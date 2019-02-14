import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
from copy import deepcopy
from trajectoryReweight.gmm import GaussianMixture
from scipy import spatial


class WeightedCrossEntropyLoss(nn.Module):
	"""
	Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
	vector of shape (batch_size,).
	"""
	def __init__(self, aggregate='mean'):
		super(WeightedCrossEntropyLoss, self).__init__()
		assert aggregate in ['sum', 'mean', None]
		self.aggregate = aggregate
		self.base_loss = nn.CrossEntropyLoss(reduction='none')

	def forward(self, data, target, weights=None):
		if self.aggregate == 'sum':
			return self.cross_entropy_with_weights(data, target, weights).sum()
		elif self.aggregate == 'mean':
			return self.cross_entropy_with_weights(data, target, weights).mean()
		elif self.aggregate is None:
			return self.cross_entropy_with_weights(data, target, weights)

	def cross_entropy_with_weights(self, data, target, weights=None):
		loss = self.base_loss(data, target)
		if weights is not None:
			loss = loss * weights
		return loss

class TrajectoryReweightNN:
	def __init__(self, torchnn, 
				burnin=2, num_cluster=6, 
				batch_size=100, num_iter=10, 
				learning_rate=5e-5, early_stopping=5, 
				device='cpu', traj_step = 3,iprint=0):
		
		self.torchnn = torchnn
		self.burnin = burnin
		self.num_cluster = num_cluster
		self.loss_func = WeightedCrossEntropyLoss()
		self.num_iter = num_iter
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.early_stopping = early_stopping
		self.device = device
		self.traj_step = traj_step
		self.iprint = iprint

	def correct_prob(self, output, y):
		prob = []
		for idx in range(len(output)):
			output_prob = self.softmax(output[idx])
			prob.append(output_prob[y[idx]] + np.var(output_prob) + np.var(np.concatenate([output_prob[:y[idx]], output_prob[y[idx]+1:]])))
		return prob

	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def log(self, msg, level):
		if self.iprint >= level:
			print(msg)

	def fit(self, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor, x_test_tensor, y_test_tensor, special_index=None):

		self.weight_tensor = torch.from_numpy(np.ones_like(y_train_tensor,dtype=np.float32))
		train_dataset = Data.TensorDataset(x_train_tensor, y_train_tensor, self.weight_tensor)
		valid_dataset = Data.TensorDataset(x_valid_tensor, y_valid_tensor)
		test_dataset = Data.TensorDataset(x_test_tensor, y_test_tensor)

		L2 = 0.0005
		patience = 0
		best_epoch = 0
		best_score = np.inf
		hiatus = 0
		best_params = {}

		self.optimizer = torch.optim.Adam(self.torchnn.parameters(), lr=self.learning_rate, weight_decay=L2)
		train_loader= Data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = Data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
		reweight_loader= Data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False)
		valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=True)
		
		"""
		burn-in epoch
		"""
		self.log('Train {} burn-in epoch...'.format(self.burnin), 1)
		
		self.traject_matrix = []
		epoch = 1
		scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.95)
		while epoch <= self.burnin:
			self.torchnn.train()
			scheduler.step()
			for step, (data, target, weight) in enumerate(train_loader):
				data, target, weight = data.to(self.device), target.to(self.device), weight.to(self.device)
				output = self.torchnn(data)
				loss = self.loss_func(output, target, None)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()

			with torch.no_grad():
				train_output = []
				for step, (data, target, weight) in enumerate(reweight_loader):
					data = data.to(self.device)
					train_output.extend(self.torchnn(data).data.cpu().numpy().tolist())
				self.traject_matrix.append(self.correct_prob(train_output, y_train_tensor.cpu().numpy()))
			test_loss, correct = self.evaluate(test_loader)
			self.log('epoch = {} | test loss = {:.4f} | test accuarcy = {}% [{}/{}]'.format(epoch, test_loss, 100*correct/len(test_loader.dataset), correct, len(test_loader.dataset)), 2)
			epoch += 1
		self.traject_matrix = np.array(self.traject_matrix).T
		self.log('Train {} burn-in epoch complete.\n'.format(self.burnin) + '-'*60, 1)

		"""
		trajectory clustering after burn-in.
		"""
		self.log('Trajectory clustering for burn-in epoch...',1)
		self.cluster_output = self.cluster()
		train_loader = self.reweight(x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor, special_index)
		self.log('Trajectory clustering for burn-in epoch complete.\n' + '-'*60, 1)
		"""
		training with reweighting starts
		"""
		self.log('Trajectory based training start ...\n',1)
		while epoch <= self.num_iter and patience < self.early_stopping:

			if hiatus == self.traj_step:
				hiatus = 0
				self.cluster_output = self.cluster()
				train_loader = self.reweight(x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor, special_index)
			
			train_losses = []
			self.torchnn.train()
			scheduler.step()
			for step, (data, target, weight) in enumerate(train_loader):
				data, target, weight = data.to(self.device), target.to(self.device), weight.to(self.device)
				output = self.torchnn(data)
				loss = self.loss_func(output, target, weight)
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				train_losses.append(loss.item())
			train_loss = np.mean(train_losses)
			
			self.torchnn.eval()
			with torch.no_grad():
				train_output = []
				for step, (data, target, weight) in enumerate(reweight_loader):
					data = data.to(self.device)
					output = self.torchnn(data)
					train_output.extend(output.data.cpu().numpy().tolist())
				new_trajectory = np.array(self.correct_prob(train_output,y_train_tensor.cpu().numpy())).reshape(-1,1)
				self.traject_matrix = np.append(self.traject_matrix,new_trajectory,1)

				valid_loss, correct = self.evaluate(valid_loader)
				valid_accuracy = 100 * correct / len(valid_loader.dataset)

			# early stopping
			if valid_loss <= best_score:
				patience = 0
				best_score = valid_loss
				best_epoch = epoch
				torch.save(self.torchnn.state_dict(), 'checkpoint.pt')
			else:
				patience += 1

			test_loss, correct = self.evaluate(test_loader)
			self.log('epoch = {} | training loss = {:.4f} | valid loss = {:.4f} | valid accuarcy = {}% | early stopping = {}/{} | test loss = {:.4f} | test accuarcy = {}% [{}/{}]'.format(epoch, train_loss, valid_loss, valid_accuracy, patience, self.early_stopping, test_loss, 100*correct/len(test_loader.dataset), correct, len(test_loader.dataset)), 1)
			epoch += 1
			hiatus += 1

		"""
		training finsihed
		"""
		self.torchnn.load_state_dict(torch.load('checkpoint.pt'))
		self.log('Trajectory based training complete, best validation loss = {} at epoch = {}.'.format(best_score, best_epoch), 1)

	def reweight(self, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor, special_index):
		valid_grad = []
		validNet = deepcopy(self.torchnn)
		valid_output = validNet(x_valid_tensor.to(self.device))
		valid_loss = self.loss_func(valid_output, y_valid_tensor.to(self.device), None)
		self.optimizer.zero_grad()
		valid_loss.backward()
		for w in validNet.parameters():
			if w.requires_grad:
				valid_grad.extend(list(w.grad.cpu().detach().numpy().flatten()))
		valid_grad = np.array(valid_grad)
		
		for cid in range(self.num_cluster):
			subset_grads = []
			cidx = (self.cluster_output==cid).nonzero()[0].tolist()
			x_cluster = x_train_tensor[cidx]
			y_cluster = y_train_tensor[cidx]
			size = len(cidx)
			if size == 0:
				continue
			sample_size = min(int(size), 2000)
			sample_idx = np.random.choice(range(size), sample_size, replace=False).tolist()
			x_subset = x_cluster[sample_idx]
			y_subset = y_cluster[sample_idx]

			subset_output = validNet(x_subset.to(self.device))
			subset_loss = self.loss_func(subset_output, y_subset.to(self.device), None)
			self.optimizer.zero_grad()
			subset_loss.backward()
			for w in validNet.parameters():
				if w.requires_grad:
					subset_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))
			subset_grads = np.array(subset_grads)
			sim = 1 - spatial.distance.cosine(valid_grad, subset_grads)

			self.weight_tensor[cidx] += 0.05 * sim
			self.weight_tensor[cidx] = self.weight_tensor[cidx].clamp(0.001)
			if special_index != []:
				num_special = self.special_ratio(cidx,special_index)
				self.log('| - ' + str({cid:cid, 'size': size, 'sim': '{:.4f}'.format(sim), 'num_special': num_special, 'spe_ratio':'{:.4f}'.format(num_special/size)}),2)
			else:
				self.log('| - ' + str({cid:cid, 'size': size, 'sim': sim}),2)

		train_dataset = Data.TensorDataset(x_train_tensor, y_train_tensor, self.weight_tensor)
		train_loader = Data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		return train_loader

	def cluster(self):
		self.gmmCluster = GaussianMixture(self.num_cluster,self.traject_matrix.shape[1], iprint=0)
		self.gmmCluster.fit(self.traject_matrix)
		cluster_output = self.gmmCluster.predict(self.traject_matrix, prob=False)
		return cluster_output

	def predict(self, x_test_tensor):
		test_output = self.torchnn(x_test_tensor.to(self.device))
		return torch.max(test_output, 1)[1].data.cpu().numpy()

	def evaluate(self, data_loader):
		loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in data_loader:
				data, target = data.to(self.device), target.to(self.device)
				output = self.torchnn(data)
				loss += self.loss_func(output, target).item() # sum up batch loss
				pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

		loss /= len(data_loader.dataset)

		return loss, correct

	def special_ratio(self, cidx, noise_index):
		spe = set(cidx) - (set(cidx) - set(noise_index))
		return len(spe)
