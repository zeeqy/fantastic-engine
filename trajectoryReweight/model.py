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

	def forward(self, input, target, weights=None):
		if self.aggregate == 'sum':
			return self.cross_entropy_with_weights(input, target, weights).sum()
		elif self.aggregate == 'mean':
			return self.cross_entropy_with_weights(input, target, weights).mean()
		elif self.aggregate is None:
			return self.cross_entropy_with_weights(input, target, weights)

	def cross_entropy_with_weights(self, logits, target, weights=None):
		assert logits.dim() == 2
		assert not target.requires_grad
		target = target.squeeze(1) if target.dim() == 2 else target
		assert target.dim() == 1
		loss = self.log_sum_exp(logits) - self.class_select(logits, target)
		if weights is not None:
			# loss.size() = [N]. Assert weights has the same shape
			assert list(loss.size()) == list(weights.size())
			# Weight the loss
			loss = loss * weights
		return loss

	def log_sum_exp(self, x):
	    b, _ = torch.max(x, 1)
	    y = b + torch.log(torch.exp(x - b.unsqueeze(dim=1).expand_as(x)).sum(1))
	    return y

	def class_select(self, logits, target):
		# in numpy, this would be logits[:, target].
		batch_size, num_classes = logits.size()
		if target.is_cuda:
			device = target.data.get_device()
			one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
												   .long()
												   .repeat(batch_size, 1)
												   .cuda(device)
												   .eq(target.data.repeat(num_classes, 1).t()))
		else:
			one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
												   .long()
												   .repeat(batch_size, 1)
												   .eq(target.data.repeat(num_classes, 1).t()))
		return logits.masked_select(one_hot_mask)



class TrajectoryReweightNN:
	def __init__(self, torchnn, burnin=2, num_cluster=6, batch_size=100, num_iter=10, learning_rate=5e-5, early_stopping=5, device='cpu', iprint=0):
		self.torchnn = torchnn
		self.burnin = burnin
		self.num_cluster = num_cluster
		self.loss_func = WeightedCrossEntropyLoss()
		self.num_iter = num_iter
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.early_stopping = early_stopping
		self.device = device
		self.iprint = iprint

	def correct_prob(self, output, y_valid):
		prob = []
		for idx in range(len(output)):
			output_prob = self.softmax(output[idx])
			prob.append(output_prob[y_valid[idx]])
		return prob

	def softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def log(self, msg, level):
		if self.iprint >= level:
			print(msg)

	def fit(self, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor, x_test_tensor, y_test_tensor):

		self.weight_tensor = torch.from_numpy(np.ones_like(y_train_tensor,dtype=np.float32))
		train_dataset = Data.TensorDataset(x_train_tensor, y_train_tensor, self.weight_tensor)
		valid_dataset = Data.TensorDataset(x_valid_tensor, y_valid_tensor)
		test_dataset = Data.TensorDataset(x_test_tensor, y_test_tensor)

		L2 = 0.0005
		patience = 0
		best_params = {}
		best_epoch = 0
		best_score = np.Inf

		optimizer = torch.optim.Adam(self.torchnn.parameters(), lr=self.learning_rate, weight_decay=L2)
		train_loader= Data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = Data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
		reweight_loader= Data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=False)
		valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=True)
		
		"""
		burn-in epoch
		"""
		self.log('Train {} burn-in epoch...'.format(self.burnin), 1)
		
		corr_prob = []
		epoch = 1
		while epoch <= self.burnin:
			self.torchnn.train()
			for step, (data, target, weight) in enumerate(train_loader):
				data, target, weight = data.to(self.device), target.to(self.device), weight.to(self.device)
				output = self.torchnn(data)
				loss = self.loss_func(output, target, None)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			with torch.no_grad():
				train_output = []
				for step, (data, target, weight) in enumerate(reweight_loader):
					data = data.to(self.device)
					train_output.extend(self.torchnn(data).data.cpu().numpy().tolist())
				corr_prob.append(self.correct_prob(train_output, y_train_tensor.cpu().numpy()))
			epoch += 1
		corr_prob = np.array(corr_prob).T
		self.log('Train {} burn-in epoch complete.\n'.format(self.burnin) + '-'*60, 1)

		"""
		trajectory clustering after burn-in.
		"""
		self.log('Trajectory clustering for burn-in epoch...',1)
		cluster_output = self.cluster(corr_prob)
		train_loader = self.reweight(cluster_output, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor)
		self.log('Trajectory clustering for burn-in epoch complete.\n' + '-'*60, 1)
		"""
		training with reweighting starts
		"""
		self.log('Trajectory based training start ...\n',1)
		epoch = 1
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
		while epoch <= self.num_iter and patience < self.early_stopping:
			train_losses = []
			valid_losses = []

			if epoch % 3 == 0:
				cluster_output = self.cluster(corr_prob)
				train_loader = self.reweight(cluster_output, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor)

			self.torchnn.train()
			scheduler.step()
			for step, (data, target, weight) in enumerate(train_loader):
				data, target, weight = data.to(self.device), target.to(self.device), weight.to(self.device)
				output = self.torchnn(data)
				loss = self.loss_func(output, target, weight)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
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
				corr_prob = np.append(corr_prob,new_trajectory,1)

				for step, (data, target) in enumerate(valid_loader):
					data, target = data.to(self.device), target.to(self.device)
					output = self.torchnn(data)
					loss = self.loss_func(output, target)
					valid_losses.append(loss.item())
				valid_loss = np.mean(valid_losses)

			# early stopping
			if valid_loss < best_score:
				patience = 0
				best_score = valid_loss
				best_epoch = self.burnin + epoch
				torch.save(self.torchnn.state_dict(), 'checkpoint.pt')
			else:
				patience += 1

			test_loss, correct = self.test(test_loader)
			self.log('epoch = {} | training loss = {:.4f} | valid loss = {:.4f} | early stopping = {}/{} | test loss = {:.4f} | test accuarcy = {}% [{}/{}]'.format(self.burnin + epoch, train_loss, valid_loss, patience, self.early_stopping, test_loss, 100*correct/len(test_loader.dataset), correct, len(test_loader.dataset)), 1)
			epoch += 1

		"""
		training finsihed
		"""
		self.torchnn.load_state_dict(torch.load('checkpoint.pt'))
		self.log('Trajectory based training complete, best validation loss = {} at epoch = {}.'.format(best_score, best_epoch), 1)

	def reweight(self, cluster_output, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor):
		eval_grads = []
		validNet = deepcopy(self.torchnn)
		optimizer = torch.optim.Adam(self.torchnn.parameters(), lr=self.learning_rate)
		valid_output = validNet(x_valid_tensor.to(self.device))
		valid_loss = self.loss_func(valid_output, y_valid_tensor.to(self.device), None)
		optimizer.zero_grad()
		valid_loss.backward()
		for w in validNet.parameters():
			if w.requires_grad:
				eval_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))
		eval_grads = np.array(eval_grads)
		
		for cid in range(self.num_cluster):
			subset_grads = []
			cidx = (cluster_output==cid).nonzero()[0].tolist()
			x_cluster = x_train_tensor[cidx]
			y_cluster = y_train_tensor[cidx]
			size = y_cluster.shape[0]
			if size == 0:
				continue
			sample_size = min(int(size), 2000)
			sample_idx = np.random.choice(range(size), sample_size, replace=False).tolist()
			x_subset = x_cluster[sample_idx]
			y_subset = y_cluster[sample_idx]

			subset_output = validNet(x_subset.to(self.device))
			subset_loss = self.loss_func(subset_output, y_subset.to(self.device), None)
			optimizer.zero_grad()
			subset_loss.backward()
			for w in validNet.parameters():
				if w.requires_grad:
					subset_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))
			subset_grads = np.array(subset_grads)
			sim = 1 - spatial.distance.cosine(eval_grads, subset_grads)

			self.weight_tensor[cidx] = 0.5 * (sim+1)
			self.weight_tensor[cidx] = self.weight_tensor[cidx].clamp(0.01)
			self.log('| - ' + str({cid:cid, 'size': size, 'weight':self.weight_tensor[cidx][0].data.numpy().tolist()}),2)

		train_dataset = Data.TensorDataset(x_train_tensor, y_train_tensor, self.weight_tensor)
		train_loader = Data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		return train_loader

	def cluster(self, corr_prob):
		self.gmmCluster = GaussianMixture(self.num_cluster,corr_prob.shape[1], iprint=0)
		self.gmmCluster.fit(corr_prob)
		cluster_output = self.gmmCluster.predict(corr_prob, prob=False)
		return cluster_output

	def predict(self, x_test_tensor):
		test_output = self.torchnn(x_test_tensor.to(self.device))
		return torch.max(test_output, 1)[1].data.cpu().numpy()


	def test(self, test_loader):
		test_loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in test_loader:
				data, target = data.to(self.device), target.to(self.device)
				output = self.torchnn(data)
				test_loss += self.loss_func(output, target).item() # sum up batch loss
				pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

		test_loss /= len(test_loader.dataset)

		return test_loss, correct