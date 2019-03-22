import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
from trajectoryPlugin.gmm import GaussianMixture
from trajectoryPlugin.collate import default_collate as core_collate
from scipy import spatial


class WeightedCrossEntropyLoss(nn.Module):
	"""
	Cross entropy with instance-wise weights. Leave `aggregate` to None to obtain a loss
	vector of shape (batch_size,).
	"""
	def __init__(self):
		super(WeightedCrossEntropyLoss, self).__init__()
		self.base_loss = nn.CrossEntropyLoss(reduction='none')

	def forward(self, data, target, weights=None, aggregate='mean'):
		assert aggregate in ['sum', 'mean', None]
		if aggregate == 'sum':
			return self.cross_entropy_with_weights(data, target, weights).sum()
		elif aggregate == 'mean':
			return self.cross_entropy_with_weights(data, target, weights).mean()
		elif aggregate is None:
			return self.cross_entropy_with_weights(data, target, weights)

	def cross_entropy_with_weights(self, data, target, weights=None):
		loss = self.base_loss(data, target)
		if weights is not None:
			loss = loss * weights
		return loss

class ConcatDataset(torch.utils.data.Dataset):
	def __init__(self, *datasets):
		self.datasets = datasets

	def __getitem__(self, i):
		return tuple(d[i] for d in self.datasets)

	def __len__(self):
		return min(len(d) for d in self.datasets)

class API:
	"""
	This API will take care of recording trajectory, clustering trajectory and reweigting dataset
	Args:
		batch_size: mini batch size when processing, avioding memory error;
		x_train_tensor: training data in tensor;
		y_train_tensor: training label in tensor;
		x_valid_tensor, y_valid_tensor: validation dataset;
		num_cluster: number of clunters

		note: this api will handle dataset during training, see example.
	"""
	
	def __init__(self, num_cluster=6, device='cpu', iprint=0):
		self.num_cluster = num_cluster
		self.loss_func = WeightedCrossEntropyLoss()
		self.device = device
		self.iprint = iprint #output level

	def _collateFn(self, batch):
		transposed = zip(*batch)
		res = []
		for samples in transposed:
			res += core_collate(samples)
		return res

	# def dataTensor(self, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor, batch_size=100):
	# 	self.batch_size = batch_size
	# 	self.weight_tensor = torch.from_numpy(np.ones_like(y_train_tensor,dtype=np.float32))
	# 	self.weight_tensor.requires_grad = False
	# 	self.train_dataset = Data.TensorDataset(x_train_tensor, y_train_tensor, self.weight_tensor)
	# 	self.train_loader = Data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
	# 	self.reweight_loader = Data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False)
	# 	self.valid_dataset = Data.TensorDataset(x_valid_tensor, y_valid_tensor)
	# 	self.traject_matrix = np.empty((y_train_tensor.size()[0],0))

	def dataLoader(self, trainset, validset, batch_size=100):
		self.batch_size = batch_size
		self.train_dataset = trainset
		self.valid_dataset = validset
		self.valid_loader = Data.DataLoader(self.valid_dataset, batch_size=self.batch_size,shuffle=False)
		self.weight_tensor = torch.tensor(np.ones(self.train_dataset.__len__(), dtype=np.float32), requires_grad=False)
		self.weightset = Data.TensorDataset(self.weight_tensor)
		self.train_loader = Data.DataLoader(
			ConcatDataset(
				self.train_dataset,
				self.weightset
			),
			batch_size=self.batch_size, shuffle=True,collate_fn=self._collateFn)
		self.reweight_loader = Data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False)
		self.traject_matrix = np.empty((self.train_dataset.__len__(),0))

	def log(self, msg, level):
		if self.iprint >= level:
			print(msg)
		if self.iprint == 99:
			pass # if we need to dump json to file in the future

	def _correctProb(self, output, y):
		prob = []
		for idx in range(len(output)):
			output_prob = self._softmax(output[idx])
			prob.append(output_prob[y[idx]]) # could be more like + np.var(output_prob) + np.var(np.concatenate([output_prob[:y[idx]], output_prob[y[idx]+1:]])))
		return prob

	def _softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def createTrajectory(self, torchnn):
		torchnn.eval()
		with torch.no_grad():
			prob_output = []
			for step, (data, target) in enumerate(self.reweight_loader):
				data = data.to(self.device)
				output = torchnn(data).data.cpu().numpy().tolist()
				prob_output += self._correctProb(output, target.data.cpu().numpy())
			self.traject_matrix = np.append(self.traject_matrix,np.matrix(prob_output).T,1)

	def _validGrad(self, validNet):
		valid_grad = []
		validNet.eval()
		validNet.zero_grad()
		for step, (data, target) in enumerate(self.reweight_loader):
			data, target = data.to(self.device), target.to(self.device)
			valid_output = validNet(data)
			valid_loss = self.loss_func(valid_output, target, None)
			valid_loss.backward()
		for w in validNet.parameters():
			if w.requires_grad:
				valid_grad.extend(list(w.grad.cpu().detach().numpy().flatten()))
		return np.array(valid_grad)


	def reweightData(self, validNet, num_sample, special_index=[]):
		valid_grad = self._validGrad(validNet)
		for cid in range(self.num_cluster):
			subset_grads = []
			cidx = (self.cluster_output==cid).nonzero()[0].tolist()
			size = len(cidx)
			if size == 0:
				continue
			sample_size = min(int(size), num_sample)
			sample_idx = [cidx[i] for i in np.random.choice(range(size), sample_size, replace=False).tolist()]
			subset_loader = torch.utils.data.DataLoader(Data.Subset(self.train_dataset, sample_idx), batch_size=self.batch_size, shuffle=False)

			validNet.eval() # eval mode, important!
			validNet.zero_grad()
			for step, (data, target) in enumerate(subset_loader):
				data, target = data.to(self.device), target.to(self.device)
				subset_output = validNet(data)
				subset_loss = self.loss_func(subset_output, target, None)
				subset_loss.backward()
			for w in validNet.parameters():
				if w.requires_grad:
					subset_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))
			subset_grads = np.array(subset_grads)
			sim = 1 - spatial.distance.cosine(valid_grad, subset_grads)

			self.weight_tensor[cidx] += 0.05 * sim # how to update weight?
			self.weight_tensor[cidx] = self.weight_tensor[cidx].clamp(0.001)
			if special_index != []:
				num_special = self._specialRatio(cidx,special_index)
				self.log('| - ' + str({cid:cid, 'size': size, 'sim': '{:.4f}'.format(sim), 'num_special': num_special, 'spe_ratio':'{:.4f}'.format(num_special/size)}),2)
			else:
				self.log('| - ' + str({cid:cid, 'size': size, 'sim': sim}),2)
		norm_fact = self.weight_tensor.size()[0] / torch.sum(self.weight_tensor) # normalizing weights
		self.weight_tensor = norm_fact * self.weight_tensor
		self.weightset = Data.TensorDataset(self.weight_tensor)
		self.train_loader = Data.DataLoader(
			ConcatDataset(
				self.train_dataset,
				self.weightset
			),
			batch_size=self.batch_size, shuffle=True, collate_fn=self._collateFn)
		validNet.zero_grad()

	def clusterTrajectory(self):
		self.gmmCluster = GaussianMixture(self.num_cluster, self.traject_matrix.shape[1], iprint=0)
		self.gmmCluster.fit(self.traject_matrix)
		self.cluster_output = self.gmmCluster.predict(self.traject_matrix, prob=False)


	def _specialRatio(self, cidx, special_index):
		spe = set(cidx) - (set(cidx) - set(special_index))
		return len(spe)