import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
from copy import deepcopy
from trajectoryPlugin.gmm import GaussianMixture
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
	
	def __init__(self, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor, num_cluster=6, batch_size=100, device='cpu', iprint=0):
		self.batch_size = batch_size
		self.weight_tensor = torch.from_numpy(np.ones_like(y_train_tensor,dtype=np.float32))
		self.train_dataset = Data.TensorDataset(x_train_tensor, y_train_tensor, self.weight_tensor)
		self.train_loader = Data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)
		self.reweight_loader = Data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False)
		self.valid_dataset = Data.TensorDataset(x_valid_tensor, y_valid_tensor)

		self.traject_matrix = np.empty((y_train_tensor.size()[0],0))
		self.num_cluster = num_cluster
		self.loss_func = WeightedCrossEntropyLoss()
		self.device = device
		self.iprint = iprint #output level

	def log(self, msg, level):
		if self.iprint >= level:
			print(msg)

	def _correctProb(self, output, y):
		prob = []
		for idx in range(len(output)):
			output_prob = self._softmax(output[idx])
			prob.append(output_prob[y[idx]]) # could be more like + np.var(output_prob) + np.var(np.concatenate([output_prob[:y[idx]], output_prob[y[idx]+1:]])))
		return prob

	def _softmax(self, x):
		return np.exp(x) / np.sum(np.exp(x), axis=0)

	def createTrajectory(self, torchnn):
		validNet = deepcopy(torchnn)
		prob_output = []
		for step, (data, target, weight) in enumerate(self.reweight_loader):
			data = data.to(self.device)
			output = validNet(data).data.cpu().numpy().tolist()
			prob_output += self._correctProb(output, target.data.cpu().numpy())
		self.traject_matrix = np.append(self.traject_matrix,np.matrix(prob_output).T,1)

	def _validGrad(self, validNet):
		valid_grad = []
		valid_output = validNet(self.valid_dataset.tensors[0].to(self.device))
		valid_loss = self.loss_func(valid_output, self.valid_dataset.tensors[1].to(self.device), None)
		validNet.zero_grad()
		valid_loss.backward()
		for w in validNet.parameters():
			if w.requires_grad:
				valid_grad.extend(list(w.grad.cpu().detach().numpy().flatten()))
		return np.array(valid_grad)


	def reweightData(self, torchnn, num_sample, special_index=[]):
		validNet = deepcopy(torchnn)
		valid_grad = self._validGrad(validNet)
		for cid in range(self.num_cluster):
			subset_grads = []
			cidx = (self.cluster_output==cid).nonzero()[0].tolist()
			x_cluster = self.train_dataset.tensors[0][cidx] #x_train_tensor
			y_cluster = self.train_dataset.tensors[1][cidx] #y_train_tensor
			size = len(cidx)
			if size == 0:
				continue
			sample_size = min(int(size), num_sample)
			sample_idx = np.random.choice(range(size), sample_size, replace=False).tolist()
			x_subset = x_cluster[sample_idx]
			y_subset = y_cluster[sample_idx]

			subset_output = validNet(x_subset.to(self.device))
			subset_loss = self.loss_func(subset_output, y_subset.to(self.device), None)
			validNet.zero_grad()
			subset_loss.backward()
			for w in validNet.parameters():
				if w.requires_grad:
					subset_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))
			subset_grads = np.array(subset_grads)
			sim = 1 - spatial.distance.cosine(valid_grad, subset_grads)

			self.weight_tensor[cidx] += 0.05 * sim #how to update weight?
			self.weight_tensor[cidx] = self.weight_tensor[cidx].clamp(0.001)
			if special_index != []:
				num_special = self._specialRatio(cidx,special_index)
				self.log('| - ' + str({cid:cid, 'size': size, 'sim': '{:.4f}'.format(sim), 'num_special': num_special, 'spe_ratio':'{:.4f}'.format(num_special/size)}),2)
			else:
				self.log('| - ' + str({cid:cid, 'size': size, 'sim': sim}),2)

		self.train_dataset = Data.TensorDataset(self.train_dataset.tensors[0], self.train_dataset.tensors[1], self.weight_tensor)
		self.train_loader = Data.DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

	def clusterTrajectory(self):
		self.gmmCluster = GaussianMixture(self.num_cluster, self.traject_matrix.shape[1], iprint=0)
		self.gmmCluster.fit(self.traject_matrix)
		self.cluster_output = self.gmmCluster.predict(self.traject_matrix, prob=False)


	def _specialRatio(self, cidx, special_index):
		spe = set(cidx) - (set(cidx) - set(special_index))
		return len(spe)