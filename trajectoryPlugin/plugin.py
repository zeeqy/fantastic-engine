import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
from trajectoryPlugin.gmm import GaussianMixture
from trajectoryPlugin.collate import default_collate as core_collate
from sklearn import mixture
from scipy import spatial
import sys, logging


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


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

class RandomBatchSampler(torch.utils.data.sampler.Sampler):
	def __init__(self, shuffle, batch_size):
		self.batch_size = batch_size
		self.shuffle = shuffle

	def __iter__(self):
		data_iter = iter(self.shuffle)
		return data_iter

	def __len__(self):
		return (len(sum(self.shuffle,[])) + self.batch_size - 1)//self.batch_size

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

		note: this api will handle dataset during training, see example.
	"""
	
	def __init__(self, num_cluster=6, device='cpu', update_rate=0.1, iprint=0):
		self.num_cluster = num_cluster
		self.update_rate = update_rate
		self.loss_func = WeightedCrossEntropyLoss()
		self.device = device
		self.logger = logging.getLogger(__name__)
		self.iprint = iprint #output level

	def _collateFn(self, batch):
		transposed = zip(*batch)
		res = []
		for samples in transposed:
			res += core_collate(samples)
		return res


	def _shuffleIndex(self):
		rand_idx = torch.randperm(self.train_dataset.__len__()).tolist()
		return [rand_idx[i:i+self.batch_size] for i in range(0, len(rand_idx), self.batch_size)]

	def generateTrainLoader(self):
		self.rand_idx = self._shuffleIndex()
		self.batch_sampler = RandomBatchSampler(self.rand_idx, self.batch_size)
		self.weightset = Data.TensorDataset(self.weight_tensor)
		self.train_loader = Data.DataLoader(
			ConcatDataset(
				self.train_dataset,
				self.weightset
			),
			batch_sampler=self.batch_sampler, shuffle=False, collate_fn=self._collateFn)

	def dataLoader(self, trainset, validset, batch_size=100):
		self.batch_size = batch_size
		self.train_dataset = trainset
		self.valid_loader = Data.DataLoader(validset, batch_size=self.batch_size, shuffle=True)
		self.weight_raw = torch.tensor(np.ones(self.train_dataset.__len__(), dtype=np.float32), requires_grad=False)
		self.weight_tensor = self._normalize(self.weight_raw)
		self.traject_matrix = np.empty((self.train_dataset.__len__(),0))
		self.generateTrainLoader()
		
	def log(self, msg, level):
		if self.iprint >= level:
			self.logger.info(msg)
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
			prob_output = np.empty(self.train_dataset.__len__())
			for step, (data, target, weight) in enumerate(self.train_loader):
				data = data.to(self.device)
				output = torchnn(data).data.cpu().numpy().tolist()
				prob_output[self.rand_idx[step]] = self._correctProb(output, target.data.cpu().numpy())
			self.traject_matrix = np.append(self.traject_matrix, np.matrix(prob_output).T, 1)

	def trajectoryBins(self):
		bins = [self.traject_matrix[:,i:i+3] for i in range(0, self.traject_matrix.shape[1], 1)]
		self.traject_bins = np.empty((self.train_dataset.__len__(), 0))
		mean_traject = np.empty((self.train_dataset.__len__(), 0))
		std_traject = np.empty((self.train_dataset.__len__(), 0))
		for b in bins:
			mean_traject = np.append(mean_traject, np.matrix(np.mean(b, axis=1)), 1)
			std_traject = np.append(std_traject, np.matrix(np.std(b, axis=1)), 1)
		self.traject_bins = np.append(self.traject_bins, mean_traject, 1)
		self.traject_bins = np.append(self.traject_bins, std_traject, 1)

	def _validGrad(self, validNet):
		valid_grads = []
		validNet.eval()
		validNet.zero_grad()
		for step, (data, target) in enumerate(self.valid_loader):
			data, target = data.to(self.device), target.to(self.device)
			valid_output = validNet(data)
			valid_loss = self.loss_func(valid_output, target, None, 'mean')
			valid_loss.backward()
		for w in validNet.parameters():
			if w.requires_grad:
				valid_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))
		validNet.zero_grad()
		return np.array(valid_grads)

	def _normalize(self, tensor):
		norm_fact = tensor.size()[0] / torch.sum(tensor)
		return norm_fact * tensor

	def reweightData(self, validNet, special_index=[]):
		valid_grads = self._validGrad(validNet)
		sim_dict = {}
		validNet.eval() # eval mode, important!
		for cid in range(self.num_cluster):
			subset_grads = []
			cidx = (self.cluster_output==cid).nonzero()[0].tolist()
			size = len(cidx)
			if size == 0:
				continue
			subset = torch.utils.data.dataset.Subset(self.train_dataset, cidx)
			subset_loader = torch.utils.data.DataLoader(subset, batch_size=self.batch_size, shuffle=True)

			validNet.zero_grad()
			for step, (data, target) in enumerate(subset_loader):
				data, target = data.to(self.device), target.to(self.device)
				subset_output = validNet(data)
				subset_loss = self.loss_func(subset_output, target, None)
				subset_loss.backward()
			for w in validNet.parameters():
				if w.requires_grad:
					subset_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))

			sim = 1 - spatial.distance.cosine(valid_grads, subset_grads)
			sim_dict.update({cid : sim})

		#update weights
		for cid in range(self.num_cluster):
			cidx = (self.cluster_output==cid).nonzero()[0].tolist()
			size = len(cidx)
			if size == 0:
				continue
			self.weight_raw[cidx] = sim_dict[cid] #+= self.update_rate * sim_dict[cid]
			
			#print some insights about noisy data
			if special_index != []:
				num_special = self._specialRatio(cidx, special_index)
				self.log('| - ' + str({cid:cid, 'size': size, 'sim': '{:.4f}'.format(sim_dict[cid]), 'num_special': num_special, 'spe_ratio':'{:.4f}'.format(num_special/size)}),2)
			else:
				self.log('| - ' + str({cid:cid, 'size': size, 'sim': sim_dict[cid]}),2)

		#normalize weight tensor
		self.weight_raw = self.weight_raw.clamp(0.001)
		self.weight_tensor = self._normalize(self.weight_raw)

		validNet.zero_grad()

	def clusterTrajectory(self):
		self.gmmCluster = mixture.GaussianMixture(n_components=self.num_cluster, covariance_type='full', max_iter=500, tol=1e-5, init_params='kmeans', verbose=0)
		#self.gmmCluster = GaussianMixture(self.num_cluster, self.traject_matrix.shape[1], iprint=0)
		self.gmmCluster.fit(self.traject_matrix)
		self.cluster_output = self.gmmCluster.predict(self.traject_matrix)

	def clusterBins(self):
		self.gmmCluster = mixture.GaussianMixture(n_components=self.num_cluster, covariance_type='full', max_iter=500, tol=1e-5, init_params='kmeans', verbose=0)
		#self.gmmCluster = GaussianMixture(self.num_cluster, self.traject_matrix.shape[1], iprint=0)
		self.gmmCluster.fit(self.traject_bins)
		self.cluster_output = self.gmmCluster.predict(self.traject_bins)


	def _specialRatio(self, cidx, special_index):
		spe = set(cidx) - (set(cidx) - set(special_index))
		return len(spe)