import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np
from trajectory_classifier.gmm import GaussianMixture
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
	def __init__(self, torchnn, burnin=2, num_cluster=6, batch_size=100, num_iter=10, learning_rate=5e-5, early_stopping=5):
		self.torchnn = torchnn
		self.burnin = burnin
		self.num_cluster = num_cluster
		self.loss_func = WeightedCrossEntropyLoss()
		self.num_iter = num_iter
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.early_stopping = early_stopping

	def correct_prob(self, output, y_valid):
		prob = []
		for idx in range(len(output)):
			output_prob = self.softmax(output[idx])
			prob.append(output_prob[y_valid[idx]])
		return prob

	def softmax(self, x):
	    return np.exp(x) / np.sum(np.exp(x), axis=0)


	def fit(self, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor, device='cpu'):

		self.weight_tensor = torch.from_numpy(np.ones_like(y_train_tensor,dtype=np.float32))
		tensor_dataset = Data.TensorDataset(x_train_tensor, y_train_tensor, self.weight_tensor)

		L2 = 0.0005
		patience = 0
		best_params = {}
		best_valid = best_epoch = 0

		optimizer = torch.optim.Adam(self.torchnn.parameters(), lr=self.learning_rate, weight_decay=L2)
		train_loader= Data.DataLoader(dataset=tensor_dataset, batch_size=self.batch_size, shuffle=True)
		
		"""
		burn-in epoch
		"""
		print('Train {} burn-in epoch...'.format(self.burnin))
		
		corr_prob = []
		for epoch in range(self.burnin):
			for step, (b_x, b_y, b_w) in enumerate(train_loader):
				b_x, b_y, b_w = b_x.to(device), b_y.to(device), b_w.to(device)
				output = self.torchnn(b_x)
				loss = self.loss_func(output, b_y, None)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			with torch.no_grad():
				train_output = []
				for step, (b_x, b_y, b_w) in enumerate(Data.DataLoader(dataset=tensor_dataset, batch_size=self.batch_size, shuffle=False)):
					b_x = b_x.to(device)
					train_output.extend(self.torchnn(b_x).data.cpu().numpy().tolist())
				corr_prob.append(self.correct_prob(train_output, y_train_tensor.cpu().numpy()))
		
		print('Train {} burn-in epoch complete.\n'.format(self.burnin) + '-'*60)

		"""
		trajectory clustering after burn-in.
		"""
		print('Trajectory clustering for burn-in epoch...')
		self.gmmCluster = GaussianMixture(self.num_cluster,self.burnin)
		corr_prob = np.array(corr_prob).T
		self.gmmCluster.fit(corr_prob, iprint=False)
		cluster = self.gmmCluster.predict(corr_prob, prob=False)
		print('Trajectory clustering for burn-in epoch complete.\n' + '-'*60)

		"""
		training with reweighting starts
		"""
		print('Trajectory based training start ...\n')
		epoch = 1
		while epoch <= self.num_iter and patience < self.early_stopping:
			print('|' + '-'*67)
			print('| - epoch = {}'.format(epoch))
			print('| - compute valid set grad...')
			eval_grads = []
			valid_output = self.torchnn(x_valid_tensor.to(device))
			valid_loss = self.loss_func(valid_output, y_valid_tensor.to(device))
			optimizer.zero_grad()
			valid_loss.backward()
			for w in self.torchnn.parameters():
				if w.requires_grad:
					eval_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))
			eval_grads = np.array(eval_grads)
			
			print('| - update cluster weight based on valid set...')
			#if len(np.unique(cluster)) != 1:
			for cid in range(self.num_cluster):
				subset_grads = []
				cidx = (cluster==cid).nonzero()[0].tolist()
				x_cluster = x_train_tensor[cidx]
				y_cluster = y_train_tensor[cidx]
				size = y_cluster.shape[0]
				if size == 0:
					continue
				sample_size = min(int(size), 2000)
				sample_idx = np.random.choice(range(size), sample_size, replace=False).tolist()
				x_subset = x_cluster[sample_idx]
				y_subset = y_cluster[sample_idx]

				subset_output = self.torchnn(x_subset.to(device))
				subset_loss = self.loss_func(subset_output, y_subset.to(device), None)
				optimizer.zero_grad()
				subset_loss.backward()
				for w in self.torchnn.parameters():
					if w.requires_grad:
						subset_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))
				subset_grads = np.array(subset_grads)
				sim = 1 - spatial.distance.cosine(eval_grads, subset_grads)

				self.weight_tensor[cidx] += 0.1 * sim
				self.weight_tensor[cidx] = self.weight_tensor[cidx].clamp(0.01)
				print('| - ', {cid:cid, 'size': size, 'weight':self.weight_tensor[cidx][0].data.numpy().tolist()})

			#else:
				#self.weight_tensor = torch.from_numpy(np.ones_like(y_train_tensor,dtype=np.float32))
			tensor_dataset = Data.TensorDataset(x_train_tensor, y_train_tensor, self.weight_tensor)
			train_loader= Data.DataLoader(dataset=tensor_dataset, batch_size=self.batch_size, shuffle=True)

			print('| - mini-batch training based on cluster weights...')
			for step, (b_x, b_y, b_w) in enumerate(train_loader):
				b_x, b_y, b_w = b_x.to(device), b_y.to(device), b_w.to(device)
				output = self.torchnn(b_x)
				loss = self.loss_func(output, b_y, b_w)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			with torch.no_grad():
				train_output = []
				for step, (b_x, b_y, b_w) in enumerate(Data.DataLoader(dataset=tensor_dataset, batch_size=self.batch_size, shuffle=False)):
					b_x = b_x.to(device)
					train_output.extend(self.torchnn(b_x).data.cpu().numpy().tolist())
				tmp = np.array(self.correct_prob(train_output,y_train_tensor.cpu().numpy())).reshape(-1,1)
				corr_prob = np.append(corr_prob,tmp,1)
				valid_output = self.torchnn(x_valid_tensor.to(device))
				valid_output_y = torch.max(valid_output, 1)[1].data.cpu().numpy()
				valid_accuracy = float((valid_output_y == y_valid_tensor.data.numpy()).astype(int).sum()) / float(y_valid_tensor.size(0))
				print('| - epoch = {} | loss = {:.4f} | valid error = {:.4f}'.format(epoch+1, loss, 1 - valid_accuracy))
				if valid_accuracy > best_valid:
					best_valid = valid_accuracy
					best_epoch = self.burnin + epoch
					for name, tensor in self.torchnn.state_dict(keep_vars=True).items():
						if tensor.requires_grad:
							best_params[name] = tensor.data.clone()
				else:
					patience += 1

			print('| - update trajectory cluster...')
			if epoch % 3 == 0:
				#self.gmmCluster.append_fit(corr_prob, 1, iprint=False)
				self.gmmCluster = GaussianMixture(self.num_cluster,self.burnin + epoch)
				self.gmmCluster.fit(corr_prob, iprint=False)
				cluster = self.gmmCluster.predict(corr_prob, prob=False)
			epoch += 1
			print('|' + '-'*67 + '\n')

		"""
		training finsihed
		"""
		self.torchnn.load_state_dict(best_params, strict=False)
		print('Trajectory based training complete, best validation error = {} at epoch = {}.'.format(1-best_valid, best_epoch))

	def predict(self, x_test_tensor, device='cpu'):
		test_output = self.torchnn(x_test_tensor.to(device))
		return torch.max(test_output, 1)[1].data.cpu().numpy()


