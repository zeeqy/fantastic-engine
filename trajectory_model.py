import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from trajectory_classifier.gmm import GaussianMixture
import matplotlib.pyplot as plt
import time
from scipy import spatial
from util.util import mnist_noise, correct_prob

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""
CNN
"""
class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(7 * 7 * 64, 1000)
		self.fc2 = nn.Linear(1000, 10)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		out = self.drop_out(out)
		out = self.fc1(out)
		out = self.fc2(out)
		return out

"""
MNIST DATA
"""
dataset = np.load('data/mnist.npz')
valid_idx = np.random.choice(range(60000), size=1000, replace=False) # Sample 6K validation set
x_valid = dataset['x_train'][valid_idx]
y_valid = dataset['y_train'][valid_idx]
x_train = np.delete(dataset['x_train'], valid_idx, axis=0)
y_train = np.delete(dataset['y_train'], valid_idx)
x_test = dataset['x_test']
y_test = dataset['y_test']

"""
Add Noise to training data
"""
y_train_noisy = mnist_noise(y_train,0.2)

"""
Plot an example
""" 
# plt.imshow(x_train[0], cmap='gray')
# plt.title('%i' % y_train_noisy[0])
# plt.show()


"""
Train CNN
"""
cnn = CNN()
cnn.to(device)

num_iter = 10
lr = 0.001
batch_size = 100
L2 = 0.0005

optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, weight_decay=L2)
loss_func = nn.CrossEntropyLoss()
train_idx = np.arange(len(x_train))

x_train = np.transpose(x_train,(2,1,0))
x_valid = np.transpose(x_valid,(2,1,0))
x_test = np.transpose(x_test,(2,1,0))
x_train_tensor = torchvision.transforms.ToTensor()(x_train).unsqueeze(1)
x_valid_tensor = torchvision.transforms.ToTensor()(x_valid).unsqueeze(1)
x_test_tensor = torchvision.transforms.ToTensor()(x_test).unsqueeze(1)
y_train_tensor = torch.from_numpy(y_train.astype(np.long))
y_train_noisy_tensor = torch.from_numpy(y_train_noisy.astype(np.long))
y_valid_tensor = torch.from_numpy(y_valid.astype(np.long))
y_test_tensor = torch.from_numpy(y_test.astype(np.long))



"""
Part1: train on clean data
"""
if False:
	tensor_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
	train_loader= Data.DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=True)

	print("First, try training on clean dataset")

	for epoch in range(num_iter):
		for step, (b_x, b_y) in enumerate(train_loader):
			b_x, b_y = b_x.to(device), b_y.to(device, torch.long)
			output = cnn(b_x)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			test_output = cnn(x_test_tensor.to(device))
			test_output_y = torch.max(test_output, 1)[1].data.cpu().numpy()
			test_accuracy = float((test_output_y == y_test_tensor.data.numpy()).astype(int).sum()) / float(y_test_tensor.size(0))
			print('epoch ', str(epoch), ' test accuracy is ', test_accuracy)

#TODO: Save model


"""
Part2: train on noisy data
"""
if False:
	print('---'*50)
	print("Then, try training on noisy dataset")

	cnn = CNN()
	cnn.to(device)
	optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, weight_decay=L2)

	tensor_noisy_dataset = Data.TensorDataset(x_train_tensor,y_train_noisy_tensor)
	train_noisy_loader= Data.DataLoader(dataset=tensor_noisy_dataset, batch_size=batch_size, shuffle=True)

	for epoch in range(num_iter):
		for step, (b_x, b_y) in enumerate(train_noisy_loader):
			b_x, b_y = b_x.to(device), b_y.to(device)
			output = cnn(b_x)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			test_output = cnn(x_test_tensor.to(device))
			test_output_y = torch.max(test_output, 1)[1].data.cpu().numpy()
			test_accuracy = float((test_output_y == y_test_tensor.data.numpy()).astype(int).sum()) / float(y_test_tensor.size(0))
			print('epoch ', str(epoch), ' test accuracy is ', test_accuracy)

#TODO: Save model


"""
Part3: use trajectorysampler
"""

if True:
	print('---'*50)
	print("Finally, try trajectorysampler on noisy dataset")

	#starting for 2nd epoch
	clu = GaussianMixture(6,2)

	cnn = CNN()
	cnn.to(device)
	optimizer = torch.optim.Adam(cnn.parameters(), lr=lr, weight_decay=L2)

	tensor_noisy_dataset = Data.TensorDataset(x_train_tensor,y_train_noisy_tensor)
	train_noisy_loader= Data.DataLoader(dataset=tensor_noisy_dataset, batch_size=batch_size, shuffle=False)

	corr_prob = []
	"""
	burn-in epoch
	"""

	for epoch in range(2):
		print('epoch = {}'.format(epoch+1))
		for step, (b_x, b_y) in enumerate(train_noisy_loader):
			b_x, b_y = b_x.to(device), b_y.to(device)
			output = cnn(b_x)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			train_output = []
			for step, (b_x, b_y) in enumerate(train_noisy_loader):
				b_x = b_x.to(device)
				train_output.extend(cnn(b_x.to(device)).data.cpu().numpy().tolist())
			corr_prob.append(correct_prob(train_output,y_train))

	corr_prob = np.array(corr_prob).T
	clu.fit(corr_prob,iprint=True)
	cluster = clu.predict(corr_prob,prob=False)

	for epoch in range(8):
		print('epoch = {}'.format(epoch+3))
		eval_grads = []
		valid_output = cnn(x_valid_tensor.to(device))
		valid_loss = loss_func(valid_output, y_valid_tensor.to(device))
		optimizer.zero_grad()
		valid_loss.backward()
		for w in cnn.parameters():
			if w.requires_grad:
				eval_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))
		eval_grads = np.array(eval_grads)

		for com in range(6):
			subset_grads = []
			x_cluster = x_train[:,:,cluster==com]
			y_cluster = y_train_noisy_tensor.data.cpu().numpy()[cluster==com]
			
			size = y_cluster.shape[0]
			sample_size = min(int(size/4), 2000)
			sample_idx = np.random.choice(range(size), sample_size, replace=False).tolist()
			x_subset = x_cluster[:,:,sample_idx]
			y_subset = y_cluster[sample_idx]


			x_train_subset_tensor = torchvision.transforms.ToTensor()(x_subset).unsqueeze(1)
			y_train_subset_tensor = torch.from_numpy(y_subset.astype(np.long))
			subset_output = cnn(x_train_subset_tensor.to(device))
			subset_loss = loss_func(subset_output, y_train_subset_tensor.to(device))
			optimizer.zero_grad()
			subset_loss.backward()
			for w in cnn.parameters():
				if w.requires_grad:
					subset_grads.extend(list(w.grad.cpu().detach().numpy().flatten()))
			subset_grads = np.array(subset_grads)
			sim = 1 - spatial.distance.cosine(eval_grads, subset_grads)

			### TODO: How to use sim?

		for step, (b_x, b_y) in enumerate(train_noisy_loader):
			b_x, b_y = b_x.to(device), b_y.to(device)
			output = cnn(b_x)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		with torch.no_grad():
			train_output = []
			for step, (b_x, b_y) in enumerate(train_noisy_loader):
				b_x = b_x.to(device)
				train_output.extend(cnn(b_x.to(device)).data.cpu().numpy().tolist())
			tmp = np.array(correct_prob(train_output,y_train)).reshape(-1,1)
			print(tmp.shape)
			corr_prob = np.append(corr_prob,tmp,1)
			print(corr_prob.shape)

		clu.append_fit(corr_prob,1,iprint=True)
		cluster = clu.predict(corr_prob,prob=False)