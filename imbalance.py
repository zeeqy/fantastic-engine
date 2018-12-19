import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
from util.util import ImbalancedDatasetSampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def name(self):
        return "LeNet"

"""
MNIST DATA
"""
dataset = np.load('data/mnist.npz')

four_index = dataset['y_train'] == 4
nine_index = dataset['y_train'] == 9

y_fours = dataset['y_train'][four_index]-4
y_nines = dataset['y_train'][nine_index]-8
x_fours = dataset['x_train'][four_index]
x_nines = dataset['x_train'][nine_index]

#print(fours.shape,nines.shape) # 5842 : 5949

valid_four_index = np.random.choice(range(len(y_fours)), size=250, replace=False)
valid_nine_index = np.random.choice(range(len(y_nines)), size=250, replace=False)

x_valid_four =  x_fours[valid_four_index]
x_valid_nine =  x_nines[valid_nine_index]
y_valid_four =  y_fours[valid_four_index]
y_valid_nine =  y_nines[valid_nine_index]

x_valid = np.append(x_valid_four,x_valid_nine,0)
y_valid = np.append(y_valid_four,y_valid_nine)
indices = np.arange(x_valid.shape[0])
np.random.shuffle(indices)
x_valid = x_valid[indices]
y_valid = y_valid[indices]

x_train_four = np.delete(x_fours, valid_four_index, axis=0)
y_train_four = np.delete(y_fours, valid_four_index)
x_train_nine = np.delete(x_nines, valid_four_index, axis=0)
y_train_nine = np.delete(y_nines, valid_four_index)

four_index = dataset['y_test'] == 4
nine_index = dataset['y_test'] == 9

y_fours = dataset['y_test'][four_index]-4
y_nines = dataset['y_test'][nine_index]-8
x_fours = dataset['x_test'][four_index]
x_nines = dataset['x_test'][nine_index]
x_test = np.append(x_fours,x_nines,0)
y_test = np.append(y_fours,y_nines)
indices = np.arange(x_test.shape[0])
np.random.shuffle(indices)
x_test = x_test[indices]
y_test = y_test[indices]

"""
Imbalance MNIST Study
""" 

for ratio in [500,250,125,62,31]:
	print('ratio: 4:9 = {}:{}'.format(ratio, 5000-ratio))
	four_part = np.random.choice(range(len(x_train_four)), size=ratio, replace=False)
	nine_part = np.random.choice(range(len(x_train_nine)), size=5000-ratio, replace=False)
	x_train = np.append(x_train_four[four_part],x_train_nine[nine_part],0)
	y_train = np.append(y_train_four[four_part],y_train_nine[nine_part])
	indices = np.arange(x_train.shape[0])
	np.random.shuffle(indices)
	x_train = x_train[indices]
	y_train = y_train[indices]


	x_train = np.transpose(x_train,(2,1,0))
	x_valid = np.transpose(x_valid,(2,1,0))
	x_test = np.transpose(x_test,(2,1,0))
	x_train_tensor = torchvision.transforms.ToTensor()(x_train).unsqueeze(1)
	x_valid_tensor = torchvision.transforms.ToTensor()(x_valid).unsqueeze(1)
	x_test_tensor = torchvision.transforms.ToTensor()(x_test).unsqueeze(1)
	y_train_tensor = torch.from_numpy(y_train.astype(np.long))
	y_valid_tensor = torch.from_numpy(y_valid.astype(np.long))
	y_test_tensor = torch.from_numpy(y_test.astype(np.long))

	"""
	baseline
	"""

	lenet = LeNet()
	num_iter = 80
	lr = 0.001
	batch_size = 100
	L2 = 0.0005

	tensor_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
	train_loader= Data.DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=True)
	optimizer = torch.optim.Adam(lenet.parameters(), lr=lr, weight_decay=L2)
	loss_func = nn.CrossEntropyLoss()

	for epoch in range(num_iter):
		for step, (b_x, b_y) in enumerate(train_loader):
			b_x, b_y = b_x.to(device), b_y.to(device, torch.long)
			output = lenet(b_x)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	with torch.no_grad():
		test_output = lenet(x_test_tensor.to(device))
		test_output_y = torch.max(test_output, 1)[1].data.cpu().numpy()
		acc = 0
		for pred, y in zip(test_output_y,y_test_tensor.data.numpy()):
			if pred == y:
				acc += 1
		test_accuracy = acc / float(y_test_tensor.size(0))
		print('lenet baseline test error is ', 1-test_accuracy)

	"""
	adjust class weight
	"""
	lenet = LeNet()
	weight = [float(5000/ratio), float(5000/(5000-ratio))]
	weight = torch.FloatTensor(weight).to(device)
	loss_func = nn.CrossEntropyLoss(weight=weight)
	for epoch in range(num_iter):
		for step, (b_x, b_y) in enumerate(train_loader):
			b_x, b_y = b_x.to(device), b_y.to(device, torch.long)
			output = lenet(b_x)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	with torch.no_grad():
		test_output = lenet(x_test_tensor.to(device))
		test_output_y = torch.max(test_output, 1)[1].data.cpu().numpy()
		acc = 0
		for pred, y in zip(test_output_y,y_test_tensor.data.numpy()):
			if pred == y:
				acc += 1
		test_accuracy = acc / float(y_test_tensor.size(0))
		print('lenet with class weight test error is ', 1-test_accuracy)

	"""
	resample to balance
	"""
	lenet = LeNet()
	tensor_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
	train_loader= Data.DataLoader(dataset=tensor_dataset, batch_size=batch_size, sampler=ImbalancedDatasetSampler(tensor_dataset), shuffle=False)
	loss_func = nn.CrossEntropyLoss()
	for epoch in range(num_iter):
		for step, (b_x, b_y) in enumerate(train_loader):
			b_x, b_y = b_x.to(device), b_y.to(device, torch.long)
			output = lenet(b_x)
			loss = loss_func(output, b_y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

	with torch.no_grad():
		test_output = lenet(x_test_tensor.to(device))
		test_output_y = torch.max(test_output, 1)[1].data.cpu().numpy()
		acc = 0
		for pred, y in zip(test_output_y,y_test_tensor.data.numpy()):
			if pred == y:
				acc += 1
		test_accuracy = acc / float(y_test_tensor.size(0))
		print('lenet with imbalance sampling test error is ', test_accuracy)


	"""
	weight model here
	"""





