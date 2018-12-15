import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

"""
CNN
"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),           
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),
        )
        
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.out(x)


"""
MNIST DATA
"""
dataset = np.load('data/mnist.npz')
valid_idx = np.random.choice(range(60000), size=5000, replace=False) # Sample 5K validation set
x_valid = dataset['x_train'][valid_idx]
y_valid = dataset['y_train'][valid_idx]
x_train = np.delete(dataset['x_train'], valid_idx, axis=0)
y_train = np.delete(dataset['y_train'], valid_idx)

"""
Save memory (Full data need at least 16G GPU RAM and 32G RAM)
"""
x_train = x_train[0:5000]
y_train = y_train[0:5000]


"""
Plot an example
""" 
# plt.imshow(x_train[0], cmap='gray')
# plt.title('%i' % y_train[0])
# plt.show()


"""
Add Noise to training data
"""
def noise(y_train, ratio=0.5):
	label = range(10)
	label_copy = np.copy(y_train)
	noise_idx = np.random.choice(range(len(y_train)), size=int(len(y_train)*ratio), replace=False)
	for idx in noise_idx:
		noise_label = [lab for lab in label if lab != y_train[idx]]
		label_copy[idx] = np.random.choice(noise_label, size=1)
	return label_copy

# Add 50% noise to training data
y_train_noisy = noise(y_train,0.5)


"""
Train CNN
"""
cnn = CNN()
cnn.to(device)

num_iter = 80
lr = 0.01
batch_size = 50
L2 = 0.0005
momentum = 0.95
tol = 1e-5

optimizer = torch.optim.SGD(cnn.parameters(), lr=lr, momentum=momentum, weight_decay=L2)
loss_func = nn.CrossEntropyLoss()
train_idx = np.arange(len(x_train))

x_train = np.transpose(x_train,(2,1,0))
x_valid = np.transpose(x_valid,(2,1,0))
x_train_tensor = torchvision.transforms.ToTensor()(x_train).unsqueeze(1)
x_valid_tensor = torchvision.transforms.ToTensor()(x_valid).unsqueeze(1)
y_train_tensor = torch.from_numpy(y_train.astype(np.int))
y_valid_tensor = torch.from_numpy(y_valid.astype(np.int))
y_train_noisy_tensor = torch.from_numpy(y_train_noisy.astype(np.int))


"""
Part1: train on clean data
"""

tensor_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
train_loader= Data.DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=True)

previous_loss = np.inf
training_acc = []
valid_acc = []

print("First, try training on clean dataset")

for epoch in range(num_iter):
	for step, (b_x, b_y) in enumerate(train_loader):
		b_x, b_y = b_x.to(device), b_y.to(device)
		output = cnn(b_x)
		loss = loss_func(output, b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	cnn.to('cpu')

	output = cnn(x_train_tensor)
	output_y = torch.max(output, 1)[1].data.numpy()
	train_accuracy = float((output_y == y_train_tensor.data.numpy()).astype(int).sum()) / float(y_train_tensor.size(0))

	valid_output = cnn(x_valid_tensor)
	valid_output_y = torch.max(valid_output, 1)[1].data.numpy()
	valid_accuracy = float((valid_output_y == y_valid_tensor.data.numpy()).astype(int).sum()) / float(y_valid_tensor.size(0))

	epoch_loss = loss_func(output, y_train_tensor)
	if abs(epoch_loss - previous_loss) < tol:
		break
	print("CNN on clean dataset at epoch = {}, loss = {}, training accuracy = {}, validation_accracy = {}".format(epoch+1, epoch_loss, train_accuracy,valid_accuracy))
	cnn.to(device)
	training_acc.append(train_accuracy)
	valid_acc.append(valid_accuracy)
	previous_loss = epoch_loss

#TODO: Save model


"""
Part2: train on noisy data
"""

time.sleep(15)
print('---'*50)
print("Then, try training on noisy dataset")

cnn.to(device)

tensor_noisy_dataset = Data.TensorDataset(x_train_tensor,y_train_noisy_tensor)
train_noisy_loader= Data.DataLoader(dataset=tensor_noisy_dataset, batch_size=batch_size, shuffle=True)

previous_loss = np.inf
training_noisy_acc = []
valid_noisy_acc = []

for epoch in range(num_iter):
	for step, (b_x, b_y) in enumerate(train_noisy_loader):
		b_x, b_y = b_x.to(device), b_y.to(device)
		output = cnn(b_x)
		loss = loss_func(output, b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	cnn.to('cpu')

	output = cnn(x_train_tensor)
	output_y = torch.max(output, 1)[1].data.numpy()
	train_accuracy = float((output_y == y_train_noisy_tensor.data.numpy()).astype(int).sum()) / float(y_train_tensor.size(0))

	valid_output = cnn(x_valid_tensor)
	valid_output_y = torch.max(valid_output, 1)[1].data.numpy()
	valid_accuracy = float((valid_output_y == y_valid_tensor.data.numpy()).astype(int).sum()) / float(y_valid_tensor.size(0))

	epoch_loss = loss_func(output, y_train_tensor)
	if abs(epoch_loss - previous_loss) < tol:
		break
	print("CNN on noisy dataset at epoch = {}, loss = {}, training accuracy = {}, validation_accracy = {}".format(epoch+1, epoch_loss, train_accuracy,valid_accuracy))
	cnn.to(device)
	training_noisy_acc.append(train_accuracy)
	valid_noisy_acc.append(valid_accuracy)
	previous_loss = epoch_loss

#TODO: Save model



