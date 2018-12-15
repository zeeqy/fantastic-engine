import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

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
valid_idx = np.random.choice(range(60000), size=10000, replace=False)
x_valid = dataset['x_train'][valid_idx]
y_valid = dataset['y_train'][valid_idx]
x_train = np.delete(dataset['x_train'], valid_idx, axis=0)
y_train = np.delete(dataset['y_train'], valid_idx)

#print(x_valid.shape,y_valid.shape,x_train.shape,y_train.shape)

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

# add 50% noise to training data
y_train_noisy = noise(y_train,0.5)


"""
Train CNN
"""
cnn = CNN()

num_iter = 80
lr = 0.01
batch_size = 64
L2 = 0.0005
momentum = 0.95

optimizer = torch.optim.SGD(cnn.parameters(), lr=lr, momentum=momentum, weight_decay=L2)
loss_func = nn.CrossEntropyLoss()
train_idx = np.arange(len(x_train))

#x_train_tensor = torch.from_numpy(x_train.astype(np.float32))
x_train = np.transpose(x_train,(2,1,0))
x_train_tensor = torchvision.transforms.ToTensor()(x_train).unsqueeze(1)
y_train_tensor = torch.from_numpy(y_train.astype(np.int))
y_train_noisy_tensor = torch.from_numpy(y_train_noisy.astype(np.int))

tensor_dataset = Data.TensorDataset(x_train_tensor,y_train_tensor)
train_loader= Data.DataLoader(dataset=tensor_dataset, batch_size=batch_size, shuffle=True)


for epoch in range(num_iter):
	for step, (b_x, b_y) in enumerate(train_loader):
		output = cnn(b_x)
		loss = loss_func(output, b_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	output = cnn(x_train_tensor)
	print("CNN loss at {} epoch = {}".format(epoch+1, loss_func(output, y_train_tensor)))









