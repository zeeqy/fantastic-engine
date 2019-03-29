import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import numpy as np
import json, time

from networks import *

from trajectoryPlugin.plugin import API

def train_fn(model, device, optimizer, api):
	model.train()
	for batch_idx, (data, target, weight) in enumerate(api.train_loader):
		data, target, weight = data.to(device), target.to(device), weight.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = api.loss_func(output, target, weight, 'mean')
		loss.backward()
		optimizer.step()

def forward_fn(model, device, api, forward_type, test_loader=None):
	model.eval()
	loss = 0
	correct = 0
	if forward_type == 'test':
		with torch.no_grad():
			for data, target in test_loader:
				data, target = data.to(device), target.to(device)
				output = model(data)
				loss += api.loss_func(output, target, None, 'sum').item() # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()
		
		loss /= len(test_loader.dataset)
		accuracy = 100. * correct / len(test_loader.dataset)
	
	elif forward_type == 'train':
		with torch.no_grad():
			for batch_idx, (data, target, weight) in enumerate(api.train_loader): 
				data, target, weight = data.to(device), target.to(device), weight.to(device)
				output = model(data)
				loss += api.loss_func(output, target, weight, 'sum').item() # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

		loss /= len(api.train_loader.dataset)
		accuracy = 100. * correct / len(api.train_loader.dataset)
				
	elif forward_type == 'validation':
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(api.valid_loader): 
				data, target = data.to(device), target.to(device)
				output = model(data)
				loss += api.loss_func(output, target, None, 'sum').item() # sum up batch loss
				pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

		loss /= len(api.valid_loader.dataset)
		accuracy = 100. * correct / len(api.valid_loader.dataset)

	return loss, accuracy

def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Baseline Training')
	parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
	parser.add_argument('--batch_size', type=int, default=128, help='input batch size for training (default: 128)')
	parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 10)')
	parser.add_argument('--depth', default=28, type=int, help='depth of model')
	parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
	parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
	parser.add_argument('--noise_level', type=float, default=0.1, help='percentage of noise data (default: 0.1)')
	parser.add_argument('--valid_size', type=int, default=1000, help='input validation size (default: 1000)')
	parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')	
	
	args = parser.parse_args()

	if args.seed != 0:
		torch.manual_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	transform_train = transforms.Compose([
		transforms.RandomCrop(32, padding=4),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	transform_test = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	])

	cifardata = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)
	testset = datasets.CIFAR10(root='../data', train=False, download=False, transform=transform_test)
	num_classes = 10

	testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	valid_index = np.random.choice(range(len(cifardata)), size=args.valid_size, replace=False).tolist()
	train_index = np.delete(range(len(cifardata)), valid_index).tolist()
	trainset = torch.utils.data.dataset.Subset(cifardata, train_index)
	validset = torch.utils.data.dataset.Subset(cifardata, valid_index)

	#nosiy data
	if args.noise_level == 0:
		noise_idx = []
	else:
		noise_idx = np.random.choice(range(len(trainset)), size=int(len(trainset)*args.noise_level), replace=False)
		label = range(10)
		for idx in noise_idx:
			true_label = trainset.dataset.targets[train_index[idx]]
			noise_label = [lab for lab in label if lab != true_label]
			trainset.dataset.targets[train_index[idx]] = int(np.random.choice(noise_label))
	
	Net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
	model_standard = Net().to(device)
	optimizer_standard = optim.SGD(model_standard.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
	
	def learning_rate(epoch):
		optim_factor = 0
		if(epoch > 160):
			optim_factor = 3
		elif(epoch > 120):
			optim_factor = 2
		elif(epoch > 60):
			optim_factor = 1

		return args.lr*math.pow(0.2, optim_factor)

	standard_train_loss = []
	standard_train_accuracy = []
	standard_valid_loss = []
	standard_valid_accuracy = []
	standard_test_loss = []
	standard_test_accuracy = []

	api = API(num_cluster=args.num_cluster, device=device, iprint=2)
	api.dataLoader(trainset, validset, batch_size=args.batch_size)
	
	scheduler_standard = torch.optim.lr_scheduler.LambdaLR(optimizer_standard, learning_rate)

	for epoch in range(1, args.epochs + 1):

		scheduler_standard.step()
		train_fn(model_standard, device, optimizer_standard, api)
		
		loss, accuracy = forward_fn(model_standard, device, api, 'train')
		standard_train_loss.append(loss)
		standard_train_accuracy.append(accuracy)
		
		loss, accuracy = forward_fn(model_standard, device, api, 'validation')
		standard_valid_loss.append(loss)
		standard_valid_accuracy.append(accuracy)

		loss, accuracy = forward_fn(model_standard, device, api, 'test', test_loader)
		standard_test_loss.append(loss)
		standard_test_accuracy.append(accuracy)


	res = vars(args)
	timestamp = int(time.time())

	res.update({'standard_train_loss':standard_train_loss})
	res.update({'standard_train_accuracy':standard_train_accuracy})
	res.update({'standard_valid_loss':standard_valid_loss})
	res.update({'standard_valid_accuracy':standard_valid_accuracy})
	res.update({'standard_test_loss':standard_test_loss})
	res.update({'standard_test_accuracy':standard_test_accuracy})
	
	res.update({'timestamp': timestamp})

	with open('cifar10_experiments/cifar10_wildresnet_baseline_response.data', 'a+') as f:
		f.write(json.dumps(res) + '\n')
	f.close()

if __name__ == '__main__':
	main()





