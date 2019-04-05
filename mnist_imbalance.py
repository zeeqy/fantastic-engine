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
	for batch_idx, (data, target, idx) in enumerate(api.train_loader):
		data, target = data.to(device), target.to(device)
		weight = api.weight_tensor[idx].to(device)
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
			for batch_idx, (data, target, idx) in enumerate(api.train_loader): 
				data, target = data.to(device), target.to(device)
				weight = api.weight_tensor[idx].to(device)
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
	parser = argparse.ArgumentParser(description='MNIST Baseline Reweight Comparison')
	parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
	parser.add_argument('--burn_in', type=int, default=5, help='number of burn-in epochs (default: 5)')
	parser.add_argument('--valid_size', type=int, default=1000, help='input validation size (default: 1000)')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
	parser.add_argument('--imbalance_rate', type=float, default=0, help='percentage of imbalance (default: 0.1)')
	parser.add_argument('--num_cluster', type=int, default=3, help='number of cluster (default: 3)')
	parser.add_argument('--reweight_interval', type=int, default=1, help='number of epochs between reweighting')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
	parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
	
	args = parser.parse_args()

	if args.seed != 0:
		torch.manual_seed(args.seed)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	mnistdata = datasets.MNIST('../data', train=True, download=True,
				 transform=transforms.Compose([
					 transforms.ToTensor(),
					 transforms.Normalize((0.1307,), (0.3081,))
				 ]))

	testdata = datasets.MNIST('../data', train=False, transform=transforms.Compose([
					transforms.ToTensor(),
					transforms.Normalize((0.1307,), (0.3081,))
				]))
	test_49 = [i for i, e in enumerate(testdata.targets) if e == 4 or e == 9]
	testset = torch.utils.data.dataset.Subset(testdata, test_49)
	test_loader =  torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True)

	mnist_4 = [i for i, e in enumerate(mnistdata.targets) if e == 4]
	mnist_9 = [i for i, e in enumerate(mnistdata.targets) if e == 9]

	valid_mnist_4 = np.random.choice(mnist_4, size=args.valid_size//2, replace=False).tolist()
	valid_mnist_9 = np.random.choice(mnist_9, size=args.valid_size//2, replace=False).tolist()
	valid_index = valid_mnist_9 + valid_mnist_4

	mnist_4 = np.delete(mnist_4, valid_mnist_4).tolist()
	mnist_9 = np.delete(mnist_9, valid_mnist_9).tolist()
	mnist_4 = np.random.choice(mnist_4, size=int(args.imbalance_rate * len(mnist_9)), replace=False).tolist()

	trainset = torch.utils.data.dataset.Subset(mnistdata, mnist_4 + mnist_9)
	validset = torch.utils.data.dataset.Subset(mnistdata, valid_index)

	model_standard = LeNet()
	if torch.cuda.device_count() > 1:
		model_standard = nn.DataParallel(model_standard)
	model_standard.to(device)
	optimizer_standard = optim.SGD(model_standard.parameters(), lr=args.lr, momentum=args.momentum)

	model_reweight = LeNet()
	if torch.cuda.device_count() > 1:
		model_reweight = nn.DataParallel(model_reweight)
	model_reweight.to(device)
	optimizer_reweight = optim.SGD(model_reweight.parameters(), lr=args.lr, momentum=args.momentum)
	
	standard_train_loss = []
	standard_train_accuracy = []
	standard_valid_loss = []
	standard_valid_accuracy = []
	standard_test_loss = []
	standard_test_accuracy = []

	reweight_train_loss = []
	reweight_train_accuracy = []
	reweight_valid_loss = []
	reweight_valid_accuracy = []
	reweight_test_loss = []
	reweight_test_accuracy = []

	api = API(num_cluster=args.num_cluster, device=device, iprint=2)
	api.dataLoader(trainset, validset, batch_size=args.batch_size)
	scheduler_standard = torch.optim.lr_scheduler.StepLR(optimizer_standard, step_size=1, gamma=0.95)

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

	api = API(num_cluster=args.num_cluster, device=device, iprint=2)
	api.dataLoader(trainset, validset, batch_size=args.batch_size)
	scheduler_reweight = torch.optim.lr_scheduler.StepLR(optimizer_reweight, step_size=1, gamma=0.95)
	epoch_reweight = []

	for epoch in range(1, args.epochs + 1):

		scheduler_reweight.step()
		train_fn(model_reweight, device, optimizer_reweight, api)
		api.createTrajectory(model_reweight)
		if epoch >= args.burn_in and (epoch - args.burn_in) % args.reweight_interval == 0:
			api.clusterTrajectory() 
			api.reweightData(model_reweight, 1e6)
			epoch_reweight.append({'epoch':epoch, 'weight_tensor':api.weight_tensor.data.cpu().numpy().tolist()})

		loss, accuracy = forward_fn(model_reweight, device, api, 'train')
		reweight_train_loss.append(loss)
		reweight_train_accuracy.append(accuracy)
		
		loss, accuracy = forward_fn(model_reweight, device, api, 'validation')
		reweight_valid_loss.append(loss)
		reweight_valid_accuracy.append(accuracy)

		loss, accuracy = forward_fn(model_reweight, device, api, 'test', test_loader)
		reweight_test_loss.append(loss)
		reweight_test_accuracy.append(accuracy)

	if (args.save_model):
		torch.save(model.state_dict(),"mnist_imbalance_baseline_reweight.pt")

	res = vars(args)
	timestamp = int(time.time())

	res.update({'standard_train_loss':standard_train_loss})
	res.update({'standard_train_accuracy':standard_train_accuracy})
	res.update({'standard_valid_loss':standard_valid_loss})
	res.update({'standard_valid_accuracy':standard_valid_accuracy})
	res.update({'standard_test_loss':standard_test_loss})
	res.update({'standard_test_accuracy':standard_test_accuracy})

	res.update({'reweight_train_loss':reweight_train_loss})
	res.update({'reweight_train_accuracy':reweight_train_accuracy})
	res.update({'reweight_valid_loss':reweight_valid_loss})
	res.update({'reweight_valid_accuracy':reweight_valid_accuracy})
	res.update({'reweight_test_loss':reweight_test_loss})
	res.update({'reweight_test_accuracy':reweight_test_accuracy})
	
	res.update({'timestamp': timestamp})

	with open('mnist_experiments/mnist_imbalance_baseline_reweight_response.data', 'a+') as f:
		f.write(json.dumps(res) + '\n')
	f.close()

	with open('mnist_experiments/weights/mnist_imbalance_baseline_reweight_{}.data'.format(timestamp), 'a+') as f:
		for ws in epoch_reweight:
			f.write(json.dumps(ws) + '\n')
	f.close()

if __name__ == '__main__':
	main()