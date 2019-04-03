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
	parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 10)')
	parser.add_argument('--depth', default=28, type=int, help='depth of model')
	parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
	parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
	parser.add_argument('--noise_level', type=float, default=0.1, help='percentage of noise data (default: 0.1)')
	parser.add_argument('--valid_size', type=int, default=1000, help='input validation size (default: 1000)')
	parser.add_argument('--dropout', default=0.3, type=float, help='dropout_rate')
	parser.add_argument('--num_cluster', type=int, default=3, help='number of cluster (default: 3)')
	parser.add_argument('--reweight_interval', type=int, default=1, help='number of epochs between reweighting')
	parser.add_argument('--burn_in', type=int, default=5, help='number of burn-in epochs (default: 5)')
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

	test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

	# Read same valid index for consistence
	with open('cifar10_experiments/valid_index.data', 'r') as f:
		valid_json = json.loads(f.read())
	f.close()

	valid_index = valid_json['valid_index']
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
	
	model_reweight = WideResNet(args.depth, num_classes, args.widen_factor, args.dropout)
	if torch.cuda.device_count() > 1:
		model_reweight = nn.DataParallel(model_reweight)
	model_reweight.to(device)
	optimizer_reweight = optim.SGD(model_reweight.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

	reweight_train_loss = []
	reweight_train_accuracy = []
	reweight_valid_loss = []
	reweight_valid_accuracy = []
	reweight_test_loss = []
	reweight_test_accuracy = []

	api = API(num_cluster=args.num_cluster, device=device, iprint=2)
	api.dataLoader(trainset, validset, batch_size=args.batch_size)
	scheduler_reweight = torch.optim.lr_scheduler.MultiStepLR(optimizer_reweight, milestones=[60,120,160], gamma=0.2)
	epoch_reweight = []

	for epoch in range(1, args.epochs + 1):

		scheduler_reweight.step()
		train_fn(model_reweight, device, optimizer_reweight, api)
		api.createTrajectory(model_reweight)
		if epoch >= args.burn_in and (epoch - args.burn_in) % args.reweight_interval == 0:
			api.clusterTrajectory() 
			api.reweightData(model_reweight, 1e6, noise_idx)
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


	res = vars(args)
	timestamp = int(time.time())

	res.update({'reweight_train_loss':reweight_train_loss})
	res.update({'reweight_train_accuracy':reweight_train_accuracy})
	res.update({'reweight_valid_loss':reweight_valid_loss})
	res.update({'reweight_valid_accuracy':reweight_valid_accuracy})
	res.update({'reweight_test_loss':reweight_test_loss})
	res.update({'reweight_test_accuracy':reweight_test_accuracy})
	
	res.update({'timestamp': timestamp})

	with open('cifar10_experiments/cifar10_wideresnet_reweight_response.data', 'a+') as f:
		f.write(json.dumps(res) + '\n')
	f.close()

	with open('cifar10_experiments/weights/cifar10_wideresnet_baseline_reweight_{}.data'.format(timestamp), 'a+') as f:
		for ws in epoch_reweight:
			f.write(json.dumps(ws) + '\n')
	f.close()

if __name__ == '__main__':
	main()