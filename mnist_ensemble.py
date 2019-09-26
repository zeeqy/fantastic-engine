import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import numpy as np
import json, time, sys
import copy

from networks import *

from trajectoryPlugin.plugin import API

def train_fn(model, device, optimizer, api, reweight=False):
	model.train()
	batch_valid = {'batch_loss':[], 'batch_accuracy':[]}
	for batch_idx, (data, target, weight) in enumerate(api.train_loader):
		data, target, weight = data.to(device), target.to(device), weight.to(device)
		optimizer.zero_grad()
		output = model(data)
		if reweight:
			loss = api.loss_func(output, target, weight, 'mean')
		else:
			loss = api.loss_func(output, target, None, 'mean')
		loss.backward()
		optimizer.step()
		loss, accuracy = valid_fn(model, device, api)
		batch_valid['batch_loss'].append(loss)
		batch_valid['batch_accuracy'].append(accuracy)
	optimizer.zero_grad()
	return batch_valid

def train_loss_fn(model, device, api, reweight=False):
	model.eval()
	loss = 0
	correct = 0
	with torch.no_grad():
		for batch_idx, (data, target, weight) in enumerate(api.train_loader):
			data, target, weight = data.to(device), target.to(device), weight.to(device)
			output = model(data)
			if reweight:
				loss += api.loss_func(output, target, weight, 'sum').item() # sum up batch loss
			else:
				loss += api.loss_func(output, target, None, 'sum').item()
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()
	
	loss /= len(api.train_loader.dataset)
	accuracy = 100. * correct / len(api.train_loader.dataset)

	return loss, accuracy

def valid_fn(model, device, api):
	model.eval()
	loss = 0
	correct = 0
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

def test_fn(model, device, api, test_loader):
	model.eval()
	loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss += api.loss_func(output, target, None, 'sum').item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			correct += pred.eq(target.view_as(pred)).sum().item()

	loss /= len(test_loader.dataset)
	accuracy = 100. * correct / len(test_loader.dataset)

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
	parser.add_argument('--noise_level', type=float, default=0.1, help='percentage of noise data (default: 0.1)')
	parser.add_argument('--num_cluster', type=int, default=3, help='number of cluster (default: 3)')
	parser.add_argument('--reweight_interval', type=int, default=1, help='number of epochs between reweighting')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
	parser.add_argument('--weight_update_rate', type=float, default=0.1, help='weight update rate (default: 0.1)')
	parser.add_argument('--save_model', action='store_true', default=False, help='For Saving the current Model')
	
	args = parser.parse_args()

	timestamp = int(time.time())

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	if args.seed != 0:
		np.random.seed(args.seed)
		torch.manual_seed(args.seed)
		if device != 'cpu':
			torch.cuda.manual_seed(args.seed)
			torch.cuda.manual_seed_all(args.seed)

	mnistdata = datasets.MNIST('../data', train=True, download=True,
				 transform=transforms.Compose([
					 transforms.ToTensor(),
					 transforms.Normalize((0.1307,), (0.3081,))
				 ]))

	test_loader = torch.utils.data.DataLoader(
		datasets.MNIST('../data', train=False, transform=transforms.Compose([
						   transforms.ToTensor(),
						   transforms.Normalize((0.1307,), (0.3081,))
					   ])),
		batch_size=args.batch_size, shuffle=True)
	
	if args.seed != 0:
		np.random.seed(args.seed)

	valid_index = np.random.choice(range(len(mnistdata)), size=args.valid_size, replace=False).tolist()
	train_index = np.delete(range(len(mnistdata)), valid_index).tolist()
	trainset = torch.utils.data.dataset.Subset(mnistdata, train_index)
	validset = torch.utils.data.dataset.Subset(mnistdata, valid_index)

	#nosiy data
	if args.noise_level == 0:
		noise_idx = []
	else:
		if args.seed != 0:
			np.random.seed(args.seed)
		noise_idx = np.random.choice(range(len(trainset)), size=int(len(trainset)*args.noise_level), replace=False)
		label = range(10)
		for idx in noise_idx:
			true_label = trainset.dataset.targets[train_index[idx]]
			noise_label = [lab for lab in label if lab != true_label]
			trainset.dataset.targets[train_index[idx]] = int(np.random.choice(noise_label))

	model_standard = ConvNet1()
	if torch.cuda.device_count() > 1:
		model_standard = nn.DataParallel(model_standard)
	model_standard.to(device)
	#optimizer_standard = optim.SGD(model_standard.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
	optimizer_standard = optim.Adam(model_standard.parameters(), lr=args.lr)

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

	api = API(num_cluster=args.num_cluster, device=device, update_rate=args.weight_update_rate, iprint=2)
	api.dataLoader(trainset, validset, batch_size=args.batch_size)
	#scheduler_standard = torch.optim.lr_scheduler.StepLR(optimizer_standard, step_size=1, gamma=0.95)

	for epoch in range(1, args.burn_in + 1):

		api.log('| - ' + 'at epoch {}, standard lr = {}'.format(epoch, optimizer_standard.param_groups[0]['lr']),2)

		#scheduler_standard.step()
		train_fn(model_standard, device, optimizer_standard, api, False)
		api.createTrajectory(model_standard)
		
		loss, accuracy = train_loss_fn(model_standard, device, api, False)
		standard_train_loss.append(loss)
		standard_train_accuracy.append(accuracy)
		reweight_train_loss.append(loss)
		reweight_train_accuracy.append(accuracy)
		
		loss, accuracy = valid_fn(model_standard, device, api)
		standard_valid_loss.append(loss)
		standard_valid_accuracy.append(accuracy)
		reweight_valid_loss.append(loss)
		reweight_valid_accuracy.append(accuracy)

		loss, accuracy = test_fn(model_standard, device, api, test_loader)
		standard_test_loss.append(loss)
		standard_test_accuracy.append(accuracy)
		reweight_test_loss.append(loss)
		reweight_test_accuracy.append(accuracy)

		api.generateTrainLoader()

		sys.stdout.flush()
	
	model_standard.train()
	optimizer_standard.zero_grad()
	model_reweight = ConvNet1()
	if torch.cuda.device_count() > 1:
		model_reweight = nn.DataParallel(model_reweight)
	model_reweight.load_state_dict(model_standard.state_dict())
	model_reweight.to(device)

	#optimizer_reweight = optim.SGD(model_reweight.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
	optimizer_reweight = optim.Adam(model_reweight.parameters(), lr=args.lr)
	optimizer_reweight.load_state_dict(optimizer_standard.state_dict())
	#scheduler_reweight = torch.optim.lr_scheduler.StepLR(optimizer_reweight, step_size=1, gamma=0.95, last_epoch=args.burn_in)
	# optimizer_reweight.param_groups[0]['lr'] = optimizer_standard.param_groups[0]['lr']

	# for p1, p2 in zip(model_standard.parameters(), model_reweight.parameters()):
	# 	if p1.data.ne(p2.data).sum() > 0:
	# 		api.log('| - ' + 'some parts of reweight model initialized incorrectly',2)

	epoch_reweight = []
	epoch_trajectory = []

	api.trajectoryBins()
	api.clusterBins()
	api.reweightData(model_reweight, noise_idx)
	api.generateTrainLoader()

	epoch_reweight.append({'epoch':epoch, 'weight_tensor':api.weight_tensor.data.cpu().numpy().tolist()})
	mean_trajectory = {}
	for cid in range(api.num_cluster):
		cidx = (api.cluster_output==cid).nonzero()[0].tolist()
		mean_trajectory.update({cid:np.mean(api.traject_bins[cidx], axis=0).tolist()})
	epoch_trajectory.append({'epoch':epoch, 'trajectory':mean_trajectory})

	standard_batch_valid = []
	reweight_batch_valid = []

	model_standard.train()
	optimizer_standard.zero_grad()
	model_reweight.train()
	optimizer_reweight.zero_grad()

	for epoch in range(args.burn_in + 1, args.epochs + 1):

		#api.log('| - ' + 'at epoch {}, standard lr = {}, reweight lr = {}'.format(epoch, optimizer_standard.param_groups[0]['lr'], optimizer_reweight.param_groups[0]['lr']),2)

		#scheduler_standard.step()
		batch_valid = train_fn(model_standard, device, optimizer_standard, api, False)
		standard_batch_valid.append({'epoch':epoch,'data':batch_valid})
		
		loss, accuracy = train_loss_fn(model_standard, device, api, False)
		standard_train_loss.append(loss)
		standard_train_accuracy.append(accuracy)
		
		loss, accuracy = valid_fn(model_standard, device, api)
		standard_valid_loss.append(loss)
		standard_valid_accuracy.append(accuracy)

		loss, accuracy = test_fn(model_standard, device, api, test_loader)
		standard_test_loss.append(loss)
		standard_test_accuracy.append(accuracy)

		#scheduler_reweight.step()
		batch_valid = train_fn(model_reweight, device, optimizer_reweight, api, True)
		reweight_batch_valid.append({'epoch':epoch,'data':batch_valid})
		api.createTrajectory(model_reweight)

		loss, accuracy = train_loss_fn(model_reweight, device, api, True)
		reweight_train_loss.append(loss)
		reweight_train_accuracy.append(accuracy)
		
		loss, accuracy = valid_fn(model_reweight, device, api)
		reweight_valid_loss.append(loss)
		reweight_valid_accuracy.append(accuracy)

		loss, accuracy = test_fn(model_reweight, device, api, test_loader)
		reweight_test_loss.append(loss)
		reweight_test_accuracy.append(accuracy)

		if epoch >= args.burn_in and (epoch - args.burn_in) % args.reweight_interval == 0:
			
			model_standard.train()
			optimizer_standard.zero_grad()
			model_reweight.train()
			optimizer_reweight.zero_grad()

			api.trajectoryBins()
			api.clusterBins()
			api.reweightData(model_reweight, noise_idx)
			epoch_reweight.append({'epoch':epoch, 'weight_tensor':api.weight_tensor.data.cpu().numpy().tolist()})
			mean_trajectory = {}
			for cid in range(api.num_cluster):
				cidx = (api.cluster_output==cid).nonzero()[0].tolist()
				mean_trajectory.update({cid:np.mean(api.traject_bins[cidx], axis=0).tolist()})
			epoch_trajectory.append({'epoch':epoch, 'trajectory':mean_trajectory})
		api.generateTrainLoader()
		sys.stdout.flush()

		model_standard.train()
		optimizer_standard.zero_grad()
		model_reweight.train()
		optimizer_reweight.zero_grad()

	if (args.save_model):
		torch.save(model.state_dict(),"mnist_cnn_ensemble.pt")

	res = vars(args)

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

	with open('mnist_experiments/mnist_cnn_baseline_reweight_response.data', 'a+') as f:
		f.write(json.dumps(res) + '\n')
	f.close()

	with open('mnist_experiments/weights/mnist_cnn_baseline_reweight_{}.data'.format(timestamp), 'a+') as f:
		for ws in epoch_reweight:
			f.write(json.dumps(ws) + '\n')
	f.close()

	with open('mnist_experiments/trajectory/mnist_cnn_baseline_trajectory_{}.data'.format(timestamp), 'a+') as f:
		for tr in epoch_trajectory:
			f.write(json.dumps(tr) + '\n')
	f.close()

	with open('mnist_experiments/weights/mnist_cnn_baseline_batch_valid_{}.data'.format(timestamp), 'a+') as f:
		for vad in standard_batch_valid:
			f.write(json.dumps(vad) + '\n')
	f.close()

	with open('mnist_experiments/weights/mnist_cnn_reweight_batch_valid_{}.data'.format(timestamp), 'a+') as f:
		for vad in reweight_batch_valid:
			f.write(json.dumps(vad) + '\n')
	f.close()

if __name__ == '__main__':
	main()