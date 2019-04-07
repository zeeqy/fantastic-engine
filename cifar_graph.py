import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser(description='CIFAR Graph')
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
	parser.add_argument('--cifar', type=int, default=10, help='which cifar is this?')

	args = parser.parse_args()
	args_dict = vars(args)
	
	with open('cifar_experiments/cifar{}_wideresnet_baseline_response.data'.format(args.cifar), 'r+') as f:
		rec = f.read().split('\n')[:-1]
	f.close()
	

	baseline_keys = ['lr','batch_size','epochs','depth','widen_factor','momentum','noise_level','dropout','seed','valid_size']
	res_1 = []
	for item in rec:
		item_dict = json.loads(item)
		match = True
		for key in baseline_keys:
			if args_dict[key] != item_dict[key]:
				match = False
		if match:
			res_1.append(item_dict)
	
	if len(res_1) == 0:
		print("No config matches, please check your arguments!")
		return None
	
	elif len(res_1) > 1:
		print("More than one trail found, select the most recent one!")
		res_1 = sorted(res_1, key=lambda k: k['timestamp'], reverse=True)[0]
	else:
		res_1 = res_1[0]


	with open('cifar_experiments/cifar{}_wideresnet_reweight_response.data'.format(args.cifar), 'r+') as f:
		rec = f.read().split('\n')[:-1]
	f.close()
	

	reweight_keys = baseline_keys + ['burn_in','reweight_interval','num_cluster']
	res_2 = []
	for item in rec:
		item_dict = json.loads(item)
		match = True
		for key in reweight_keys:
			if args_dict[key] != item_dict[key]:
				match = False
		if match:
			res_2.append(item_dict)
	
	if len(res_2) == 0:
		print("No config matches, please check your arguments!")
		return None
	
	elif len(res_2) > 1:
		print("More than one trail found, select the most recent one!")
		res_2 = sorted(res_2, key=lambda k: k['timestamp'], reverse=True)[0]
	else:
		res_2 = res_2[0]
	
	fig, axs = plt.subplots(2,3, figsize=(20, 10))
	
	x = range(1, res_1['epochs']+1, 1)
	axs[0,0].plot(x, res_1['standard_train_loss'], '--', color='blue', label='Standard')
	axs[0,0].plot(x, res_2['reweight_train_loss'], '--', color='red', label='Reweighted')
	axs[0,0].axvline(x=res_2['burn_in'], linestyle='--', color='black')
	axs[0,0].set_title("Train (Weighted) Loss")
	axs[0,0].legend()

	axs[0,1].plot(x, res_1['standard_valid_loss'], '--', color='blue', label='Standard')
	axs[0,1].plot(x, res_2['reweight_valid_loss'], '--', color='red', label='Reweighted')
	axs[0,1].axvline(x=res_2['burn_in'], linestyle='--', color='black')
	axs[0,1].set_title("Validation Loss")
	axs[0,1].legend()

	axs[0,2].plot(x, res_1['standard_test_loss'], '--', color='blue', label='Standard')
	axs[0,2].plot(x, res_2['reweight_test_loss'], '--', color='red', label='Reweighted')
	axs[0,2].axvline(x=res_2['burn_in'], linestyle='--', color='black')
	axs[0,2].set_title("Test Loss")
	axs[0,2].legend()
	
	axs[1,0].plot(x, res_1['standard_train_accuracy'], '--', color='blue', label='Standard')
	axs[1,0].plot(x, res_2['reweight_train_accuracy'], '--', color='red', label='Reweighted')
	axs[1,0].axvline(x=res_2['burn_in'], linestyle='--', color='black')
	axs[1,0].set_title("Train Accuracy")
	axs[1,0].legend()

	print("standard_valid_accuracy: ", res_1['standard_valid_accuracy'][-5:])
	print("reweight_valid_accuracy: ", res_2['reweight_valid_accuracy'][-5:])
	axs[1,1].plot(x, res_1['standard_valid_accuracy'], '--', color='blue', label='Standard')
	axs[1,1].plot(x, res_2['reweight_valid_accuracy'], '--', color='red', label='Reweighted')
	axs[1,1].axvline(x=res_2['burn_in'], linestyle='--', color='black')
	axs[1,1].set_title("Validation Accuracy")
	axs[1,1].legend()

	print("standard_test_accuracy:", res_1['standard_test_accuracy'][-5:])
	print("reweight_valid_accuracy:", res_2['reweight_test_accuracy'][-5:])
	axs[1,2].plot(x, res_1['standard_test_accuracy'], '--', color='blue', label='Standard')
	axs[1,2].plot(x, res_2['reweight_test_accuracy'], '--', color='red', label='Reweighted')
	axs[1,2].axvline(x=res_2['burn_in'], linestyle='--', color='black')
	axs[1,2].set_title("Test Accuracy")
	axs[1,2].legend()
	
	plt.savefig('figures/loss_accuracy_{}.pdf'.format(res_1['timestamp']), format='pdf', dpi=1000)

	with open('cifar_experiments/weights/cifar{}_wideresnet_baseline_reweight_{}.data'.format(args.cifar, res_2['timestamp']), 'r+') as f:
		rec = f.read().split('\n')[:-1]
	f.close()
	
	grid = int(np.ceil(np.sqrt(len(rec))))
	fig, axs = plt.subplots(grid,grid, figsize=(20, 10))
	i = 0
	j = 0
	for item in rec:
		item_dict = json.loads(item)
		axs[i,j].hist(item_dict['weight_tensor'],bins=10, range=(0,1.5))
		axs[i,j].set_title("Weights Distirbution at {} epoch".format(item_dict['epoch']))
		if j < grid-1:
			j += 1
		else:
			i += 1
			j = 0
	plt.tight_layout()
	plt.savefig('figures/weights_distribution_{}.pdf'.format(res_2['timestamp']), format='pdf', dpi=1000)
	
if __name__ == '__main__':
	main()




