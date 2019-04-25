import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
	parser = argparse.ArgumentParser(description='MNIST Graph')
	parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
	parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
	parser.add_argument('--burn_in', type=int, default=5, help='number of burn-in epochs (default: 5)')
	parser.add_argument('--valid_size', type=int, default=1000, help='input validation size (default: 1000)')
	parser.add_argument('--lr', type=float, default=0.01, help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, help='SGD momentum (default: 0.5)')
	parser.add_argument('--noise_level', type=float, default=0, help='percentage of noise data (default: 0.0)')
	parser.add_argument('--num_cluster', type=int, default=3, help='number of cluster (default: 3)')
	parser.add_argument('--reweight_interval', type=int, default=1, help='number of epochs between reweighting')
	parser.add_argument('--weight_update_rate', type=float, default=0.1, help='weight update rate (default: 0.1)')
	parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

	args = parser.parse_args()
	args_dict = vars(args)
	
	with open('mnist_experiments/mnist_cnn_baseline_reweight_response.data', 'r+') as f:
		rec = f.read().split('\n')[:-1]
	f.close()
	
	res = []
	for item in rec:
		item_dict = json.loads(item)
		match = True
		for key in args_dict.keys():
			if (key not in item_dict.keys()) or (args_dict[key] != item_dict[key]):
				match = False
		if match:
			res.append(item_dict)
	
	if len(res) == 0:
		print("No config matches, please check your arguments!")
		return None
	
	elif len(res) > 1:
		print("More than one trail found, select the most recent one!")
		res = sorted(res, key=lambda k: k['timestamp'], reverse=True)[0]
	else:
		res = res[0]
	
	fig, axs = plt.subplots(2,3, figsize=(20, 10))
	
	x = range(1, res['epochs']+1, 1)
	axs[0,0].plot(x, res['standard_train_loss'], '--', color='blue', label='Standard')
	axs[0,0].plot(x, res['reweight_train_loss'], '--', color='red', label='Reweighted')
	axs[0,0].axvline(x=res['burn_in'], linestyle='--', color='black')
	axs[0,0].set_title("Train (Weighted) Loss")
	axs[0,0].legend()

	axs[0,1].plot(x, res['standard_valid_loss'], '--', color='blue', label='Standard')
	axs[0,1].plot(x, res['reweight_valid_loss'], '--', color='red', label='Reweighted')
	axs[0,1].axvline(x=res['burn_in'], linestyle='--', color='black')
	axs[0,1].set_title("Validation Loss")
	axs[0,1].legend()

	axs[0,2].plot(x, res['standard_test_loss'], '--', color='blue', label='Standard')
	axs[0,2].plot(x, res['reweight_test_loss'], '--', color='red', label='Reweighted')
	axs[0,2].axvline(x=res['burn_in'], linestyle='--', color='black')
	axs[0,2].set_title("Test Loss")
	axs[0,2].legend()
	
	axs[1,0].plot(x, res['standard_train_accuracy'], '--', color='blue', label='Standard')
	axs[1,0].plot(x, res['reweight_train_accuracy'], '--', color='red', label='Reweighted')
	axs[1,0].axvline(x=res['burn_in'], linestyle='--', color='black')
	axs[1,0].set_title("Train Accuracy")
	axs[1,0].legend()


	print("standard_valid_accuracy: ", res['standard_valid_accuracy'][-5:])
	print("reweight_valid_accuracy: ", res['reweight_valid_accuracy'][-5:])
	axs[1,1].plot(x, res['standard_valid_accuracy'], '--', color='blue', label='Standard')
	axs[1,1].plot(x, res['reweight_valid_accuracy'], '--', color='red', label='Reweighted')
	axs[1,1].axvline(x=res['burn_in'], linestyle='--', color='black')
	axs[1,1].set_title("Validation Accuracy")
	axs[1,1].legend()

	print("standard_test_accuracy:", res['standard_test_accuracy'][-5:])
	print("reweight_test_accuracy:", res['reweight_test_accuracy'][-5:])
	axs[1,2].plot(x, res['standard_test_accuracy'], '--', color='blue', label='Standard')
	axs[1,2].plot(x, res['reweight_test_accuracy'], '--', color='red', label='Reweighted')
	axs[1,2].axvline(x=res['burn_in'], linestyle='--', color='black')
	axs[1,2].set_title("Test Accuracy")
	axs[1,2].legend()
	
	plt.savefig('figures/loss_accuracy_{}.pdf'.format(res['timestamp']), format='pdf', dpi=1000)

	try:	
		with open('mnist_experiments/weights/mnist_cnn_baseline_reweight_{}.data'.format(res['timestamp']), 'r+') as f:
			rec = f.read().split('\n')[:-1]
		f.close()
		
		fig, axs = plt.subplots(2,3, figsize=(20, 10))
		i = 0
		j = 0
		for item in rec[:3] + rec[-3:]:
			item_dict = json.loads(item)
			axs[i,j].hist(item_dict['weight_tensor'],bins='auto', range=(min(item_dict['weight_tensor'])-0.2,max(item_dict['weight_tensor'])+0.2))
			axs[i,j].set_title("Weights Distirbution at {} epoch".format(item_dict['epoch']))
			if j < 3-1:
				j += 1
			else:
				i += 1
				j = 0
		plt.tight_layout()
		plt.savefig('figures/weights_distribution_{}.pdf'.format(res['timestamp']), format='pdf', dpi=1000)

		weight_idx = input("index weight trajectory: ")
		
		if weight_idx != None:
			fig, axs = plt.subplots(1,1, figsize=(5, 5))
			weight_idx = int(weight_idx)
			weight_trajectory = [json.loads(item)['weight_tensor'][weight_idx] for item in rec]
			weight_epochs = [json.loads(item)['epoch'] for item in rec]
			plt.plot(weight_epochs,weight_trajectory, 'x-')
			plt.title("Weight changes for points {}".format(weight_idx))
			plt.tight_layout()
			plt.savefig('figures/weights_inspect_{}.pdf'.format(res['timestamp']), format='pdf', dpi=1000)

	except:
		print("mnist_cnn_baseline_reweight_{}.data didn't found, skip weight graph.".format(res['timestamp']))

	try:	
		with open('mnist_experiments/trajectory/mnist_cnn_baseline_trajectory_{}.data'.format(res['timestamp']), 'r+') as f:
			rec = f.read().split('\n')[:-1]
		f.close()

		fig, axs = plt.subplots(2,3, figsize=(20, 10))
		i = 0
		j = 0
		for item in rec[:3] + rec[-3:]:
			item_dict = json.loads(item)
			x = range(1, item_dict['epoch']+1, 1)
			for cid in range(args.num_cluster):
				axs[i,j].plot(x, item_dict['trajectory']['{}'.format(cid)][0], '--')
			axs[i,j].set_title("Average cluster trajectory at {} epoch".format(item_dict['epoch']))
			if j < 3-1:
				j += 1
			else:
				i += 1
				j = 0
		plt.tight_layout()
		plt.savefig('figures/mean_trajectory_{}.pdf'.format(res['timestamp']), format='pdf', dpi=1000)
	
	except:
		print("mnist_cnn_baseline_trajectory_{}.data didn't found, skip trajectory graph.".format(res['timestamp']))
	
if __name__ == '__main__':
	main()




