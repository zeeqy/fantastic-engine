from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import numpy as np


class StandardTrainingNN:
	def __init__(self, torchnn, batch_size=100, num_iter=10, learning_rate=5e-5, early_stopping=5, device='cpu', iprint=0):
		self.torchnn = torchnn
		self.num_iter = num_iter
		self.batch_size = batch_size
		self.loss_func = nn.CrossEntropyLoss()
		self.learning_rate = learning_rate
		self.early_stopping = early_stopping
		self.device = device
		self.iprint = iprint

	def log(self, msg, level):
		if self.iprint >= level:
			print(msg)

	def fit(self, x_train_tensor, y_train_tensor, x_valid_tensor, y_valid_tensor, x_test_tensor, y_test_tensor):

		train_dataset = Data.TensorDataset(x_train_tensor, y_train_tensor)
		test_dataset = Data.TensorDataset(x_test_tensor, y_test_tensor)
		valid_dataset = Data.TensorDataset(x_valid_tensor, y_valid_tensor)

		L2 = 0.0005
		patience = 0
		best_params = {}
		best_epoch = 0
		best_score = np.inf

		optimizer = torch.optim.Adam(self.torchnn.parameters(), lr=self.learning_rate, weight_decay=L2)
		train_loader = Data.DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True)
		test_loader = Data.DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=True)
		valid_loader = Data.DataLoader(dataset=valid_dataset, batch_size=self.batch_size, shuffle=True)

		self.log('Standard NN training...',1)
		"""
		standard training starts
		"""
		epoch = 1
		scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
		while epoch <= self.num_iter and patience < self.early_stopping:
			train_losses = []
			self.torchnn.train()
			scheduler.step()
			for step, (data, target) in enumerate(train_loader):
				data, target = data.to(self.device), target.to(self.device)
				output = self.torchnn(data)
				loss = self.loss_func(output, target)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				train_losses.append(loss.item())
			train_loss = np.mean(train_losses)

			self.torchnn.eval()
			valid_loss, correct = self.evaluate(valid_loader)
			valid_accuracy = 100 * correct / len(valid_loader.dataset)

			#early stopping
			if valid_loss <= best_score:
				patience = 0
				best_score = valid_loss
				best_epoch = epoch
				torch.save(self.torchnn.state_dict(), 'checkpoint.pt')
			else:
				patience += 1

			test_loss, correct = self.evaluate(test_loader)
			self.log('epoch = {} | training loss = {:.4f} | valid loss = {:.4f} | valid accuarcy = {}% | early stopping = {}/{} | test loss = {:.4f} | test accuarcy = {}% [{}/{}]'.format(epoch, train_loss, valid_loss, valid_accuracy, patience, self.early_stopping, test_loss, 100*correct/len(test_loader.dataset), correct, len(test_loader.dataset)), 1)
			epoch += 1

		"""
		training finsihed
		"""
		self.torchnn.load_state_dict(torch.load('checkpoint.pt'))
		self.log('Standard training complete, best validation loss = {} at epoch = {}.'.format(best_score, best_epoch), 1)

	def predict(self, x_test_tensor):
		test_output = self.torchnn(x_test_tensor.to(self.device))
		return torch.max(test_output, 1)[1].data.cpu().numpy()

	def evaluate(self, data_loader):
		loss = 0
		correct = 0
		with torch.no_grad():
			for data, target in data_loader:
				data, target = data.to(self.device), target.to(self.device)
				output = self.torchnn(data)
				loss += self.loss_func(output, target).item() # sum up batch loss
				pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(target.view_as(pred)).sum().item()

		loss /= len(data_loader.dataset)

		return loss, correct


