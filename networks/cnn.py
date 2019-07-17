import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet1(nn.Module):
	def __init__(self):
		super(ConvNet1, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.layer2 = nn.Sequential(
			nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2))
		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(7 * 7 * 64, 1000)
		self.fc2 = nn.Linear(1000, 10)

	def forward(self, x):
		out = self.layer1(x)
		out = self.layer2(out)
		out = out.reshape(out.size(0), -1)
		out = self.drop_out(out)
		out = self.fc1(out)
		out = self.fc2(out)
		return out

class ConvNet2(nn.Module):

	def __init__(self):
		super(ConvNet2, self).__init__()
		self.feats = nn.Sequential(
			nn.Conv2d(1, 32, 5, 1, 1),
			nn.MaxPool2d(2, 2),
			nn.ReLU(),
			nn.BatchNorm2d(32),

			nn.Conv2d(32, 64, 3,  1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(64),

			nn.Conv2d(64, 64, 3,  1, 1),
			nn.MaxPool2d(2, 2),
			nn.ReLU(),
			nn.BatchNorm2d(64),

			nn.Conv2d(64, 128, 3, 1, 1),
			nn.ReLU(),
			nn.BatchNorm2d(128)
		)

		self.classifier = nn.Conv2d(128, 10, 1)
		self.avgpool = nn.AvgPool2d(6, 6)
		self.dropout = nn.Dropout(0.5)

	def forward(self, inputs):
		out = self.feats(inputs)
		out = self.dropout(out)
		out = self.classifier(out)
		out = self.avgpool(out)
		out = out.view(-1, 10)
		return out