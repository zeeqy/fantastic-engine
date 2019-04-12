import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
	 def __init__(self):
		  super(LeNet, self).__init__()
		  self.conv1 = nn.Sequential( # (1,28,28)
					nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3,
							  stride=1, padding=0), # (32,26,26)
					nn.ReLU(),
					)
					
		  self.conv2 = nn.Sequential( # (32,26,26)
					nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
							  stride=1, padding=0), # (64,24,24)
					nn.ReLU(),
					nn.MaxPool2d(kernel_size=2), # (64,12,12)
					nn.Dropout2d(p=0.25),

					)

		  self.fc1 = nn.Sequential(
				  nn.Linear(64*12*12,128),
				  nn.ReLU(),
				  nn.Dropout2d(p=0.5),
				  )
		  
		  self.fc2 = nn.Linear(128,10)
		  self.softmax = nn.Softmax(dim=1)
		  
	 def forward(self, x):
		  x = self.conv1(x)
		  x = self.conv2(x)
		  x = x.view(x.size(0), -1)
		  x = self.fc1(x)
		  x = self.fc2(x)
		  output = self.softmax(x)
		  return output