from __future__ import print_function

import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import config as cf

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Subset
import torch.cuda as cutorch

import os
import sys
import time
import argparse
import datetime

from networks import *
from torch.autograd import Variable

from trajectoryPlugin.plugin import API

import matplotlib.pyplot as plt
from scipy import spatial

parser = argparse.ArgumentParser(description='PyTorch CIFAR-10 Training')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--exp_num', default=0, type=int, help='experiment number')
parser.add_argument('--valid_size', default=500, type=int, help='validation set size')
parser.add_argument('--cluster_num', default=10, type=int, help='cluster number')
parser.add_argument('--burn_in', default=3, type=int, help='burn-in epoch number')
parser.add_argument('--interval', default=1, type=int, help='interval epoch number')
parser.add_argument('--update_rate', default=0.1, type=int, help='weight update rate')
parser.add_argument('--lr', default=0.1, type=float, help='learning_rate')
parser.add_argument('--net_type', default='wide-resnet', type=str, help='model')
parser.add_argument('--depth', default=28, type=int, help='depth of model')
parser.add_argument('--widen_factor', default=10, type=int, help='width of model')
parser.add_argument('--dropout', default=0., type=float, help='dropout_rate')
parser.add_argument('--dataset', default='cifar10', type=str, help='dataset = [cifar10/cifar100]')

args = parser.parse_args()

# Hyper Parameter settings
use_cuda = torch.cuda.is_available()
best_acc = 0
start_epoch, num_epochs, batch_size, optim_type = cf.start_epoch, cf.num_epochs, cf.batch_size, cf.optim_type

# Data Uplaod
print('\n[Phase 1] : Data Preparation')
transform_train = transforms.Compose([
    #transforms.RandomCrop(32, padding=4),
    #transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
]) # meanstd transformation

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cf.mean[args.dataset], cf.std[args.dataset]),
])

if(args.dataset == 'cifar10'):
    print("| Preparing CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 10
elif(args.dataset == 'cifar100'):
    print("| Preparing CIFAR-100 dataset...")
    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    num_classes = 100

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

np.random.seed(args.seed)

full_idx = np.arange(10000)
valid_idx = np.random.choice(10000, args.valid_size, replace=False).tolist()
test_idx = np.delete(full_idx, valid_idx).tolist()

# sanity check 1: making sure the drawn validset is the same
print('sanity check on drawn validation set:')
print(valid_idx[:20])

validset = Subset(testset, valid_idx)
testset = Subset(testset, test_idx)


validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size, shuffle=False)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

api = API(num_cluster=args.cluster_num, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"), iprint=2)
api.dataLoader(trainset, validset, batch_size=batch_size)

# Return network & file name
def getNetwork(args):
    if (args.net_type == 'lenet'):
        net = LeNet(num_classes)
        file_name = 'lenet'
    elif (args.net_type == 'vggnet'):
        net = VGG(args.depth, num_classes)
        file_name = 'vgg-'+str(args.depth)
    elif (args.net_type == 'resnet'):
        net = ResNet(args.depth, num_classes)
        file_name = 'resnet-'+str(args.depth)
    elif (args.net_type == 'wide-resnet'):
        net = Wide_ResNet(args.depth, args.widen_factor, args.dropout, num_classes)
        file_name = 'wide-resnet-'+str(args.depth)+'x'+str(args.widen_factor)
    else:
        print('Error : Network should be either [LeNet / VGGNet / ResNet / Wide_ResNet')
        sys.exit(0)

    return net, file_name


# Model
print('\n[Phase 2] : Model setup')
print('| Building net type [' + args.net_type + ']...')
net, file_name = getNetwork(args)
#random.seed(args.seed)
#np.random.seed(args.seed)
#torch.manual_seed(args.seed)
#if torch.cuda.is_available():
#   torch.cuda.manual_seed_all(args.seed)
net.apply(conv_init)

if use_cuda:
    net.cuda()
    net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

train_acc_his = []
valid_acc_his = []
test_acc_his = []
train_loss_his = []
valid_loss_his = []
test_loss_his = []

weight_his = []

criterion = nn.CrossEntropyLoss()

# Training
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer = optim.SGD(net.parameters(), lr=cf.learning_rate(args.lr, epoch), momentum=0.9, weight_decay=5e-4)

    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, cf.learning_rate(args.lr, epoch)))

    for batch_idx, (inputs, targets, weights) in enumerate(api.train_loader):
        if use_cuda:
            inputs, targets, weights = inputs.cuda(), targets.cuda(), weights.cuda() # GPU settings

        optimizer.zero_grad()

        #print('data shapes ', 'inputs ', inputs.shape, 'targets ', targets.shape, 'weights ', weights.shape)

        inputs, targets = Variable(inputs), Variable(targets)

        outputs = net(inputs)               # Forward Propagation
        loss = api.loss_func(outputs, targets, weights)  # Loss

        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()


    # cluster trajectory + reweight data
    if epoch >= args.burn_in and ((epoch - args.burn_in) % args.interval) == 0:
        api.clusterTrajectory()  # run gmm cluster
        api.reweightData(net, 1000000)  # update train_loader
        weight_his.append(np.expand_dims(api.weight_tensor.detach().numpy(), axis=1).tolist())


    api.generateTrainLoader()
    
    train_loss = train_loss / total
    acc = 100.*correct.item()/total

    print('train loss\t\t', train_loss)
    print('correct\t\t', correct, '\t\ttotal\t\t', total)
    print('acc\t\t', acc)

    train_loss_his.append(train_loss)
    train_acc_his.append(acc)

    # record trajectory
    api.createTrajectory(net)

    print('| Epoch [%3d/%3d] \t\tLoss: %.4f Acc@1: %.3f%%'
               %(epoch, num_epochs, train_loss, 100.*correct/total))

def valid(epoch):
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(validloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    valid_loss = valid_loss / total
    acc = 100.*correct.item()/total
    print('\nvalid loss\t\t', valid_loss)
    print('correct\t\t', correct, '\t\ttotal\t\t', total)
    print('acc\t\t', acc)
    valid_loss_his.append(valid_loss)
    valid_acc_his.append(acc)
    print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, valid_loss, acc))

def test(epoch):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            inputs, targets = Variable(inputs), Variable(targets)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

    test_loss = test_loss / total
    acc = 100.*correct.item()/total
    print('\ntest loss\t\t', test_loss)
    print('correct\t\t', correct, '\t\ttotal\t\t', total)
    print('acc\t\t', acc)
    test_loss_his.append(test_loss)
    test_acc_his.append(acc)
    print("\n| Test Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, test_loss, acc))

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(args.lr))
print('| Optimizer = ' + str(optim_type))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch+num_epochs):
    start_time = time.time()

    train(epoch)
    valid(epoch)
    test(epoch)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Epoch time : %d:%02d:%02d' % (cf.get_hms(epoch_time)))
    print('| Elapsed time : %d:%02d:%02d'  %(cf.get_hms(elapsed_time)))


print('\n[Phase 4] : Saving history')
history = np.savez(args.dataset + '_' + str(args.valid_size) + '_reweight_' + str(args.exp_num) + '_history', train_loss=train_loss_his, valid_loss=valid_loss_his, test_loss=test_loss_his,
                   train_acc=train_acc_his, valid_acc=valid_acc_his, test_acc=test_acc_his, weight=weight_his, traject=api.traject_matrix, cluster=api.cluster_matrix)