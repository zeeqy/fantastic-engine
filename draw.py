import numpy as np
import matplotlib.pyplot as plt

s_train_loss = None
s_valid_loss = None
s_test_loss = None
s_train_acc = None
s_valid_acc = None
s_test_acc = None

r_train_loss = None
r_valid_loss = None
r_test_loss = None
r_train_acc = None
r_valid_acc = None
r_test_acc = None
weight = None

M = 1

for i in range(M):
    standard = np.load('history/per-epoch/cifar100_500_standard_' + str(i) + '_history.npz')
    reweight = np.load('history/per-epoch/cifar100_500_reweight_' + str(i) + '_history.npz')

    if s_train_loss is None:
        s_train_loss = standard['train_loss']
        s_valid_loss = standard['valid_loss']
        s_test_loss = standard['test_loss']
        s_train_acc = standard['train_acc']
        s_valid_acc = standard['valid_acc']
        s_test_acc = standard['test_acc']

        N = s_train_loss.shape[0]
        plt.plot(np.arange(0, N, 1), s_valid_loss, c='b')

        r_train_loss = reweight['train_loss']
        r_valid_loss = reweight['valid_loss']
        r_test_loss = reweight['test_loss']
        r_train_acc = reweight['train_acc']
        r_valid_acc = reweight['valid_acc']
        r_test_acc = reweight['test_acc']

        N = s_train_loss.shape[0]
        plt.plot(np.arange(0, N, 1), r_valid_loss, c='r')

        #weight = reweight['weight']

    else:
        s_train_loss = s_train_loss + standard['train_loss']
        s_valid_loss = s_valid_loss + standard['valid_loss']
        s_test_loss = s_test_loss + standard['test_loss']
        s_train_acc = s_train_acc + standard['train_acc']
        s_valid_acc = s_valid_acc + standard['valid_acc']
        s_test_acc = s_test_acc + standard['test_acc']

        N = s_train_loss.shape[0]
        plt.plot(np.arange(0, N, 1), standard['valid_loss'], c='b')

        r_train_loss = r_train_loss + reweight['train_loss']
        r_valid_loss = r_valid_loss + reweight['valid_loss']
        r_test_loss = r_test_loss + reweight['test_loss']
        r_train_acc = r_train_acc + reweight['train_acc']
        r_valid_acc = r_valid_acc + reweight['valid_acc']
        r_test_acc = r_test_acc + reweight['test_acc']

        N = s_train_loss.shape[0]
        plt.plot(np.arange(0, N, 1), reweight['valid_loss'], c='r')

        #weight = weight + reweight['weight']
    plt.show()

#plt.show()


s_train_loss = s_train_loss / M
s_valid_loss = s_valid_loss / M
s_test_loss = s_test_loss / M
s_train_acc = s_train_acc / M
s_valid_acc = s_valid_acc / M
s_test_acc = s_test_acc / M

r_train_loss = r_train_loss / M
r_valid_loss = r_valid_loss / M
r_test_loss = r_test_loss / M
r_train_acc = r_train_acc / M
r_valid_acc = r_valid_acc / M
r_test_acc = r_test_acc / M

#weight = weight / M

#print(weight)

N = s_train_loss.shape[0]

x = np.arange(0, N, 1)

fig = plt.figure()

ax1 = fig.add_subplot(231)
ax1.plot(x, s_train_loss, ls=':', c='b', label='Standard')
ax1.plot(x, r_train_loss, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_train_loss, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_train_loss, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Train Loss')

ax1 = fig.add_subplot(232)
ax1.plot(x, s_valid_loss, ls=':', c='b', label='Standard')
ax1.plot(x, r_valid_loss, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_valid_loss, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_valid_loss, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Valid Loss')

ax1 = fig.add_subplot(233)
ax1.plot(x, s_test_loss, ls=':', c='b', label='Standard')
ax1.plot(x, r_test_loss, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_test_loss, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_test_loss, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Test Loss')

ax1 = fig.add_subplot(234)
ax1.set_ylim([0,100])
ax1.plot(x, s_train_acc, ls=':', c='b', label='Standard')
ax1.plot(x, r_train_acc, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_train_acc, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_train_acc, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Train Accuracy')

ax1 = fig.add_subplot(235)
ax1.set_ylim([0,100])
ax1.plot(x, s_valid_acc, ls=':', c='b', label='Standard')
ax1.plot(x, r_valid_acc, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_valid_acc, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_valid_acc, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Valid Accuracy')

ax1 = fig.add_subplot(236)
ax1.set_ylim([0,100])
ax1.plot(x, s_test_acc, ls=':', c='b', label='Standard')
ax1.plot(x, r_test_acc, ls=':', c='r', label='Reweighted')

'''
ax2 = ax1.twinx()
ax2.bar(left=N/2-1, height=s_f_test_acc, width=0.4, alpha=0.8, color='b', label="Standard")
ax2.bar(left=N/2+1, height=r_f_test_acc, width=0.4, alpha=0.8, color='r', label="Reweighted")
'''

ax1.legend(['Standard', 'Reweighted'])
ax1.set_title('Test Accuracy')

plt.show()