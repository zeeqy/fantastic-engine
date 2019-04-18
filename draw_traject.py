import matplotlib.pyplot as plt
import numpy as np

A = np.load('history/reweight/cifar100_500_reweight_0_history.npz')
cluster = A['cluster']
weight = A['weight'].squeeze()
print(weight.shape)
#weight = A['weight'].squeeze().sum(axis=0) / 23
#print(weight.shape)

#B = np.load('history/check-cluster/cifar10_1000_check_1_history.npz')

A = A['traject']
#B = B['traject']

for i in range(23):
    fig = plt.figure()
    for j in range(10):
        sect = cluster[:,i]
        #print('sect\t', sect.shape)
        ind = np.argwhere(sect==j)
        size =ind.shape[0]
        if size > 10:
            ind = ind.squeeze()
            sample = A[ind,:10+5*i]
            sample_w = weight[i][ind].mean()
            print(sample.shape)

            ax1 = fig.add_subplot(2,5,j+1)
            ax1.set_ylim([0,2.0])
            ax1.set_xlim([0,120])
            ax1.errorbar(np.arange(sample.shape[1]), sample.mean(axis=0), np.std(sample, axis=0), ls=':', c='b', label='A')
            ax1.bar(left=55, height=sample_w, width=10, alpha=0.8, color='red', label="weight")
            ax1.text(60, sample_w + 0.1, '{0:.3}'.format(sample_w), ha="center", va="top")
            ax1.text(60, sample_w + 0.3, size, ha="center", va="top")

    plt.savefig('record/cifar100-reweight-cosine/reweight-figure-' + str(i) + '.png', dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format=None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)