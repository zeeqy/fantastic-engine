import numpy as np


def mnist_noise(y_train, ratio=0.5):
	label = range(10)
	label_copy = np.copy(y_train)
	noise_idx = np.random.choice(range(len(y_train)), size=int(len(y_train)*ratio), replace=False)
	for idx in noise_idx:
		noise_label = [lab for lab in label if lab != y_train[idx]]
		label_copy[idx] = np.random.choice(noise_label, size=1)
	return label_copy


def correct_prob(output, y_valid):
	prob = []
	for idx in range(len(output)):
		output_prob = softmax(output[idx])
		prob.append(output_prob[y_valid[idx]])
	return prob

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)