import numpy as np
import torch

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z))

def get_batch(x, y, c=0.8):
	k = int(x.shape[0] * c)
	batch_index = np.random.permutation(np.arange(x.shape[0]))[:k]
	batch_index = sorted(batch_index)
	batch_index = torch.tensor(batch_index).type(torch.int32)
	x_batch = x.index_select(0, batch_index)
	y_batch = y.index_select(0, batch_index)
	# x_batch = x[batch_index, :]
	# y_batch = y[batch_index, :]
	return x_batch, y_batch