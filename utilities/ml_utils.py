import numpy as np
import torch
from sklearn.model_selection import train_test_split

def softmax(z):
	return np.exp(z) / np.sum(np.exp(z))

def get_batch(x, y, c=0.8, random_state=None):
	# k = int(x.shape[0] * c)
	# batch_index = np.random.permutation(np.arange(x.shape[0]))[:k]
	# batch_index = sorted(batch_index)
	# batch_index = torch.tensor(batch_index).type(torch.int32)
	# x_batch = x.index_select(0, batch_index)
	# y_batch = y.index_select(0, batch_index)
	if c >= 1.0:
		return x, y
	else:
		x_batch, _, y_batch, _ = train_test_split(x, y, test_size=1-c, random_state=random_state, stratify=y)
		return x_batch, y_batch