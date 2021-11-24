import pandas as pd
import numpy as np
from scipy.io import arff

class MicroarrayDataset:

	def __init__(self, filename):
		self.filename = filename
		self.x, self.y = self.load_dataset()

	def load_dataset(self):
		dataset = arff.loadarff(self.filename)
		df = pd.DataFrame(dataset[0])
		labels = {label:i for i, label in enumerate(df.iloc[:, -1].unique())}
		df.iloc[:, -1] = df.iloc[:, -1].replace(labels)
		x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
		y = df.iloc[:, -1].to_numpy(dtype=np.float32)
		return x, y

	def get_full_dataset(self):
		return self.x, self.y

	def split(self, trn=0.8):
		k = int(self.x.shape[0] * trn)
		index = np.random.permutation(np.arange(self.x.shape[0]))
		x_trn, y_trn = self.x[index[:k], :], self.y[index[:k]]
		x_tst, y_tst = self.x[index[k:], :], self.y[index[k:]]
		return x_trn, y_trn, x_tst, y_tst
