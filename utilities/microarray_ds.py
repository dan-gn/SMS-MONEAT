from random import random
import pandas as pd
import numpy as np
from scipy.io import arff
import arff as arfff
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, train_test_split

class MicroarrayDataset:

	def __init__(self, filename):
		self.x, self.y, self.df = self.load_dataset(filename)

	def load_dataset(self, filename):
		dataset = arff.loadarff(filename)
		df = pd.DataFrame(dataset[0])
		self.labels = {label:i for i, label in enumerate(df.iloc[:, -1].unique())}
		df.iloc[:, -1] = df.iloc[:, -1].replace(self.labels)
		x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
		y = df.iloc[:, -1].to_numpy(dtype=np.float32)
		return x, y, pd.DataFrame(dataset[0])

	def get_full_dataset(self):
		return self.x, self.y
	
	def get_labels(self):
		return self.labels

	def split(self, trn_size=0.8, random_state=None):
		return train_test_split(self.x, self.y, test_size=1-trn_size, random_state=random_state, stratify=self.y)

	def cross_validation(self, k_folds=10, n_repeats=1, random_state=None):
		if n_repeats <= 1:
			rkf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
		else:
			rkf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=n_repeats, random_state=random_state)
		for i, (train_index, test_index) in enumerate(rkf.split(self.x, self.y)):
			yield i, self.x[train_index], self.x[test_index], self.y[train_index], self.y[test_index]


		
def merge_datasets(a, b, filename):
	df = pd.concat([a.df, b.df])
	df.iloc[:, -1] = df.iloc[:, -1].str.decode("utf-8")
	print(df.iloc[:, -1].dtype)

	arfff.dump(filename
		, df.values
		, relation='relation name'
		, names=df.columns)

	ds = MicroarrayDataset(filename)
	return ds


if __name__ == '__main__':

	filename = 'prostate_tumorVSNormal_train'
	ds_train = MicroarrayDataset(f'./datasets/{filename}.arff')
	x, y = ds_train.get_full_dataset()
	print(x.shape, y.shape)

	filename = 'prostate_tumorVSNormal_test'
	ds_test = MicroarrayDataset(f'./datasets/{filename}.arff')
	x, y = ds_test.get_full_dataset()
	print(x.shape, y.shape)

	# new_filename = './datasets/prostate_tumorVSNormal-full.arff'
	# ds_full = merge_datasets(ds_train, ds_test, new_filename)
	# x, y = ds_full.get_full_dataset()
	# print(x.shape, y.shape)

	filename = 'prostate_tumorVSNormal-full'
	ds_full = MicroarrayDataset(f'./datasets/{filename}.arff')
	x, y = ds_full.get_full_dataset()
	print(x.shape, y.shape)
	


