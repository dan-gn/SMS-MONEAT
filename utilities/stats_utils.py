import numpy as np
from scipy import stats

import pandas as pd

def kruskal_wallis(x, y):
	labels = np.unique(y)
	p_value = np.ones(x.shape[1])
	for feature in range(x.shape[1]):
		if np.max(x[:, feature]) != np.min(x[:, feature]):
			classes = []
			for label_i in labels:
				class_i = x[np.argwhere(y == label_i), feature]
				class_i = list(np.squeeze(class_i))
				classes.append(class_i)
			_, p_value[feature] = stats.kruskal(*classes)
	return p_value

class KruskalWallisFilter:

	def __init__(self, threshold=0.01, attempts=10):
		self.threshold = threshold
		self.attempts = attempts

	def fit_transform(self, x, y):
		temp_threshold = self.threshold
		for i in range(self.attempts):
			self.p_value = kruskal_wallis(x, y)
			self.feature_selected = np.argwhere(self.p_value < temp_threshold)
			if self.feature_selected.shape[0] > 0:
				break
			temp_threshold = self.threshold * i
			print(f'All features were removed. Trying with new threshold = {temp_threshold}')
		return self.transform(x)

	def transform(self, x):
		x_filtered = x[:, self.feature_selected[:, 0]]
		return x_filtered, self.feature_selected


if __name__ == '__main__':

	"""
	Dataset and pre-processing
	"""
	# Read dataset
	df = pd.read_csv('datasets/breast-cancer-wisconsin.data', delimiter=',', header=None)
	df = df.drop([6], axis=1)

	# Remove column with incomplete data
	x = df.iloc[:, 1:-1].to_numpy(dtype=np.float32)

	# Set label data between 0 and 1
	y = df.iloc[:, -1].to_numpy(dtype=np.float32)
	y = np.round((y - 2) / 2)

	# Divide dataset in training set and validation set
	x_train = x[:640, :]
	y_train = y[:640]
	x_test = x[640:, :]
	y_test = y[640:]


	# Count class distribution from both datasets
	n_relapsed_train = np.sum(y_train)
	n_non_relapsed_train = y_train.shape[0] - n_relapsed_train
	n_relapsed_test = np.sum(y_test)
	n_non_relapsed_test = y_test.shape[0] - n_relapsed_test

	# Print information
	print(f"Train dataset shape: {x_train.shape[0]}, Relapsed instances: {n_relapsed_train}, Non-Relapsed instances: {n_non_relapsed_train}")
	print(f"Test dataset shape: {x_test.shape[0]}, Relapsed instances: {n_relapsed_test}, Non-Relapsed instances: {n_non_relapsed_test}")


	filter = KruskalWallisFilter(threshold=0.01)
	x_train_kw, feature_selected = filter.fit_transform(x_train, y_train)
	x_test_kw, _ = filter.transform(x_test)

	print(x_train_kw.shape, x_test_kw.shape)