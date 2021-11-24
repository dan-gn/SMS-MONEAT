import numpy as np
import torch
from scipy import stats
from sklearn.metrics import confusion_matrix

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


def geometric_mean(y_true, y_pred):
	tn, fp, fn, tp = confusion_matrix(y_true, torch.round(y_pred)).ravel()
	tp_rate = tp / (tp + fn)
	tn_rate = tn / (tn + fp)
	return np.sqrt(tp_rate * tn_rate)


if __name__ == '__main__':

	y_true = torch.tensor([0, 1, 0, 1]).type(torch.float32)
	y_pred = torch.tensor([0, 1, 0, 1]).type(torch.float32)

	print(geometric_mean(y_true, y_pred))
