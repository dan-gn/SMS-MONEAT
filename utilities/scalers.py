import numpy as np

class MeanScaler:

	def __init__(self):
		self.mu = None
		self.x_min = None
		self.x_max = None

	def fit_transform(self, x):
		self.mu = np.mean(x, axis=0)
		self.x_min = np.min(x, axis=0)
		self.x_max = np.max(x, axis=0)
		return self.transform(x)

	def transform(self, x):
		x_norm = np.zeros_like(x)
		for i, xi in enumerate(x_norm):
			x_norm[i] = (xi - self.mu) / ((self.x_max - self.x_min) + 1e-6)
		return x_norm
