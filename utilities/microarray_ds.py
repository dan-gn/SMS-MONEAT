import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split

class MicroarrayDataset:

	def __init__(self, filename):
		self.filename = filename
		self.x, self.y = self.load_dataset()

	def load_dataset(self):
		dataset = arff.loadarff(self.filename)
		df = pd.DataFrame(dataset[0])
		self.labels = {label:i for i, label in enumerate(df.iloc[:, -1].unique())}
		df.iloc[:, -1] = df.iloc[:, -1].replace(self.labels)
		x = df.iloc[:, :-1].to_numpy(dtype=np.float32)
		y = df.iloc[:, -1].to_numpy(dtype=np.float32)
		return x, y

	def get_full_dataset(self):
		return self.x, self.y
	
	def get_labels(self):
		return self.labels

	def split(self, trn_size=0.8, random_state=None):
		return train_test_split(self.x, self.y, test_size=1-trn_size, random_state=random_state, stratify=self.y)