from random import random
import pandas as pd
import numpy as np
from scipy.io import arff
import arff as arfff
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold, StratifiedKFold, train_test_split

import tempfile

def load_arff_ignore_names(path):
    with open(path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    attr_counter = 0
    for line in lines:
        if line.lower().startswith('@attribute'):
            # Replace attribute name with a unique placeholder
            parts = line.split()
            if len(parts) >= 3:
                new_name = f"attr_{attr_counter}"
                new_line = f"@attribute {new_name} {' '.join(parts[2:])}\n"
                new_lines.append(new_line)
                attr_counter += 1
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    # Write modified content to a temporary file
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
        temp_file.writelines(new_lines)
        temp_file_path = temp_file.name

    # Now load with arff
    data, meta = arff.loadarff(temp_file_path)
    return data, meta


class MicroarrayDataset:

	def __init__(self, filename):
		self.x, self.y, self.df = self.load_dataset(filename)

	def load_dataset(self, filename):
		# dataset = arff.loadarff(filename)
		dataset = load_arff_ignore_names(filename)
		df = pd.DataFrame(dataset[0])
		self.labels = {label:i for i, label in enumerate(df.iloc[:, -1].unique())}
		df[df.columns[-1]] = df.iloc[:, -1].replace(self.labels)
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

	def cross_validation_3splits(self, k_folds=10, n_repeats=1, random_state=None):
		if n_repeats <= 1:
			rkf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
		else:
			rkf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=n_repeats, random_state=random_state)
		for i, (train_index, test_index) in enumerate(rkf.split(self.x, self.y)):
			x_train, x_val, y_train, y_val = train_test_split(self.x[train_index], self.y[train_index], test_size=len(test_index)*2, random_state=random_state, stratify=self.y[train_index])
			yield i, x_train, x_val, self.x[test_index], y_train, y_val, self.y[test_index]
		
	def cross_validation_experiment(self, k_folds=10, n_repeats=1, random_state=None):
		if n_repeats <= 1:
			rkf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
		else:
			rkf = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=n_repeats, random_state=random_state)
		cv = [{'train': train_index, 'test': test_index} for train_index, test_index in rkf.split(self.x, self.y)]
		for i in range(k_folds*n_repeats):
			if ((i+2) % k_folds) == 0:
				cv[i]['val'] = np.concatenate((cv[i+1]['test'], cv[i+2-k_folds]['test']))	
			elif ((i+1) % k_folds) == 0:
				cv[i]['val'] = np.concatenate((cv[i+1-k_folds]['test'], cv[i+2-k_folds]['test']))	
			else:
				cv[i]['val'] = np.concatenate((cv[i+1]['test'], cv[i+2]['test']))	
			cv[i]['train'] = np.array([x for x in cv[i]['train'] if x not in cv[i]['val']])

			yield i, self.x[cv[i]['train']], self.x[cv[i]['val']], self.x[cv[i]['test']], self.y[cv[i]['train']], self.y[cv[i]['val']], self.y[cv[i]['test']]

			


		




		
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
	
