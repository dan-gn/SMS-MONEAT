import numpy as np
import os
import sys
import pickle
import csv

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import SEED, N_POPULATION
from experiment_info import datasets
from utilities.microarray_ds import MicroarrayDataset

alg = 'SMS_MONEAT'
iterations = 18000
experiment = 0

data = {}

for ds in datasets:
	df = MicroarrayDataset(f'./datasets/CUMIDA/{ds}.arff')
	x, y = df.get_full_dataset()
	distribution = f'({(y.shape[0]-np.sum(y))/y.shape[0]:.2f}, {np.sum(y)/y.shape[0]:.2f})'
	data[ds] = {'n_samples' : x.shape[0], 'n_features' : x.shape[1], 'distribution' : distribution}
	print(f'Dataset: {ds}; Samples {x.shape[0]}; Features {x.shape[1]}, Distribution {distribution}')
    
with open('datasets_info_irace.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	all_rows = []
	header = ['Dataset', 'Samples', 'Features', 'Distribution']
	all_rows.append(header)
	for ds in datasets:
		row = [ds]
		row.extend([value for value in data[ds].values()])
		all_rows.append(row)
	writer.writerows(all_rows)

