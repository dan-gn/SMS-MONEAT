import numpy as np

import os
from pathlib import Path
import pickle

seed = 0
n_experiments = 30
n_population = 100

algorithms = ['sms_moneat']
iterations = [18000]

datasets = []
datasets.append('breastCancer-full') 
datasets.append('ALL-AML-full')
datasets.append('prostate_tumorVSNormal-full')

for i, alg in enumerate(algorithms):
	for ds in datasets:
		results_path = os.getcwd() + f"\\results\\{alg}-pop_{n_population}-it_{iterations[i]}_seed{seed}_oldhv\\{ds}"
		time = [0] * n_experiments
		for k in range(n_experiments):
			results_filename = f"{ds}_MinMaxSc_{k}.pkl"
			with open(f'{results_path}/{results_filename}', 'rb') as f:
				results = pickle.load(f)
				time[k] = results[2]['time']
		print(f'Algorithm: {alg}; Dataset: {ds}; Time {np.mean(time)}')