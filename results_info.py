from re import A
import numpy as np

import os
from pathlib import Path
import pickle

import csv

from utilities.choose_solutions import choose_solution_train, choose_solution_val

seed = 0
n_experiments = 30
n_population = 100

algorithms = []
# algorithms.append('n3o')
algorithms.append('sms_moneat')

datasets = []
# datasets.append('breastCancer-full') 
# datasets.append('ALL-AML-full')
# datasets.append('prostate_tumorVSNormal-full')
datasets.append('Breast_GSE22820') 
datasets.append('Breast_GSE42568')
datasets.append('Breast_GSE59246') 
# datasets.append('Breast_GSE70947')
# datasets.append('Colorectal_GSE8671') 
# datasets.append('Colorectal_GSE32323') # SMS-MONEAT 18
# datasets.append('Colorectal_GSE44076')
# datasets.append('Colorectal_GSE44861')
# datasets.append('Leukemia_GSE22529_U133A') 
# datasets.append('Leukemia_GSE22529_U133B') # SMS-MONEAT 8 
# datasets.append('Leukemia_GSE33615')
# datasets.append('Leukemia_GSE63270') 
# datasets.append('Liver_GSE14520_U133A') #N3O 15
# datasets.append('Liver_GSE50579')
# datasets.append('Liver_GSE62232') 
# datasets.append('Prostate_GSE6919_U95Av2')
# datasets.append('Prostate_GSE11682')
# datasets.append('Prostate_GSE46602')


data = {}

for i, alg in enumerate(algorithms):
	data[alg] = {}
	iterations = 200 if alg=='n3o' else 18000
	for ds in datasets:
		data[alg][ds] = {}
		results_path = os.getcwd() + f"\\results\\{alg}-pop_{n_population}-it_{iterations}_seed{seed}-cv_hpt\\{ds}"
		time = [0] * n_experiments
		train = [0] * n_experiments
		train_fs = [0] * n_experiments
		val = [0] * n_experiments
		val_fs = [0] * n_experiments
		arch = [0] * n_experiments
		arch_fs = [0] * n_experiments
		for k in range(n_experiments):
			results_filename = f"{ds}_MinMaxSc_{k}.pkl"
			with open(f'{results_path}/{results_filename}', 'rb') as f:
				results = pickle.load(f)
			time[k] = results[2]['time']
			model = results[2]['model']
			if alg == 'sms_moneat':
				model.best_solution = choose_solution_train(model.population, model.x_train, model.y_train)
				model.best_solution_val = choose_solution_val(model.population, model.x_train, model.y_train, model.x_val, model.y_val)
				model.best_solution_archive = choose_solution_val(model.archive.get_full_population(), model.x_train, model.y_train, model.x_val, model.y_val)
			model.best_solution.valid, model.best_solution_val.valid, model.best_solution_archive.valid = True, True, True	
			_, _, train[k] = model.evaluate(model.best_solution, model.x_test, model.y_test)
			_, _, val[k] = model.evaluate(model.best_solution_val, model.x_test, model.y_test)
			_, _, arch[k] = model.evaluate(model.best_solution_archive, model.x_test, model.y_test)
			train_fs[k] = model.best_solution.selected_features.shape[0]
			val_fs[k] = model.best_solution_val.selected_features.shape[0]
			arch_fs[k] = model.best_solution_archive.selected_features.shape[0]
		data[alg][ds]['time'] = np.mean(time)		
		data[alg][ds]['train'] = np.mean(train)		
		data[alg][ds]['train_fs'] = np.mean(train_fs)		
		data[alg][ds]['val'] = np.mean(val)		
		data[alg][ds]['val_fs'] = np.mean(val_fs)		
		data[alg][ds]['arch'] = np.mean(arch)		
		data[alg][ds]['arch_fs'] = np.mean(arch_fs)		
		print(f'Algorithm: {alg}; Dataset: {ds}; Time {np.mean(time)}; Train {np.mean(train)}, Val {np.mean(val)}, Arch {np.mean(arch)}')

# with open('results_sms-moneat_hpt2.csv', 'w', newline='') as file:
# 	writer = csv.writer(file)
# 	all_rows = []
# 	header = ['Dataset']
# 	subheader = ['']
# 	for alg in algorithms:
# 		header.extend([alg] * 7)
# 		temp = ['time']
# 		temp.extend(['gmean', 'fs'] * 3)
# 		subheader.extend(temp)
# 	all_rows.append(header)
# 	all_rows.append(subheader)
# 	for ds in datasets:
# 		row = [ds]
# 		row.extend([value for alg in algorithms for value in data[alg][ds].values()])
# 		all_rows.append(row)
# 	writer.writerows(all_rows)