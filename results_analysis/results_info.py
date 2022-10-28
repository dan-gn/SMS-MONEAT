import numpy as np
import os
import sys
import pickle
import csv

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import SEED, N_EXPERIMENTS, N_POPULATION
from experiment_info import datasets, algorithms
from utilities.choose_solutions import SolutionSelector


data = {}

selector = SolutionSelector(method='WSum', pareto_front=False)

for i, alg in enumerate(algorithms):
	data[alg] = {}
	iterations = 200 if alg=='n3o' else 18000
	for ds in datasets:
		data[alg][ds] = {}
		results_path = os.getcwd() + f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt_final\\{ds}"
		time = [0] * N_EXPERIMENTS
		train = [0] * N_EXPERIMENTS
		train_fs = [0] * N_EXPERIMENTS
		val = [0] * N_EXPERIMENTS
		val_fs = [0] * N_EXPERIMENTS
		arch = [0] * N_EXPERIMENTS
		arch_fs = [0] * N_EXPERIMENTS
		for k in range(N_EXPERIMENTS):
			results_filename = f"{ds}_MinMaxSc_{k}.pkl"
			with open(f'{results_path}/{results_filename}', 'rb') as f:
				results = pickle.load(f)
			time[k] = results[2]['time']
			model = results[2]['model']

			# Choose solutions
			model.best_solution = selector.choose(model.population, model.x_train, model.y_train)
			model.best_solution_val = selector.choose(model.population, model.x_train, model.y_train, model.x_val, model.y_val)
			model.best_solution_archive = selector.choose(model.archive.get_full_population(), model.x_train, model.y_train, model.x_val, model.y_val)

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

with open('results_sms-moneat_hpt_wsum_2obj_merge-35_15.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	all_rows = []
	header = ['Dataset']
	subheader = ['']
	for alg in algorithms:
		header.extend([alg] * 7)
		temp = ['time']
		temp.extend(['gmean', 'fs'] * 3)
		subheader.extend(temp)
	all_rows.append(header)
	all_rows.append(subheader)
	for ds in datasets:
		row = [ds]
		row.extend([value for alg in algorithms for value in data[alg][ds].values()])
		all_rows.append(row)
	writer.writerows(all_rows)