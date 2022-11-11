import numpy as np
import os
import sys
import pickle
import csv
from statistics import stdev

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import SEED, N_EXPERIMENTS, N_POPULATION
from experiment_info import datasets, algorithms
from utilities.choose_solutions import SolutionSelector, SolutionSelector2, evaluate3


data = {}

selector = SolutionSelector(method='WSum', pareto_front=False)

for i, alg in enumerate(algorithms):
	data[alg] = {}
	iterations = 200 if alg=='n3o' else 18000
	for ds in datasets:
		data[alg][ds] = {}
		results_path = os.getcwd() + f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt_final_6\\{ds}"
		time = [0] * N_EXPERIMENTS
		train = [0] * N_EXPERIMENTS
		train_fs = [0] * N_EXPERIMENTS
		val = [0] * N_EXPERIMENTS
		val_fs = [0] * N_EXPERIMENTS
		train_arch = [0] * N_EXPERIMENTS
		train_arch_fs = [0] * N_EXPERIMENTS
		arch = [0] * N_EXPERIMENTS
		arch_fs = [0] * N_EXPERIMENTS
		for k in range(N_EXPERIMENTS):
			results_filename = f"{ds}_MinMaxSc_{k}.pkl"
			with open(f'{results_path}/{results_filename}', 'rb') as f:
				results = pickle.load(f)
			time[k] = results[2]['time']
			model = results[2]['model']

			# Choose solutions
			if alg != 'sms_emoa':
				model.best_solution = selector.choose(model.population, model.x_train, model.y_train)
				model.best_solution_val = selector.choose(model.population, model.x_train, model.y_train, model.x_val, model.y_val)
				model.best_solution_t_archive = selector.choose(model.archive.get_full_population(), model.x_train, model.y_train)
				model.best_solution_archive = selector.choose(model.archive.get_full_population(), model.x_train, model.y_train, model.x_val, model.y_val)
				model.best_solution.valid, model.best_solution_val.valid, model.best_solution_t_archive.valid, model.best_solution_archive.valid = True, True, True, True
				_, _, train[k] = model.evaluate(model.best_solution, model.x_test, model.y_test)
				_, _, val[k] = model.evaluate(model.best_solution_val, model.x_test, model.y_test)
				_, _, train_arch[k] = model.evaluate(model.best_solution_t_archive, model.x_test, model.y_test)
				_, _, arch[k] = model.evaluate(model.best_solution_archive, model.x_test, model.y_test)
				train_fs[k] = model.best_solution.selected_features.shape[0]
				val_fs[k] = model.best_solution_val.selected_features.shape[0]
				train_arch_fs[k] = model.best_solution_t_archive.selected_features.shape[0]
				arch_fs[k] = model.best_solution_archive.selected_features.shape[0]
			else:
				model.best_solution = selector.choose(model.population, model.x_train, model.y_train)
				_, train_fitness, train[k] = evaluate3(model.best_solution, model.x_train, model.y_train, model.x_test, model.y_test)
				train_fs[k] = train_fitness[1]

			
		data[alg][ds]['time'] = np.mean(time)		
		data[alg][ds]['time_std'] = stdev(time)		
		data[alg][ds]['train'] = np.mean(train)		
		data[alg][ds]['train_std'] = stdev(train)		
		data[alg][ds]['train_fs'] = np.mean(train_fs)		
		data[alg][ds]['train_fs_std'] = stdev(train_fs)		
		data[alg][ds]['val'] = np.mean(val)		
		data[alg][ds]['val_std'] = stdev(val)		
		data[alg][ds]['val_fs'] = np.mean(val_fs)		
		data[alg][ds]['val_fs_std'] = stdev(val_fs)		
		data[alg][ds]['arch_t'] = np.mean(train_arch)		
		data[alg][ds]['arch_t'] = stdev(train_arch)		
		data[alg][ds]['arch_t_fs'] = np.mean(train_arch_fs)		
		data[alg][ds]['arch_t_fs_std'] = stdev(train_arch_fs)		
		data[alg][ds]['arch'] = np.mean(arch)		
		data[alg][ds]['arch_std'] = stdev(arch)		
		data[alg][ds]['arch_fs'] = np.mean(arch_fs)		
		data[alg][ds]['arch_fs_std'] = stdev(arch_fs)		
		print(f'Algorithm: {alg}; Dataset: {ds}; Time {np.mean(time)}; Train {np.mean(train)}, Val {np.mean(val)}, Train Arch: {np.mean(train_arch)}, Arch {np.mean(arch)}')


with open('results_sms-moneat_final6.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	all_rows = []
	header = ['Dataset']
	subheader = ['']
	for alg in algorithms:
		header.extend([alg] * 18)
		temp = ['time', 'time_stdev']
		temp.extend(['gmean', 'g_mean_std', 'fs', 'fs_std'] * 4)
		subheader.extend(temp)
	all_rows.append(header)
	all_rows.append(subheader)
	for ds in datasets:
		row = [ds]
		row.extend([value for alg in algorithms for value in data[alg][ds].values()])
		all_rows.append(row)
	writer.writerows(all_rows)