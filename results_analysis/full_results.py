import numpy as np
import os
import sys
import pickle
import csv

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import SEED, N_EXPERIMENTS, N_POPULATION
from experiment_info import datasets, algorithms, iter_num
from utilities.choose_solutions import SolutionSelector, SolutionSelector2, evaluate3

selector = SolutionSelector2(method='WSum', pareto_front=False)

data = {}
for i, alg in enumerate(algorithms):
	data[alg] = {}
	iterations = iter_num[alg]
	for ds in datasets:
		data[alg][ds] = {}
		results_path = os.getcwd() + f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt_final_6\\{ds}"
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
			if alg != 'sms_emoa':
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
			else:
				model.best_solution = selector.choose(model.population, model.x_train, model.y_train)
				_, train_fitness, train[k] = evaluate3(model.best_solution, model.x_train, model.y_train, model.x_test, model.y_test)
				train_fs[k] = train_fitness[1]

			
		data[alg][ds]['time'] = time
		data[alg][ds]['train'] = train
		data[alg][ds]['train_fs'] = train_fs
		data[alg][ds]['val'] = val
		data[alg][ds]['val_fs'] = val_fs
		data[alg][ds]['arch'] = arch
		data[alg][ds]['arch_fs'] = arch_fs
		print(f'Algorithm: {alg}; Dataset: {ds}; Time {np.mean(time)}; Train {np.mean(train)}, Val {np.mean(val)}, Arch {np.mean(arch)}')

def store_results(data, alg, filename, population):
	with open(f'{filename}_{population}.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		all_rows = []
		header = ['Dataset']
		header.extend([i for i in range(N_EXPERIMENTS)])
		all_rows.append(header)
		for ds in datasets:
			row = [ds]
			row.extend(list(data[alg][ds][population]))
			all_rows.append(row)
		for ds in datasets:
			row = [ds]
			row.extend(list(data[alg][ds][f'{population}_fs']))
			all_rows.append(row)
		writer.writerows(all_rows)

alg = 'sms_moneat'
store_results(data, alg, f'results_{alg}_final6_full', 'train')
store_results(data, alg, f'results_{alg}_final6_full', 'val')
store_results(data, alg, f'results_{alg}_final6_full', 'arch')alg = 'sms_emoa'
