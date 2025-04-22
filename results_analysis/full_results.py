import numpy as np
import os
import sys
import pickle
import csv

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import SEED, N_EXPERIMENTS, N_POPULATION
from experiment_info import datasets, algorithms, iter_num, experiment
from utilities.choose_solutions import SolutionSelector, SolutionSelector2, evaluate3

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", FutureWarning)

selector = SolutionSelector(method='WSum', pareto_front=False)
selector2 = SolutionSelector2(method='WSum', pareto_front=False)

data = {}
for i, alg in enumerate(algorithms):
	data[alg] = {}
	iterations = iter_num[alg]
	exp = experiment[alg]
	for ds in datasets:
		data[alg][ds] = {}
		# results_path = os.getcwd() + f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt_final{exp}\\{ds}"
		results_path = os.getcwd() + f'\\results_asc\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-exp{exp}\\{ds}'
		time = [0] * N_EXPERIMENTS
		train = [0] * N_EXPERIMENTS
		train_fs = [0] * N_EXPERIMENTS
		train_acc = [0] * N_EXPERIMENTS
		val = [0] * N_EXPERIMENTS
		val_fs = [0] * N_EXPERIMENTS
		val_acc = [0] * N_EXPERIMENTS
		train_arch = [0] * N_EXPERIMENTS
		train_arch_fs = [0] * N_EXPERIMENTS
		train_arch_acc = [0] * N_EXPERIMENTS
		arch = [0] * N_EXPERIMENTS
		arch_fs = [0] * N_EXPERIMENTS
		arch_acc = [0] * N_EXPERIMENTS
		for k in range(N_EXPERIMENTS):
			results_filename = f"{ds}_MinMaxSc_{k}.pkl"
			with open(f'{results_path}/{results_filename}', 'rb') as f:
				results = pickle.load(f)
			time[k] = results[2]['time']
			model = results[2]['model']

			# Choose solutions
			if alg in ['sms_moneat', 'n3o']:
				model.best_solution = selector.choose(model.population, model.x_train, model.y_train)
				model.best_solution_val = selector.choose(model.population, model.x_train, model.y_train, model.x_val, model.y_val)
				model.best_solution_t_archive = selector.choose(model.archive.get_full_population(), model.x_train, model.y_train)
				model.best_solution_archive = selector.choose(model.archive.get_full_population(), model.x_train, model.y_train, model.x_val, model.y_val)
				model.best_solution.valid, model.best_solution_val.valid, model.best_solution_archive.valid = True, True, True
				train_acc[k], _, train[k] = model.evaluate(model.best_solution, model.x_test, model.y_test)
				val_acc[k], _, val[k] = model.evaluate(model.best_solution_val, model.x_test, model.y_test)
				train_arch_acc[k], _, train_arch[k] = model.evaluate(model.best_solution_t_archive, model.x_test, model.y_test)
				arch_acc[k], _, arch[k] = model.evaluate(model.best_solution_archive, model.x_test, model.y_test)
				train_fs[k] = model.best_solution.selected_features.shape[0]
				val_fs[k] = model.best_solution_val.selected_features.shape[0]
				train_arch_fs[k] = model.best_solution_t_archive.selected_features.shape[0]
				arch_fs[k] = model.best_solution_archive.selected_features.shape[0]
			elif alg in ['sms_emoa']:
				model.best_solution = selector.choose(model.population, model.x_train, model.y_train)
				_, train_fitness, train[k] = evaluate3(model.best_solution, model.x_train, model.y_train, model.x_test, model.y_test)
				train_fs[k] = train_fitness[1] # LOSS
			elif alg in ['mochc']:
				model.best_solution = selector2.choose(model.archive, model.x_train, model.y_train)
				_, train_fitness, train[k] = evaluate3(model.best_solution, model.x_train, model.y_train, model.x_test, model.y_test)
				train_fs[k] = train_fitness[1]
				train[k] = float(train_fitness[0]) # LOSS
			elif alg == 'sfe':
				_, train_fitness, train[k] = model.final_evaluate(model.best_solution, model.x_train, model.y_train, model.x_test, model.y_test)
				train_fs[k] = train_fitness[1]
				train[k] = float(train_fitness[0]) # LOSS
			elif alg == 'sfe_pso':
				# print(f'{k}')
				if hasattr(model, 'population'):
					model.best_solution = selector2.choose(model.population, model.x_train, model.y_train)
				_, train_fitness, train[k] = model.final_evaluate(model.best_solution.position, model.x_train, model.y_train, model.x_test, model.y_test)
				train_fs[k] = train_fitness[1]
				train[k] = float(train_fitness[0]) # LOSS



			
		data[alg][ds]['time'] = time
		data[alg][ds]['train'] = train_acc
		data[alg][ds]['train_fs'] = train_fs
		data[alg][ds]['val'] = val_acc
		data[alg][ds]['val_fs'] = val_fs
		data[alg][ds]['arch_t'] = train_arch_acc
		data[alg][ds]['arch_t_fs'] = train_arch_fs		
		data[alg][ds]['arch'] = arch_acc
		data[alg][ds]['arch_fs'] = arch_fs
		print(f'Algorithm: {alg}; Dataset: {ds}; Time {np.mean(time)}; Train {np.mean(train)}, Val {np.mean(val)}, Arch {np.mean(arch)}')
		print(f'Algorithm: {alg}; Dataset: {ds}; Time {np.mean(time)}; Train {np.mean(train_fs)}, Val {np.mean(val_fs)}, Arch {np.mean(arch_fs)}')
		# print(f'Algorithm: {alg}; Dataset: {ds}; Time {np.mean(time)}; Train {np.mean(train)}, Val {np.mean(val)}, Train Arch: {np.mean(train_arch)}, Arch {np.mean(arch)}')
		# print(f'Algorithm: {alg}; Dataset: {ds}; Time {np.mean(time)}; Train {np.mean(train_fs)}, Val {np.mean(val_fs)}, Train Arch: {np.mean(train_arch_fs)}, Arch {np.mean(arch_fs)}')

def store_results(data, alg, filename, population):
	with open(f'final_results_asc/{alg}_2025/{filename}_{population}_acc.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		all_rows = []
		header = ['Dataset']
		header.extend([i for i in range(N_EXPERIMENTS)])
		all_rows.append(header)
		for ds in datasets:
			row = [ds]
			row.extend(list(data[alg][ds][population]))
			all_rows.append(row)
		writer.writerows(all_rows)
	with open(f'final_results_asc/{alg}_2025/{filename}_{population}_fs.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		all_rows = []
		header = ['Dataset']
		header.extend([i for i in range(N_EXPERIMENTS)])
		all_rows.append(header)
		for ds in datasets:
			row = [ds]
			row.extend(list(data[alg][ds][f'{population}_fs']))
			all_rows.append(row)
		writer.writerows(all_rows)
		
store_results(data, alg, f'results_{alg}_final{exp}_full', 'train')
store_results(data, alg, f'results_{alg}_final{exp}_full', 'val')
# store_results(data, alg, f'results_{alg}_final{exp}_full', 'arch_t')
store_results(data, alg, f'results_{alg}_final{exp}_full', 'arch')
# for alg in algorithms:
# 	store_results(data, alg, f'results_{alg}_final{exp}_full_ws2', 'train')
# 	store_results(data, alg, f'results_{alg}_final{exp}_full_ws2', 'val')
# 	store_results(data, alg, f'results_{alg}_final{exp}_full_ws2', 'arch')