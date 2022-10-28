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

alg = 'sms_moneat'
iterations =  18000
data = {}


for ds in datasets:
    for i in range(0, 6):
        alpha = i * 0.1
        beta = 0.5 - alpha
        w = np.array([alpha, beta] * 2)
        selector = SolutionSelector(method='WSum', pareto_front=False, w = w)

        data[ds] = {}
        results_path = os.getcwd() + f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt\\{ds}"
        time = [0] * N_EXPERIMENTS
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
            model.best_solution_val = selector.choose(model.population, model.x_train, model.y_train, model.x_val, model.y_val)
            model.best_solution_archive = selector.choose(model.archive.get_full_population(), model.x_train, model.y_train, model.x_val, model.y_val)
            model.best_solution.valid, model.best_solution_val.valid, model.best_solution_archive.valid = True, True, True	
            _, _, val[k] = model.evaluate(model.best_solution_val, model.x_test, model.y_test)
            _, _, arch[k] = model.evaluate(model.best_solution_archive, model.x_test, model.y_test)
            val_fs[k] = model.best_solution_val.selected_features.shape[0]
            arch_fs[k] = model.best_solution_archive.selected_features.shape[0]
        data[ds]['time'] = np.mean(time)		
        data[ds]['val'] = np.mean(val)		
        data[ds]['val_fs'] = np.mean(val_fs)		
        data[ds]['arch'] = np.mean(arch)		
        data[ds]['arch_fs'] = np.mean(arch_fs)		
        print(f'Algorithm: {alg}; Dataset: {ds}; w {w}; Val {np.mean(val)}, Arch {np.mean(arch)}')

with open('results_weights.csv', 'w', newline='') as file:
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