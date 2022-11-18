import numpy as np
import torch
from sklearn import svm
import os
import sys
from pathlib import Path
import pickle
import csv
import pygmo

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import SEED, N_EXPERIMENTS, N_POPULATION
from experiment_info import datasets, algorithms, iter_num, experiment
from utilities.choose_solutions import evaluate

from utilities.choose_solutions import evaluate3
from utilities.moea_utils import non_dominated_sorting_2

C = np.array([1, 0.1], dtype=np.float32)
REF = np.array([60, 15], dtype=np.float32)

def hv_test(population, X, y):
    # Evaluate data
    for member in population:
        _, member.fitness, _ = evaluate(member, X, y, True)
    front = non_dominated_sorting_2(population)
    fitness = np.array([np.array(member.fitness, np.float32) for member in front[0]])
    fitness = fitness * C
    # Compute HV
    hv = pygmo.hypervolume(fitness)
    return hv.compute(REF), fitness.max(axis=0)

def hv_test2(population, X_train, y_train, X_test, y_test):
    # Evaluate data
    for member in population:
        _, member.fitness, _ = evaluate3(member, X_train, y_train, X_test, y_test)
    front = non_dominated_sorting_2(population)
    fitness = np.array([np.array(member.fitness, np.float32) for member in front[0]])

    fitness = fitness * C
    # Compute HV
    hv = pygmo.hypervolume(fitness)
    return hv.compute(REF), fitness.max(axis=0)

max_values = [0, 0] 

data = {}

for i, alg in enumerate(algorithms):
    data[alg] = {}
    iterations = iter_num[alg]
    exp = experiment[alg]
    for ds in datasets:
        data[alg][ds] = {}
        results_path = os.getcwd() + \
                                    f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt_final{exp}\\{ds}"
        train = [0] * N_EXPERIMENTS
        arch = [0] * N_EXPERIMENTS
        for k in range(N_EXPERIMENTS):
            results_filename = f"{ds}_MinMaxSc_{k}.pkl"
            with open(f'{results_path}/{results_filename}', 'rb') as f:
                results = pickle.load(f)
            model = results[2]['model']
            if alg != 'sms_emoa':
                train[k], m = hv_test(model.population, model.x_test, model.y_test)
                max_values[0] = max(max_values[0], m[0])
                max_values[1] = max(max_values[1], m[1])
                arch[k], m = hv_test(model.archive.get_full_population(), model.x_test, model.y_test)
                max_values[0] = max(max_values[0], m[0])
                max_values[1] = max(max_values[1], m[1])
            else:
                train[k], m = hv_test2(model.population, model.x_train, model.y_train, model.x_test, model.y_test)
                max_values[0] = max(max_values[0], m[0])
                max_values[1] = max(max_values[1], m[1])
        data[alg][ds]['train'] = train		
        data[alg][ds]['arch'] = arch		
        print(f'Algorithm: {alg}; Dataset: {ds}; Train {np.mean(train)}, Arch {np.mean(arch)}, Max values {max_values}')


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
		writer.writerows(all_rows)

store_results(data, alg, f'results_{alg}_final{exp}_full_hv', 'train')
store_results(data, alg, f'results_{alg}_final{exp}_full_hv', 'arch')
