import numpy as np
import torch
from sklearn import svm
import os
import sys
from pathlib import Path
import pickle
import csv
from sklearn.metrics import confusion_matrix

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import SEED, N_EXPERIMENTS, N_POPULATION
from experiment_info import datasets, algorithms
from utilities.choose_solutions import SolutionSelector

from utilities.choose_solutions import choose_solution_train, choose_solution_val
from models.genotype import MultiObjectiveGenome as Genome
from algorithms.sms_moneat import SMS_MONEAT
from utilities.stats_utils import geometric_mean


def svm_fs_test(model: SMS_MONEAT, genome: Genome):
    x_train = model.x_train.index_select(1, genome.selected_features)
    x_test = model.x_test.index_select(1, genome.selected_features)
    clf = svm.SVC(kernel ='rbf', class_weight = 'balanced')
    clf.fit(x_train, model.y_train.ravel())
    y_pred = torch.tensor(clf.predict(x_test))
    return geometric_mean(model.y_test, y_pred)


data = {}

for i, alg in enumerate(algorithms):
    data[alg] = {}
    iterations = 200 if alg == 'n3o' else 18000
    for ds in datasets:
        data[alg][ds] = {}
        results_path = os.getcwd() + \
                                    f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt_final2\\{ds}"
        train = [0] * N_EXPERIMENTS
        val = [0] * N_EXPERIMENTS
        arch = [0] * N_EXPERIMENTS
        for k in range(N_EXPERIMENTS):
            results_filename = f"{ds}_MinMaxSc_{k}.pkl"
            with open(f'{results_path}/{results_filename}', 'rb') as f:
                results = pickle.load(f)
            model = results[2]['model']
            model.best_solution = choose_solution_train(model.population, model.x_train, model.y_train)
            model.best_solution_val = choose_solution_val(model.population, model.x_train, model.y_train, model.x_val, model.y_val)
            model.best_solution_archive = choose_solution_val(model.archive.get_full_population(), model.x_train, model.y_train, model.x_val, model.y_val)
            train[k] = svm_fs_test(model, model.best_solution)
            val[k] = svm_fs_test(model, model.best_solution_val)
            arch[k] = svm_fs_test(model, model.best_solution_archive)
        data[alg][ds]['train'] = np.mean(train)		
        data[alg][ds]['val'] = np.mean(val)		
        data[alg][ds]['arch'] = np.mean(arch)		
        print(f'Algorithm: {alg}; Dataset: {ds}; Train {np.mean(train)}, Val {np.mean(val)}, Arch {np.mean(arch)}')

with open('results_sms_moneat_svm.csv', 'w', newline='') as file:
	writer = csv.writer(file)
	all_rows = []
	header = ['Dataset']
	for alg in algorithms:
		header.extend([alg for _ in range(3)])
	all_rows.append(header)
	for ds in datasets:
		row = [ds]
		for alg in algorithms:
			row.extend([data[alg][ds]['train'], data[alg][ds]['val'], data[alg][ds]['arch']])
		all_rows.append(row)
	writer.writerows(all_rows)