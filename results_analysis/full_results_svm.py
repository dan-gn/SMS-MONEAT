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
from experiment_info import datasets, algorithms, iter_num, experiment
from utilities.choose_solutions import SolutionSelector, SolutionSelector2

from models.genotype import MultiObjectiveGenome as Genome
from utilities.stats_utils import geometric_mean


def svm_fs_test(genome: Genome, X_train, y_train, X_test, y_test):
    x_train = X_train.index_select(1, genome.selected_features)
    x_test = X_test.index_select(1, genome.selected_features)
    clf = svm.SVC(kernel ='rbf', class_weight = 'balanced')
    clf.fit(x_train, y_train.ravel())
    y_pred = torch.tensor(clf.predict(x_test))
    y_real = y_test.squeeze(dim=1)
    acc = (y_real == torch.round(y_pred)).type(torch.float32).mean()
    # return geometric_mean(y_test, y_pred)
    return acc


data = {}

selector = SolutionSelector(method='WSum', pareto_front=False)
selector2 = SolutionSelector2(method='WSum', pareto_front=False)

for i, alg in enumerate(algorithms):
    data[alg] = {}
    iterations = iter_num[alg]
    exp = experiment[alg]
    for ds in datasets:
        data[alg][ds] = {}
        # results_path = os.getcwd() + f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt_final{exp}\\{ds}"
        results_path = os.getcwd() + f'\\results_asc\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-exp{exp}_rest14\\{ds}'
        train = [0] * N_EXPERIMENTS
        val = [0] * N_EXPERIMENTS
        arch = [0] * N_EXPERIMENTS
        for k in range(N_EXPERIMENTS):
            results_filename = f"{ds}_MinMaxSc_{k}.pkl"
            with open(f'{results_path}/{results_filename}', 'rb') as f:
                results = pickle.load(f)
            model = results[2]['model']
            if alg != 'sms_emoa':
                model.best_solution = selector.choose(model.population, model.x_train, model.y_train)
                model.best_solution_val = selector.choose(model.population, model.x_train, model.y_train, model.x_val, model.y_val)
                model.best_solution_archive = selector.choose(model.archive.get_full_population(), model.x_train, model.y_train, model.x_val, model.y_val)
                X_train = torch.concat((model.x_train, model.x_val))
                y_train = torch.concat((model.y_train, model.y_val))
                train[k] = svm_fs_test(model.best_solution, X_train, y_train, model.x_test, model.y_test)
                val[k] = svm_fs_test(model.best_solution_val, X_train, y_train, model.x_test, model.y_test)
                arch[k] = svm_fs_test(model.best_solution_archive, X_train, y_train, model.x_test, model.y_test)
            else:
                model.best_solution = selector2.choose(model.population, model.x_train, model.y_train)
                features_selected = [i for i, xi in enumerate(model.best_solution.genome) if xi == 1]
                model.best_solution.selected_features = torch.tensor(features_selected)
                train[k] = svm_fs_test(model.best_solution, model.x_train, model.y_train, model.x_test, model.y_test)
        data[alg][ds]['train'] = train		
        data[alg][ds]['val'] = val		
        data[alg][ds]['arch'] = arch		
        print(f'Algorithm: {alg}; Dataset: {ds}; Train {np.mean(train)}, Val {np.mean(val)}, Arch {np.mean(arch)}')


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

# store_results(data, alg, f'results_{alg}_final{exp}_full_svm', 'train')
# store_results(data, alg, f'results_{alg}_final{exp}_full_svm', 'val')
# store_results(data, alg, f'results_{alg}_final{exp}_full_svm', 'arch')

store_results(data, alg, f'results_{alg}_final{exp}_full_it{iterations}_rest14_svm', 'train')
store_results(data, alg, f'results_{alg}_final{exp}_full_it{iterations}_rest14_svm', 'val')
store_results(data, alg, f'results_{alg}_final{exp}_full_it{iterations}_rest14_svm', 'arch')