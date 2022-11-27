import numpy as np
import os
import sys
import pickle
import csv
import copy

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from models.genotype import Genome
from experiment_info import SEED, N_EXPERIMENTS, N_POPULATION
from experiment_info import datasets, algorithms, iter_num, experiment
from utilities.choose_solutions import SolutionSelector, SolutionSelector2, evaluate3

selector = SolutionSelector(method='WSum', pareto_front=False)
selector2 = SolutionSelector2(method='WSum', pareto_front=False)

class ANNInfo:

    def __init__(self) -> None:
        self.fs = [0] * N_EXPERIMENTS
        self.hidden = [0] * N_EXPERIMENTS
        self.connections = [0] * N_EXPERIMENTS
        self.fs_active = [0] * N_EXPERIMENTS
        self.hidden_active = [0] * N_EXPERIMENTS
        self.connections_active = [0] * N_EXPERIMENTS

    def add(self, model: Genome, n_experiment):
        self.fs[n_experiment] = len([node for node in model.node_genes if node.node_type == 'input'])
        self.hidden[n_experiment] = len([node for node in model.node_genes if node.node_type == 'hidden'])
        self.connections[n_experiment] = len(model.connection_genes)
        model.tag_layers()
        self.fs_active[n_experiment] = len([node for node in model.node_genes if node.layer == 0])
        self.hidden_active[n_experiment] = len([node for node in model.node_genes if node.layer is not None and node.node_type == 'hidden'])
        self.connections_active[n_experiment] = model.n_active_connections.numpy()
    
    def get_data(self):
        return [self.fs, self.hidden, self.connections, self.fs_active, self.hidden_active, self.connections_active]

data = {}
for i, alg in enumerate(algorithms):
    data[alg] = {}
    iterations = iter_num[alg]
    exp = experiment[alg]
    for ds in datasets:
        data[alg][ds] = {}
        results_path = os.getcwd() + f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt_final{exp}\\{ds}"
        pop_info = ANNInfo()
        arch_info = ANNInfo()
        for k in range(N_EXPERIMENTS):
            results_filename = f"{ds}_MinMaxSc_{k}.pkl"
            with open(f'{results_path}/{results_filename}', 'rb') as f:
                results = pickle.load(f)
            model = results[2]['model']

            # Choose solutions
            if alg != 'sms_emoa':
                model.best_solution = selector.choose(model.population, model.x_train, model.y_train, model.x_val, model.y_val)
                model.best_solution_archive = selector.choose(model.archive.get_full_population(), model.x_train, model.y_train, model.x_val, model.y_val)
                pop_info.add(model.best_solution, k)
                arch_info.add(model.best_solution_archive, k)
            else:
                model.best_solution = selector2.choose(model.population, model.x_train, model.y_train)
                pop_info.add(model.best_solution, k)

			
        data[alg][ds]['val'] = copy.deepcopy(pop_info)
        data[alg][ds]['arch'] = copy.deepcopy(arch_info)
        print(f'Algorithm: {alg}; Dataset: {ds}; FS {np.mean(pop_info.fs_active)}; Hidden {np.mean(pop_info.hidden_active)}, Connections {np.mean(pop_info.connections_active)}')

def store_results(data, alg, filename, population):
	with open(f'final_exp/{alg}/{filename}_{population}.csv', 'w', newline='') as file:
		writer = csv.writer(file)
		all_rows = []
		header = ['Dataset']
		header.extend([i for i in range(N_EXPERIMENTS)])
		all_rows.append(header)
		for ds in datasets:
			row = [ds]
			row.extend(list(data[alg][ds][population].fs))
			all_rows.append(row)
		for ds in datasets:
			row = [ds]
			row.extend(list(data[alg][ds][population].hidden))
			all_rows.append(row)
		for ds in datasets:
			row = [ds]
			row.extend(list(data[alg][ds][population].connections))
			all_rows.append(row)
		for ds in datasets:
			row = [ds]
			row.extend(list(data[alg][ds][population].fs_active))
			all_rows.append(row)
		for ds in datasets:
			row = [ds]
			row.extend(list(data[alg][ds][population].hidden_active))
			all_rows.append(row)
		for ds in datasets:
			row = [ds]
			row.extend(list(data[alg][ds][population].connections_active))
			all_rows.append(row)
		writer.writerows(all_rows)

store_results(data, alg, f'results_{alg}_final{exp}_full_ANNInfo', 'val')
store_results(data, alg, f'results_{alg}_final{exp}_full_ANNInfo', 'arch')
