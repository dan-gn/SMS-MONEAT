import numpy as np
import torch
import torch.nn as nn

import os
from pathlib import Path
import pickle

import time

from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utilities.microarray_ds import MicroarrayDataset
from utilities.stats_utils import KruskalWallisFilter
from utilities.scalers import MeanScaler
from utilities.fitness_functions import torch_fitness_function
from utilities.activation_functions import Gaussian
from algorithms.n3o import N3O
from algorithms.sms_moneat import SMS_MONEAT as SMS_NEAT

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", FutureWarning)


datasets = []


""" TESTING """
datasets.append('breastCancer-full') 
datasets.append('ALL-AML-full')
datasets.append('prostate_tumorVSNormal-full')

datasets.append('Breast_GSE22820') 
datasets.append('Breast_GSE59246') 
datasets.append('Breast_GSE70947')	
datasets.append('Colorectal_GSE25070')
datasets.append('Colorectal_GSE32323')
datasets.append('Colorectal_GSE44076')
datasets.append('Colorectal_GSE44861')
datasets.append('Leukemia_GSE22529_U133A') 
datasets.append('Leukemia_GSE22529_U133B') 
datasets.append('Leukemia_GSE33615')
datasets.append('Leukemia_GSE63270') 
datasets.append('Liver_GSE14520_U133A') 
datasets.append('Liver_GSE50579')
datasets.append('Liver_GSE62232') 
datasets.append('Prostate_GSE6919_U95Av2')
datasets.append('Prostate_GSE6919_U95B')
datasets.append('Prostate_GSE6919_U95C')
datasets.append('Prostate_GSE11682')

""" IRACE """
# datasets.append('Leukemia_GSE71935') # Only 9
# datasets.append('Liver_GSE57957')
# datasets.append('Prostate_GSE46602')
# datasets.append('Colorectal_GSE8671') # SMS-MONEAT 
# datasets.append('Breast_GSE42568')

seed = 0
k_folds = 10
n_repeats = 3
save_results = True
debug = False
algorithm = 'sms_moneat'
	

params = {
	'fitness_function': torch_fitness_function, 
	'n_population' : 100, 
	'max_iterations' : 200 if algorithm=='n3o' else 18000,
	'hidden_activation_function' : nn.Tanh(),
	'hidden_activation_coeff' : 4.9 * 0.5,
	'output_activation_function' : Gaussian(),
	'output_activation_coeff' : 1,
	'regularization_parameter' : 0.5,
	'crossover_prob' : 0.5925 if algorithm=='n3o' else 0.5662,
	'n_competitors' : 2,
	'disable_node_prob' : 0.75,
	'interspecies_mating_rate' : 0.001,
	'add_input_prob' : 0.0016 if algorithm=='n3o' else 0.1226,
	'swap_input_prob' : 0.1239 if algorithm=='n3o' else 0.0972,
	'add_connection_prob' : 0.0874 if algorithm=='n3o' else 0.1352,
	'add_node_prob' : 0.1339 if algorithm=='n3o' else 0.1390,
	'weight_mutation_prob' : 0.0980 if algorithm=='n3o' else 0.0498,
	'pol_mutation_distr' : 5,
	'weight_mutation_sustitution_prob' : 0.1,
	'compatibility_threshold' : 3,
	'compatibility_distance_coeff' : [1.0, 1.0, 0.4],
	'stagnant_generations_threshold' : 18001,
	'champion_elitism_threshold' : 5,
	'elitism_prop' : 0.1,
	'initial_weight_limits' : [-1, 1],
}


if __name__ == '__main__':
	
	for filename in datasets:

		# Read microarray dataset
		print(f'Reading dataset: {filename}')
		ds = MicroarrayDataset(f'./datasets/CUMIDA/{filename}.arff')
		print(f'Dataset labels: {ds.get_labels()}')
		x, y = ds.get_full_dataset()
		print(f'Total samples = {x.shape[0]}, Total features = {x.shape[1]}')
		print(f'Proportion of classes = ({np.sum(y)/y.shape[0]:.2f}, {(y.shape[0]-np.sum(y))/y.shape[0]:.2f})')

		if save_results:
			results_path = os.getcwd() + f"\\results_asc\\{algorithm}-pop_{params['n_population']}-it_{params['max_iterations']}_seed{seed}-exp2024_abc\\{filename}"
			Path(results_path).mkdir(parents=True, exist_ok=True)

		for i, x_train, x_val, x_test, y_train, y_val, y_test in ds.cross_validation_experiment(k_folds, n_repeats, seed):

			# if i < -1:
			# if i < 12 or i >= 15:	
			# if i >= 15:
			# if i < 15:
			# 	continue

			print(f'Seed = {seed}, test = {i}')
			print(f'Traning dataset samples = {x_train.shape[0]}, Test dataset samples = {x_test.shape[0]}')

			# Statistical Filtering
			# print(f'Statistical Filtering...')
			filter = KruskalWallisFilter(threshold=0.01)
			x_train, features_selected = filter.fit_transform(x_train, y_train)
			x_val, _ = filter.transform(x_val)
			x_test, _ = filter.transform(x_test)
			print(f'Remaining features after Kruskal Wallis H Test: {features_selected.shape[0]} features')

			# Re-Scale datasets
			# print(f'Scaling datasets...')
			scaler = MinMaxScaler()
			x_train = scaler.fit_transform(x_train)
			x_val = scaler.transform(x_val)
			x_test = scaler.transform(x_test)

			# print(f'Final preprocessing data steps...')
			y_train = np.expand_dims(y_train, axis=1)
			y_val = np.expand_dims(y_val, axis=1)
			y_test = np.expand_dims(y_test, axis=1)

			x_train = torch.from_numpy(x_train).type(torch.float32)
			y_train = torch.from_numpy(y_train).type(torch.float32)
			x_val = torch.from_numpy(x_val).type(torch.float32)
			y_val = torch.from_numpy(y_val).type(torch.float32)
			x_test = torch.from_numpy(x_test).type(torch.float32)
			y_test = torch.from_numpy(y_test).type(torch.float32)

			problem = {
				'x_train' : x_train,
				'y_train' : y_train,
				'x_val' : x_val,
				'y_val' : y_val,
				'x_test' : x_test,
				'y_test' : y_test,
				'kw_htest_pvalue' : filter.p_value,
				'labels' : ds.get_labels()
			}

			"""
			TRAIN MODEL
			"""

			print(f'Traning model...')
			record = {}
			if algorithm == 'n3o':
				model = N3O(problem, params)
			elif algorithm == 'sms_moneat':
				model = SMS_NEAT(problem, params)
			start = time.time()
			record['final'], record['archive'] = model.run(i, debug)
			time_exec = time.time()- start
			# neat.best_solution.describe()

			"""
			DISPLAY RESULTS
			"""
			print(f'Time Execution: {time_exec}, invalid_nets: {model.n_invalid_nets}')
			print(f"Record Size: {len(record['final'].population)}, Archive record size: {len(record['archive'].population)}")

			print('Best solution: Train Dataset')
			input_nodes, hidden_nodes, output_nodes = model.best_solution.count_nodes()
			print(f'Best solution topology: [{input_nodes}, {hidden_nodes}, {output_nodes}], FS: {model.best_solution.selected_features.shape[0]}')
			acc, fitness, g_mean = model.evaluate(model.best_solution, model.x_train, model.y_train)
			print(f'Train dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')
			acc, fitness, g_mean = model.evaluate(model.best_solution, model.x_val, model.y_val)
			print(f'Val dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')
			acc, fitness, g_mean = model.evaluate(model.best_solution, model.x_test, model.y_test)
			print(f'Test dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')

			print('Best solution: Val Dataset')
			input_nodes, hidden_nodes, output_nodes = model.best_solution_val.count_nodes()
			print(f'Best solution topology: [{input_nodes}, {hidden_nodes}, {output_nodes}], FS: {model.best_solution_val.selected_features.shape[0]}')
			acc, fitness, g_mean = model.evaluate(model.best_solution_val, model.x_train, model.y_train)
			print(f'Train dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')
			acc, fitness, g_mean = model.evaluate(model.best_solution_val, model.x_val, model.y_val)
			print(f'Val dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')
			acc, fitness, g_mean = model.evaluate(model.best_solution_val, model.x_test, model.y_test)
			print(f'Test dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')

			print('Best solution: Arch Dataset')
			input_nodes, hidden_nodes, output_nodes = model.best_solution_archive.count_nodes()
			print(f'Best solution topology: [{input_nodes}, {hidden_nodes}, {output_nodes}], FS: {model.best_solution_archive.selected_features.shape[0]}')
			acc, fitness, g_mean = model.evaluate(model.best_solution_archive, model.x_train, model.y_train)
			print(f'Train dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')
			acc, fitness, g_mean = model.evaluate(model.best_solution_archive, model.x_val, model.y_val)
			print(f'Val dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')
			acc, fitness, g_mean = model.evaluate(model.best_solution_archive, model.x_test, model.y_test)
			print(f'Test dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')

			print('\n')

			if save_results:
				result = {'seed' : seed, 'cv_it' : i, 'model' : model, 'time' : time_exec}
				problem['fitness_function'] = 'torch_fitness_function'	
				results_filename = f"{filename}_MinMaxSc_{i}.pkl"
				with open(f'{results_path}/{results_filename}', 'wb') as f:
					pickle.dump([problem, params, result], f)
				with open(f'{results_path}/record_{results_filename}', 'wb') as f:
					pickle.dump(record, f)
				problem['fitness_function'] = torch_fitness_function	