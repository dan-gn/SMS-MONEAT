import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utilities.microarray_ds import MicroarrayDataset
from utilities.stats_utils import KruskalWallisFilter
from utilities.scalers import MeanScaler
from utilities.fitness_functions import torch_fitness_function
from utilities.activation_functions import Gaussian
from algorithms.n3o import N3O
from algorithms.sms_moneat import SMS_NEAT


datasets = []

datasets.append('breastCancer-full')
# datasets.append('ALL-AML-full')
# datasets.append('prostate_tumorVSNormal-full')

# datasets.append('Breast_GSE42568')
# datasets.append('Colorectal_GSE8671')
# datasets.append('Colorectal_GSE32323')
# datasets.append('Colorectal_GSE44076')
# datasets.append('Colorectal_GSE44861')
# datasets.append('Leukemia_GSE14317')
# datasets.append('Leukemia_GSE63270')
# datasets.append('Leukemia_GSE71935')

# datasets.append('Breast_GSE22820')
# datasets.append('Breast_GSE59246')
# datasets.append('Breast_GSE70947')
# datasets.append('Liver_GSE14520_U133A')
# datasets.append('Liver_GSE50579')
# datasets.append('Liver_GSE62232')
# datasets.append('Prostate_GSE6919_U95Av2')
# datasets.append('Prostate_GSE46602')
# datasets.append('Prostate_GSE11682')
# datasets.append('Leukemia_GSE33615')

trn_size = 0.7
initial_seed = 0
n_tests = 1
save_results = True
algorithm = 'sms_neat'
	

params = {
	'fitness_function': torch_fitness_function, 
	'n_population' : 100, 
	'max_iterations' : 100,
	'hidden_activation_function' : nn.Tanh(),
	'hidden_activation_coeff' : 4.9 * 0.5,
	'output_activation_function' : Gaussian(),
	'output_activation_coeff' : 1,
	'regularization_parameter' : 0.5,
	'crossover_prob' : 0.75,
	'n_competitors' : 2,
	'disable_node_prob' : 0.75,
	'interspecies_mating_rate' : 0.001,
	'add_input_prob' : 0.05,
	'swap_input_prob' : 0.05,
	'add_connection_prob' : 0.05,
	'add_node_prob' : 0.03,
	'weight_mutation_prob' : 0.04,
	'pol_mutation_distr' : 5,
	'weight_mutation_sustitution_prob' : 0.1,
	'compatibility_threshold' : 3,
	'compatibility_distance_coeff' : [1.0, 1.0, 0.4],
	'stagnant_generations_threshold' : 200,
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
			results_path = os.getcwd() + f"\\results\\{algorithm}-pop_{params['n_population']}-it_{params['max_iterations']}_2\\{filename}"
			Path(results_path).mkdir(parents=True, exist_ok=True)

		for seed in range(initial_seed, initial_seed+n_tests):

			print(f'Execution = {seed}, seed = {seed}')

			debug = True if seed == -1 else False

			# Split dataset into training and testing dataset
			print(f'Spliting into training and testin datasets. Proportion of training dataset = {trn_size * 100:.2f}%')
			x_train, x_test, y_train, y_test = ds.split(trn_size=trn_size, random_state=seed)

			print(f'Traning dataset samples = {x_train.shape[0]}, Test dataset samples = {x_test.shape[0]}')

			# Statistical Filtering
			# print(f'Statistical Filtering...')
			filter = KruskalWallisFilter(threshold=0.01)
			x_train, features_selected = filter.fit_transform(x_train, y_train)
			x_test, _ = filter.transform(x_test)
			print(f'Remaining features after Kruskal Wallis H Test: {features_selected.shape[0]} features')

			# Re-Scale datasets
			# print(f'Scaling datasets...')
			scaler = MinMaxScaler()
			x_train = scaler.fit_transform(x_train)
			x_test = scaler.transform(x_test)

			# print(f'Final preprocessing data steps...')
			y_train = np.expand_dims(y_train, axis=1)
			y_test = np.expand_dims(y_test, axis=1)

			x_train = torch.from_numpy(x_train).type(torch.float32)
			y_train = torch.from_numpy(y_train).type(torch.float32)
			x_test = torch.from_numpy(x_test).type(torch.float32)
			y_test = torch.from_numpy(y_test).type(torch.float32)

			problem = {
				'x_train' : x_train,
				'y_train' : y_train,
				'x_test' : x_test,
				'y_test' : y_test,
				'kw_htest_pvalue' : filter.p_value,
				'labels' : ds.get_labels()
			}

			"""
			TRAIN MODEL
			"""

			print(f'Traning model...')
			if algorithm == 'n3o':
				model = N3O(problem, params)
			elif algorithm == 'sms_neat':
				model = SMS_NEAT(problem, params)

			model.run(seed, debug)
			# neat.best_solution.describe()

			"""
			DISPLAY RESULTS
			"""
			input_nodes, hidden_nodes, output_nodes = model.best_solution_test.count_nodes()
			print(f'Best solution topology: [{input_nodes}, {hidden_nodes}, {output_nodes}]')

			acc, fitness, g_mean = model.evaluate(model.best_solution_test, model.x_train, model.y_train)
			print(f'Train dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')

			acc, fitness, g_mean = model.evaluate(model.best_solution_test, model.x_test, model.y_test)
			print(f'Test dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')


			if algorithm == 'sms_neat':
				acc, fitness, g_mean = model.evaluate(model.best_solution_archive, model.x_test, model.y_test)
				print(f'Test dataset: fitness = {fitness}, accuracy = {acc}, g mean = {g_mean}')

			if save_results:
				result = {'seed' : seed, 'model' : model}
				problem['fitness_function'] = 'torch_fitness_function'	
				results_filename = f"{filename}_{seed}_MinMaxSc.pkl"
				with open(f'{results_path}/{results_filename}', 'wb') as f:
					pickle.dump([problem, params, result], f)
				problem['fitness_function'] = torch_fitness_function	