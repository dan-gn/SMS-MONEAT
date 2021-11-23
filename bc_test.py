"""
Import required libraries and modules
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from scipy.io import arff
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import pickle

from algorithms.n3o import N3O
from utilities.activation_functions import Gaussian
from utilities.fitness_functions import torch_fitness_function
from utilities.scalers import MeanScaler
from utilities.stats_utils import KruskalWallisFilter



params = {
	'fitness_function': torch_fitness_function,
	'n_population' : 1000, 
	'max_iterations' : 200,
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
	'stagnant_generations_threshold' : 15,
	'champion_elitism_threshold' : 5,
	'elitism_prop' : 0.1,
	'initial_weight_limits' : [-1, 1],
}


if __name__ == '__main__':

	"""
	DATASET
	"""

	# Load train dataset
	data = arff.loadarff('datasets/breastCancer-train.arff')
	df_train = pd.DataFrame(data[0])

	# Load test dataset
	data = arff.loadarff('datasets/breastCancer-test.arff')
	df_test = pd.DataFrame(data[0])

	# Change category class label to binary class label
	labels = {b'relapse' : 1, b'non-relapse' : 0}
	df_train['Class'] = df_train['Class'].replace(labels)
	df_test['Class'] = df_test['Class'].replace(labels)

	# Count class distribution from both datasets
	n_relapsed_train = np.sum(df_train['Class'].to_numpy(dtype=np.float32))
	n_non_relapsed_train = df_train.shape[0] - n_relapsed_train
	n_relapsed_test = np.sum(df_test['Class'].to_numpy(dtype=np.float32))
	n_non_relapsed_test = df_test.shape[0] - n_relapsed_test

	# Print information
	print(f"Train dataset shape: {df_train.shape}, Relapsed instances: {n_relapsed_train}, Non-Relapsed instances: {n_non_relapsed_train}")
	print(f"Test dataset shape: {df_test.shape}, Relapsed instances: {n_relapsed_test}, Non-Relapsed instances: {n_non_relapsed_test}")

	"""
	Preprocess data
	"""
	# Convert train dataset to Numpy array
	x_train = df_train.iloc[:, :-1].to_numpy(dtype=np.float32)
	y_train = df_train.iloc[:, -1].to_numpy(dtype=np.float32)

	# Convert test dataset to Numpy array
	x_test = df_test.iloc[:, :-1].to_numpy(dtype=np.float32)
	y_test = df_test.iloc[:, -1].to_numpy(dtype=np.float32)

	# Kruskal Wallis H Test
	filter = KruskalWallisFilter(threshold=0.01)
	x_train_kw, kw_feature_selected = filter.fit_transform(x_train, y_train)
	x_test_kw, _ = filter.transform(x_test)
	kw_pvalue = filter.p_value

	print(f'Attributes selected after KW H Test: {kw_feature_selected.shape[0]}')

	# Normalize data
	scaler = MinMaxScaler()
	x_train_norm = scaler.fit_transform(x_train_kw)
	x_test_norm = scaler.transform(x_test_kw)

	# Preprocess training and testing dataset
	y_train = np.expand_dims(y_train, axis=1)
	y_test = np.expand_dims(y_test, axis=1)
	
	x_train = torch.from_numpy(x_train_norm).type(torch.float32)
	y_train = torch.from_numpy(y_train).type(torch.float32)
	x_test = torch.from_numpy(x_test_norm).type(torch.float32)
	y_test = torch.from_numpy(y_test).type(torch.float32)

	problem = {
		'x_train' : x_train,
		'y_train' : y_train,
		'x_test' : x_test,
		'y_test' : y_test,
		'kw_htest_pvalue' : kw_pvalue
	}

	"""
	TRAIN MODEL
	"""
	
	for seed in range(20):

		print(f'Execution = {seed}, seed = {seed}')

		debug = True if seed == -1 else False

		problem['fitness_function'] = torch_fitness_function	
		model = N3O(problem, params)
		model.run(seed, debug)
		# neat.best_solution.describe()
		result = {'seed' : seed, 'model' : model}

		"""
		DISPLAY RESULTS
		"""

		acc, fitness = model.evaluate(model.best_solution, model.x_train, model.y_train)
		print(f'Train dataset: fitness = {fitness}, accuracy = {acc} ')

		acc, fitness = model.evaluate(model.best_solution, model.x_test, model.y_test)
		print(f'Test dataset: fitness = {fitness}, accuracy = {acc} ')

		problem['fitness_function'] = 'torch_fitness_function'	
		filename = f"breast_{seed}_it{params['max_iterations']}_MinMaxSc_f.pkl"
		with open(f'results/{filename}', 'wb') as f:
			pickle.dump([problem, params, result], f)