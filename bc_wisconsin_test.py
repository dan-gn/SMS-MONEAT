from algorithms.neat import NEAT, set_seed
from algorithms.fs_neat import FS_NEAT
from algorithms.n3o import N3O
from models.ann import ArtificialNeuralNetwork
from models.ann_pytorch import Ann_PyTorch, eval_model
from models.genotype import Genome, NodeGene, ConnectionGene
from utilities.activation_functions import gaussian, Gaussian
from utilities.scalers import MeanScaler
from utilities.fitness_functions import fitness_function, torch_fitness_function
from utilities.stats_utils import KruskalWallisFilter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import MinMaxScaler

params = {
	'fitness_function' : torch_fitness_function,
	'n_population' : 1000, 
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
	'stagnant_generations_threshold' : 15,
	'champion_elitism_threshold' : 5,
	'elitism_prop' : 0.1,
	'initial_weight_limits' : [-1, 1],
}


if __name__ == '__main__':

	"""
	Dataset and pre-processing
	"""
	# Read dataset
	df = pd.read_csv('datasets/breast-cancer-wisconsin.data', delimiter=',', header=None)
	df = df.drop([6], axis=1)

	# Remove column with incomplete data
	x = df.iloc[:, 1:-1].to_numpy(dtype=np.float32)

	# Set label data between 0 and 1
	y = df.iloc[:, -1].to_numpy(dtype=np.float32)
	y = np.round((y - 2) / 2)

	# Divide dataset in training set and validation set
	x_train = x[:640, :]
	y_train = y[:640]
	x_test = x[640:, :]
	y_test = y[640:]


	# Count class distribution from both datasets
	n_relapsed_train = np.sum(y_train)
	n_non_relapsed_train = y_train.shape[0] - n_relapsed_train
	n_relapsed_test = np.sum(y_test)
	n_non_relapsed_test = y_test.shape[0] - n_relapsed_test

	# Print information
	print(f"Train dataset shape: {x_train.shape[0]}, Relapsed instances: {n_relapsed_train}, Non-Relapsed instances: {n_non_relapsed_train}")
	print(f"Test dataset shape: {x_test.shape[0]}, Relapsed instances: {n_relapsed_test}, Non-Relapsed instances: {n_non_relapsed_test}")

	# Kruskal Wallis H Test
	filter = KruskalWallisFilter(threshold=0.01)
	x_train_kw, kw_feature_selected = filter.fit_transform(x_train, y_train)
	x_test_kw, _ = filter.transform(x_test)
	kw_pvalue = filter.p_value

	print(f'Attributes selected after KW H Test: {kw_feature_selected.shape[0]}')

	# Normalized datasets
	scaler = MinMaxScaler()
	x_train_norm = scaler.fit_transform(x_train_kw)
	x_test_norm = scaler.transform(x_test_kw)

	print(x_train_norm.shape)

	# Preprocess dataset
	# bias = np.ones((x_test.shape[0], 1))
	# x_train_norm = np.concatenate((bias, x_train_norm), axis=1)
	# bias = np.ones((x_test.shape[0], 1))
	# x_test_norm = np.concatenate((bias, x_test_norm), axis=1)

	y_train = np.expand_dims(y_train, axis=1)
	y_test = np.expand_dims(y_test, axis=1)
	
	x_train_torch = torch.from_numpy(x_train_norm).type(torch.float32)
	y_train_torch = torch.from_numpy(y_train).type(torch.float32)
	x_test_torch = torch.from_numpy(x_test_norm).type(torch.float32)
	y_test_torch = torch.from_numpy(y_test).type(torch.float32)

	"""
	Training ANNs
	"""
	problem = {
		'x_train' : x_train_torch,
		'y_train' : y_train_torch,
		'x_test' : x_test_torch,
		'y_test' : y_test_torch,
		'kw_htest_pvalue' : kw_pvalue
	}

	for seed in range(1):

		print(f'Execution = {seed}, seed = {seed}')

		debug = True if seed == -1 else False

		problem['fitness_function'] = torch_fitness_function	
		model = N3O(problem, params)
		model.run(seed, debug)
		# neat.best_solution.describe()
		result = {'seed' : seed, 'model' : model}

		"""
		Print results
		"""
		acc, fitness = model.evaluate(model.best_solution, model.x_train, model.y_train)
		print(f'Train dataset: fitness = {fitness}, accuracy = {acc} ')

		acc, fitness = model.evaluate(model.best_solution, model.x_test, model.y_test)
		print(f'Test dataset: fitness = {fitness}, accuracy = {acc} ')

		# problem['fitness_function'] = 'torch_fitness_function'	
		# filename = f"neat_bc_wis_test_seed_{seed}_it{params['max_iterations']}_f.pkl"
		# with open(f'results/{filename}', 'wb') as f:
		# 	pickle.dump([problem, params, result], f)

		"""
		Test results
		"""
		# activation = {}
		# activation['hidden_activation_function'] = np.tanh
		# activation['hidden_activation_coeff'] = params['hidden_activation_coeff']
		# activation['output_activation_function'] = gaussian
		# activation['output_activation_coeff'] = params['output_activation_coeff']

		# model = ArtificialNeuralNetwork(neat.best_solution, activation)

		# loss, acc = model.eval_model(x_train_norm, y_train, fitness_function, params['regularization_parameter'])
		# fitness = 100 - loss
		# print(f'Train dataset: fitness = {fitness}, accuracy = {acc} ')

		# loss, acc = model.eval_model(x_test_norm, y_test, fitness_function, params['regularization_parameter'])
		# fitness = 100 - loss
		# print(f'Train dataset: fitness = {fitness}, accuracy = {acc} ')



	