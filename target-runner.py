
import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys

from utilities.microarray_ds import MicroarrayDataset
from utilities.stats_utils import KruskalWallisFilter
from utilities.scalers import MeanScaler
from utilities.fitness_functions import torch_fitness_function
from utilities.activation_functions import Gaussian
from algorithms.n3o import N3O
from algorithms.sms_neat import SMS_NEAT


N_POPULATION = sys.argv[1]
N_ITERATIONS = sys.argv[2]
CROSSOVER_PROB = sys.argv[3]
ADD_INPUT_PROB = sys.argv[4]
SWAP_INPUT_PROB = sys.argv[5]
ADD_CONNECTION_PROB = sys.argv[6]
ADD_NODE_PROB = sys.argv[7]

dataset = 'breastCancer-full'
trn_size = 0.7
seed = 0
debug = False


params = {
	'fitness_function': torch_fitness_function,
	'n_population' : N_POPULATION, 
	'max_iterations' : N_ITERATIONS,
	'hidden_activation_function' : nn.Tanh(),
	'hidden_activation_coeff' : 4.9 * 0.5,
	'output_activation_function' : Gaussian(),
	'output_activation_coeff' : 1,
	'regularization_parameter' : 0.5,
	'crossover_prob' : CROSSOVER_PROB,
	'n_competitors' : 2,
	'disable_node_prob' : 0.75,
	'interspecies_mating_rate' : 0.001,
	'add_input_prob' : ADD_INPUT_PROB,
	'swap_input_prob' : SWAP_INPUT_PROB,
	'add_connection_prob' : ADD_CONNECTION_PROB,
	'add_node_prob' : ADD_NODE_PROB,
	'weight_mutation_prob' : 0.04,
	'pol_mutation_distr' : 5,
	'weight_mutation_sustitution_prob' : 0.1,
	'compatibility_threshold' : 3,
	'compatibility_distance_coeff' : [1.0, 1.0, 0.4],
	'stagnant_generations_threshold' : 100,
	'champion_elitism_threshold' : 5,
	'elitism_prop' : 0.1,
	'initial_weight_limits' : [-1, 1],
}


# Read microarray dataset
ds = MicroarrayDataset(f'./datasets/CUMIDA/{dataset}.arff')
x, y = ds.get_full_dataset()

# Split dataset into training and testing dataset
x_train, x_test, y_train, y_test = ds.split(trn_size=trn_size, random_state=seed)

# Statistical Filtering
filter = KruskalWallisFilter(threshold=0.01)
x_train, features_selected = filter.fit_transform(x_train, y_train)
x_test, _ = filter.transform(x_test)

# Re-Scale datasets
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)	

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
# Training the model
model = SMS_NEAT(problem, params)
model.run(seed, debug)


acc, fitness, g_mean = model.evaluate(model.best_solution_test, model.x_test, model.y_test)

print(fitness)