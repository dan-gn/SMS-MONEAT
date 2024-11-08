import numpy as np
import torch
import torch.nn as nn
import math
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings
warnings.simplefilter("ignore", UserWarning)

import sys
sys.path.insert(0, '../..')

from utilities.microarray_ds import MicroarrayDataset
from utilities.stats_utils import KruskalWallisFilter
from utilities.fitness_functions import torch_fitness_function
from utilities.activation_functions import Gaussian
from pygmo import hypervolume
from utilities.moea_utils import non_dominated_sorting
from algorithms.sms_moneat import SMS_MONEAT
from algorithms.sms_emoa import SMS_EMOA

configuration_id = sys.argv[1]
instance_id = sys.argv[2]
seed = int(sys.argv[3])
instance = sys.argv[4]
irace_params = sys.argv[5:]
N_POPULATION = 100
N_ITERATIONS = 6000
CROSSOVER_PROB = float(irace_params[1])
MUTATION_PROB = float(irace_params[3])

dataset = instance.split('/')[-1]
trn_size = 0.70
debug = False


params = {
	'fitness_function': torch_fitness_function,
	'n_population' : int(N_POPULATION), 
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
	'mutation_prob' : MUTATION_PROB,
	'pol_mutation_distr' : 5,
	'weight_mutation_sustitution_prob' : 0.1,
	'compatibility_threshold' : 3,
	'compatibility_distance_coeff' : [1.0, 1.0, 0.4],
	'stagnant_generations_threshold' : math.inf,
	'champion_elitism_threshold' : 5,
	'elitism_prop' : 0.1,
	'initial_weight_limits' : [-1, 1],
}


# Read microarray dataset
import os
ds = MicroarrayDataset(f'{os.path.abspath(os.getcwd())}\\..\\..\\datasets\\CUMIDA\\{dataset}.arff')
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
	'x_val' : x_test,
	'y_val' : y_test,
	'x_test' : x_test,
	'y_test' : y_test,
	'kw_htest_pvalue' : filter.p_value,
	'labels' : ds.get_labels()
}


"""
TRAIN MODEL
"""
# Training the model
model = SMS_EMOA(problem, params)
model.run(seed, debug=False)


reference = np.array([60, 15])
alpha = 10
for member in model.population:
	member.accuracy, member.fitness, _ = model.evaluate(member, model.x_test, model.y_test, True)
front = non_dominated_sorting(model.population)
fitness = np.array([list([f.fitness[0], f.fitness[1]/alpha]) for f in front[0]])
unq, count = np.unique(fitness, axis=0, return_counts=True)
hv = hypervolume(unq)

target = -hv.compute(reference)
# target = np.mean([member.fitness[0] for member in model.population if member.accuracy is not None])
print(target)