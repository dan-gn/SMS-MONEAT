import numpy as np
import math
import torch
import torch.nn as nn

from typing import List, Tuple

from models.genotype import Genome
from utilities.evaluation import eval_model
from utilities.fitness_functions import torch_fitness_function
from utilities.activation_functions import Gaussian
from utilities.moea_utils import non_dominated_sorting_2

activation = {}
activation['hidden_activation_function'] = nn.Tanh()
activation['hidden_activation_coeff'] = 4.9 * 0.5
activation['output_activation_function'] = Gaussian()
activation['output_activation_coeff'] = 1
fitness_function = torch_fitness_function
l2_parameter = 0.5
		
def evaluate(genome: Genome, x: torch.Tensor, y: torch.Tensor, build_model: bool = True) -> Tuple[np.float32, np.array, np.float32]:
		if build_model:
			genome.compute_phenotype(activation)
		if genome.selected_features.shape[0] == 0:
			return None, np.array([math.inf, math.inf]), 0
		x_prima = x.index_select(1, genome.selected_features)
		loss, acc, gmean = eval_model(genome.phenotype, x_prima, y, fitness_function, l2_parameter, genome.mean_square_weights)
		fitness = np.array([loss, genome.selected_features.shape[0]])
		return acc, fitness, gmean

def choose_solution_train(population: List[Genome], x, y) -> Genome:
	# solution = Genome()
	# n_objectives = len(population[0].fitness)
	# solution.fitness = np.ones(n_objectives) * math.inf
	# for member in population:
	# 	member.accuracy, member.fitness, member.g_mean = evaluate(member, x, y, True)
	# 	if member.fitness[0] < solution.fitness[0]:
	# 		solution = member.copy(True)
	# 	elif member.fitness[0] == solution.fitness[0] and member.g_mean > solution.g_mean:
	# 		solution = member.copy(True)
	for member in population:
		member.valid = True
		member.accuracy, member.fitness, member.g_mean = evaluate(member, x, y, True)
	# front = non_dominated_sorting_2(population)
	sorted_front = sorted(population, key=lambda x: (-x.g_mean, -x.accuracy, x.fitness[0], x.fitness[1]))
	solution = sorted_front[0].copy(True)
	return solution

def choose_solution_val(population: List[Genome], x_train, y_train, x_val, y_val):
	# solution_train = choose_solution_train(population, x_train, y_train)
	# sorted_population = sorted(population, key=lambda x: x.fitness[0])
	# solution = solution_train.copy(True)
	# for member in sorted_population:
	# 	member.accuracy, member.fitness, member.g_mean = evaluate(member, x_val, y_val, True)
	# 	if member.fitness[0] < solution.fitness[0]:
	# 		solution = member.copy(True)
	# 	elif member.fitness[0] == solution.fitness[0] and member.g_mean > solution.g_mean:
	# 		solution = member.copy(True)
	for member in population:
		member.valid = True
		member.accuracy, member.fitness, member.g_mean = evaluate(member, x_train, y_train, True)
	sorted_population = sorted(population, key=lambda x: (-x.g_mean, -x.accuracy, x.fitness[0], x.fitness[1]))
	for member in sorted_population:
		member.accuracy, member.fitness, member.g_mean = evaluate(member, x_val, y_val, True)
	# front = non_dominated_sorting_2(sorted_population)
	sorted_front = sorted(population, key=lambda x: (-x.g_mean, -x.accuracy, x.fitness[0], x.fitness[1]))
	solution = sorted_front[0].copy(True)
	return solution
