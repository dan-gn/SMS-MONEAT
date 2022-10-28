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
	sorted_front = sorted(population, key=lambda x: (x.fitness[0], -x.g_mean, x.fitness[1]))
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
	sorted_population = sorted(population, key=lambda x: (x.fitness[0], -x.g_mean, x.fitness[1]))
	for member in sorted_population:
		member.accuracy, member.fitness, member.g_mean = evaluate(member, x_val, y_val, True)
	# front = non_dominated_sorting_2(sorted_population)
	sorted_front = sorted(population, key=lambda x: (x.fitness[0], -x.g_mean, x.fitness[1]))
	solution = sorted_front[0].copy(True)
	return solution

def weighted_sum(x, w):
	return np.sum(np.array(x) * w)

class SolutionSelector:

	def __init__(self, method, pareto_front=False, w=None) -> None:
		self.method = method
		self.pareto_front = pareto_front
		self.w = w

	def choose(self, population: List[Genome], x_train, y_train, x_val=None, y_val=None):
		if self.method == 'WeightedSum':
			if x_val is None:
				solution = self.weighted_sum_selector(population, x_train, y_train)
			else:
				solution = self.weighted_sum_selector(population, x_val, y_val)
		elif self.method == 'Sorting':
			solution = self.sorting_selector(population, x_train, y_train, x_val, y_val)
		elif self.method == 'WSum':
			if x_val is None:
				solution = self.weighted_sum_selector(population, x_train, y_train)
			else:
				solution = self.wsum_selector(population, x_train, y_train, x_val, y_val)
		return solution

	def weighted_sum_selector(self, population: List[Genome], x, y):
		w = self.w if self.w is not None else np.array([0.35, 0.15])
		for member in population:
			member.valid = True
			member.accuracy, member.fitness, member.g_mean = evaluate(member, x, y, True)
			z = np.array([member.fitness[0], 1-member.g_mean])
			member.wsum = weighted_sum(z, w)
		front = non_dominated_sorting_2(population) if self.pareto_front else population
		sorted_population = sorted(front, key=lambda x: x.wsum)
		return sorted_population[0]

	def wsum_selector(self, population: List[Genome], x_train, y_train, x_val, y_val):
		w = self.w if self.w is not None else np.array([0.35, 0.15, 0.35, 0.15])
		for member in population:
			z = []
			member.valid = True
			member.accuracy, member.fitness, member.g_mean = evaluate(member, x_train, y_train, True)
			z.extend([member.fitness[0], 1-member.g_mean])
			member.accuracy, member.fitness, member.g_mean = evaluate(member, x_val, y_val, True)
			z.extend([member.fitness[0], 1-member.g_mean])
			member.wsum = weighted_sum(np.array(z), w)
		front = non_dominated_sorting_2(population) if self.pareto_front else population
		sorted_population = sorted(front, key=lambda x: x.wsum)
		return sorted_population[0]
		
	def sorting_selector(self, population: List[Genome], x_train, y_train, x_val=None, y_val=None):
		for member in population:
			member.valid = True
			member.accuracy, member.fitness, member.g_mean = evaluate(member, x_train, y_train, True)
		front = non_dominated_sorting_2(population) if self.pareto_front else population
		sorted_population = sorted(front, key=lambda x: (x.fitness[0], -x.g_mean, x.fitness[1]))
		if x_val is None:
			return sorted_population[0]
		else:
			return self.sorting_selector(sorted_population, x_val, y_val)