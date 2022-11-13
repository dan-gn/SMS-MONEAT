import numpy as np
import torch
import math
import copy
from typing import Tuple, List

import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from algorithms.neat import set_seed
from utilities.ga_utils import single_point_crossover, tournament_selection, binary_mutation
from utilities.moea_utils import choose_min_hv_contribution, non_dominated_sorting_2
from utilities.moea_utils import add_genome_nds, remove_genome_nds, create_fronts
from utilities.data_utils import choose_repeated_index
from utilities.stats_utils import geometric_mean
from utilities.fitness_functions import torch_fitness_function
from utilities.record import KRecord

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold

class Individual:

	def __init__(self, n_variables: int = 1) -> None:
		self.n_variables = n_variables
		self.genome = None	
		self.fitness = None
		self.accuracy = None
		self.g_mean = None
		self.rank = None
		self.dominates_to = []
		self.n_dominated_by = 0
		self.reduced_genome = []

	def initialize(self):
		self.genome = np.zeros(self.n_variables)
		index = np.random.randint(self.n_variables)
		self.genome[index] = 1

	def copy(self):
		new_individual = Individual(self.n_variables)
		new_individual.genome = np.copy(self.genome)
		new_individual.rank = self.rank
		new_individual.fitness = self.fitness
		new_individual.accuracy = self.accuracy
		new_individual.g_mean = self.g_mean
		new_individual.n_dominated_by = self.n_dominated_by
		new_individual.dominates_to = list(self.dominates_to)
		return new_individual

	def expand_genome(self):
		self.genome = np.array([1 if i in self.reduced_genome else 0 for i in range(self.n_variables)])

	def reduced_copy(self):
		new_individual = Individual(self.n_variables)
		new_individual.reduced_genome = [i for i in self.genome if i == 1]
		new_individual.rank = self.rank
		new_individual.fitness = self.fitness
		new_individual.accuracy = self.accuracy
		new_individual.g_mean = self.g_mean
		new_individual.n_dominated_by = self.n_dominated_by
		new_individual.dominates_to = list(self.dominates_to)
		return new_individual







class SMS_EMOA:

	def __init__(self, problem: dict, params: dict) -> None:
		# Execution algorithm parameters
		self.max_iterations = params['max_iterations']
		self.n_population = params['n_population']
		# Parent selection paramenters
		self.n_competitors = params['n_competitors']
		self.beta = 1
		# Crossover parameters
		self.crossover_prob = params['crossover_prob']
		# Mutation parameters
		self.mutation_prob = params['mutation_prob']
		# Problem parameters
		self.x_train, self.y_train = problem['x_train'], problem['y_train']
		self.x_test, self.y_test = problem['x_test'], problem['y_test']
		self.n_var = self.x_train.shape[1]
		self.objective_norm = np.array([1, 0.1])

	def initialize_population(self):
		self.population = []
		for _ in range(self.n_population):
			member = Individual(n_variables=self.n_var)
			member.initialize()
			member.accuracy, member.fitness, member.g_mean = self.evaluate(member, self.x_train, self.y_train)
			self.population.append(member)

	def weighted_sum(self, fitness):
		return np.sum(np.array(fitness, dtype=np.float32) * self.objective_norm * np.array((0.9, 0.1)))
	
	def compute_selection_prob(self) -> np.array:
		c = np.array([member.rank for member in self.population])
		# c = np.array([self.weighted_sum(member.fitness) for member in self.population])
		mean_cost = np.mean(c)
		if mean_cost != 0:
			c = c / mean_cost
		return np.exp(-self.beta * c)

	def evaluate(self, member, x, y, n_folds = 5):
		features_selected = [i for i, xi in enumerate(member.genome) if xi == 1]
		features_selected = torch.tensor(features_selected)
		if features_selected.shape[0] < 1:
			return None, np.array([math.inf, math.inf]), 0
		x_prima = x.index_select(1, features_selected)
		min_class = min(int(torch.sum(y)), y.shape[0] - int(torch.sum(y)))
		k = min(min_class, n_folds)
		loss = np.zeros(k)
		acc = np.zeros(k)
		g_mean = np.zeros(k)
		skf = StratifiedKFold(n_splits=k)
		for i, (train_index, test_index) in enumerate(skf.split(x_prima, y)):
			model = KNeighborsClassifier(n_neighbors=2)
			model.fit(x_prima[train_index], y[train_index].ravel())
			y_predict = torch.tensor(model.predict(x_prima[test_index]))
			y_real = y[test_index].squeeze(dim=1)
			loss[i] = torch_fitness_function(y_real, y_predict) 
			acc[i] = (y_real == torch.round(y_predict)).type(torch.float32).mean()
			g_mean[i] = geometric_mean(y_real, y_predict)
		return acc.mean(), [loss.mean(), features_selected.shape[0]], g_mean.mean()

	def final_evaluate(self, member, x_train, y_train, x_test, y_test):
		features_selected = [i for i, xi in enumerate(member.genome) if xi == 1]
		features_selected = torch.tensor(features_selected)
		if features_selected.shape[0] < 1:
			return None, np.array([math.inf, math.inf]), 0
		x_train_prima = x_train.index_select(1, features_selected)
		x_test_prima = x_test.index_select(1, features_selected)
		model = KNeighborsClassifier(n_neighbors=2)
		model.fit(x_train_prima, y_train.ravel())
		y_predict = torch.tensor(model.predict(x_test_prima))
		y_real = y_test.squeeze(dim=1)
		loss = torch_fitness_function(y_real, y_predict) 
		acc = (y_real == torch.round(y_predict)).type(torch.float32).mean()
		g_mean = geometric_mean(y_real, y_predict)
		return acc, [loss, features_selected.shape[0]], g_mean


	def select_parents(self) -> Tuple[Individual, Individual]:
		index = [i for i in range(self.n_population)]
		parent1 = tournament_selection(index, self.probs, self.n_competitors)
		index.remove(parent1)
		parent2 = tournament_selection(index, self.probs, self.n_competitors)
		return self.population[parent1], self.population[parent2]

	def crossover(self, parent1, parent2):
		# parent1.ws, parent2.ws = self.weighted_sum(parent1.fitness), self.weighted_sum(parent2.fitness)
		# if parent1.ws == parent2.ws:
		# 	child = Individual(self.n_var)
		# 	child.genome = np.zeros(self.n_var)
		# 	for i in range(self.n_var):
		# 		if parent1.genome[i] or parent2.genome[i]:
		# 			if np.random.uniform(0, 1) < 0.75:
		# 				child.genome[i] = 1
		# else:
		# 	parents = sorted([parent1, parent2], key=lambda x: x.ws)
		# 	child = parents[0].copy()
		# 	for i, x in enumerate(parents[1].genome):
		# 		if x and not child.genome[i]:
		# 			if np.random.uniform(0, 1) < 0.5:
		# 				child.genome[i] = x
		child1 = Individual(n_variables=self.n_var)
		child2 = Individual(n_variables=self.n_var)
		child1.genome, child2.genome = single_point_crossover(parent1.genome, parent2.genome)
		if np.sum(child1.genome) >= 1:
			return child1
		else:
			return child2

	def mutate(self, genome):
		r = np.random.uniform(0, 1, self.n_var)
		mutated_genome = [x if r[i] > self.mutation_prob else binary_mutation(x) for i, x in enumerate(genome)]
		return mutated_genome

	def next_generation(self):
		self.probs = self.compute_selection_prob()
		parent1, parent2 = self.select_parents()
		if np.random.uniform(0, 1) <= self.crossover_prob:
			child = self.crossover(parent1, parent2)
		else: 
			child = parent1.copy()
		child.genome = self.mutate(child.genome)
		child.accuracy, child.fitness, child.g_mean = self.evaluate(child, self.x_train, self.y_train)
		return child

	def reduce_population(self):
		front = create_fronts(self.population)
		if len(front[-1]) == 1:
			remove_index = 0
		else:
			front_fitness = np.array([list(p.fitness) for p in front[-1]])
			front_fitness *= self.objective_norm # Normalize objective
			remove_index, _ = choose_repeated_index(front_fitness)
			if remove_index is None:
				remove_index = choose_min_hv_contribution(front_fitness)
		self.population.remove(front[-1][remove_index])
		remove_genome_nds(self.population, front[-1][remove_index])

	def choose_solution(self, population: List[Individual], x, y) -> Individual:
		solution = copy.deepcopy(self.best_solution)
		for member in population:
			member.accuracy, member.fitness, member.g_mean = self.evaluate(member, x, y)
			if member.fitness[0] < solution.fitness[0]:
				solution = member.copy()
			elif member.fitness[0] == solution.fitness[0] and member.g_mean > solution.g_mean:
				solution = member.copy()
		return solution

	def run(self, seed: int = None, debug: bool = False):
		if seed is not None:
			set_seed(seed)
		self.initialize_population()
		_ = non_dominated_sorting_2(self.population)
		record = KRecord(self.max_iterations)
		record.update(self.population, iteration_num=0)
		for i in range(self.max_iterations):
			offspring = self.next_generation()
			add_genome_nds(self.population, offspring)
			self.population.append(offspring)
			self.reduce_population()
			if (i+1) % 180 == 0:
				record.update(self.population, iteration_num=i+1)
		n_objectives = len(self.population[0].fitness)
		self.best_solution = Individual()
		self.best_solution.fitness = np.ones(n_objectives) * math.inf
		self.best_solution = self.choose_solution(self.population, self.x_train, self.y_train)
		return record
