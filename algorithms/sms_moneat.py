import numpy as np
from sklearn.feature_selection import GenericUnivariateSelect
import torch
import random
import math
import copy

from typing import Tuple
from typing import List

from models.genotype import MultiObjectiveGenome as Genome
from algorithms.n3o import N3O
from algorithms.neat import set_seed
from utilities.evaluation import eval_model
from utilities.ga_utils import tournament_selection
from utilities.moea_utils import choose_min_hv_contribution, non_dominated_sorting_2
from utilities.moea_utils import add_genome_nds, remove_genome_nds, create_fronts
from utilities.data_utils import choose_repeated_index
from utilities.ml_utils import get_batch

from algorithms.archive import SpeciesArchive
from sklearn.model_selection import train_test_split

BATCH_PROP = 1.0

class SMS_MONEAT(N3O):

	def __init__(self, problem: dict, params: dict) -> None:
		super().__init__(problem, params)
		self.beta = 1
		self.objective_norm = np.array([0, 0.1])

	def initialize_population(self) -> None:
		"""
		For each network in the intial population, randomly select an input and an output and add a link connecting them.
		"""
		# Global genome to keep track of all node and connection genes
		self.global_genome = Genome()
		self.global_genome.create_node_genes(self.x_train.shape[1], self.y_train.shape[1])
		self.global_genome.set_weight_limits(self.weight_min_value, self.weight_max_value)
		# Global variables for node id and innovation number
		self.node_id = len(self.global_genome.node_genes)
		self.innovation_number = 0
		# Create population from copies of the global genome 
		self.population = []
		for i in range(self.n_population):
			# Create a random connection
			connection, self.innovation_number = self.global_genome.connect_one_input(self.innovation_number)
			# Create an individual with this connection
			input_node = self.global_genome.get_node_gene(connection.input_node)
			output_node = self.global_genome.get_node_gene(connection.output_node)
			node_genes = [input_node, output_node]
			connection_genes = [connection.copy()]
			member = Genome(node_genes, connection_genes)	
			member.set_weight_limits(self.weight_min_value, self.weight_max_value)
			# Evaluate member fitness
			member.accuracy, member.fitness, member.g_mean = self.evaluate(member, self.x_train, self.y_train, True)
			# Add member to population
			self.population.append(member)

	def evaluate(self, genome: Genome, x: torch.Tensor, y: torch.Tensor, build_model: bool = True) -> Tuple[np.float32, np.array, np.float32]:
		if build_model:
			genome.compute_phenotype(self.activation)
		if genome.selected_features.shape[0] == 0:
			return None, np.array([math.inf, math.inf]), 0
		x_prima = x.index_select(1, genome.selected_features)
		loss, acc, gmean = eval_model(genome.phenotype, x_prima, y, self.fitness_function, self.l2_parameter, genome.mean_square_weights)
		fitness = np.array([loss, genome.selected_features.shape[0]])
		# fitness = np.array([false_negative, false_positive, genome.selected_features.shape[0]])
		return acc, fitness, gmean

	def evaluate_population(self, x: torch.Tensor, y: torch.Tensor) -> None:
		for member in self.population:
			member.accuracy, member.fitness, member.g_mean = self.evaluate(member, x, y, False)
		_ = create_fronts(self.population)

	def compute_selection_prob(self) -> np.array:
		c = np.array([member.rank for member in self.population], dtype="float32") 
		mean_cost = np.mean(c)
		if mean_cost != 0:
			c /= mean_cost
		return np.exp(-self.beta * c)

	def select_parents(self) -> Tuple[Genome, Genome]:
		index = [i for i in range(self.n_population)]
		parent1 = tournament_selection(index, self.probs, self.n_competitors)
		index.remove(parent1)
		parent2 = tournament_selection(index, self.probs, self.n_competitors)
		return self.population[parent1], self.population[parent2]

	def crossover(self, genome1: Genome, genome2: Genome) -> Tuple[Genome, bool]:
		"""
		Works similary as the NEAT crossover operator, but if the parent with lower fitness has an input that the other
		parent does not, and the input is connected to a node present in the other parent, there is a 50% chance of the
		offspring inheriting that input.
		"""
		if genome1.rank != genome2.rank:
			parents = sorted([genome1, genome2], key=lambda x: x.rank)
			offspring = parents[0].copy()
			for connection_a in offspring.connection_genes:
				connection_b = parents[1].get_connection_gene(connection_a.innovation_number)
				if connection_b:
					if random.uniform(0, 1) < 0.5:
						connection_a.weight = connection_b.weight
					if not connection_a.enabled or not connection_b.enabled:
						if random.uniform(0, 1) < self.disable_node_prob:
							connection_a.enabled = False
						else:
							connection_a.enabled = True
			# Add inputs from parent with lowest fitness to offspring if this input has a connection to an existing node
			# in offspring
			current_input_nodes = []
			for node in offspring.node_genes:
				if node.node_type == 'input':
					current_input_nodes.append(node.id)
			current_input_nodes = list(set(current_input_nodes))

			for connection_b in parents[1].connection_genes:
				input_node = parents[1].get_node_gene(connection_b.input_node)
				if input_node.node_type == 'input' and connection_b.input_node not in current_input_nodes:
					if offspring.get_node_gene(connection_b.output_node) is not None:
						if random.uniform(0, 1) < 0.5:
							if offspring.get_node_gene(connection_b.input_node) is None:
								offspring.node_genes.append(input_node.copy())
							offspring.connection_genes.append(connection_b.copy())
			return offspring, offspring.validate_network()
		else:
			offspring = Genome()
			offspring.set_weight_limits(self.weight_min_value, self.weight_max_value)
			temp, validation_flag = self.crossover_same_fitness(genome1, genome2)
			offspring.node_genes = list(temp.node_genes)
			offspring.connection_genes = list(temp.connection_genes)
			return offspring, validation_flag

	def next_generation(self) -> Genome:
		self.generation_add_node = []
		self.generation_new_connections = {}
		self.probs = self.compute_selection_prob()
		if random.uniform(0, 1) < self.crossover_prob:
			parent1, parent2 = self.select_parents()
			attempts = 0
			while attempts < 5:
				attempts += 1
				child, succeeded = self.crossover(parent1, parent2)
				if succeeded:
					self.mutate(child)
					break
		else:
			parent, _ = self.select_parents()
			child = parent.copy()
			self.mutate(child)
		child.accuracy, child.fitness, child.g_mean = self.evaluate(child, self.x_train, self.y_train, True)
		return child

	def reduce_population(self) -> None:
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

	def choose_solution(self, population: List[Genome], x, y) -> Genome:
		solution = copy.deepcopy(self.best_solution)
		for member in population:
			member.accuracy, member.fitness, member.g_mean = self.evaluate(member, x, y, True)
			if member.fitness[0] < solution.fitness[0]:
				solution = member.copy(True)
			elif member.fitness[0] == solution.fitness[0] and member.g_mean > solution.g_mean:
				solution = member.copy(True)
		return solution

	def run(self, seed: int = None, debug: bool = False) -> None:
		if seed is not None:
			set_seed(seed)
		self.initialize_population()
		_ = non_dominated_sorting_2(self.population)
		self.archive = SpeciesArchive(self.n_population, self.objective_norm, self.population)
		for i in range(self.max_iterations):
			# Get batch
			# if i % 100 == 0 and BATCH_PROP < 1.0:
			# 	x_batch, y_batch = get_batch(self.x_train, self.y_train, BATCH_PROP, random_state=i)
			# 	self.evaluate_population(x_batch, y_batch)
			# Get offspring
			offspring = self.next_generation()
			# if BATCH_PROP < 1.0:
			# 	offspring.accuracy, offspring.fitness, offspring.g_mean = self.evaluate(offspring, x_batch, y_batch, False)
			# Update Non-Dominated Sorting variables from population and offspring
			add_genome_nds(self.population, offspring)
			# Add Offspring to population
			self.population.append(offspring)
			# Reduce population
			self.reduce_population()
			# Add to archive
			self.archive.add(offspring)
			# Display run info
			# if i % 5000 == 0:
			# 	if BATCH_PROP < 1.0:
			# 		self.evaluate_population(self.x_train, self.y_train)
			# 	population_fitness = np.array([member.fitness for member in self.population]).mean(axis=0)
			# 	population_gmean = np.array([member.g_mean for member in self.population]).mean(axis=0)
			# 	print(f'Iteration {i}: population fitness = {population_fitness}, g mean = {population_gmean:.4f}, species = {self.archive.species_count()}')
		n_objectives = len(self.population[0].fitness)
		self.best_solution = Genome()
		self.best_solution.fitness = np.ones(n_objectives) * math.inf
		self.best_solution = self.choose_solution(self.population, self.x_train, self.y_train)
		self.best_solution_val = self.choose_solution(sorted(self.population, key=lambda x:x.fitness[0]), self.x_val, self.y_val)
		self.best_solution_archive = self.choose_solution(sorted(self.archive.get_full_population(), key=lambda x:x.fitness[0]), self.x_val, self.y_val)
		
