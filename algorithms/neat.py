#!/usr/bin/env python

"""
NeuroEvolution of Augmenting Topologies (NEAT)

Description:
"""

__author__ = "Daniel García Núñez"
__copyright__ = ""
__credits__ = ["Daniel García Núñez"]
__license__ = ""
__version__ = "0.1"
__maintainer__ = "Daniel García Núñez"
__email__ = "daniel_gn@comunidad.unam.mx"
__status__ = "Production"
__date__ = "30/09/2021"

"""
Required libraries
"""
import numpy as np
import math
import random
import torch
import logging
from typing import Tuple
from sklearn.model_selection import train_test_split

"""
Models
"""
from models.genotype import Genome, NodeGene
from models.species import Species
from utilities.evaluation import eval_model
from utilities.ga_utils import polynomial_mutation, tournament_selection, compute_parent_selection_prob
from utilities.ml_utils import get_batch
from utilities.record import Record, BestInidividualRecord

"""
Constants
"""
BATCH_PROP = 1.0
VALIDATION_PROP = 0.4


"""
NEAT
"""
def set_seed(seed = 0):
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)

class NEATArchive:

	def __init__(self, size=100) -> None:
		self.size = size
		self.population = []

	def add(self, population) -> None:
		population_copy = [member.copy() for member in population]
		self.population.extend(population_copy)
		self.population = sorted(self.population, key=lambda x: x.fitness)[:self.size]



class NEAT:

	# Constructor method
	def __init__(self, problem: dict, params: dict) -> None:
		# Execution algorithm parameters
		self.max_iterations = params['max_iterations']
		self.n_population = params['n_population']
		self.n_offspring = params['n_population']
		# Parent selection paramenters
		self.n_competitors = params['n_competitors']
		# Crossover parameters
		self.crossover_prob = params['crossover_prob']
		self.disable_node_prob = params['disable_node_prob']
		self.interspecies_mating_rate = params['interspecies_mating_rate']
		# Mutation parameters
		self.weight_mutation_prob = params['weight_mutation_prob']
		self.eta_m = params['pol_mutation_distr']
		self.weight_mutation_sustitution_prob = params['weight_mutation_sustitution_prob']
		self.add_node_prob = params['add_node_prob']
		self.add_connection_prob = params['add_connection_prob']
		# Speciation parameters
		self.compatibility_threshold = params['compatibility_threshold']
		self.stagnant_generations_threshold = params['stagnant_generations_threshold']
		self.champion_elitism_threshold = params['champion_elitism_threshold']
		self.c = params['compatibility_distance_coeff']
		# Artificial Neural Network parameters
		self.weight_min_value = params['initial_weight_limits'][0]
		self.weight_max_value = params['initial_weight_limits'][1]
		# self.activation_function = lambda x: 1 / (1 + np.exp(-4.9*x))
		self.activation = {}
		self.activation['hidden_activation_function'] = params['hidden_activation_function']
		self.activation['hidden_activation_coeff'] = params['hidden_activation_coeff']
		self.activation['output_activation_function'] = params['output_activation_function']
		self.activation['output_activation_coeff'] = params['output_activation_coeff']
		self.fitness_function = params['fitness_function']
		self.l2_parameter = params['regularization_parameter']
		# Problem parameters
		self.x_train, self.y_train = problem['x_train'], problem['y_train']
		self.x_val, self.y_val = problem['x_val'], problem['y_val']
		self.x_test, self.y_test = problem['x_test'], problem['y_test']
		
		
	def split_test_dataset(self, random_state: int=None):
		return train_test_split(self.x_train, self.y_train, test_size=VALIDATION_PROP, random_state=random_state, stratify=self.y_train)

	def initialize_population(self) -> None:
		"""
		NEAT starts with a population of Artificial Neural Networks with a minimal structure. It means that there
		are no hidden nodes, but only a fully connected layer between the input and output nodes. 
		Weights are chosen randomly.
		A global genome is created to keep track of all existing node and connections genes.
		"""
		# Global genome to keep track of all node and connection genes
		self.global_genome = Genome()
		self.global_genome.create_node_genes(self.x_train.shape[1], self.y_train.shape[1])
		self.global_genome.set_weight_limits(self.weight_min_value, self.weight_max_value)
		# Global variables for node id and innovation number
		self.node_id = len(self.global_genome.node_genes)
		self.innovation_number = self.global_genome.connect_all()
		# Create population from copies of the global genome 
		self.population = [self.global_genome.copy() for i in range(self.n_population)]
		for member in self.population:
			# Randomize weights from each member of the population
			member.randomize_weights()
			# Evaluate member fitness
			member.accuracy, member.fitness, member.g_mean = self.evaluate(member, self.x_train, self.y_train, True)
			# Keep track of the best solution found
			if member.fitness < self.best_solution.fitness:
				self.best_solution = member.copy(with_phenotype=True)

	def evaluate(self, genome: Genome, x: torch.Tensor, y: torch.Tensor, build_model: bool = True) -> Tuple[np.float32, np.float32, np.float32]:
		"""
		Encoding genomes to actual artifial neural network (ANN) to compute fitness.
		We should consider that NEAT tries to maximize the fitness, while most common methods
		for trainng ANNs try to minimize the loss. Therefore, the fitness function should be
		modified so it can work as maximization task.
		"""
		# model = ArtificialNeuralNetwork(genome, self.activation)
		# loss, acc = model.eval_model(x, y, self.fitness_function, self.l2_parameter)
		# fitness = 100 - loss
		# return acc, fitness

		if build_model:
			genome.compute_phenotype(self.activation)

		x_prima = x.index_select(1, genome.selected_features)
		connection_weights = [connection.weight for connection in genome.connection_genes if connection.enabled]
		mean_weight = np.mean(np.square(np.array(connection_weights))) if connection_weights else 0
		loss, acc, g_mean = eval_model(genome.phenotype, x_prima, y, self.fitness_function, self.l2_parameter, mean_weight)
		#fitness = 100 - loss
		return acc, loss.detach().numpy(), g_mean


	def get_offspring_distribution(self, remaining_offspring_space: int) -> list:
		"""
		The proportion of offspring for each species is defined by the sum of shared fitness of their members.
		Stagnant species for more generations than a defined threshold or species with no members should not be taking in count 
		for generating new offspring.
		"""
		# Compute offspring_distribution for each species
		total_sum_shared_fitness = np.sum(np.array([s.sum_shared_fitness for s in self.species]))
		offspring_distribution = [math.ceil(s.sum_shared_fitness * remaining_offspring_space / total_sum_shared_fitness) for s in self.species]
		# If the offspring distribution is not the same as the remainig space, randomly add or substract an element for a species
		while np.sum(offspring_distribution) > remaining_offspring_space:
			sp_options = [i for i, x in enumerate(offspring_distribution) if x > 1]
			if len(sp_options) == 0:
				sp_options = [i for i, x in enumerate(offspring_distribution)]
			i = random.choice(sp_options)
			offspring_distribution[i] -= 1
		return offspring_distribution

	def select_parents(self, species: list) -> Tuple[Genome, Genome]:
		parent1 = tournament_selection(species.member_index, self.probs, self.n_competitors)
		if random.uniform(0, 1) <= self.interspecies_mating_rate or species.get_size() == 1:
			other_species = random.choice(self.species)
			parent2 = tournament_selection(other_species.member_index, self.probs, self.n_competitors)
		else:
			parent2 = tournament_selection(species.member_index, self.probs, self.n_competitors)
		return self.population[parent1], self.population[parent2]

	def crossover(self, genome1: Genome, genome2: Genome) -> Tuple[Genome, bool]:
		"""
		When crossing over, the genes in both genomes with the same innovation numbers are lined up.
		These genes are called matching genes. Genes that do not match are either disjoint or exceess,
		depending on whether they occur within or outside the range of the parent's innovation numbers.
		Genes are randomly chosen from either parent at matching genes, whereas all excess and disjoint
		genes are always included from the more fit parent.
		If both parents have the same fitness, then the excess and disjoint genes from each parent are 
		inherit randomly.
		"""
		if genome1.fitness != genome2.fitness:
			parents = sorted([genome1, genome2], key=lambda x: x.fitness)
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
			return offspring, offspring.validate_network()
		else:
			return self.crossover_same_fitness(genome1, genome2)

	def crossover_same_fitness(self, genome1: Genome, genome2: Genome) -> Tuple[Genome, bool]:
		node_genes_id = []
		connection_genes = []
		# Find matching genes
		for connection_a in genome1.connection_genes:
			connection_b = genome2.get_connection_gene(connection_a.innovation_number)
			if connection_b:
				connection_genes.append(connection_a.copy())
				node_genes_id.extend([connection_a.input_node, connection_b.output_node])
				if not connection_a.enabled or not connection_b.enabled:
					if random.uniform(0, 1) < self.disable_node_prob:
						connection_genes[-1].enabled = False
					else:
						connection_genes[-1].enabled = True
		node_genes_id = list(set(node_genes_id))
		node_genes = []
		for id in node_genes_id:
			node_gene = genome1.get_node_gene(id)
			node_genes.append(node_gene.copy())
		offspring = Genome(node_genes, connection_genes)
		offspring.set_weight_limits(self.weight_min_value, self.weight_max_value)
		# Randomly add excess and disjoint genes from parent one
		for parent_connection in genome1.connection_genes:
			connection = genome2.get_connection_gene(parent_connection.innovation_number)
			if not connection and random.uniform(0, 1) < 0.8:
				if parent_connection.input_node in node_genes_id and parent_connection.output_node in node_genes_id:
					possible = offspring.get_possible_connections(parent_connection.input_node)
					if parent_connection.output_node in possible:
						offspring.connection_genes.append(parent_connection.copy())
				else:
					offspring.connection_genes.append(parent_connection.copy())
					if parent_connection.input_node not in node_genes_id:
						node_genes_id.append(parent_connection.input_node)
						node = genome1.get_node_gene(parent_connection.input_node)
						offspring.node_genes.append(node.copy())
					if parent_connection.output_node not in node_genes_id:
						node_genes_id.append(parent_connection.output_node)
						node = genome1.get_node_gene(parent_connection.output_node)
						offspring.node_genes.append(node.copy())
				if not offspring.connection_genes[-1].enabled:
					if random.uniform(0, 1) > self.disable_node_prob:
						offspring.connection_genes[-1].enabled = True
		# Randomly add excess and disjoint genes from parent two
		for parent_connection in genome2.connection_genes:
			connection = genome1.get_connection_gene(parent_connection.innovation_number)
			if not connection and random.uniform(0, 1) < 0.8:
				if parent_connection.input_node in node_genes_id and parent_connection.output_node in node_genes_id:
					possible = offspring.get_possible_connections(parent_connection.input_node)
					if parent_connection.output_node in possible:
						offspring.connection_genes.append(parent_connection.copy())
				else:
					offspring.connection_genes.append(parent_connection.copy())
					if parent_connection.input_node not in node_genes_id:
						node_genes_id.append(parent_connection.input_node)
						node = genome2.get_node_gene(parent_connection.input_node)
						offspring.node_genes.append(node.copy())
					if parent_connection.output_node not in node_genes_id:
						node_genes_id.append(parent_connection.output_node)
						node = genome2.get_node_gene(parent_connection.output_node)
						offspring.node_genes.append(node.copy())
				if not offspring.connection_genes[-1].enabled:
					if random.uniform(0, 1) > self.disable_node_prob:
						offspring.connection_genes[-1].enabled = True
		# if not offspring.connection_genes:
		# 	offspring.valid = False
		# 	return offspring, False
		return offspring, offspring.validate_network()

	def mutate_weights(self, genome: Genome) -> None:
		for connection in genome.connection_genes:
			if random.uniform(0, 1) <= self.weight_mutation_prob:
				# Polynomial mutation
				if random.uniform(0, 1) > self.weight_mutation_sustitution_prob:
					connection.weight = polynomial_mutation(connection.weight, self.eta_m)
				# Substitute weight for random number
				else:
					connection.weight = random.uniform(self.weight_min_value, self.weight_max_value) 
	
	def add_node_mutation(self, genome: Genome) -> None:
		old_connection = random.choice(genome.connection_genes)
		# Check if this innovation already happend during this generation
		if old_connection.innovation_number not in self.generation_add_node:
			new_node = NodeGene(self.node_id)
			self.innovation_number = genome.add_node(new_node, old_connection.innovation_number, self.innovation_number)
			self.innovation_number = self.global_genome.add_node(new_node, old_connection.innovation_number, self.innovation_number-2)
			self.generation_add_node.append(old_connection.innovation_number)
			self.generation_new_connections[str(old_connection.innovation_number)] = {
				'node_id' : self.node_id,
				'innovation_number' : self.innovation_number - 2
			}
			self.node_id += 1
		else:
			node_id = self.generation_new_connections[str(old_connection.innovation_number)]['node_id']
			innovation_number = self.generation_new_connections[str(old_connection.innovation_number)]['innovation_number']
			new_node = NodeGene(node_id)
			genome.add_node(new_node, old_connection.innovation_number, innovation_number)

	def add_connection_mutation(self, genome: Genome) -> None:
		# Search possible input nodes
		possible_input_nodes = []
		for node in genome.node_genes:
			if node.node_type != 'output':
				possible_input_nodes.append(node.id)
		# Loop for possible input nodes until a connection has been added
		while possible_input_nodes:
			# Choose an input node
			input_node_id = random.choice(possible_input_nodes)
			# Seach possible output nodes
			possible_output_nodes = genome.get_possible_connections(input_node_id)
			if possible_output_nodes:
				# Check if this connection already exists in global genome
				output_node_id = random.choice(possible_output_nodes)
				innovation_number = self.global_genome.get_innovation_number(input_node_id, output_node_id)
				if innovation_number:
					genome.add_connection(input_node_id, output_node_id, innovation_number)
				else:
					self.innovation_number = genome.add_connection(input_node_id, output_node_id, self.innovation_number)
					self.innovation_number = self.global_genome.add_connection(input_node_id, output_node_id, self.innovation_number-1)
				break
			else:
				# If no possible output node was found for the chosen input node, this input node is actually not possible
				possible_input_nodes.remove(input_node_id)

	def mutate(self, genome: Genome) -> None:
		"""
		Mutation in NEAT can change both connection weights and network structures.
		Connection weights mutate as in any NE system, with each connection either perturbated or not in each generation.
		Structural mutation occurs in two ways: add node and add connection operators.
		"""
		# Weight mutation
		self.mutate_weights(genome)

		# Structural mutation	
		# Add node mutation
		if random.uniform(0, 1) <= self.add_node_prob:
			self.add_node_mutation(genome)

		# Add connection mutation 
		if random.uniform(0, 1) <= self.add_connection_prob:
			self.add_connection_mutation(genome)

	def elitism(self):
		# Offspring population initalize empty
		offspring = []
		remaining_offspring_space = self.n_offspring
		# Add champion of large population species to next generation
		for s in self.species:
			if s.get_size() > self.champion_elitism_threshold:
				s.champion.accuracy, s.champion.fitness, s.champion.g_mean = self.evaluate(s.champion, self.x_batch, self.y_batch, False)
				offspring.append(s.champion.copy(with_phenotype=True))
				remaining_offspring_space -= 1
		return offspring, remaining_offspring_space

	def next_generation(self) -> Genome:
		"""
		Get next generation population. First add champion of each species which has a population 
		larger than a defined threshold. Then, compute the distribution of offspring by each species
		according to the sum of shared fitness of their members. Then get offspring by crossover and
		mutation operators.
		"""
		# Current generation innovation
		self.generation_add_node = []
		self.generation_new_connections = {}
		self.probs = compute_parent_selection_prob(self.population)
		# Add champion of large population species to next generation
		offspring, remaining_offspring_space = self.elitism()
		self.k_offspring = remaining_offspring_space
		# Compute offspring distribution by species
		offspring_distribution = self.get_offspring_distribution(remaining_offspring_space)
		# Get offspring for each species
		for i, s in enumerate(self.species):
			for _ in range(offspring_distribution[i]):
				if random.uniform(0, 1) < self.crossover_prob:
					cross = True
					parent1, parent2 = self.select_parents(s)
					attempts = 0
					while attempts < 5:
						attempts += 1
						child, succeed = self.crossover(parent1, parent2)
						if succeed:
							break
					if not succeed:
						self.n_invalid_nets +=1
				else:
					cross = False
					parent, _ = self.select_parents(s)
					child = parent.copy()
				if self.debug:
					temp_child = child.copy()
				self.mutate(child)
				child.accuracy, child.fitness, child.g_mean = self.evaluate(child, self.x_batch, self.y_batch, True)
				offspring.append(child)
				if child.fitness < self.best_solution.fitness:
					self.best_solution = child.copy(with_phenotype=True)
				if self.debug and np.isnan(child.fitness):
					if cross:
						logging.info('Parents')
						self.describe(parent1)
						self.describe(parent2)
					logging.info('Temp Child')
					self.describe(temp_child)	
					logging.info('Child')
					self.describe(child)

		return offspring

	def compute_shared_fitness(self) -> None:
		for s in self.species:
			species_size = s.get_size()
			for index in s.member_index:
				# Compute shared fitness for each member
				self.population[index].shared_fitness = self.population[index].fitness / species_size
				# Compute sum of shared fitness for each species 
				s.sum_shared_fitness += self.population[index].shared_fitness
			s.sum_shared_fitness = 1 / s.sum_shared_fitness

	def speciation(self) -> None:
		# Remove past population index from species
		if self.species:
			for s in self.species:
				s.member_index = []
				s.stagnant_generations += 1
		# Select a species for each member in population
		for i, member in enumerate(self.population):
			if member.fitness == math.inf:
				continue
			species_found = False
			for s in self.species:
				if member.compatibility_distance(s.representant, self.c) <= self.compatibility_threshold:
					s.add_member(i, member)
					species_found = True
					break
			# If member does not match in an existing create a new one 
			if not species_found:
				new_species = Species()
				new_species.member_index.append(i)
				new_species.representant = member.copy()
				new_species.champion = member.copy(with_phenotype=True)
				self.species.append(new_species)
		# Remove species with stagnant generations higher than threshold or species with no members
		self.species = [s for s in self.species if (s.stagnant_generations < self.stagnant_generations_threshold and s.get_size() > 0)]
		# Choose each species representant
		for s in self.species:
			if s.member_index:
				s.choose_representant(self.population)
		# Compute shared fitness for each member and the sum of shared fitness for each species
		self.compute_shared_fitness()

	def describe(self, genome: Genome) -> None:
		logging.info('Node genes:')
		for node in genome.node_genes:
			logging.info(f'Node {node.id}, type {node.node_type}, layer {node.layer}')
		logging.info('Connection genes:')
		for connection in genome.connection_genes:
			logging.info(f'Connection {connection.innovation_number}: Input node id = {connection.input_node}, Output node id = {connection.output_node}, Weight = {connection.weight}, Enabled = {connection.enabled}')
		logging.info(f'Fitness: {genome.fitness}')

	def choose_solution(self, population) -> None:
		solution = self.best_solution.copy(True)
		for member in population:
			member.accuracy, member.fitness, member.g_mean = self.evaluate(member, self.x_val, self.y_val, True)
			if member.fitness < solution.fitness:
				solution = member.copy(True)
			elif member.fitness == solution.fitness and member.g_mean > solution.g_mean:
				solution = member.copy(True)
		return solution

	# Run algorithm
	def run(self, seed: int = None, debug: bool = False) -> None:
		self.debug = debug
		if debug:
			logging.basicConfig(filename="test_mean.log", level=logging.INFO)
		if seed is not None:
			set_seed(seed)
		# Variable to store best solution
		self.best_solution = Genome()
		self.best_solution.fitness = math.inf
		# Initalize population
		self.initialize_population()
		# List to store species, initalized empty
		self.species = []
		# Initialize Archive
		self.archive = NEATArchive(self.n_population)
		self.archive.add(self.population)
		self.n_invalid_nets = 0
		# Create arrays to store model fitness and accuracy on each iteration for training and test datasets
		self.record = Record(self.max_iterations)
		self.record.update(self.population, iteration_num=0)
		self.record_archive = Record(self.max_iterations)
		self.record_archive.update(self.archive.population)
		self.best_solution_record = BestInidividualRecord(self.max_iterations)
		self.best_solution_record.update(self.best_solution, iteration_num=0)
		# Evolve population
		for i in range(self.max_iterations):
			if debug:
				logging.info(f'Iteration: {i}')
			# Split population into species
			self.speciation()
			# Stop condition
			if len(self.species) < 1:
				break
			# Generate training batch
			self.x_batch, self.y_batch = get_batch(self.x_train, self.y_train, BATCH_PROP)
			self.best_solution.accuracy, self.best_solution.fitness, self.best_solution.g_mean = self.evaluate(self.best_solution, self.x_batch, self.y_batch, False)
			# Compute new generation by crossover and mutation operators
			self.population = self.next_generation()
			self.archive.add(self.population[:-self.k_offspring])
			# Evaluate best solution on full dataset
			# self.best_solution.accuracy, self.best_solution.fitness, self.best_solution.g_mean = self.evaluate(self.best_solution, self.x_train, self.y_train, False)
			# Store history of fitness and accuracy from best solution in both datasets
			self.record.update(self.population, iteration_num=i+1, n_invalid_nets=self.n_invalid_nets)
			self.record_archive.update(self.archive.population)
			self.best_solution_record.update(self.best_solution, iteration_num=i+1)
			# Display progress
			# if i % 5 == 0:
			# 	n_input_nodes, n_hidden_nodes, n_output_nodes = self.best_solution.count_nodes()
			# 	print(f'It: {i}: Train fit = {self.training_fitness[i+1][0]:.4f}, Acc = {self.training_accuracy[i+1][0]:.4f}, Gmean = {self.training_gmean[i+1][0]:.4f}; Nodes = [{n_input_nodes}, {n_hidden_nodes}, {n_output_nodes}]; Species = {len(self.species)}')
		self.best_solution_val = self.choose_solution(self.population)
		self.best_solution_archive = self.choose_solution(self.archive.population)
		
