
"""
Required libraries
"""
import numpy as np
import random
import logging
from typing import Tuple

"""
Class
"""
from algorithms.fs_neat import FS_NEAT
from models.genotype import Genome
from utilities.ml_utils import softmax

"""
N3O
"""

class N3O(FS_NEAT):

	def __init__(self, problem: dict, params: dict) -> None:
		super().__init__(problem, params)
		pvalue = problem['kw_htest_pvalue']
		self.guided_input_probs = self.get_guided_input_prob(pvalue)
		self.elitism_prop = params['elitism_prop']
		self.add_input_prob = params['add_input_prob']
		self.swap_input_prob = params['swap_input_prob']
		self.input_ids = [i for i in range(self.x_train.shape[1])]
		self.output_ids = [i for i in range(self.x_train.shape[1], self.x_train.shape[1] + self.y_train.shape[1])]

	def initialize_population(self) -> None:
		"""
		For each network in the intial population, randomly select an input and an output and add a link connecting them.
		"""
		# Global genome to keep track of all node and connection genes
		self.global_genome = Genome()
		self.global_genome.create_node_genes(self.x_train.shape[1], self.y_train.shape[1])
		self.global_genome.set_weight_limits(self.weight_min_value, self.weight_max_value)
		if self.debug:
			logging.info(f'Global genome')
			for connection in self.global_genome.connection_genes:
				logging.info(f'{connection.innovation_number}, {connection.input_node}, {connection.output_node}')
			
		# Global variables for node id and innovation number
		self.node_id = len(self.global_genome.node_genes)
		self.innovation_number = 0
		# Create population from copies of the global genome 
		self.population = []
		if self.debug:
			logging.info('Initialize population')
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
			# Keep track of the best solution found
			if member.fitness > self.best_solution.fitness:
				self.best_solution = member.copy(with_phenotype=True)
			if self.debug:
				self.describe(member)

	def get_guided_input_prob(self, pvalue : np.array) -> np.array:
		return softmax(-np.log10(pvalue))

	def crossover(self, genome1: Genome, genome2: Genome) -> Tuple[Genome, bool]:
		"""
		Works similary as the NEAT crossover operator, but if the parent with lower fitness has an input that the other
		parent does not, and the input is connected to a node present in the other parent, there is a 50% chance of the
		offspring inheriting that input.
		"""
		if genome1.fitness != genome2.fitness:
			parents = sorted([genome1, genome2], key=lambda x: -x.fitness)
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
			return self.crossover_same_fitness(genome1, genome2)

	def swap_input_mutation(self, genome: Genome) -> None:
		"""
		Randomly swaps one of the network inputs by another input not present in the ANN.
		"""
		# Create a list of current input nodes and new possiblle input nodes
		possible_new_input_ids = list(self.input_ids)
		current_input_ids = []
		for connection in genome.connection_genes:
			input_node = genome.get_node_gene(connection.input_node)
			if input_node.node_type == 'input':
				current_input_ids.append(connection.input_node)
				if connection.input_node in possible_new_input_ids:
					possible_new_input_ids.remove(connection.input_node)
		current_input_ids = list(set(current_input_ids))
		# If any possible new input node
		if possible_new_input_ids:
			# Choose a current input to remove and a new input to add
			old_input = random.choice(current_input_ids)
			new_input = random.choice(possible_new_input_ids)
			# Swap input nodes
			remove_connection = []
			for connection in genome.connection_genes:
				if connection.input_node == old_input:
					innovation_number = self.global_genome.get_innovation_number(new_input, connection.output_node)
					if innovation_number:
						connection.input_node = new_input
						connection.innovation_number = innovation_number
					else:
						remove_connection.append(connection.innovation_number)
						self.innovation_number = genome.add_connection(new_input, connection.output_node, self.innovation_number, connection.weight) - 1
						self.innovation_number = self.global_genome.add_connection(new_input, connection.output_node, self.innovation_number, connection.weight)
			new_node = self.global_genome.get_node_gene(new_input)
			genome.node_genes.append(new_node)
			genome.node_genes = sorted(genome.node_genes, key = lambda x: x.id)
			# Remove old node and its connections
			for i_num in remove_connection:
				genome.remove_connection(i_num)
			genome.remove_node(old_input)

	def guided_add_input_mutation(self, genome: Genome) -> None:
		possible_new_input_ids = list(self.input_ids)
		for connection in genome.connection_genes:
			if connection.input_node in possible_new_input_ids:
				possible_new_input_ids.remove(connection.input_node)
		if possible_new_input_ids:
			input_probs = self.guided_input_probs[possible_new_input_ids]
			input_probs = np.cumsum(input_probs)
			r = random.uniform(0, 1) * input_probs[-1]
			input_index = next(index for index, prob in enumerate(input_probs) if prob >= r)
			choosen_input = possible_new_input_ids[input_index]
			choosen_output = random.choice(self.output_ids)
			if genome.get_node_gene(choosen_input) is None:
				new_node = self.global_genome.get_node_gene(choosen_input)
				genome.node_genes.append(new_node)
				genome.node_genes = sorted(genome.node_genes, key = lambda x: x.id)
			innovation_number = self.global_genome.get_innovation_number(choosen_input, choosen_output)
			if innovation_number:
				genome.add_connection(choosen_input, choosen_output, innovation_number)
			else:
				self.innovation_number = genome.add_connection(choosen_input, choosen_output, self.innovation_number) - 1
				self.innovation_number = self.global_genome.add_connection(choosen_input, choosen_output, self.innovation_number) 

	def mutate(self, genome: Genome) -> None:
		super().mutate(genome)
		# # Add input mutation
		if random.uniform(0, 1) <= self.add_input_prob:
			self.guided_add_input_mutation(genome)

		# # Add swap input mutation
		if random.uniform(0, 1) <= self.swap_input_prob:
			self.swap_input_mutation(genome)

	def elitism(self) -> Tuple[list, int]:
		"""
		Preserves the best individuals from each generation.
		"""
		offspring = []
		sorted_pop = sorted(self.population, key=lambda x: -x.fitness)
		k = int(self.n_population * self.elitism_prop)
		remaining_offspring_space = self.n_population - k
		for i in range(k):
			sorted_pop[i].accuracy, sorted_pop[i].fitness, sorted_pop[i].g_mean = self.evaluate(sorted_pop[i], self.x_batch, self.y_batch, False)
			offspring.append(sorted_pop[i].copy(with_phenotype=True))
		return offspring, remaining_offspring_space