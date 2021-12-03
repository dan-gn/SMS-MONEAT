import numpy as np
import random
import math

from algorithms.n3o import N3O
from algorithms.neat import set_seed
from models.genotype import Genome
from models.ann_pytorch import eval_model
from utilities.ga_utils import tournament_selection
from utilities.moea_utils import non_dominated_sorting, get_hv_contribution
from utilities.moea_utils import check_repeated_rows, choose_repeated_index


class SMS_NEAT(N3O):

	def __init__(self, problem, params):
		super().__init__(problem, params)
		self.beta = 1

	def initialize_population(self):
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

	def evaluate(self, genome, x, y, build_model=True):
		if build_model:
			genome.compute_phenotype(self.activation)
		if genome.selected_features.shape[0] == 0:
			return None, np.array([math.inf, math.inf]), 0
		x_prima = x.index_select(1, genome.selected_features)
		connection_weights = [connection.weight for connection in genome.connection_genes if connection.enabled]
		mean_weight = np.mean(np.square(np.array(connection_weights))) if connection_weights else 0
		loss, acc, gmean = eval_model(genome.phenotype, x_prima, y, self.fitness_function, self.l2_parameter, mean_weight)
		fitness = np.array([loss, genome.selected_features.shape[0]])
		return acc, fitness, gmean

	def compute_selection_prob(self):
		c = np.array([member.rank+1 for member in self.population], dtype="float32") 
		mean_cost = np.mean(c)
		if mean_cost != 0:
			c /= mean_cost
		return np.exp(-self.beta * c)

	def select_parents(self):
		index = [i for i in range(self.n_population)]
		parent1 = tournament_selection(index, self.probs, self.n_competitors)
		index.remove(parent1)
		parent2 = tournament_selection(index, self.probs, self.n_competitors)
		return self.population[parent1], self.population[parent2]

	def crossover(self, genome1, genome2):
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
			return self.crossover_same_fitness(genome1, genome2)

	def next_generation(self):
		self.generation_add_node = []
		self.generation_new_connections = {}
		self.probs = self.compute_selection_prob()
		if random.uniform(0, 1) < self.crossover_prob:
			parent1, parent2 = self.select_parents()
			while True:
				child, succeeded = self.crossover(parent1, parent2)
				if succeeded:
					break
		else:
			parent, _ = self.select_parents()
			child = parent.copy()
		self.mutate(child)
		child.accuracy, child.fitness, child.g_mean = self.evaluate(child, self.x_train, self.y_train, True)
		return child

	def reduce_population(self):
		front = non_dominated_sorting(self.population)
		if len(front[-1]) == 1:
			self.population.remove(front[-1][0])
		else:
			f = np.array([list(p.fitness) for p in front[-1]])
			if check_repeated_rows(f):
				r = choose_repeated_index(f)
			else:
				r = np.argmin(get_hv_contribution(f))
			self.population.remove(front[-1][r])

	def choose_solution(self):
		self.population = sorted(self.population, key=lambda x: x.fitness[0])
		self.best_solution = self.population[0].copy(with_phenotype=True)
		self.best_solution_test = Genome()
		self.best_solution_test.fitness = math.inf
		self.best_solution_test.g_mean = -math.inf
		for member in self.population:
			_, fitness, g_mean = self.evaluate(member, self.x_test, self.y_test, True)
			if (g_mean > self.best_solution_test.g_mean) or (g_mean == self.best_solution_test.g_mean and fitness[1] < self.best_solution_test.fitness[1]):
				self.best_solution_test = member.copy(with_phenotype=True)

	def run(self, seed=None, debug=False):
		set_seed()
		self.initialize_population()
		_ = non_dominated_sorting(self.population)
		for i in range(self.max_iterations):
			offspring = self.next_generation()
			self.population.append(offspring)
			self.reduce_population()
			population_fitness = np.array([member.fitness for member in self.population])
			mean_fitness = np.mean(population_fitness, axis=0)
			if i % 500 == 0:
				print(f'Iteration {i}: population fitness {mean_fitness}')
		self.choose_solution()
		
