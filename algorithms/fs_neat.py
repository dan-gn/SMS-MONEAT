#!/usr/bin/env python

"""
Feature Selective NeuroEvolution of Augmented Topologies (FS-NEAT)

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
import torch

from typing import Tuple

"""
Class
"""
from algorithms.neat import NEAT
from models.genotype import Genome
from utilities.evaluation import eval_model

"""
FS-NEAT
"""
class FS_NEAT(NEAT):

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
		self.population = [self.global_genome.copy() for i in range(self.n_population)]
		for member in self.population:
			# Create a random connection
			connection, self.innovation_number = self.global_genome.connect_one_input(self.innovation_number)
			member.connection_genes.append(connection.copy())
			# Evaluate member fitness
			member.accuracy, member.fitness, member.g_mean = self.evaluate(member, self.x_train, self.y_train, True)
			# Keep track of the best solution found
			if member.fitness > self.best_solution.fitness:
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
		# if fitness < 0:
		# 	print(f'Fitness negativo: {fitness}')
		# return acc, fitness

		if build_model:
			genome.compute_phenotype(self.activation)
		if genome.selected_features.shape[0] == 0:
			return None, 0, 0
		x_prima = x.index_select(1, genome.selected_features)
		connection_weights = [connection.weight for connection in genome.connection_genes if connection.enabled]
		mean_weight = np.mean(np.square(np.array(connection_weights))) if connection_weights else 0
		loss, acc, gmean = eval_model(genome.phenotype, x_prima, y, self.fitness_function, self.l2_parameter, mean_weight)
		fitness = 100 - loss
		if fitness < 0:
			print(f'Fitness negativo: {fitness}')
		return acc, fitness.detach().numpy(), gmean
