import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import numpy as np
from statistics import mean
import torch
import torch.nn as nn

from models.genotype import Genome, NodeGene, ConnectionGene
from models.ann_pytorch import Ann_PyTorch, eval_model
from utilities.fitness_functions import torch_fitness_function, fitness_function
from utilities.activation_functions import Gaussian, gaussian

class ArtificialNeuralNetwork:

	def __init__(self, genome, activation):
		self.nodes = genome.node_genes
		self.connections = genome.connection_genes
		self.hidden_activation_function = activation['hidden_activation_function']
		self.hidden_activation_coeff = activation['hidden_activation_coeff']
		self.output_activation_function = activation['output_activation_function']
		self.output_activation_coeff = activation['output_activation_coeff']
		self.structure_recognition()

	def structure_recognition(self):
		self.input_nodes = []
		self.hidden_nodes = []
		self.output_nodes = []
		for node in self.nodes:
			if node.node_type == 'input':
				self.input_nodes.append(node)
			elif node.node_type == 'hidden':
				self.hidden_nodes.append(node)
			else:
				self.output_nodes.append(node)

	def restart(self):
		for node in self.nodes:
			node.value = None

	def set_input(self, input):
		for node in self.input_nodes:
			node.value = input[node.id]

	def get_node(self, id):
		for node in self.nodes:
			if node.id == id:
				return node
		return None

	def compute_node(self, id):
		node = self.get_node(id)

		if node.value is None:
			value = []
			for connection in self.connections:
				if connection.enabled and connection.output_node == node.id:
					value.append(connection.weight * self.compute_node(connection.input_node))
			value = np.mean(value)
			if node.node_type == 'hidden':
				node.value = self.hidden_activation_function(self.hidden_activation_coeff * value)
			elif node.node_type == 'output':
				node.value = self.output_activation_function(self.output_activation_coeff * value)
		return node.value	

	def evaluate_input(self, input):
		output = np.ones(len(self.output_nodes))
		self.restart()
		self.set_input(input)
		# try:
		for i, node in enumerate(self.output_nodes):
			output[i] = self.compute_node(node.id)
		return output
		# except:
		# 	return None

	def predict(self, X):
		y = np.zeros(X.shape[0])
		for i, xi in enumerate(X):
			y[i] = self.evaluate_input(xi)
		return y

	def score(self, X, y):
		y_predict = self.predict(X)
		y_predict = np.expand_dims(y_predict, axis=1)
		return y.shape[0] - np.sum(np.square(y - y_predict))

	def eval_model(self, X, y, fitness_function, l2_parameter):
		n = y.shape[0]
		w = np.mean([connection.weight**2 for connection in self.connections if connection.enabled])
		y_predict = self.predict(X)
		y_predict = np.expand_dims(y_predict, axis=1)
		loss = fitness_function(y, y_predict) + ((l2_parameter * w) / (2 * n))
		acc = np.mean(y == np.round(y_predict))
		return loss, acc


if __name__ == '__main__':

	node_genes = []
	node_genes.append(NodeGene(0, 'input'))
	node_genes.append(NodeGene(1, 'input'))

	node_genes.append(NodeGene(2, 'hidden'))
	node_genes.append(NodeGene(3, 'output'))

	connection_genes = []
	connection_genes.append(ConnectionGene(0, 2, 0, 1, True))
	connection_genes.append(ConnectionGene(0, 3, 1, -2, True))
	connection_genes.append(ConnectionGene(2, 3, 2, -3, True))
	connection_genes.append(ConnectionGene(1, 2, 2, -4, False))

	genome = Genome(node_genes, connection_genes)

	x = np.array([[1, 2]])
	y = np.array([[0]])
	l2_parameter = 0.5

	"ANN PyTorch"

	f = torch_fitness_function
	activation = {}
	activation['hidden_activation_function'] = nn.Tanh()
	activation['hidden_activation_coeff'] = 4.9 * 0.5
	activation['output_activation_function'] = Gaussian()
	activation['output_activation_coeff'] = 1
	
	x_torch = torch.from_numpy(x).type(torch.float32)
	y_torch = torch.from_numpy(y).type(torch.float32)

	genome.compute_phenotype(activation)
	x_prima = x_torch.index_select(1, genome.selected_features)
	connection_weights = [connection.weight for connection in genome.connection_genes if connection.enabled]
	mean_weight = np.mean(np.square(np.array(connection_weights))) if connection_weights else 0
	loss, acc = eval_model(genome.phenotype, x_prima, y_torch, f, l2_parameter, mean_weight)
	fitness = 100 - loss
	print(f'Fitness: {fitness}, Accuracy: {acc}')


	"ANN"

	f2 = fitness_function
	activation['hidden_activation_function'] = np.tanh
	activation['output_activation_function'] = gaussian

	model = ArtificialNeuralNetwork(genome, activation)
	loss, acc = model.eval_model(x, y, f2, l2_parameter)
	fitness = 100 - loss
	print(f'Fitness: {fitness}, Accuracy: {acc}')