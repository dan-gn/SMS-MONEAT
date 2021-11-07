import numpy as np
from statistics import mean
import torch
import torch.nn as nn

from .genotype import Genome, NodeGene, ConnectionGene
from .ann_pytorch import Ann_PyTorch

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

	def compute_node(self, id):
		for node in self.nodes:
			if node.id == id:
				break

		if node.value is None:
			value = []
			for connection in self.connections:
				if connection.enabled and connection.output_node == node.id:
					value.append(connection.weight * self.compute_node(connection.input_node))
			value = mean(value)
			if node.node_type == 'hidden':
				node.value = self.hidden_activation_function(self.hidden_activation_coeff * value)
			elif node.node_type == 'output':
				node.value = self.output_activation_function(self.output_activation_coeff * value)
		return node.value	

	def evaluate_input(self, input):
		output = np.ones(len(self.output_nodes))
		self.restart()
		self.set_input(input)
		try:
			for i, node in enumerate(self.output_nodes):
				output[i] = self.compute_node(node.id)
			return output
		except:
			return None

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
		w = mean([connection.weight**2 for connection in self.connections])
		y_predict = self.predict(X)
		y_predict = np.expand_dims(y_predict, axis=1)
		loss = fitness_function(y, y_predict) + ((l2_parameter * w) / (2 * n))
		acc = np.mean(y == np.round(y_predict))
		return loss, acc

