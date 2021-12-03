"""
NEAT's genetic encoding scheme
"""
import random
import numpy as np
import torch
import torch.nn as nn
from models.ann_pytorch import Ann_PyTorch

class NodeGene:
	"""
	A node gene to represent an input, hidden or output node from a Artificial Neural Network.
	"""

	def __init__(self, id, node_type='hidden'):
		self.id = id
		self.node_type = node_type
		self.value = None
		self.layer = None

	def copy(self):
		new_node = NodeGene(self.id, self.node_type)
		new_node.layer = self.layer
		return new_node


class ConnectionGene:
	"""
	A connection gene refers to two node gene being connected.
	Each connection gene specifies the in-node, the out-node, the weight of the connection,
	wether or not the connection gene is expressed (an enabled bit), and an innovation number, 
	which allows finding corresponding genes.
	"""

	def __init__(self, input_node, output_node, innovation_number=-1, weight=1, enabled=True):
		self.input_node = input_node
		self.output_node = output_node
		self.innovation_number = innovation_number
		self.weight = weight
		self.enabled = enabled

	def copy(self):
		return ConnectionGene(self.input_node, self.output_node, self.innovation_number, self.weight, self.enabled)


class Genome:
	"""
	Genomes are linear representations of network connectivity. Each genome includes a list of node genes and a list of
	connection genes.
	"""

	def __init__(self, node_genes=None, connection_genes=None, min_value=1e-6, max_value=1):
		self.node_genes = [] if node_genes is None else node_genes
		self.connection_genes = [] if connection_genes is None else connection_genes
		self.fitness = 0
		self.shared_fitness = 0
		self.weight_min_value = min_value
		self.weight_max_value = max_value
		self.accuracy = None
		self.g_mean = None
		self.phenotype = None

	def create_node_genes(self, n_inputs, n_outputs):
		for id in range(n_inputs):
			self.node_genes.append(NodeGene(id, 'input'))
		for id in range(n_inputs, n_inputs+n_outputs):
			self.node_genes.append(NodeGene(id, 'output'))
	
	def connect_all(self):
		innovation_number = 0
		input_nodes_index = []
		output_nodes_index = []
		for i, node in enumerate(self.node_genes):
			if node.node_type == 'input':
				input_nodes_index.append(i)
			else:
				output_nodes_index.append(i)

		for i in input_nodes_index:
			for j in output_nodes_index:
				connection = ConnectionGene(self.node_genes[i].id, self.node_genes[j].id, innovation_number)
				self.connection_genes.append(connection)
				innovation_number += 1

		return innovation_number

	def connect_one_input(self, innovation_number):
		# Get input and output nodes
		input_node_ids = []
		output_node_ids = []
		for node in self.node_genes:
			if node.node_type == 'input':
				input_node_ids.append(node.id)
			elif node.node_type == 'output':
				output_node_ids.append(node.id)
		# Choose one input and one output
		input_node = random.choice(input_node_ids)	
		output_node = random.choice(output_node_ids)	
		# Check if a connection already exists with that input and output
		inum = self.get_innovation_number(input_node, output_node)
		if inum:
			connection = self.get_connection_gene(inum)
		else:
			# if don't, create a new one
			connection = ConnectionGene(input_node, output_node, innovation_number)
			self.connection_genes.append(connection)
			innovation_number += 1
		# Randomize weight
		connection.weight = random.uniform(self.weight_min_value, self.weight_max_value)
		return connection, innovation_number 

	def set_weight_limits(self, min_value, max_value):
		self.weight_min_value = min_value
		self.weight_max_value = max_value

	def randomize_weights(self):
		for connection in self.connection_genes:
			connection.weight = random.uniform(self.weight_min_value, self.weight_max_value)

	def get_node_gene(self, id):
		for node in self.node_genes:
			if node.id == id:
				return node
		return None

	def get_connection_gene(self, innovation_number):
		for connection in self.connection_genes:
			if connection.innovation_number == innovation_number:
				return connection
		return None

	def get_innovation_number(self, input_node_id, output_node_id):
		for connection in self.connection_genes:
			if connection.input_node == input_node_id and connection.output_node == output_node_id:
				return connection.innovation_number
		return None

	def add_node(self, node, old_connection_inum, innovation_number):
		""" 
		In the add node mutation, an existing connection is split and the new node placed where the old connection used
		to be. The old connection is disabled and two new connections are added to the genome. The new connection leading
		into the new node receives a weight of 1, and the new connection leading out receives the same weight as old connection.
		"""
		self.node_genes.append(node)
		self.node_genes = sorted(self.node_genes, key = lambda x: x.id)
		connection = self.get_connection_gene(old_connection_inum)
		connection.enabled = False
		innovation_number = self.add_connection(connection.input_node, node.id, innovation_number, 1)
		innovation_number = self.add_connection(node.id, connection.output_node, innovation_number, connection.weight)
		return innovation_number

	def remove_node(self, node_id):
		for index, node in enumerate(self.node_genes):
			if node.id == node_id:
				self.node_genes.pop(index)
				break

	def add_connection(self, in_node, out_node, innovation_number, weight=None):
		"""
		In add connection mutation, a single new connection gene with a random weight is added connecting two previously
		unconnected nodes.
		"""
		if not weight:
			weight = random.uniform(self.weight_min_value, self.weight_max_value)
		connection = ConnectionGene(in_node, out_node, innovation_number, weight)
		self.connection_genes.append(connection)
		self.connection_genes = sorted(self.connection_genes, key = lambda x: x.innovation_number)
		return innovation_number + 1

	def remove_connection(self, innovation_number):
		for index, connection in enumerate(self.connection_genes):
			if connection.innovation_number == innovation_number:
				self.connection_genes.pop(index)
				break

	def get_possible_connections(self, input_node_id):
		possible_connections = []
		for node in self.node_genes:
			# Discard same node and input nodes
			if node.id != input_node_id and node.node_type != 'input':
				possible_connections.append(node.id)
		for connection in self.connection_genes:
			# Discard nodes which are already connected to this node
			if connection.input_node == input_node_id:
				if connection.output_node in possible_connections:
					possible_connections.remove(connection.output_node)
			elif connection.output_node == input_node_id:
				if connection.input_node in possible_connections:
					possible_connections.remove(connection.input_node)
					# Discard nodes that are behind this node 
					self.discard_backwards(connection.input_node, possible_connections)	
		return possible_connections
	
	def discard_backwards(self, node_id, possible_connections):
		if possible_connections:
			for connection in self.connection_genes:
				if connection.input_node in possible_connections and connection.output_node == node_id:
					possible_connections.remove(connection.input_node)
					possible_connections = self.discard_backwards(connection.input_node, possible_connections)
		return possible_connections

	def search_loop(self, node, nodes_checked):
		if node.id in nodes_checked:
			return False
		else:
			found = False
			nodes_checked.append(node.id)
			for connection in self.connection_genes:
				if connection.input_node == node.id:
					found = True
					output_node = self.get_node_gene(connection.output_node)
					if output_node.node_type == 'hidden':
						if not self.search_loop(output_node, list(nodes_checked)):	
							return False
		return found

	def check_connectivity(self):
		any_input = False
		any_output = False
		for node in self.node_genes:
			if node.node_type == 'input':
				any_input = True
				nodes_checked = []
				if not self.search_loop(node, nodes_checked):
					return False
			elif node.node_type == 'output':
				any_output = True
		return any_input and any_output

	def connection_from_input(self, node):
		if node.node_type == 'input':
			return True
		for connection in self.connection_genes:
			if connection.output_node == node.id and connection.enabled:
				input_node = self.get_node_gene(connection.input_node)
				if self.connection_from_input(input_node):
					return True
		return False
	
	def validate_network(self):
		any_output = False
		for node in self.node_genes:
			if node.node_type == 'output':
				any_output = True
				if not self.connection_from_input(node):
					return False
		return any_output



	def depth_first_search(self, node, unchecked_connections, max_depth=0):
		if node.layer is None:
			node.layer = max_depth
		elif node.layer < max_depth:
			node.layer = max_depth
		else:
			return max_depth
		max_depth += 1
		current_depth = max_depth
		for connection in list(unchecked_connections):
			if connection.input_node == node.id:
				if node.node_type == 'input':
					unchecked_connections.remove(connection)
				output_node = self.get_node_gene(connection.output_node)
				if output_node.node_type == 'hidden':
					depth = self.depth_first_search(output_node, unchecked_connections, current_depth)
					max_depth = max(depth, max_depth)
		return max_depth

	def tag_layers(self):
		for node in self.node_genes:
			node.layer = None
		input_nodes = [node for node in self.node_genes if node.node_type == 'input']
		output_nodes = [node for node in self.node_genes if node.node_type == 'output']
		unchecked_connections = [connection for connection in self.connection_genes if connection.enabled]
		max_depth = 0
		for node in input_nodes:
			depth = self.depth_first_search(node, unchecked_connections)
			max_depth = max(depth, max_depth)
		for node in output_nodes:
			node.layer = max_depth
		return max_depth

	def build_layers(self):
		layers = []
		layer_weights = []
		n_inputs = []

		max_depth = self.tag_layers()

		self.node_genes = sorted(self.node_genes, key = lambda x: x.id)
		for i in range(max_depth + 1):
			layer_i = [node.id for node in self.node_genes if node.layer == i]
			layers.append(layer_i)

		selected_features = []
		for connection in self.connection_genes:
			if connection.input_node in layers[0] and connection.enabled:
				selected_features.append(connection.input_node)
		selected_features = sorted(set(selected_features))
		layers[0] = list(selected_features)

		for i, layer_i in enumerate(layers):
			if i == max_depth:
				break
			in_features = len(layer_i)
			out_features = len(layers[i+1])
			weights = np.zeros((out_features, in_features), dtype=np.float32)
			inputs = np.zeros(out_features)
			for connection in self.connection_genes:
				if connection.input_node in layer_i and connection.output_node in layers[i+1]:
					row = layers[i+1].index(connection.output_node)
					col = layer_i.index(connection.input_node)
					weights[row, col] = connection.weight
					inputs[row] += 1

			layer_i.extend(layers[i+1])
			layers[i+1] = list(layer_i)
			layer_weights.append(weights)
			n_inputs.append(inputs)

		return selected_features, layer_weights, n_inputs

	def count_nodes(self):
		n_input_nodes = 0
		n_hidden_nodes = 0
		n_output_nodes = 0
		for node in self.node_genes:
			if node.node_type == 'input':
				n_input_nodes += 1
			elif node.node_type == 'hidden':
				n_hidden_nodes += 1
			else:
				n_output_nodes += 1
		return n_input_nodes, n_hidden_nodes, n_output_nodes

	def compatibility_distance(self, other, c):
		# Get value of N
		n_max = max(len(self.connection_genes), len(other.connection_genes))
		n = 1 if n_max < 20 else n_max
		# Initialize variables: disjoint and matching genes counters, weight difference between matching genes
		# and index of connection genes
		n_disjoint = 0
		n_matching = 0
		weight_difference = 0
		i = 0
		j = 0
		# Count matching and disjoint genes and the sum of weight difference of matching genes
		while i < len(self.connection_genes) and j < len(other.connection_genes):
			if self.connection_genes[i].innovation_number == other.connection_genes[j].innovation_number:
				n_matching += 1
				weight_difference += abs(self.connection_genes[i].weight - other.connection_genes[j].weight)
				i += 1
				j += 1
			elif self.connection_genes[i].innovation_number < other.connection_genes[j].innovation_number:
				n_disjoint += 1
				i += 1
			else:
				n_disjoint += 1
				j += 1
		# Infer excess genes
		n_excess = (len(self.connection_genes) - i) + (len(other.connection_genes) - j)
		# Compute compatibility distance
		if n_matching == 0:
			n_matching = 1
		return (c[0] * n_excess / n) + (c[1] * n_disjoint / n) + (c[2] * weight_difference / n_matching)

	def compute_phenotype(self, activation):
		selected_features, layer_weights, n_inputs = self.build_layers()
		self.selected_features = torch.tensor(selected_features).type(torch.int32)
		layer_weights = [torch.from_numpy(w).type(torch.float32) for w in layer_weights]
		n_inputs = [torch.from_numpy(n).type(torch.float32) for n in n_inputs]
		self.phenotype = Ann_PyTorch(layer_weights, n_inputs, activation)

	def copy(self, with_phenotype=False):
		genome_copy = Genome()
		genome_copy.node_genes = [node.copy() for node in self.node_genes]
		genome_copy.connection_genes = [connection.copy() for connection in self.connection_genes]
		genome_copy.set_weight_limits(self.weight_min_value, self.weight_max_value)
		genome_copy.fitness = self.fitness
		genome_copy.shared_fitness = self.shared_fitness
		genome_copy.accuracy = self.accuracy
		genome_copy.g_mean = self.g_mean
		if with_phenotype:
			genome_copy.selected_features = self.selected_features
			genome_copy.phenotype = self.phenotype.copy()
		return genome_copy

	def describe(self):
		print('Node genes:')
		for node in self.node_genes:
			print(f'Node {node.id}, type {node.node_type}, layer {node.layer}')
		print('Connection genes:')
		for connection in self.connection_genes:
			print(f'Connection {connection.innovation_number}: Input node id = {connection.input_node}, Output node id = {connection.output_node}, Weight = {connection.weight}, Enabled = {connection.enabled}')
		print(f'Fitness: {self.fitness}')


