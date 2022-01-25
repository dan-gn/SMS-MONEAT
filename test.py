import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from models.genotype import Genome, NodeGene, ConnectionGene
from models.ann_pytorch import Ann_PyTorch, eval_model
from utilities.fitness_functions import torch_fitness_function
from utilities.activation_functions import Gaussian
from utilities.microarray_ds import MicroarrayDataset
from utilities.stats_utils import KruskalWallisFilter


filename = 'Leukemia_GSE71935'
seed = 2
trn_size = 0.7

ds = MicroarrayDataset(f'./datasets/CUMIDA/{filename}.arff')

debug = True if seed == -1 else False
x_train, x_test, y_train, y_test = ds.split(trn_size=trn_size, random_state=seed)

filter = KruskalWallisFilter(threshold=0.01)
x_train, features_selected = filter.fit_transform(x_train, y_train)
x_test, _ = filter.transform(x_test)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

x_train = torch.from_numpy(x_train).type(torch.float32)
y_train = torch.from_numpy(y_train).type(torch.float32)
x_test = torch.from_numpy(x_test).type(torch.float32)
y_test = torch.from_numpy(y_test).type(torch.float32)

nodes = []
nodes.append(NodeGene(118, 'input'))
nodes.append(NodeGene(1615, 'hidden'))
nodes.append(NodeGene(947, 'output'))

connections = []
connections.append(ConnectionGene(118, 1615, 2977, 0.89, True))
connections.append(ConnectionGene(1615, 947, 2978, -0.89, True))

genome = Genome(nodes, connections)


activation = {}
activation['hidden_activation_function'] = nn.Tanh()
activation['hidden_activation_coeff'] = 4.9 * 0.5
activation['output_activation_function'] = Gaussian()
activation['output_activation_coeff'] = 1
fitness_function = torch_fitness_function
l2_parameter = 0.5

genome.compute_phenotype(activation)
if genome.selected_features.shape[0] == 0:
	print(None, 0, 0)
else:
	x_prima = x_train.index_select(1, genome.selected_features)
	connection_weights = [connection.weight for connection in genome.connection_genes if connection.enabled]
	mean_weight = np.mean(np.square(np.array(connection_weights))) if connection_weights else 0
	loss, acc, gmean = eval_model(genome.phenotype, x_prima, y_train, fitness_function, l2_parameter, mean_weight)
	fitness = 100 - loss
	if fitness < 0:
		print(f'Fitness negativo: {fitness}')
	print(acc, fitness.detach().numpy(), gmean)