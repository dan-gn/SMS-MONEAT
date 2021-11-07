import torch
import torch.nn as nn

import numpy as np


class Ann_PyTorch(nn.Module):

	def __init__(self, layer_weights, activation):

		super().__init__()

		self.activation_f1 = activation['hidden_activation_function']
		self.activation_c1 = activation['hidden_activation_coeff']
		self.activation_f2 = activation['output_activation_function']
		self.activation_c2 = activation['output_activation_coeff']

		self.layers = []

		for weights in layer_weights:
			in_features = weights.shape[0]			
			out_features = weights.shape[1]
			layer = nn.Linear(in_features, out_features, bias=False)
			self.layers.append(layer)

		with torch.no_grad():
			for i, layer in enumerate(self.layers):
				layer.weight = nn.Parameter(layer_weights[i])

	def forward(self, x):

		for layer in self.layers[:-1]:
			y = layer(x) / layer.weight.shape[0]
			y = self.activation_f1(self.activation_c1 * y)
			x = torch.cat((x, y), dim=1)
		
		y = self.layers[-1](x) / self.layers[-1].weight.shape[0]
		y = self.activation_f2(self.activation_c2 * y)

		return y




def eval_model(model, x, y, fitness_function, l2_parameter, w):
	with torch.no_grad():
		y_predict = model(x)
		n = x.shape[0]
		# loss = torch.nn.functional.binary_cross_entropy(y_predict, y)
		loss = fitness_function(y, y_predict) + ((l2_parameter * w) / (2 * n))
		acc = (y == torch.round(y_predict)).type(torch.float32).mean()
	return loss, acc
