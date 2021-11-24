import torch
import torch.nn as nn

from utilities.stats_utils import geometric_mean

class Ann_PyTorch(nn.Module):

	def __init__(self, layer_weights, n_inputs, activation):

		super().__init__()

		self.layer_weights = layer_weights
		self.n_inputs = n_inputs
		self.activation = activation

		self.set_activation_functions()

		self.layers = []
		for weights in layer_weights:
			in_features = weights.shape[0]			
			out_features = weights.shape[1]
			layer = nn.Linear(in_features, out_features, bias=False)
			self.layers.append(layer)

		self.set_layer_weights(layer_weights)


	def set_activation_functions(self):
		self.activation_f1 = self.activation['hidden_activation_function']
		self.activation_c1 = self.activation['hidden_activation_coeff']
		self.activation_f2 = self.activation['output_activation_function']
		self.activation_c2 = self.activation['output_activation_coeff']

	def set_layer_weights(self, layer_weights):
		with torch.no_grad():
			for i, layer in enumerate(self.layers):
				layer.weight = nn.Parameter(layer_weights[i])

	def forward(self, x):
		for i, layer in enumerate(self.layers[:-1]):
			y = layer(x) / self.n_inputs[i]
			y = self.activation_f1(self.activation_c1 * y)
			x = torch.cat((x, y), dim=1)
		y = self.layers[-1](x) / self.n_inputs[-1]
		y = self.activation_f2(self.activation_c2 * y)
		return y

	def copy(self):
		return Ann_PyTorch(self.layer_weights, self.n_inputs, self.activation)





def eval_model(model, x, y, fitness_function, l2_parameter, w):
	with torch.no_grad():
		y_predict = model(x)
		n = x.shape[0]
		# loss = torch.nn.functional.binary_cross_entropy(y_predict, y)
		loss = fitness_function(y, y_predict) + ((l2_parameter * w) / (2 * n))
		acc = (y == torch.round(y_predict)).type(torch.float32).mean()
		g_mean = geometric_mean(y, y_predict)
	return loss, acc, g_mean
