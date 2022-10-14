import numpy as np
import torch

def fitness_function(y, y_predict):
	classes = np.unique(y)
	fitness = np.zeros_like(classes, dtype=np.float32)
	y_predict[y_predict == 0] = np.nextafter(0., 1.)
	y_predict[y_predict == 1] = np.nextafter(1., 0.)
	for i, q in enumerate(classes):
		idx = np.argwhere(y == q)
		fitness[i] = -np.mean(y[idx] * np.log(y_predict[idx]) + (1 - y[idx]) * np.log(1 - y_predict[idx]))
	return np.mean(fitness)

def torch_fitness_function(y, y_predict):
	classes = torch.unique(y)
	fitness = torch.zeros_like(classes)
	y_predict[y_predict == 0] = torch.nextafter(torch.tensor([0.]), torch.tensor([1.]))
	y_predict[y_predict == 1] = torch.nextafter(torch.tensor([1.]), torch.tensor([0.]))
	for i, q in enumerate(classes):
		idx = torch.where(y == q)
		fitness[i] = -torch.mean(y[idx] * torch.log(y_predict[idx]) + (1 - y[idx]) * torch.log(1 - y_predict[idx]))
	return torch.mean(fitness)
