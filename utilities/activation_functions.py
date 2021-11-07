import torch
import torch.nn as nn
import numpy as np

def gaussian(x, c=5, mean=0, std=1):
    return np.exp((- c * (x - mean) ** 2)/(2* std ** 2))

def torch_gaussian(x, c=5, mean=0, std=1):
	return np.exp((-c * (x-mean)**2) / (2 * std**2))

def sigmoid(x, c=4.9):
	return nn.Sigmoid(c * x)

def tanh(x, c=4.9*0.5):
	return nn.Tanh(c * x)

class Gaussian(nn.Module):

	def __init__(self):
		super().__init__()


	def forward(self, input):
		return torch_gaussian(input)
