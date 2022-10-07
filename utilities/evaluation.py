import torch
from torchmetrics import ConfusionMatrix

from typing import Tuple, Callable, Any

from models.genotype import Genome
from utilities.stats_utils import geometric_mean

def eval_model(model: Genome, x: torch.Tensor, y: torch.Tensor, fitness_function: Callable[[torch.Tensor, torch.Tensor], Any], l2_parameter: torch.float32, w: torch.float32) -> Tuple[Any, Any, Any]:
	with torch.no_grad():
		y_predict = model(x)
		n = x.shape[0]
		# loss = torch.nn.functional.binary_cross_entropy(y_predict, y)
		loss = fitness_function(y, y_predict) + ((l2_parameter * w) / (2 * n))
		acc = (y == torch.round(y_predict)).type(torch.float32).mean()
		g_mean = geometric_mean(y, y_predict)
	return loss, acc, g_mean

def eval_in_population(population: list, x: torch.Tensor, y: torch.Tensor) -> Tuple[Any, Any]:
	predictions = torch.zeros((len(population), y.shape[0]))
	with torch.no_grad():
		for i, member in enumerate(population):
			x_prima = x.index_select(1, member.selected_features)
			predictions[i, :] = member.phenotype(x_prima)
	y_predict = predictions.mean(axis=0)
	acc = (y == torch.round(y_predict)).type(torch.float32).mean()
	g_mean = geometric_mean(y, y_predict)
	return acc, g_mean
	

def eval_model_2(model: Genome, x: torch.Tensor, y: torch.Tensor, fitness_function: Callable[[torch.Tensor, torch.Tensor], Any], l2_parameter: torch.float32, w: torch.float32) -> Tuple[Any, Any, Any, Any, Any]:
	with torch.no_grad():
		y_predict = model(x)
		n = x.shape[0]
		loss = fitness_function(y, y_predict) + ((l2_parameter * w) / (2 * n))
		acc = (y == torch.round(y_predict)).type(torch.float32).mean()
		g_mean = geometric_mean(y, y_predict)
		cofmat = ConfusionMatrix(num_classes=2)
		results = cofmat(y_predict, y)
		false_negative = results[0, 1]
		false_positive = results[1, 0]
	return loss, acc, g_mean, false_negative, false_positive

