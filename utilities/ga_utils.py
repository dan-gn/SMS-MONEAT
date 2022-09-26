import numpy as np
import random
import math

def simulated_binary_crossover(x, eta):
	pass

def polynomial_mutation(x, eta):
	r = random.uniform(0, 1)
	if r < 0.5:
		delta = (2*r) ** (1 / (eta+1)) - 1
	else:
		delta = 1 - (2 * (1-r)) ** (1 / (eta+1))
	return x + delta

def roulette_wheel(p):
	r = random.uniform(0, 1) * sum(p)	
	q = np.cumsum(p)
	return next(idx for idx, value in enumerate(q) if value >= r)

def tournament_selection(population, probs, n_competitors):
	draw = np.random.permutation(population)
	competitors = draw[:n_competitors]
	winner = roulette_wheel(probs[competitors])
	return competitors[winner]

# Get parent selection probabilities
def compute_parent_selection_prob(population, beta=1):
	# Get an array of all cost of current population, add acceptance criteria value
	# and divide by the mean of the array to avoid overflow while computing exponential
	fitness = np.array([member.shared_fitness for member in population]) 
	mean_fitness = np.mean(fitness)
	if mean_fitness != 0 and mean_fitness != math.inf:
		fitness /= mean_fitness
	return np.exp(-beta * fitness)
