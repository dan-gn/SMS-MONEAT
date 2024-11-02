import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utilities.moea_utils import non_dominated_sorting
from algorithms.chc import Individual

import numpy as np
import pygmo
import copy
import time

def function_a(population):
    fronts = non_dominated_sorting(population)

def function_b(population):
    points = [member.fitness for member in population]
    ndf, _, _, _ = pygmo.fast_non_dominated_sorting(points = points)
    fronts = [[] for _ in range(len(ndf))]
    for index_front, front in enumerate(ndf):
        for index_pop in front:
            fronts[index_front].append(population[index_pop])


def initialize_population(population_size=100, n_var = 10000):
    population = [] 
    for _ in range(population_size):
        member = Individual(n_variables=n_var)
        member.initialize()
        # member.accuracy, member.fitness, member.g_mean = self.evaluate(member, self.x_train, self.y_train)
        member.fitness = np.random.randint(0, 14, size=2)
        population.append(member)
    return population

if __name__ == '__main__':
    population = initialize_population()

    start_time = time.time()
    for _ in range(10000):
         function_a(population)
    end_time = time.time() - start_time
    print(f'Time A: {end_time}')

    start_time = time.time()
    for _ in range(10000):
         function_b(population)
    end_time = time.time() - start_time
    print(f'Time B: {end_time}')
