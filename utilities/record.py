import numpy as np
from models.genotype import MultiObjectiveGenome as Genome


class Record:

    def __init__(self, max_iterations) -> None:
        self.loss = np.empty((max_iterations + 1, 1))
        self.loss[:] = np.nan
        self.fs = np.copy(self.loss)
        self.accuracy = np.copy(self.loss)
        self.g_mean = np.copy(self.loss)
        self.n_invalid_nets = np.copy(self.loss)
        self.population = []

    def update(self, population: Genome, iteration_num, n_invalid_nets=0):
        self.loss[iteration_num] = np.mean([member.fitness[0] for member in population])
        self.fs[iteration_num] = np.mean([member.fitness[1] for member in population])
        self.accuracy[iteration_num] = np.mean([member.accuracy for member in population])
        self.g_mean[iteration_num] = np.mean([member.g_mean for member in population])
        self.n_invalid_nets[iteration_num] = n_invalid_nets
        self.population.append([member.copy() for member in population])

class BestInidividualRecord(Record):

    def __init__(self, max_iterations) -> None:
        super().__init__(max_iterations)

    def update(self, individual: Genome, iteration_num):
        self.loss[iteration_num] = individual.fitness
        self.fs[iteration_num] = individual.selected_features.shape[0]
        self.accuracy[iteration_num] = individual.accuracy
        self.g_mean[iteration_num] = individual.g_mean