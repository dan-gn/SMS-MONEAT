import numpy as np
import copy
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
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
        self.iteration_num = 0

    def update(self, population: Genome, iteration_num, n_invalid_nets=0):
        self.population.append([copy.deepcopy(member) for member in population])
        self.iteration_num = iteration_num
        self.n_invalid_nets[iteration_num] = n_invalid_nets

    def compute_info(self):
        for i in range(self.iteration_num):
            if type(self.population[0][0].fitness) == list:
                self.loss[i] = np.mean([member.fitness[0] for member in self.population[i] if member.accuracy is not None])
                self.fs[i] = np.mean([member.fitness[1] for member in self.population[i] if member.accuracy is not None])
            else:
                self.loss[i] = np.mean([member.fitness for member in self.population[i] if member.accuracy is not None])
                self.fs[i] = np.mean([member.selected_features.shape[0] for member in self.population[i] if member.accuracy is not None])
            self.accuracy[i] = np.mean([member.accuracy for member in self.population[i] if member.accuracy is not None])
            self.g_mean[i] = np.mean([member.g_mean for member in self.population[i] if member.accuracy is not None])
        
            


class BestInidividualRecord(Record):

    def __init__(self, max_iterations) -> None:
        super().__init__(max_iterations)

    def update(self, individual: Genome, iteration_num):
        self.loss[iteration_num] = individual.fitness
        self.fs[iteration_num] = individual.selected_features.shape[0]
        self.accuracy[iteration_num] = individual.accuracy
        self.g_mean[iteration_num] = individual.g_mean
