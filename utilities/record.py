import numpy as np
import copy
import sys
import os

from typing import List

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
        self.n_hidden = np.copy(self.loss)
        self.n_active_connections = np.copy(self.loss)
        self.n_nodes = np.copy(self.loss)
        self.n_connections = np.copy(self.loss)
        self.population = []
        self.iteration_num = 0

    def update(self, population: List[Genome], iteration_num, n_invalid_nets=0):
        self.population.append([member.copy(with_phenotype=False) for member in population])
        self.iteration_num = iteration_num
        self.n_invalid_nets[iteration_num] = n_invalid_nets
        # if isinstance(population[0], Genome):
        #     self.loss[iteration_num] = np.mean([member.fitness[0] for member in population if member.accuracy is not None])
        #     self.fs[iteration_num] = np.mean([member.fitness[1] for member in population if member.accuracy is not None])
        # else:
        #     self.loss[iteration_num] = np.mean([member.fitness for member in population if member.accuracy is not None])
        #     self.fs[iteration_num] = np.mean([member.selected_features.shape[0] for member in population if member.accuracy is not None])
        # self.accuracy[iteration_num] = np.mean([member.accuracy for member in population if member.accuracy is not None])
        # self.g_mean[iteration_num] = np.mean([member.g_mean for member in population if member.accuracy is not None])
        # self.n_nodes[iteration_num] = np.mean([len(member.node_genes) for member in population if member.accuracy is not None])
        # self.n_connections[iteration_num] = np.mean([len(member.connection_genes) for member in population if member.accuracy is not None])
        # self.n_active_connections[iteration_num] = np.mean([member.n_active_connections for member in population if member.accuracy is not None])
        # self.n_hidden[iteration_num] = np.mean([member.n_hidden for member in population if member.accuracy is not None])


    def compute_info(self):
        for i in range(self.iteration_num):
            if isinstance(self.population[0][0], Genome):
                self.loss[i] = np.mean([member.fitness[0] for member in self.population[i] if member.accuracy is not None])
                self.fs[i] = np.mean([member.fitness[1] for member in self.population[i] if member.accuracy is not None])
            else:
                self.loss[i] = np.mean([member.fitness for member in self.population[i] if member.accuracy is not None])
                self.fs[i] = np.mean([member.selected_features.shape[0] for member in self.population[i] if member.accuracy is not None])
            self.accuracy[i] = np.mean([member.accuracy for member in self.population[i] if member.accuracy is not None])
            self.g_mean[i] = np.mean([member.g_mean for member in self.population[i] if member.accuracy is not None])
            self.n_nodes[i] = np.mean([len(member.node_genes) for member in self.population[i] if member.accuracy is not None])
            self.n_connections[i] = np.mean([len(member.connection_genes) for member in self.population[i] if member.accuracy is not None])
            self.n_active_connections[i] = np.mean([member.n_active_connections for member in self.population[i] if member.accuracy is not None])
            self.n_hidden[i] = np.mean([member.n_hidden for member in self.population[i] if member.accuracy is not None])
        
            


class BestInidividualRecord(Record):

    def __init__(self, max_iterations) -> None:
        super().__init__(max_iterations)

    def update(self, individual: Genome, iteration_num):
        self.loss[iteration_num] = individual.fitness
        self.fs[iteration_num] = individual.selected_features.shape[0]
        self.accuracy[iteration_num] = individual.accuracy
        self.g_mean[iteration_num] = individual.g_mean


class KRecord(Record):

    def __init__(self, max_iterations) -> None:
        super().__init__(max_iterations)

    def update(self, population, iteration_num):
        self.population.append([member.reduced_copy() for member in population])
        self.iteration_num = iteration_num