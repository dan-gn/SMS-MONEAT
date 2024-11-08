import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import numpy as np
import pandas as pd
import scipy.stats as stats
import copy
import math
import pygmo
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
import torch

from utilities.ga_utils import tournament_selection, single_point_crossover
from algorithms.neat import set_seed
from utilities.moea_utils import non_dominated_sorting
from utilities.stats_utils import geometric_mean
from utilities.fitness_functions import torch_fitness_function

import time
import itertools

class Individual:

    def __init__(self, n_variables: int = 1) -> None:
        self.n_variables = n_variables
        self.genome = None	
        self.fitness = None
        self.accuracy = None
        self.g_mean = None
        self.rank = None
        self.crowding_distance = None
        self.dominates_to = []
        self.n_dominated_by = 0
        self.reduced_genome = []

    def initialize(self):
        # self.genome = np.zeros(self.n_variables)
        # for _ in range(1):
        #     index = np.random.randint(self.n_variables)
        #     self.genome[index] = 1
        # self.genome = np.random.randint(0, 2, size= self.n_variables)
        threshold = 0.01
        self.genome = np.random.rand(self.n_variables)
        self.genome = [0 if x >= threshold else 1 for x in self.genome]

    def copy(self):
        new_individual = Individual(self.n_variables)
        new_individual.genome = np.copy(self.genome)
        new_individual.rank = self.rank
        new_individual.fitness = self.fitness
        new_individual.accuracy = self.accuracy
        new_individual.g_mean = self.g_mean
        new_individual.n_dominated_by = self.n_dominated_by
        new_individual.dominates_to = list(self.dominates_to)
        return new_individual



class CHC:

    def __init__(self, problem: dict, params: dict) -> None:
        self.population_size = 100
        self.cataclysmic_muation_prob = 0.35
        self.preserved_population = 0.05
        self.initial_convergence_count = 0.02
        self.convergence_value_k = 1
        self.max_evaluations = 6000
        self.current_evaluations = 0
        # Crossover -> HUX
        # Parent selection -> Random with incest threshold
        # New generation selection -> Elitist selection
        # Ordering criterion -> Sort by value
        # Problem parameters
        self.x_train, self.y_train = problem['x_train'], problem['y_train']
        self.x_test, self.y_test = problem['x_test'], problem['y_test']
        self.n_var = self.x_train.shape[1]

    def initialize_convergence_count(self):
        return int(self.initial_convergence_count * self.n_var)

    def initialize_population(self):
        population = [] 
        for _ in range(self.population_size):
            member = Individual(n_variables=self.n_var)
            member.initialize()
            member.accuracy, member.fitness, member.g_mean = self.evaluate(member, self.x_train, self.y_train)
            population.append(member)
        return population

    def ending_condition(self, iteration):
        return False # No ending condition
    
    def select_parents(self, population, convergence_count):
        # parents = []
        # for _ in range(int(self.population_size/2)):
        #     parent_index = np.random.choice(self.population_size, 2, replace=False)
        #     parents.append([population[parent_index[0]], population[parent_index[1]]])
        # return parents
        # parents = []
        # for parent1 in population:
            # for parent2 in population:
            #     if self.hamming_distance(parent1.genome, parent2.genome) > convergence_count:
            #         parents.append([parent1, parent2])
        parents = [(parent1, parent2) for parent1, parent2 in itertools.combinations(population, 2) if self.hamming_distance(parent1.genome, parent2.genome) > convergence_count]
        if len(parents) > self.population_size:
            random_index = np.random.choice(len(parents), int(self.population_size/2), replace=False)
            return [parents[i] for i in random_index]
        return parents
    
    def hamming_distance(self, a, b):
        return np.sum(np.array(a) != np.array(b))
    
    def half_uniform_crossover(self, parent1, parent2):
        parent1 = np.array(parent1)
        parent2 = np.array(parent2)
        differing_indices = np.where(parent1 != parent2)[0]
        num_swaps = len(differing_indices) // 2
        swap_indices = np.random.choice(differing_indices, num_swaps, replace=False)
        offspring1 = parent1.copy()
        offspring2 = parent2.copy()
        offspring1[swap_indices] = parent2[swap_indices]
        offspring2[swap_indices] = parent1[swap_indices]
        return offspring1, offspring2

    def crossover(self, parents, convergence_count):
        """
        Perform crossover only if hamming distance is greater than convergance count
        Crossover is HUX
        """
        offspring = []
        for p in parents:
            child1 = Individual(n_variables=self.n_var) 
            child2 = Individual(n_variables=self.n_var)
            # if self.hamming_distance(p[0].genome, p[1].genome) <= convergence_count:
            #      child1.genome, child1.fitness = np.copy(p[0].genome), np.copy(p[0].fitness)
            #      child2.genome, child2.fitness = np.copy(p[1].genome), np.copy(p[1].fitness)
            # else:
            #     child1.genome, child2.genome = self.half_uniform_crossover(p[0].genome, p[1].genome)
            #     _, child1.fitness, _ = self.evaluate(child1, self.x_train, self.y_train)
            #     _, child2.fitness, _ = self.evaluate(child2, self.x_train, self.y_train)
            child1.genome, child2.genome = self.half_uniform_crossover(p[0].genome, p[1].genome)
            _, child1.fitness, _ = self.evaluate(child1, self.x_train, self.y_train)
            _, child2.fitness, _ = self.evaluate(child2, self.x_train, self.y_train)
            offspring.extend([child1, child2])
            if self.current_evaluations >= self.max_evaluations:
                break
        return offspring

    def elitism(self, population, offspring):
        offspring.extend(population)
        sorted_pop = sorted(offspring, key=lambda x:(x.fitness[0], x.fitness[1]))
        return sorted_pop[:self.population_size]

    def modified(self, population, new_population, parents):
        # population = sorted(population, key=lambda x:(x.fitness[0], x.fitness[1]))
        # genomes_a = np.array([member.genome for member in population])
        # genomes_b = np.array([member.genome for member in new_population])
        # if np.array_equal(genomes_a, genomes_b):
        #      return False
        # return True
        return True if len(parents) else False
    
    def mutate(self, member):
        random_array = np.random.rand(self.n_var)
        member.genome = [1-x if random_array[i]<self.cataclysmic_muation_prob else x for i, x in enumerate(member.genome)]
        return member
         
    def restart(self, population):
        sorted_pop = sorted(population, key=lambda x:(x.fitness[0], x.fitness[1]))
        for i in range(self.population_size):
             if i > (self.population_size * self.preserved_population):
                    sorted_pop[i] = self.mutate(sorted_pop[i])
                    _, sorted_pop[i].fitness, _ = self.evaluate(sorted_pop[i], self.x_train, self.y_train)
        return sorted_pop

    def choose_solution(self, population, x, y) -> Individual:
        solution = copy.deepcopy(self.best_solution)
        for member in population:
            member.accuracy, member.fitness, member.g_mean = self.evaluate(member, x, y)
            if member.fitness[0] < solution.fitness[0]:
                solution = member.copy()
            elif member.fitness[0] == solution.fitness[0] and member.g_mean > solution.g_mean:
                solution = member.copy()
        return solution

    def evaluate(self, member, x, y, n_folds = 3):
        self.current_evaluations += 1
        features_selected = [i for i, xi in enumerate(member.genome) if xi == 1]
        features_selected = torch.tensor(features_selected)
        if features_selected.shape[0] < 1:
            return None, np.array([math.inf, math.inf]), 0
        x_prima = x.index_select(1, features_selected)
        min_class = min(int(torch.sum(y)), y.shape[0] - int(torch.sum(y)))
        k = min(min_class, n_folds)
        loss = np.zeros(k)
        acc = np.zeros(k)
        g_mean = np.zeros(k)
        skf = StratifiedKFold(n_splits=k)
        for i, (train_index, test_index) in enumerate(skf.split(x_prima, y)):
            model = KNeighborsClassifier(n_neighbors=2)
            model.fit(x_prima[train_index], y[train_index].ravel())
            y_predict = torch.tensor(model.predict(x_prima[test_index]))
            y_real = y[test_index].squeeze(dim=1)
            loss[i] = torch_fitness_function(y_real, y_predict) 
            acc[i] = (y_real == torch.round(y_predict)).type(torch.float32).mean()
            g_mean[i] = geometric_mean(y_real, y_predict)
        return acc.mean(), [loss.mean(), features_selected.shape[0]], g_mean.mean()
  
    def final_evaluate(self, member, x_train, y_train, x_test, y_test):
        features_selected = [i for i, xi in enumerate(member.genome) if xi == 1]
        features_selected = torch.tensor(features_selected)
        if features_selected.shape[0] < 1:
            return None, np.array([math.inf, math.inf]), 0
        x_train_prima = x_train.index_select(1, features_selected)
        x_test_prima = x_test.index_select(1, features_selected)
        model = KNeighborsClassifier(n_neighbors=2)
        model.fit(x_train_prima, y_train.ravel())
        y_predict = torch.tensor(model.predict(x_test_prima))
        y_real = y_test.squeeze(dim=1)
        loss = torch_fitness_function(y_real, y_predict) 
        acc = (y_real == torch.round(y_predict)).type(torch.float32).mean()
        g_mean = geometric_mean(y_real, y_predict)
        return acc, [loss, features_selected.shape[0]], g_mean

    def run(self, seed = None, debug = False):
        self.archive = []
        if seed is not None:
            set_seed(seed)
        convergence_count = self.initialize_convergence_count()
        population = self.initialize_population()
        iteration = 0
        while self.current_evaluations < self.max_evaluations:
            iteration += 1
            if self.ending_condition(iteration):
                break
            parents = self.select_parents(population, convergence_count)
            offspring = self.crossover(parents, convergence_count)
            new_population = self.elitism(population, offspring)
            # if (iteration+1) % (int(self.max_iterations / 100)) == 0:
            if debug:
                mean_loss = np.mean([x.fitness[0] for x in new_population])
                mean_fs = np.mean([x.fitness[1] for x in new_population])
                print(f'Iteration {iteration}, Evaluations {self.current_evaluations}, Convergence {convergence_count}, mean loss {mean_loss}, mean fs {mean_fs}')
            if not self.modified(population, new_population, parents):
                # distance = [self.hamming_distance(parent1.genome, parent2.genome) for parent1, parent2 in itertools.combinations(population, 2)]
                # convergence_count = max(distance)
                convergence_count = convergence_count - 1
                if convergence_count <= -self.convergence_value_k:
                    self.archive = self.elitism(self.archive, new_population)
                    new_population = self.restart(population)
                    convergence_count = self.initialize_convergence_count()
            # population = copy.deepcopy(new_population)
            population = list(new_population)
        self.archive = self.elitism(self.archive, new_population)
        n_objectives = len(population[0].fitness)
        self.best_solution = Individual()
        self.best_solution.fitness = np.ones(n_objectives) * math.inf
        self.best_solution = self.choose_solution(self.archive, self.x_train, self.y_train)



class MOCHC(CHC):

    def __init__(self, problem, params):
        super().__init__(problem, params)

    def sort_by_crowding_distance(self, population):
        if len(population) < 2:
            return list(population)
        fitness_matrix = [member.fitness for member in population]
        crowding_distance_vector = pygmo.crowding_distance(fitness_matrix)
        crowding_distance_vector = np.nan_to_num(crowding_distance_vector, nan=0.0)
        for i in range(len(population)):
            population[i].crowding_distance = crowding_distance_vector[i]
        sorted_population = sorted(population, key = lambda x: -x.crowding_distance)
        crowding_distance_vector = [member.crowding_distance for member in population]
        return sorted_population

    def elitism(self, population, offspring):
        if len(offspring) == 0:
            return population
        offspring.extend(population)
        # fronts = non_dominated_sorting(offspring)
        points = [member.fitness for member in offspring]
        ndf, _, _, _ = pygmo.fast_non_dominated_sorting(points = points)
        fronts = [[] for _ in range(len(ndf))]
        for index_front, front in enumerate(ndf):
            for index_pop in front:
                fronts[index_front].append(offspring[index_pop])
        new_population = []
        for front in fronts:
            if len(new_population) + len(front) > self.population_size:
                n = self.population_size - len(new_population)
                sorted_front = self.sort_by_crowding_distance(front)
                new_population.extend(sorted_front[:n])
                break
            else:
                new_population.extend(front)
        return new_population
    
    def update_archive(self, population):
        self.archive = self.elitism(self.archive, copy.deepcopy(population))

    def restart(self, population):
        self.update_archive(population)
        # fronts = non_dominated_sorting(population)
        points = [member.fitness for member in population]
        ndf, _, _, _ = pygmo.fast_non_dominated_sorting(points = points)
        fronts = [[] for _ in range(len(ndf))]
        for index_front, front in enumerate(ndf):
            for index_pop in front:
                fronts[index_front].append(population[index_pop])
        sorted_pop = []
        for front in fronts:
            sorted_front = self.sort_by_crowding_distance(front)
            sorted_pop.extend(sorted_front)
        for i in range(self.population_size):
             if i > (self.population_size * self.preserved_population):
                    sorted_pop[i] = self.mutate(sorted_pop[i])
                    _, sorted_pop[i].fitness, _ = self.evaluate(sorted_pop[i], self.x_train, self.y_train)
        return sorted_pop



if __name__ == '__main__':

    seed = 0
    
    data = pd.read_csv("experiments_2024\\SFE Python Code\\colon.csv")
    data = data.values
    Input = np.asarray(data[:, 0:-1])
    Input = stats.zscore(Input)
    Target = np.asarray(data[:, -1])




    problem = {
        'x_train' : Input,
        'y_train' : Target,
        'x_test' : None,
        'y_test' : None
    }

    alg = MOCHC(problem)

    alg.run(seed)