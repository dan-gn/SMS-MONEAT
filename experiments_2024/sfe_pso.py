import numpy as np
import math
import copy

from SFE_2 import SFE

class Individual: # Particle

    def __init__(self, n_variables: int = 1, w = 1, c1 = 2, c2 = 1.5) -> None:
        self.n_variables = n_variables
        self.position = None	
        self.velocity = None
        self.fitness = None
        self.accuracy = None
        self.g_mean = None
        self.best_position = None
        self.best_fitness = math.inf
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def update_best_position(self):
        if self.fitness < self.best_fitness:
            self.best_position = self.position
            self.best_fitness = self.fitness

    def update_velocity(self, global_best):
        r1, r2 = np.random.rand(), np.random.rand()
        # Cognitive and social components based on XOR difference
        cognitive = self.c1 * r1 * (self.best_position ^ self.position).astype(int)
        social = self.c2 * r2 * (global_best.position ^ self.position).astype(int)
        # cognitive = self.c1 * r1 * (self.best_position - self.position)
        # social = self.c2 * r2 * (global_best.position - self.position)
        # Update velocity as a weighted combination
        self.velocity = self.w * self.velocity + cognitive + social
        # Apply sigmoid to velocity to interpret it as a probability
        self.velocity = 1 / (1 + np.exp(-self.velocity))


    def update_position(self):
        self.position = (np.random.rand(len(self.position)) < self.velocity).astype(int)

    def initialize(self):
        # self.position = np.zeros(self.n_variables)
        # index = np.random.randint(self.n_variables)
        # self.position[index] = 1
        # threshold = 0.01
        # self.position = np.random.rand(self.n_variables)
        # self.position = [0 if x >= threshold else 1 for x in self.position]
        self.position = np.random.randint(0, 2, self.n_variables)
        self.velocity = np.random.uniform(-1, 1, self.n_variables)
        self.best_position = self.position.copy()

    def copy(self):
        individual_copy = Individual()
        individual_copy.position = self.position.copy()
        individual_copy.best_position = self.best_position.copy()
        individual_copy.fitness = self.fitness
        individual_copy.best_fitness = self.best_fitness
        individual_copy.update_best_position()
        individual_copy.w = self.w
        individual_copy.c1 = self.c1
        individual_copy.c2 = self.c2
        return individual_copy


class SFE_PSO(SFE):

    def __init__(self, problem: dict, params: dict) -> None:
        super().__init__(problem, params)
        self.x_train_full = self.x_train.clone()
        self.x_test_full = self.x_test.clone()
        self.max_evaluations = self.max_iterations
        self.population_size = 100
        self.w = 1
        self.c1 = 2
        self.c2 = 1.5

    def initialize_population(self):
        population = [] 
        for i in range(self.population_size):
            member = Individual(n_variables=self.n_var)
            member.initialize()
            if i == 0:
                member.position = np.ones(self.n_var).astype(int)
            member.accuracy, fitness, member.g_mean = self.evaluate(member.position, self.x_train, self.y_train)
            member.fitness = fitness[0]
            member.update_best_position()
            self.current_evaluation += 1
            population.append(member)
            if self.current_evaluation >= self.max_evaluations:
                break
        return population
    
    def get_best_solution(self, population):
        sorted_pop = sorted(population, key=lambda x:(x.best_fitness, np.sum(x.best_position)))
        return sorted_pop[0].copy()

    def update_population(self, population):
        for i, particle in enumerate(population):
            population[i].update_velocity(self.best_solution)
            population[i].update_position()
            _, fitness, _ = self.evaluate(population[i].position, self.x_train, self.y_train)
            self.current_evaluation += 1
            population[i].fitness = fitness[0]
            population[i].update_best_position()
            if self.current_evaluation >= self.max_evaluations:
                break

    def run_pso(self, best_solution, current_evaluation):
        self.current_evaluation = current_evaluation
        self.x_train = self.x_train[:, best_solution.astype(bool)]
        self.x_test = self.x_test[:, best_solution.astype(bool)]
        self.n_var = self.x_train.shape[1]

        population = self.initialize_population()
        self.best_solution = self.get_best_solution(population)

        while self.current_evaluation < self.max_evaluations:
            self.update_population(population)
            self.best_solution = self.get_best_solution(population)
            # mean = np.mean([member.best_fitness for member in population])
            # print(f'Evaluation = {self.current_evaluation}, fitness = {self.best_solution.best_fitness}, FS = {np.sum(self.best_solution.best_position)}, Mean = {mean}')

        return population

    def choose_solution(self, population, x, y):
        solution = copy.deepcopy(self.best_solution)
        solution.update_best_position()
        solution.accuracy, solution.fitness, solution.g_mean = self.evaluate(solution.best_position, x, y)
        for member in population:
            member.accuracy, member.fitness, member.g_mean = self.evaluate(member.best_position, x, y)
            if member.fitness[0] < solution.fitness[0]:
                solution = copy.deepcopy(member)
            elif member.fitness[0] == solution.fitness[0] and member.g_mean > solution.g_mean:
                solution = copy.deepcopy(member)
        return solution
        
    def run(self, seed: int = None, debug: bool = False):
        np.random.seed(seed)
        fitness_vector = np.zeros(self.max_iterations)
        EFs = 1
    
        individual = np.random.randint(0, 2, np.size(self.x_train, 1))   # Initialize an Individual X
        # Fit_X = fit(self.x_train, self.y_train.squeeze(1), individual)                    # Calculate the Fitness of X
        _, fitness, _ = self.evaluate(individual, self.x_train, self.y_train)
        Fit_X = fitness[0]
        Nvar = np.size(self.x_train, 1)                         # Number of Features in Dataset

        fitness_vector[EFs - 1] = Fit_X

        pso_flag = False
        while (EFs <= self.max_iterations):
            new_individual = np.copy(individual)
            # Non-selection operation:

            U_Index = np.where(individual == 1)                      # Find Selected Features in X
            NUSF_X = np.size(U_Index, 1)                    # Number of Selected Features in X
            UN = math.ceil(self.UR*Nvar)                         # The Number of Features to Unselect: Eq(2)
            # SF=randperm(20,1)                             # The Number of Features to Unselect: Eq(4)
            # UN=ceil(rand*Nvar/SF);                        # The Number of Features to Unselect: Eq(4)
            K1 = np.random.randint(0, NUSF_X, UN)           # Generate UN random number between 1 to the number of slected features in X
            res = np.array([*set(K1)])
            res1 = np.array(res)
            K = U_Index[0][[res1]]                          # K=index(U)
            new_individual[K] = 0                                    # Set X_New (K)=0 


            # Selection operation:
            if np.sum(new_individual) == 0:
                S_Index = np.where(individual == 0)                  # Find non-selected Features in X
                NSF_X = np.size(S_Index, 1)                 # Number of non-selected Features in X
                SN = 1                                      # The Number of Features to Select
                K1 = np.random.randint(0, NSF_X, SN)        # Generate SN random number between 1 to the number of non-selected features in X
                res = np.array([*set(K1)])
                res1 = np.array(res)
                K = S_Index[0][[res1]]
                new_individual = np.copy(individual)
                new_individual[K] = 1                                # Set X_New (K)=1

            # print(Input, Target, X_New)
            # Fit_X_New = fit(self.x_train, self.y_train.squeeze(1), new_individual)             # Calculate the Fitness of X_New
            _, new_fitness, _ = self.evaluate(new_individual, self.x_train, self.y_train)
            Fit_X_New = new_fitness[0]
            # print(Fit_X_New)
            if Fit_X_New <= Fit_X:
                individual = np.copy(new_individual)
                Fit_X = Fit_X_New

            UR = (self.UR_Max-self.UR_Min)*((self.max_iterations-EFs)/self.max_iterations)+self.UR_Min  # Eq(3)
            
            fitness_vector[EFs] = Fit_X

            # Run PSO
            if EFs > 2000:
                if fitness_vector[EFs] == fitness_vector[EFs - 1000]:
                    self.population = self.run_pso(individual, EFs + 1)
                    self.best_solution = self.choose_solution(self.population, self.x_train, self.y_train)
                    pso_flag = True
                    break
            # if EFs % 200 == 0:
            #     print(f'Evaluation = {EFs}, fitness = {Fit_X}, FS = {np.sum(individual)}')

            # print('Iteration = {} :   Accuracy = {} :   Number of Selected Features= {} :  Run= {}'.format( EFs, Fit_X, np.sum(new_individual), self.Run))
            EFs = EFs+1

        if not pso_flag:
            self.best_solution = Individual()
            self.best_solution.position = np.copy(individual)
