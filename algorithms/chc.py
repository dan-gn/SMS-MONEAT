import numpy as np

class CHC:

    def __init__(self) -> None:
        self.population_size = 100
        self.cataclysmic_muation_prob = 0.35
        self.preserved_population = 0.05
        self.initial_convergence_count = 0.25
        self.convergence_value_k = 1
        self.max_iterations = 100
        # Crossover -> HUX
        # Parent selection -> Random with incest threshold
        # New generation selection -> Elitist selection
        # Ordering criterion -> Ranking and crowding distance

    def initialize_convergence_count(self):
        pass

    def initialize_population(self):
        pass

    def ending_condition(self):
        pass

    def select_parents(self):
        pass

    def crossover(self):
        pass

    def evaluate(self):
        pass

    def elitism(self):
        pass

    def modified(self):
        pass

    def restart(self):
        pass

    def run(self):
        convergence_count = self.initialize_convergence_count()
        population = self.initialize_population()
        for iteration in range(self.max_iterations):
            if self.ending_condition():
                break
            parents = self.select_parents(population, convergence_count)
            offspring = self.crossover(parents)
            self.evaluate(offspring)
            new_population = self.elitism(offspring, population)
            if not self.modified(population, new_population):
                convergence_count = convergence_count - 1
                if convergence_count <= -self.convergence_value_k:
                    new_population = self.restart(population)
                    convergence_count = self.initialize_convergence_count()
            population = np.copy(new_population)



class MOCHC(CHC):

    def __init__(self):
        super().__init__()

    def evaluate(self):
        pass

    def elitism(self):
        pass

    def restart(self):
        pass


