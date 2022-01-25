from utilities.ga_utils import simulated_binary_crossover, polynomial_mutation, tournament_selection

class SMS_EMOA:

	def __init__(self):
		self.n_iterations = 0
		self.population = []
		self.mutation_eta = 0

	def initialize_population(self):
		pass

	def select_parents(self):
		pass

	def crossover(self, parent1, parent2):
		return simulated_binary_crossover(parent1, parent2)

	def mutate(self, offspring):
		return polynomial_mutation(offspring, self.mutation_eta)

	def next_generation(self):
		parent1, parent2 = self.select_parents()
		offspring = self.crossover(parent1, parent2)
		offspring = self.mutate(offspring)
		return offspring

	def reduce_population(self):
		pass

	def run(self):
		self.initialize_population()
		for i in range(self.n_iterations):
			offspring = self.next_generation()
			self.population.extend(offspring)
			self.reduce_population()