import random

class Species:

	def __init__(self):
		self.member_index = []
		self.champion = None
		self.representant = None
		self.stagnant_generations = 0
		self.sum_shared_fitness = 0

	def add_member(self, index, member):
		self.member_index.append(index)
		if member.fitness > self.champion.fitness:
			self.champion = member.copy(with_phenotype=True)
			self.stagnant_generations = 0

	def choose_representant(self, population):
		self.representant = population[random.choice(self.member_index)].copy()
	
	def get_size(self):
		return len(self.member_index)
