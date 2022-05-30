import random

from models.genotype import Genome

class Species:

	def __init__(self) -> None:
		self.member_index = []
		self.champion = None
		self.representant = None
		self.stagnant_generations = 0
		self.sum_shared_fitness = 0

	def add_member(self, index: int, member: Genome) -> None:
		self.member_index.append(index)
		if member.fitness > self.champion.fitness:
			self.champion = member.copy(with_phenotype=True)
			self.stagnant_generations = 0

	def choose_representant(self, population: list) -> None:
		self.representant = population[random.choice(self.member_index)].copy()
	
	def get_size(self) -> int:
		return len(self.member_index)
