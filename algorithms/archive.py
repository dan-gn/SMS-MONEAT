import numpy as np
import torch
import random
from itertools import chain
import copy

from models.genotype import Genome
from utilities.moea_utils import choose_min_hv_contribution, non_dominated_sorting
from utilities.data_utils import choose_repeated_index

class Species:

	def __init__(self) -> None:
		self.members = []

	def add_member(self, new_member: Genome) -> None:
		self.members.append(new_member)

	def remove_member(self, member: Genome) -> None:
		self.members.remove(member)

	def get_random_member(self) -> Genome:
		return random.choice(self.members)

	def get_size(self) -> int:
		return len(self.members)

	def sort_members(self) -> None:
		self.members = sorted(self.members, key=lambda x: -x.fitness[0])


class SpeciesArchive:

	def __init__(self, max_size: int, objective_norm: np.array, population: list = None) -> None:
		self.max_size = max_size
		self.objective_norm = objective_norm
		self.current_size = 0
		self.archive = []
		if population is not None:
			self.add_population(population)

	def add_population(self, population: list) -> None:
		for member in population:
			self.add(member)

	def species_count(self) -> int:
		return len(self.archive)

	def compare(self, a : Genome, b : Genome) -> bool:
		fs1, fs2 = a.selected_features, b.selected_features
		return torch.equal(fs1, fs2)

	def check_overflow(self) -> bool:
		return True if self.current_size > self.max_size else False

	def choose_element_to_remove(self, population: list) -> Genome:
		front = non_dominated_sorting(population)
		if len(front[-1]) == 1:
			remove_index = 0
		elif len(front[-1]) == 2:
			remove_index =  0 if front[-1][0].fitness[0] <= front[-1][1].fitness[0] else 1
		elif len(front[-1]) == 3:
			front_fitness = np.array([list(p.fitness) for p in front[-1]])
			remove_index = 3 - np.argmin(front_fitness, axis=0).sum()
		else:
			front_fitness = np.array([list(p.fitness) for p in front[-1]])
			remove_index, _ = choose_repeated_index(front_fitness)
			if remove_index is None:
				front_fitness *= self.objective_norm # Normalize objective
				remove_index = choose_min_hv_contribution(front_fitness)
		return front[-1][remove_index]

	def get_full_population(self) -> list:
		return list(chain.from_iterable([species.members for species in self.archive]))

	def get_worst_of_each_species(self) -> list:
		return [species.members[-1] for species in self.archive]

	def reduce_archive(self, species: Species = None) -> None:
		if species is None:
			all_species = self.get_worst_of_each_species()
			x = self.choose_element_to_remove(all_species)
			for species in self.archive:
				if x in species.members:
					species.remove_member(x)
					break
		else:
			species.members.pop()
		if species.get_size() == 0:
			self.archive.remove(species)
		self.current_size -= 1

	def add(self, genome: Genome) -> None:
		new_member = copy.deepcopy(genome)
		# Set flag to know if new member has been already added to any existing species to False
		added = False
		self.current_size += 1
		# Compare new member with existing species
		for species in self.archive:
			# Add new member to species which belongs to
			if self.compare(species.get_random_member(), new_member):
				species.add_member(new_member)
				species.sort_members()
				added = True
				break
		# If new member does not belong to any existing species create a new one
		if not added:
			species = Species()
			species.add_member(new_member)
			self.archive.append(species)
		# Check for archive overflow
		if self.check_overflow():
			if species.get_size() > 5:
				self.reduce_archive(species)
			else:
				self.reduce_archive()
