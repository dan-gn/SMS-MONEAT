import numpy as np
import pygmo 

import sys
import os

from typing import List

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from utilities.hv import HyperVolume
from models.genotype import MultiObjectiveGenome as Genome

def dominate(p: Genome, q: Genome):
	fp, fq = p.fitness, q.fitness
	same = True		# Flag to check that they are not equal solutions
	for i, fpi in enumerate(fp):
		# Check that if fi(p) <= fi(q)
		if fpi > fq[i]:
			return False
		# Check that at least one element of f(p) != f(q)
		if fpi != fq[i]:
			same = False
	return False if same else True

def non_dominated_sorting(population: List[Genome]):
	# Fronts
	front = [[]]
	i = 0	# Front counter
	# np -> how many individuals dominate p in the population
	n = np.zeros(len(population))
	# Get Sp and np for each individual 'p' in population
	for j, p in enumerate(population):
		# Sp -> individuals dominated by p in the population
		p.s = []
		# Iterate for each individual 'q' different from 'p'
		for k, q in enumerate(population):
			if j != k:
				# if 'p' dominates 'q', add 'q' to Sp
				if dominate(p, q):
					p.s.append(k)
				# if 'q' domiantes 'p', add 1 to np
				elif dominate(q, p):
					n[j] += 1
		# if np is not dominated by any other individual in the population, then p belongs to the first front
		if n[j] == 0:
			p.rank = i
			front[i].append(p)
	# Loop for filling the fronts
	while front[i]:
		# Empty array for the next front 
		next_front = []
		# For each individual 'p' of the current front
		for p in front[i]:
			# For each individual 'q' dominated by 'p'
			for q in p.s:
				n[q] -= 1
				# if no other individual than 'p' dominated 'q'
				if n[q] == 0:
					# it belongs to the next front
					population[q].rank = i + 1
					next_front.append(population[q])
		# Continue with next front
		i += 1
		front.append(list(next_front))
	front.pop()
	return front

def non_dominated_sorting_2(population: List[Genome]):
	# Fronts
	front = [[]]
	i = 0	# Front counter
	# Get Sp and np for each individual 'p' in population
	for j, p in enumerate(population):
		# Sp -> individuals dominated by p in the population
		p.dominates_to = []
		# Iterate for each individual 'q' different from 'p'
		for k, q in enumerate(population):
			if j != k:
				# if 'p' dominates 'q', add 'q' to Sp
				if dominate(p, q):
					p.dominates_to.append(q)
				# if 'q' domiantes 'p', add 1 to np
				elif dominate(q, p):
					p.n_dominated_by += 1
		# if np is not dominated by any other individual in the population, then p belongs to the first front
		p.n_temp = p.n_dominated_by
		if p.n_dominated_by == 0:
			p.rank = i
			front[i].append(p)
	# Loop for filling the fronts
	while front[i]:
		# Empty array for the next front 
		next_front = []
		# For each individual 'p' of the current front
		for p in front[i]:
			# For each individual 'q' dominated by 'p'
			for q in p.dominates_to:
				q.n_temp -= 1
				# if no other individual than 'p' dominated 'q'
				if q.n_temp == 0:
					# it belongs to the next front
					q.rank = i + 1
					next_front.append(q)
		# Continue with next front
		i += 1
		front.append(list(next_front))
	front.pop()
	return front

def add_genome_nds(population: List[Genome], genome: Genome):
	genome.dominated_to = []
	genome.n_dominated_by = 0
	for member in population:
		if dominate(genome, member):
			genome.dominates_to.append(member)
			member.n_dominated_by += 1
		elif dominate(member, genome):
			member.dominates_to.append(genome)
			genome.n_dominated_by += 1
	if genome.n_dominated_by == 0:
		genome.rank = 0

def remove_genome_nds(population: List[Genome], genome: Genome):
	for member in population:
		if dominate(genome, member):
			member.n_dominated_by -= 1
		elif dominate(member, genome):
			member.dominates_to.remove(genome)


def create_fronts(population: List[Genome]):
	i = 0
	for member in population:
		member.n_temp = member.n_dominated_by
	front = [[member for member in population if member.n_dominated_by == 0]]
	# Loop for filling the fronts
	while front[i]:
		# Empty array for the next front 
		next_front = []
		# For each individual 'p' of the current front
		for p in front[i]:
			# For each individual 'q' dominated by 'p'
			for q in p.dominates_to:
				q.n_temp -= 1
				# if no other individual than 'p' dominated 'q'
				if q.n_temp == 0:
					# it belongs to the next front
					q.rank = i + 1
					next_front.append(q)
		# Continue with next front
		i += 1
		front.append(list(next_front))
	front.pop()
	return front


def get_hv_contribution(front: List[float], delta: float=0.1):
	# Find best for each objective
	best_single_objective = np.argmin(front, axis=0)
	# Define reference value
	reference = np.max(front, axis=0) + delta
	hv = HyperVolume(reference)
	# Compute hypervolume with all solutions
	volume = hv.compute(front)
	# Compute hypervolume contribution for each solution
	volume_contribution = np.ones(front.shape[0]) * volume
	for i, p in enumerate(front):
		front_p = np.delete(front, i, axis=0)
		volume_i = hv.compute(front_p)
		volume_contribution[i] -= volume_i
	# Keep best for each objective
	volume_contribution[best_single_objective] += max(volume_contribution)
	return volume_contribution

def compute_hv_contribution(front: List[float], delta: float=0.1):
	# Find best for each objective
	best_single_objective = np.argmin(front, axis=0)
	# Define reference value
	reference = np.max(front, axis=0) + delta
	hv = pygmo.hypervolume(front)
	total_hv = hv.compute(reference)
	hv_contribution = np.ones(front.shape[0]) * total_hv	
	for i in range(front.shape[0]):
		hv = pygmo.hypervolume([x for j, x in enumerate(front) if j!=i])
		hv_contribution -= hv.compute(reference)
	# Keep best for each objective
	hv_contribution[best_single_objective] += max(hv_contribution)
	return hv_contribution


def choose_min_hv_contribution(front: List[float], delta: float=0.1):
	volume_contribution = compute_hv_contribution(front, delta)
	min_contribution_index = np.argwhere(volume_contribution == np.min(volume_contribution))
	return np.random.choice(min_contribution_index.squeeze(axis=1))


if __name__ == '__main__':
	import time
	np.random.seed(0)

	n = 1000
	front = [[i, n-i] for i in range(n)]
	front = np.array(front)

	# volume_contribution = get_hv_contribution(front)
	# min_contribution_index = np.argwhere(volume_contribution == np.min(volume_contribution))

	start_time = time.time()
	print(choose_min_hv_contribution(front))
	print(time.time() - start_time)