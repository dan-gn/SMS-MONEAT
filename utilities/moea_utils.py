import numpy as np

from utilities.hv import HyperVolume

def dominate(p, q):
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


def non_dominated_sorting(population):
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

def get_hv_contribution(front):
	reference = np.max(front, axis=0)
	hv = HyperVolume(reference)
	volume = hv.compute(front)
	volume_contribution = np.ones(front.shape[0]) * volume
	for i, p in enumerate(front):
		front_p = np.delete(front, i, axis=0)
		volume_i = hv.compute(front_p)
		volume_contribution[i] -= volume_i
	return volume_contribution

def check_repeated_rows(A):
	unq, count = np.unique(A, axis=0, return_counts=True)
	repeated = unq[count > 1]
	return True if repeated.shape[0] >= 1 else False

def choose_repeated_index(A):
	unq, count = np.unique(A, axis=0, return_counts=True)
	repeated = unq[count > 1]
	if repeated.shape[0] >= 1:
		row = np.random.randint(0, repeated.shape[0])
		row_idx = np.argwhere((A == repeated[row]).all(axis=1)).squeeze()
		chosen_row = np.random.choice(row_idx)
		return chosen_row
	else:
		return None

if __name__ == '__main__':
	A = np.array([[0,0], [1,1], [1,1]])
	print(choose_repeated_index(A))