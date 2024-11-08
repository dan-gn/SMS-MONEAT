import numpy as np

def hamming_distance(a, b):
    return np.sum(a != b)

a = np.array([1, 0, 0, 0])
b = np.array([0, 1, 0, 0])
c = np.array([0, 1, 1, 1])

print(hamming_distance(a, a))
print(hamming_distance(a, b))
print(hamming_distance(a, c))

def select_parents(self, population, convergence_count):
    parents = []
    for parent1 in population:
        for parent2 in population:
            if self.hamming_distance(parent1.genome, parent2.genome) > convergence_count:
                parents.append([parent1, parent2])
    if len(parents) > self.population_size:
        random_index = np.random.choice(len(parents), int(self.population_size/2), replace=False)
        return [parents[i] for i in random_index]
    return parents