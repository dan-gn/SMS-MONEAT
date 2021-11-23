from models.genotype import Genome
from models.ann_pytorch import Ann_PyTorch

class Individual:

	def __init__(self, genotype, phenotype):
		self.genotype = genotype
		self.phenotype = phenotype

	def copy(self):
		return Individual(self.genotype.copy(), self.phenotype.copy())

