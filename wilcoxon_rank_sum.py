import numpy as np
import pickle
from scipy.stats import ranksums


n_tests = 20

"""
DATASETS
"""
datasets = []

datasets.append('breastCancer-full')
datasets.append('ALL-AML-full')
datasets.append('prostate_tumorVSNormal-full')

datasets.append('Breast_GSE42568')
datasets.append('Colorectal_GSE8671')
datasets.append('Colorectal_GSE32323')
datasets.append('Colorectal_GSE44076')
datasets.append('Colorectal_GSE44861')
datasets.append('Leukemia_GSE14317')
datasets.append('Leukemia_GSE63270')
datasets.append('Leukemia_GSE71935')

"""
N3O
"""

n_population = 100
n_iterations = 100
n3o_gmean = np.zeros((len(datasets), n_tests))
n3o_fs = np.zeros((len(datasets), n_tests))

for i, ds in enumerate(datasets):
	for test in range(n_tests):
		filename = f'{ds}_{test}_MinMaxSc.pkl'
		# print(f'Reading test: {filename}')
		with open(f'results-n3o-pop_{n_population}-it_{n_iterations}/{ds}/{filename}', 'rb') as f:
			problem, params, res = pickle.load(f)
		model = res['model']
		_, _, n3o_gmean[i, test] = model.evaluate(model.best_solution_test, model.x_test, model.y_test)
		n3o_fs[i, test] = model.best_solution_test.selected_features.shape[0]


"""
SMS-NEAT
"""

n_population = 100
n_iterations = 9000
sms_neat_gmean = np.zeros((len(datasets), n_tests))
sms_neat_fs = np.zeros((len(datasets), n_tests))

for i, ds in enumerate(datasets):
	for test in range(n_tests):
		filename = f'{ds}_{test}_MinMaxSc.pkl'
		# print(f'Reading test: {filename}')
		with open(f'results-sms_neat-pop_{n_population}-it_{n_iterations}/{ds}/{filename}', 'rb') as f:
			problem, params, res = pickle.load(f)
		model = res['model']
		_, _, sms_neat_gmean[i, test] = model.evaluate(model.best_solution_test, model.x_test, model.y_test)
		sms_neat_fs[i, test] = model.best_solution_test.selected_features.shape[0]


""" 
WILCOXON RANK SUM TEST
"""

sample1 = n3o_gmean.mean(axis=1)
sample2 = sms_neat_gmean.mean(axis=1)

print(sample1)
print(sample2)
print(ranksums(sample1, sample2))


sample1 = n3o_fs.mean(axis=1)
sample2 = sms_neat_fs.mean(axis=1)

print(sample1)
print(sample2)
print(ranksums(sample1, sample2))