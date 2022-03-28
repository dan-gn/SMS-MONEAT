import numpy as np
import pickle
from scipy.stats import ranksums, mannwhitneyu
import statistics


n_tests = 20

"""
DATASETS
"""
ds_names = ['GSE22820', 'GSE42568', 'GSE59246', 'GSE70947', 'Van De Vijver et al. \cite{van2002gene\}', 'GSE8671', 'GSE32323']
ds_names.extend(['GSE44076', 'GSE44861', 'GSE14317', 'GSE33615', 'GSE63270', 'GSE71935', 'Golub et al. \cite{golub1999molecular\}'])
ds_names.extend(['GSE14520_U133A','GSE62232', 'GSE6919_U95Av2', 'GSE11682', 'GSE46602', 'Singh et al. \cite{singh2002gene\}'])

datasets = []

datasets.append('Breast_GSE22820')
datasets.append('Breast_GSE42568')
datasets.append('Breast_GSE59246')
datasets.append('Breast_GSE70947')
datasets.append('breastCancer-full')

datasets.append('Colorectal_GSE8671')
datasets.append('Colorectal_GSE32323')
datasets.append('Colorectal_GSE44076')
datasets.append('Colorectal_GSE44861')

datasets.append('Leukemia_GSE14317')
datasets.append('Leukemia_GSE33615')
datasets.append('Leukemia_GSE63270')
datasets.append('Leukemia_GSE71935')
datasets.append('ALL-AML-full')

datasets.append('Liver_GSE14520_U133A')
datasets.append('Liver_GSE62232')

datasets.append('Prostate_GSE6919_U95Av2')
datasets.append('Prostate_GSE11682')
datasets.append('Prostate_GSE46602')
datasets.append('prostate_tumorVSNormal-full')


reference = [1, 1]

# """
# N3O
# """

n_population = 100
n_iterations = 200
n3o_gmean = np.zeros((len(datasets), n_tests))
n3o_fs = np.zeros((len(datasets), n_tests))

fitness = np.zeros((n_population, 2))

for i, ds in enumerate(datasets):
	for test in range(n_tests):
		filename = f'{ds}_{test}_MinMaxSc.pkl'
		# print(f'Reading test: {filename}')
		with open(f'results-n3o-pop_{n_population}-it_{n_iterations}_2/{ds}/{filename}', 'rb') as f:
			problem, params, res = pickle.load(f)
		model = res['model']
		_, _, n3o_gmean[i, test] = model.evaluate(model.best_solution_test, model.x_test, model.y_test)
		n3o_fs[i, test] = model.best_solution_test.selected_features.shape[0]

		for i, member in enumerate(model.population):
			_, fitness[i, :], _ = model.evaluate(member, model.x_test, model.y_test)
			
		
	# fs_min = n3o_fs[i, :].min()
	# fs_max = n3o_fs[i, :].max()
	# fs_mean = n3o_fs[i, :].mean()
	# fs_std = statistics.stdev(n3o_fs[i, :])
	# gmean_min = n3o_gmean[i, :].min()
	# gmean_max = n3o_gmean[i, :].max()
	# gmean_mean = n3o_gmean[i, :].mean()
	# gmean_std = statistics.stdev(n3o_gmean[i, :])
	# print(f'{ds_names[i]} & {fs_min:.2f} & {fs_max:.2f} & {fs_mean:.2f} $\pm$ {fs_std:.2f} & {gmean_min:.4f} & {gmean_max:.4f} & {gmean_mean:.4f} $\pm$ {gmean_std:.4f} \\\\')





# """
# SMS-NEAT
# """

n_population = 100
n_iterations = 18000
sms_neat_gmean = np.zeros((len(datasets), n_tests))
sms_neat_fs = np.zeros((len(datasets), n_tests))
kw = np.zeros((len(datasets), n_tests))

for i, ds in enumerate(datasets):
	for test in range(n_tests):
		filename = f'{ds}_{test}_MinMaxSc.pkl'
		# print(f'Reading test: {filename}')
		with open(f'results-sms_neat-pop_{n_population}-it_{n_iterations}_2/{ds}/{filename}', 'rb') as f:
			problem, params, res = pickle.load(f)
		model = res['model']
		_, _, sms_neat_gmean[i, test] = model.evaluate(model.best_solution_test, model.x_test, model.y_test)
		sms_neat_fs[i, test] = model.best_solution_test.selected_features.shape[0]
		kw[i, test] = model.x_test.shape[1]

archive_gmean = np.zeros((len(datasets), n_tests))
archive_fs = np.zeros((len(datasets), n_tests))

for i, ds in enumerate(datasets):
	for test in range(n_tests):
		filename = f'{ds}_{test}_MinMaxSc.pkl'
		# print(f'Reading test: {filename}')
		with open(f'results-sms_neat-pop_{n_population}-it_{n_iterations}_2/{ds}/{filename}', 'rb') as f:
			problem, params, res = pickle.load(f)
		model = res['model']
		_, _, archive_gmean[i, test] = model.evaluate(model.best_solution_archive, model.x_test, model.y_test)
		archive_fs[i, test] = model.best_solution_archive.selected_features.shape[0]


def wilcoxon_rank_sum(a, b):
	_, pvalue =  mannwhitneyu(a, b, alternative="two-sided")
	if pvalue < 0.05:
		return False
	return True

# for i, ds in enumerate(datasets):
# 	print(f'{ds_names[i]} = {wilcoxon_rank_sum(archive_fs[i, :], sms_neat_fs[i, :])}, {wilcoxon_rank_sum(archive_fs[i, :], n3o_fs[i, :])}, {wilcoxon_rank_sum(sms_neat_fs[i, :], n3o_fs[i, :])}')


# for i, ds in enumerate(datasets):
# 	moneatP_gmean = archive_gmean[i, :].mean()
# 	moneatP_gmean_std = statistics.stdev(archive_gmean[i, :])
# 	moneatP_fs = archive_fs[i, :].mean()
# 	moneatP_fs_std = statistics.stdev(archive_fs[i, :])
# 	moneatQ_gmean = sms_neat_gmean[i, :].mean()
# 	moneatQ_gmean_std = statistics.stdev(sms_neat_gmean[i, :])
# 	moneatQ_fs = sms_neat_fs[i, :].mean()
# 	moneatQ_fs_std = statistics.stdev(sms_neat_fs[i, :])
# 	n3o_gmean_a = n3o_gmean[i, :].mean()
# 	n3o_gmean_std = statistics.stdev(n3o_gmean[i, :])
# 	n3o_fs_a = n3o_fs[i, :].mean()
# 	n3o_fs_std = statistics.stdev(n3o_fs[i, :])

# 	print(f'{ds_names[i]} & {moneatP_gmean:.4f} ({moneatP_gmean_std:.4f}) & {moneatP_fs:.2f} ({moneatP_fs_std:.2f}) \leftrightarrow & {moneatQ_gmean:.4f} ({moneatQ_gmean_std:.4f}) & {moneatQ_fs:.2f} ({moneatQ_fs_std:.2f}) & {n3o_gmean_a:.4f} ({n3o_gmean_std:.4f}) \leftrightarrow & {n3o_fs_a:.2f} ({n3o_fs_std:.2f}) \\\\')


	# fs_min = sms_neat_fs[i, :].min()
	# fs_max = sms_neat_fs[i, :].max()
	# fs_mean = sms_neat_fs[i, :].mean()
	# fs_std = statistics.stdev(sms_neat_fs[i, :])
	# gmean_min = sms_neat_gmean[i, :].min()
	# gmean_max = sms_neat_gmean[i, :].max()
	# gmean_mean = sms_neat_gmean[i, :].mean()
	# gmean_std = statistics.stdev(sms_neat_gmean[i, :])
	# print(f'{kw[i, :].mean():.2f}')
	# print(f'{ds_names[i]} & {fs_min:.2f} & {fs_max:.2f} & {fs_mean:.2f} $\pm$ {fs_std:.2f} & {gmean_min:.4f} & {gmean_max:.4f} & {gmean_mean:.4f} $\pm$ {gmean_std:.4f} \\\\')


# print(n3o_fs.mean())
# print(n3o_gmean.mean())
# print(sms_neat_fs.mean())
# print(sms_neat_gmean.mean())

# """ 
# WILCOXON RANK SUM TEST
# """

# sample1 = n3o_gmean.mean(axis=1)
# sample2 = sms_neat_gmean.mean(axis=1)

# print(sample1)
# print(sample2)
# print(ranksums(sample1, sample2))


# sample1 = n3o_fs.mean(axis=1)
# sample2 = sms_neat_fs.mean(axis=1)

# print(sample1)
# print(sample2)
# print(ranksums(sample1, sample2))