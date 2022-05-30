import numpy as np
import pickle

from utilities.hv import HyperVolume
from utilities.moea_utils import non_dominated_sorting

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
# datasets.append('Breast_GSE59246')
# datasets.append('Breast_GSE70947')
# datasets.append('breastCancer-full')

# datasets.append('Colorectal_GSE8671')
# datasets.append('Colorectal_GSE32323')
# datasets.append('Colorectal_GSE44076')
# datasets.append('Colorectal_GSE44861')

# datasets.append('Leukemia_GSE14317')
# datasets.append('Leukemia_GSE33615')
# datasets.append('Leukemia_GSE63270')
# datasets.append('Leukemia_GSE71935')
# datasets.append('ALL-AML-full')

# datasets.append('Liver_GSE14520_U133A')
# datasets.append('Liver_GSE62232')

# datasets.append('Prostate_GSE6919_U95Av2')
# datasets.append('Prostate_GSE11682')
# datasets.append('Prostate_GSE46602')
# datasets.append('prostate_tumorVSNormal-full')

# output_filename = 'hv_results.pkl'
# with open(output_filename, 'rb') as f:
# 	hv_results = pickle.load(f)

# print(hv_results['sms_moneat_arch'])

alpha = 10

# reference = [6, 528]
reference = np.array([1.2, 1.2]) # 2
# reference = np.array([1, 12]) # 3
# reference = np.array([1.0, 2.5])	# 4
# reference = np.array([0.8, 5.6])	# 5
new_reference = np.zeros(2)

hv = HyperVolume(reference)

n3o_hv = np.zeros((len(datasets), n_tests))
sms_moneat_hv = np.zeros((len(datasets), n_tests))
archive_hv = np.zeros((len(datasets), n_tests))

n3o_loss = np.zeros((len(datasets), n_tests))
sms_moneat_loss = np.zeros((len(datasets), n_tests))
archive_loss = np.zeros((len(datasets), n_tests))

n3o_fs = np.zeros((len(datasets), n_tests))
sms_moneat_fs = np.zeros((len(datasets), n_tests))
archive_fs = np.zeros((len(datasets), n_tests))

# """
# N3O
# """

n_population = 100
n_iterations = 200

for i, ds in enumerate(datasets):
	print(i, ds)
	for test in range(n_tests):
		filename = f'{ds}_{test}_MinMaxSc.pkl'
		# print(f'Reading test: {filename}')
		with open(f'results-n3o-pop_{n_population}-it_{n_iterations}_2/{ds}/{filename}', 'rb') as f:
			problem, params, res = pickle.load(f)
		model = res['model']

		for j, member in enumerate(model.population):
			_, member.fitess, _ = model.evaluate(member, model.x_test, model.y_test)
			member.fitness = np.array([100 - member.fitness, member.selected_features.shape[0]])
			# reference[0] = member.fitness[0] if reference[0] < member.fitness[0] else reference[0]
			# reference[1] = member.fitness[1] if reference[1] < member.fitness[1] else reference[1]

		front = non_dominated_sorting(model.population)
		fitness = np.array([list([f.fitness[0], f.fitness[1]/alpha]) for f in front[0]])
		unq, count = np.unique(fitness, axis=0, return_counts=True)
		n3o_hv[i, test] = hv.compute(unq)

		n3o_loss[i, test], n3o_fs[i, test] = unq[:, 0].mean(), unq[:, 1].mean()

		max_unq = unq.max(axis=0)
		new_reference[0] = max_unq[0] if new_reference[0] < max_unq[0] else new_reference[0]
		new_reference[1] = max_unq[1] if new_reference[1] < max_unq[1] else new_reference[1]

	# print(ds, n3o_hv[i, :].mean())


n_population = 100
n_iterations = 18000

for i, ds in enumerate(datasets):
	print(i, ds)
	for test in range(n_tests):
		filename = f'{ds}_{test}_MinMaxSc.pkl'
		# print(f'Reading test: {filename}')
		with open(f'results-sms_neat-pop_{n_population}-it_{n_iterations}_2/{ds}/{filename}', 'rb') as f:
			problem, params, res = pickle.load(f)
		model = res['model']

		for j, member in enumerate(model.population):
			_, member.fitess, _ = model.evaluate(member, model.x_test, model.y_test)
			# reference[0] = member.fitness[0] if reference[0] < member.fitness[0] else reference[0]
			# reference[1] = member.fitness[1] if reference[1] < member.fitness[1] else reference[1]

		front = non_dominated_sorting(model.population)
		fitness = np.array([list([f.fitness[0], f.fitness[1]/alpha]) for f in front[0]])
		unq, count = np.unique(fitness, axis=0, return_counts=True)
		sms_moneat_hv[i, test] = hv.compute(unq)
		sms_moneat_loss[i, test], sms_moneat_fs[i, test] = unq[:, 0].mean(), unq[:, 1].mean()

		max_unq = unq.max(axis=0)
		new_reference[0] = max_unq[0] if new_reference[0] < max_unq[0] else new_reference[0]
		new_reference[1] = max_unq[1] if new_reference[1] < max_unq[1] else new_reference[1]


		archive_population = model.archive.get_full_population()
		for j, member in enumerate(archive_population):
			_, member.fitess, _ = model.evaluate(member, model.x_test, model.y_test)
			# reference[0] = member.fitness[0] if reference[0] < member.fitness[0] else reference[0]
			# reference[1] = member.fitness[1] if reference[1] < member.fitness[1] else reference[1]

		front = non_dominated_sorting(archive_population)
		fitness = np.array([list([f.fitness[0], f.fitness[1]/alpha]) for f in front[0]])
		unq, count = np.unique(fitness, axis=0, return_counts=True)
		archive_hv[i, test] = hv.compute(unq)
		archive_loss[i, test], archive_fs[i, test] = unq[:, 0].mean(), unq[:, 1].mean()

		max_unq = unq.max(axis=0)
		new_reference[0] = max_unq[0] if new_reference[0] < max_unq[0] else new_reference[0]
		new_reference[1] = max_unq[1] if new_reference[1] < max_unq[1] else new_reference[1]

	# print(ds, sms_moneat_hv[i, :].mean(), archive_hv[i, :].mean())

for i, ds in enumerate(datasets):
	print(ds, n3o_hv[i, :].mean(), sms_moneat_hv[i, :].mean(), archive_hv[i, :].mean())
	print(ds, n3o_loss[i, :].mean(), sms_moneat_loss[i, :].mean(), archive_loss[i, :].mean())
	print(ds, n3o_fs[i, :].mean(), sms_moneat_fs[i, :].mean(), archive_fs[i, :].mean())
	print('\n')

print(new_reference[0], new_reference[1])

output_filename = 'hv_results_4.pkl'

hv_results = {
	'n3o' : [n3o_hv, n3o_loss, n3o_fs],
	'sms_moneat' : [sms_moneat_hv, sms_moneat_loss, sms_moneat_fs],
	'sms_moneat_arch' : [archive_hv, archive_loss, archive_fs]
}

# with open(output_filename, 'wb') as f:
# 	pickle.dump(hv_results, f)