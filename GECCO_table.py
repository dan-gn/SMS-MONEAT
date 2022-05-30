import numpy as np
import statistics
import pickle

from utilities.microarray_ds import MicroarrayDataset
from utilities.hv import HyperVolume
from utilities.moea_utils import non_dominated_sorting

def compute_hypervolume(population, reference, alpha=10):
	hv = HyperVolume(reference)
	front = non_dominated_sorting(population)
	fitness = np.array([list([f.fitness[0], f.fitness[1]/alpha]) for f in front[0]])
	unq, count = np.unique(fitness, axis=0, return_counts=True)
	return hv.compute(unq)

class TableRow:

	def __init__(self, name, dataset) -> None:
		self.name = name
		self.dataset=dataset
		self.get_dataset_info()
		self.get_results_info()
		
	def get_dataset_info(self):
		ds = MicroarrayDataset(f'./datasets/CUMIDA/{self.dataset}.arff')
		x, _ = ds.get_full_dataset()
		self.n_samples, self.n_features = x.shape[0], x.shape[1]

	def get_results_info(self, n_tests=20):
		n_population = 100
		n3o_iterations = 200
		sms_moneat_iterations = 18000

		reference = np.array([1.2, 1.2])	

		gmean = np.zeros(n_tests)
		fs = np.zeros(n_tests)
		kw = np.zeros(n_tests)
		hv = np.zeros(n_tests)

		for i in range(n_tests):
			filename = f'{self.dataset}_{i}_MinMaxSc.pkl'
			with open(f'results-n3o-pop_{n_population}-it_{n3o_iterations}_2/{self.dataset}/{filename}', 'rb') as f:
				_, _, res = pickle.load(f)
				model = res['model']
				kw[i] = model.x_train.shape[1]
				_, _, gmean[i] = model.evaluate(model.best_solution_test, model.x_test, model.y_test)
				fs[i] = model.best_solution_test.selected_features.shape[0]
				for _, member in enumerate(model.population):
					_, member.fitess, _ = model.evaluate(member, model.x_test, model.y_test)
					member.fitness = np.array([100 - member.fitness, member.selected_features.shape[0]])
				hv[i] = compute_hypervolume(model.population, reference)
		self.kw_features = kw.mean()
		self.n3o_gmean, self.n3o_gmean_std = gmean.mean(), statistics.stdev(gmean)
		self.n3o_fs, self.n3o_fs_std = fs.mean(), statistics.stdev(fs)
		self.n3o_hv, self.n3o_hv_std = hv.mean(), statistics.stdev(hv)

		self.n3o_gmean_, self.n3o_fs_, self.n3o_hv_ = list(gmean), list(fs), list(hv)

		gmean_arch = np.zeros(n_tests)
		fs_arch = np.zeros(n_tests)
		hv_arch = np.zeros(n_tests)

		for i in range(n_tests):
			filename = f'{self.dataset}_{i}_MinMaxSc.pkl'
			with open(f'results-sms_neat-pop_{n_population}-it_{sms_moneat_iterations}_2/{self.dataset}/{filename}', 'rb') as f:
				_, _, res = pickle.load(f)
				model = res['model']
				kw[i] = model.x_train.shape[1]
				_, _, gmean[i] = model.evaluate(model.best_solution_test, model.x_test, model.y_test)
				fs[i] = model.best_solution_test.selected_features.shape[0]
				hv[i] = compute_hypervolume(model.population, reference)
				_, _, gmean_arch[i] = model.evaluate(model.best_solution_archive, model.x_test, model.y_test)
				fs_arch[i] = model.best_solution_archive.selected_features.shape[0]
				hv_arch[i] = compute_hypervolume(model.archive.get_full_population(), reference)
		self.sms_moneat_gmean, self.sms_moneat_gmean_std = gmean.mean(), statistics.stdev(gmean)
		self.sms_moneat_fs, self.sms_moneat_fs_std = fs.mean(), statistics.stdev(fs)
		self.sms_moneat_hv, self.sms_moneat_hv_std = hv.mean(), statistics.stdev(hv)
		self.archive_gmean, self.archive_gmean_std = gmean_arch.mean(), statistics.stdev(gmean_arch)
		self.archive_fs, self.archive_fs_std = fs_arch.mean(), statistics.stdev(fs_arch)
		self.archive_hv, self.archive_hv_std = hv_arch.mean(), statistics.stdev(hv_arch)

		self.sms_moneat_gmean_, self.sms_moneat_fs_, self.sms_moneat_hv_ = gmean, fs, hv
		self.archive_gmean_, self.archive_fs_, self.archive_hv_ = gmean_arch, fs_arch, hv_arch

	def display_data(self, with_hv=True):
		arrow = '$\\leftrightarrow$'
		ds_info = f'{self.name}\t '
		if with_hv:
			archive_info = f'{self.archive_gmean:.3f}({self.archive_gmean_std:.2f})\t& {self.archive_fs:.2f}({self.archive_fs_std:.2f})\t& {self.archive_hv:.3f}({self.archive_hv_std:.2f})'
			sms_moneat_info = f'{self.sms_moneat_gmean:.3f}({self.sms_moneat_gmean_std:.2f}){arrow}\t& {self.sms_moneat_fs:.2f}({self.sms_moneat_fs_std:.2f}){arrow}\t& {self.sms_moneat_hv:.3f}({self.archive_hv_std:.2f}){arrow}'
			n3o_info = f'{self.n3o_gmean:.3f}({self.n3o_gmean_std:.2f}){arrow}\t& {self.n3o_fs:.2f}({self.n3o_fs_std:.2f}){arrow}\t& {self.n3o_hv:.3f}({self.n3o_hv_std:.2f}){arrow}'
		else:
			archive_info = f'{self.archive_gmean:.4f}({self.archive_gmean_std:.2f})\t& {self.archive_fs:.2f}({self.archive_fs_std:.2f})'
			sms_moneat_info = f'{self.sms_moneat_gmean:.4f}({self.sms_moneat_gmean_std:.2f}){arrow}\t& {self.sms_moneat_fs:.2f}({self.sms_moneat_fs_std:.2f}){arrow}'
			n3o_info = f'{self.n3o_gmean:.4f}({self.n3o_gmean_std:.2f}){arrow}\t& {self.n3o_fs:.2f}({self.n3o_fs_std:.2f}){arrow}'
		print(f'{ds_info}\t& {archive_info}\t& {sms_moneat_info}\t& {n3o_info} \\\\')



class Table:

	def __init__(self, dataset_names):
		self.dataset_names = dataset_names
		self.rows = self.get_table()

		self.n3o_gmean = np.mean([x.n3o_gmean for x in self.rows.values()])
		self.n3o_gmean_std = statistics.stdev(np.array([x.n3o_gmean_ for x in self.rows.values()]).flatten())
		self.n3o_fs = np.mean([x.n3o_fs for x in self.rows.values()])
		self.n3o_fs_std = statistics.stdev(np.array([x.n3o_fs_ for x in self.rows.values()]).flatten())
		self.n3o_hv = np.mean([x.n3o_hv for x in self.rows.values()])
		self.n3o_hv_std = statistics.stdev(np.array([x.n3o_hv_ for x in self.rows.values()]).flatten())
		self.sms_moneat_gmean = np.mean([x.sms_moneat_gmean for x in self.rows.values()])
		self.sms_moneat_gmean_std = statistics.stdev(np.array([x.sms_moneat_gmean_ for x in self.rows.values()]).flatten())
		self.sms_moneat_fs = np.mean([x.sms_moneat_fs for x in self.rows.values()])
		self.sms_moneat_fs_std = statistics.stdev(np.array([x.sms_moneat_fs_ for x in self.rows.values()]).flatten())
		self.sms_moneat_hv = np.mean([x.sms_moneat_hv for x in self.rows.values()])
		self.sms_moneat_hv_std = statistics.stdev(np.array([x.sms_moneat_hv_ for x in self.rows.values()]).flatten())
		self.archive_gmean = np.mean([x.archive_gmean for x in self.rows.values()])
		self.archive_gmean_std = statistics.stdev(np.array([x.archive_gmean_ for x in self.rows.values()]).flatten())
		self.archive_fs = np.mean([x.archive_fs for x in self.rows.values()])
		self.archive_fs_std = statistics.stdev(np.array([x.archive_fs_ for x in self.rows.values()]).flatten())
		self.archive_hv = np.mean([x.archive_hv for x in self.rows.values()])
		self.archive_hv_std = statistics.stdev(np.array([x.archive_hv_ for x in self.rows.values()]).flatten())

	def get_table(self):
		rows = {}
		for name, dataset in self.dataset_names.items():
			rows[name] = TableRow(name, dataset)
		return rows

	def print_table(self, with_hv=True):
		for row in self.rows.values():
			row.display_data()
		ds_info = f'Average\t '
		if with_hv:
			archive_info = f'{self.archive_gmean:.3f}({self.archive_gmean_std:.2f})\t& {self.archive_fs:.2f}({self.archive_fs_std:.2f})\t& {self.archive_hv:.3f}({self.archive_hv_std:.2f})'
			sms_moneat_info = f'{self.sms_moneat_gmean:.3f}({self.sms_moneat_gmean_std:.2f})\t& {self.sms_moneat_fs:.2f}({self.sms_moneat_fs_std:.2f})\t& {self.sms_moneat_hv:.3f}({self.archive_hv_std:.2f})'
			n3o_info = f'{self.n3o_gmean:.3f}({self.n3o_gmean_std:.2f})\t& {self.n3o_fs:.2f}({self.n3o_fs_std:.2f})\t& {self.n3o_hv:.3f}({self.n3o_hv_std:.2f})'
		else:
			archive_info = f'{self.archive_gmean:.4f}({self.archive_gmean_std:.2f})\t& {self.archive_fs:.2f}({self.archive_fs_std:.2f})'
			sms_moneat_info = f'{self.sms_moneat_gmean:.4f}({self.sms_moneat_gmean_std:.2f})\t& {self.sms_moneat_fs:.2f}({self.sms_moneat_fs_std:.2f})'
			n3o_info = f'{self.n3o_gmean:.4f}({self.n3o_gmean_std:.2f})\t& {self.n3o_fs:.2f}({self.n3o_fs_std:.2f})'
		print(f'{ds_info}\t& {archive_info}\t& {sms_moneat_info}\t& {n3o_info} \\\\')

if __name__ == '__main__':

	dataset_names = {}
	dataset_names['GSE22820'] = 'Breast_GSE22820'
	dataset_names['GSE42568'] = 'Breast_GSE42568'
	dataset_names['GSE59246'] = 'Breast_GSE59246'
	dataset_names['GSE70947'] = 'Breast_GSE70947'
	dataset_names['Breast\\tnote{a}'] = 'breastCancer-full'
	dataset_names['GSE8671'] = 'Colorectal_GSE8671'
	dataset_names['GSE32323'] = 'Colorectal_GSE32323'
	dataset_names['GSE44076'] = 'Colorectal_GSE44076'
	dataset_names['GSE44861'] = 'Colorectal_GSE44861'
	dataset_names['GSE14317'] = 'Leukemia_GSE14317'
	dataset_names['GSE33615'] = 'Leukemia_GSE33615'
	dataset_names['GSE63270'] = 'Leukemia_GSE63270'
	dataset_names['GSE71935'] = 'Leukemia_GSE71935'
	dataset_names['Leukemia\\tnote{b}'] = 'ALL-AML-full'
	dataset_names['GSE14520_U133A'] = 'Liver_GSE14520_U133A'
	dataset_names['GSE62232'] = 'Liver_GSE62232'
	dataset_names['GSE6919_U95Av2'] = 'Prostate_GSE6919_U95Av2'
	dataset_names['GSE11682'] = 'Prostate_GSE11682'
	dataset_names['GSE46602'] = 'Prostate_GSE46602'
	dataset_names['Prostate\\tnote{c}'] = 'prostate_tumorVSNormal-full'

	tab = Table(dataset_names)
	tab.print_table()