
import numpy as np

from utilities.microarray_ds import MicroarrayDataset

datasets = []

# datasets.append('breastCancer-full')
# datasets.append('Breast_GSE22820')
# datasets.append('Breast_GSE42568')
# datasets.append('Breast_GSE59246')
# datasets.append('Breast_GSE70947')

# datasets.append('Colorectal_GSE8671')
# datasets.append('Colorectal_GSE32323')
# datasets.append('Colorectal_GSE44076')
# datasets.append('Colorectal_GSE44861')

# datasets.append('Leukemia_GSE14317')
datasets.append('Leukemia_GSE33615')
# datasets.append('Leukemia_GSE63270')
# datasets.append('Leukemia_GSE71935')
# datasets.append('ALL-AML-full')

# datasets.append('prostate_tumorVSNormal-full')
# datasets.append('Prostate_GSE11682')
# datasets.append('Prostate_GSE46602')
# datasets.append('Prostate_GSE6919_U95Av2')

# datasets.append('Liver_GSE14520_U133A')
# datasets.append('Liver_GSE50579')
# datasets.append('Liver_GSE62232')

if __name__ == '__main__':
	
	for filename in datasets:

		# Read microarray dataset
		print(f'Reading dataset: {filename}')
		ds = MicroarrayDataset(f'./datasets/CUMIDA/{filename}.arff')
		print(f'Dataset labels: {ds.get_labels()}')
		x, y = ds.get_full_dataset()
		print(f'Total samples = {x.shape[0]}, Total features = {x.shape[1]}')
		print(f'Proportion of classes = ({(y.shape[0]-np.sum(y))/y.shape[0]:.2f}, {(np.sum(y))/y.shape[0]:.2f})')
		print('\n')