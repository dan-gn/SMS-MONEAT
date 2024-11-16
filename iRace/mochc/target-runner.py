import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def log_file_call(file_name="file.txt"):
    # log_file = os.path.abspath(os.getcwd()) + "irace_mochc_log.txt"
	log_file = "C:/Users/23252359/Documents/SMS-MONEAT/iRace/mochc/irace_counter.txt"

	# Initialize count
	count = 0
	if os.path.exists(log_file):
		with open(log_file, "r") as log:
			count = int(log.read())

	# Increment count and write back to log
	count += 1
	with open(log_file, "w") as log:
		log.write(str(count))

log_file_call('target-runnet.py')

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", FutureWarning)

import sys
# sys.path.insert(0, '../..')
import os
os.environ["OMP_NUM_THREADS"] = "1"
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
sys.path.append(parent)

from utilities.microarray_ds import MicroarrayDataset
from utilities.stats_utils import KruskalWallisFilter
from pygmo import hypervolume, fast_non_dominated_sorting
# from utilities.moea_utils import non_dominated_sorting
from algorithms.chc import MOCHC

configuration_id = sys.argv[1]
instance_id = sys.argv[2]
seed = int(sys.argv[3])
instance = sys.argv[4]
irace_params = sys.argv[5:]
INITIAL_CONVERGENCE_COUNT = float(irace_params[1])
MUTATION_PROB = float(irace_params[3])

N_POPULATION = 100
N_ITERATIONS = 6000

# instance = "C:/Users/23252359/Documents/SMS-MONEAT/datasets/CUMIDA/Leukemia_GSE14317"
# seed = 1234567
# INITIAL_CONVERGENCE_COUNT = 0.02
# MUTATION_PROB = 0.35

dataset = instance.split('/')[-1]
trn_size = 0.70
debug = False


params = {
	'n_population' : int(N_POPULATION), 
	'max_iterations' : N_ITERATIONS,
	'initial_convergence_count' : INITIAL_CONVERGENCE_COUNT,
	'mutation_prob' : MUTATION_PROB,
}


# Read microarray dataset
import os
ds = MicroarrayDataset(f'{os.path.abspath(os.getcwd())}/../../datasets/CUMIDA/{dataset}.arff')
# ds = MicroarrayDataset(f'{os.path.abspath(os.getcwd())}\\datasets\\CUMIDA\\{dataset}.arff')
x, y = ds.get_full_dataset()

# Split dataset into training and testing dataset
x_train, x_test, y_train, y_test = ds.split(trn_size=trn_size, random_state=seed)

# Statistical Filtering
filter = KruskalWallisFilter(threshold=0.01)
x_train, features_selected = filter.fit_transform(x_train, y_train)
x_test, _ = filter.transform(x_test)

# Re-Scale datasets
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)	

y_train = np.expand_dims(y_train, axis=1)
y_test = np.expand_dims(y_test, axis=1)

x_train = torch.from_numpy(x_train).type(torch.float32)
y_train = torch.from_numpy(y_train).type(torch.float32)
x_test = torch.from_numpy(x_test).type(torch.float32)
y_test = torch.from_numpy(y_test).type(torch.float32)

problem = {
	'x_train' : x_train,
	'y_train' : y_train,
	'x_val' : x_test,
	'y_val' : y_test,
	'x_test' : x_test,
	'y_test' : y_test,
	'kw_htest_pvalue' : filter.p_value,
	'labels' : ds.get_labels()
}


"""
TRAIN MODEL
"""
# Training the model
model = MOCHC(problem, params)
model.run(seed, debug=False)

A = 60
B = 100
reference = np.array([A, B])
alpha = 10
for i, member in enumerate(model.archive):
	_, model.archive[i].fitness, _ = model.evaluate(member, model.x_test, model.y_test)

points = [member.fitness for member in model.archive]
ndf, _, _, _ = fast_non_dominated_sorting(points = points)
fronts = [[] for _ in range(len(ndf))]
for index_front, front in enumerate(ndf):
	for index_pop in front:
		fronts[index_front].append(model.archive[index_pop])

fitness = np.array([[f.fitness[0], f.fitness[1]/alpha] for f in fronts[0] if (f.fitness[0] <= A and f.fitness[1] <= alpha*B)])

if len(fitness) < 1:
	target = 1
	print(target)
else:
	unq, count = np.unique(fitness, axis=0, return_counts=True)
	hv = hypervolume(unq)
	target = -hv.compute(reference)
	print(target)