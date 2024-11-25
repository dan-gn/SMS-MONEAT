import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def log_file_call(file_name="file.txt"):
    # log_file = os.path.abspath(os.getcwd()) + "irace_mochc_log.txt"
	log_file = "C:/Users/23252359/Documents/ASC/iRace/sfe_pso/irace_counter.txt"

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
# warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", FutureWarning)

import sys
# sys.path.insert(0, '../../')

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
parent = os.path.dirname(parent)
sys.path.append(parent)

from utilities.microarray_ds import MicroarrayDataset
from utilities.stats_utils import KruskalWallisFilter
from experiments_2024.sfe_pso import SFE_PSO

configuration_id = sys.argv[1]
instance_id = sys.argv[2]
seed = int(sys.argv[3])
instance = sys.argv[4]
irace_params = sys.argv[5:]
UR_MIN = int(irace_params[1])
UR_MAX = float(irace_params[3])
SN = int(irace_params[5])
W = float(irace_params[7])
C1 = float(irace_params[9])
C2 = float(irace_params[11])

N_POPULATION = 100
N_ITERATIONS = 6000

# instance = "C:/Users/23252359/Documents/SMS-MONEAT/datasets/CUMIDA/Leukemia_GSE14317"
# seed = 1234567
# UR_MAX = 0.2691
# SN = 4

dataset = instance.split('/')[-1]
trn_size = 0.70
debug = False


params = {
	'max_iterations' : N_ITERATIONS,
	'UR' : UR_MAX,
	'UR_max' : UR_MAX,
	'UR_min' : 1/10**UR_MIN,
	'SN' : SN,
	'w' : W,
	'c1' : C1,
	'c2' : C2
}

# Read microarray dataset
ds = MicroarrayDataset(f'{os.path.abspath(os.getcwd())}/../../datasets/CUMIDA/{dataset}.arff')
# ds = MicroarrayDataset(f'{os.path.abspath(os.getcwd())}\\..\\..\\datasets\\CUMIDA\\{dataset}.arff')
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
	'x_test' : x_test,
	'y_test' : y_test,
	'kw_htest_pvalue' : filter.p_value,
	'labels' : ds.get_labels()
}

"""
TRAIN MODEL
"""
# Training the model
model = SFE_PSO(problem, params)
model.run(seed, debug=False)

_, fitness, _ = model.final_evaluate(model.best_solution.best_position, model.x_train, model.y_train, model.x_test, model.y_test)
target = float(fitness[0])

# for i, member in enumerate(model.population):
#     model.population[i].accuracy, model.population[i].fitness, _ = model.evaluate(member.best_position, model.x_test, model.y_test)
    
# fitness = [x.fitness[0] for x in model.population if x.accuracy is not None]
# target = np.mean(fitness)
print(target)