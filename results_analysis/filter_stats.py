import numpy as np
import os
import sys
import pickle
import csv
from statistics import stdev

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import SEED, N_EXPERIMENTS, N_POPULATION
from experiment_info import datasets

alg = 'sms_moneat'
iterations = 18000

data = {}

for ds in datasets:
    results_path = os.getcwd() + f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt_final_6\\{ds}"
    kw_fs = [0] * N_EXPERIMENTS
    for k in range(N_EXPERIMENTS):
        results_filename = f"{ds}_MinMaxSc_{k}.pkl"
        with open(f'{results_path}/{results_filename}', 'rb') as f:
            results = pickle.load(f)
        model = results[2]['model']
        kw_fs[k] = model.x_train.shape[1]
    data[ds] = {'mean': np.mean(kw_fs), 'std': stdev(kw_fs)}
    print(f'Dataset: {ds}; FS: {np.mean(kw_fs)}, Std: {stdev(kw_fs)}')


with open('results_kw.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    all_rows = []
    header = ['Dataset', 'FS', 'Std']
    all_rows.append(header)
    all_rows.extend([[ds, data[ds]['mean'], data[ds]['std']] for ds in datasets])
    writer.writerows(all_rows)

