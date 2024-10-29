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
from experiment_info import datasets, algorithms


data = {}
alg = 'sfe'
iterations = 6000

for ds in datasets:
    data[ds] = {}
    results_path = os.getcwd() + f"\\results\\{alg}-pop_1-it_{iterations}_seed0-exp2024\\{ds}"
    time = [0] * N_EXPERIMENTS
    loss = [0] * N_EXPERIMENTS
    fs = [0] * N_EXPERIMENTS
    g_mean = [0] * N_EXPERIMENTS
    for k in range(N_EXPERIMENTS):
        results_filename = f"{ds}_MinMaxSc_{k}.pkl"
        # print(ds, k)
        with open(f'{results_path}/{results_filename}', 'rb') as f:
            results = pickle.load(f)
        time[k] = results[2]['time']
        model = results[2]['model']
        _, fitness, g_mean[k] = model.final_evaluate(model.individual, model.x_train, model.y_train, model.x_test, model.y_test)
        loss[k] = fitness[0]
        fs[k] = fitness[1]
    data[ds]['time'] = np.mean(time)		
    data[ds]['loss'] = np.mean(loss)		
    data[ds]['fs'] = np.mean(fs)		
    data[ds]['g_mean'] = np.mean(g_mean)		
    print(f'Algorithm: {alg}; Dataset: {ds}; Time {np.mean(time)}; Loss {np.mean(loss)}; FS {np.mean(fs)}; Gmean {np.mean(g_mean)}')


