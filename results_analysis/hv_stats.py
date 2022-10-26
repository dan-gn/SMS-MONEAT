import numpy as np
import os
import sys
import pickle
import csv
import pygmo

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import SEED, N_EXPERIMENTS, N_POPULATION
from experiment_info import datasets, algorithms
from utilities.choose_solutions import evaluate

norm_vector = np.array([1, 0.1])
reference = [15, 10]

data = {}

for i, alg in enumerate(algorithms):
    data[alg] = {}
    iterations = 200 if alg=='n3o' else 18000
    for ds in datasets:
        data[alg][ds] = {}
        results_path = os.getcwd() + f"\\results\\{alg}-pop_{N_POPULATION}-it_{iterations}_seed{SEED}-cv_hpt\\{ds}"
        hv_pop= [0] * N_EXPERIMENTS
        hv_archive = [0] * N_EXPERIMENTS
        for k in range(N_EXPERIMENTS):
            results_filename = f"{ds}_MinMaxSc_{k}.pkl"
            with open(f'{results_path}/{results_filename}', 'rb') as f:
                results = pickle.load(f)
            model = results[2]['model']
            for member in model.population:
                member.valid = True
                member.accuracy, member.fitness, member.g_mean = evaluate(member, model.x_test, model.y_test, True)
            for member in model.archive.get_full_population():
                member.valid = True
                member.accuracy, member.fitness, member.g_mean = evaluate(member, model.x_test, model.y_test, True)
            hv = pygmo.hypervolume([np.array(member.fitness)*norm_vector for member in model.population if member.accuracy is not None])
            hv_pop[k] = hv.compute(reference)
            hv = pygmo.hypervolume([np.array(member.fitness)*norm_vector for member in model.archive.get_full_population() if member.accuracy is not None])
            hv_archive[k] = hv.compute(reference)
        data[alg][ds]['hv_pop'] = np.mean(hv_pop)
        data[alg][ds]['hv_arch'] = np.mean(hv_archive)
        print(f'Algorithm: {alg}; Dataset: {ds}; Population HV {np.mean(hv_pop)}; Archive HV {np.mean(hv_archive)}')
            