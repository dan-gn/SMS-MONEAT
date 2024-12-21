import numpy as np
import pandas as pd
import os
import sys
import csv
from scipy.stats import ranksums
import scikit_posthocs as sp

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import N_EXPERIMENTS
from experiment_info import datasets

def csv2expdata(filename):
    metric = {}
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            metric[row[0]] = row[1:]
    return metric
 
# def csv2expdata2(filename):
#     metric = {}
#     with open(filename) as f:
#         reader = csv.reader(f)
#         for i, row in enumerate(reader):
#             metric[row[0]] = row[1:]
#     return metric 

metric = 'hv'

files = []
# files.append(f'final_exp/n3o/results_n3o_final2_full{metric}2_arch.csv')
# files.append(f'final_exp/sms_emoa/results_sms_emoa_final5x_full{metric}_train.csv')
# files.append(f'final_exp/sms_moneat/results_sms_moneat_final6_full{metric}2_train.csv')
# files.append(f'final_exp/sms_moneat/results_sms_moneat_final6_full{metric}2_arch.csv')

files.append(f'final_results_asc/sms_moneat/results_sms_moneat_final6_full_{metric}2_train.csv')
files.append(f'final_results_asc/sms_moneat/results_sms_moneat_final6_full_{metric}2_arch.csv')
files.append(f'final_results_asc/n3o/results_n3o_final2_full_{metric}2_arch.csv')
files.append(f'final_results_asc/sms_emoa/results_sms_emoa_final5x_full_{metric}_train.csv')
files.append(f'final_results_asc/mochc/results_mochc_final2024_full_ws_train_{metric}2.csv')
# files.append(f'final_results_asc/sfe/results_sfe_final2024_full_ws_train_{metric}.csv')
# files.append(f'final_results_asc/sfe_pso/results_sfe_pso_final2024_full_ws_train_{metric}.csv')

res = {}
for i, file in enumerate(files):
    res[file] = csv2expdata(file)
    

data = {}
# for ds in datasets:
#     data[ds] = {}
#     samples = [np.array(g_mean[f][ds], np.float32) for f in files]
#     print(ds)
#     print(np.mean(samples, axis=1))
#     data[ds]['g_mean'] = sp.posthoc_dunn(samples, p_adjust='bonferroni')
#     print(data[ds]['g_mean'])
#     samples = [np.array(fs[f][ds], np.float32) for f in files]
#     data[ds]['fs'] = sp.posthoc_dunn(samples, p_adjust='bonferroni')

for ds in datasets:
    data[ds] = {}
    samples = np.array([np.array(res[f][ds], np.float32) for f in files]).transpose()
    # print(ds)
    # print(np.mean(samples, axis=0))
    data[ds]['res'] = sp.posthoc_miller_friedman(samples)
    # print(data[ds]['res'] < 0.05)


condorcet = np.zeros((len(files), len(files)+1))
for ds in datasets:
    significance = data[ds]['res'] < 0.05
    significance.to_csv(f'final_results_asc/statistical_analysis/posthoc/{metric}/{ds}.csv')
    significance = significance.astype(float).to_numpy()
    for i in range(len(files)):
        for j in range(len(files)):
            if significance[i][j]:
                mean_a = np.array(res[files[i]][ds], np.float32).mean()
                mean_b = np.array(res[files[j]][ds], np.float32).mean()
                if metric in ['loss', 'fs']:
                    if mean_a < mean_b:
                        condorcet[i][j] += 1
                else:
                    if mean_a > mean_b:
                        condorcet[i][j] += 1
for i in range(len(files)):
    condorcet[i][len(files)] = np.sum(condorcet[i])




print(condorcet)
condorcet = pd.DataFrame(condorcet)
condorcet.to_csv(f'final_results_asc/statistical_analysis/condorcet/{metric}.csv')

