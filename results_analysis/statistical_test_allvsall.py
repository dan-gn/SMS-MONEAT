import numpy as np
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
    g_mean = {}
    fs = {}
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 0 and i <= len(datasets) + 1:
                g_mean[row[0]] = row[1:]
            elif i > len(datasets) + 1:
                fs[row[0]] = row[1:]
    return g_mean, fs
 
def csv2expdata2(filename):
    g_mean = {}
    fs = {}
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 0 and i <= len(datasets):
                g_mean[row[0]] = row[1:]
            elif i > len(datasets) + 1:
                fs[row[0]] = row[1:]
    return g_mean, fs

files = []
files.append('results_n3o_final2_full_arch.csv')
files.append('results_sms_emoa_final5x_full_train.csv')
files.append('results_sms_moneat_final6_full_val.csv')
files.append('results_sms_moneat_final6_full_arch.csv')

g_mean = {}
fs = {}
for i, file in enumerate(files):
    g_mean[file], fs[file] = csv2expdata(file)
    

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
    samples = np.array([np.array(g_mean[f][ds], np.float32) for f in files]).transpose()
    print(ds)
    print(np.mean(samples, axis=0))
    data[ds]['g_mean'] = sp.posthoc_miller_friedman(samples)
    print(data[ds]['g_mean'] < 0.05)
    # samples = np.array([np.array(fs[f][ds], np.float32) for f in files]).transpose()
    # data[ds]['fs'] = sp.posthoc_miller_friedman(samples)


# for ds in datasets:
#     significance = data[ds]['g_mean'] < 0.05
#     significance.to_csv(f'kw_final_svm/gmean/{ds}.csv')
    # significance = data[ds]['fs'] < 0.05
    # significance.to_csv(f'kw_final_svm/fs/{ds}_fs.csv')
