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
 
files = []
files.append('results_sms_moneat_final6_full_svm_val.csv')
files.append('results_sms_moneat_final6_full_svm_arch.csv')
files.append('results_n3o_final2_full_svm_arch.csv')
files.append('results_sms_emoa_final4_full_svm_train.csv')

g_mean = {}
fs = {}
for file in files:
    g_mean[file], fs[file] = csv2expdata(file)

data = {}
for ds in datasets:
    data[ds] = {}
    samples = [np.array(g_mean[f][ds], np.float32) for f in files]
    data[ds]['g_mean'] = sp.posthoc_dunn(samples, p_adjust='bonferroni')
    # samples = [np.array(fs[f][ds], np.float32) for f in files]
    # data[ds]['fs'] = sp.posthoc_dunn(samples, p_adjust='bonferroni')

for ds in datasets:
    significance = data[ds]['g_mean'] < 0.05
    significance.to_csv(f'kw_svm/gmean/{ds}.csv')
    # significance = data[ds]['fs'] < 0.05
    # significance.to_csv(f'kw/fs/{ds}_fs.csv')
