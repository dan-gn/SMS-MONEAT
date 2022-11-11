import numpy as np
import os
import sys
import csv
from scipy.stats import ranksums

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
            if i > 0 and i <= len(datasets):
                g_mean[row[0]] = row[1:]
            elif i > len(datasets):
                fs[row[0]] = row[1:]
    return g_mean, fs
 
files = []
# files.append('results_sms_moneat_final6_full_val.csv')
# files.append('results_sms_moneat_final6_full_arch.csv')
# files.append('results_n3o_final2_full_val.csv')
files.append('results_sms_moneat_final6_full_svm_val.csv')
files.append('results_sms_moneat_final6_full_svm_arch.csv')
               
g_mean = {}
fs = {}
for file in files:
    g_mean[file], fs[file] = csv2expdata(file)

data = {}
for ds in datasets:
    data[ds] = {}
    sample1 = np.array(g_mean[files[0]][ds], np.float32)
    sample2 = np.array(g_mean[files[1]][ds], np.float32)
    _, data[ds]['g_mean'] = ranksums(sample1, sample2)
    # sample1 = np.array(fs[files[0]][ds], np.float32)
    # sample2 = np.array(fs[files[1]][ds], np.float32)
    # _, data[ds]['fs'] = ranksums(sample1, sample2)

with open('sms-moneat-val-svm_vs_sms-moneat-arch-svm.csv', 'w', newline='') as file:
    writer = csv.writer(file) 
    all_rows = []
    header = ['Dataset', 'gmean', 'fs']
    all_rows.append(header)
    for ds in datasets:
        # all_rows.append([ds, data[ds]['g_mean'], data[ds]['fs']])
        all_rows.append([ds, data[ds]['g_mean']])
    writer.writerows(all_rows)