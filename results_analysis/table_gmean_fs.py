import numpy as np
import os
import sys
import csv
import re

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import datasets

def csv2expdata(filename, cols):
    data = {}
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 1:
                data[row[0]] = np.array(row[cols[0]:cols[1]], dtype=np.float32)
    return data

files = []
files.append('results_n3o_final2.csv')
files.append('results_sms-emoa_final4.csv')
files.append('results_sms-moneat_final6.csv')
files.append('results_sms-moneat_final6.csv')

cols = []
cols.append([15, 19])
cols.append([3, 7])
cols.append([7, 11])
cols.append([15, 19])

data = []
for f, c in zip(files, cols):
    data.append(csv2expdata(f, c))

rm_words = ['Breast_', 'Colorectal_', 'Leukemia_', 'Liver_', 'Prostate_']

with open('table_performance_comparison.txt', 'w') as f:
    all_rows = []
    for ds in datasets:
        ds_name = re.sub(r'|'.join(map(re.escape, rm_words)), '', ds)
        ds_name = re.sub(r'_', '\_', ds_name)
        line = f'{ds_name}\t'
        row = []
        for x in data:
            row.extend(list(x[ds]))
        for i in range(4):
            line = line + f'& {row[i*4]:.4f} &$\pm$ {row[i*4+1]:.2f} & {row[i*4+2]:.2f} &$\pm$ {row[i*4+3]:.2f} '
        line = line + ' \\\\ \n'
        f.write(line)
        all_rows.append(row)
    row = np.mean(all_rows, axis=0)
    line = 'Promedio\t'
    for i in range(4):
        line = line + f'& {row[i*4]:.4f} &$\pm$ {row[i*4+1]:.2f} & {row[i*4+2]:.2f} &$\pm$ {row[i*4+3]:.2f} '
    line = line + ' \\\\'
    f.write(line)

 