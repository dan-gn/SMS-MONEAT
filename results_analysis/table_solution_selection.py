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

filename = 'results_sms-emoa_final4.csv'
cols = [3, 7]

data = csv2expdata(filename, cols)


rm_words = ['Breast_', 'Colorectal_', 'Leukemia_', 'Liver_', 'Prostate_']

with open('table_solutions_sms-emoa.txt', 'w') as f:
    all_rows = []
    for ds in datasets:
        row = data[ds]
        ds_name = re.sub(r'|'.join(map(re.escape, rm_words)), '', ds)
        ds_name = re.sub(r'_', '\_', ds_name)
        line = f'{ds_name}\t& {row[0]:.4f} $\pm$& {row[1]:.2f} & {row[2]:.2f} $\pm$& {row[3]:.2f} \\\\ \n'#& {row[4]:.4f} $\pm$& {row[5]:.2f} & {row[6]:.2f} $\pm$& {row[7]:.2f} \\\\ \n'
        f.write(line)
        all_rows.append(row)
    row = np.mean(all_rows, axis=0)
    line = f'Promedio\t& {row[0]:.4f} $\pm$& {row[1]:.2f} & {row[2]:.2f} $\pm$& {row[3]:.2f} \\\\ \n' #& {row[4]:.4f} $\pm$& {row[5]:.2f} & {row[6]:.2f} $\pm$& {row[7]:.2f} \\\\ \n'
    f.write(line)

    