import numpy as np
import os
import sys
import csv
import re
import pandas as pd

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

filename = []
alg = 'n3o'
filename.append(f'final_exp/{alg}/results_{alg}_final2_full_time.csv')
alg = 'sms_emoa'
filename.append(f'final_exp/{alg}/results_{alg}_final5x_full_time.csv')
alg = 'sms_moneat'
filename.append(f'final_exp/{alg}/results_{alg}_final6_full_time.csv')


df = [pd.read_csv(file) for file in filename]

rm_words = ['Breast_', 'Colorectal_', 'Leukemia_', 'Liver_', 'Prostate_']
with open('table_time.txt', 'w') as f:
    all_rows = [[] for _ in range(len(filename))]
    for ds in datasets:
        row = [x.loc[x['Dataset'] == ds].iloc[0, 1:] for x in df]
        ds_name = re.sub(r'|'.join(map(re.escape, rm_words)), '', ds)
        ds_name = re.sub(r'_', '\_', ds_name)
        line = f'{ds_name}\t'
        for r in row:
            line += f'& {r.mean():.4f} &$\pm$ {r.std():.2f}'
        line += '\\\\ \n'
        f.write(line)
        for i, r in enumerate(row):
            all_rows[i].append(r)
    all_rows = [pd.DataFrame(r).stack() for r in all_rows]
    line = f'Todos\t'
    for r in all_rows:
        line += f'& {r.mean():.4f} &$\pm$ {r.std():.2f}'
    line += '\\\\ \n'
    f.write(line)

# cols = [3, 7]

# data = csv2expdata(filename, cols)
# rm_words = ['Breast_', 'Colorectal_', 'Leukemia_', 'Liver_', 'Prostate_']

# with open('table_solutions_sms-emoa.txt', 'w') as f:
#     all_rows = []
#     for ds in datasets:
#         row = data[ds]
#         ds_name = re.sub(r'|'.join(map(re.escape, rm_words)), '', ds)
#         ds_name = re.sub(r'_', '\_', ds_name)
#         line = f'{ds_name}\t& {row[0]:.4f} $\pm$& {row[1]:.2f} & {row[2]:.2f} $\pm$& {row[3]:.2f} \\\\ \n'#& {row[4]:.4f} $\pm$& {row[5]:.2f} & {row[6]:.2f} $\pm$& {row[7]:.2f} \\\\ \n'
#         f.write(line)
#         all_rows.append(row)
#     row = np.mean(all_rows, axis=0)
#     line = f'Promedio\t& {row[0]:.4f} $\pm$& {row[1]:.2f} & {row[2]:.2f} $\pm$& {row[3]:.2f} \\\\ \n' #& {row[4]:.4f} $\pm$& {row[5]:.2f} & {row[6]:.2f} $\pm$& {row[7]:.2f} \\\\ \n'
#     f.write(line)

    