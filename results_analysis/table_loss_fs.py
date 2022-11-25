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

alg = 'sms_moneat'
filename_loss = f'final_exp/{alg}/results_{alg}_final6_full_arch.csv'
filename_fs = f'final_exp/{alg}/results_{alg}_final6_full_fs_arch.csv'


df_loss = pd.read_csv(filename_loss)
df_fs = pd.read_csv(filename_fs)

rm_words = ['Breast_', 'Colorectal_', 'Leukemia_', 'Liver_', 'Prostate_']
with open('table_sms_moneat_Q_loss_fs.txt', 'w') as f:
    all_rows_loss = []
    all_rows_fs = []
    for ds in datasets:
        row_loss = df_loss.loc[df_loss['Dataset'] == ds].iloc[0, 1:] 
        row_fs = df_fs.loc[df_fs['Dataset'] == ds].iloc[0, 1:] 
        ds_name = re.sub(r'|'.join(map(re.escape, rm_words)), '', ds)
        ds_name = re.sub(r'_', '\_', ds_name)
        line = f'{ds_name}\t'
        line += f'& {row_loss.mean():.4f} &$\pm$ {row_loss.std():.2f}'
        line += f'& {row_fs.mean():.2f} &$\pm$ {row_fs.std():.2f}'
        line += '\\\\ \n'
        f.write(line)
        all_rows_loss.append(row_loss)
        all_rows_fs.append(row_fs)
    all_rows_loss = pd.DataFrame(all_rows_loss).stack()
    all_rows_fs = pd.DataFrame(all_rows_fs).stack()
    line = f'Todos\t'
    line += f'& {all_rows_loss.mean():.4f} &$\pm$ {all_rows_loss.std():.2f}'
    line += f'& {all_rows_fs.mean():.2f} &$\pm$ {all_rows_fs.std():.2f}'
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

    