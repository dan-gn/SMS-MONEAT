import numpy as np
import os
import sys
import csv
import re
import pandas as pd
import copy

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import datasets, dataset_names

def csv2expdata(filename, cols):
    data = {}
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i > 1:
                data[row[0]] = np.array(row[cols[0]:cols[1]], dtype=np.float32)
    return data

filename = []
alg = 'sms_moneat'
filename = f'final_exp/{alg}/results_{alg}_final6_full_ANNInfo_arch.csv'

df = pd.read_csv(filename)
df_fs = df.iloc[:20, :]
df_hidden = df.iloc[20:40, :]
df_connections = df.iloc[40:60, :]
df_fs_active = df.iloc[60:80, :]
df_hidden_active = df.iloc[80:100, :]
df_connections_active = df.iloc[100:120, :]

rm_words = ['Breast_', 'Colorectal_', 'Leukemia_', 'Liver_', 'Prostate_']
with open('table_ANNInfo_sms_moneat_Q.txt', 'w') as f:
    for j, ds in enumerate(datasets):
        row_fs = df_fs.loc[df_fs['Dataset'] == ds].iloc[0, 1:]
        row_fs_active = df_fs_active.loc[df_fs_active['Dataset'] == ds].iloc[0, 1:]
        row_hidden = df_hidden.loc[df_hidden['Dataset'] == ds].iloc[0, 1:]
        row_hidden_active = df_hidden_active.loc[df_hidden_active['Dataset'] == ds].iloc[0, 1:]
        row_connections = df_connections.loc[df_connections['Dataset'] == ds].iloc[0, 1:]
        row_connections_active = df_connections_active.loc[df_connections_active['Dataset'] == ds].iloc[0, 1:]

        ds_name = re.sub(r'|'.join(map(re.escape, rm_words)), '', ds)
        ds_name = re.sub(r'_', '\_', ds_name)
        # line = f'{ds_name}\t'
        line = f'{dataset_names[j]}\t'
        line += f'& {row_fs.mean():.2f} &$\pm$ {row_fs.std():.2f}'
        line += f'& {row_fs_active.mean():.2f} &$\pm$ {row_fs_active.std():.2f}'
        line += f'& {row_hidden.mean():.2f} &$\pm$ {row_hidden.std():.2f}'
        line += f'& {row_hidden_active.mean():.2f} &$\pm$ {row_hidden_active.std():.2f}'
        line += f'& {row_connections.mean():.2f} &$\pm$ {row_connections.std():.2f}'
        line += f'& {row_connections_active.mean():.2f} &$\pm$ {row_connections_active.std():.2f}'
        line += '\\\\ \n'
        f.write(line)
    all_rows_fs = df_fs.iloc[:, 1:].stack()
    all_rows_fs_active = df_fs_active.iloc[:, 1:].stack()
    all_rows_hidden= df_hidden.iloc[:, 1:].stack()
    all_rows_hidden_active = df_hidden_active.iloc[:, 1:].stack()
    all_rows_connections = df_connections.iloc[:, 1:].stack()
    all_rows_connections_active = df_connections_active.iloc[:, 1:].stack()
    line = f'Todos\t'
    line += f'& {all_rows_fs.mean():.2f} &$\pm$ {all_rows_fs.std():.2f}'
    line += f'& {all_rows_fs_active.mean():.2f} &$\pm$ {all_rows_fs_active.std():.2f}'
    line += f'& {all_rows_hidden.mean():.2f} &$\pm$ {all_rows_hidden.std():.2f}'
    line += f'& {all_rows_hidden_active.mean():.2f} &$\pm$ {all_rows_hidden_active.std():.2f}'
    line += f'& {all_rows_connections.mean():.2f} &$\pm$ {all_rows_connections.std():.2f}'
    line += f'& {all_rows_connections_active.mean():.2f} &$\pm$ {all_rows_connections_active.std():.2f}'
    line += '\\\\ \n'
    f.write(line)
