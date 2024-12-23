import numpy as np
import os
import sys
import csv
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import N_EXPERIMENTS, datasets, experiment

def csv2expdata(filename):
    metric = {}
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            metric[row[0]] = row[1:]
    return metric

alg_names = ['SMS-MONEAT', 'N3O', 'SMS-EMOA', 'MOCHC', 'SFE', 'SFE-PSO']

data = {}
metric = '_time'
metric_names = {
    '_loss_': 'Entropía Cruzada',
    '_fs_': 'Características seleccionadas',
    '_hv_': 'Hipervolumen',
    '_': 'Promedio geométrico',
    '_svm_': 'Promedio geométrico (SVM)',
    '_time': 'Tiempo de ejecución (s)'
}

alg = 'sms_moneat'
exp = experiment[alg]
filename = f'final_exp/{alg}/results_{alg}_final{exp}_full{metric}.csv'
data[alg_names[0]] = pd.read_csv(filename, header=0, index_col=0).transpose()
# filename = f'final_exp/{alg}/results_{alg}_final{exp}_full{metric}arch.csv'
# data[alg_names[1]] = pd.read_csv(filename, header=0, index_col=0).transpose()

alg = 'n3o'
exp = experiment[alg]
filename = f'final_exp/{alg}/results_{alg}_final{exp}_full{metric}.csv'
data[alg_names[1]] = pd.read_csv(filename, header=0, index_col=0).transpose()

alg = 'sms_emoa'
exp = experiment[alg]
filename = f'final_exp/{alg}/results_{alg}_final{exp}_full{metric}.csv'
data[alg_names[2]] = pd.read_csv(filename, header=0, index_col=0).transpose()

alg = 'mochc'
exp = experiment[alg]
filename = f'final_results_asc/{alg}/results_{alg}_final{exp}_full{metric}.csv'
data[alg_names[3]] = pd.read_csv(filename, header=0, index_col=0).transpose()

alg = 'sfe'
exp = experiment[alg]
filename = f'final_results_asc/{alg}/results_{alg}_final{exp}_full{metric}.csv'
data[alg_names[4]] = pd.read_csv(filename, header=0, index_col=0).transpose()

alg = 'sfe_pso'
exp = experiment[alg]
filename = f'final_results_asc/{alg}/results_{alg}_final{exp}_full{metric}.csv'
data[alg_names[5]] = pd.read_csv(filename, header=0, index_col=0).transpose()



# alg_names = ['N3O', 'SMS-MONEAT (P)', 'SMS-MONEAT (Q)']
# alg_names = ['N3O', 'SMS-EMOA']

x = []
dataset_names = []
dataset_names.append('Colon1') 
dataset_names.append('Colon2') 
dataset_names.append('Colon3') 
dataset_names.append('Colon4')
dataset_names.append('Higado1')
dataset_names.append('Higado2')
dataset_names.append('Higado3')
dataset_names.append('Leucemia1')
dataset_names.append('Leucemia2')
dataset_names.append('Leucemia3')
dataset_names.append('Leucemia4')
dataset_names.append('Leucemia5')
dataset_names.append('Mama1')
dataset_names.append('Mama2')
dataset_names.append('Mama3')
dataset_names.append('Mama4')
dataset_names.append('Próstata1')
dataset_names.append('Próstata2')
dataset_names.append('Próstata3')
dataset_names.append('Próstata4')


# for ds in dataset_names:
#     x.extend([ds for _ in range(N_EXPERIMENTS)])

# fig = go.Figure()
# for alg in alg_names: 
#     y = []
#     for ds in datasets:
#         y.extend(list(data[alg][ds]))
#     fig.add_trace(go.Box(y=y, x=x, name=alg))

# fig.update_layout(
#     xaxis_title='Conjunto de datos',
#     yaxis_title=metric_names[metric],
#     boxmode='group'
# )
# fig.show()

# seaborn.set(style='whitegrid')
# seaborn.boxplot(data=df)
# plt.xticks(rotation='vertical')
# plt.show()

for name in alg_names:
    x.extend([name for _ in range(N_EXPERIMENTS*len(dataset_names))])

fig = go.Figure()
for alg in alg_names: 
    y = []
    for ds in datasets:
        y.extend(list(np.log(data[alg][ds])))
    print(np.mean(y))
    fig.add_trace(go.Box(y=y, x=x, name=alg))

fig.update_layout(
    xaxis_title='Conjunto de datos',
    yaxis_title=metric_names[metric],
    boxmode='group'
)

# fig.show()

boxplot_data = []
for alg in alg_names:
    temp = []
    for ds in datasets:
        temp.extend(list(np.log(data[alg][ds])))
        # temp.extend(list(data[alg][ds]))
    boxplot_data.append(temp)

# colors = ['#6666ff', '#ff6666', '#66ff66'] * 2
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
colors = ['#ff9999', '#ffcc99', '#ffff99', '#99ff99', '#99ccff', '#cc99ff']

fig, ax = plt.subplots()
ax.set_title('Execution time from microarray experiments')
ax.set_xlabel('Algorithm')
ax.set_ylabel('Time (s) in log scale')
print(len(boxplot_data), len(boxplot_data[0]))
bplot = ax.boxplot(boxplot_data,
                   patch_artist=True,  # fill with color
                   labels=alg_names)  # will be used to label x-ticks


# fill with colors
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

plt.show()