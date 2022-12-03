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

alg = 'n3o'
exp = experiment[alg]
filename = f'final_exp/{alg}/results_{alg}_final{exp}_full_loss_arch.csv'
df = pd.read_csv(filename, header=0, index_col=0).transpose()

fig = px.box(df[datasets[0]])
fig.show()

# seaborn.set(style='whitegrid')
# seaborn.boxplot(data=df)
# plt.xticks(rotation='vertical')
# plt.show()
