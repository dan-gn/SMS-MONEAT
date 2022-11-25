import numpy as np
import os
import sys
import csv
import pandas as pd
import plotly.express as px

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from experiment_info import N_EXPERIMENTS
from experiment_info import datasets

def csv2expdata(filename):
    metric = {}
    with open(filename) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            metric[row[0]] = row[1:]
    return metric

filename = 'final_exp/sms_moneat/results_sms_moneat_final6_full_loss_arch.csv'
data = pd.read_csv(filename)

print(data.shape)