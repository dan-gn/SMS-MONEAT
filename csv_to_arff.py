import os
import sys

current = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current)
# parent = os.path.dirname(current)
# sys.path.append(parent)

import pandas as pd

path = 'datasets/ASOC_comp/'
dataset_name = 'DLBCL'
filename = path + dataset_name

df = pd.read_csv(f'{filename}.csv')

import arff
arff.dump(f'{filename}.arff'
      , df.values
      , relation='relation name'
      , names=df.columns)