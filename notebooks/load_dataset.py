import pandas as pd
import numpy as np
from scipy.io import arff

filename = './datasets/Breast_GSE42568.arff'

def load_dataset(filename):
	dataset = arff.loadarff(filename)
	return pd.DataFrame(dataset[0])

df = load_dataset(filename)


