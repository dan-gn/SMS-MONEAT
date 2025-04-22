import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import arff
import numpy as np
import pandas as pd


# This is important if you will be using rpy2
import os
os.environ['R_USER'] = 'D:\Anaconda3\Lib\site-packages\rpy2'

# Enable automatic conversion between R and pandas
pandas2ri.activate()

# Load the RData file
def load_rdata(file_path):
    robjects.r['load'](file_path)
    # List of all R objects loaded
    return list(robjects.r.objects())

# Convert pandas DataFrame to ARFF format and save
def save_as_arff(df, filename):
    # Build ARFF dictionary
    arff_data = {
        'description': '',
        'relation': filename,
        'attributes': [(col, 'REAL' if pd.api.types.is_numeric_dtype(df[col]) else 'STRING') for col in df.columns],
        'data': df.values.tolist()
    }
    with open(filename + '.arff', 'w') as f:
        arff.dump(arff_data, f)

# Main logic
if __name__ == '__main__':
    dataset_name = 'golub'
    file_path = f'datasets/ASoC_comp/{dataset_name}.RData'  # Replace with your actual file
    dataset_names = load_rdata(file_path)

    ds = [dataset_name, dataset_name+'_train', dataset_name+'_test']

    for name in ds:
        if name in dataset_names:
            df = pandas2ri.rpy2py(robjects.r[name])
            df.head()
            # df = pandas2ri.rpy2py_dataframe(r_list.rx2[name])
            # df = robjects.r[name]
            # x = df
            # y = df[1]
            # print(type(x), x.size)
            print(type(df))

            
            save_as_arff(df, name)
            print(f'Saved {name}.arff')
        else:
            print(f"Dataset '{name}' not found in the RData file.")
