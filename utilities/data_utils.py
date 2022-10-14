import numpy as np
import time

from utilities.moea_utils import choose_min_hv_contribution

def check_repeated_rows(A):
	unq, count = np.unique(A, axis=0, return_counts=True)
	repeated = unq[count > 1]
	return True if repeated.shape[0] >= 1 else False

def choose_repeated_index(A):
	unq, count = np.unique(A, axis=0, return_counts=True)
	repeated = unq[count > 1]
	if repeated.shape[0] >= 1:
		row = 0 if repeated.shape[0] == 1 else choose_min_hv_contribution(repeated)
		# row = np.random.randint(0, repeated.shape[0])
		row_idx = np.argwhere((A == repeated[row]).all(axis=1)).squeeze()
		chosen_row = np.random.choice(row_idx)
		return chosen_row, True
	else:
		return None, False
