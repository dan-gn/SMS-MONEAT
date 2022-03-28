import numpy as np
import pickle
import statistics
from scipy.stats import ranksums, mannwhitneyu

ds_names = ['GSE22820', 'GSE42568', 'GSE59246', 'GSE70947', 'Van De Vijver et al. \cite{van2002gene\}', 'GSE8671', 'GSE32323']
ds_names.extend(['GSE44076', 'GSE44861', 'GSE14317', 'GSE33615', 'GSE63270', 'GSE71935', 'Golub et al. \cite{golub1999molecular\}'])
ds_names.extend(['GSE14520_U133A','GSE62232', 'GSE6919_U95Av2', 'GSE11682', 'GSE46602', 'Singh et al. \cite{singh2002gene\}'])

output_filename = 'hv_results_4.pkl'
with open(output_filename, 'rb') as f:
	hv_results = pickle.load(f)

n3o_hv = hv_results['n3o'][0]
sms_moneat_hv = hv_results['sms_moneat'][0]
archive_hv = hv_results['sms_moneat_arch'][0]

# counter = 0
# for i in range(n3o_hv.shape[0]):
# 	if sms_moneat_hv[i] > n3o_hv[i]:
# 		counter += 1

# print(counter)
# print(sms_moneat_hv)

def wilcoxon_rank_sum(a, b):
	_, pvalue =  mannwhitneyu(a, b, alternative="two-sided")
	if pvalue < 0.05:
		return False
	return True

counter = np.zeros(3)
for i, name in enumerate(ds_names):
	a = wilcoxon_rank_sum(archive_hv[i], sms_moneat_hv[i])
	b = wilcoxon_rank_sum(archive_hv[i], n3o_hv[i])
	c = wilcoxon_rank_sum(sms_moneat_hv[i], n3o_hv[i])
	if a:
		counter[0] += 1
	if b:
		counter[1] += 1
	if c:
		counter[2] += 1

	print(f'{name}: \t {a}, {b}, {c}')
print(counter)



# for i, name in enumerate(ds_names):
# 	a = archive_hv[i].mean()
# 	a_std = statistics.stdev(archive_hv[i])
# 	b = sms_moneat_hv[i].mean()
# 	b_std = statistics.stdev(sms_moneat_hv[i])
# 	c = n3o_hv[i].mean()
# 	c_std = statistics.stdev(n3o_hv[i])
# 	print(f'{name} & {a:.4f} ({a_std:.4f}) & {b:.4f} ({b_std:.4f}) & {c:.4f} ({c_std:.4f}) \\\\')
# print(f'Average & {archive_hv.mean():.4f} ({statistics.stdev(archive_hv.flatten()):.4f}) & {sms_moneat_hv.mean():.4f} ({statistics.stdev(sms_moneat_hv.flatten()):.4f}) & {n3o_hv.mean()} ({statistics.stdev(n3o_hv.flatten()):.4f})')