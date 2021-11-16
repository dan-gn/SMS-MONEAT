import numpy as np
import pickle
import matplotlib.pyplot as plt

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

n_tests = 2
n_iterations = 100

models = []
for i in range(n_tests):
	filename = f'bc_test_seed_{i}_it{n_iterations}.pkl'
	print(f'Reading test: {filename}')
	with open(f'results/{filename}', 'rb') as f:
		problem, params, res = pickle.load(f)
	models.append(res[0]['model'])

train_acc = np.zeros((n_tests, n_iterations + 1))
train_fit = np.zeros((n_tests, n_iterations + 1))
test_acc = np.zeros((n_tests, n_iterations + 1))
test_fit = np.zeros((n_tests, n_iterations + 1))

for i in range(n_tests):
	train_acc[i, :] = models[i].training_accuracy.T
	train_fit[i, :] = models[i].training_fitness.T
	test_acc[i, :] = models[i].testing_accuracy.T
	test_fit[i, :] = models[i].testing_fitness.T

fig = plt.figure(figsize=(6,6), dpi= 100, facecolor='w', edgecolor='k')
ax = plt.axes()
ax.set_title('Best Solution')
ax.set_xlabel('Iteration')
ax.set_ylabel('Fitness')

for fit in train_fit:
    plt.plot(fit, color='0.8')

plt.plot(np.mean(train_fit, axis=1), color='b')
# fig.show()
fig.savefig('bc_test_0001_train_fit.png')