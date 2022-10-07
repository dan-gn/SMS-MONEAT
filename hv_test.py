import numpy as np
import time

import pygmo
from utilities.hv import HyperVolume


n = 1000
referencePoint = np.array([n, n])

# start = time.time()
# hv = HyperVolume(referencePoint)
front = [[i, n-i] for i in range(n+1)]
# for i in range(n):
# 	volume = hv.compute(np.array([x for j, x in enumerate(front) if j!=i]))
# print(volume)
# print(time.time() - start)

start = time.time()
for i in range(n):
	hv = pygmo.hypervolume([x for j, x in enumerate(front) if j!=i])
	volume = hv.compute(referencePoint)
print(volume)
print(time.time() - start)
