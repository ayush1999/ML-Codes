import numpy as np
from matplotlib import pylab as plt

A = np.fromfile('train-images.idx3-ubyte',dtype='int16', sep='')
A = A.reshape([23520008,23520008])
plt.imshow(A)