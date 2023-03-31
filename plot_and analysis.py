import numpy as np
from matplotlib import pyplot as plt

Tcs, mgs = np.load("Tc_mg_sweep.npy")

plt.plot(mgs, Tcs)
plt.show()