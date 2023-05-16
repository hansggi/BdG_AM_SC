import numpy as np
from matplotlib import pyplot as plt


# Tcs = np.load("Tcs.npy", allow_pickle=True)
# mgs = np.linspace(0, 0.3, 20)
# plt.plot(mgs, Tcs, label = "straight")
# TcsSkew = np.load("TcsSkewed.npy", allow_pickle=True)
# plt.plot(mgs, TcsSkew, label = "skewed", linestyle = "dashed")
# plt.legend()
# plt.show()

# mgs = np.linspace(0, 2, 20)
mgs, Tcs = np.load("mgdata.npy")

plt.plot(mgs, Tcs)
plt.show()