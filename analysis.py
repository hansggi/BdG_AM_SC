import numpy as np
from matplotlib import pyplot as plt


Tcs = np.load("Tcs2_mgs=[(0.0, 1.0)].npy", allow_pickle=True)
mgs = np.linspace(0, 1, 20)
plt.plot(mgs, Tcs)


plt.show()