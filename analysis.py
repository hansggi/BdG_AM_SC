import numpy as np
from matplotlib import pyplot as plt


mgs, Tcs = np.load("Tcs.npy")


plt.plot(Tcs, mgs)