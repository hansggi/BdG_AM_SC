import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


# -- The single material Delta (CC limit), not heterostructure part --

mzs, NDelta, Ny, Deltas, Tcs     = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATDDELTASMzND=10Ny=10Nx=20mu=-0.5.npy", allow_pickle=True)
mgs, NDelta2, Ny2, Deltas2, Tcs2 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATDDELTASMgND=10Ny=10Nx=20mu=-0.5.npy", allow_pickle=True)
# mgs3, NDelta2, Ny2, result3 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=0.0.npy", allow_pickle=True)
# mgs4, NDelta2, Ny2, result4 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=0.5.npy", allow_pickle=True)
# mgs5, NDelta2, Ny2, result5 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=-0.5.npy", allow_pickle=True)

font = {'size'   : 14}

mpl.rc('font', **font)

fig, ax = plt.subplots()

ax.set_xlabel("Magnetic strength $m / \Delta(T=0)$")
ax.set_ylabel(r"Zero-termperture gap $\Delta(T=0)$")

ax.plot(mzs[::2], Deltas[::2], label = "FM")
ax.plot(mgs / Deltas2[0], Deltas2, label = "AM")#, mu = -0.5")

# ax.plot(mgs3, result3, label = "AM, mu = 0.0")
# ax.plot(mgs4, result4, label = "AM, mu = 0.5", ls = "dashed")
# ax.plot(mgs5, result5, label = "AM")

ax.legend(loc = "best")


plt.tight_layout()

# fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/onemat_Deltas.pdf", format = "pdf", bbox_inches="tight")
plt.show()
# -- End --------------------------------------


# -- The single material Tc, not heterostructure part --

# mzs, NDelta, Ny, result = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/MzOneMatND=10Ny=10.npy", allow_pickle=True)

# # mgs, NDelta2, Ny2, result2 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/MgOneMatND=10Ny=10.npy", allow_pickle=True)
# mgs, NDelta2, Ny2, result2 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=-0.5.npy", allow_pickle=True)
# mgs3, NDelta2, Ny2, result3 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=0.0.npy", allow_pickle=True)
# mgs4, NDelta2, Ny2, result4 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=0.5.npy", allow_pickle=True)
# # mgs5, NDelta2, Ny2, result5 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=-0.5.npy", allow_pickle=True)

font = {'size'   : 14}

mpl.rc('font', **font)

fig, ax = plt.subplots()

ax.set_xlabel("Magnetic strength $m / \Delta(T=0)$")
ax.set_ylabel(r"Critical temperature $T_c/T_c(m=0)$")

ax.plot(mzs / Deltas[0], Tcs/ Tcs[0], label = "FM")
ax.plot(mgs / Deltas[0], Tcs2/ Tcs[0], label = "AM, mu = -0.5")
# ax.plot(mgs3, result3, label = "AM, mu = 0.0")
# ax.plot(mgs4, result4, label = "AM, mu = 0.5", ls = "dashed")
# ax.plot(mgs5, result5, label = "AM")

ax.legend(loc = "best")


plt.tight_layout()

# fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/onemat_Tcs.pdf", format = "pdf", bbox_inches="tight")
plt.show()
# -- End --------------------------------------






# -- Junction geometry part --
"""mgsStraight, NDelta, Ny, resultStraight = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/STRAIGHT=30Ny=10.npy", allow_pickle=True)

mgsSkewed, NDelta2, Ny2, resultSkewed = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/SKEWED=30Ny=10.npy", allow_pickle=True)

font = {'size'   : 14}

mpl.rc('font', **font)

fig, ax = plt.subplots()
print(resultStraight[0], resultSkewed[0])
resultStraight = resultStraight / resultStraight[0]
resultSkewed = resultSkewed / resultSkewed[0]
ax.plot(mgsStraight, resultStraight, label = "Straight")
ax.set_ylim(np.amin(resultStraight)*0.99, np.amax(resultStraight)*1.01)

ax.plot(mgsSkewed, resultSkewed, label = "Skewed")
ax.set_xlabel("Altermagnetic strength m")
ax.set_ylabel(r"Critical temperature $T_c(m)/T_c(m=0)$")
ax.legend(loc = "best")


plt.tight_layout()

fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/geometry.pdf", format = "pdf", bbox_inches="tight")
plt.show()"""
# -- End --------------------------

# -- P/AP part --
"""mgsP, NDelta, Ny, resultP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/P=30Ny=10.npy", allow_pickle=True)

mgsAP, NDelta2, Ny2, resultAP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/AP=30Ny=10.npy", allow_pickle=True)

font = {'size'   : 14}

mpl.rc('font', **font)

fig, ax = plt.subplots()
print(resultP[0], resultAP[0])
resultP = resultP / resultP[0]
resultAP = resultAP / resultAP[0]
ax.plot(mgsP, resultP, label = "P")
ax.set_ylim(np.amin(resultP)*0.99, np.amax(resultP)*1.01)

ax.plot(mgsAP, resultAP, label = "Skewed")
ax.set_xlabel("Altermagnetic strength m")
ax.set_ylabel(r"Critical temperature $T_c(m)/T_c(m=0)$")
ax.legend(loc = "best")


plt.tight_layout()

fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/PAP.pdf", format = "pdf", bbox_inches="tight")
plt.show()"""
# -- End -----------------------

# -- OLD P/AP part --
# mgsP, resultP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Parallell_mgdata.npy", allow_pickle=True)
# print(mgsP)
# mgsAP, resultAP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/AParallell_mgdata.npy", allow_pickle=True)

# font = {'size'   : 14}

# mpl.rc('font', **font)

# fig, ax = plt.subplots()
# print(resultP[0], resultAP[0])
# resultP = resultP / resultP[0]
# resultAP = resultAP / resultAP[0]
# ax.plot(mgsP, resultP, label = "P")
# ax.set_ylim(np.amin(resultP)*0.99, np.amax(resultP)*1.01)

# ax.plot(mgsAP, resultAP, label = "Skewed")
# ax.set_xlabel("Altermagnetic strength m")
# ax.set_ylabel(r"Critical temperature $T_c(m)/T_c(m=0)$")
# ax.legend(loc = "best")


# plt.tight_layout()

# fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/PAP.pdf", format = "pdf", bbox_inches="tight")
# plt.show()
# -- End -----------------------

