import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from icecream import ic
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
# # -- The single material Delta (CC limit), not heterostructure part --
# mzs, NDelta, Ny, Deltas, Tcs1     = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData2/ONEMATDDELTASMzND=5Ny=20Nx=20mu=-0.5.npy", allow_pickle=True)
# mgs, NDelta2, Ny2, Deltas2, Tcs2 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData2/ONEMATDDELTASMgND=5Ny=20Nx=20mu=-0.5.npy", allow_pickle=True)
# # mgs3, NDelta2, Ny2, result3 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=0.0.npy", allow_pickle=True)
# # mgs4, NDelta2, Ny2, result4 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=0.5.npy", allow_pickle=True)
# # mgs5, NDelta2, Ny2, result5 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=-0.5.npy", allow_pickle=True)

# font = {'size'   : 14}

# mpl.rc('font', **font)

# fig, ax = plt.subplots()

# ax.set_xlabel("Magnetic strength $m / /Delta(T=0)$")
# ax.set_ylabel(r"Zero-termperture gap $/Delta(T=0)$")

# ax.plot(mzs / Deltas[0],  Tcs1, label = "FM")
# ax.plot(mgs[:] / Deltas2[0], Tcs2[:], label = "AM")#, mu = -0.5")
# plt.show()
# ic(Deltas[0], Deltas2[0])
# # ic(Deltas[0])
# # mzs2, NDelta, Ny, Deltas, Tcs3     = np.load(r"C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATDDELTASMzND=10Ny=20Nx=20mu=-0.5.npy", allow_pickle=True)
# # mgs2, NDelta2, Ny2, Deltas2, Tcs4 = np.load(r"C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATDDELTASMgND=10Ny=20Nx=20mu=-0.5.npy", allow_pickle=True)
# # ax.plot(mzs2 / Deltas[0],  Deltas, label = "FM, Ny=20")
# # ax.plot(mgs2 / Deltas2[0], Deltas2, label = "AM, Ny=20")
# # # ax.plot(mgs3, result3, label = "AM, mu = 0.0")
# # # ax.plot(mgs4, result4, label = "AM, mu = 0.5", ls = "dashed")
# # # ax.plot(mgs5, result5, label = "AM")
# # ax.legend(loc = "best")
# # plt.tight_layout()

# fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/onemat_Tcs.pdf", format = "pdf", bbox_inches="tight")
# -- End --------------------------------------

items, DeltasFM, mzs = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/onemat_Delta/FM(20, 20, 1e-08, 0.3, 0, 0, 0.01).npy", allow_pickle=True)

items, DeltasAM, mgs = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/onemat_Delta/AM(20, 20, 1e-08, 0, 0, 0.3, 0.01).npy", allow_pickle=True)


plt.style.use('bmh')

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 20
    })


fig, ax = plt.subplots()

# ax.set_xlabel("Magnetic strength $m$")
ax.set_ylabel(r"$\Delta/\Delta_0$")

# assert TcsFM[0] == TcsAM[0]
# ax.plot(mzs , TcsAM/ TcsAM[0], label = "AM")
ax.plot(mgs , DeltasAM / DeltasAM[0], label = "AM")
ax.plot(mzs , DeltasFM / DeltasFM[0], label = "FM")
ax.set_xlim(0, 0.2)
# ax.plot(mgs2 , TcsAM2/ TcsAM2[0], label = "AM")
# ax.plot(mzs2 , TcsFM2/ TcsAM2[0], label = "FM")


ax.legend(loc = "best", fontsize = "medium", frameon = False)

# x-axis label
xbox1 = TextArea("$m/t$ ", textprops=dict(color="#348ABD", size=22))
xbox2 = TextArea(", ", textprops=dict(color="k", size=22))
xbox3 = TextArea("$m_z/t$ ", textprops=dict(color="#A60628", size=22))

xbox = HPacker(children=[xbox1, xbox2, xbox3],
                  align="center", pad=0, sep=5)

anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0., frameon=False,
                                      bbox_to_anchor=(0.38, -0.19),
                                      bbox_transform=ax.transAxes, borderpad=0.)

ax.add_artist(anchored_xbox)
plt.tight_layout()
fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figs/onemat_TDeltas.pdf", format = "pdf", bbox_inches="tight")
plt.show()
# -- The single material Tc, not heterostructure part --

# mzs, NDelta, Ny, result = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/MzOneMatND=10Ny=10.npy", allow_pickle=True)

# mgs, NDelta2, Ny2, result2 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/MgOneMatND=10Ny=10.npy", allow_pickle=True)
items, TcsAM, mgs  = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/onemat/AM(20, 20, 30, 0, 0, 0.3).npy", allow_pickle=True)
items, TcsFM, mzs  = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/onemat/FM(20, 20, 30, 0.3, 0, 0).npy", allow_pickle=True)

# items, TcsAM2, mgs2  = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/onemat/AM(10, 10, 20, 0, 0, 0.3).npy", allow_pickle=True)
# items, TcsFM2, mzs2  = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/onemat/FM(10, 10, 20, 0.3, 0, 0).npy", allow_pickle=True)


plt.style.use('bmh')

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 20
    })


fig, ax = plt.subplots()

# ax.set_xlabel("Magnetic strength $m$")
ax.set_ylabel(r"$T_c/T_c(m=0)$")

assert TcsFM[0] == TcsAM[0]
ax.plot(mzs , TcsAM/ TcsAM[0], label = "AM")
ax.plot(mgs , TcsFM/ TcsAM[0], label = "FM")

# ax.plot(mgs2 , TcsAM2/ TcsAM2[0], label = "AM")
# ax.plot(mzs2 , TcsFM2/ TcsAM2[0], label = "FM")


ax.legend(loc = "best", fontsize = "medium", frameon = False)

# x-axis label
xbox1 = TextArea("$m/t$ ", textprops=dict(color="#348ABD", size=22))
xbox2 = TextArea(", ", textprops=dict(color="k", size=22))
xbox3 = TextArea("$m_z/t$ ", textprops=dict(color="#A60628", size=22))

xbox = HPacker(children=[xbox1, xbox2, xbox3],
                  align="center", pad=0, sep=5)

anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0., frameon=False,
                                      bbox_to_anchor=(0.38, -0.19),
                                      bbox_transform=ax.transAxes, borderpad=0.)

ax.add_artist(anchored_xbox)
plt.tight_layout()
fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figs/onemat_Tcs.pdf", format = "pdf", bbox_inches="tight")
plt.show()
# -- End --------------------------------------
