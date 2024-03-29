import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from icecream import ic

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


# -- The single material Tc, not heterostructure part --

# mzs, NDelta, Ny, result = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/MzOneMatND=10Ny=10.npy", allow_pickle=True)

# # mgs, NDelta2, Ny2, result2 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/MgOneMatND=10Ny=10.npy", allow_pickle=True)
# mgs, NDelta2, Ny2, result2 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=-0.5.npy", allow_pickle=True)
# mgs3, NDelta2, Ny2, result3 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=0.0.npy", allow_pickle=True)
# mgs4, NDelta2, Ny2, result4 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=0.5.npy", allow_pickle=True)
# # mgs5, NDelta2, Ny2, result5 = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/ONEMATMg=10Ny=10Nx=20mu=-0.5.npy", allow_pickle=True)
#
# font = {'size'   : 14}

# mpl.rc('font', **font)

# fig, ax = plt.subplots()

# ax.set_xlabel("Magnetic strength $m / \Delta(T=0)$")
# ax.set_ylabel(r"Critical temperature $T_c/T_c(m=0)$")
# ic(Tcs1[0])
# ic(len(mgs))
# ic(len(mzs))
# ic(len(mgs2))
# ic(len(mzs2))

# ax.plot(mzs / Deltas[0], Tcs1/ Tcs1[0], label = "FM")
# ax.plot(mgs / Deltas[0], Tcs2/ Tcs2[0], label = "AM")
# ax.plot(mzs2 / Deltas[0], Tcs3/ Tcs3[0], label = "FM, Ny=20")
# ax.plot(mgs2 / Deltas[0], Tcs4/ Tcs4[0], label = "AM, Ny=20")

# # ax.plot(mgs3, result3, label = "AM, mu = 0.0")
# # ax.plot(mgs4, result4, label = "AM, mu = 0.5", ls = "dashed")
# # ax.plot(mgs5, result5, label = "AM")

# ax.legend(loc = "best")


# plt.tight_layout()

# # fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/onemat_Tcs.pdf", format = "pdf", bbox_inches="tight")
# plt.show()
# -- End --------------------------------------


# -- Junction geometry part --
plt.style.use('bmh')

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 20
    })
items, mgs_straight,  TcsStraight = np.load(
    "C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed_15/ShorterAMFalse(15, 20, 30, 0, 0, 1.0, False, 0, 0).npy", allow_pickle=True)

items, mgsSkewed,  TcsSkewed = np.load(
    "C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed_15/ShorterAMTrue(15, 20, 30, 0, 0, 1.0, True, 0, 0).npy", allow_pickle=True)


items, mzsStraight,  TcsFMStraight = np.load(
    "C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed_15/ShorterFMFalse(15, 20, 30, 1.0, 0, 0, False, 0, 0).npy", allow_pickle=True)

items, mzsSkewed,  TcsFMSkewed = np.load(
    "C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed_15/ShorterFMTrue(15, 20, 30, 1.0, 0, 0, True, 0, 0).npy", allow_pickle=True)

print(items[0])

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Times",
    'font.size': 18
})

fig, ax = plt.subplots()
print(TcsFMStraight[0], TcsStraight[0])
print(TcsFMSkewed[0], TcsSkewed[0])
TcsStraight /= TcsStraight[0]
TcsSkewed /= TcsSkewed[0]
TcsFMStraight /= TcsFMStraight[0]
TcsFMSkewed /= TcsFMSkewed[0]

ax.plot(mgs_straight, TcsStraight, label="St AM")

ax.plot(mgsSkewed, TcsSkewed, label="Sk AM")
ax.plot(mzsStraight, TcsFMStraight, label="St FM")
ax.plot(mzsSkewed, TcsFMSkewed, label="Sk FM")

ax.set_xlabel("Altermagnetic strength $m$")
ax.set_ylabel(r"$T_c(m)/T_{c}(m=0)$")
ax.legend(loc="upper left", frameon=False, ncols=2)

plt.tight_layout()

# fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/geometry.pdf", format = "pdf", bbox_inches="tight")
plt.show()
plt.style.use('bmh')

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 20
    })

items, mgs_straight,  TcsStraight = np.load(
    "C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed_16/ShorterAMFalse(16, 20, 50, 0, 0, 1.0, False, 0, 0).npy", allow_pickle=True)

items, mgsSkewed,  TcsSkewed = np.load(
    "C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed_16/ShorterAMTrue(16, 20, 50, 0, 0, 1.0, True, 0, 0).npy", allow_pickle=True)


items, mzsStraight,  TcsFMStraight = np.load(
    "C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed_16/ShorterFMFalse(16, 20, 50, 1.0, 0, 0, False, 0, 0).npy", allow_pickle=True)

items, mzsSkewed,  TcsFMSkewed = np.load(
    "C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed_16/ShorterFMTrue(16, 20, 50, 1.0, 0, 0, True, 0, 0).npy", allow_pickle=True)


fig, ax = plt.subplots()
print(TcsFMStraight[0], TcsStraight[0])
print(TcsFMSkewed[0], TcsSkewed[0])
TcsStraight /= TcsStraight[0]
TcsSkewed /= TcsSkewed[0]
TcsFMStraight /= TcsFMStraight[0]
TcsFMSkewed /= TcsFMSkewed[0]

ax.plot(mgs_straight, TcsStraight, label="St AM")
# ax.set_ylim(np.amin(resultStraight)*0.99, np.amax(resultStraight)*1.01)

ax.plot(mgsSkewed, TcsSkewed, label="Sk AM", color = "#348ABD", linestyle = "dashed")
ax.plot(mzsStraight, TcsFMStraight, label="St FM")
ax.plot(mzsSkewed, TcsFMSkewed, label="Sk FM", color = "#A60628", linestyle = "dashed")

# ax.set_xlabel("Altermagnetic strength $m$")
ax.set_ylabel(r"$T_c(m)/T_{c}(m=0)$")
ax.legend(loc="best", frameon=False)

plt.tight_layout()

# x-axis label
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

xbox1 = TextArea("$m/t$ ", textprops=dict(color="#348ABD", size=22))
xbox2 = TextArea(", ", textprops=dict(color="k", size=22))
xbox3 = TextArea("$m_z/t$ ", textprops=dict(color="#A60628", size=22))

xbox = HPacker(children=[xbox1, xbox2, xbox3],
                  align="center", pad=0, sep=5)

anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0., frameon=False,
                                      bbox_to_anchor=(0.38, -0.19),
                                      bbox_transform=ax.transAxes, borderpad=0.)

ax.add_artist(anchored_xbox)
fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figs/ss16.pdf", format = "pdf", bbox_inches="tight")
plt.show()
# -- End --------------------------

# -- P/AP part --
# mgsP, NDelta, Ny, Deltas, resultP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/PND=20Ny=10Nx=30mu=-0.5U=1.7.npy", allow_pickle=True)

# mgsAP, NDelta2, Ny2, Deltas, resultAP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/APND=20Ny=10Nx=30mu=-0.5U=1.7.npy", allow_pickle=True)

# font = {'size'   : 14}

# mpl.rc('font', **font)

# fig, ax = plt.subplots()
# print(resultP[0], resultAP[0])
# resultP = resultP / resultP[0]
# resultAP = resultAP / resultAP[0]
# ax.plot(mgsP, resultP, label = "P")
# ax.set_ylim(np.amin(resultP)*0.99, np.amax(resultP)*1.01)

# ax.plot(mgsAP, resultAP, label = "AP")
# ax.set_xlabel("Altermagnetic strength m")
# ax.set_ylabel(r"Critical temperature $T_c(m)/T_c(m=0)$")
# ax.legend(loc = "best")


# plt.tight_layout()

# # fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/PAP.pdf", format = "pdf", bbox_inches="tight")
# plt.show()
# -- End -----------------------


# # --  P/AP part --
# mgsP,NDelta, Ny, _, resultP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/PND=10Ny=10Nx=40mu=-0.5U=1.7.npy", allow_pickle=True)
# # print(mgsP)
# mgsAP,NDelta, Ny, _,  resultAP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/NewData/APND=10Ny=10Nx=40mu=-0.5U=1.7.npy", allow_pickle=True)

# font = {'size'   : 14}

# mpl.rc('font', **font)

# fig, ax = plt.subplots()
# print(resultP[0], resultAP[0])
# resultP = resultP / resultP[0]
# resultAP = resultAP / resultAP[0]
# ax.plot(mgsP, resultP, label = "P")
# ax.set_ylim(np.amin(resultP)*0.99, np.amax(resultP)*1.01)

# ax.plot(mgsAP, resultAP, label = "AP")
# ax.set_xlabel("Altermagnetic strength m")
# ax.set_ylabel(r"Critical temperature $T_c(m)/T_c(m=0)$")
# ax.legend(loc = "best")


# plt.tight_layout()

# fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/PAPNx=40.pdf", format = "pdf", bbox_inches="tight")
# plt.show()
# -- End -----------------------
