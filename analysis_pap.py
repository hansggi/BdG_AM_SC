import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from icecream import ic


# -- P/AP part --
_, resultP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/PAP/AMP(32, 20, [10, 22], 30, 0, 0, 1.0, 'P', 0, 0).npy", allow_pickle=True)

_, resultAP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/PAP/AMAP(32, 20, [10, 22], 30, 0, 0, 1.0, 'AP', 0, 0).npy", allow_pickle=True)

# _, result_P_FM = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/PAP/FMP(31, 20, [10, 21], 30, 1.0, 0, 0, 'P', 0, 0).npy", allow_pickle= True)
plt.style.use('bmh')

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 20
    })

# ic(resultAP[0],  result_P_FM[0])

# assert resultAP[0] == result_P_FM[0]
# assert result_P_FM[0] == resultP[0]
fig, ax = plt.subplots(frameon=False)
print(resultP[0], resultAP[0])
resultP = resultP / resultP[0]
resultAP = resultAP / resultAP[0]
# result_P_FM = result_P_FM / result_P_FM[0]
mgs = np.linspace(0, 1, 100)
ax.plot(mgs, resultP, label = "P")
ax.plot(mgs, resultAP, label = "AP")

# ax.plot(mgs, result_P_FM, label ="FM P")
# ax.set_ylim(np.amin(resultP)*0.99, np.amax(resultP)*1.01)

# ax.plot(mgs, resultAP, label = "AP")
ax.set_ylabel(r"$T_c(m)/T_{c}(m=0)$")
ax.set_xlabel(r"$m/t$")
ax.legend(loc = "best")

# x-axis label
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# xbox1 = TextArea("$m$ ", textprops=dict(color="#348ABD", size=22))
# xbox2 = TextArea(", ", textprops=dict(color="k", size=22))
# xbox3 = TextArea("$m_z$ ", textprops=dict(color="#A60628", size=22))

# xbox = HPacker(children=[xbox1, xbox2, xbox3],
#                   align="center", pad=0, sep=5)

# anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0., frameon=False,
#                                       bbox_to_anchor=(0.45, -0.17),
#                                       bbox_transform=ax.transAxes, borderpad=0.)

# ax.add_artist(anchored_xbox)
plt.tight_layout()

fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figs/PAP32.pdf", format = "pdf", bbox_inches="tight")
plt.show()
# -- End -----------------------


# -- P/AP part --
_, resultP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/PAP/AMP(31, 20, [10, 21], 30, 0, 0, 1.0, 'P', 0, 0).npy", allow_pickle=True)

_, resultAP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/PAP/AMAP(31, 20, [10, 21], 30, 0, 0, 1.0, 'AP', 0, 0).npy", allow_pickle=True)

_, result_P_FM = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/PAP/FMP(31, 20, [10, 21], 30, 1.0, 0, 0, 'P', 0, 0).npy", allow_pickle= True)

plt.style.use('bmh')

plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 20
    })

fig, ax = plt.subplots(frameon=False)
print(resultP[0], resultAP[0])
# resultP = resultP / resultP[0]
# resultAP = resultAP / resultAP[0]
# result_P_FM = result_P_FM / result_P_FM[0]

mgs = np.linspace(0, 1, 100)
ax.plot(mgs, resultP, label = "P")
ax.plot(mgs, resultAP, label = "AP")
ax.plot(mgs, result_P_FM, label ="FM P")

# ax.set_ylim(np.amin(resultP)*0.99, np.amax(resultP)*1.01)

# ax.plot(mgs, resultAP, label = "AP")
ax.set_ylabel(r"$T_c(m)/T_{c}(m=0)$")
# ax.set_xlabel(r"Altermagnetic strength $m$")
ax.legend(loc = "best", frameon = False, fontsize = "medium")

# x-axis label
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

xbox1 = TextArea("$m$ ", textprops=dict(color="#348ABD", size=22))
xbox2 = TextArea(", ", textprops=dict(color="k", size=22))
xbox3 = TextArea("$m_z$ ", textprops=dict(color="#A60628", size=22))

xbox = HPacker(children=[xbox1, xbox2, xbox3],
                  align="center", pad=0, sep=5)

anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=0., frameon=False,
                                      bbox_to_anchor=(0.45, -0.17),
                                      bbox_transform=ax.transAxes, borderpad=0.)

ax.add_artist(anchored_xbox)
plt.tight_layout()

# fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/PAP32.pdf", format = "pdf", bbox_inches="tight")
plt.show()
# -- End -----------------------"""