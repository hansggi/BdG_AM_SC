import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from icecream import ic



# -- P/AP part --
_, resultP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/PAP/AMP(32, 20, [10, 22], 30, 0, 0, 1.0, 'P', 0, 0).npy", allow_pickle=True)

_, resultAP = np.load("C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/PAP/AMAP(32, 20, [10, 22], 30, 0, 0, 1.0, 'AP', 0, 0).npy", allow_pickle=True)


plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 18
    })

fig, ax = plt.subplots()
print(resultP[0], resultAP[0])
resultP = resultP / resultP[0]
resultAP = resultAP / resultAP[0]
mgs = np.linspace(0, 1, 100)
ax.plot(mgs, resultP, label = "P")
ax.plot(mgs, resultAP, label = "AP")

# ax.set_ylim(np.amin(resultP)*0.99, np.amax(resultP)*1.01)

# ax.plot(mgs, resultAP, label = "AP")
ax.set_xlabel("Altermagnetic strength m")
ax.set_ylabel(r"Critical temperature $T_c(m)/T_c(m=0)$")
ax.legend(loc = "best")


plt.tight_layout()

fig.savefig("C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/PAP32.pdf", format = "pdf", bbox_inches="tight")
plt.show()
# -- End -----------------------



# --  P/AP part --
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
# # -- End -----------------------