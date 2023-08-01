import numpy as np
from matplotlib import pyplot as plt
from icecream import ic
def getms(items, el):
    n = len(items)
    ms = np.zeros(n)
    for i in range(n):
        # print(items[i])
        ms[i] = items[i][el]
        # print(ms[i])
    
    return ms

# NDelta = 10
# itemsssAMF, TcsssAMF = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed/AMFalse(20, 20, {NDelta}, 0, 0, 1.0, False, 0, 0).npy", allow_pickle=True)
# mgssAMF = getms(itemsssAMF, 5)
# itemsssAMT, TcsssAMT = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed/AMTrue(20, 20, {NDelta}, 0, 0, 1.0, True, 0, 0).npy", allow_pickle=True)
# mgssAMT = getms(itemsssAMT, 5)

# itemsssFMF, TcsssFMF = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/straightskewed/FMFalse(20, 20, {NDelta}, 1.0, 0, 0, False, 0, 0).npy", allow_pickle=True)
# mgssFMF = getms(itemsssFMF, 3)

# fig, ax = plt.subplots()

# ax.plot(mgssAMF, TcsssAMF, label = "AMF")
# ax.plot(mgssAMT, TcsssAMT, label = "AMT")
# ax.plot(mgssFMF, TcsssFMF, label = "FMF")
# plt.legend()

# plt.tight_layout()
# plt.rcParams.update({
#         "text.usetex": True,
#         "font.family": "Times",
#     })

# plt.show()


# NDelta = 10
# Tcs0 : m = mg, but w = 0
# Tcs1 : m= 0, w = 0
"""items, Tcs, Tc0, Tc1 = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps_new/AMm=0.0mz=0.0((16, 20), 30, 1.0, 0.2).npy", allow_pickle=True)
Tcs, Tc0, Tc1 = Tcs/Tc1, Tc0/Tc1, Tc1/Tc1 
mgsImps = getms(items, 5)
TcsAv = np.average(Tcs)
dTcs = np.std(Tcs)
ic(TcsAv, Tc0, Tc1, dTcs)
plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 28
    })

# plt.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
fig, ax = plt.subplots(1,3, sharey=True, figsize = (15, 5))

ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
ax[2].set_xticklabels([])
ax[0].set_ylabel("$T_c$", fontsize = 32)
# fig, ax = plt.subplots()

ax[0].plot(np.arange(len(mgsImps)), Tcs, label = "$T_{c,i}$")
ax[0].axhline(y=TcsAv, label = r"$\langle T_c \rangle_{i}$", color = "orange")
ax[0].axhline(y=Tc0, label = "$T_{c,0}$", color = "green")
ax[0].axhline(y=Tc1, label = "$T_{c,0}(m=0)$", color = "purple")

ax[0].axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
ax[0].axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)
ax[0].set_xlabel("$i$")
ax[0].set_title("m=0")


items, Tcs, Tc0, Tc1 = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps_new/AMm=0.75mz=0.0((16, 20), 30, 1.0, 0.2).npy", allow_pickle=True)
Tcs, Tc0, Tc1 = Tcs/Tc1, Tc0/Tc1, Tc1/Tc1 

mgsImps = getms(items, 5)
TcsAv = np.average(Tcs)
dTcs = np.std(Tcs)
ic(TcsAv, Tc0, Tc1, dTcs)
ax[1].plot(np.arange(len(mgsImps)), Tcs, label = "$T_{c,i}$")
ax[1].axhline(y=TcsAv, label = r"$\langle T_c \rangle_{i}$", color = "orange")
ax[1].axhline(y=Tc0, label = "$T_{c,0}$", color = "green")
ax[1].axhline(y=Tc1, label = "$T_{c,0}(m=0)$", color = "purple")

ax[1].axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
ax[1].axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)
ax[1].set_xlabel("$i$")

ax[1].set_title("m=0.75")

items, Tcs, Tc0, Tc1 = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps_new/AMm=0.25mz=0.0((16, 20), 30, 1.0, 0.2).npy", allow_pickle=True)
Tcs, Tc0, Tc1 = Tcs/Tc1, Tc0/Tc1, Tc1/Tc1 

mgsImps = getms(items, 5)
TcsAv = np.average(Tcs)
dTcs = np.std(Tcs)
ic(TcsAv, Tc0, Tc1, dTcs)
ax[2].plot(np.arange(len(mgsImps)), Tcs, label = "$T_{c,i}$")
ax[2].axhline(y=TcsAv, label = r"$\langle T_c \rangle_{i}$", color = "orange")
ax[2].axhline(y=Tc0, label = "$T_{c,0}$", color = "green")
ax[2].axhline(y=Tc1, label = "$T_{c,0}(m=0)$", color = "purple")

ax[2].axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
ax[2].axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)
ax[2].set_xlabel("$i$")

ax[2].set_title("m=0.25")

plt.legend(loc = "best", fontsize = "small")

plt.tight_layout()
plt.show()"""


def extract_impval(mg):
    if mg == 0.75:
        items, Tcs, Tc0, Tc1 = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps_new/AMm={mg:.2f}mz=0.0((16, 20), 30, 1.0, 0.2)2.npy", allow_pickle=True)
    else:
        items, Tcs, Tc0, Tc1 = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps_new/AMm={mg:.2f}mz=0.0((16, 20), 30, 1.0, 0.2).npy", allow_pickle=True)
    Tcs, Tc0, Tc1 = Tcs/Tc1, Tc0/Tc1, Tc1/Tc1 
    ic(Tcs, Tc0, Tc1)
    TcsAv = np.average(Tcs)
    dTcs = np.std(Tcs)
    return Tcs, Tc0, TcsAv, dTcs

def plot_imps(mgs):
    Tcs = np.zeros((len(mgs), 100))
    Tcs_without_imp = np.zeros_like(mgs)
    Tcs_upper = np.zeros_like(mgs)
    Tcs_lower = np.zeros_like(mgs)
    Tcs_av = np.zeros_like(mgs)

    dTs = np.zeros_like(mgs)
    for i, mg in enumerate(mgs):
        ic(i, mg)
        ic(extract_impval(mg))
        Tcs[i], Tcs_without_imp[i], Tcs_av[i] , dT = extract_impval(mg)
        # Tcs_av[i] = np.average(Tcs[i])
        Tcs_upper[i] = Tcs_av[i] + dT
        Tcs_lower[i] = Tcs_av[i] - dT
        dTs[i] = dT

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 28
    })
    fig, ax = plt.subplots()

    ax.plot(mgs, Tcs_av / Tcs_av[0], "-x", label = "Impurity average")
    ic(dTs)
    # plt.errorbar(mgs, Tcs_av/ Tcs_av[0], dTs, label = "Impurity average")
    # ax.scatter(mgs, Tcs_av, label = "Imp av", marker ="x")

    ax.plot(mgs, Tcs_without_imp/Tcs_without_imp[0], "--x" ,  label = "Clean system")
    # ax.scatter(mgs, Tcs_without_imp,, marker ="x")

    plt.legend(loc = "upper left", fontsize = "small")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.9, 1.15)
    plt.tight_layout()
    plt.show()

mgs = np.array([0.0, 0.01, 0.02,0.03, 0.04, 0.05, 0.06, 0.07, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,0.14, 0.15, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25,0.65,0.66, 0.67, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72,0.73, 0.74, 0.75])
plot_imps(mgs)
# NDelta = 10
# itemsssAM, TcsssImps_AM, Tc0AM = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps/AMm=0.5mz=0((20, 20), 10, 1.0, 0.2).npy", allow_pickle=True)
# mgsImps = getms(itemsssAM, 5)
# # print(mgsImps)
# TcsAv = np.average(TcsssImps_AM)
# dTcs = np.std(TcsssImps_AM)
# ic(dTcs)
# ic(Tc0AM, TcsAv, TcsAv / Tc0AM)
# # plt.rcParams.update({
# #         "text.usetex": True,
# #         "font.family": "Times",
# #         'font.size'   : 28
# #     })
# fig, ax = plt.subplots()


# ax.plot(np.arange(len(mgsImps)), TcsssImps_AM, label = "$T_{c,i}$")
# ax.axhline(y=np.average(TcsssImps_AM), label = r"$\langle T_c \rangle_{i}$", color = "orange")
# ax.axhline(y=Tc0AM, label = "$T_{c,0}$", color = "green")
# ax.axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
# ax.axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)

# ax0[1].plot(np.arange(len(mgsImps)), TcsssImps_AM, label = "$T_{c,i}$")
# ax0[1].axhline(y=np.average(TcsssImps_AM), label = r"$\langle T_c \rangle_{i}$", color = "orange")
# ax0[1].axhline(y=Tc0AM, label = "$T_{c,0}$", color = "green")
# ax0[1].axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
# ax0[1].axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)

# ax.set_ylabel("$T_c$")
# ax.set_xlabel("$i$")
# plt.legend(loc = "best", fontsize = "small")

# plt.tight_layout()
# # plt.rcParams.update({
# #         "text.usetex": True,
# #         "font.family": "Times",
# #         'font.size'   : 22
# #     })

# # plt.show()

# NDelta = 10
# itemsssFM, TcsssImps_FM, Tc0FM = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps/FMm=0.0mz=0.5((20, 20), 10, 1.0, 0.2).npy", allow_pickle=True)
# mzsImps = getms(itemsssFM, 3)
# TcsAv = np.average(TcsssImps_FM)
# dTcs = np.std(TcsssImps_FM)
# ic(dTcs)
# ic(Tc0FM, TcsAv, TcsAv / Tc0FM)
# # print(mzsImps)
# fig, ax = plt.subplots()

# ax.plot(np.arange(len(mzsImps)), TcsssImps_FM, label = "$T_{c,i}$")
# ax.axhline(y=np.average(TcsssImps_FM), label = r"$\langle T_c \rangle_{i}$", color = "orange")
# ax.axhline(y=Tc0FM, label = "$T_{c,0}$", color = "green")
# ax.axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
# ax.axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)

# ax0[2].plot(np.arange(len(mzsImps)), TcsssImps_FM, label = "$T_{c,i}$")
# ax0[2].axhline(y=np.average(TcsssImps_FM), label = r"$\langle T_c \rangle_{i}$", color = "orange")
# ax0[2].axhline(y=Tc0FM, label = "$T_{c,0}$", color = "green")
# ax0[2].axhline(y = TcsAv + dTcs, color = "orange", ls = "dashed", lw = 1.)
# ax0[2].axhline(y = TcsAv - dTcs, color = "orange", ls = "dashed", lw = 1.)


# ax.set_ylabel("$T_c$")
# ax.set_xlabel("$i$")
# plt.legend(loc = "best", fontsize = "small")

# plt.tight_layout()
# # plt.rcParams.update({
# #         "text.usetex": True,
# #         "font.family": "Times",
# #     })

# plt.show()


