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
    items, Tcs, Tc0, Tc1 = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/imps_new/AMm={mg:.2f}mz=0.0((16, 20), 30, 1.0, 0.2).npy", allow_pickle=True)
    Tcs, Tc0, Tc1 = Tcs/Tc1, Tc0/Tc1, Tc1/Tc1 
    ic(Tcs, Tc0, Tc1)
    TcsAv = np.average(Tcs)
    dTcs = np.std(Tcs)
    return Tcs, Tc0, TcsAv, dTcs


def plot_imps_scaled_by_noimp(mgs):
    Tcs = np.zeros((len(mgs), 100))
    Tcs_without_imp = np.zeros_like(mgs)
    Tcs_upper = np.zeros_like(mgs)
    Tcs_lower = np.zeros_like(mgs)
    Tcs_av = np.zeros_like(mgs)

    dTs = np.zeros_like(mgs)
    for i, mg in enumerate(mgs):
        # ic(i, mg)
        # ic(extract_impval(mg))
        Tcs[i], Tcs_without_imp[i], Tcs_av[i] , dT = extract_impval(mg)
        # Tcs_av[i] = np.average(Tcs[i])
        Tcs_upper[i] = Tcs_av[i] + dT
        Tcs_lower[i] = Tcs_av[i] - dT
        dTs[i] = dT
    plt.style.use('seaborn-v0_8-dark-palette')

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 16
    })
    fig, ax = plt.subplots()

    ax.plot(mgs, Tcs_av / Tcs_without_imp[0], "-x",markersize = 4, label = "Impurity average")
    # ic(dTs)
    # plt.errorbar(mgs, Tcs_av/ Tcs_av[0], dTs, label = "Impurity average")
    # ax.scatter(mgs, Tcs_av, label = "Imp av", marker ="x")

    ax.plot(mgs, Tcs_without_imp/Tcs_without_imp[0], "-x" , markersize = 4,  label = "Clean system")
    # ax.scatter(mgs, Tcs_without_imp,, marker ="x")

    plt.legend(loc = "upper left", fontsize = "small", frameon = False )
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0.91, 1.15)
    ax.set_ylabel(r"$T_c(m)/T_{c, 0}(m=0)$")
    ax.set_xlabel(r"Altermagnetic strength $m$")
    plt.tight_layout()
    # plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figs/imp_both.pdf", format = "pdf", bbox_inches="tight")
    # plt.show()

def plot_imps_norm_to_one(mgs):
    Tcs = np.zeros((len(mgs), 100))
    Tcs_without_imp = np.zeros_like(mgs)
    Tcs_upper = np.zeros_like(mgs)
    Tcs_lower = np.zeros_like(mgs)
    Tcs_av = np.zeros_like(mgs)

    dTs = np.zeros_like(mgs)
    for i, mg in enumerate(mgs):
        # ic(i, mg)
        # ic(extract_impval(mg))
        Tcs[i], Tcs_without_imp[i], Tcs_av[i] , dT = extract_impval(mg)
        # Tcs_av[i] = np.average(Tcs[i])
        Tcs_upper[i] = Tcs_av[i] + dT
        Tcs_lower[i] = Tcs_av[i] - dT
        dTs[i] = dT

    plt.style.use('ggplot')

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 18
    })
    fig, ax = plt.subplots(figsize = (6, 4))

    ax.plot(mgs, Tcs_av / Tcs_av[0], "-x", markersize = 4, label = "Impurity average")
    # ic(dTs)
    # plt.errorbar(mgs, Tcs_av/ Tcs_av[0], dTs, label = "Impurity average")
    # ax.scatter(mgs, Tcs_av, label = "Imp av", marker ="x")

    ax.plot(mgs, Tcs_without_imp/Tcs_without_imp[0], "-x" , markersize = 4,  label = "Clean system")
    # ax.scatter(mgs, Tcs_without_imp,, marker ="x")

    plt.legend(loc = "upper left", fontsize = "small", frameon=False)
    ax.set_xlim(0, 1.01)
    ax.set_ylim(0.91, 1.15)
    ax.set_ylabel(r"$T_c(m)/T_{c}(m=0)$")
    ax.set_xlabel(r"Altermagnetic strength $m$")
    plt.tight_layout
    # plt.show()


def plot_imps_both(mgs):
    Tcs = np.zeros((len(mgs), 100))
    Tcs_without_imp = np.zeros_like(mgs)
    Tcs_upper = np.zeros_like(mgs)
    Tcs_lower = np.zeros_like(mgs)
    Tcs_av = np.zeros_like(mgs)

    dTs = np.zeros_like(mgs)
    for i, mg in enumerate(mgs):
        # ic(i, mg)
        # ic(extract_impval(mg))
        Tcs[i], Tcs_without_imp[i], Tcs_av[i] , dT = extract_impval(mg)
        # Tcs_av[i] = np.average(Tcs[i])
        Tcs_upper[i] = Tcs_av[i] + dT
        Tcs_lower[i] = Tcs_av[i] - dT
        dTs[i] = dT

    plt.style.use('bmh')

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        'font.size'   : 20
    })
    fig, ax = plt.subplots(1,2, figsize = (14, 6), sharey=True)

    ax[0].plot(mgs, Tcs_av / Tcs_av[0], "-x", markersize = 4, label = "Impurity average")
    # plt.errorbar(mgs, Tcs_av/ Tcs_av[0], dTs, label = "Impurity average")
    # ax.scatter(mgs, Tcs_av, label = "Imp av", marker ="x")

    ax[0].plot(mgs, Tcs_without_imp/Tcs_without_imp[0], "-x" , markersize = 4,  label = "Clean system")
    # ax.scatter(mgs, Tcs_without_imp,, marker ="x")

    ax[0].set_xlim(0, 1.01)
    ax[0].set_ylim(0.91, 1.15)
    ax[0].set_ylabel(r"$T_c(m)/T_{c}(m=0)$")
    ax[0].set_xlabel(r"$m/t$")

    ax[1].plot(mgs, Tcs_av / Tcs_without_imp[0], "-x",markersize = 4)#, label = "Impurity average")
    # ic(dTs)
    # plt.errorbar(mgs, Tcs_av/ Tcs_av[0], dTs, label = "Impurity average")
    # ax.scatter(mgs, Tcs_av, label = "Imp av", marker ="x")

    ax[1].plot(mgs, Tcs_without_imp/Tcs_without_imp[0], "-x" , markersize = 4)#,  label = "Clean system")
    # ax.scatter(mgs, Tcs_without_imp,, marker ="x")
    # plt.legend(loc = "upper left", fontsize = "small", frameon = False )
    ax[1].set_xlim(0, 1.01)
    ax[1].set_ylim(0.91, 1.15)
    ax[1].set_ylabel(r"$T_c(m)/T_{c, 0}(m=0)$")
    ax[1].set_xlabel(r"$m/ t$")

    fig.legend(loc = (0.1, 0.75), fontsize = "large", frameon=False)

    plt.tight_layout()
    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figs/imp_both.pdf", format = "pdf", bbox_inches="tight")

    plt.show()

mgs = np.array([0.0, 0.01, 0.02,0.03, 0.04, 0.05, 0.06, 0.07, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13,0.14, 0.15, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25,0.65,0.66, 0.67, 0.67, 0.68, 0.69, 0.70, 0.71, 0.72,0.73, 0.74, 0.75])

mgs = np.concatenate([np.arange(0, 0.37, 0.01), [0.41, 0.42], np.arange(0.44, 0.62, 0.01), np.arange(0.63, 0.84, 0.01), np.arange(0.85, 1.01, 0.01)])#, np.arange(0.79, 0.82, 0.01), [0.86, 0.91], np.arange(0.93, 0.94, 0.01), [0.96], np.arange(0.99, 1.01, 0.01)])
# plot_imps_norm_to_one(mgs)
# plot_imps_scaled_by_noimp(mgs)
mgs = np.arange(0, 1.02, 0.01)
plot_imps_both(mgs)