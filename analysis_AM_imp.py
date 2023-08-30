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


def plot_w_N(m):
    items,cx_up_av, cy_up_av, cx_dn_av, cy_dn_av, wMax, NfracMax = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/AM_imps_wN/m={m:.2f}((20, 20), 10, 10, 100).npy", allow_pickle=True)
    plt.style.use('bmh')
    plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Times",
                'font.size'   : 20
            })
    fig, ax = plt.subplots(figsize = (6.6, 5.5))
    # plt.tight_layout()

    ic(np.amin(cx_up_av - cy_up_av))
    ic(np.amax(cx_up_av - cy_up_av))
    im = ax.imshow(cx_up_av - cy_up_av, vmin = 0, vmax = np.amax(cx_up_av - cy_up_av), interpolation= "nearest", extent = [0, wMax, 0, NfracMax], origin = "lower", aspect="auto", cmap = "RdBu_r") #, interpolarion = "nearest"

    # ax[1].imshow(cy_up_noimp)
    # im = ax[1].imshow(cy_up_av, vmin = 0, vmax = 0.3)
    ax.set_xlabel("$w_i$")
    ax.set_ylabel(r"$N_{i}$")

    fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.75])
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)    
    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figs/wN_m={m:.2f}.pdf", format = "pdf", bbox_inches="tight")

    plt.show()


    # fig, ax = plt.subplots(2, sharex = True)
    # ax[0].plot(ws, cx_up_noimp[1, :], label = "cx noimp", color = "red", linestyle = "dashed")
    # ax[0].plot(ws,    cx_up_av[1, :], label = "cx imp av", color = "red")
    # ax[0].plot(ws, cy_up_noimp[1, :], label = "cy noimp", color = "b", linestyle = "dashed")
    # ax[0].plot(ws,    cy_up_av[1, :], label = "cy imp av", color = "b" )
    # ax[0].set_ylabel(r"Correlations")


    # ax[1].plot(ws, cx_up_noimp[-1, :], label = "cx noimp", color = "red", linestyle = "dashed")
    # ax[1].plot(ws,    cx_up_av[-1, :], label = "cx imp av", color = "red")
    # ax[1].plot(ws, cy_up_noimp[-1, :], label = "cy noimp", color = "b", linestyle = "dashed")
    # ax[1].plot(ws,    cy_up_av[-1, :], label = "cy imp av", color = "b" )

    # plt.legend(loc = "best")
    # ax[1].set_xlabel(r"$w_i$")
    # ax[1].set_ylabel(r"Correlations")

    # # plt.ylabel("Correlations")
    # plt.show()


def plot_m_N(w):
    items, cx_up_av, cy_up_av, cx_dn_av, cy_dn_av, mMax, NfracMax = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/AM_imps_mN/w={w:.1f}((20, 20), 20, 20, 100).npy", allow_pickle=True)
    plt.style.use('bmh')
    plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Times",
                'font.size'   : 20
            })
    fig, ax = plt.subplots(figsize = (6.6, 5.5))
    # plt.tight_layout()

    ic(np.amin(cx_up_av - cy_up_av))
    ic(np.amax(cx_up_av - cy_up_av))
    im = ax.imshow(cx_up_av - cy_up_av, vmin = 0, vmax = np.amax(cx_up_av - cy_up_av), interpolation= "nearest", extent = [0, mMax, 0, NfracMax], origin = "lower", aspect="auto", cmap = "RdBu_r") #, interpolarion = "nearest"

    # ax[1].imshow(cy_up_noimp)
    # im = ax[1].imshow(cy_up_av, vmin = 0, vmax = 0.3)
    ax.set_xlabel("$m/t$")
    ax.set_ylabel(r"$N_{imp}$")

    fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.75])
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)    
    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figs/mN_w={w:.2f}.pdf", format = "pdf", bbox_inches="tight")

    plt.show()


    """numvals_Ni = len(cx_up_av[0, :])
    numvals_m = len(cx_up_av[:, 0])
    mgs = np.linspace(0, 1, numvals_m)
    Nis = np.linspace(0, NfracMax, numvals_Ni)
    print("plotting for ", Nis[numvals_Ni//2])
    fig, ax = plt.subplots(figsize = (8, 5.5))

    ax.plot(mgs, cx_up_av[0, :], label = "cx noimp", color = "red", linestyle = "dashed")
    ax.plot(mgs,    cx_up_av[ numvals_Ni//2, :], label = "cx imp av", color = "red")

    ax.plot(mgs, cy_up_av[ 0, :], label = "cy noimp", color = "b", linestyle = "dashed")
    ax.plot(mgs,    cy_up_av[ numvals_Ni//2, :], label = "cy imp av", color = "b" )
    ax.set_ylabel(r"Correlations")


    # ax[1].plot(mgs, cx_up_noimp[-1, :], label = "cx noimp", color = "red", linestyle = "dashed")
    # ax[1].plot(mgs,    cx_up_av[-1, :], label = "cx imp av", color = "red")
    # # ax[1].plot(mgs, cy_up_noimp[-1, :], label = "cy noimp", color = "b", linestyle = "dashed")
    # ax[1].plot(mgs,    cy_up_av[-1, :], label = "cy imp av", color = "b" )

    plt.legend(loc = "best")
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"Correlations")
    plt.show()"""



def plot_m_mu():
    items,cx_up_av, cy_up_av, cx_dn_av, cy_dn_av, mmMax, muMin, muMax  = np.load(f"C:/Users/hansggi/OneDrive - NTNU/BdG/Newdata4/AM_imps_mmu/((20, 20), 100, 100, 1).npy", allow_pickle=True)
    plt.style.use('bmh')
    plt.rcParams.update({
                "text.usetex": True,
                "font.family": "Times",
                'font.size'   : 20
            })
    

    fig, ax = plt.subplots(figsize = (6.6, 5.5))
    # plt.tight_layout()

    ic(np.amin(cx_up_av - cy_up_av))
    ic(np.amax(cx_up_av - cy_up_av))
    ic(muMin, muMax)
    # im = ax.imshow((cx_up_av - cy_up_av)[ :, :], vmin = 0, vmax = np.amax(cx_up_av - cy_up_av), interpolation= "nearest", extent = [0, mmMax, muMin, muMax], origin = "lower", aspect="auto", cmap = "RdBu_r") #, interpolarion = "nearest"

    im = ax.imshow((cx_up_av - cy_up_av)[ ::-1, :], vmin = 0, vmax = np.amax(cx_up_av - cy_up_av), interpolation= "nearest", extent = [0, mmMax, np.abs(muMax), np.abs(muMin)], origin = "lower", aspect="auto", cmap = "RdBu_r") #, interpolarion = "nearest"

    # ax[1].imshow(cy_up_noimp)
    # im = ax[1].imshow(cy_up_av, vmin = 0, vmax = 0.3)
    ax.set_xlabel("$m/t$")
    ax.set_ylabel(r"$-\mu/t$")

    fig.subplots_adjust(right=0.8)
    # cbar_ax = fig.add_axes([0.85, 0.25, 0.05, 0.75])
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)    
    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figs/mmu.pdf", format = "pdf", bbox_inches="tight")
    plt.show()

plot_m_N(w=1)

plot_m_N(w=3)

# plot_w_N(m=0.25)
# plot_w_N(m=0.75)

plot_m_mu()