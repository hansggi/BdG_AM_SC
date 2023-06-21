import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve
from icecream import ic
import time

def f(E):
    return 1 / (1 + np.exp(beta * E))

def xi(kx, ky):
    return - 2 * t * (np.cos(kx) + np.cos(ky)) - mu

def E(kx, ky, sigma, Delta, args):
    hz, m, hx = args
    # ic(args)
    M = 2* m * (np.cos(kx) - np.cos(ky))
    # ic(np.allclose(-np.sqrt(M**2), M))
    # return - hz  - np.sqrt(M**2 + hx**2) + sigma * np.sqrt(xi(kx, ky)**2 + Delta**2)
    # ic((xi(kx, ky)**2 + Delta**2)* ( M**2 + hx**2))
    
    return  - M - hx  + sigma* np.sqrt(xi(kx, ky)**2 + Delta**2)

def integrand(kx, ky, Delta, args):
    # ic(f(E(kx, ky, +1, Delta, args)) -f(-E(kx, ky, -1, Delta, args)))
    hz, m, hx = args

    M = 2* m * (np.cos(kx) - np.cos(ky))
    # dEdDpp = dEdDelta(kx, ky, 1, 1, Delta, args)
    # dEdDpn = dEdDelta(kx, ky, 1, -1, Delta, args)
    # dEdDnp = dEdDelta(kx, ky, -1, 1, Delta, args)
    # dEdDnn = dEdDelta(kx, ky, -1, -1, Delta, args)

    # return   (1/ dEdDnn - 1/dEdDnp - f(E(kx, ky, +1, +1, Delta, args)) / dEdDpp - f(E(kx, ky, +1, -1, Delta, args))/dEdDpn - f(E(kx, ky, -1, +1, Delta, args))/dEdDnp -f(E(kx, ky, -1, -1, Delta, args)/dEdDnn ) )
    return  (1 - f(E(kx, ky, +1, Delta, args)) - f(-E(kx, ky, -1, Delta, args))) / np.sqrt(xi(kx, ky)**2 + Delta**2)

# def dEdDelta(kx, ky, alpha, beta, Delta, args):
#     hz, m, hx = args

#     M = 2* m * (np.cos(kx) - np.cos(ky))

#     return alpha * Delta * (1 + beta * np.sqrt(M**2 + hx**2) / np.sqrt(Delta**2 + xi(kx, ky)**2)) / np.sqrt( M**2 + hx**2 + xi(kx, ky)**2 + Delta**2 + 2*beta * np.sqrt((xi(kx, ky)**2 + Delta**2)* ( M**2 + hx**2) ))

def g(Delta, args):
    # h, m = arg
    # ic(args)
    kx = np.linspace(0, np.pi, Nx)
    ky = np.linspace(0, np.pi, Ny)
    XX, YY = np.meshgrid(kx, ky)
    N = Nx*Ny

    gsum = 1 - U / 2 / N*  np.sum(integrand(XX, YY, Delta, args))
    # print(np.sum(integrand(XX, YY, Delta, args)))
    return gsum   

# Es = np.linspace(- 5, 5, 1000)

# plt.plot(Es, E(Es)) 
# plt.show()


def analytical_run_m():

    Delta0 = fsolve(g, 0.01, args = np.array([0,0, 0]))[0]
    ic(Delta0)

    n = 200 # number of Delta values
    Delta_plot = np.linspace(0, 1.4*Delta0, n)
    N = 7 # number of h/m values
    hz_plots= np.linspace(0, 0, N)
    hx_plots= np.linspace(0, 0, N)

    # h = 0.4*Delta0
    # ic(h)
    # ic(h/Delta0)

    # h_plots = np.ones(N) * h

    m_plots = np.linspace(0, 0.35*Delta0, N)

    args = np.zeros((len(Delta_plot), 3))
    for i in range(N):
        args[i] = [hz_plots[i], m_plots[i], hx_plots[i]]


    gs = np.zeros((N, len(Delta_plot)))

    for j in range(N):
        for i in range(len(Delta_plot)):
            gs[j,i] = g(Delta_plot[i], args = args[j])


    for j in range(N):
        plt.plot(Delta_plot / Delta0, gs[j, :], label = f"m={(m_plots[j]/ Delta0):.2f} $\Delta_0$")


    plt.axhline(0, linestyle = "dashed", color = "black", lw = 1)
    plt.axvline(1, linestyle = "dashed", color = "black", lw = 1)

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
    })
    # plt.xlim(0, 1.25 )
    plt.ylim(-0.5, 0.15)

    plt.xlabel(f"$\Delta(m)/\Delta_0$", fontsize = 22)
    plt.ylabel(f"g[$\Delta(m)$]", fontsize = 22)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
    })
    plt.tight_layout()
    plt.legend(loc = "best", fontsize = 14)
    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/g_m_mu={mu}_h={hx:.2f}T={T:.2f}.pdf", format = "pdf", bbox_inches="tight")

    plt.show()


def analytical_run_h():

    Delta0 = fsolve(g, 0.01, args = np.array([0,0, 0]))[0]
    # ic(Delta0)
    # Delta0 = 0.5
    # print(Delta0)
    n = 200 # number of Delta values
    Delta_plot = np.linspace(0, 1.4*Delta0, n)
    N = 7 # number of h/m values

    hz_plots= np.linspace(0, 1.2*Delta0*0, N)
    hx_plots= np.linspace(0, 1.0*Delta0, N)

    m = 0.*Delta0

    m_plots = np.ones(N)*m

    args = np.zeros((len(Delta_plot), 3))
    for i in range(N):
        args[i] = [hz_plots[i], m_plots[i], hx_plots[i]]

    gs = np.zeros((N, len(Delta_plot)))

    for j in range(N):
        for i in range(len(Delta_plot)):
            gs[j,i] = g(Delta_plot[i], args = args[j])
            # ic(gs[j,i])
    linestyles = ["dashdot", "dashdot", "dashed", "dashed"]
    linestyles = ["solid"]*N
    for j in range(N):#, len(h_plots)):
        # plt.plot(Delta_plot / Delta0, gs[j, :], label = f"h={(h_plots[j]/ Delta0):.2f} $\Delta_0$")
        plt.plot(Delta_plot / Delta0, gs[j, :], label = f"h={(hx_plots[j]/ Delta0):.2f} $\Delta_0$", ls = linestyles[j])

        # plt.plot(Delta_plot, gs[1, :])
        # plt.plot(Delta_plot, gs[2, :])

    plt.axhline(0, linestyle = "dashed", color = "black", lw = 1)
    plt.axvline(1, linestyle = "dashed", color = "black", lw = 1)

    # plt.xlim(0, 1.25 )


    plt.ylim(-0.5, 0.3)
    plt.xlabel(f"$\Delta(h)/\Delta_0$", fontsize = 22)
    plt.ylabel(f"g[$\Delta(h)$]", fontsize = 22)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
    })
    plt.tight_layout()
    plt.legend(loc = "best", fontsize = 14)
    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/g_hx_mu={mu}_m={m:.2f}_neghT={T:.2f}.pdf", format = "pdf", bbox_inches="tight")

    plt.show()



def F(Delta, args):
    # The free energy
    hz, m, hx = args
    kx = np.linspace(- np.pi, np.pi, Nx)
    ky = np.linspace(- np.pi, np.pi, Ny)
    XX, YY = np.meshgrid(kx, ky)
    # Nx = len(kx)
    # Ny = len(ky)
    N = Nx*Ny
    # H0 = np.sum(xi(XX, YY))  + N* Delta**2  / U # First term is a constant wrt Delta
    # ic(np.sum(xi(XX, YY)) / N)
    # ic(Delta**2  / U)
    # ic(- 1/ N*np.sum(E(XX, YY, -1, Delta, args)) )
    # ic(- 1/ N*T *np.sum(np.log(1 + np.exp(- beta * 1*E(XX, YY, 1, Delta, args))) + np.log(1 + np.exp(+ beta *E(XX, YY, -1, Delta, args))) ))
    # F = H0 - np.sum(E(XX, YY, -1, Delta, args)) - T *np.sum(np.log(1 + np.exp(- beta *E(XX, YY, 1, Delta, args))) + np.log(1 + np.exp(+ beta *E(XX, YY, -1, Delta, args))) )

    F = N / U * np.abs(Delta)**2 -  T * np.sum(np.log(np.cosh(beta *E(XX, YY, 1, Delta, args) / 2)*np.cosh(-beta *E(XX, YY, -1, Delta, args)/2)))
    # alpha = 0
    # try: np.exp(- beta * E(kx, ky, +1, Delta, args))
    # except RuntimeWarning:alpha = 1

    # F = 1 / U * np.abs(Delta)**2 -  T * np.sum(np.log((1+ np.exp(- beta * E(XX, YY, +1, Delta, args)))+ np.log(1 + np.exp(+ beta * E(XX, YY, -1, Delta, args)))))

    return F / N



U = 1.7
Nx = 500
Ny = 500
T = 0.00
beta = np.inf
t = 1.
mu = -0.5
E0 = E(0, 0, 1, 0.1, args= np.array([0,0,0]))
print(E0)

Delta0 = fsolve(g, 0.01, args = np.array([0,0,0]))[0]
ic(Delta0)
T = 0.05*Delta0


hz = 0.0
m = 0.
hx = 0.
if T == 0:
    beta = np.inf
else:
    beta = 1 / T

# m = 0.
# h = 0
# Deltag = 0.01
# analytical_run_m()

# Deltas = np.linspace(-18*Delta0, 18*Delta0, 40)
def calc_free():
    Deltas = np.linspace(-1.5*Delta0, 1.5*Delta0, 200)
    args = np.array((hz,m, hx)) #h, 1
    n = 7
    m_plots = np.linspace(0, 0.35*Delta0, n)
    hz_plots = np.linspace(0, 0 * Delta0, n)
    hx_plots = np.linspace(0, 0*Delta0, n)


    args = np.zeros((len(Deltas), 3))
    for i in range(n):
        args[i] = [hz_plots[i], m_plots[i], hx_plots[i]]
        
    Fs = np.zeros((n, len(Deltas)))
    for j in range(n):
        for i in range(len(Deltas)):
            Fs[j, i] = F(Deltas[i], args[j])

        plt.plot(Deltas / Delta0 , Fs[j, :], label = f"m = {(m_plots[j]/Delta0):.2f} $\Delta_0$")


    # plt.xlim(Deltas[0] / Delta0, Deltas[-1] / Delta0)
    # plt.plot(Deltas , Fs, label = f"h = {h}")
    plt.axvline(1, ls = "dashed", lw = 1, color = "black")
    plt.legend(loc = "best", fontsize = 14)
    plt.xlabel("$\Delta/\Delta_0$", fontsize = 22)
    plt.ylabel("$F/N$", fontsize = 22)

    plt.tight_layout()
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
    })
    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figsAMSC/F_mT={T:.2f}.pdf", format = "pdf", bbox_inches="tight")
    plt.show()
    # analytical_run_h()


    # analytical_run_h()



    # Delta0 = fsolve(g, 0.01, args = np.array([0,0]))[0]
    # ic(Delta0)
    # hs = np.linspace(0., 0., 5)
    # ms = np.linspace(0, 0.8*Delta0, len(hs))
    # tic = time.time()
    # Deltas = np.zeros_like(hs) 
    # Deltas2 = np.zeros_like(hs)
    # DeltaOld = 0.2
    # for i, m in enumerate(ms):
    #     ic(m)
    #     args = [hs[i],ms[i]]
    #     root = fsolve(g, 0.0, args = args)#, xtol = 1e-6)
    #     Deltas[i] = np.abs(root)
    #     Deltas2[i] = np.abs( fsolve(g, 0.2, args = args))
    #     ic(root)
    #     # ic(g(root, args))
    #     DeltaOld = Deltas[i]
    #     print(np.isclose(g(root, args), 0))

    # ic(time.time() -tic)
    # plt.plot(ms, Deltas, label = "Guess 0")
    # plt.plot(ms, Deltas2, label = "Guess 0.2")
    # plt.legend()
    # plt.show()
    # print(root)
    # print(g(root, args))
# T = 0.0001
if T == 0:
    beta = np.inf
else:
    beta = 1 / T
calc_free()