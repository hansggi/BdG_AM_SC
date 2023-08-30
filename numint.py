import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fsolve, root
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
    
    return  - M - hz  + sigma* np.sqrt(xi(kx, ky)**2 + Delta**2)

def integrand(kx, ky, Delta, args):
    # ic(f(E(kx, ky, +1, Delta, args)) -f(-E(kx, ky, -1, Delta, args)))
    hz, m, hx = args

    M = 2* m * (np.cos(kx) - np.cos(ky))

    return  (1 - f(E(kx, ky, +1, Delta, args)) - f(-E(kx, ky, -1, Delta, args))) / np.sqrt(xi(kx, ky)**2 + Delta**2)

# def dEdDelta(kx, ky, alpha, beta, Delta, args):
#     hz, m, hx = args

#     M = 2* m * (np.cos(kx) - np.cos(ky))

#     return alpha * Delta * (1 + beta * np.sqrt(M**2 + hx**2) / np.sqrt(Delta**2 + xi(kx, ky)**2)) / np.sqrt( M**2 + hx**2 + xi(kx, ky)**2 + Delta**2 + 2*beta * np.sqrt((xi(kx, ky)**2 + Delta**2)* ( M**2 + hx**2) ))


U = 1.7
Nx = 500
Ny = 500

t = 1.
mu = -0.5
E0 = E(0, 0, 1, 0.1, args= np.array([0,0,0]))
print(E0)


# T = 0.05*Delta0
T = 0.01*t

hz = 0.0
m = 0.
hx = 0.
if T == 0:
    beta = np.inf
else:
    beta = 1 / T

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

Delta0 = fsolve(g, 0.01, args = np.array([0,0,0]))[0]
ic(Delta0)






def analytical_run_m():
    Delta0 = fsolve(g, 0.01, args = np.array([0,0,0]))[0]
    # ic(Delta0)

    n = 200 # number of Delta values
    Delta_plot = np.linspace(0, 1.4*Delta0, n)
    N = 7 # number of h/m values
    hz_plots= np.linspace(0, 0, N)
    hx_plots= np.linspace(0, 0, N)

    m_plots = np.linspace(0, 0.35*Delta0, N)

    args = np.zeros((N, 3))
    for i in range(N):
        args[i] = [hz_plots[i], m_plots[i], hx_plots[i]]


    gs = np.zeros((N, len(Delta_plot)))

    for j in range(N):
        for i in range(len(Delta_plot)):
            gs[j,i] = g(Delta_plot[i], args = args[j])

    plt.style.use('bmh')
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times",
            'font.size'   : 20
        })
    
    fig, ax = plt.subplots()

    for j in range(N):
        # ax.plot(Delta_plot / Delta0, gs[j, :], label = f"m={(m_plots[j]/ Delta0):.2f}$\Delta_0$")
        ax.plot(Delta_plot / Delta0, gs[j, :], label = f"$m$ = {(m_plots[j]):.2f}$t$")



    ax.axhline(0, linestyle = "dashed", color = "black", lw = 1)
    ax.axvline(1, linestyle = "dashed", color = "black", lw = 1)

    ax.set_ylim(-0.75, 0.15)

    ax.set_xlabel(f"$\Delta(m)/\Delta_0$")
    ax.set_ylabel(f"g[$\Delta(m)$]")
    
    plt.tight_layout()
    ax.legend(loc = (0.56, -0.03), fontsize = "medium" , ncol = 1, frameon=False)
    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figs/gm.pdf", format = "pdf", bbox_inches="tight")

    plt.show()


def analytical_run_h():

    Delta0 = fsolve(g, 0.01, args = np.array([0,0, 0]))[0]

    n = 200 # number of Delta values
    Delta_plot = np.linspace(0, 1.4*Delta0, n)
    N = 7 # number of h/m values

    hz_plots= np.linspace(0, 1.0*Delta0, N)
    hx_plots= np.linspace(0, 1.0*Delta0*0, N)

    m = 0.*Delta0

    m_plots = np.ones(N)*m

    args = np.zeros((N, 3))
    for i in range(N):
        args[i] = [hz_plots[i], m_plots[i], hx_plots[i]]

    gs = np.zeros((N, len(Delta_plot)))

    for j in range(N):
        for i in range(len(Delta_plot)):
            gs[j,i] = g(Delta_plot[i], args = args[j])
            # ic(gs[j,i])
    linestyles = ["dashdot", "dashdot", "dashed", "dashed"]
    linestyles = ["solid"]*N

    plt.style.use('bmh')
    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times",
            'font.size'   : 20
        })
    fig, ax = plt.subplots()

    for j in range(N):#, len(h_plots)):
        # ax.plot(Delta_plot / Delta0, gs[j, :], label = f"h={(hz_plots[j]/ Delta0):.2f}$\Delta_0$", ls = linestyles[j])
        ax.plot(Delta_plot / Delta0, gs[j, :], label = f"$m_z$ = {(hz_plots[j]):.2f}$t$", ls = linestyles[j])

    
    
    ax.axhline(0, linestyle = "dashed", color = "black", lw = 1)
    ax.axvline(1, linestyle = "dashed", color = "black", lw = 1)

    # ax.xlim(0, 1.25 )


    ax.set_ylim(-0.8, 0.15)
    # ax.set_yticks([-0.6, -0.2, 0, 0.2])
    ax.set_xlabel(f"$\Delta(m_z)/\Delta_0$")
    ax.set_ylabel(f"g[$\Delta(m_z)$]")

    plt.tight_layout()
    ax.legend(loc = (0.56, -0.03), fontsize = "medium" , ncol = 1, frameon=False)
    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figs/gh.pdf", format = "pdf", bbox_inches="tight")

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


def Delta_plot_m():
    N = 50
    T = 0.1
    Delta0 = fsolve(g, 0.01, args = np.array([0,0, 0]))[0]

    hz_plots= np.linspace(0, 1.2*Delta0, N)
    hx_plots= np.linspace(0, 0, N)

    m_plots = np.linspace(0, 0.35*Delta0*0, N)

    args = np.zeros((N, 3))
    for i in range(N):
        args[i] = [hz_plots[i], m_plots[i], hx_plots[i]]

    Deltas = np.zeros(N)

    for i in range(N):
        # Deltas[i] = np.abs(fsolve(g, -Delta0/100, args = args[i], maxfev=1000, )[0])
        Deltas[i] = np.abs(root(g, [Delta0*2*0,], args = args[i]).x)
        ic(args[i, 0]/Delta0, Deltas[i])
    ic(Deltas)
    
    fig, ax = plt.subplots()
    # ax.plot(m_plots, Deltas)
    ax.plot(hz_plots/ Delta0, Deltas)

    plt.show()


def calc_free_mg():
    N = 200
    Deltas = np.linspace(-1.5*Delta0, 1.5*Delta0, N)
    args = np.array((hz,m, hx)) #h, 1
    n = 7
    m_plots = np.linspace(0, 0.35*Delta0, n)
    hz_plots = np.linspace(0, 0 * Delta0, n)
    hx_plots = np.linspace(0, 0*Delta0, n)


    args = np.zeros((N, 3))
    for i in range(n):
        args[i] = [hz_plots[i], m_plots[i], hx_plots[i]]
        
    Fs = np.zeros((n, N))


    plt.style.use('bmh')

    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times",
            'font.size'   : 20
        })

    fig, ax = plt.subplots()

    for j in range(n):
        for i in range(N):
            Fs[j, i] = F(Deltas[i], args[j])

        # ax.plot(Deltas / Delta0 , Fs[j, :], label = f"m = {(m_plots[j]/Delta0):.2f}$\Delta_0$")
        ax.plot(Deltas / Delta0 , Fs[j, :], label = f"$m$ = {(m_plots[j]):.2f}$t$")




    ax.axvline(1, ls = "dashed", lw = 1, color = "black")
    ax.legend(loc = (0.05, -0.02), fontsize = "medium" , ncol = 2, frameon=False)
    ax.set_xlabel("$\Delta(m)/\Delta_0$")
    ax.set_ylabel("$F/N$")
    ax.set_ylim(-1.683, -1.673)

    ax.set_yticks([-1.68, -1.675])
    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figs/F_m.pdf", format = "pdf", bbox_inches="tight")
    plt.show()
    
def calc_free_mz():
    N = 200
    Deltas = np.linspace(-1.5*Delta0, 1.5*Delta0, N)
    args = np.array((hz,m, hx)) #h, 1
    n = 7
    m_plots = np.linspace(0, 0.35*Delta0*0, n)
    hz_plots = np.linspace(0, 1 * Delta0, n)
    hx_plots = np.linspace(0, 0*Delta0, n)


    args = np.zeros((N, 3))
    for i in range(n):
        args[i] = [hz_plots[i], m_plots[i], hx_plots[i]]
        
    Fs = np.zeros((n, N))


    plt.style.use('bmh')

    plt.rcParams.update({
            "text.usetex": True,
            "font.family": "Times",
            'font.size'   : 20
        })

    fig, ax = plt.subplots()

    for j in range(n):
        for i in range(N):
            Fs[j, i] = F(Deltas[i], args[j])

        # ax.plot(Deltas / Delta0 , Fs[j, :], label = f"m = {(hz_plots[j]/Delta0):.2f}$\Delta_0$")
        ax.plot(Deltas / Delta0 , Fs[j, :], label = f"$m_z$ = {(hz_plots[j]):.2f}$t$")




    ax.axvline(1, ls = "dashed", lw = 1, color = "black")
    ax.legend(loc = (0.06, -0.02), fontsize = "medium" , ncol = 2, frameon=False)
    ax.set_xlabel("$\Delta(m_z)/\Delta_0$")
    ax.set_ylabel("$F/N$")
    ax.set_ylim(-1.685, -1.673)
    ax.set_yticks([-1.685, -1.673])

    plt.savefig(f"C:/Users/hansggi/OneDrive - NTNU/BdG/figs/F_h.pdf", format = "pdf", bbox_inches="tight")
    plt.show()


# T = 0.0001
if T == 0:
    beta = np.inf
else:
    beta = 1 / T

# analytical_run_m()
# analytical_run_h()

# calc_free_mg()
# calc_free_mz()

# Delta_plot_m()