from matplotlib import pyplot as plt
import numpy as np
import time
# import cProfile
from numba import njit
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from funcs import make_sigmas, nb_block, nb_block2, make_H_numba, fd, Delta_sc, unpack_block_matrix

from multiprocessing import Pool

import sys
from icecream import ic
from concurrent.futures import ProcessPoolExecutor
# @njit()
def does_Delta_increase(Nx, Ny,m_arr, mz_arr,  Deltag, T, param, Delta_arr1, bd,  NDelta, skewed = False):
    # Here, Deltag must be the guess, if Delta < Deltag, 
    t, U, mu, mg, mz = param

    Delta = Delta_arr1.copy()
    for i in range(NDelta):
            H = make_H_numba(Nx, Ny, m_arr, mz_arr, Delta, param, bd,  skewed = skewed)

            D, gamma = np.linalg.eigh(H)
            D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
            Delta = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)

    if bd[1] < Nx:
        Delta_bulk = np.abs(Delta[Ny//2, (bd[0] + bd[1])//2])

    else:
        Delta_bulk = np.abs(Delta[Ny//2, (bd[0] + Nx)//2])

        
        # if Delta_bulk > DeltaT[1] :
        #     # print("Term after", i)
        #     return True
        
        # elif Delta_bulk <= DeltaT[0]:
        #     # print("Term after", i)
        #     return False

    if Delta_bulk <= np.abs(Deltag):
        # print("Ran through,", Delta_bulk)
        return False
    else:
        # print("Ran through", Delta_bulk)
        return True

# @njit(cache = True)
"""def does_Delta_increase_steff(Nx, Ny,m_arr, mz_arr, Deltag, T, param, Delta_arr1, bd,  NDelta, skewed = False):
    # Here, Deltag must be the guess, if Delta < Deltag, 
    t, U, mu, mg, mz = param
    StefNum = 8000
    # Delta_tab = np.zeros((NDelta, Ny, Nx), dtype="complex128")
    # Delta_tab[0, :, :] = Delta_arr1.copy()


    Deltapp = np.zeros_like(Delta_arr1)
    Deltap  = np.zeros_like(Delta_arr1)
    Delta   = Delta_arr1.copy()
    ind = np.nonzero(Delta)

    ic(T)
    for i in range(1, NDelta):
        H = make_H_numba(Nx, Ny, m_arr, mz_arr, Delta, param, bd, skewed = skewed)
        D, gamma = np.linalg.eigh(H)
        D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
        # print(i, np.allclose(Delta_tab[i], 0))
        Deltapp = Deltap.copy()
        Deltap = Delta.copy()
        Delta = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)
        # if np.amin(np.abs(Delta[ind] - Deltap[ind])) < 1e-18:
        #     if np.amax(np.abs(Delta[ind] - Deltap[ind])) < 1e-12:
        #        break

        #     print("Oh Ow, ", np.amin(np.abs(Delta[ind] - Deltap[ind])))
        #     print("Max error is ", np.amax(np.abs(Delta[ind] - Deltap[ind])))
        #     time.sleep(5)
        if i%StefNum==0 and i > 100:
            print("error")
            Delta[ind] = Deltapp[ind] - (Deltap[ind] - Deltapp[ind])**2 / (Delta[ind] - 2 * Deltap[ind] + Deltapp[ind])


    if bd[1] < Nx:
        Delta_bulk = np.abs(Delta[ Ny//2, (bd[0] + bd[1])//2])

    else:
        Delta_bulk = np.abs(Delta[Ny//2, (bd[0] + Nx)//2])

    if Delta_bulk <= np.abs(Deltag):
        return False
    else:
        return True

"""
# @njit(cache = True)
def calc_Delta_sc(Nx, Ny, Deltag, tol, T, param, bd,  skewed):
    t, U, mu, mg, mz = param

    done = False
    Delta_old = (np.ones((Nx*Ny))*Deltag).reshape(Ny, Nx)*(1 + 0j)
    # Delta_old[:, :bd] = 0

    it = 0
    if bd[1] < Nx:
        Delta_old_bulk = np.abs(Delta_old[Ny//2, (bd[0] + bd[1])//2])

    else:
        Delta_old_bulk = np.abs(Delta_old[Ny//2, (bd[0] + Nx)//2])    

    while not done:
        H = make_H_numba(Nx, Ny, Delta_old, param, bd, skewed = skewed)

        D, gamma = np.linalg.eigh(H)
        D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
        # ----------------------------------------------------------------------------------

        Delta_new = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)
        # Delta_new = Delta_new_i.reshape(Ny, Nx)
        if bd[1] < Nx:
            Delta_bulk = np.abs(Delta_new)[Ny//2, (bd[0] + bd[1])//2]
        else:
            Delta_bulk = np.abs(Delta_new)[Ny//2, (bd[0] + Nx)//2]
        it += 1
        # Bulk method
        # if np.abs(Delta_bulk - Delta_old_bulk)  <= tol :
        #     done = True

        # Using max difference instead. Will not give the same plot as in the article, since T is a function of bulk Tc here
        if np.amax(np.abs(np.abs(Delta_new) - np.abs(Delta_old)))  <= tol :
            done = True

        # Delta_old = Delta_new
        Delta_old = Delta_new
        
        Delta_old_bulk = Delta_bulk

    # print("Used ", it, " iterations to calculate Delta self-consist.")
    return Delta_new, gamma, D

# @njit()
def calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd, num_it, skewed, alignment = None):
    N = 15 # Look at, maybe not needed this accuracy
    t, U, mu, mg, mz = param
    if alignment == None:
        assert bd[1] >= Nx
    # The first calculation is the same for all temperatures --------------
    # x = np.arange(0, Nx)
    # Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)#*(x - bd)**2*0.05 / 40**2
    Delta_arr = (np.ones((Nx*Ny))*Deltag).reshape(Ny, Nx)# To make it complex?
    Delta_arr[:, :bd[0]] = 0
    if bd[1] < Nx:
        Delta_arr[:, bd[1]:] = 0

    m_arr = (np.ones((Nx*Ny)) * mg).reshape(Ny, Nx)
    m_arr[:, bd[0]:bd[1]] = 0

    if alignment == "AP":
        m_arr[:, bd[1]:] *= -1
    # plt.imshow(m_arr)
    # plt.show()

    if mz != 0:

        assert alignment == None

    mz_arr = (np.ones((Ny*Nx))*mz).reshape(Ny, Nx)
    mz_arr[:, bd[0]:] = 0
    H = make_H_numba(Nx, Ny, m_arr, mz_arr, Delta_arr, param, bd, skewed)
    # plt.imshow(np.abs(H[::4, ::4]))
    # plt.colorbar()
    # plt.show()
    assert np.allclose(H, np.conjugate(H.T), rtol=1e-08, atol=1e-08, equal_nan=False)

    D, gamma = np.linalg.eigh(H)
    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
    # ---------------------------------------------------------------

    Ts_lower = 0
    Ts_upper = Tc0

    for i in range(N):
        T = (Ts_upper + Ts_lower ) / 2 
        assert Ts_upper > Ts_lower
        Delta_arr1 = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)



        if does_Delta_increase(Nx, Ny,m_arr, mz_arr, Deltag, T, param, Delta_arr1, bd, num_it, skewed=skewed): # Meaning that there is SC at this temp, need to go higher in T to find Tc
            Ts_lower = T
        else:
            Ts_upper = T 

    return (Ts_upper + Ts_lower ) / 2
 

@njit()
def gaussian(E, mu, sigma):
    return 1 / ( sigma * np.sqrt(2 * np.pi)) * np.exp(- (E - mu)**2 / ( 2 * sigma**2))

@njit()
def Ldos(gamma, D):
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]
    s = np.zeros((u_up.shape[0], 5000))
    E = np.linspace(- 4, 4, 5000)
    for i in range(u_up.shape[0]):
        for j in range(len(D)):

            s[i, :] += np.abs(u_up[i, j])**2 * gaussian(E, D[j], 0.1) + np.abs(v_up[i, j])**2 * gaussian(E, -D[j], 0.1)
            s[i, :] += np.abs(u_dn[i, j])**2 * gaussian(E, D[j], 0.1) + np.abs(v_dn[i, j])**2 * gaussian(E, -D[j], 0.1)

    return s

def N_sigma(gamma, D, T):
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]

    # Fuu = np.sum(u_up *np.conjugate(v_up) , axis = 1)
    # print(Fuu)
    f  = (1 - np.tanh(D / (2 * T))) / 2
    N_up = np.sum(u_up *np.conjugate(u_up) * f + v_up * np.conjugate(v_up)* (1 - f) , axis = 1)
    N_dn = np.sum(u_dn *np.conjugate(u_dn) * f + v_dn * np.conjugate(v_dn)* (1 - f) , axis = 1)

    return N_up, N_dn

def pairing_amplitude(gamma, D, T):
    # NB: gamma and energies D must already be just positive eigenvalues
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]

    # Fuu = np.sum(u_up *np.conjugate(v_up) , axis = 1)
    # print(Fuu)
    f  = (1 - np.tanh(D / (2 * T))) / 2
    # -------------------------------------------------------------------
    Fud = np.sum(u_dn *np.conjugate(v_up) * f + u_up * np.conjugate(v_dn)* (1 - f) , axis = 1)
    Fdu = np.sum(u_up *np.conjugate(v_dn) * f + u_dn * np.conjugate(v_up)* (1 - f) , axis = 1)
    # P
    
    # Fpx =  np.sum(u_dn *np.conjugate(np.roll(v_up, 1, axis = 0)) * f + u_up * np.conjugate(v_dn)* (1 - f) , axis = 1)
    #----------
    # print(np.allclose(Fud, -Fdu))
    Fs_i = 0.5*(Fud - Fdu)  # Here, we used equilibrium explicitly to say (1-f(E)) = f(-E) 
    Ft_i = 0.5*(Fud + Fdu)
    return Fs_i, Ft_i


def sweep_Delta(Nx, Ny, mg,mz, U, mu, Deltag, bd, tol, num_it, skewed ):
    Tc0 = 0.3
    t = 1.
    # Calc Tc in the mg = 0 case
    param0 = (t, U, mu, 0, 0)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag, param0, Tc0, bd, num_it, skewed = skewed) 
    # print(Tc)
    # -----------------------------------------------------------
    # Tc in this case
    param = (t, U, mu, mg, mz)
    Tc2 = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd, num_it, skewed = skewed) 
    print(f"Tc :{Tc:.4f}, \nTc2:{Tc2:.4f}.")
    T_plots = np.array([1e-8, 0.5 * Tc , 0.95 * Tc, 0.96*Tc,0.97*Tc, 0.98*Tc, 1.0 * Tc, 1.05*Tc])
    T_plots = np.linspace(0.001, Tc, 10)
    fig, ax = plt.subplots()
    for i, T in enumerate(T_plots):
        Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T_plots[i], param, bd,  skewed=skewed)
        Fs, Ft = pairing_amplitude(gamma, D, T)
        Fs = Fs.reshape(Ny, Nx)
        Ft = Ft.reshape(Ny, Nx)
        if i == 2:
                fig2, ax2 = plt.subplots()
                # divider = make_axes_locatable(ax2)
                # cax = divider.append_axes('right', size='5%', pad=0.05)

                im = ax2.imshow(np.abs(Fs))
                # ax2[1].imshow(np.abs(Ft))
                fig.colorbar(im)

                # fig2.colorbar(im, cax=cax, orientation='vertical')    # plt.savefig(f"1d_Delta/N=({Nx},{Ny}),mg={mg:.1f}.pdf", bbox_inches='tight')
                # ax2.set_title(f"T = {T}")
                # fig2.legend()

        
        # Delta_i = Delta_i.reshape((Ny, Nx))
        ax.plot(np.abs(Fs[Ny//2, :]), label = f"T/Tc = {T/Tc:.2f}S")
        # ax.plot(np.abs(Ft[Ny//2, :]), label = f"T/Tc = {T/Tc:.2f}T", linestyle = "dashed")


    fig.suptitle(f" m ={mg:.1f} mz = {mz} U = {U}  mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    ax.set_ylabel("F_i")
    fig.legend()
    # plt.savefig(f"1d_Delta/N=({Nx},{Ny}),mg={mg:.1f}.pdf", bbox_inches='tight')
    plt.show()

def sweep_Delta_mz(Nx, Ny, mzs, T, U, mu, Deltag, bd, tol, num_it,  skewed ):
    mg = 0
    Tc0 = 0.3
    t = 1.
    # Calc Tc in the mg = 0 case
    param0 = (t, U, mu, 0, 0)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag, param0, Tc0, bd, num_it, skewed = skewed) 
    print("Tc: ", Tc)
    fig3, ax3 = plt.subplots()
    fig, ax = plt.subplots()
    for i, mz in enumerate(mzs):
        print("Mz = ", mz)

        param = (t, U, mu, mg, mz)
        Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T, param, bd,  skewed=skewed) 
        Fs, Ft = pairing_amplitude(gamma, D, T)
        Fs = Fs.reshape(Ny, Nx)
        Ft = Ft.reshape(Ny, Nx)

        ax.plot(np.abs(Fs[Ny//2, :]), label = f"mz = {mz:.2f}Sing")
        Nup,Ndn = N_sigma(gamma, D, T)
        print("Nup tot: ", np.sum(Nup))
        print("Ndn tot: ", np.sum(Ndn))
        print("Ntot: ", np.sum(Ndn) + np.sum(Nup))
        ax3.plot(np.abs(Nup), label = f"mz = {mz:.2f}up")
        ax3.plot(np.abs(Ndn), label = f"mz = {mz:.2f}dn")

        # ax.plot(np.abs(Ft[Ny//2, :]), linestyle = "dashed", label = f"mz = {mz:.2f}Trip")
        if i ==2:
            fig2, ax2 = plt.subplots()
            # divider = make_axes_locatable(ax2)
            # cax = divider.append_axes('right', size='5%', pad=0.05)

            ax2.imshow(np.abs(Delta_i.reshape(Ny, Nx)))
            # ax2[1].imshow(np.abs(Ft))

        # ax.plot(np.abs(Delta_i[Ny//2, :]), label = f"mz = {mz:.2f}")
    fig3.legend()
    ax.set_ylabel("Singlet pairing")
    fig.suptitle(f" mz ={mz:.1f}, U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    fig.legend()
    plt.savefig(f"Delta_mz/N=({Nx},{Ny}),mg={mz:.1f}.pdf", bbox_inches='tight')
    plt.show()

def sweep_Delta_mg(Nx, Ny, mgs, T, U, mu, Deltag,   bd, tol, num_it, skewed ):
    t=1.
    mz = 0.
    Tc0 = 0.3

    # Calc Tc in the mg = 0 case
    param0 = (t, U, mu, 0, 0)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag,   param0, Tc0, bd, num_it, skewed = skewed) 
    fig, ax = plt.subplots()
    fig3, ax3 = plt.subplots()
    for i, mg in enumerate(mgs):
        print("Mg = ", mg)
        param = (t, U, mu, mg, mz)
        Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T, param, bd,  skewed=skewed) 
        # Delta_i = Delta_i.reshape((Ny, Nx))
        Fs, Ft = pairing_amplitude(gamma, D, T)
        Fs = Fs.reshape(Ny, Nx)
        Ft = Ft.reshape(Ny, Nx)
        Nup,Ndn = N_sigma(gamma, D, T)
        print("Nup tot: ", np.sum(Nup))
        print("Ndn tot: ", np.sum(Ndn))
        print("Ntot: ", np.sum(Ndn) + np.sum(Nup))
        # ax3.plot(np.abs(Nup), label = f"mz = {mz:.2f}up")
        # ax3.plot(np.abs(Ndn), label = f"mz = {mz:.2f}dn")
        ax3.plot(np.abs(Nup), label = f"mg = {mg:.2f}up")
        ax3.plot(np.abs(Ndn), label = f"mg = {mg:.2f}dn")

        ax.plot(np.abs(Fs[Ny//2, :]), label = f"mg = {mg:.2f}Sing")
        # ax.plot(np.abs(Ft[Ny//2, :]), linestyle = "dashed", label = f"mg = {mg:.2f}Trip")
    fig3.legend()
    fig.suptitle(f" mz ={0}, U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    fig.legend()
    plt.savefig(f"1d_Delta/N=({Nx},{Ny}),mg={mg:.1f}numit={num_it}_T={T}.pdf", bbox_inches='tight')

    plt.show()

def Tc_fo_mg(Nx, Ny, mgs, U, mu, Deltag,   bd, num_it, skewed ):
    t = 1.
    mz = 0
    Tc0 = 0.3
    Tcs = np.zeros_like(mgs)
    fig, ax = plt.subplots()
    for i, mg in enumerate(mgs):
        print(f"running for mg = {mg}")
        param = (t, U, mu, mg, mz)
        Tcs[i] = calc_Tc_binomial(Nx, Ny, Deltag,   param, Tc0, bd, num_it, skewed=skewed)
        print(f"Tc0 = {Tcs[0]:.4f}, Tc:{Tcs[i]:.4f}.")

    ax.plot(mgs, Tcs)
    fig.suptitle(f" mg ={mg:.1f}, U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    # fig.legend()
    ax.set_xlabel("m")
    ax.set_ylabel("Tc")

    plt.savefig(f"mg_sweep/N=({Nx},{Ny})numit={num_it}sw={skewed}.pdf", bbox_inches='tight')
    # plt.show()

def Tc_one(Nx, Ny, mg,mz,t,Tc0, U, mu, Deltag,   bd, num_it, skewed, alignment):
    param = (t, U, mu, mg, mz)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd, num_it, skewed, alignment)

    return Tc

def Tc_fo_mz(Nx, Ny, mzs, U, mu, Deltag, bd, NDelta, skewed ):
    mg = 0
    t = 1.
    Tc0 = 0.3
    Tcs = np.zeros_like(mzs)
    fig, ax = plt.subplots()
    for i, mz in enumerate(mzs):
        print(f"running for mz = {mz}")



        param = (t, U, mu, mg, mz)
        Tcs[i] = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd, NDelta,  skewed=skewed)
        print(Tcs[i])
    ax.plot(mzs, Tcs)
    fig.suptitle(f" U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    # fig.legend()
    ax.set_xlabel("mz")
    ax.set_ylabel("Tc")

    plt.savefig(f"mz_sweep/N=({Nx},{Ny})N_D={NDelta}.pdf", bbox_inches='tight')


def Tc_fo_Nx(Nxs, Ny, mg, mz, U, mu, Deltag,   bd, tol, num_it, skewed = False):
    Tcs = np.zeros_like(Nxs, dtype = float)
    Tc0s = np.zeros_like(Nxs, dtype = float)
    fig, ax = plt.subplots()
    t = 1.
    Tc0 = 0.3
    param = (t, U, mu, mg, mz)

    for i, Nx in enumerate(Nxs):
        print(f"running for Nx = {Nx}")
        # Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T_plots[i], param, bd,  skewed=skewed) 
        # Delta_i = Delta_i.reshape((Ny, Nx))
        Tc0s[i] = calc_Tc_binomial(Nx - bd, Ny, Deltag,   param, Tc0, 0, num_it,  skewed=skewed)
        Tcs[i] = calc_Tc_binomial(Nx, Ny, Deltag,   param, Tc0, bd, num_it,  skewed=skewed)
        # print(f"Tc :{Tc:.4f}, \nTc2:{Tc2:.4f}.")
        # Tcs[i] = Tc2
        # ax.plot(np.abs(Delta_i[Ny//2, :]), label = f"T/Tc = {T/Tc:.2f}")

    print(Tcs)
    ax.plot(Nxs, Tcs)
    ax.plot(Nxs, Tc0s, label = "bd=0")
    fig.suptitle(f"Nxs=({Nxs[0]},{Nxs[-1]}, Ny={Ny}),num_it={num_it}m={mg}mz={mz}")
    # fig.legend()
    fig.legend()
    ax.set_xlabel("Nx")
    ax.set_ylabel("Tc")
    plt.savefig(f"Nxsweep/Nxs=({Nxs[0]},{Nxs[-1]}Ny={Ny})num_it={num_it}m={mg}mz={mz}.pdf", bbox_inches='tight')

# @njit(cache = True)
def Tc_fo_Ny(Nx, Nys, mg, mz, U, mu, Deltag,   bd, tol, num_it, skewed = False):
    t = 1.
    Tc0 = 0.3
    # Tcs = np.zeros_like(Nys, dtype = float)
    Tcs = np.zeros(len(Nys))
    param = (t, U, mu, mg, mz)

    for i, Ny in enumerate(Nys):
        print(f"running for Ny = {Ny}")
        Tcs[i] = calc_Tc_binomial(Nx, Ny, Deltag,   param, Tc0, bd, num_it,  skewed=skewed)
        print("Tc = ", Tcs[i])

    fig, ax = plt.subplots()

    ax.plot(Nys, Tcs)
    fig.suptitle(f"Ny-sweep")
    # fig.legend()
    ax.set_xlabel("Ny")
    ax.set_ylabel("Tc")
    plt.savefig(f"Ny_sweep/PNx={Nx}_numit={num_it}m={mg}mz={mz}sw={skewed}.pdf", bbox_inches='tight')
    return Tcs

def numitSweep(Nx, Ny, mg, mz, U, mu, Deltag, bd, num_its, skewed, alignment):
    Tc0 = 0.3
    t = 1.
    Tcs = np.zeros_like(num_its, dtype = float)
    # Tc0s = np.zeros_like(Nxs, dtype = float)
    fig, ax = plt.subplots()
    param = (t, U, mu, mg, mz)

    for i, num_it in enumerate(num_its):
        print(f"Running for numit = {num_it}")
        Tcs[i] = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd, num_its[i],  skewed=skewed)

    ax.plot(num_its, Tcs/0.1343)
    fig.suptitle(f"num_its=({num_its[0]},{num_its[-1]},Nx={Nx} Ny={Ny})m={mg}mz={mz}")
    ax.set_xlabel("num_iter")
    ax.set_ylabel("Tc")
    ic(Tcs)
    ic(Tcs/0.1343)
    ic(num_its)
    plt.savefig(f"numitSweepStef/Nx={Nx}Ny={Ny}numits=({num_its[0]},{num_its[-1]}m={mg}mz={mz}sw={skewed}.pdf", bbox_inches='tight')


def main(mg):
    # Method: decide on system parameters, make sure Ny is large enough to avoid-finite size effects (by running Ny_sweep), then decide on number of iterations that give a the correct Tc
    # (by running numit_sweep). Expect it to go down as Ny increases.

    t = 1.
    Nx = 20
    Ny = 1
    Tc0 = 0.2

    # Nys = np.arange(2, 12, 2)
    NDelta = 100
    ic(NDelta)
    # mg = 0.0
    mz = 0.
    bd = np.array([Nx//2, Nx])
    U = 1.7
    mu = -0.5
    Deltag = 5e-3 + 0.j
    # tol = 1e-8 # Remember to adjust this as how as possible
    # T = 0.07
    # Calc Tc for the SC before putting it into contact with the magnet:
    # param = (t, U, mu, mg, mz)
    # param0 = (t, U, mu, 0, 0)
    # Tc_sc0 = calc_Tc_binomial(Nx, Ny, Deltag, param0, Tc0, bd, NDelta, skewed = False)

    # Tc_sc = calc_Tc_binomial(Nx - bd[0], Ny, Deltag, param, Tc0, np.array([0, bd[1] - bd[0]]), NDelta, skewed = False)
    # print("Only SC with hard wall in x-dir: Tc is : ", Tc_sc, "including normal metal(s) we get: ", Tc_sc0)
    # ---------------------------------
    # mzs = np.linspace(0., 0.5, 5)
    # mgs = np.linspace(0., 0.2, 5)
    # Tcs = Tc_fo_Ny(Nx, Nys, mg, mz, U, mu, Deltag, bd, tol, NDelta, skewed = False)
    # plot_Ny(Nys, Tcs, Nx, NDelta, mg, mz)

    # Tcs = Tc_fo_Ny(Nx, Nys, mg, mz, U, mu, Deltag, bd, tol, NDelta, skewed = True)
    # plot_Ny(Nys, Tcs, Nx, NDelta, mg, mz)
    # mzs = np.array((0, 0))
    # sweep_Delta(Nx = Nx, Ny = Ny, mg = mg , mz = mz, U = U, mu = mu, Deltag = Deltag, bd = bd, tol = tol, num_it=NDelta, skewed = True)
    # mzs = np.linspace(0, 1.5, 15)
    # mgs = np.linspace(0, 5, 50)
    # Mz: ----------------------------------------------------------------------- 
    # Tc_fo_mz(Nx, Ny, mzs, U, mu, Deltag, bd, NDelta=NDelta, skewed = False)
    # # mzs= mzs[::10]
    # sweep_Delta_mz(Nx, Ny, mzs, T, U, mu, Deltag, bd, tol, NDelta, skewed=False)
    #-----------------------------------------

    # Mg: 
    Tc = Tc_one(Nx, Ny, mg, mz, t, Tc0,  U, mu, Deltag, bd, num_it=NDelta, skewed = False, alignment = "P")
    # ic(Tc)
    # numitSweep(Nx= Nx, Ny=Ny, mg=mg, mz=mz, U=U, mu= mu, Deltag = Deltag, bd = bd, num_its = np.linspace(20, 500 , 10).astype(int), skewed = False, alignment = None)
    # plt.show()
        # mzs = np.linspace(0, 1.5, 15)

    # mgs = np.linspace(0, 0.2, 5)

    # Tc_fo_mg(Nx, Ny, mgs, U, mu, Deltag,   bd, num_it=NDelta, skewed = False)
    # ic("Straight done")

    # Tc_fo_mg(Nx, Ny, mgs, U, mu, Deltag,   bd, num_it=NDelta, skewed = True)
    # ic("Skewed done")
    # sweep_Delta_mg(Nx, Ny, mgs, T, U, mu, Deltag, bd, tol, NDelta, skewed=True)

    # Tcs = Ny_sweep_Tc(Nx = Nx, Nys = Nys, mg=mg, mz=mz, U=1.7, mu=-0.5, Deltag = 1e-5, DeltaT= np.array([0.5e-5, 2.e-5]), bd = 10, tol = 5e-8, num_it = numit,  skewed = False)
    # print(Tcs)

    # plot_Ny(Nys, Tcs, Nx, numit, mg, mz)

    #-----------------------

    # -- Running numit_sweep ---------------------------------------------------------------------------------
    # param = (t, U, mu, mg, mz)
    # T = 0.01
    # Delta_arr1 = np.ones((Ny, Nx), dtype="complex128")*Deltag
    # Delta_arr1[:, :bd[0]] = 0
    # ic(does_Delta_increase_steff(Nx, Ny, Deltag, T, param, Delta_arr1, bd, NDelta, skewed = False))

    # numitSweep(Nx= 30, Ny=1, mg=1.5, mz=0., U=1.7, mu=-0.5, Deltag = 1e-5, DeltaT= [0.5e-5, 2.e-5], bd = 10, tol = 5e-8, num_its = np.linspace(100, 5000, 10).astype(int),  skewed = False)
    # ------------------------------------------------------------------------------------------------------------
    # -- Running Ny_sweep -----------------------------------------
    """  num_iter = 200
    Ny_sweep_Tc(Nx = 20, Nys = np.arange(1, 30, 5), mg=0., mz=0, U=1.7, mu=-0.5, Deltag = 1e-5, DeltaT= [0.5e-5, 2.e-5], bd = 10, tol = 5e-8, num_it = num_iter,  skewed = False)
    """
    # Nx_sweep_Tc(Nxs = np.arange(20, 100 + 2, 2), Ny=1, mg=0., mz=0, U=1.7, mu=-0.5, Deltag = 1e-5, DeltaT= [0.5e-5, 2.e-5], bd = 10, tol = 5e-8, num_it = num_iter,  skewed = False)
    # plt.show()
    # ------------------------------------------------------------------

    # -- Plot Delta values, remember to set T to something close to Tc -----------------------
    # sweep_Delta(Nx = 25, Ny = 1, mg = 0. , mz = 0., U = 1.7, mu = -0.5, Deltag = 1e-5, DeltaT= [0.5e-5, 2.e-5], bd = 10, tol = 5e-8, num_it = num_iter, skewed=False)

    # sweep_Delta_mz(Nx = 25, Ny = 1, mzs = np.linspace(0, 3, 5), T = 0.13, U = 1.7, mu = -0.5, Deltag = 1e-5, DeltaT= [0.5e-5, 2.e-5], bd = 10, tol = 5e-8, num_it = num_iter, skewed= False )
    # --------------------------------------------------------------------------------------------

    # -- Sweep over values of alter- or ferromagnetism -----------------------------------------------------------------------
    # mz_sweep_Tc(Nx= 20, Ny= 1, mz_start=0, mz_end= 1.5, num_sweep_vals= 20,  U = 1.7, mu= -0.5, Deltag= 1e-5 ,DeltaT= [0.5e-5, 2.e-5], bd = 10, num_it = num_iter, skewed= False)
    # mz_
    # plt.show()
    # sweep_Delta_mg(Nx = 30, Ny = 1, mgs = np.linspace(0, 1.5, 10), T = 0.133, U = 1.7, mu = -0.5, Deltag = 1e-5, DeltaT= [0.5e-5, 2.e-5], bd = 10, tol = 1e-8, num_it = num_iter, skewed= False )
    # mg_sweep_Tc(Nx= 20, Ny= 5, mg_start=0, mg_end= 1.5, num_sweep_vals= 20,  U = 1.7, mu= -0.5, Deltag= 1e-5 ,DeltaT= [0.5e-5, 2.e-5], bd = 10, num_it = num_iter, skewed= False)
    # plt.show()
    #------------------------------------------------------------------------
    return Tc


# if __name__ == "__main__":
#     tic = time.time()

#     mgs = np.linspace(0.5, 5., 1)
#     Tcs = np.zeros(len(mgs))
#     with ProcessPoolExecutor(max_workers=20) as executor:
#         for i, result in enumerate(executor.map(main, mgs)):
#             Tcs[i] = result

#     print(Tcs)
#     np.save(f"TcsReentranceMgs=({mgs[0]}, {mgs[-1]})ND=20Nx=8", Tcs)
#     print("Time: ", time.time() - tic)

#     # # print(Tcs)
#     # plt.plot(mgs, Tcs)
#     # plt.show()
# # main(0.0)
# # main(1.0)

# # Paralellization --------------------------------------
# """def mpwrap(mg):
#     param = (t, U, mu, mg, mz)
#     Tc = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, skewed = skewed)
#     return Tc"""

'''if __name__ == "__main__":
    
    p = Pool(1)
    Tcs = p.map(mpwrap, mgs)

    print(f"took {time.time()- tic} seconds")
    plt.title(f"U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}, mz={mz}")
#     plt.xlabel("m/t")
#     plt.ylabel("Tc")
#     plt.plot(mgs, Tcs)
#     plt.show()
'''

# if __name__ == "__main__":
#     procs = []
#     for mg in mgs:
#         proc = Process(target=mpwrap, args=(mg,))
#         procs.append(proc)
#         proc.start()
#     # complete the processes
#     for proc in procs:
#         proc.join()

# # ----------------------------------------------------------------------

# from multiprocess import Process, Queue

from multiprocess import Pool
if __name__ == "__main__":
    tic = time.time()
    # def f(x): return x*x
    mgs = np.linspace(0, 0.5, 20)
    with  Pool(len(mgs)) as p:
       

        result = p.map(main, mgs)
    ic(result)
    np.save("mgdata", np.array([mgs, result]))
    # ic(result.get())
    tac = time.time()
    ic(tac - tic)