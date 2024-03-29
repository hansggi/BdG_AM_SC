from multiprocessing.pool import Pool
from matplotlib import pyplot as plt
import numpy as np
import time
# import cProfile
from numba import njit
# from scipy import sparse
# from scipy.sparse.linalg import eigsh
# from scipy.sparse.linalg import eigs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from funcs import make_sigmas, nb_block, nb_block2, make_H_numba, fd, Delta_sc, unpack_block_matrix  # , make_H_FT
# from scipy.linalg import eigh
# from multiprocessing import Pool
from tqdm import tqdm

import sys
from icecream import ic
# from concurrent.futures import ProcessPoolExecutor

# @njit()


def does_Delta_increase(Nx, Ny, m_arr, mz_arr, hx_arr, Deltag, T, Ui, mu, imps, Delta_arr1, bd,  NDelta, skewed, periodic):
    # Here, Deltag must be the guess, if Delta < Deltag,
    Delta = Delta_arr1.copy()
    for i in range(NDelta):
        H = make_H_numba(Nx, Ny, m_arr, mz_arr, hx_arr,
                         Delta, mu, imps, skewed, periodic)

        D, gamma = np.linalg.eigh(H)
        # D, gamma = eigh(H)
        D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
        Delta = Delta_sc(gamma, D, Ui, T).reshape(Ny, Nx)

    # ind = np.nonzero(Delta)
    # Delta_bulk = np.average(np.abs(Delta[ind]))
    ind = np.nonzero(Delta)

    Delta_bulk = np.median(np.abs(Delta[ind]))

    if Delta_bulk <= np.abs(Deltag):
        # print("Ran through,", Delta_bulk)
        return False
    else:
        # print("Ran through", Delta_bulk)
        return True

# def does_Delta_increase_steff(Nx, Ny, m_arr, mz_arr, hx_arr, Deltag, T, Ui, mu, imps, Delta_arr1, bd,  NDelta, skewed, periodic):
#     # Here, Deltag must be the guess, if Delta < Deltag,
#     StefNum = 8
#     Deltapp = np.zeros_like(Delta_arr1)
#     Deltap  = np.zeros_like(Delta_arr1)
#     Delta   = Delta_arr1.copy()
#     ind = np.nonzero(Delta)

#     ic(T)
#     Change = np.zeros(NDelta)
#     for i in range(1, NDelta):
#         H = make_H_numba(Nx, Ny, m_arr, mz_arr, hx_arr, Delta, mu, imps, skewed, periodic)
#         D, gamma = np.linalg.eigh(H)
#         D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]


#         Deltapp = Deltap.copy()
#         Deltap = Delta.copy()
#         Delta = Delta_sc(gamma, D, Ui, T).reshape(Ny, Nx)
#         # print(Delta[-1, -1])
#         if i%StefNum==0:
#             # print("error")
#             Delta[ind] = Deltapp[ind] - (Deltap[ind] - Deltapp[ind])**2 / (Delta[ind] - 2 * Deltap[ind] + Deltapp[ind])

#         # Change[i] = np.average(Delta - Deltap)
#         # print(i, Change[i])

#     Delta_bulk = np.median(np.abs(Delta[ind]))

#     if Delta_bulk <= np.abs(Deltag):
#         # print("Ran through,", Delta_bulk)
#         return False
#     else:
#         # print("Ran through", Delta_bulk)
#         return True

# @njit(cache = True)
def calc_Delta_sc(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, imps, Deltag, tol, Ui, mu, T, skewed, alignment, periodic):

    done = False
    Delta_old = Delta_arr.copy()
    # Delta_old = (np.ones((Nx*Ny))*Deltag).reshape(Ny, Nx)*(1 + 0j)
    # Delta_old[:, :bd] = 0
    it = 0
    # if bd[1] < Nx:
    #     Delta_old_bulk = np.abs(Delta_old[Ny//2, (bd[0] + bd[1])//2])

    # else:
    #     Delta_old_bulk = np.abs(Delta_old[Ny//2, (bd[0] + Nx)//2])

    while not done:
        H = make_H_numba(Nx, Ny, m_arr, mz_arr, hx_arr,
                         Delta_old, mu, imps, skewed, periodic)

        D, gamma = np.linalg.eigh(H)
        # D, gamma = eigh(H)
        D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
        # ----------------------------------------------------------------------------------

        Delta_new = Delta_sc(gamma, D, Ui, T).reshape(Ny, Nx)
        # Delta_new = Delta_new_i.reshape(Ny, Nx)
        # Delta_bulk = np.average(np.abs(Delta_new))
        # if bd[1] < Nx:
        #     Delta_bulk = np.abs(Delta_new)[Ny//2, (bd[0] + bd[1])//2]
        # else:
        #     Delta_bulk = np.abs(Delta_new)[Ny//2, (bd[0] + Nx)//2]
        it += 1
        # Bulk method
        # if np.abs(Delta_bulk - Delta_old_bulk)  <= tol :
        #     done = True

        # Using max difference instead. Will not give the same plot as in the article, since T is a function of bulk Tc here
        if np.median(np.abs(np.abs(Delta_new) - np.abs(Delta_old))) <= tol:
            done = True
            ic(it)
        # Delta_old = Delta_new
        Delta_old = Delta_new

        # Delta_old_bulk = Delta_bulk

    # print("Used ", it, " iterations to calculate Delta self-consist.")
    return Delta_new, gamma, D

# @njit()


def calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, Deltag, Ui, imps, mu, Tc0, bd, num_it, skewed, alignment, periodic):
    N = 15  # Look at, maybe not needed this accuracy
    if alignment == None:
        assert bd[1] == Nx
    # The first calculation is the same for all temperatures --------------
    # x = np.arange(0, Nx)
    # Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)#*(x - bd)**2*0.05 / 40**2

    H = make_H_numba(Nx, Ny, m_arr, mz_arr, hx_arr,
                     Delta_arr, mu, imps, skewed, periodic)
    # plt.imshow(np.real(H - np.conjugate(H.T))[::4, ::4])
    # plt.colorbar()
    # plt.show()
    assert np.allclose(H, np.conjugate(H.T), rtol=1e-08,
                       atol=1e-08, equal_nan=False)

    D, gamma = np.linalg.eigh(H)
    # D, gamma = eigh(H)

    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]

    # ---------------------------------------------------------------

    Ts_lower = 0
    Ts_upper = Tc0

    for i in range(N):
        T = (Ts_upper + Ts_lower) / 2
        assert Ts_upper > Ts_lower
        Delta_arr1 = Delta_sc(gamma, D, Ui, T).reshape(Ny, Nx)

        # Meaning that there is SC at this temp, need to go higher in T to find Tc
        if does_Delta_increase(Nx, Ny, m_arr, mz_arr, hx_arr, Deltag, T, Ui, mu, imps,  Delta_arr1, bd, num_it, skewed, periodic):
            Ts_lower = T
        else:
            Ts_upper = T

    return (Ts_upper + Ts_lower) / 2


@njit()
def gaussian(E, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- (E - mu)**2 / (2 * sigma**2))


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

            s[i, :] += np.abs(u_up[i, j])**2 * gaussian(E, D[j], 0.1) + \
                np.abs(v_up[i, j])**2 * gaussian(E, -D[j], 0.1)
            s[i, :] += np.abs(u_dn[i, j])**2 * gaussian(E, D[j], 0.1) + \
                np.abs(v_dn[i, j])**2 * gaussian(E, -D[j], 0.1)

    return s


def N_sigma(gamma, D, T, Nx, Ny):
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]

    # Fuu = np.sum(u_up *np.conjugate(v_up) , axis = 1)
    # print(Fuu)
    f = (1 - np.tanh(D / (2 * T))) / 2
    N_up = np.sum(u_up * np.conjugate(u_up) * f + v_up *
                  np.conjugate(v_up) * (1 - f), axis=1).reshape(Ny, Nx)
    N_dn = np.sum(u_dn * np.conjugate(u_dn) * f + v_dn *
                  np.conjugate(v_dn) * (1 - f), axis=1).reshape(Ny, Nx)

    return N_up, N_dn


def pairing_amplitude(gamma, D, T):
    # NB: gamma and energies D must already be just positive eigenvalues
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]

    # Fuu = np.sum(u_up *np.conjugate(v_up) , axis = 1)
    # print(Fuu)
    f = (1 - np.tanh(D / (2 * T))) / 2
    # -------------------------------------------------------------------
    Fud = np.sum(u_dn * np.conjugate(v_up) * f + u_up *
                 np.conjugate(v_dn) * (1 - f), axis=1)
    Fdu = np.sum(u_up * np.conjugate(v_dn) * f + u_dn *
                 np.conjugate(v_up) * (1 - f), axis=1)
    # P

    # Fpx =  np.sum(u_dn *np.conjugate(np.roll(v_up, 1, axis = 0)) * f + u_up * np.conjugate(v_dn)* (1 - f) , axis = 1)
    # ----------
    # print(np.allclose(Fud, -Fdu))
    # Here, we used equilibrium explicitly to say (1-f(E)) = f(-E)
    Fs_i = 0.5*(Fud - Fdu)
    Ft_i = 0.5*(Fud + Fdu)
    return Fs_i, Ft_i


def make_system_normal(bd, U, mz, hx, mg, impdata, Deltag, Nx, Ny, alignment):
    # ic(bd)
    w_AM, w_SC, NfracAM, NfracSC = impdata
    # ic(w_AM, w_SC, NfracAM, NfracSC)
    imps = np.zeros((Ny, Nx))
    imps[:, :bd[0]] = make_impurities(imps[:, :bd[0]], w_AM, NfracAM)
    imps[:, bd[1]:] = make_impurities(imps[:, bd[1]:], w_AM, NfracAM)

    imps[:, bd[0]:bd[1]] = make_impurities(imps[:, bd[0]:bd[1]], w_SC, NfracSC)
    # plt.imshow(imps)
    # plt.colorbar()
    # plt.show()
    Delta_arr = (np.ones((Nx*Ny))*Deltag).reshape(Ny,
                                                  Nx)  # To make it complex?
    Delta_arr[:, :bd[0]] = 0
    if bd[1] < Nx:
        Delta_arr[:, bd[1]:] = 0

    U = U * (1. + 0j)
    Ui = np.ones(Nx*Ny).reshape(Ny, Nx)*U

    Ui[:, :bd[0]] = 0
    if bd[1] < Nx:
        Ui[:, bd[1]:] = 0
    Ui = Ui.reshape((Nx*Ny))

    mg_arr = (np.ones((Nx*Ny)) * mg).reshape(Ny, Nx)
    mg_arr[:, bd[0]:bd[1]] = 0

    hx_arr = (np.ones((Nx*Ny)) * hx).reshape(Ny, Nx)
    hx_arr[:, bd[0]:bd[1]] = 0

    mz_arr = (np.ones((Ny*Nx))*mz).reshape(Ny, Nx)
    mz_arr[:, bd[0]:bd[1]] = 0

    if alignment == "AP":
        mg_arr[:, bd[1]:] *= -1
        hx_arr[:, bd[1]:] *= -1
        mz_arr[:, bd[1]:] *= -1

    # plt.imshow(mg_arr)
    # plt.colorbar()
    # plt.show()
    # plt.imshow(mz_arr)
    # # plt.colorbar()
    # plt.show()
    # plt.imshow(np.abs(Delta_arr))
    # plt.colorbar()
    # plt.show()
    # plt.imshow(np.abs(Ui).reshape(Ny, Nx))
    # plt.colorbar()
    # plt.show()
    return Ui, mz_arr, hx_arr, mg_arr, Delta_arr, imps


def make_system_one_material(U, mz, hx, m, Deltag, Nx, Ny, w, Nfrac):
    Delta_arr = (np.ones((Nx*Ny))*Deltag).reshape(Ny,
                                                  Nx)  # To make it complex?

    U = U * (1. + 0j)
    Ui = np.ones(Nx*Ny).reshape(Ny, Nx)*U
    Ui = Ui.reshape((Nx*Ny))

    m_arr = (np.ones((Nx*Ny)) * m).reshape(Ny, Nx)
    # plt.imshow(m_arr)
    # plt.colorbar()
    # plt.show()
    mz_arr = (np.ones((Ny*Nx))*mz).reshape(Ny, Nx)
    hx_arr = (np.ones((Nx*Ny)) * hx).reshape(Ny, Nx)

    # ic("Running for one material, not a heterostructure. Vals:")

    imps = np.zeros((Ny, Nx))
    imps = make_impurities(imps, w, Nfrac)
    return Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps


def plot_observables_constantT(Delta, gamma, D, T, Nx, Ny, NDelta, mz, bd, U, mu, Deltag, tol, alignment, skewed):
    x = np.arange(1, Nx + 1, 1)
    ic(T, Nx, Ny, NDelta, mz, bd, U, mu, Deltag, tol, alignment, skewed)
    # -- Plot Delta as f.o. x --
    fig, ax = plt.subplots()
    ax.plot(x, np.abs(Delta[Ny//2, :]))
    # ax.axvline(x = bd[0])
    # ax.axvline(x = bd[1])
    ax.set_ylabel(r"$\Delta$ at y = " + str(Ny//2))
    ax.set_xlabel(f"x")
    plt.show()

    # -- Plot the particle number as f.o. x --
    N_up, N_dn = np.abs(N_sigma(gamma, D, T, Nx, Ny))

    fig, ax = plt.subplots()
    ax.plot(x, N_up[Ny//2, :], label=r"N_\uparrow")
    ax.plot(x, N_dn[Ny//2, :], label=r"N_\downarrow")

    ax.set_ylabel(r"Particle density at y = " + str(Ny//2))
    ax.set_xlabel(f"x")

    fig.legend()

    print("Nup tot: ", np.sum(N_up))
    print("Ndn tot: ", np.sum(N_dn))
    print("Ntot: ", np.sum(N_dn) + np.sum(N_up))
    plt.show()


def make_impurities(A, w, Nfrac):
    # Fill A with fraction impfrac of strength w
    shape = np.shape(A)
    size = shape[0] * shape[1]
    impnum = int(size*Nfrac)

    if Nfrac > 1:
        raise ValueError("Number of ones exceeds the size of the matrix.")

    # Flatten the matrix to a 1D array
    flattened = np.zeros(size)

    # Generate random indices without replacement
    indices = np.random.choice(size, impnum, replace=False)

    # Set the chosen indices to 1
    flattened[indices] = w

    # Reshape the flattened array back to the original shape
    A[:, :] = flattened.reshape(shape)
    # ic(impnum)
    # ic(size)
    # ic(Nfrac)
    # plt.imshow(A)
    # plt.show()

    return A


def task_onematerial(Nx, Ny, NDelta, mz, hx, mg):
    # Constants
    U = 1.7
    mu = -0.5
    Deltag = 1e-4 + 0.j
    Tc0 = 0.2

    # Problem spesific constants
    # x-direction, y-direction. Should only have x.dir for the one material thing.
    periodic = np.array([True, True])
    bd = np.array([Nx, Nx])
    w = 0
    Nfrac = 0
    skewed = False
    alignment = None

    # Make Hamiltonian parameters
    Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps = make_system_one_material(
        U, mz, hx, mg, Deltag, Nx, Ny, w, Nfrac)

    Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr,
                          Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)

    return Tc

def task_onematerial_Delta(Nx, Ny, tol, mz, hx, mg, T):
    # Constants
    U = 1.7
    mu = -0.5
    Deltag = 1e-4 + 0.j
    Tc0 = 0.2

    # Problem spesific constants
    # x-direction, y-direction. Should only have x.dir for the one material thing.
    periodic = np.array([True, True])
    bd = np.array([Nx, Nx])
    w = 0
    Nfrac = 0
    skewed = False
    alignment = None

    # Make Hamiltonian parameters
    Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps = make_system_one_material(
        U, mz, hx, mg, Deltag, Nx, Ny, w, Nfrac)

    # Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr,
    #                       Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)
    Delta = calc_Delta_sc(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, imps, Deltag, tol, Ui, mu, T, skewed, alignment, periodic)[0]
    ic(Delta.shape)
    return np.average(np.real(Delta))


def task_straightskewed(Nx, Ny, NDelta, mz, hx, mg, skewed, wAM, NfracAM):
    # Constants
    U = 1.7
    mu = -0.5
    Deltag = 1e-4 + 0.j
    Tc0 = 0.2

    # Problem spesific constants
    # x-direction, y-direction. Should only have x.dir for the one material thing.
    periodic = np.array([False, True])
    bd = np.array([10, Nx])  # SC to left, AM to right
    NfracSC = 0.0  # Fraction of lattice sites that will get impurities in SC
    w_SC = 0.0
    impdata = [wAM, w_SC, NfracAM, NfracSC]
    alignment = None

    # Make Hamiltonian parameters
    Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps = make_system_normal(
        bd, U, mz, hx, mg, impdata, Deltag, Nx, Ny, alignment)
    Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr,
                          Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)
    print(f"bd={bd}mg={mg}, mz={mz}, Tc={Tc}")
    return Tc


def task_imp_oneval(Nx, Ny, NDelta, mz, hx, mg, wAM, NfracAM):
    ic(mg, mz, wAM, NfracAM)
    # Constants
    U = 1.7
    mu = -0.5
    Deltag = 1e-4 + 0.j
    Tc0 = 0.2

    # Problem spesific constants
    # x-direction, y-direction. Should only have x.dir for the one material thing.
    periodic = np.array([False, True])
    bd = np.array([10, Nx])  # SC to the right, AM to the left
    NfracSC = 0.0  # Fraction of lattice sites that will get impurities in SC
    w_SC = 0.0
    impdata = [wAM, w_SC, NfracAM, NfracSC]
    alignment = None
    skewed = False
    # Make Hamiltonian parameters

    Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps = make_system_normal(
        bd, U, mz, hx, mg, impdata, Deltag, Nx, Ny, alignment)
    Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr,
                          Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)
    return Tc

# def task_imp_oneval_pap(Nx, Ny, NDelta, mz, hx, mg, wAM, NfracAM):
#     ic(mg, mz, wAM, NfracAM)
#     # Constants
#     U = 1.7
#     mu = -0.5
#     Deltag = 1e-4 + 0.j
#     Tc0 = 0.2

#     # Problem spesific constants
#     # x-direction, y-direction. Should only have x.dir for the one material thing.
#     periodic = np.array([False, True])
#     bd = np.array([10, Nx - 10])  # PAP structure
#     NfracSC = 0.0  # Fraction of lattice sites that will get impurities in SC
#     w_SC = 0.0
#     impdata = [wAM, w_SC, NfracAM, NfracSC]
#     alignment = "P"
#     skewed = False
#     # Make Hamiltonian parameters

#     Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps = make_system_normal(
#         bd, U, mz, hx, mg, impdata, Deltag, Nx, Ny, alignment)
#     Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr,
#                           Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)
#     return Tc


def task_PAP(Nx, Ny, bd, NDelta, mz, hx, mg, alignment, wAM, NfracAM):
    # Constants
    U = 1.7
    mu = -0.5
    Deltag = 1e-4 + 0.j
    Tc0 = 0.2
    ic(mg, mz)

    # Problem spesific constants
    # x-direction, y-direction. Should only have x.dir for the one material thing.
    periodic = np.array([False, True])
    NfracSC = 0.0  # Fraction of lattice sites that will get impurities in SC
    w_SC = 0.0
    impdata = [wAM, w_SC, NfracAM, NfracSC]
    skewed = False

    # Make Hamiltonian parameters
    Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps = make_system_normal(
        bd, U, mz, hx, mg, impdata, Deltag, Nx, Ny, alignment)
    Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr,
                          Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)
    ic(mg, Tc)
    return Tc


def run_onemat(magnettype):
    # Prepare for multiprocessing in the Onemat case

    # Run spesific parameters:
    Nx = 20
    Ny = 20
    NDelta = 5
    mz = 0
    hx = 0
    items = []
    numsteps = 30
    tic = time.time()
    if magnettype == "AM":
        mgs = np.linspace(0, 0.3, numsteps)
        mz = 0
        for mg in mgs:
            items.append((Nx, Ny, NDelta, mz, hx, mg))
    elif magnettype == "FM":
        mzs = np.linspace(0, 0.3, numsteps)
        mg = 0
        for mz in mzs:
            items.append((Nx, Ny, NDelta, mz, hx, mg))
    else:
        raise NameError

    print(items)

    Tcs = np.zeros(numsteps)
    # print(np.array(items).shape[1])
    for i in tqdm(range(numsteps)):
        Tcs[i] = task_onematerial(*items[i])

    tac = time.time()
    print(tac - tic)
    if magnettype =="AM":
        np.save(f"Newdata4/onemat/{magnettype}{(Nx, Ny, NDelta, mz, hx, mg)}", np.array([items, Tcs, mgs], dtype = object))
    else:
        np.save(f"Newdata4/onemat/{magnettype}{(Nx, Ny, NDelta, mz, hx, mg)}", np.array([items, Tcs, mzs], dtype = object))

    # print(Tcs)
    # plt.plot(mgs, Tcs)
    # plt.show()
    return Tcs

def run_onemat_Delta(magnettype, tol, T):
    # Prepare for multiprocessing in the Onemat case

    # Run spesific parameters:
    Nx = 20
    Ny = 20
    # NDelta = 5
    mz = 0
    hx = 0
    items = []
    numsteps = 100
    tic = time.time()
    if magnettype == "AM":
        mgs = np.linspace(0, 0.3, numsteps)
        mz = 0
        for mg in mgs:
            items.append((Nx, Ny, tol, mz, hx, mg, T))
    elif magnettype == "FM":
        mzs = np.linspace(0, 0.3, numsteps)
        mg = 0
        for mz in mzs:
            items.append((Nx, Ny, tol, mz, hx, mg, T))
    else:
        raise NameError

    print(items)

    Deltas = np.zeros(numsteps)
    # print(np.array(items).shape[1])
    for i in tqdm(range(numsteps)):
        ic(task_onematerial_Delta(*items[i]).shape)
        Deltas[i] = np.average(task_onematerial_Delta(*items[i]))

    tac = time.time()
    print(tac - tic)
    if magnettype =="AM":
        np.save(f"Newdata4/onemat_Delta/{magnettype}{(Nx, Ny, tol, mz, hx, mg, T)}", np.array([items, Deltas, mgs], dtype = object))
    else:
        np.save(f"Newdata4/onemat_Delta/{magnettype}{(Nx, Ny, tol, mz, hx, mg, T)}", np.array([items, Deltas, mzs], dtype = object))

    # print(Tcs)
    # plt.plot(mgs, Tcs)
    # plt.show()
    return Deltas

def run_straightskewed(skewed, magnettype):
    print(
        f"Running run_straightskewed for a {magnettype} in the {skewed} geometry.")

    # Run spesific parameters:
    Nx = 15
    Ny = 20
    NDelta = 30
    hx = 0
    wAM = 0
    NfracAM = 0
    items = []
    tic = time.time()
    # Prepare for multiprocessing in the straight/skewed case
    numsteps = 100
    if magnettype == "AM":
        mgs = np.linspace(0, 1.0, numsteps)
        mz = 0
        for i, mg in enumerate(mgs):
            items.append((Nx, Ny, NDelta, mz, hx, mg, skewed, wAM, NfracAM))
    elif magnettype == "FM":
        mzs = np.linspace(0, 1.0, numsteps)
        mg = 0
        for mz in mzs:
            items.append((Nx, Ny, NDelta, mz, hx, mg, skewed, wAM, NfracAM))
    else:
        raise NameError

    print(items)
    with Pool() as pool:
        Tcs = pool.starmap(task_straightskewed, items)
    tac = time.time()
    print(Tcs)
    print(tac - tic)

    if magnettype == "AM":
        np.save(f"Newdata4/straightskewed/Shorter{magnettype}{skewed}{(Nx, Ny, NDelta, mz, hx, mg, skewed, wAM, NfracAM)}", np.array(
            [items, mgs,  Tcs], dtype=object))
    elif magnettype == "FM":
        np.save(f"Newdata4/straightskewed/Shorter{magnettype}{skewed}{(Nx, Ny, NDelta, mz, hx, mg, skewed, wAM, NfracAM)}", np.array(
            [items, mzs,  Tcs], dtype=object))

    # plt.plot(mgs, Tcs)
    # plt.show()
    return Tcs


def run_PAP(alignment, magnettype):
    # Prepare for multiprocessing in the straight/skewed case
    numsteps = 100
    # Run spesific parameters:
    Nx = 32
    Ny = 20
    bd = [10, Nx - 10]
    NDelta = 30
    hx = 0
    wAM = 0
    NfracAM = 0
    items = []
    tic = time.time()
    print(f"Running run_PAP for a {magnettype} in the {alignment} alignment with boundaries {bd}. Parameters: (without magnets): {(Nx, Ny, bd, NDelta,  hx,  alignment, wAM, NfracAM)}")

    if magnettype == "AM":
        mgs = np.linspace(0.0, 1.0, numsteps)
        # mgs = np.zeros(4)
        mz = 0
        for i, mg in enumerate(mgs):
            items.append((Nx, Ny, bd, NDelta, mz, hx,
                         mg, alignment, wAM, NfracAM))

    elif magnettype == "FM":
        mzs = np.linspace(0, 1.0, numsteps)
        mg = 0
        for mz in mzs:
            items.append((Nx, Ny, bd, NDelta, mz, hx,
                         mg, alignment, wAM, NfracAM))
    else:
        raise NameError

    print(items)

    Tcs = np.zeros(len(items))
    for i in tqdm(range(numsteps)):
        Tcs[i] = task_PAP(*items[i])


    tac = time.time()
    print(tac - tic)
    print(f"Finished running run_PAP for a {magnettype} in the {alignment} alignment with boundaries {bd}. Parameters: (including last value of magnetic strength): {(Nx, Ny, bd, NDelta, mz, hx, mg, alignment, wAM, NfracAM)}")
    print(Tcs)
    if magnettype =="AM":
        np.save(f"Newdata4/PAP/{magnettype}{alignment}{(Nx, Ny, bd, NDelta, mz, hx, mg, alignment, wAM, NfracAM)}",
            np.array([items, Tcs, mgs], dtype=object))
    else:
        np.save(f"Newdata4/PAP/{magnettype}{alignment}{(Nx, Ny, bd, NDelta, mz, hx, mg, alignment, wAM, NfracAM)}",
            np.array([items, Tcs, mzs], dtype=object))
    # plt.plot(mgs, Tcs)
    # plt.show()
    return Tcs


def run_imp_oneval(mg, mz, wAM, NfracAM, magnettype):
    # Run spesific parameters:
    Nx = 16
    Ny = 20
    NDelta = 30
    print("Set to 30")
    hx = 0
    items = []
    tic = time.time()
    # Prepare for multiprocessing   in the straight/skewed case
    numvals = 1  # Number of impurity configurations
    if magnettype == "AM":
        assert mz == 0
        for i in range(numvals):
            items.append((Nx, Ny, NDelta, mz, hx, mg, wAM, NfracAM))
    elif magnettype == "FM":
        assert mg == 0
        for i in range(numvals):
            items.append((Nx, Ny, NDelta, mz, hx, mg,  wAM, NfracAM))
    else:
        raise NameError

    print(items)

    Tcs = np.zeros(numvals)
    for i in tqdm(range(numvals)):
        Tcs[i] = task_imp_oneval(*items[i])

    # with Pool() as pool:
    #     Tcs = pool.starmap(task_imp_oneval, items)
    tac = time.time()
    print(Tcs)

    item0 = (Nx, Ny, NDelta, mz, hx, mg, 0, 0)
    item1 = (Nx, Ny, NDelta, 0, 0, 0, 0, 0)
    Tc0 = task_imp_oneval(*item0)
    Tc1 = task_imp_oneval(*item1)

    ic(tac - tic)
    np.save(f"Newdata4/imps_new/{magnettype}m={mg:.2f}mz={mz}{(Nx, Ny), NDelta, wAM, NfracAM}",
            np.array([items, Tcs, Tc0, Tc1], dtype=object))

    # plt.plot(np.arange(numvals), Tcs)
    # plt.axhline(y=np.average(Tcs), label = "Average over imps", color = "orange")
    # plt.axhline(y=Tc0, label = "Without imps", color = "green")
    # plt.legend()
    # plt.show()
    return Tcs



# --- Just AM, without SC part -----------------------------------------------------
def run_imp_just_AM_mNfracplot( mz, wAM, magnettype, T):
    Nx = 20
    Ny = 20
    mu = -0.5 

    items = []
    tic = time.time()
    # Prepare for multiprocessing   in the straight/skewed case
    numvals_mg = 20
    numvals_Nfrac = 20
    NfracMax = 0.6
    Nfracs = np.linspace(0, NfracMax, numvals_Nfrac)
    mgs = np.linspace(0, 1., numvals_mg)
    numvals = 100  # Number of impurity configurations

    cx_up_imp = np.zeros((numvals_Nfrac, numvals_mg, numvals))
    cy_up_imp = np.zeros((numvals_Nfrac, numvals_mg, numvals))
    cx_dn_imp = np.zeros((numvals_Nfrac, numvals_mg, numvals))
    cy_dn_imp = np.zeros((numvals_Nfrac, numvals_mg, numvals))

    # for k in range(numvals_Nfrac):
    #     NfracAM = Nfracs[k]
    #     ic(wAM, NfracAM)
    #     for i in range(numvals_mg):
    #         for j in range(numvals):
    #             items.append((Nx, Ny, mgs[i], wAM, Nfracs[k], T, mu))

    # ic(items[0])
    cx_up_av    = np.zeros((numvals_Nfrac, numvals_mg))
    cy_up_av    = np.zeros((numvals_Nfrac, numvals_mg))
    # cx_up_noimp = np.zeros((numvals_Nfrac, numvals_mg))
    # cy_up_noimp = np.zeros((numvals_Nfrac, numvals_mg))

    cx_dn_av    = np.zeros((numvals_Nfrac, numvals_mg))
    cy_dn_av    = np.zeros((numvals_Nfrac, numvals_mg))
    # cx_dn_noimp = np.zeros((numvals_Nfrac, numvals_mg))
    # cy_dn_noimp = np.zeros((numvals_Nfrac, numvals_mg))

    for k in tqdm(range(numvals_Nfrac)):
        for i in tqdm(range(numvals_mg)):
            # cx_up_noimp[k, i], cy_up_noimp[k, i], cx_dn_noimp[k, i], cy_dn_noimp[k, i] = task_just_AM(*(Nx, Ny, mgs[i], 0, 0, T, mu))
            for j in range(numvals):
                cx_up_imp[k, i, j], cy_up_imp[k, i, j], cx_dn_imp[k, i, j], cy_dn_imp[k, i, j] = task_just_AM(*(Nx, Ny, mgs[i], wAM, Nfracs[k], T, mu))

            # ic(np.average(cx_up_imp, axis = 1).shape)
            cx_up_av[k, i] = np.average(cx_up_imp[k, i, :])
            cy_up_av[k, i] = np.average(cy_up_imp[k, i, :])
            cx_dn_av[k, i] = np.average(cx_dn_imp[k, i, :])
            cy_dn_av[k, i] = np.average(cy_dn_imp[k, i, :])

    
    np.save(f"Newdata4/AM_imps_mN/w={wAM}{(Nx, Ny), numvals_mg, numvals_Nfrac, numvals}",
                np.array([items,cx_up_av, cy_up_av, cx_dn_av, cy_dn_av, 1, NfracMax ], dtype=object))



def run_imp_just_AM_wNfracplot(mg, mz, magnettype, T):
    Nx = 20
    Ny = 20
    mu = -0.5
    items = []  
    tic = time.time()
    # Prepare for multiprocessing   in the straight/skewed case
    numvals_w = 20
    numvals_Nfrac = 20
    NfracMax = 0.6
    Nfracs = np.linspace(0, NfracMax, numvals_Nfrac)
    wMax = 5
    ws = np.linspace(0, wMax, numvals_w)
    numvals = 100  # Number of impurity configurations

    cx_up_imp = np.zeros((numvals_Nfrac, numvals_w, numvals))
    cy_up_imp = np.zeros((numvals_Nfrac, numvals_w, numvals))
    cx_dn_imp = np.zeros((numvals_Nfrac, numvals_w, numvals))
    cy_dn_imp = np.zeros((numvals_Nfrac, numvals_w, numvals))

    # for k in range(numvals_Nfrac):
    #     # NfracAM = Nfracs[k]
    #     # ic(wAM, NfracAM)
    #     for i in range(numvals_w):
    #         wAM = ws[i]
    #         for j in range(numvals):
    #             items.append((Nx, Ny, mg, ws[i], Nfracs[k], T, mu))


    # ic(items[0])
    cx_up_av    = np.zeros((numvals_Nfrac, numvals_w))
    cy_up_av    = np.zeros((numvals_Nfrac, numvals_w))
    # cx_up_noimp = np.zeros((numvals_Nfrac, numvals_w))
    # cy_up_noimp = np.zeros((numvals_Nfrac, numvals_w))

    cx_dn_av    = np.zeros((numvals_Nfrac, numvals_w))
    cy_dn_av    = np.zeros((numvals_Nfrac, numvals_w))
    # cx_dn_noimp = np.zeros((numvals_Nfrac, numvals_w))
    # cy_dn_noimp = np.zeros((numvals_Nfrac, numvals_w))

    for k in tqdm(range(numvals_Nfrac)):
        for i in range(numvals_w):
            # cx_up_noimp[k, i], cy_up_noimp[k, i], cx_dn_noimp[k, i], cy_dn_noimp[k, i] = task_just_AM(*(Nx, Ny, mg, 0, 0, T, mu))
            for j in range(numvals):
                cx_up_imp[k, i, j], cy_up_imp[k, i, j], cx_dn_imp[k, i, j], cy_dn_imp[k, i, j] = task_just_AM(*(Nx, Ny, mg, ws[i], Nfracs[k], T, mu))

            # ic(np.average(cx_up_imp, axis = 1).shape)
            cx_up_av[k, i] = np.average(cx_up_imp[k, i, :])
            cy_up_av[k, i] = np.average(cy_up_imp[k, i, :])
            cx_dn_av[k, i] = np.average(cx_dn_imp[k, i, :])
            cy_dn_av[k, i] = np.average(cy_dn_imp[k, i, :])

    
    # ic(cx_up_noimp[10], cy_up_noimp[10])

    np.save(f"Newdata4/AM_imps_wN/m={mg}{(Nx, Ny), numvals_w,numvals_Nfrac, numvals}",
            np.array([items,cx_up_av, cy_up_av, cx_dn_av, cy_dn_av, wMax, NfracMax ], dtype=object))


def run_imp_just_AM_mmufracplot(mz, magnettype, T):
    Nx = 20
    Ny = 20
    wAM = 0
    Nfrac = 0

    items = []
    tic = time.time()
    # Prepare for multiprocessing   in the straight/skewed case
    numvals_mg = 100
    numvals_mu = 100
    muMin = -3
    muMax = -0.5
    mus = np.linspace(muMin, muMax, numvals_mu)
    mgs = np.linspace(0, 1., numvals_mg)
    numvals = 1  # Number of impurity configurations

    cx_up_imp = np.zeros((numvals_mu, numvals_mg, numvals))
    cy_up_imp = np.zeros((numvals_mu, numvals_mg, numvals))
    cx_dn_imp = np.zeros((numvals_mu, numvals_mg, numvals))
    cy_dn_imp = np.zeros((numvals_mu, numvals_mg, numvals))

    for k in range(numvals_mu):
        # NfracAM = mus[k]
        for i in range(numvals_mg):
            for j in range(numvals):
                items.append((Nx, Ny, mgs[i], wAM, mus[k], T))


    cx_up_av    = np.zeros((numvals_mu, numvals_mg))
    cy_up_av    = np.zeros((numvals_mu, numvals_mg))
    # cx_up_noimp = np.zeros((numvals_mu, numvals_mg))
    # cy_up_noimp = np.zeros((numvals_mu, numvals_mg))

    cx_dn_av    = np.zeros((numvals_mu, numvals_mg))
    cy_dn_av    = np.zeros((numvals_mu, numvals_mg))
    # cx_dn_noimp = np.zeros((numvals_mu, numvals_mg))
    # cy_dn_noimp = np.zeros((numvals_mu, numvals_mg))

    for k in tqdm(range(numvals_mu)):
        for i in tqdm(range(numvals_mg)):
            # cx_up_noimp[k, i], cy_up_noimp[k, i], cx_dn_noimp[k, i], cy_dn_noimp[k, i] = task_just_AM(*(Nx, Ny, mgs[i], 0, 0, T, mus[k]))
            for j in range(numvals):
                cx_up_imp[k, i, j], cy_up_imp[k, i, j], cx_dn_imp[k, i, j], cy_dn_imp[k, i, j] = task_just_AM(*(Nx, Ny, mgs[i], wAM, Nfrac, T, mus[k]))

            # ic(np.average(cx_up_imp, axis = 1).shape)
            cx_up_av[k, i] = np.average(cx_up_imp[k, i, :])
            cy_up_av[k, i] = np.average(cy_up_imp[k, i, :])
            cx_dn_av[k, i] = np.average(cx_dn_imp[k, i, :])
            cy_dn_av[k, i] = np.average(cy_dn_imp[k, i, :])


    np.save(f"Newdata4/AM_imps_mmu/{(Nx, Ny), numvals_mg, numvals_mu, numvals}",
            np.array([items,cx_up_av, cy_up_av, cx_dn_av, cy_dn_av, 1, muMin, muMax ], dtype=object))



def task_just_AM(Nx, Ny, mg, wAM, NfracAM, T, mu):
    # Constants
    U = 1.7
    # mu = -0.5
    # ic(mg, mz)

    # Problem spesific constants
    # x-direction, y-direction. Should only have x.dir for the one material thing.
    periodic = np.array([False, False])
    # NfracSC = 0.0  # Fraction of lattice sites that will get impurities in SC
    # w_SC = 0.0
    # impdata = [wAM, w_SC, NfracAM, NfracSC]
    skewed = False

    # Make Hamiltonian parameters
    mz, hx = 0, 0
    Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps = make_system_one_material(U, mz, hx, mg, 0, Nx, Ny, wAM, NfracAM)

    H = make_H_numba(Nx, Ny, m_arr, mz_arr, hx_arr,
                         Delta_arr, mu, imps, skewed, periodic)
    

    D, gamma = np.linalg.eigh(H)
    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
    cx_up, cy_up, cx_dn, cy_dn = calc_corr(D, gamma, T, Nx, Ny)
    return np.average(np.abs(cx_up)), np.average(np.abs(cy_up)), np.average(np.abs(cx_dn)), np.average(np.abs(cy_dn))

def calc_corr(D, gamma, T, Nx, Ny):
    # NB: gamma and energies D must already be just positive eigenvalues
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]

    f = (1 - np.tanh(D / (2 * T))) / 2


    u_up_px =  np.roll(u_up, -1, axis = 0)
    v_up_px =  np.roll(v_up, -1, axis = 0)

    u_up_py =  np.roll(u_up, -Nx, axis = 0)
    v_up_py =  np.roll(v_up, -Nx, axis = 0)
    

    u_dn_px =  np.roll(u_dn, -1, axis = 0)
    v_dn_px =  np.roll(v_dn, -1, axis = 0)

    u_dn_py =  np.roll(u_dn, -Nx, axis = 0)
    v_dn_py =  np.roll(v_dn, -Nx, axis = 0)

    cx_up = np.sum(np.conjugate(u_up) * u_up_px * f +
                        v_up * np.conjugate(v_up_px) * (1 - f), axis=1).reshape((Ny, Nx))
    
    cy_up = np.sum(np.conjugate(u_up) * u_up_py * f +
                        v_up * np.conjugate(v_up_py) * (1 - f), axis=1).reshape((Ny, Nx))
    
    cx_dn = np.sum(np.conjugate(u_dn) * u_dn_px * f +
                        v_dn * np.conjugate(v_dn_px) * (1 - f), axis=1).reshape((Ny, Nx))
    
    cy_dn = np.sum(np.conjugate(u_dn) * u_dn_py * f +
                        v_dn * np.conjugate(v_dn_py) * (1 - f), axis=1).reshape((Ny, Nx))
    
    return cx_up, cy_up, cx_dn, cy_dn

# --- end just AM ------------------------------------------------------------



if __name__ == "__main__":
    # run_onemat()
    # run_onemat("AM")
    # run_straightskewed(False, "AM")
    # run_straightskewed(True, "AM")
    # run_straightskewed(False, "FM")
    # run_straightskewed(True, "FM")

    # run_PAP("P", "AM")
    # run_PAP("AP", "AM")
    run_PAP("P", "FM")
    run_PAP("AP", "FM")

    # run_imp_oneval(0.5, 0., 1.0, 0.4, "AM")

    """ms = np.arange(0.42, 0.45, 0.01)
    print(ms)
    for m in ms:
        run_imp_oneval(m, 0., 1.0, 0.2, "AM")
    """
    
    # run_imp_oneval(0., 0.5, 1.0, 0.4, "FM")
    # main()

    # run_imp_just_AM_mNfracplot(0, 1.0, "AM", 0.001)

    # run_imp_just_AM_mNfracplot(0, 3.0, "AM", 0.001)


    # run_imp_just_AM_wNfracplot(0.75, 0, "AM", 0.001)

    # run_imp_just_AM_wNfracplot(0.25, 0, "AM", 0.001)

    # run_imp_just_AM_mmufracplot(0, "AM", 0.001)

    # run_onemat("AM")
    # run_onemat("FM")

    # run_onemat_Delta("AM", 1e-8, 0.01)
    # run_onemat_Delta("FM", 1e-8, 0.01)