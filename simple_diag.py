from matplotlib import pyplot as plt
import numpy as np
import time
# import cProfile
from numba import njit
# from scipy import sparse
# from scipy.sparse.linalg import eigsh
# from scipy.sparse.linalg import eigs
from mpl_toolkits.axes_grid1 import make_axes_locatable
from funcs import make_sigmas, nb_block, nb_block2, make_H_numba, fd, Delta_sc, unpack_block_matrix#, make_H_FT

# from multiprocessing import Pool

import sys
from icecream import ic
# from concurrent.futures import ProcessPoolExecutor

# @njit()
def does_Delta_increase(Nx, Ny, m_arr, mz_arr, hx_arr, Deltag, T, Ui, mu, imps, Delta_arr1, bd,  NDelta, skewed, periodic):
    # Here, Deltag must be the guess, if Delta < Deltag, 
    Delta = Delta_arr1.copy()
    for i in range(NDelta):
            H = make_H_numba(Nx, Ny, m_arr, mz_arr, hx_arr, Delta, mu, imps, skewed, periodic)

            D, gamma = np.linalg.eigh(H)
            D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
            Delta = Delta_sc(gamma, D, Ui, T).reshape(Ny, Nx)

    # ind = np.nonzero(Delta)
    # Delta_bulk = np.average(np.abs(Delta[ind]))
    Delta_bulk = np.median(np.abs(Delta))

    # if bd[1] < Nx:
    #     Delta_bulk = np.abs(Delta[Ny//2, (bd[0] + bd[1])//2])

    # else:
    #     Delta_bulk = np.abs(Delta[Ny//2, (bd[0] + Nx)//2])

        
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
def calc_Delta_sc(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, Deltag,tol, Ui, mu,T, skewed, alignment, periodic):

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
        H = make_H_numba(Nx, Ny,m_arr, mz_arr, hx_arr, Delta_old, mu, skewed, periodic)

        D, gamma = np.linalg.eigh(H)
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
        if np.median(np.abs(np.abs(Delta_new) - np.abs(Delta_old)))  <= tol :
            done = True
            ic(it)
        # Delta_old = Delta_new
        Delta_old = Delta_new
        
        # Delta_old_bulk = Delta_bulk

    # print("Used ", it, " iterations to calculate Delta self-consist.")
    return Delta_new, gamma, D

# @njit()
def calc_Tc_binomial(Nx, Ny, m_arr, mz_arr,hx_arr, Delta_arr, Deltag, Ui, imps, mu, Tc0, bd, num_it, skewed, alignment, periodic):
    N = 18 # Look at, maybe not needed this accuracy
    if alignment == None:
        assert bd[1] >= Nx
    # The first calculation is the same for all temperatures --------------
    # x = np.arange(0, Nx)
    # Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)#*(x - bd)**2*0.05 / 40**2
    
    H = make_H_numba(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, mu, imps, skewed, periodic)
    # plt.imshow(np.real(H - np.conjugate(H.T))[::4, ::4])
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
        Delta_arr1 = Delta_sc(gamma, D, Ui, T).reshape(Ny, Nx)


        if does_Delta_increase(Nx, Ny, m_arr, mz_arr, hx_arr, Deltag, T, Ui, mu,imps,  Delta_arr1, bd, num_it, skewed, periodic): # Meaning that there is SC at this temp, need to go higher in T to find Tc
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

def N_sigma(gamma, D, T, Nx, Ny):
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]

    # Fuu = np.sum(u_up *np.conjugate(v_up) , axis = 1)
    # print(Fuu)
    f  = (1 - np.tanh(D / (2 * T))) / 2
    N_up = np.sum(u_up *np.conjugate(u_up) * f + v_up * np.conjugate(v_up)* (1 - f) , axis = 1).reshape(Ny, Nx)
    N_dn = np.sum(u_dn *np.conjugate(u_dn) * f + v_dn * np.conjugate(v_dn)* (1 - f) , axis = 1).reshape(Ny, Nx)

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



def make_system_normal(bd, U, mz,hx, mg, impdata, Deltag, Nx, Ny, alignment):

    w_AM, w_SC, NfracAM, NfracSC = impdata
    imps = np.zeros((Ny, Nx))
    imps[:, :bd[0]] = make_impurities(imps[:, :bd[0]], w_SC, NfracSC )
    imps[:, bd[1]:] = make_impurities(imps[:, bd[1]:], w_SC, NfracSC )

    imps[:, bd[0]:bd[1]] = make_impurities(imps[:, bd[0]:bd[1]], w_AM, NfracAM )
    # plt.imshow(imps)
    # plt.show()
    Delta_arr = (np.ones((Nx*Ny))*Deltag).reshape(Ny, Nx)# To make it complex?
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
        mg_arr[:, bd[1]:]  *= -1
        hx_arr[:, bd[1]:] *= -1
        mz_arr[:, bd[1]:] *= -1



    # plt.imshow(m_arr)
    # # plt.colorbar()
    # plt.show()
    # plt.imshow(np.abs(Delta_arr))
    # plt.colorbar()
    # plt.show()    
    # plt.imshow(np.abs(Ui))
    # plt.colorbar()
    # plt.show()
    if mz != 0:

        assert alignment == None or alignment == "P"



    return Ui, mz_arr, hx_arr, mg_arr, Delta_arr, imps

def make_system_one_material(U, mz, hx, m, Deltag, Nx, Ny, w, Nfrac):
    Delta_arr = (np.ones((Nx*Ny))*Deltag).reshape(Ny, Nx)# To make it complex?


    U = U * (1. + 0j)
    Ui = np.ones(Nx*Ny).reshape(Ny, Nx)*U
    Ui = Ui.reshape((Nx*Ny))   


    m_arr = (np.ones((Nx*Ny)) * m).reshape(Ny, Nx)
    # plt.imshow(m_arr)
    # plt.colorbar()
    # plt.show()
    mz_arr = (np.ones((Ny*Nx))*mz).reshape(Ny, Nx)
    hx_arr = (np.ones((Nx*Ny)) * hx).reshape(Ny, Nx)

    ic("Running for one material, not a heterostructure. Vals:")

    imps = np.zeros((Ny, Nx))
    imps = make_impurities(imps, w, Nfrac)
    return Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps


def plot_observables_constantT(Delta, gamma, D, T, Nx, Ny, NDelta, mz, bd, U, mu, Deltag, tol, alignment, skewed):
    x = np.arange(1, Nx + 1, 1)
    ic( T, Nx, Ny, NDelta, mz, bd, U, mu, Deltag, tol, alignment, skewed)
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
    ax.plot(x, N_up[Ny//2, :], label = r"N_\uparrow")
    ax.plot(x, N_dn[Ny//2, :], label = r"N_\downarrow")

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
    periodic = np.array([True, True]) # x-direction, y-direction. Should only have x.dir for the one material thing.
    bd = np.array([Nx, Nx])
    w = 0
    Nfrac = 0
    periodic = np.array([True, True]) # x-direction, y-direction. Should only have x.dir for the one material thing.
    skewed = False
    alignment = None

    # Make Hamiltonian parameters
    Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps = make_system_one_material(U, mz, hx, mg, Deltag, Nx, Ny, w, Nfrac)

    Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)

    return Tc


def task_straightskewed(Nx, Ny, NDelta, mz, hx, mg, skewed, wAM, NfracAM):
    # Constants
    U = 1.7
    mu = -0.5
    Deltag = 1e-4 + 0.j
    Tc0 = 0.2

    # Problem spesific constants
    periodic = np.array([False, True]) # x-direction, y-direction. Should only have x.dir for the one material thing.
    bd = np.array([Nx//2, Nx]) # SC to left, AM to right
    NfracSC = 0.0 # Fraction of lattice sites that will get impurities in SC
    w_SC = 0.0
    impdata = [wAM, w_SC, NfracAM, NfracSC]
    alignment = None

    # Make Hamiltonian parameters
    Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps = make_system_normal(bd, U, mz, hx, mg, impdata, Deltag, Nx, Ny, alignment)
    Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)
    return Tc

def task_PAP(Nx, Ny, bd, NDelta, mz, hx, mg, alignment, wAM, NfracAM):
    # Constants
    U = 1.7
    mu = -0.5
    Deltag = 1e-4 + 0.j
    Tc0 = 0.2
    ic(mg, mz)

    # Problem spesific constants
    periodic = np.array([False, True]) # x-direction, y-direction. Should only have x.dir for the one material thing.
    NfracSC = 0.0 # Fraction of lattice sites that will get impurities in SC
    w_SC = 0.0
    impdata = [wAM, w_SC, NfracAM, NfracSC]
    skewed = False

    # Make Hamiltonian parameters
    Ui, mz_arr, hx_arr, m_arr, Delta_arr, imps = make_system_normal(bd, U, mz, hx, mg, impdata, Deltag, Nx, Ny, alignment)
    Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)
    return Tc



# def main():
#     Nx = 30
#     Ny = 20

#     NDelta = 10
#     Tc0 = 0.2

#     bd = np.array([10, 20])
#     # bd_Onemat = np.array([Nx, Nx])
#     # mg = 0.0
#     # hx = 0.1
#     # Make sure to set magnetism to zero if not already defined.
#     # mg = 0.02
#     # mg = 0.0
#     # make_impurities(Nx, Ny, bd, 0.1, int(Nx*Ny / 10))
#     try: mz
#     except NameError: mz = 0
#     try: mg
#     except NameError: mg = 0
#     try: hx
#     except NameError: hx = 0
#     # mg = 0.
#     # bd = np.array([Nx//3,  (2 *Nx)//3 ])
#     # bd = np.array([10, Nx - 10])
#     U = 1.7
#     mu = -0.5
#     Deltag = 1e-5 + 0.j
#     # tol = 1e-10
#     alignment = "AP"
#     ic(alignment)
#     skewed = False
#     periodic = np.array([False, True]) # x-direction, y-direction. Should only have x.dir for the one material thing.
#     ic(bd)

#     NfracAM = 0.2*0 # Fraction of lattice sites that will get impurities in AM
#     NfracSC = 0.0 # Fraction of lattice sites that will get impurities in SC
#     w_AM = 1.0*0
#     w_SC = 0.0
#     # A = np.zeros((10, 20))
#     # make_impurities(A, w_AM, NfracAM)

#     impdata = [w_AM, w_SC, NfracAM, NfracSC]


#     # Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)
#     # try: Tc
#     # except: Tc = 0
#     # ic(Tc)
#     # T = Tc
#     # Tc = None
#     # T = 0
#     # Delta, gamma, D = calc_Delta_sc(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, Deltag, tol, Ui, mu, T, skewed, alignment, periodic)
#     # ic(np.average(np.abs(Delta)))

#     # try: Delta
#     # except: Delta = 1
#     # plot_observables_constantT(Delta, gamma, D, T, Nx, Ny, NDelta, mz, bd, U, mu, Deltag, tol, alignment, skewed)
#     tic = time.time()
#     # def f(x): return x*x
#     # mgs = np.linspace(0, .1, 20)
#     # mgs = np.linspace(0.0, 0.05, 40)
#     # mzs = np.arange(0., 0.2 + 0.001, 0.001)
#     # mzs = np.arange(0., 0.2 + 0.02, 0.01)
#     # mgs =  np.arange(0.0, 1.0 + 0.01, 0.01)
#     mgs =  np.arange(0.0, 1.0 + 0.01, 0.01)

#     Tcs = np.zeros_like(mgs)
#     tic =  time.time()
#     Deltas = np.zeros_like(mgs)
#     ic( mz, hx, alignment, skewed, Nx, Ny, bd, U, mu, impdata)

#     for i, mg in enumerate(mgs):    
#             ic(mg)

#             Ui, mz_arr, hx_arr,  m_arr, Delta_arr, imps = make_system_normal(bd, U, mz, hx, mg, impdata, Deltag, Nx, Ny, alignment)

#             Tcs[i] = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, hx_arr, Delta_arr, Deltag, Ui, imps, mu, Tc0, bd, NDelta, skewed, alignment, periodic)
        

#     np.save(f"NewData2/mg_al{alignment}sw{skewed}_ND={NDelta}Ny={Ny}Nx={Nx}mu={mu}bd{bd}.npy",  np.array([mgs, NDelta, Ny, Deltas, Tcs], dtype=object))
#     # np.save(f"IMPDATA/mg_al{alignment}sw{skewed}_ND={NDelta}Ny={Ny}Nx={Nx}mu={mu}deltag={deltag}.npy",  np.array([mgs, NDelta, Ny, Deltas, Tcs], dtype=object))
#     # Tcs_noimp = np.load("Noimp.npy")    
#     # np.save("NoImp", Tcs)
#     tac = time.time()
#     ic(tac-tic)
#     plt.plot(mgs, Tcs, label = "Tc")

#     plt.legend()
#     plt.title(f"m={0.0}")
#     plt.show()



from multiprocessing.pool import Pool

def run_onemat(magnettype):
    # Prepare for multiprocessing in the Onemat case
    
    # Run spesific parameters:
    Nx = 5
    Ny = 5
    NDelta = 10
    mz = 0
    hx = 0
    items = []
    numsteps = 100

    tic = time.time()
    if magnettype == "AM":
        mgs = np.linspace(0, 1.0 , numsteps )
        mz = 0
        for mg in mgs:
            items.append((Nx, Ny, NDelta, mz, hx, mg))
    elif magnettype =="FM":
        mzs = np.linspace(0, 1.0 , numsteps )
        mg = 0
        for mz in mzs:
            items.append((Nx, Ny, NDelta, mz, hx, mg))
    else:
        raise NameError

    print(items)

    with Pool() as pool:
        Tcs = pool.starmap(task_onematerial, items, chunksize=10)

    tac = time.time()
    ic(tac - tic)
    np.save(f"Newdata4/onemat/{magnettype}{(Nx, Ny, NDelta, mz, hx, mg)}", np.array(items))

    plt.plot(mgs, Tcs)
    plt.show()
    return Tcs

def run_straightskewed(skewed, magnettype):

    # Run spesific parameters:
    Nx = 20
    Ny = 20
    NDelta = 10
    hx = 0
    wAM = 0
    NfracAM = 0
    items = []
    tic = time.time()
    # Prepare for multiprocessing in the straight/skewed case
    numsteps = 100

    if magnettype == "AM":
        mgs = np.linspace(0, 1.0, numsteps )
        mz = 0
        for mg in mgs:
            items.append((Nx, Ny, NDelta, mz, hx, mg, skewed, wAM, NfracAM))
    elif magnettype =="FM":
        mzs = np.linspace(0, 1.0 , numsteps )
        mg = 0
        for mz in mzs:
            items.append((Nx, Ny, NDelta, mz, hx, mg, skewed, wAM, NfracAM))
    else:
        raise NameError


    print(items)
    with Pool() as pool:
        Tcs = pool.starmap(task_straightskewed, items, chunksize = 10)
    tac = time.time()
    ic(tac - tic)
    np.save(f"Newdata4/straightskewed/{magnettype}{skewed}{(Nx, Ny, NDelta, mz, hx, mg, skewed, wAM, NfracAM)}", np.array(items, dtype=object))

    plt.plot(mgs, Tcs)
    plt.show()
    return Tcs

def run_PAP(alignment, magnettype):
    # Prepare for multiprocessing in the straight/skewed case
    numsteps = 100
    # Run spesific parameters:
    Nx = 35
    Ny = 20
    bd = [10, Nx - 10]
    NDelta = 10
    hx = 0
    wAM = 0
    NfracAM = 0
    items = []
    tic = time.time()
    print(f"Running run_PAP for a {magnettype} in the {alignment} alignment with boundaries {bd}. Parameters: (without magnets): {(Nx, Ny, bd, NDelta,  hx,  alignment, wAM, NfracAM)}")

    if magnettype == "AM":
        mgs = np.linspace(0, 1.0, numsteps )
        mz = 0
        for mg in mgs:
            items.append((Nx, Ny, bd, NDelta, mz, hx, mg, alignment, wAM, NfracAM))

    elif magnettype =="FM":
        assert alignment == "P"
        mzs = np.linspace(0, 1.0 , numsteps )
        mg = 0
        for mz in mzs:
            items.append((Nx, Ny, bd, NDelta, mz, hx, mg, alignment, wAM, NfracAM))
    else:
        raise NameError
    
    print(items)
    with Pool() as pool:
        Tcs = pool.starmap(task_PAP, items, chunksize=5)

    tac = time.time()
    print(tac - tic)
    print(f"Finished running run_PAP for a {magnettype} in the {alignment} alignment with boundaries {bd}. Parameters: (including last value of magnetic strength): {(Nx, Ny, bd, NDelta, mz, hx, mg, alignment, wAM, NfracAM)}")
    
    np.save(f"Newdata4/PAP/{magnettype}{alignment}{(Nx, Ny, bd, NDelta, mz, hx, mg, alignment, wAM, NfracAM)}", np.array(items, dtype=object))

    plt.plot(mgs, Tcs)
    plt.show()
    return Tcs

# import sys
if __name__ == "__main__":
    # run_onemat()
    # run_onemat("AM")
    run_straightskewed(False, "AM")
    # run_PAP("P", "AM")
    # run_PAP("AP", "AM")
    # run_PAP("P", "FM")
    # main()
