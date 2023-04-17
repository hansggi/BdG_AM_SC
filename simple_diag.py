from matplotlib import pyplot as plt
import numpy as np
import time
# import cProfile
from numba import njit
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs

from funcs import make_sigmas, nb_block, nb_block2, make_H_numba, f, Delta_sc, unpack_block_matrix

from multiprocessing import Pool
# print("Number of cpuf : ", multiprocessing.cpu_count())

def does_Delta_increase(Nx, Ny, Deltag, T, param, Delta_arr1, bd,  skewed = False):
    # Here, Deltag must be the guess, if Delta < Deltag, 
    t, U, mu, mg, mz = param
    num_it = 5

    done = False
    Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_arr[:, :bd] = 0

    it = 0
    while not done:
        if it ==0 :
            Delta_arr = Delta_arr1
            # Deltag = np.abs(Delta_arr[(3 * Nx) // 4 + Nx *(Ny//2)  ] )
            # Deltag = np.abs(Delta_arr[Ny //2, - Nx //4] )
            # print(Deltag, "hh")
            # Deltag = np.sum(np.abs(Delta_arr))/ ( Nx * Ny / 2)

        else:
            H = make_H_numba(Nx, Ny, Delta_arr, param, bd,  skewed = skewed)
            # print(np.allclose(H, np.conjugate(H).T))
            D, gamma = np.linalg.eigh(H)
            D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]

            Delta_arr = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)

        
        Delta_bulk = np.abs(Delta_arr[Ny//2, (bd + Nx)//2])

        it += 1
        
        if it > num_it and Delta_bulk > Deltag :
            increase = True

            return increase
        
        elif it > num_it and Delta_bulk <= Deltag:
            increase = False
            return increase

def calc_Delta_sc(Nx, Ny, Deltag, tol, T, param, bd,  skewed):
    t, U, mu, mg, mz = param

    # tol = 0.001 # Absolute. Think  better way to truncate, maybe Ali's article?
    done = False
    Delta_old = Deltag
    Delta_old_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_old_arr[:, :bd] = 0

    Delta_old_bulk = Deltag
    it = 0

    while not done:
        H = make_H_numba(Nx, Ny, Delta_old_arr, param, bd, skewed = skewed)
        # print(np.allclose(H, np.conjugate(H).T))
        # plt.imshow(np.abs(H[::4, ::4]))
        # plt.colorbar()
        # plt.show()
        # Sparse --------------------------------------------------------------------------
        # num_eigvals = 5
        # sH = sparse.csr_matrix(H)  # NB, seems H is not Hermitian, which is not good
        # sD, sgamma = eigs(H, k = num_eigvals, which = "SM")
        # # sD, sgamma = eigsh(H, k = num_eigvals, which = "SM")

        # # # print(sorted(sD))
        # sgamma = sgamma[:, sD > 0]
        # sD = sD[sD >= 0]
        # num_eigvals_remaining = len(sD)
        # # print(num_eigvals_remaining)
        # # print(f(np.max(sD), T))
        # if f(np.max(sD), T) > 0.00001:
        #     print(f"ERROR, chose too few e.v., got f(Emax) = {f(np.max(sD), T)}")
        # D = sD
        # gamma = sgamma
        # #        ----------------------------------------------------------------------------
        # Exact, keep so that we can compare

        D, gamma = np.linalg.eigh(H)
        D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]

       
        # print()
        # print(D[:num_eigvals_remaining])
        # time.sleep(20)
        # # print(" From exact ( just pos part): \n", D[:8])
        # ----------------------------------------------------------------------------------

        Delta_new_i = Delta_sc(gamma, D, U, T, Nx, Ny, bd)
        Delta_new = Delta_new_i.reshape(Ny, Nx)

        Delta_bulk = np.abs(Delta_new)[Ny//2, (bd + Nx)//2]
        it += 1
        # Bulk method
        if np.abs(np.abs(Delta_bulk) - np.abs(Delta_old_bulk))  <= tol :
            done = True

        # Using max difference instead. Will not give the same plot as in the article, since T is a function of bulk Tc here
        # if np.amax(np.abs(np.abs(Delta_bulk) - np.abs(Delta_old_bulk)))  <= tol :
        #     done = True
        Delta_old = Delta_new
        Delta_old_arr = Delta_new
        
        Delta_old_bulk = np.abs(Delta_old[Ny//2, (bd + Nx)//2])
    # print("Used ", it, " iterations to calculate Delta self-consist.")
    return Delta_new_i, gamma, D

def calc_Tc(Nx, Ny, Deltag, param, skewed ):
    t, U, mu, mg, mz = param
    # Delta0 = Deltag
    nTs = 50
    # Deltas = np.zeros((nTs))
    Ts = np.linspace(0.0001*t, 0.5*t, nTs)
    # found = False
    # tol = 0.02

    # The first calculation is the same for all temperatures --------------
    Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)

    H = make_H_numba(Nx, Ny, Delta_arr, param, bd, skewed = skewed)
    
    D, gamma = np.linalg.eigh(H)
    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
    # --------------------------------------------------------------------
    for i, T in enumerate(Ts):
        print(f"Checking for T = {T}")
        Delta_arr1 = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)

        Ts[i] = T
        # print("NOW: ", type(Delta_arr1))
        if not does_Delta_increase(Nx, Ny, Deltag, T, param, Delta_arr1, bd, skewed=skewed):
            Tc = T
            return Tc, Ts #, Deltas, Delta_i, gamma, D
        # cProfile.run("Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Delta0, 0.001, 0.0001, param, skewed=skewed)")
        # time.sleep(100)

    #     Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T, param, skewed=skewed)
    #     Delta_arr = Delta_i.reshape(Ny , Nx)
    #     Deltas[i] = np.abs(np.sum(Delta_i))/(Nx * Ny / 2)
    #     Delta_bulk = Delta_arr[Ny//2, (3 * Nx) // 4]
        
    #     # print(Ny//2, (3 * Nx) // 4)
    #     # print(np.abs(Delta_bulk))
    #     # print("Deltas i", Deltas[i])

    #     # time.sleep(5)
    #     # print(does_Delta_increase(Nx, Ny, Delta0, tol, T, param, skewed=skewed))
    #     if i > 1 and Deltas[i]  < Deltag  and not found:
    #         found = True
    #         Tc = T
    #         print(f" No sc at T = {T}")
        
    
    #         return Tc, Ts, Deltas, Delta_i, gamma, D

    # print(" Too low range, no cutoff found")

def calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd,  skewed ):
    N = 11 # Look at, maybe not needed this accuracy
    t, U, mu, mg, mz = param

    # The first calculation is the same for all temperatures --------------
    Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_arr[:, :bd] = 0
    H = make_H_numba(Nx, Ny, Delta_arr, param, bd, skewed = skewed)

    D, gamma = np.linalg.eigh(H)
    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
    # ---------------------------------------------------------------

    Ts_lower = 0
    Ts_upper = Tc0

    for i in range(N):
        T = (Ts_upper + Ts_lower ) / 2
        Delta_arr1 = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)

        if does_Delta_increase(Nx, Ny, Deltag, T, param, Delta_arr1, bd,  skewed=skewed):
            Ts_lower = T
        else:
            Ts_upper = T #, Deltas, Delta_i, gamma, D
    return T
 
# Run one time
"""Nx = 30
Ny = 5
skewed = False
t = 1.
U = 2.
mu = +0.1
mg = 0.
Deltag = 1e-5
param = (t, U, mu, mg, mz)

# Initial guess for Delta is a constant in the SC, 0 in the AM
Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
Delta_arr[:, :Nx//2] = 0

#H_toreturn = np.zeros((4*Nx*Ny, 4*Nx*Ny), dtype = complex)
H = make_H_numba_boundaryjumping(Nx, Ny, Delta_arr, param, skewed = skewed)
print(" Is Herm? : ", np.allclose(H, np.conjugate(H).T))
# plt.imshow(np.abs(H))
# plt.show()
# H = unpack_block_matrix(H_block, H_toreturn, Nx, Ny)

D, gamma = np.linalg.eigh(H)
D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
# plt.plot(D)
# plt.show()
# Calculating Delta for this one temperature
tic = time.time()
Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, 0.00001, 0.00001, param, skewed=skewed) #tol, T
print(f"Calculating one self-consistently took {time.time()- tic} seconds")
Delta_i = Delta_i.reshape((Ny, Nx))
plt.imshow(np.abs(Delta_i), aspect="auto")
plt.colorbar()
plt.show()"""
# ---------------------------------------------------------------
# Calc one Delta plot over temp
# Ts = np.linspace(0.0001, 0.2, 20)
# Deltas = np.zeros(20, dtype=complex)
# print(Ny//2, -Nx//4)
# for i, T in enumerate(Ts):
#     H = make_H_numba_boundaryjumping(Nx, Ny, Delta_arr, param, skewed = skewed)
#     D, gamma = np.linalg.eigh(H)
#     D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
#     print(f"Checking for T = {T}")
#     Delta_arr1, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, 0.001, T, param, skewed = False)
#     # Delta_arr1 = Delta_sc(gamma, D, U, T, Nx, Ny)
#     # Delta_arr1[:, :Nx//2]= 0
#     Deltas[i] = np.reshape(Delta_arr1, (Ny, Nx))[Ny//2, -Nx//4]

# # tic = time.time() 

# # Tc, Ts,  Deltas, Delta_i, gamma, D = calc_Tc(Nx, Ny, Deltag, param,  skewed = skewed)
# # print(f"Took {time.time()- tic} seconds to calculate Tc")
# plt.xlabel("T/t")
# plt.ylabel(r"$\Delta_{avr}$ / t")
# plt.plot(Ts, np.abs(Deltas))
# plt.show()

# Tc, Ts = calc_Tc(Nx, Ny, Deltag, param, skewed = skewed)
# print(f"Tc using the increase method, linear search: {Tc}")

# Tc0 = 1
# Tc = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, skewed = skewed)
# print(f"Tc using the increase method, binomial search: {Tc}")

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


"""Nx = 100
Ny = 1
skewed = False
t = 1.0 # Assume nn hopping term to be isotropic and constant for all lattice points. Set all quantities from this
U = 1.5 * t # Strength of attractive potential, causing s.c. Assumed to be constant here
Deltag =  1e-5 #  First guess: constant, small Delta inside the SC, will be updated self-consistently
mu =    -0.5*t # So far, constant chemical potential
# mz = 0. # Ferromagetism in the AM
# mg = 0
mg = 0.0
bd = Nx // 7*0
mz = 0
param = (t, U, mu, mg, mz )"""
# def mg_sweep_Delta(Nx, Ny, mg_start, mg_end, num_sweep_vals, U, mu, Deltag, bd, skewed ):

def sweep_Delta(Nx, Ny, mg, U, mu, Deltag, bd, tol, skewed ):
    mz = 0
    t = 1.
    Tc0 = 0.3
    # Calc Tc in the mg = 0 case
    param0 = (t, U, mu, 0, 0)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag, param0, Tc0, bd, skewed = skewed) 
    print(Tc)
    # -----------------------------------------------------------
    # Tc in this case
    param = (t, U, mu, mg, mz)
    Tc2 = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd, skewed = skewed) 
    print(f"Tc :{Tc:.4f}, \nTc2:{Tc2:.4f}.")
    T_plots = np.array([1e-8, 0.5 * Tc , 0.95 * Tc, 0.98*Tc, 1.0*Tc])
    fig, ax = plt.subplots()
    for i, T in enumerate(T_plots):
        Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T_plots[i], param, bd,  skewed=skewed) 
        Delta_i = Delta_i.reshape((Ny, Nx))
        ax.plot(np.abs(Delta_i[Ny//2, :]), label = f"T/Tc = {T/Tc:.2f}")

    fig.suptitle(f" mg ={mg:.1f}, U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    fig.legend()
    plt.savefig(f"1d_Delta/N=({Nx},{Ny}),mg={mg:.1f}.pdf", bbox_inches='tight')

    # plt.savefig(f"1d_Delta/N=({Nx},{Ny}),mg={mg:.1f}.pdf", bbox_inches='tight')
    # plt.show()

tic = time.time()
sweep_Delta(Nx = 20, Ny = 20, mg = 0. , U = 1.7, mu = -0.5, Deltag = 1e-5, bd = 20, tol = 1e-9, skewed=False)
# # sweep_Delta(Nx = 50, Ny = 1, mg = 0.5, U = 1.7, mu = -0.5, Deltag = 1e-5, bd = 10, tol = 1e-9, skewed=False)
# # sweep_Delta(Nx = 50, Ny = 1, mg = 1.0, U = 1.7, mu = -0.5, Deltag = 1e-5, bd = 10, tol = 1e-9, skewed=False)
# # sweep_Delta(Nx = 50, Ny = 1, mg = 2.0, U = 1.7, mu = -0.5, Deltag = 1e-5, bd = 10, tol = 1e-9, skewed=False)
# # sweep_Delta(Nx = 50, Ny = 1, mg = 4.0, U = 1.7, mu = -0.5, Deltag = 1e-5, bd = 10, tol = 1e-9, skewed=False)
# sweep_Delta(Nx = 40, Ny = 1, mg = 8.0, U = 1.7, mu = -0.5, Deltag = 1e-5, bd = 10, tol = 1e-9, skewed=False)
print(f"took {time.time()- tic} seconds")

# plt.show()

def mg_sweep_Tc(Nx, Ny, mg_start, mg_end, num_sweep_vals,  U, mu, Deltag, bd, skewed ):
    mz = 0
    t = 1.
    Tc0 = 0.3
    # Calc Tc in the mg = 0 case
    param0 = (t, U, mu, 0, 0)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag, param0, Tc0, bd, skewed = skewed) 
    print(Tc)
    # -----------------------------------------------------------
    # Tc in this case
    # param = (t, U, mu, mg, mz)
    # Tc2 = calc_Tc_binomial(Nx, Ny, Deltag, param0, Tc0, bd, skewed = skewed) 
    # print(f"Tc :{Tc:.4f}, \nTc2:{Tc2:.4f}.")
    T_plots = np.array([1e-8, 0.5 * Tc , 0.95 * Tc, 0.98*Tc, 1.0*Tc])
    mgs = np.linspace(mg_start, mg_end, num_sweep_vals)
    Tcs = np.zeros_like(mgs)
    fig, ax = plt.subplots()
    for i, mg in enumerate(mgs):
        print(f"running for mg = {mg}")
        param = (t, U, mu, mg, mz)
        # Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T_plots[i], param, bd,  skewed=skewed) 
        # Delta_i = Delta_i.reshape((Ny, Nx))
        Tc2 = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd, skewed=skewed)
        print(f"Tc :{Tc:.4f}, \nTc2:{Tc2:.4f}.")
        Tcs[i] = Tc2
        # ax.plot(np.abs(Delta_i[Ny//2, :]), label = f"T/Tc = {T/Tc:.2f}")

    ax.plot(mgs, Tcs)
    fig.suptitle(f" mg ={mg:.1f}, U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    # fig.legend()
    ax.set_xlabel("m")
    ax.set_ylabel("Tc")

    plt.savefig(f"mg_sweep/N=({Nx},{Ny}).pdf", bbox_inches='tight')
    # plt.show()

tic = time.time()
mg_sweep_Tc(Nx= 20, Ny= 20, mg_start=0, mg_end= 5, num_sweep_vals= 5,  U = 1.7, mu= -0.5, Deltag= 1e-5, bd = 10, skewed= False)
print(f"took {time.time()- tic} seconds")

plt.show()
# Calculate the kf
# Delta_arrF = np.zeros((Nx*Ny), dtype = complex).reshape(Ny, Nx)

# paramF = (t, 0, mu, mg, mz)
# H = make_H_numba(Nx, Ny, Delta_arrF, paramF, bd, skewed = skewed)

# D, gamma = np.linalg.eigh(H)
# D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
# plt.plot(D)
# plt.show()
# idx = (np.abs(D - mu)).argmin()
# vF = np.gradient(D)[idx]
# vF = 2 * t * np.sin()
# vF3 = np.arccos( - mu / 4 / t)
# print(vF3)
# print(vF)
# ----------------------------------------------------

num_sweep_vals = 1
tic = time.time()
# mg = 0
mz = 0
# print(bd)
mgs = np.linspace(0., 1.5, num_sweep_vals)
# mzs = np.linspace(0, 0.5, num_sweep_vals)
Tcs = np.zeros(num_sweep_vals)

tic = time.time()
fnum = 0
fig, ax = plt.subplots(nrows = 1, ncols= 1)

param = (t, U, mu, mg, mz)
Tc = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd,  skewed = skewed)
print(Tc)
# Tc = 0.20068359375
print(Tc)
T_plots = np.array([1e-5, 0.85 * Tc , 0.95 * Tc, 0.97*Tc,   Tc, 1.01 * Tc, 1.02*Tc])
Delta_is = np.zeros((7, Ny, Nx), dtype=complex)
# T_plot = 0.21
for i, T in enumerate(T_plots):
    print("Running for T/Tc = ", T / Tc)
    param = (t, U, mu, mg, mz)
    # if i == 0: #or i == num_sweep_vals -1 or i == num_sweep_vals//2:
        # Basically zero termperature, and low tolerance
    Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, 1e-9, T_plots[i], param, bd,  skewed=skewed) #tol, T
    Delta_i = Delta_i.reshape((Ny, Nx))
    Delta_is[i] = Delta_i
    print(np.amax(np.abs(Delta_i)))
    im = ax.imshow(np.abs(Delta_i), aspect="auto")
    # ax.set_title((f"mg = {mg:.1f}"))
    # fig.colorbar(im, ax=ax)
    # fnum += 1

    # Tc = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd,  skewed = skewed)
    # Tcs[i] = Tc
# Delta_is = np.array(Delta_is)
print(Delta_is.shape)
print(f"took {time.time()- tic} seconds")

# xi = np.sqrt(2 * np.abs(mu)) / np.pi /np.amax(np.abs(Delta_i))    # First factor is k_F
# xi2 = vF / np.pi /np.amax(np.abs(Delta_i))    # First factor is k_F
# xi3 = vF3 / np.pi /np.amax(np.abs(Delta_i))    # First factor is k_F

# print("Coherence length: ", xi)

# print("Coherence length2: ", xi2)

# print("Coherence length3: ", xi3)

print(f"Tc = {Tc}")
# fig.colorbar(im, ax=ax.ravel().tolist())
fig.suptitle(f"U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}, mz ={mz:.1f}")
plt.show()

for i in range(len(T_plots)):
    plt.plot(np.arange(Nx), np.abs(Delta_is[i][Ny//2, :]), label = f"T/Tc = {T_plots[i]/Tc:.2f}")

plt.legend()
plt.ylabel("abs(Delta)")
plt.title(f"U = {U}, mu = {mu}, N = {Nx, Ny},\n k_B Tc = {Tc:.2f}")
plt.show()
plt.plot(D)
plt.show()

# plt.title(f"U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}, mg={mg:.1f}")
# plt.xlabel("m/t")
# plt.ylabel("Tc")
# plt.plot(mgs, Tcs)
# plt.show()


# Nx = 20
# Ny = 20
# skewed = False
# t = 1.0 # Assume nn hopping term to be isotropic and constant for all lattice points. Set all quantities from this
# U = 2 * t*0 # Strength of attractive potential, causing s.c. Assumed to be constant here
# Deltag =  1e-5*0  # Assume constant Delta. First guess, will be updated self-consistently
# mu =  -0.5*t # So far, constant chemical potential
# mz = 0. # Ferromagetism in the AM
# mg = 0

# param = (t, U, mu, mg, mz)
# Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, 0.00001, 0.00001, param, skewed=skewed)

# plt.plot(D)
# plt.show()

# ldos = Ldos(gamma, D)
# E = np.linspace(-4, 4, 5000)
# ldos = ldos.reshape(Ny, Nx, 5000)
# plt.plot(E, ldos[Ny//2,- Nx //2,  :])
# plt.show()
# ----------------------------------------------------------
"""
plt.imshow(np.abs(Delta_i.reshape(Ny, Nx)), origin = "upper")
plt.colorbar()
plt.show()
u_up = gamma[0::4, :]
u_dn = gamma[1::4, :]
v_up = gamma[2::4, :]
v_dn = gamma[3::4, :]
plt.imshow(np.abs(u_up[:, 0].reshape(Ny, Nx)), origin = "upper")
plt.colorbar()
plt.show()
plt.imshow(np.abs(u_dn[:, 0].reshape(Ny, Nx)), origin = "upper")
plt.colorbar()
plt.show()"""

# Hs = sparse.lil_matrix((4*Nx*Ny, 4*Nx*Ny), dtype = complex)#, dtype= np.float32)

# H = unpack_block_matrix(H4, Nx, Ny)
# print(f"Took {time.time()- tic} seconds to make H unpacked")

# tic = time.time() 
# D, gamma = np.linalg.eigh(H)
# # Remove negative eigenvalues and their corresponding eigenvectors, as these are linearly dependent on the positive one
# D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
# gamma0 = gamma[:, 0] # The lowest energy eigenvector

# u_up = gamma[0::4, :]
# u_dn = gamma[1::4, :]
# v_up = gamma[2::4, :]
# v_dn = gamma[3::4, :]

# Self-consistent:


# Delta_avr = np.abs(np.sum(Delta_new_i))/ ( Nx * Ny / 2)
# print(f"Completed after {it} iterations. We have Delta > DElta_guess = {Delta_avr > Deltag }, and T = {T}")
# plt.imshow(AM, origin="upper")
# plt.show()
# print(Delta_new_i.shape)
# Delta_i = np.reshape(Delta_new_i, (Ny, Nx))
# plt.imshow(np.real(Delta_i), origin = "lower")
# plt.colorbar()
# plt.show()
# # u_up = gamma[0::4, -1].reshape(Ny, Nx)
# # plt.imshow(np.abs(u_up), origin = "lower")
# # plt.colorbar()
# # plt.show()
# print(f"Took {time.time()- tic} seconds to solve eigenvalue problem using dense")
# # Sparse way ------------------------------------------------------------
# # tic = time.time() 
# # num_ev = Nx*Ny
# # sD, sgamma = eigsh(H, k = num_ev, which = "LR")
# # sD_low, sgamma = eigsh(H, k = num_ev, which = "SR")
# # print(f"Took {time.time()- tic} seconds to solve eigenvalue problem for sparse matrix")
# # ----------------------------------------------------------------------------------
# ar = np.arange(0, Nx, 1) 

# # for nr in range(0, 50, 5):
# #     print(nr)
# #     gamma0 = np.reshape(gamma[::4, nr], (Nx, Ny))
# #     gamma1 = np.reshape(gamma[2:][::4, nr], (Nx, Ny))
# #     print(gamma1.shape)

# #     plt.imshow(np.abs(gamma0).T)
# #     plt.show()
# #     plt.plot(ar, np.real(gamma0[:, Ny//2]))
# #     plt.plot(ar, np.real(gamma1[:, Ny//2]))

# #     plt.show()
# # plt.plot(k, -mu -2 * t * np.cos(k), label = "anal")
# # # plt.plot(k, D, label = "num solid")
# # # plt.ylim(-2, 2)
# # plt.plot(k, D2[:N], label = "even")
# # plt.plot(k, D2[N:], label = "even")

# # # plt.plot(k[::2], D2[1:][::2 ], label = "odd")

# # plt.legend()

# # plt.show()

# # kx = np.linspace(-np.pi, np.pi, N)
# # ky = np.linspace(-np.pi, np.pi, N)
# # xx, yy = np.meshgrid(kx, ky)
# # Z = -mu - 2 * t * (np.cos(xx) + np.cos(yy))
# # plt.contour(Z)
# # plt.plot(k, -mu -2 * t * np.cos(k), label = "anal")
# # plt.plot(k, D, label = "num solid")
# # plt.ylim(-2, 2)
# # plt.contourf( D3[:N**2], label = "spin up")
# # plt.plot(D3[N**2:], label = "spin down")

# # plt.plot(k[::2], D2[1:][::2 ], label = "odd")
# # plt.plot(np.arange(2*N**2), np.abs(D3))
# # # plt.legend()

# plt.plot(np.arange(0, 2*Nx*Ny, 1), D[:], label = "H, abs of the energy")

# plt.legend()
# plt.show()
# # E = np.linspace(-3 * t, 3 * t, 1000)
# # plt.plot(E, f(E- mu))
# # plt.show()



# Paralellization --------------------------------------
"""def mpwrap(mg):
    param = (t, U, mu, mg, mz)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, skewed = skewed)
    return Tc"""

'''if __name__ == "__main__":
    
    p = Pool(1)
    Tcs = p.map(mpwrap, mgs)

    print(f"took {time.time()- tic} seconds")
    plt.title(f"U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}, mz={mz}")
    plt.xlabel("m/t")
    plt.ylabel("Tc")
    plt.plot(mgs, Tcs)
    plt.show()
'''
# if __name__ == "__main__":
#     procs = []
#     for mg in mgs:
#         print(mg)
#         proc = Process(target=mpwrap, args=(mg,))
#         procs.append(proc)
#         proc.start()
#     # complete the processes
#     for proc in procs:
#         proc.join()
# print(procs)
# print(f"Took {time.time() -tic} seconds")

# ----------------------------------------------------------------------