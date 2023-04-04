from matplotlib import pyplot as plt
import numpy as np
import time
import cProfile
from numba import njit
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs

from funcs import make_sigmas, nb_block, nb_block2, make_H_numba, f, Delta_sc, unpack_block_matrix


def does_Delta_increase(Nx, Ny, Deltag, T, param, Delta_arr1,  skewed = False):
    # Here, Deltag must be the guess, if Delta < Deltag, 
    t, U, mu, mg, mz = param
    num_it = 15

    done = False
    Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_arr[:, :Nx//2] = 0

    it = 0
    while not done:
        if it ==0 :
            Delta_arr = Delta_arr1
            # Deltag = np.abs(Delta_arr[(3 * Nx) // 4 + Nx *(Ny//2)  ] )
            # Deltag = np.abs(Delta_arr[Ny //2, - Nx //4] )
            # print(Deltag, "hh")
            # Deltag = np.sum(np.abs(Delta_arr))/ ( Nx * Ny / 2)

        else:
            H = make_H_numba(Nx, Ny, Delta_arr, param, skewed = skewed)

            D, gamma = np.linalg.eigh(H)

            D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]

            Delta_arr = Delta_sc(gamma, D, U, T, Nx, Ny).reshape(Ny, Nx)

        # Delta_arr = Delta_arr.reshape((Ny, Nx))
        # Delta_avr = np.sum(np.abs(Delta_arr))/ ( Nx * Ny / 2)
        Delta_bulk = np.abs(Delta_arr)[Ny//2, - Nx // 4]
        # print(Delta_bulk)
        it += 1
        
        if it > num_it and Delta_bulk > Deltag :
            # done = True
            increase = True
            print(f"T/ Bulk value after {num_it} iterations: ",T, Delta_bulk)

            return increase
        
        elif it > num_it and Delta_bulk < Deltag:
            increase = False
            return increase

def calc_Delta_sc(Nx, Ny, Deltag, tol, T, param,  skewed = False):
    t, U, mu, mg, mz = param

    # tol = 0.001 # Absolute. Think  better way to truncate, maybe Ali's article?
    done = False
    Delta_old = Deltag
    Delta_old_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_old_arr[:, :Nx//2] = 0

    Delta_old_avr = Deltag
    it = 0
    H_toreturn = np.zeros((4*Nx*Ny, 4*Nx*Ny), dtype = complex)

    while not done:
        H = make_H_numba(Nx, Ny, Delta_old_arr, param, skewed = skewed)
        # H = unpack_block_matrix(H_block, H_toreturn, Nx, Ny)
        # print("Is herm:? ", np.allclose(H, np.conjugate(H).T))

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

        Delta_new_i = Delta_sc(gamma, D, U, T, Nx, Ny)
        Delta_new = Delta_new_i.reshape(Ny, Nx)
        Delta_avr = np.sum(np.abs(Delta_new_i))/ ( Nx * Ny / 2)
        it += 1
        if np.abs((np.abs(Delta_avr) - np.abs(Delta_old_avr)))  < tol * np.abs(Delta_old_avr):
            done = True
        Delta_old = Delta_new
        Delta_old_arr = Delta_new
        Delta_old_avr = np.sum( np.abs(Delta_old) ) / ( Nx * Ny /  2)

    return Delta_new_i, gamma, D

def calc_Tc(Nx, Ny, Deltag, param,  skewed = False):

    t, U, mu, mg, mz = param
    # Delta0 = Deltag
    nTs = 50
    # Deltas = np.zeros((nTs))
    Ts = np.linspace(0.0001*t, 0.5*t, nTs)
    # found = False
    # tol = 0.02

    # The first calculation is the same for all temperatures --------------
    Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)

    H = make_H_numba(Nx, Ny, Delta_arr, param, skewed = skewed)

    D, gamma = np.linalg.eigh(H)
    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
    # --------------------------------------------------------------------
    for i, T in enumerate(Ts):
        print(f"Checking for T = {T}")
        Delta_arr1 = Delta_sc(gamma, D, U, T, Nx, Ny).reshape(Ny, Nx)

        Ts[i] = T
        # print("NOW: ", type(Delta_arr1))
        if not does_Delta_increase(Nx, Ny, Deltag, T, param, Delta_arr1,  skewed=skewed):
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


def calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0,  skewed = False):
    N = 10 # Look at, maybe not needed this accuracy
    t, U, mu, mg, mz = param

    # The first calculation is the same for all temperatures --------------
    Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_arr[:, :Nx//2] = 0
    H = make_H_numba(Nx, Ny, Delta_arr, param, skewed = skewed)


    D, gamma = np.linalg.eigh(H)
    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
    # ---------------------------------------------------------------

    Ts_lower = 0
    Ts_upper = Tc0

    for i in range(N):
        T = (Ts_upper + Ts_lower ) / 2
        Delta_arr1 = Delta_sc(gamma, D, U, T, Nx, Ny).reshape(Ny, Nx)

        if does_Delta_increase(Nx, Ny, Deltag, T, param, Delta_arr1,  skewed=skewed):
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
H = make_H_numba(Nx, Ny, Delta_arr, param, skewed = skewed)
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
#     H = make_H_numba(Nx, Ny, Delta_arr, param, skewed = skewed)
#     D, gamma = np.linalg.eigh(H)
#     D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
#     print(f"Checking for T = {T}")
#     Delta_arr1, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, 0.001, T, param, skewed = False)
#     # Delta_arr1 = Delta_sc(gamma, D, U, T, Nx, Ny)
#     # assert np.allclose(Delta_arr1[:, :Nx//2], 0)
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

# Sweep over some parameter --------------------------------
# Nx = 32
# Ny = 6
# skewed = False
# t = 1.0 # Assume nn hopping term to be isotropic and constant for all lattice points. Set all quantities from this
# # U = 2 * t # Strength of attractive potential, causing s.c. Assumed to be constant here
# Deltag =  1e-5  # Assume constant Delta. First guess, will be updated self-consistently
# # mu =  -0.5*t # So far, constant chemical potential
# mg = 0.5*t*0
# mu = -0.5
# -- Run one time-- ------------------------------------------------------------------------------------------------------------


# param = (t, U, mu, mg, mz)

# Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
# Delta_arr[:, :Nx//2] = 0
# ish 5, 
# H = make_H_numba(Nx, Ny, Delta_arr, param, skewed = skewed)
# print(H.shape)
# print(" Is Herm? : ", np.allclose(H, np.conjugate(H).T))
# plt.imshow(np.abs(H)[::4, ::4])
# plt.show()
# D, gamma = np.linalg.eigh(H)
# D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
# plt.plot(D)
# plt.show()
# Calculating Delta for this one temperature
# tic = time.time()
# Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, 0.00001, 0.00001, param, skewed=skewed) #tol, T
# print(f"Calculating one self-consistently took {time.time()- tic} seconds")
# Delta_i = Delta_i.reshape((Ny, Nx))
# plt.imshow(np.abs(Delta_i), aspect="auto")
# plt.colorbar()
# plt.title(f"U = {U}, mu = {mu}, N = {Nx, Ny}")
# plt.show()
# # -----------------------------------------------------------------------------------------------------------------------
Nx = 100
Ny = 5
skewed = False
t = 1.0 # Assume nn hopping term to be isotropic and constant for all lattice points. Set all quantities from this
# U = 2 * t # Strength of attractive potential, causing s.c. Assumed to be constant here
Deltag =  1e-5  # Assume constant Delta. First guess, will be updated self-consistently
# mu =  -0.5*t # So far, constant chemical potential
# mg = 0.5*t*0
mu = -0.5
U = 2.0
# U = 2.0
# mz = 0.
mz = 0
num_sweep_vals = 1
tic = time.time()
mgs = np.linspace(0.5, 5.0, num_sweep_vals)
# mzs = np.linspace(0, 3, num_sweep_vals)
Tcs = np.zeros(num_sweep_vals)
Tc0 = 0.3
for i, mg in enumerate(mgs):
    print("Running for mg = ", mg)
    param = (t, U, mu, mg, mz)
    if i == 0 or i == num_sweep_vals -1 or i == num_sweep_vals//2:
        # Basically zero termperature, and low tolerance
        Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, 0.00001, 0.00001, param, skewed=skewed) #tol, T
        Delta_i = Delta_i.reshape((Ny, Nx))

        plt.imshow(np.abs(Delta_i), aspect="auto")
        plt.colorbar()
        plt.title(f"U = {U}, mu = {mu}, N = {Nx, Ny}, mz = {mz:.1f}, sw = {skewed}, m ={mg:.1f}")
        plt.show()
    # Upper bound:
    # Tc0 = 5 * t
    # Tc, Ts = calc_Tc(Nx, Ny, Deltag, param, skewed = skewed)
    # Tc, Ts,  Deltas, Delta_i, gamma, D = calc_Tc(Nx, Ny, Deltag, param,  skewed = skewed)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, skewed = skewed)
    Tcs[i] = Tc

print(f"took {time.time()- tic} seconds")
plt.title(f"U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}, mz={mz}")
plt.xlabel("m/t")
plt.ylabel("Tc")
plt.plot(mgs, Tcs)
plt.show()

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
# # print("desne?,", np.allclose(H, Hs.todense()))
# # num_ev = Nx*Ny
# # print(np.allclose(H, np.conjugate(H.T)))
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