from matplotlib import pyplot as plt
import numpy as np
import time
import cProfile
from numba import njit
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs

from funcs import make_sigmas, nb_block, nb_block2, make_H_numba, f, Delta_sc, unpack_block_matrix, make_H_numba_boundaryjumping

from multiprocessing import Pool
# print("Number of cpuf : ", multiprocessing.cpu_count())

def does_Delta_increase(Nx, Ny, Deltag, T, param, Delta_arr1,  skewed = False):
    # Here, Deltag must be the guess, if Delta < Deltag, 
    t, U, mu, mg, mz = param
    num_it = 5

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
            H = make_H_numba_boundaryjumping(Nx, Ny, Delta_arr, param, bd,  skewed = skewed)
            # print(np.allclose(H, np.conjugate(H).T))
            D, gamma = np.linalg.eigh(H)

            D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]

            Delta_arr = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)

        
        Delta_bulk = np.abs(Delta_arr)[Ny//2, - Nx // 4]
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
        H = make_H_numba_boundaryjumping(Nx, Ny, Delta_old_arr, param, bd, skewed = skewed)

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
        Delta_bulk = np.abs(Delta_new)[Ny//2, - Nx//4]
        it += 1
        if np.abs(np.abs(Delta_bulk) - np.abs(Delta_old_bulk))  <= tol :
            done = True

        Delta_old = Delta_new
        Delta_old_arr = Delta_new
        Delta_old_bulk = np.abs(Delta_old[Ny//2, - Nx//4])
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

    H = make_H_numba_boundaryjumping(Nx, Ny, Delta_arr, param, bd, skewed = skewed)

    D, gamma = np.linalg.eigh(H)
    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
    # --------------------------------------------------------------------
    for i, T in enumerate(Ts):
        print(f"Checking for T = {T}")
        Delta_arr1 = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)

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

def calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd,  skewed ):
    N = 12 # Look at, maybe not needed this accuracy
    t, U, mu, mg, mz = param

    # The first calculation is the same for all temperatures --------------
    Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_arr[:, :bd] = 0
    H = make_H_numba_boundaryjumping(Nx, Ny, Delta_arr, param, bd, skewed = skewed)

    D, gamma = np.linalg.eigh(H)
    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
    # ---------------------------------------------------------------

    Ts_lower = 0
    Ts_upper = Tc0

    for i in range(N):
        T = (Ts_upper + Ts_lower ) / 2
        Delta_arr1 = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)

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

Nx = 12
Ny = 12
skewed = True
t = 1.0 # Assume nn hopping term to be isotropic and constant for all lattice points. Set all quantities from this
U = 2 * t # Strength of attractive potential, causing s.c. Assumed to be constant here
Deltag =  1e-5 # Assume constant Delta. First guess, will be updated self-consistently
mu =  - 0.5*t # So far, constant chemical potential
# mz = 0. # Ferromagetism in the AM
# mg = 0
num_sweep_vals = 20
tic = time.time()
# mg = 0
mz = 0
bd = Nx//2
mgs = np.linspace(0., 1.5, num_sweep_vals)
# mzs = np.linspace(0, 0.5, num_sweep_vals)
Tcs = np.zeros(num_sweep_vals)
Tc0 = 0.3


tic = time.time()
fnum = 0
fig, ax = plt.subplots(nrows = 3, ncols= 1)

for i, mg in enumerate(mgs):
    print("Running for mg = ", mg)
    param = (t, U, mu, mg, mz)
    if i == 0 or i == num_sweep_vals -1 or i == num_sweep_vals//2:
        # Basically zero termperature, and low tolerance
        Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, 0.000001, 0.01, param, bd,  skewed=skewed) #tol, T
        Delta_i = Delta_i.reshape((Ny, Nx))
        print(np.amax(np.abs(Delta_i)))
        im = ax[fnum].imshow(np.abs(Delta_i), aspect="auto")
        ax[fnum].set_title((f"mg = {mg:.1f}"))
        fig.colorbar(im, ax=ax[fnum])
        fnum += 1

    Tc = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd,  skewed = skewed)
    Tcs[i] = Tc

# fig.colorbar(im, ax=ax.ravel().tolist())
fig.suptitle(f"U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}, mz ={mz:.1f}")
plt.show()

print(f"took {time.time()- tic} seconds")
plt.title(f"U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}, mg={mg:.1f}")
plt.xlabel("m/t")
plt.ylabel("Tc")
plt.plot(mgs, Tcs)
plt.show()


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
# print(ldos.shape)
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