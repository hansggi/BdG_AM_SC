from matplotlib import pyplot as plt
import numpy as np
import time
import cProfile
from numba import njit
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import eigs


sigmax = np.array([[0, 1],
                   [1, 0]], dtype = complex)

sigmay = np.array([[0, -1j],
                   [1j, 0]], dtype = complex)

sigmaz = np.array([[1, 0],
                  [0, -1]], dtype = complex)

@njit
def nb_block(X):
    xtmp1 = np.concatenate(X[0], axis=1)
    xtmp2 = np.concatenate(X[1], axis=1)
    return np.concatenate((xtmp1, xtmp2), axis=0)

@njit
def nb_block2(X):
    xtmp1 = np.hstack(X[0])
    xtmp2 = np.hstack(X[1])
    return np.vstack((xtmp1, xtmp2))

I2 = np.identity(2, dtype = complex)

@njit()
def make_H_periodic_2d_nambu(Nx, Ny,  Deltag, param, skewed = False):
    t, U, mu, mg = param
    # if skewed:
    #     return make_H_periodic_2d_nambu_skewed(Nx, Ny, Deltag, param)
    H = np.zeros((Nx*Ny, Nx*Ny, 4, 4), dtype="complex64")
    mz = 0
    bd = Nx//2  
    # bd = 0
    use_numba = True
    Deltag_arr = Deltag
    for ix in range(0, Nx):
            for iy in range(0, Ny):
                i = ix + Nx *iy 
                # Diagonal components in lattice space
                if ix < bd: # AM left part, SC right part
                    m = mg
                else:
                    m = 0
                if not use_numba:
                    H[i, i] = np.block([[mu*I2  + mz *sigmaz, Deltag_arr[iy, ix] * (-1j * sigmay)],
                                    [np.conjugate(Deltag_arr[iy, ix])*1j*sigmay, - mu*I2 - mz *sigmaz]]) # lower right component should be cc
            
                else:
                    H[i,i] = nb_block(((mu*I2  + mz *sigmaz, Deltag_arr[iy, ix] * (-1j * sigmay) )  ,
                                        (np.conjugate(Deltag_arr[iy, ix])*1j*sigmay, - mu*I2 - mz *sigmaz ) ) )


                if ix < Nx - 1:
                    # from the right
                    # if not ix == bd - 1:
                    if not use_numba:
                        H[i, i + 1] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                                [0 * I2, -t * I2 - m*sigmaz]])
                    else:   
                        H[i, i + 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                    (0 * I2, -t * I2 - m*sigmaz)))
                    # else:
                    #     if not use_numba:
                    #         H[i, i + 1] = np.block([[t * I2 , 0 * I2],
                    #                                 [0 * I2, -t * I2 ]])
                    #     else:
                    #         H[i, i + 1] = nb_block(((t * I2 , 0 * I2),
                    #                                 (0 * I2, -t * I2 )))
                if ix > 0:
                    # From the left
                    if ix ==bd: # Note mg below, this is the special boundary part, allows hopping into the SC
                        # print(ix, iy, m)
                        if not use_numba:
                            H[i, i - 1] = np.block([[t * I2 + mg*sigmaz , 0 * I2],
                                                    [0 * I2, -t * I2 - mg*sigmaz]])
                        else:
                            H[i, i - 1] = nb_block(((t * I2 + mg*sigmaz , 0 * I2),
                                                    (0 * I2, -t * I2 - mg*sigmaz)))
                    else: 
                        # print(ix, iy, m)
                        if not use_numba:
                            H[i, i - 1] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                                    [0 * I2, -t * I2 - m*sigmaz]])
                        else:
                            H[i, i - 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                   (0 * I2, -t * I2 - m*sigmaz)))

                # y hopping
                m = -m # Assume oposite here, this is the altermagnetism part
                if  iy < Ny - 1 :
                    # From down
                    if not use_numba:
                        H[i, i + Nx] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                                [0 * I2, -t * I2 - m*sigmaz]])
                    else:
                        H[i, i + Nx] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                             (0 * I2, -t * I2 - m*sigmaz)))

                if iy  > 0: # Except iy = 0
                    # From up
                    if not use_numba:
                        H[i, i - Nx] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                                [0 * I2, -t * I2 - m*sigmaz]])
                    else:
                        H[i, i - Nx] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                (0 * I2, -t * I2 - m*sigmaz)))
                    
    return H#, AM

def make_H_periodic_2d_nambu_skewed(Nx, Ny, Deltag, param):
    t, U,  mu, mg = param
    H = np.zeros((Nx*Ny, Nx*Ny, 4, 4), dtype=complex)
    mz = 0
    AM = np.zeros((Ny, Nx), dtype=bool)
    bd = Nx//2
    # bd = 0
    # if isinstance(Deltag, float) or type(Deltag) == "numpy.float64":
    #     Deltag_arr = np.zeros((Ny, Nx))
    #     Deltag_arr[:, bd:] = Deltag 
    # else:
    Deltag_arr = Deltag
    for ix in range(0, Nx):
            for iy in range(0, Ny):
                i = ix + Nx *iy 
                # Diagonal components in lattice space
                if ix < bd : # AM left part, SC right part
                    # Delta = 0
                    m = mg
                    AM[iy, ix] = True
                else:
                    # Delta = Deltag
                    m = 0

                H[i, i] = np.block([[mu*I2  + mz *sigmaz, Deltag_arr[iy, ix] * (-1j * sigmay)],
                                    [np.conjugate(Deltag_arr[iy, ix])*1j*sigmay, - mu*I2 - mz *sigmaz]]) # lower right component should be cc
                
                
                # x hopping, m = + mg
                if (ix < Nx - 1 or iy % 2 ==1) and iy > 0:
                    # to the right
                    H[i, i - Nx + 1] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                            [0 * I2, -t * I2 - m*sigmaz]])
                    
                if (ix > 0 or iy %2 ==1) and iy < Ny - 1:
                    # To the left
                    H[i, i + Nx ] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                                 [0 * I2, -t * I2 - m*sigmaz]])

                # y hopping
                m = -m # Assume oposite here, this is the altermagnetism part

                if iy > 0 and (ix > 0 or iy %2 ==1): 
                    # up (and to the left)
                    H[i, i - Nx ] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                                 [0 * I2, -t * I2 - m*sigmaz]])
                
                if  (ix < Nx -1 or iy %2 ==1) and iy < Ny - 1:
                    # Up
                    H[i, i + Nx] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                             [0 * I2, -t * I2 - m*sigmaz]])

                    
    return H, AM

def f(E, T):
    # Assume equilibrium
    return (1 - np.tanh( E / ( 2 * T ))) / 2

def Delta_sc(gamma, D, U, T):
    # NB: gamma and energies D must already be just positive eigenvalues
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]
    Delta_i = U*np.sum(u_dn *np.conjugate(v_up) * f(D, T) + u_up * np.conjugate(v_dn)* f(-D, T) , axis = 1) # Here, we used equilibrium explicitly to say (1-f(E)) = f(-E) 
    return Delta_i

@njit() # ish speedup of this function of 10-20 for Nx = Ny = 50, so I guess smaller speedup for smaller matrices
def unpack_block_matrix(H_block, H_toreturn, Nx, Ny):
    for i in range(Nx*Ny):
        for j in range(Nx*Ny):
            H_toreturn[4*i: 4*(i + 1), 4*j: 4* j + 4] = H_block[i, j]

            # Hs[4*i: 4*(i + 1), 4*j: 4* j + 4] = H4[i, j] 
    return H_toreturn

# Hz = np.zeros((4*50*50,  4 * 50*50))

# unpack_block_matrix(np.ones((50*50, 50*50, 4, 4)), Hz, 50, 50)
# tic = time.time()
# unpack_block_matrix(np.ones((50*50, 50*50, 4, 4)), Hz, 50, 50)
# print(time.time()- tic)
# time.sleep(100)
def does_Delta_increase(Nx, Ny, Deltag, tol, T, param, Delta_arr1,  skewed = False):
    # HEre, Deltag must be the guess, if Delta < Deltag, 
    t, U, mu, mg = param

    done = False
    Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_arr[:, :Nx//2] = 0

    it = 0
    H_toreturn = np.zeros((4*Nx*Ny, 4*Nx*Ny), dtype = complex)

    while not done:
        if it ==0 :
            Delta_arr = Delta_arr1
            Deltag = np.abs(Delta_arr[(3 * Nx) // 4 + Nx *(Ny//2)  ] )
            Deltag = np.sum(np.abs(Delta_arr))/ ( Nx * Ny / 2)

        else:
            H_block = make_H_periodic_2d_nambu(Nx, Ny, Delta_arr, param, skewed = skewed)
            H = unpack_block_matrix(H_block, H_toreturn, Nx, Ny)

            D, gamma = np.linalg.eigh(H)

            D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]

            Delta_arr = Delta_sc(gamma, D, U, T)

        Delta_arr = Delta_arr.reshape((Ny, Nx))

        Delta_arr[:, :Nx//2] = 0 # Set manually zero s.c. in the A.M?
        Delta_avr = np.sum(np.abs(Delta_arr))/ ( Nx * Ny / 2)
        # Delta_bulk = Delta_arr[Ny//2, (3 * Nx) // 4]

        it += 1
        
        if it > 4 and Delta_avr > Deltag :
            # done = True
            increase = True
            print("Avr: ",Delta_avr)

            return increase
        elif it > 4 and Delta_avr < Deltag:
            increase = False
            return increase



def calc_Delta_sc(Nx, Ny, Deltag, tol, T, param,  skewed = False):
    t, U, mu, mg = param

    # tol = 0.001 # Absolute. Think  better way to truncate, maybe Ali's article?
    done = False
    Delta_old = Deltag
    Delta_old_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_old_arr[:, :Nx//2] = 0

    Delta_old_avr = Deltag
    it = 0
    H_toreturn = np.zeros((4*Nx*Ny, 4*Nx*Ny), dtype = complex)

    while not done:
        H_block = make_H_periodic_2d_nambu(Nx, Ny, Delta_old_arr, param, skewed = skewed)
        H = unpack_block_matrix(H_block, H_toreturn, Nx, Ny)
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
        # plt.plot(D, f(D, T))
        # plt.show()
        D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]

       
        # print()
        # print(D[:num_eigvals_remaining])
        # time.sleep(20)
        # # print(" From exact ( just pos part): \n", D[:8])
        # ----------------------------------------------------------------------------------

        Delta_new_i = Delta_sc(gamma, D, U, T)
        Delta_new = Delta_new_i.reshape((Ny, Nx))
        # plt.imshow(np.abs(Delta_new))
        # plt.colorbar()
        # plt.show()
        Delta_new[:, :Nx//2] = 0 # Set manually zero s.c. in the A.M?
        # plt.imshow(np.abs(Delta_new))
        # plt.colorbar()
        # plt.show()
        # Delta_new = np.abs(np.sum(Delta_new_i))/ ( Nx * Ny / 2)
        Delta_avr = np.sum(np.abs(Delta_new_i))/ ( Nx * Ny / 2)
        it += 1
        if np.abs((np.abs(Delta_avr) - np.abs(Delta_old_avr)))  < tol * np.abs(Delta_old_avr):
            done = True
        # print(np.abs(Delta_avr) , np.abs(Delta_old_avr))
        Delta_old = Delta_new
        Delta_old_arr = Delta_new
        Delta_old_avr = np.sum( np.abs(Delta_old) ) / ( Nx * Ny /  2)

    return Delta_new_i, gamma, D

def calc_Tc(Nx, Ny, Deltag, param,  skewed = False):

    t, U, mu, mg = param
    Delta0 = Deltag
    nTs = 50
    Deltas = np.zeros((nTs))
    Ts = np.linspace(0.0001*t, 0.5*t, nTs)
    found = False
    tol = 0.02

    # The first calculation is the same for all temperatures --------------
    Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_arr[:, :Nx//2] = 0
    H_toreturn = np.zeros((4*Nx*Ny, 4*Nx*Ny), dtype = complex)

    H_block = make_H_periodic_2d_nambu(Nx, Ny, Delta_arr, param, skewed = skewed)
    H = unpack_block_matrix(H_block, H_toreturn, Nx, Ny)

    D, gamma = np.linalg.eigh(H)

    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
    # ---------------------------------------------------------------
    for i, T in enumerate(Ts):
        print(f"Checking for T = {T}")
        Delta_arr1 = Delta_sc(gamma, D, U, T)

        Ts[i] = T
        if not does_Delta_increase(Nx, Ny, Deltag, tol, T, param, Delta_arr1,  skewed=skewed):
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
    N = 10
    t, U, mu, mg = param
    Delta0 = Deltag
    # Deltas = np.zeros((nTs))
    # Ts = np.linspace(0.0001*t, 0.5*t, nTs)
    # found = False
    tol = 0.02

    # The first calculation is the same for all temperatures --------------
    Delta_arr = (np.ones((Nx*Ny), dtype = complex)*Deltag).reshape(Ny, Nx)
    Delta_arr[:, :Nx//2] = 0
    H_toreturn = np.zeros((4*Nx*Ny, 4*Nx*Ny), dtype = complex)
    H_block = make_H_periodic_2d_nambu(Nx, Ny, Delta_arr, param, skewed = skewed)
    H = unpack_block_matrix(H_block, H_toreturn, Nx, Ny)
    D, gamma = np.linalg.eigh(H)
    D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
    # ---------------------------------------------------------------

    Ts_lower = 0
    Ts_upper = Tc0

    for i in range(N):
        T = (Ts_upper - Ts_lower ) / 2
        print(T)
        Delta_arr1 = Delta_sc(gamma, D, U, T)

        if does_Delta_increase(Nx, Ny, Deltag, tol, T, param, Delta_arr1,  skewed=skewed):
            Ts_lower = T
            print("higher than this")
        else:
            Ts_upper = T #, Deltas, Delta_i, gamma, D
            print( "lower than this")
    return T

 
"""# Run one time
Nx = 20
Ny = 20
skewed = False
print("Running for skewed = ", skewed)
tic = time.time() 
Tc, Ts,  Deltas, Delta_i, gamma, D = calc_Tc(Nx, Ny, Deltag, param,  skewed = skewed)
print(f"Took {time.time()- tic} seconds to calculate Tc")

plt.xlabel("T/t")
plt.ylabel(r"$\Delta_{avr}$ / t")
plt.plot(Ts, Deltas)
plt.show()
"""
# Sweep over some parameter --------------------------------
Nx = 20
Ny = 10
skewed = False
t = 1.0 # Assume nn hopping term to be isotropic and constant for all lattice points. Set all quantities from this
U = 2.0 * t # Strength of attractive potential, causing s.c. Assumed to be constant here
Deltag = 0.00001 * t  # Assume constant Delta. First guess, will be updated self-consistently
mu =  -0.2*t # So far, constant chemical potential
num_sweep_vals = 5
tic = time.time()
mgs = np.linspace(0, 1*t, num_sweep_vals)
Tcs = np.zeros(num_sweep_vals)

for i, mg in enumerate(mgs):
    print("Running for mg = ", mg)
    param = (t, U, mu, mg)
    # Upper bound:
    # Tc0 = 5 * t
    Tc, Ts = calc_Tc(Nx, Ny, Deltag, param, skewed = skewed)
    # Tc, Ts,  Deltas, Delta_i, gamma, D = calc_Tc(Nx, Ny, Deltag, param,  skewed = skewed)
    # Tc = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, skewed = skewed)

    Tcs[i] = Tc

print(f"took {time.time()- tic} seconds")
# np.save( "Tc_mg_sweep", [Tcs, mgs])
plt.plot(mgs, Tcs)
plt.show()
# np.save()

# ----------------------------------------------------------
"""plt.imshow(np.abs(Delta_i.reshape(Ny, Nx)), origin = "upper")
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