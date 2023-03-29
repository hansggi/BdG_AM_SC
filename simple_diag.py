from matplotlib import pyplot as plt
import numpy as np
import time
# from scipy import sparse
# from scipy.sparse.linalg import eigsh
# from scipy.sparse.linalg import eigs

# parameters
t = 1.0 # Assume nn hopping term to be isotropic and constant for all lattice points. Set all quantities from this
U = 1.9 * t # Strength of attractive potential, causing s.c. Assumed to be constant here

Deltag = 0.001 * t  # Assume constant Delta. First guess, will be updated self-consistently

mu = - 0.5*t # So far, constant chemical potential
mg = 0.8* t

sigmax = np.array([[0, 1],
                   [1, 0]], dtype = complex)
sigmay = np.array([[0, -1j],
                   [1j, 0]], dtype = complex)
sigmaz = np.array([[1, 0],
                  [0, -1]], dtype = complex)

I2 = np.identity(2, dtype = complex)

# def make_H(N):
#     H = np.zeros((N, N, 2, 2, 2, 2), dtype = complex)
#     for i in range(N):
#         for j in range(N):
#             H[i, j] = np.array([[(mu + t)*I2                  , - Delta *1j * sigmay],
#                                 [np.conjugate(Delta)*1j*sigmay, -(np.conjugate(mu) + t)*I2 ]])  
#     return H

# def make_H_simple(N):
#     H = np.zeros((N, N))
#     for i in range(0, N):
#             H[i,i] = -mu
#             if not i == 0:
#                 H[i, i-1] = -t
#             if not i == N-1:
#                 H[i, i+1] = - t

#     return H

# def make_H_simple_periodic(N):
#     H = np.zeros((2*N, 2*N))
#     m = mg
#     for i in range(0, N):
#             H[2*i,2*i] = -mu - m
#             H[2*i+1, 2*i+1] = - mu + m
#             H[2*i, (2*i-1)%(2*N)] = -t
#             H[2*i, (2*i+1)%(2*N)] = - t
#             H[2*i + 1, (2*i-1 + 1)%(2*N)] = -t
#             H[2*i + 1, (2*i+1 + 1)%(2*N)] = - t

#     return H

# def make_H_periodic_2d(N):

#     H = np.zeros((2*N**2, 2*N**2))
#     m = mg
#     for ix in range(0, N):
#             for iy in range(0, N):
#                 i =  ix  + N*iy 
#                 H[2*i,2*i] = -mu - m
#                 H[2*i+1, 2*i+1] = - mu + m

#                 # x- hopp2*ing
#                 H[2*i, (2*i-1)%(2*N**2)] = -t
#                 H[2*i, (2*i+1)%(2*N**2)] = - t

#                 H[2*i + 1, (2*i-1 + 1)%(2*N**2)] = -t
#                 H[2*i + 1, (2*i+1 + 1)%(2*N**2)] = - t
#                 #y-hopp2*ing
#                 H[2*i, (2*i-N)%(2*N**2)] = -t
#                 H[2*i, (2*i+N)%(2*N**2)] = - t

#                 H[2*i + 1, (2*i-N + 1)%(2*N**2)] = -t
#                 H[2*i + 1, (2*i+N + 1)%(2*N**2)] = - t

#     return H

def make_H_periodic_2d_nambu(Nx, Ny,  Deltag, skewed = False):
    if skewed:
        return make_H_periodic_2d_nambu_skewed(Nx, Ny, Deltag)
    H = np.zeros((Nx*Ny, Nx*Ny, 4, 4), dtype=complex)
    mz = 0
    AM = np.zeros((Ny, Nx), dtype=bool)
    bd = Nx//2
    # bd = 0
    if isinstance(Deltag, float) or type(Deltag) == "numpy.float64":
        Deltag_arr = np.zeros((Ny, Nx))
        Deltag_arr[:, bd:] = Deltag 
    else:
        Deltag_arr = Deltag
        # print(Deltag_arr)
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
                # print(AM[ix, iy], ix, iy,  Delta, m)
                H[i, i] = np.block([[mu*I2  + mz *sigmaz, Deltag_arr[iy, ix] * (-1j * sigmay)],
                                    [np.conjugate(Deltag_arr[iy, ix])*1j*sigmay, - mu*I2 - mz *sigmaz]]) # lower right component should be cc
                

                #  hopping
                # if ix < Nx/2: # AM left part, SC right part
                #     m = mg
                # else:
                #     m = 0
                # x hopping, m = + mg
                if ix < Nx - 1 :
                    # to the right
                    H[i, i + 1] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                            [0 * I2, -t * I2 - m*sigmaz]])
                    
                if ix > 0:
                    # To the left
                    H[i, i - 1] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                            [0 * I2, -t * I2 - m*sigmaz]])

                # y hopping
                m = -m # Assume oposite here, this is the altermagnetism part
                if  iy < Ny - 1 :
                    #down

                    H[i, i + Nx] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                             [0 * I2, -t * I2 - m*sigmaz]])

                if iy  > 0: # Except iy = 0
                    # up
                    H[i, i - Nx] = np.block([[t * I2 + m*sigmaz , 0 * I2],
                                            [0 * I2, -t * I2 - m*sigmaz]])
                    
    return H, AM

def make_H_periodic_2d_nambu_skewed(Nx, Ny,  Deltag):
    H = np.zeros((Nx*Ny, Nx*Ny, 4, 4), dtype=complex)
    mz = 0
    AM = np.zeros((Ny, Nx), dtype=bool)
    bd = Nx//2
    # bd = 0
    if isinstance(Deltag, float) or type(Deltag) == "numpy.float64":
        Deltag_arr = np.zeros((Ny, Nx))
        Deltag_arr[:, bd:] = Deltag 
    else:
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
    return (1 - np.tanh(( E) / (2*T)))/2

def Delta_sc(gamma, D, U, T):
    # NB: gamma and energies D must already be just positive eigenvalues
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]
    Delta_i = U*np.sum(u_dn *np.conjugate(v_up) * f(D, T) + u_up * np.conjugate(v_dn)* f(-D, T) , axis = 1) # Here, we used equilibrium explicitly to say (1-f(E)) = f(-E) 
    return Delta_i

def unpack_block_matrix(H_block, Nx, Ny):
    H = np.zeros((4*Nx*Ny, 4*Nx*Ny), dtype = complex)

    for i in range(Nx*Ny):
        for j in range(Nx*Ny):
            H[4*i: 4*(i + 1), 4*j: 4* j + 4] = H_block[i, j]
            # Hs[4*i: 4*(i + 1), 4*j: 4* j + 4] = H4[i, j] 
    return H

def calc_Delta_sc(Nx, Ny, Deltag, tol, T, skewed = False):
    tol = 0.0001 # Relative
    done = False
    Delta_old = Deltag
    Delta_old_avr = Deltag
    it = 0
    while not done:
        H_block, AM = make_H_periodic_2d_nambu(Nx, Ny, Delta_old, skewed = skewed)
        H = unpack_block_matrix(H_block, Nx, Ny)
        D, gamma = np.linalg.eigh(H)
        D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
        Delta_new_i = Delta_sc(gamma, D, U, T)
        Delta_new = Delta_new_i.reshape((Ny, Nx))
        # Delta_new = np.abs(np.sum(Delta_new_i))/ ( Nx * Ny / 2)
        Delta_avr = np.abs(np.sum(Delta_new_i))/ ( Nx * Ny)

        it += 1
        if np.abs((np.abs(Delta_avr) - np.abs(Delta_old_avr)))  < tol:
            done = True
        Delta_old = Delta_new
        Delta_old_avr = np.abs(np.sum(Delta_old))/ ( Nx * Ny )

    return Delta_new_i

def calc_Tc(Nx, Ny, Deltag, skewed = False):
    nTs = 50
    Deltas = np.zeros((nTs))
    Ts = np.linspace(0.0001*t, 0.5*t, nTs)
    found = False
    for i, T in enumerate(Ts):
        print(f"Checking for T = {T}")
        Delta_i = calc_Delta_sc(Nx, Ny, Deltag, 0.001, T, skewed=skewed)
        Deltas[i] = np.average(np.abs(Delta_i))
        if np.average(np.abs(Delta_i))  < Deltag and not found:
            found = True
            Tc = T
            print(f" No sc at T = {T}")
        
    
            return Tc, Ts, Deltas, Delta_i

    print(" Too low range, no cutoff found")
        

Nx = 15
Ny = 20
skewed = True
print("Eunning for skewed = ", skewed)
T,Ts,  Deltas, Delta_i = calc_Tc(Nx, Ny, Deltag, skewed = skewed)
plt.xlabel("T/t")
plt.ylabel(r"$\Delta_{avr}$ / t")
plt.plot(Ts, Deltas)
# plt.savefig("only_sc_singlet")
# plt.savefig("straight_singlet")
# plt.savefig("skewed_singlet")

plt.show()

plt.imshow(np.abs(Delta_i.reshape(Ny, Nx)), origin = "upper")
plt.colorbar()
plt.show()
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