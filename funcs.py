import numpy as np
from numba import njit

def make_sigmas():
    I2 = np.identity(2, dtype = "complex128")

    sigmax = np.array([[0, 1],
                    [1, 0]], dtype = "complex128")

    sigmay = np.array([[0, -1j],
                    [1j, 0]], dtype = "complex128")

    sigmaz = np.array([[1, 0],
                    [0, -1]], dtype = "complex128")
    return I2, sigmax, sigmay, sigmaz

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


I2, sigmax, sigmay, sigmaz = make_sigmas()

@njit()
def make_H_numba(Nx, Ny, Delta_arr, param, bd, skewed):
    t, U, mu, mg, mzg = param
    H = np.zeros((Nx*Ny, Nx*Ny, 4, 4), dtype="complex128")
    for ix in range(0, Nx):
            for iy in range(0, Ny):
                i = ix + Nx *iy 
                # Diagonal components in lattice space
                if ix < bd: # AM left part, SC right part
                    m = mg
                    mz = mzg
                elif bd ==0: # When we set bd = 0, assume we want ferromagnetism/ altermagnetism in the SC. Else we can set mg = 0 etc
                    m = mg
                    mz = mzg
                else:
                    m = 0
                    mz = 0

                H[i,i] = nb_block(((mu*I2  + mz *sigmaz, Delta_arr[iy, ix] * (-1j * sigmay) )  ,
                                    (np.conjugate(Delta_arr[iy, ix])*1j*sigmay, - mu*I2 - mz *sigmaz ) ) ) # With ferromagnetism


                # H[i,i] = nb_block(((mu*I2 , Delta_arr[iy, ix] * (-1j * sigmay) )  ,
                #                     (np.conjugate(Delta_arr[iy, ix])*1j*sigmay, -mu*I2) ) )
                
                if not skewed:
    # x-hopping, m = mg --------------------------------------------------------------------
                    # Contribution from the right, only if not at the right edge, and special case at boundary
                    if ix < Nx - 1:
                        if ix == bd -1:
                            H[i, i + 1] = nb_block(((t * I2,  0 * I2),
                                                    (0 * I2, -t * I2)))
                        else:                    
                            H[i, i + 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                    (0 * I2            , -t * I2 - m*sigmaz)))
                        
                    # Contrbution from the left, only if not on the left edge
                    if ix > 0:
                        H[i, i - 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                (0 * I2, -t * I2 - m*sigmaz)))
                    # y hopping
                    m = -m  # Assume oposite here, this is the altermagnetism part

                    # If below
                    # IF not periodic, include the if tests below
                    if  iy < Ny - 1 :
                    #     # From down
                        H[i, (i + Nx) % (Nx*Ny)] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                             (0 * I2, -t * I2 - m*sigmaz)))

                    if iy  > 0: # Except iy = 0
                        # From up
                        H[i, i - Nx] = nb_block(((t * I2 + m*sigmaz , 0 * I2            ),
                                                 (0 * I2            , -t * I2 - m*sigmaz)))
# Skewed ---------------------------------------------------------------------------
                elif skewed: 
                    # print("THIS IS NOT VERIFIED!")
                    # Here, have used a system where the first row is further to the left than the second row etc,
                    # also, the boundary splits each row in two
                    # x-hopping--------------------------

                    # On even site in y- direction ----------------------------------------------------------
                    if iy % 2 == 0:

                        # -- From the right --
                        if ix < Nx - 1 and iy > 0:
                            if ix == bd -1 : # Just hopping, not atermagnetism
                                H[i, i - Nx + 1 ] =  nb_block(((t * I2,  0 * I2),
                                                            (0 * I2, -t * I2)))
                            else:
                                H[i, i - Nx + 1 ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                            (0 * I2, -t * I2 - m*sigmaz)))
                            
                        # -- From the left --
                        if  iy < Ny - 1:
                            H[i, i + Nx ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                    (0 * I2, -t * I2 - m*sigmaz)))

                        # -- y hopping --
                        m = -m # Assume oposite here, this is the altermagnetism part

                        # -- From above --
                        if iy > 0: 
                            H[i, i - Nx  ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                        (0 * I2, -t * I2 - m*sigmaz)))
                            
                        # -- From below --
                        if  ix < Nx - 1 and iy < Ny - 1:
                            if ix == bd -1 and iy % 2 == 0:
                                H[i, i + Nx + 1] = nb_block(((t * I2  , 0 * I2),
                                                            (0 * I2, -t * I2 )))
                            else:
                                H[i, i + Nx + 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                    (0 * I2, -t * I2 - m*sigmaz)))
                    # ---------------------------------------------------------------------------
                    # On odd site in y-direction ------------------------------------------------
                    else:

                        # From the right
                        if  iy > 0:
                            
                            H[i, i - Nx ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                            (0 * I2, -t * I2 - m*sigmaz)))
                            
                        # -- From the left --
                        if ix > 0  and iy < Ny - 1:
                            H[i, i + Nx  - 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                    (0 * I2, -t * I2 - m*sigmaz)))

                        # -- y hopping --
                        m = -m # Assume oposite here, this is the altermagnetism part

                        # -- From above --
                        if iy > 0 and ix > 0 : 
                            H[i, i - Nx - 1 ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                        (0 * I2, -t * I2 - m*sigmaz)))
                        
                        # -- From below --
                        if  iy < Ny - 1:
                            H[i, i + Nx] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                    (0 * I2, -t * I2 - m*sigmaz)))
                # # -------------------------------------------------------------------------------
    # Hr = unpack_block_matrix(H, Nx, Ny)
    # print(np.allclose(Hr, np.transpose(np.conjugate(Hr))))
    return unpack_block_matrix(H, Nx, Ny)


@njit()
def make_H_numba_boundaryjumping(Nx, Ny, Delta_arr, param, bd, skewed):
    t, U, mu, mg, mzg = param
    H = np.zeros((Nx*Ny, Nx*Ny, 4, 4), dtype="complex128")
    for ix in range(0, Nx):
            for iy in range(0, Ny):
                i = ix + Nx *iy 
                # Diagonal components in lattice space
                if ix < bd: # AM left part, SC right part
                    m = mg
                    mz = mzg
                elif bd ==0: # When we set bd = 0, assume we want ferromagnetism/ altermagnetism in the SC. Else we can set mg = 0 etc
                    m = mg
                    mz = mzg
                else:
                    m = 0
                    mz = 0

                H[i,i] = nb_block(((mu*I2  + mz *sigmaz, Delta_arr[iy, ix] * (-1j * sigmay) )  ,
                                    (np.conjugate(Delta_arr[iy, ix])*1j*sigmay, - mu*I2 - mz *sigmaz ) ) ) # With ferromagnetism


                # H[i,i] = nb_block(((mu*I2 , Delta_arr[iy, ix] * (-1j * sigmay) )  ,
                #                     (np.conjugate(Delta_arr[iy, ix])*1j*sigmay, -mu*I2) ) )
                
                if not skewed:
    # x-hopping, m = mg --------------------------------------------------------------------
                    # Contribution from the right, only if not at the right edge, and special case at boundary
                    if ix < Nx - 1:
                   
                        H[i, i + 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                (0 * I2            , -t * I2 - m*sigmaz)))
                        
                    # Contrbution from the left, only if not on the left edge
                    if ix > 0:
                        if ix == bd:
                            H[i, i - 1] = nb_block(((t * I2 + mg*sigmaz , 0 * I2),
                                                (0 * I2, -t * I2 - mg*sigmaz)))
                        else:
                            H[i, i - 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                (0 * I2, -t * I2 - m*sigmaz)))
                    # y hopping
                    m = -m  # Assume oposite here, this is the altermagnetism part

                    # If below
                    # IF not periodic, include the if tests below
                    if  iy < Ny - 1 :
                    #     # From down
                        H[i, (i + Nx) % (Nx*Ny)] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                             (0 * I2, -t * I2 - m*sigmaz)))

                    if iy  > 0: # Except iy = 0
                        # From up
                        H[i, i - Nx] = nb_block(((t * I2 + m*sigmaz , 0 * I2            ),
                                                 (0 * I2            , -t * I2 - m*sigmaz)))
# Skewed ---------------------------------------------------------------------------
                elif skewed: 
                    # print("THIS IS NOT VERIFIED!")
                    # Here, have used a system where the first row is further to the left than the second row etc,
                    # also, the boundary splits each row in two
                    # x-hopping--------------------------

                    # On even site in y- direction ----------------------------------------------------------
                    if iy % 2 == 0:

                        # -- From the right --
                        if ix < Nx - 1 and iy > 0:
                            H[i, i - Nx + 1 ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                            (0 * I2, -t * I2 - m*sigmaz)))
                            
                        # -- From the left --
                        if  iy < Ny - 1:
                            if ix == bd:
                                H[i, i + Nx ] = nb_block(((t * I2 + mg*sigmaz , 0 * I2),
                                                    (0 * I2, -t * I2 - mg*sigmaz)))
                            else:
                                H[i, i + Nx ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                        (0 * I2, -t * I2 - m*sigmaz)))

                        # -- y hopping --
                        m = -m # Assume oposite here, this is the altermagnetism part

                        # -- From above --
                        if iy > 0: 
                            if ix == bd:
                                H[i, i - Nx  ] = nb_block(((t * I2 + mg*sigmaz , 0 * I2),
                                                        (0 * I2, -t * I2 - mg*sigmaz)))
                            else:
                                H[i, i - Nx  ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                        (0 * I2, -t * I2 - m*sigmaz)))
                            
                        # -- From below --
                        if  ix < Nx - 1 and iy < Ny - 1:
                                H[i, i + Nx + 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                    (0 * I2, -t * I2 - m*sigmaz)))
                    # ---------------------------------------------------------------------------
                    # On odd site in y-direction ------------------------------------------------
                    else:

                        # From the right
                        if  iy > 0:
                            
                            H[i, i - Nx ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                            (0 * I2, -t * I2 - m*sigmaz)))
                            
                        # -- From the left --
                        if ix > 0  and iy < Ny - 1:
                            H[i, i + Nx  - 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                    (0 * I2, -t * I2 - m*sigmaz)))

                        # -- y hopping --
                        m = -m # Assume oposite here, this is the altermagnetism part

                        # -- From above --
                        if iy > 0 and ix > 0 : 
                            H[i, i - Nx - 1 ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                        (0 * I2, -t * I2 - m*sigmaz)))
                        
                        # -- From below --
                        if  iy < Ny - 1:
                            H[i, i + Nx] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                    (0 * I2, -t * I2 - m*sigmaz)))
                # # -------------------------------------------------------------------------------
    # Hr = unpack_block_matrix(H, Nx, Ny)
    # print(np.allclose(Hr, np.conjugate(Hr).T))
    return unpack_block_matrix(H, Nx, Ny)


def f(E, T):
    # Assume equilibrium
    return (1 - np.tanh( E / ( 2 * T ))) / 2

def Delta_sc(gamma, D, U, T, Nx, Ny, bd):
    # NB: gamma and energies D must already be just positive eigenvalues
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]
    # Make Ui
    Ui = np.ones(Nx*Ny, dtype = complex).reshape(Ny, Nx)*U
    Ui[:, :bd] = 0
    Ui = Ui.reshape((Nx*Ny))   
    # -------------------------------------------------------------------
    Delta_i = Ui*np.sum(u_dn *np.conjugate(v_up) * f(D, T) + u_up * np.conjugate(v_dn)* f(-D, T) , axis = 1) # Here, we used equilibrium explicitly to say (1-f(E)) = f(-E) 
    return Delta_i

@njit() # ish speedup of this function of 10-20 for Nx = Ny = 50, so I guess smaller speedup for smaller matrices
def unpack_block_matrix(H_block, Nx, Ny):
    H_toreturn = np.zeros((4*Nx*Ny, 4*Nx*Ny), dtype = "complex64")
    for i in range(Nx*Ny):
        for j in range(Nx*Ny):
            H_toreturn[4*i: 4*(i + 1), 4*j: 4* j + 4] = H_block[i, j]

            # Hs[4*i: 4*(i + 1), 4*j: 4* j + 4] = H4[i, j] 
    return H_toreturn