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


@njit
def make_H_numba(Nx, Ny, m_arr, mz_arr, Delta_arr, param, bd, skewed):
    t, U, mu, mg, mzg = param
    H = np.zeros((Nx*Ny, Nx*Ny, 4, 4), dtype="complex128")
    # V = 2
    # if F ==0:
    #     F = np.zeros((Nx*Ny, Nx*Ny, 4, 4), dtype="complex128")
    #     F0 = np.zeros((Nx*Ny, Nx*Ny))
    #     # for ix in range(Nx):
    #     #     for iy in range(Ny):
    #     #         i = ix + Nx *iy 

    #     #         # x-interaction
                
    #     #         if ix < Nx - 1 :
    #     #             ur = np.array( [[0,                  - 2 * V * F0[i, i+1]],
    #     #                             [2* V * F0[i, i +1], 0                    ]])
                    
    #     #             F[i, i + 1] = nb_block(((0. * I2,             ur),
    #     #                                     (- np.conjugate(ur), 0.*I2)))
    #     #         # if ix > 0:
    #     #         #     ur = np.array( [[0, - 2 * V * F[i, i - 1]],
    #     #         #                     [2* V * F[i - 1, i], 0]])
    #     #         #     F[i, i + 1] = nb_block(((0 * I2,             ur),
    #     #         #                             (- np.conjugate(ur), 0*I2))) 
    #     #         # # if iy < Ny - 1 :
    #     #         # ur = np.array( [[0, - 2 * V * F[i, i + Nx]],
    #     #         #                 [2* V * F[i + Nx, i], 0]])
    #     #         # F[i, i + 1] = nb_block(((0 * I2,             ur),
    #     #         #                         (- np.conjugate(ur), 0*I2)))    
    #     #         # # if iy > 0 :
    #     #         # ur = np.array( [[0, - 2 * V * F[i, i - Nx]],
    #     #         #                 [2* V * F[i -Nx, i], 0]])
    #     #         # F[i, i + 1] = nb_block(((0 * I2,             ur),
    #     #         #                         (- np.conjugate(ur), 0*I2)))   

    m = mg
    for ix in range(0, Nx):
        for iy in range(0, Ny):
            i = ix + Nx *iy

            # if ix <= bd: # AM left part, SC right part
            #     m = mg
            #     mz = mzg
            # elif bd == 0: # When we set bd = 0, assume we want ferromagnetism/ altermagnetism in the SC. Else we can set mg = 0 etc
            #     m = mg
            #     mz = mzg
            # else:
            #     m = 0
            #     mz = 0
            """if ix == bd:
                mz = 0.
                zetaL = -1.
            elif ix < bd:
                mz = mzg
                zetaL = 1.
            else: 
                mz = 0.
                zetaL = -1.

            if ix == bd - 1:
                zetaR = -1.
            elif ix < bd - 1:
                zetaR = 1.
            else: 
                zetaR = -1."""
            
            # Equiv:
            if ix < bd[0] or ix >= bd[1]:
                mz = mzg
                if ix < bd[0] or ix > bd[1]: 
                    zetaL = 1.
                else:
                    zetaL = -1
            else:
                mz = 0.
                zetaL = -1.
            
            if ix < bd[0] - 1 or ix >= bd[1]:
                zetaR = 1.
            else:
                zetaR = -1.
            # print(ix, iy, m, mz, Delta_arr[iy, ix])
            # Diagonal term includes mu, Delta (s-wave) and ferromagnetism
            # print(ix, iy, m, mz)
            H[i,i] = nb_block(((mu*I2  + mz_arr[iy, ix] *sigmaz, Delta_arr[iy, ix] * (-1j * sigmay) )  ,
                               (np.conjugate(Delta_arr[iy, ix])*1j*sigmay, - mu*I2 - mz_arr[iy, ix] *sigmaz ) ) ) # With ferromagnetism
    
            if not skewed:
                # x-hopping, m = mg --------------------------------------------------------------------
                # Contribution from the right, only if not at the right edge, and special case at boundary
                if ix < Nx - 1:
                    m = min(m_arr[iy, ix], m_arr[iy, ix+1])
                    H[i, i + 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                            (0 * I2                           , -t * I2 - m*sigmaz)))
                # H[i, (i + 1)%(Nx*Ny)] += F[i, i+ 1]

                # Contrbution from the left, only if not on the left edge
                if ix > 0:
                    m = min(m_arr[iy, ix], m_arr[iy, ix-1])

                    H[i, i - 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                            (0 * I2, -t * I2 - m*sigmaz)))
                    # H[i, i - 1] += F[i, i-1]

                # y hopping, hav3 set m -> -m
                # if ix <  bd:

                #     m = -m  # Assume oposite here, this is the altermagnetism part
                # else: 
                #     m = 0

                # IF not periodic, include the if tests below
                if not Ny == 1:
                    # if  iy < Ny - 1 :
                    # From down
                    assert m_arr[iy, ix] == m_arr[(iy + 1)%Ny, ix]
                    assert m_arr[iy, ix] == m_arr[iy - 1, ix]

                    m = m_arr[iy, ix]

                    H[i, (i + Nx) % (Nx*Ny)] = nb_block(((t * I2 - m*sigmaz , 0 * I2),
                                                         (0 * I2, -t * I2 + m*sigmaz)))
                    # H[i, (i + Nx) % (Nx*Ny)] += F[i, (i + Nx) % (Nx*Ny) ]
                    # if iy  > 0: # Except iy = 0
                        # From up
                    H[i, i - Nx] = nb_block(((t * I2 - m*sigmaz , 0 * I2            ),
                                             (0 * I2            , -t * I2 + m*sigmaz)))
                    
                    # H[i, i - Nx] += F[i, i - Nx]
# Skewed ---------------------------------------------------------------------------
            elif skewed: 
                # m = mg
                # print("THIS IS NOT VERIFIED!")
                if ix < bd[0]:
                    m = mg
                    mz = mzg
                else:
                    m = 0
                    mz = 0
                # Here, have used a system where the first row is further to the left than the second row etc,
                # also, the boundary splits each row in two. 
                # x-hopping------------------------ --
                assert Ny % 2 ==0
                # On even site in y- direction ----------------------------------------------------------
                if iy % 2 == 0:

                    # -- From the right --
                    # if ix == bd[0] -1 : # Just hopping, not atermagnetism, depends on both sites
                    # H[i, (i - Nx ) % (Nx*Ny) ] =nb_block(((t * I2,  0 * I2),
                    #                                     (0 * I2, -t * I2)))
                    # else:
                    H[i, (i - Nx)% (Nx*Ny)  ] += nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                          (0 * I2, -t * I2 - m*sigmaz)))
                        
                    # -- From the left --
                    if ix > 0:
                        #if ix == bd[0]: m already zero
                        H[i, (i + Nx - 1)% (Nx*Ny) ] += nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                                 (0 * I2, -t * I2 - m*sigmaz)))

                    # -- y hopping --
                    m = -m # Assume oposite here, this is the altermagnetism part

                    # -- From above --
                    if ix > 0:
                        H[i,( i - Nx -1)% (Nx*Ny) ] += nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                                (0 * I2, -t * I2 - m*sigmaz)))
                            
                    # -- From below --
                    # if ix == bd[0] -1:
                    #     H[i, (i + Nx ) % (Nx*Ny)] = nb_block(((t * I2  , 0 * I2),
                    #                                           (0 * I2, -t * I2 )))
                    # else:
                    H[i, (i + Nx ) % (Nx*Ny)] += nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                        (0 * I2, -t * I2 - m*sigmaz)))
                # ---------------------------------------------------------------------------
                # On odd site in y-direction ------------------------------------------------
                elif iy % 2 == 1 :

                    # From the right
                    if  ix  < Nx - 1:
                        if ix == bd[0] -1:
                            H[i, (i - Nx +1) % (Nx*Ny)] += nb_block(((t * I2 , 0 * I2),
                                                        (0 * I2, -t * I2 )))
                        else:
                            H[i, (i - Nx +1)% (Nx*Ny) ] += nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                         (0 * I2, -t * I2 - m*sigmaz)))
                        
                    # -- From the left --
                    H[i, (i + Nx) % (Nx*Ny)  ] += nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                (0 * I2, -t * I2 - m*sigmaz)))

                    # -- y hopping --
                    m = -m # Assume oposite here, this is the altermagnetism part

                    # -- From above --
                    H[i,( i - Nx % (Nx*Ny)) ] += nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                (0 * I2, -t * I2 - m*sigmaz)))
                
                    # -- From below --
                    if  ix < Nx - 1:
                        if ix == bd[0] -1:
                            H[i, (i + Nx + 1)% (Nx*Ny)] += nb_block(((t * I2  , 0 * I2),
                                                         (0 * I2, -t * I2 )))
                        else:
                            H[i, (i + Nx + 1)% (Nx*Ny)] += nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                         (0 * I2, -t * I2 - m*sigmaz)))
                # # -------------------------------------------------------------------------------
    
    Hr = unpack_block_matrix(H, Nx, Ny)

    # assert np.allclose(Hr, np.transpose(np.conjugate(Hr)))
    # print(np.allclose(Hr, np.transpose(np.conjugate(Hr))))
    return Hr


"""@njit()
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

"""
def fd(E, T):
    # Assume equilibrium
    return (1 - np.tanh( E / ( 2 * T ))) / 2

@njit
def Delta_sc(gamma, D, U, T, Nx, Ny, bd):
    # NB: gamma and energies D must already be just positive eigenvalues
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]
    # Make Ui
    # Ui = np.ones(Nx*Ny, dtype = complex).reshape(Ny, Nx)*U
    U = U * (1. + 0j)
    Ui = np.ones(Nx*Ny).reshape(Ny, Nx)*U

    Ui[:, :bd[0]] = 0
    if bd[1] < Nx:
        Ui[:, bd[1]:] = 0
    Ui = Ui.reshape((Nx*Ny))   
    # print(np.allclose(np.sum(u_up* np.conjugate(v_dn), axis = 1), -np.sum(u_dn* np.conjugate(v_up), axis = 1)))
    # print(np.sum(u_up* np.conjugate(v_dn), axis = 1), -np.sum(u_dn* np.conjugate(v_up), axis = 1))
    f  = (1 - np.tanh(  D / ( 2 * T ))) / 2
    # fm = (1 - np.tanh( -D / ( 2 * T ))) / 2
    # -------------------------------------------------------------------
    Delta_i = -Ui*np.sum(u_dn *np.conjugate(v_up) * f +       u_up * np.conjugate(v_dn)* (1 - f) , axis = 1) # Here, we used equilibrium explicitly to say (1-f(E)) = f(-E) 
    return Delta_i


def F_sc(gamma, D, V, T, Nx, Ny, bd):
    # NB: gamma and energies D must already be just positive eigenvalues
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]
    # Make Vi
    Vi = np.ones(Nx*Ny, dtype = complex).reshape(Ny, Nx)*V
    Vi[:, :bd] = 0
    Vi = Vi.reshape((Nx*Ny))   
    F = np.zeros((Nx*Ny, Nx*Ny))
    for ix in range(Nx):
        for iy in range(Ny):
            i = ix + Nx *iy

            if ix < Nx-1:
                F[i, i + 1] = Vi[i]* np.sum(f(D, T) * ( np.conjugate(v_up[i])*u_dn[i+1] - u_up[i]* np.conjugate(v_dn[i + 1]) ) +   u_up[i]* np.conjugate(v_dn[i + 1]), axis = 1)
                F[i + 1, i] = Vi[i]* np.sum(f(D, T) * ( np.conjugate(v_up[i+1])*u_dn[i] - u_up[i + 1]* np.conjugate(v_dn[i]) ) +   u_up[i + 1]* np.conjugate(v_dn[i]), axis = 1)

            if ix > 0:
                F[i, i - 1] = Vi[i]* np.sum(f(D, T) * ( np.conjugate(v_up[i]) * u_dn[i - 1] - u_up[i] * np.conjugate(v_dn[i - 1]) ) +   u_up[i]* np.conjugate(v_dn[i - 1]), axis = 1)
                F[i - 1, i] = Vi[i]* np.sum(f(D, T) * ( np.conjugate(v_up[i - 1]) * u_dn[i] - u_up[i - 1] * np.conjugate(v_dn[i]) ) +   u_up[i - 1]* np.conjugate(v_dn[i]), axis = 1)

            # if iy > 0:
            F[i, (i + Nx)%(Nx*Ny)] = Vi[i]* np.sum(f(D, T) * ( np.conjugate(v_up[i])*u_dn[(i + Nx) % (Nx*Ny)] - u_up[i]* np.conjugate(v_dn[(i + Nx) % (Nx*Ny)]) ) +   u_up[i]* np.conjugate(v_dn[(i+Nx)%(Nx*Ny)]), axis = 1)
            F[(i + Nx)%(Nx*Ny), i] = Vi[i]* np.sum(f(D, T) * ( np.conjugate(v_up[(i + Nx) % (Nx*Ny)])*u_dn[i] - u_up[(i + Nx) % (Nx*Ny)]* np.conjugate(v_dn[i]) ) +   u_up[(i+Nx)%(Nx*Ny)]* np.conjugate(v_dn[i]), axis = 1)

            F[i, i - Nx] = Vi[i]* np.sum(f(D, T) * ( np.conjugate(v_up[i])*u_dn[i-Nx] - u_up[i]* np.conjugate(v_dn[i-Nx]) ) +   u_up[i]* np.conjugate(v_dn[i-Nx]), axis = 1)
            F[i - Nx, i] = Vi[i]* np.sum(f(D, T) * ( np.conjugate(v_up[i-Nx])*u_dn[i] - u_up[i-Nx]* np.conjugate(v_dn[i]) ) +   u_up[i-Nx]* np.conjugate(v_dn[i]), axis = 1)            
            # if iy < Ny-1:
    # -------------------------------------------------------------------
    # Delta_i = Ui*np.sum(u_dn *np.conjugate(v_up) * f(D, T) + u_up * np.conjugate(v_dn)* f(-D, T) , axis = 1) # Here, we used equilibrium explicitly to say (1-f(E)) = f(-E) 
    return F


@njit # ish speedup of this function of 10-20 for Nx = Ny = 50, so I guess smaller speedup for smaller matrices
def unpack_block_matrix(H_block, Nx, Ny):
    H_toreturn = np.zeros((4*Nx*Ny, 4*Nx*Ny), dtype = "complex128")
    for i in range(Nx*Ny):
        for j in range(Nx*Ny):
            H_toreturn[4*i: 4*(i + 1), 4*j: 4* j + 4] = H_block[i, j]

    return H_toreturn