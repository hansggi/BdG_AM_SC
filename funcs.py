import numpy as np
from numba import njit
from icecream import ic
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
def make_H_numba(Nx, Ny, m_arr, mz_arr,hx_arr, Delta_arr, mu,imps, skewed, periodc):
    t = 1. # For clarity in the terms below
    H = np.zeros((Nx*Ny, Nx*Ny, 4, 4), dtype="complex128")
    periodicX, periodicY = periodc
    for ix in range(0, Nx):
        for iy in range(0, Ny):
            i = ix + Nx *iy

            H[i,i] = nb_block((((mu - imps[iy, ix])*I2  + mz_arr[iy, ix] *sigmaz + hx_arr[iy, ix]*sigmax,             Delta_arr[iy, ix] * (-1j * sigmay) )  ,
                               (np.conjugate(Delta_arr[iy, ix])*1j*sigmay, - (mu - imps[iy, ix])*I2 - mz_arr[iy, ix] *sigmaz - hx_arr[iy, ix]*sigmax     ) ) )
    
            if not skewed:
                # Contribution from the right, only if not at the right edge. 
                if ix < Nx - 1 or periodicX:
                    m = min(m_arr[iy, ix], m_arr[iy, (ix+1) % Nx]) # To make sure only when both sites are within the altermanget
                    # m = (m_arr[iy, ix] + m_arr[iy, ix+1])  / 2
                    H[i, (i + 1) % (Nx*Ny)] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                            (0 * I2                           , -t * I2 - m*sigmaz)))

                # Contrbution from the left, only if not on the left edge
                if ix > 0 or periodicX:
                    m = min(m_arr[iy, ix], m_arr[iy, ix-1])
                    # m = (m_arr[iy, ix] +m_arr[iy, ix-1]) / 2
                    H[i, i - 1] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                            (0 * I2, -t * I2 - m*sigmaz)))


                # IF not periodic, include the if tests below
                if not Ny == 1:

                    if iy < Ny - 1 or periodicY:
                    # -- From below --
                        m = m_arr[iy, ix]
                        H[i, (i + Nx) % (Nx*Ny)] = nb_block(((t * I2 - m*sigmaz , 0 * I2),
                                                            (0 * I2, -t * I2 + m*sigmaz))) # Note the oposite sign in the altermagnetic part.

                    # if iy  > 0: # Except iy = 0
                    # -- From above --
                    if iy >0 or periodicY:
                        H[i, i - Nx] = nb_block(((t * I2 - m*sigmaz , 0 * I2            ),
                                                (0 * I2            , -t * I2 + m*sigmaz)))
                    
# Skewed ---------------------------------------------------------------------------
            elif skewed: 
                # Here, have used a system where the first row is further to the left than the second row etc,
                # also, the boundary splits each row in two. m_arr is NyxNx matrix, where indexing is as if
                # the grid was square.
                # On even site in y- direction ----------------------------------------------------------
                assert Ny % 2 ==0 # will get errors if not, due to periodic bdc in y-direction
                if iy % 2 == 0:
                    # -- From the right --
                    m = min(m_arr[iy, ix], m_arr[(iy - 1)%Ny, ix ])
                    H[i, (i - Nx)% (Nx*Ny)  ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                          (0 * I2, -t * I2 - m*sigmaz)))
                        
                    # -- From the left --
                    if ix > 0:
                        m = min(m_arr[iy, ix], m_arr[(iy + 1)%Ny, ix - 1])
                        H[i, (i + Nx - 1)% (Nx*Ny) ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                                 (0 * I2, -t * I2 - m*sigmaz)))

                    # -- y hopping, opposite sign for the altermangetic terms --
                    # -- From above --
                    if ix > 0:
                        m = min(m_arr[iy, ix], m_arr[(iy - 1)%Ny, ix - 1])
                        H[i,( i - Nx -1)% (Nx*Ny) ] = nb_block(((t * I2 - m*sigmaz , 0 * I2),
                                                                (0 * I2, -t * I2 + m*sigmaz)))
                            
                    # -- From below --
                    # if ix == bd[0] -1:
                    #     H[i, (i + Nx ) % (Nx*Ny)] = nb_block(((t * I2  , 0 * I2),
                    #                                           (0 * I2, -t * I2 )))
                    # else:
                    m = min(m_arr[iy, ix], m_arr[(iy + 1)%Ny, ix])
                    H[i, (i + Nx ) % (Nx*Ny)] = nb_block(((t * I2 - m*sigmaz , 0 * I2),
                                                        (0 * I2, -t * I2 + m*sigmaz)))
                # ---------------------------------------------------------------------------
                # On odd site in y-direction ------------------------------------------------
                elif iy % 2 == 1 :

                    # From the right
                    if  ix  < Nx - 1:
                        m = min(m_arr[iy, ix], m_arr[(iy - 1)%Ny, ix + 1])
                        H[i, (i - Nx +1)% (Nx*Ny) ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                                 (0 * I2, -t * I2 - m*sigmaz)))
                        # if ix == bd[0] -1:
                        #     H[i, (i - Nx +1) % (Nx*Ny)] += nb_block(((t * I2 , 0 * I2),
                        #                                         (0 * I2, -t * I2 )))
                        # else:
                        #     H[i, (i - Nx +1)% (Nx*Ny) ] += nb_block(((t * I2 + m*sigmaz , 0 * I2),
                        #                                  (0 * I2, -t * I2 - m*sigmaz)))
                        
                    # -- From the left --
                    m = min(m_arr[iy, ix], m_arr[(iy + 1)%Ny, ix])
                    H[i, (i + Nx) % (Nx*Ny)  ] = nb_block(((t * I2 + m*sigmaz , 0 * I2),
                                                            (0 * I2, -t * I2 - m*sigmaz)))

                    # -- y hopping --

                    # -- From above --
                    m = min(m_arr[iy, ix], m_arr[(iy - 1)%Ny, ix])
                    H[i,( i - Nx % (Nx*Ny)) ] = nb_block(((t * I2 - m*sigmaz , 0 * I2),
                                                           (0 * I2, -t * I2 + m*sigmaz)))
                
                    # -- From below --
                    if  ix < Nx - 1:
                        m = min(m_arr[iy, ix], m_arr[(iy + 1)%Ny, ix +1])
                        H[i, (i + Nx + 1)% (Nx*Ny)] = nb_block(((t * I2 - m*sigmaz , 0 * I2),
                                                                 (0 * I2, -t * I2 + m*sigmaz)))
                        
                        # if ix == bd[0] -1:
                        #     H[i, (i + Nx + 1)% (Nx*Ny)] += nb_block(((t * I2  , 0 * I2),
                        #                                              (0 * I2, -t * I2 )))
                        # else:
                        #     H[i, (i + Nx + 1)% (Nx*Ny)] += nb_block(((t * I2 - m*sigmaz , 0 * I2),
                        #                                         (0 * I2, -t * I2 + m*sigmaz)))
                # # -------------------------------------------------------------------------------
    
    Hr = unpack_block_matrix(H, Nx, Ny)

    # assert np.allclose(Hr, np.transpose(np.conjugate(Hr)))
    # print(np.allclose(Hr, np.transpose(np.conjugate(Hr))))
    return - Hr

"""@njit
def make_H_FT(Nx, Ny, m_arr0, mz_arr0, Delta_arr0, U, mu, bd, skewed):
    t = 1. # For clarity in the terms below
    H = np.zeros((Ny, Nx, Nx, 4, 4), dtype="complex128")
    if m_arr0.shape == (Ny, Nx):
        m_arr = m_arr0[Ny//2, :]
    if mz_arr0.shape == (Ny, Nx):
        mz_arr = mz_arr0[Ny//2, :]
    if Delta_arr0.shape == (Ny, Nx):
        Delta_arr = Delta_arr0[Ny//2, :]

    
    ks = 2 * np.pi * np.arange(0, Ny, 1)

    for nk, k in enumerate(ks):
 
        for ix in range(0, Nx):
            # ic(Delta_arr[ix])
            # Diagonal term contains y-jumping!
            H[nk,ix,ix] = nb_block((((2 * t*I2 - m_arr[ix]*sigmaz) * np.cos(k)+ mu*I2  + mz_arr[ix] *sigmaz,             Delta_arr[ix] * (-1j * sigmay) )  ,
                                (np.conjugate(Delta_arr[ix])*1j*sigmay, -(2 * t*I2 - m_arr[ix]*sigmaz) * np.cos(k) - mu*I2 - mz_arr[ix] *sigmaz     ) ) )

            if not skewed:
                # Contribution from the right, only if not at the right edge. 
                if ix < Nx - 1:
                    # m = min(m_arr[iy, ix], m_arr[iy, ix+1]) # To make sure only when both sites are within the altermanget
                    H[nk, ix, ix + 1] = nb_block(((t * I2 + m_arr[ix]*sigmaz , 0 * I2),
                                              (0 * I2                           , -t * I2 - m_arr[ix]*sigmaz)))

                # Contrbution from the left, only if not on the left edge
                if ix > 0:

                    H[nk, ix, ix - 1] = nb_block(((t * I2 + m_arr[ix]*sigmaz , 0 * I2),
                                                  (0 * I2, -t * I2 - m_arr[ix]*sigmaz)))
    
    H_up = np.zeros((Ny, 4*Nx, 4*Nx), dtype="complex128")
    for nk, k in enumerate(2 * np.pi * np.arange(0, Ny, 1)):
        H_up[nk] = unpack_block_matrix(H[nk], Nx, Ny, FT=True)
    return H

"""

          
def fd(E, T):
    # Assume equilibrium
    return (1 - np.tanh( E / ( 2 * T ))) / 2

@njit
def Delta_sc(gamma, D, Ui, T):
    # NB: gamma and energies D must already be just positive eigenvalues
    u_up = gamma[0::4, :]
    u_dn = gamma[1::4, :]
    v_up = gamma[2::4, :]
    v_dn = gamma[3::4, :]
    # print(D[-1])
    # Make Ui
    # Ui = np.ones(Nx*Ny, dtype = complex).reshape(Ny, Nx)*U

    # print(np.allclose(np.sum(u_up* np.conjugate(v_dn), axis = 1), -np.sum(u_dn* np.conjugate(v_up), axis = 1)))
    # print(np.sum(u_up* np.conjugate(v_dn), axis = 1), -np.sum(u_dn* np.conjugate(v_up), axis = 1))
    f  = (1 - np.tanh(  D / ( 2 * T ))) / 2
    # fm = (1 - np.tanh( -D / ( 2 * T ))) / 2
    # -------------------------------------------------------------------
    Delta_i = Ui*np.sum(u_dn *np.conjugate(v_up) * f +       u_up * np.conjugate(v_dn)* (1 - f) , axis = 1) # Here, we used equilibrium explicitly to say (1-f(E)) = f(-E) 
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
                F[i, i + 1] = Vi[i]* np.sum(fd(D, T) * ( np.conjugate(v_up[i])*u_dn[i+1] - u_up[i]* np.conjugate(v_dn[i + 1]) ) +   u_up[i]* np.conjugate(v_dn[i + 1]), axis = 1)
                F[i + 1, i] = Vi[i]* np.sum(fd(D, T) * ( np.conjugate(v_up[i+1])*u_dn[i] - u_up[i + 1]* np.conjugate(v_dn[i]) ) +   u_up[i + 1]* np.conjugate(v_dn[i]), axis = 1)

            if ix > 0:
                F[i, i - 1] = Vi[i]* np.sum(fd(D, T) * ( np.conjugate(v_up[i]) * u_dn[i - 1] - u_up[i] * np.conjugate(v_dn[i - 1]) ) +   u_up[i]* np.conjugate(v_dn[i - 1]), axis = 1)
                F[i - 1, i] = Vi[i]* np.sum(fd(D, T) * ( np.conjugate(v_up[i - 1]) * u_dn[i] - u_up[i - 1] * np.conjugate(v_dn[i]) ) +   u_up[i - 1]* np.conjugate(v_dn[i]), axis = 1)

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
def unpack_block_matrix(H_block, Nx, Ny, FT = False):
    if not FT:
        H_toreturn = np.zeros((4*Nx*Ny, 4*Nx*Ny), dtype = "complex128")
        for i in range(Nx*Ny):
            for j in range(Nx*Ny):
                H_toreturn[4*i: 4*(i + 1), 4*j: 4* j + 4] = H_block[i, j]
    else:
        H_toreturn = np.zeros((4*Nx, 4*Nx), dtype = "complex128")
        for i in range(Nx):
            for j in range(Nx):
                H_toreturn[4*i: 4*(i + 1), 4*j: 4* j + 4] = H_block[i,j]
    return H_toreturn