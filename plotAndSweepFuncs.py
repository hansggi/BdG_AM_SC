def sweep_Delta(Nx, Ny, mg,mz, U, mu, Deltag, bd, tol, num_it, skewed , alignment = None):
    Tc0 = 0.3
    t = 1.
    # Calc Tc in the mg = 0 case
    param0 = (t, U, mu, 0, 0)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag, param0, Tc0, bd, num_it, skewed = skewed, alignment = alignment) 
    # print(Tc)
    # -----------------------------------------------------------
    # Tc in this case
    param = (t, U, mu, mg, mz)
    Tc2 = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd, num_it, skewed = skewed) 
    print(f"Tc :{Tc:.4f}, \nTc2:{Tc2:.4f}.")
    T_plots = np.array([1e-8, 0.5 * Tc , 0.95 * Tc, 0.96*Tc,0.97*Tc, 0.98*Tc, 1.0 * Tc, 1.05*Tc])
    T_plots = np.linspace(0.001, Tc, 10)
    fig, ax = plt.subplots()
    for i, T in enumerate(T_plots):
        Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T_plots[i], param, bd,  skewed=skewed)
        Fs, Ft = pairing_amplitude(gamma, D, T)
        Fs = Fs.reshape(Ny, Nx)
        Ft = Ft.reshape(Ny, Nx)
        if i == 2:
                fig2, ax2 = plt.subplots()
                # divider = make_axes_locatable(ax2)
                # cax = divider.append_axes('right', size='5%', pad=0.05)

                im = ax2.imshow(np.abs(Fs))
                # ax2[1].imshow(np.abs(Ft))
                fig.colorbar(im)

                # fig2.colorbar(im, cax=cax, orientation='vertical')    # plt.savefig(f"1d_Delta/N=({Nx},{Ny}),mg={mg:.1f}.pdf", bbox_inches='tight')
                # ax2.set_title(f"T = {T}")
                # fig2.legend()

        
        # Delta_i = Delta_i.reshape((Ny, Nx))
        ax.plot(np.abs(Fs[Ny//2, :]), label = f"T/Tc = {T/Tc:.2f}S")
        # ax.plot(np.abs(Ft[Ny//2, :]), label = f"T/Tc = {T/Tc:.2f}T", linestyle = "dashed")


    fig.suptitle(f" m ={mg:.1f} mz = {mz} U = {U}  mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    ax.set_ylabel("F_i")
    fig.legend()
    # plt.savefig(f"1d_Delta/N=({Nx},{Ny}),mg={mg:.1f}.pdf", bbox_inches='tight')
    plt.show()

def sweep_Delta_mz(Nx, Ny, mzs, T, U, mu, Deltag, bd, tol, num_it,  skewed ):
    mg = 0
    Tc0 = 0.3
    t = 1.
    # Calc Tc in the mg = 0 case
    param0 = (t, U, mu, 0, 0)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag, param0, Tc0, bd, num_it, skewed = skewed) 
    print("Tc: ", Tc)
    fig3, ax3 = plt.subplots()
    fig, ax = plt.subplots()
    for i, mz in enumerate(mzs):
        print("Mz = ", mz)

        param = (t, U, mu, mg, mz)
        Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T, param, bd,  skewed=skewed) 
        Fs, Ft = pairing_amplitude(gamma, D, T)
        Fs = Fs.reshape(Ny, Nx)
        Ft = Ft.reshape(Ny, Nx)

        ax.plot(np.abs(Fs[Ny//2, :]), label = f"mz = {mz:.2f}Sing")
        Nup,Ndn = N_sigma(gamma, D, T)
        print("Nup tot: ", np.sum(Nup))
        print("Ndn tot: ", np.sum(Ndn))
        print("Ntot: ", np.sum(Ndn) + np.sum(Nup))
        ax3.plot(np.abs(Nup), label = f"mz = {mz:.2f}up")
        ax3.plot(np.abs(Ndn), label = f"mz = {mz:.2f}dn")

        # ax.plot(np.abs(Ft[Ny//2, :]), linestyle = "dashed", label = f"mz = {mz:.2f}Trip")
        if i ==2:
            fig2, ax2 = plt.subplots()
            # divider = make_axes_locatable(ax2)
            # cax = divider.append_axes('right', size='5%', pad=0.05)

            ax2.imshow(np.abs(Delta_i.reshape(Ny, Nx)))
            # ax2[1].imshow(np.abs(Ft))

        # ax.plot(np.abs(Delta_i[Ny//2, :]), label = f"mz = {mz:.2f}")
    fig3.legend()
    ax.set_ylabel("Singlet pairing")
    fig.suptitle(f" mz ={mz:.1f}, U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    fig.legend()
    plt.savefig(f"Delta_mz/N=({Nx},{Ny}),mg={mz:.1f}.pdf", bbox_inches='tight')
    plt.show()

def sweep_Delta_mg(Nx, Ny, mgs, T, U, mu, Deltag,   bd, tol, num_it, skewed ):
    t=1.
    mz = 0.
    Tc0 = 0.3

    # Calc Tc in the mg = 0 case
    param0 = (t, U, mu, 0, 0)
    Tc = calc_Tc_binomial(Nx, Ny, Deltag,   param0, Tc0, bd, num_it, skewed = skewed) 
    fig, ax = plt.subplots()
    fig3, ax3 = plt.subplots()
    for i, mg in enumerate(mgs):
        print("Mg = ", mg)
        param = (t, U, mu, mg, mz)
        Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T, param, bd,  skewed=skewed) 
        # Delta_i = Delta_i.reshape((Ny, Nx))
        Fs, Ft = pairing_amplitude(gamma, D, T)
        Fs = Fs.reshape(Ny, Nx)
        Ft = Ft.reshape(Ny, Nx)
        Nup,Ndn = N_sigma(gamma, D, T)
        print("Nup tot: ", np.sum(Nup))
        print("Ndn tot: ", np.sum(Ndn))
        print("Ntot: ", np.sum(Ndn) + np.sum(Nup))
        # ax3.plot(np.abs(Nup), label = f"mz = {mz:.2f}up")
        # ax3.plot(np.abs(Ndn), label = f"mz = {mz:.2f}dn")
        ax3.plot(np.abs(Nup), label = f"mg = {mg:.2f}up")
        ax3.plot(np.abs(Ndn), label = f"mg = {mg:.2f}dn")

        ax.plot(np.abs(Fs[Ny//2, :]), label = f"mg = {mg:.2f}Sing")
        # ax.plot(np.abs(Ft[Ny//2, :]), linestyle = "dashed", label = f"mg = {mg:.2f}Trip")
    fig3.legend()
    fig.suptitle(f" mz ={0}, U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    fig.legend()
    plt.savefig(f"1d_Delta/N=({Nx},{Ny}),mg={mg:.1f}numit={num_it}_T={T}.pdf", bbox_inches='tight')

    plt.show()

def Tc_fo_mg(Nx, Ny, mgs, U, mu, Deltag,   bd, num_it, skewed ):
    t = 1.
    mz = 0
    Tc0 = 0.3
    Tcs = np.zeros_like(mgs)
    fig, ax = plt.subplots()
    for i, mg in enumerate(mgs):
        print(f"running for mg = {mg}")
        param = (t, U, mu, mg, mz)
        Tcs[i] = calc_Tc_binomial(Nx, Ny, Deltag,   param, Tc0, bd, num_it, skewed=skewed)
        print(f"Tc0 = {Tcs[0]:.4f}, Tc:{Tcs[i]:.4f}.")

    ax.plot(mgs, Tcs)
    fig.suptitle(f" mg ={mg:.1f}, U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    # fig.legend()
    ax.set_xlabel("m")
    ax.set_ylabel("Tc")

    plt.savefig(f"mg_sweep/N=({Nx},{Ny})numit={num_it}sw={skewed}.pdf", bbox_inches='tight')
    # plt.show()

# def Tc_one(Nx, Ny, mg,mz,t,Tc0, U, mu, Deltag,   bd, num_it, skewed, alignment):
#     param = (t, U, mu, mg, mz)
#     Tc = calc_Tc_binomial(Nx, Ny, m_arr, mz_arr, Delta_arr, Deltag, U, Tc0, bd, num_it, skewed, alignment)

#     return Tc

def Tc_fo_mz(Nx, Ny, mzs, U, mu, Deltag, bd, NDelta, skewed ):
    mg = 0
    t = 1.
    Tc0 = 0.3
    Tcs = np.zeros_like(mzs)
    fig, ax = plt.subplots()
    for i, mz in enumerate(mzs):
        print(f"running for mz = {mz}")



        param = (t, U, mu, mg, mz)
        Tcs[i] = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd, NDelta,  skewed=skewed)
        print(Tcs[i])
    ax.plot(mzs, Tcs)
    fig.suptitle(f" U = {U}, mu = {mu}, N = {Nx, Ny}, sw = {skewed}")
    # fig.legend()
    ax.set_xlabel("mz")
    ax.set_ylabel("Tc")

    plt.savefig(f"mz_sweep/N=({Nx},{Ny})N_D={NDelta}.pdf", bbox_inches='tight')


def Tc_fo_Nx(Nxs, Ny, mg, mz, U, mu, Deltag,   bd, tol, num_it, skewed = False):
    Tcs = np.zeros_like(Nxs, dtype = float)
    Tc0s = np.zeros_like(Nxs, dtype = float)
    fig, ax = plt.subplots()
    t = 1.
    Tc0 = 0.3
    param = (t, U, mu, mg, mz)

    for i, Nx in enumerate(Nxs):
        print(f"running for Nx = {Nx}")
        # Delta_i, gamma, D = calc_Delta_sc(Nx, Ny, Deltag, tol, T_plots[i], param, bd,  skewed=skewed) 
        # Delta_i = Delta_i.reshape((Ny, Nx))
        Tc0s[i] = calc_Tc_binomial(Nx - bd, Ny, Deltag,   param, Tc0, 0, num_it,  skewed=skewed)
        Tcs[i] = calc_Tc_binomial(Nx, Ny, Deltag,   param, Tc0, bd, num_it,  skewed=skewed)
        # print(f"Tc :{Tc:.4f}, \nTc2:{Tc2:.4f}.")
        # Tcs[i] = Tc2
        # ax.plot(np.abs(Delta_i[Ny//2, :]), label = f"T/Tc = {T/Tc:.2f}")

    print(Tcs)
    ax.plot(Nxs, Tcs)
    ax.plot(Nxs, Tc0s, label = "bd=0")
    fig.suptitle(f"Nxs=({Nxs[0]},{Nxs[-1]}, Ny={Ny}),num_it={num_it}m={mg}mz={mz}")
    # fig.legend()
    fig.legend()
    ax.set_xlabel("Nx")
    ax.set_ylabel("Tc")
    plt.savefig(f"Nxsweep/Nxs=({Nxs[0]},{Nxs[-1]}Ny={Ny})num_it={num_it}m={mg}mz={mz}.pdf", bbox_inches='tight')

# @njit(cache = True)
def Tc_fo_Ny(Nx, Nys, mg, mz, U, mu, Deltag,   bd, tol, num_it, skewed = False):
    t = 1.
    Tc0 = 0.3
    # Tcs = np.zeros_like(Nys, dtype = float)
    Tcs = np.zeros(len(Nys))
    param = (t, U, mu, mg, mz)

    for i, Ny in enumerate(Nys):
        print(f"running for Ny = {Ny}")
        Tcs[i] = calc_Tc_binomial(Nx, Ny, Deltag,   param, Tc0, bd, num_it,  skewed=skewed)
        print("Tc = ", Tcs[i])

    fig, ax = plt.subplots()

    ax.plot(Nys, Tcs)
    fig.suptitle(f"Ny-sweep")
    # fig.legend()
    ax.set_xlabel("Ny")
    ax.set_ylabel("Tc")
    plt.savefig(f"Ny_sweep/PNx={Nx}_numit={num_it}m={mg}mz={mz}sw={skewed}.pdf", bbox_inches='tight')
    return Tcs

def numitSweep(Nx, Ny, mg, mz, U, mu, Deltag, bd, num_its, skewed, alignment):
    Tc0 = 0.3
    t = 1.
    Tcs = np.zeros_like(num_its, dtype = float)
    # Tc0s = np.zeros_like(Nxs, dtype = float)
    fig, ax = plt.subplots()
    param = (t, U, mu, mg, mz)

    for i, num_it in enumerate(num_its):
        print(f"Running for numit = {num_it}")
        Tcs[i] = calc_Tc_binomial(Nx, Ny, Deltag, param, Tc0, bd, num_its[i],  skewed=skewed)

    ax.plot(num_its, Tcs/0.1343)
    fig.suptitle(f"num_its=({num_its[0]},{num_its[-1]},Nx={Nx} Ny={Ny})m={mg}mz={mz}")
    ax.set_xlabel("num_iter")
    ax.set_ylabel("Tc")
    ic(Tcs)
    ic(Tcs/0.1343)
    ic(num_its)
    plt.savefig(f"numitSweepStef/Nx={Nx}Ny={Ny}numits=({num_its[0]},{num_its[-1]}m={mg}mz={mz}sw={skewed}.pdf", bbox_inches='tight')


# @njit(cache = True)
"""
def does_Delta_increase_steff(Nx, Ny,m_arr, mz_arr, Deltag, T, param, Delta_arr1, bd,  NDelta, skewed = False):
    # Here, Deltag must be the guess, if Delta < Deltag, 
    t, U, mu, mg, mz = param
    StefNum = 8000
    # Delta_tab = np.zeros((NDelta, Ny, Nx), dtype="complex128")
    # Delta_tab[0, :, :] = Delta_arr1.copy()


    Deltapp = np.zeros_like(Delta_arr1)
    Deltap  = np.zeros_like(Delta_arr1)
    Delta   = Delta_arr1.copy()
    ind = np.nonzero(Delta)

    ic(T)
    for i in range(1, NDelta):
        H = make_H_numba(Nx, Ny, m_arr, mz_arr, Delta, param, bd, skewed = skewed)
        D, gamma = np.linalg.eigh(H)
        D, gamma = D[2*Nx * Ny:], gamma[:, 2*Nx * Ny:]
        # print(i, np.allclose(Delta_tab[i], 0))
        Deltapp = Deltap.copy()
        Deltap = Delta.copy()
        Delta = Delta_sc(gamma, D, U, T, Nx, Ny, bd).reshape(Ny, Nx)
        # if np.amin(np.abs(Delta[ind] - Deltap[ind])) < 1e-18:
        #     if np.amax(np.abs(Delta[ind] - Deltap[ind])) < 1e-12:
        #        break

        #     print("Oh Ow, ", np.amin(np.abs(Delta[ind] - Deltap[ind])))
        #     print("Max error is ", np.amax(np.abs(Delta[ind] - Deltap[ind])))
        #     time.sleep(5)
        if i%StefNum==0 and i > 100:
            print("error")
            Delta[ind] = Deltapp[ind] - (Deltap[ind] - Deltapp[ind])**2 / (Delta[ind] - 2 * Deltap[ind] + Deltapp[ind])


    if bd[1] < Nx:
        Delta_bulk = np.abs(Delta[ Ny//2, (bd[0] + bd[1])//2])

    else:
        Delta_bulk = np.abs(Delta[Ny//2, (bd[0] + Nx)//2])

    if Delta_bulk <= np.abs(Deltag):
        return False
    else:
        return True

"""