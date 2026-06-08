# wave_total.py


#生成总波函数，只需要把它接入可视化代码即可生成三维图像，可视化接口没写。
import numpy as np
from scipy.interpolate import interp1d
from arc import Rubidium87
from atom_parameters import params

# ============================================================
# parameters
# ============================================================

r_grid = params['r_grid']
Nr = len(r_grid)

rydberg_n = params['rydberg_n']

# ============================================================
# ARC atom
# ============================================================

Rb = Rubidium87()

# ============================================================
# build real radial wavefunctions from ARC
# ============================================================

def get_arc_radial_wavefunctions():
    """
    使用 ARC 获取 Rb 原子的真实径向波函数

    返回
    ----
    phi_g : ndarray
        基态径向波函数 R_g(r)
        shape = (Nr,)

    phi_r : ndarray
        Rydberg态径向波函数 R_r(r)
        shape = (Nr,)

    说明
    ----
    ARC 返回的是:
        u(r)=rR(r)

    这里恢复:
        R(r)=u(r)/r

    并插值到:
        atom_parameters.py 中定义的 r_grid (SI单位: m)
    """

    import numpy as np
    from scipy.interpolate import interp1d
    from arc import Rubidium87

    # ========================================================
    # parameters
    # ========================================================

    r_grid = params['r_grid']

    rydberg_n = params['rydberg_n']

    dr = r_grid[1] - r_grid[0]

    # Bohr radius
    a0 = 5.29177210903e-11  # m

    # ========================================================
    # ARC atom
    # ========================================================

    Rb = Rubidium87()

    # ========================================================
    # helper function
    # ========================================================

    def build_state(n, l, j):

        s = 0.5

        # ----------------------------------------------------
        # ARC energy
        # ARC returns energy in eV
        # radialWavefunction needs Hartree
        # ----------------------------------------------------

        E_eV = Rb.getEnergy(
            n,
            l,
            j
        )

        E_hartree = E_eV / 27.211386245988

        # ----------------------------------------------------
        # radial wavefunction from ARC
        # returns:
        #   r(a0)
        #   u(r)=rR(r)
        # ----------------------------------------------------

        r_arc, u_arc, _ = Rb.radialWavefunction(
            l=l,
            s=s,
            j=j,
            stateEnergy=E_hartree,
            innerLimit=1e-5,
            outerLimit=2.0 * rydberg_n**2,
            step=0.01
        )

        # ----------------------------------------------------
        # convert to SI
        # ARC radius unit = Bohr radius
        # ----------------------------------------------------

        r_si = r_arc * a0

        # ----------------------------------------------------
        # recover R(r)=u(r)/r
        # avoid division by zero
        # ----------------------------------------------------

        R = np.zeros_like(u_arc)

        mask = r_arc > 1e-10

        R[mask] = u_arc[mask] / r_arc[mask]

        # ----------------------------------------------------
        # interpolate onto simulation grid
        # ----------------------------------------------------

        interp_func = interp1d(
            r_si,
            R,
            kind='cubic',
            bounds_error=False,
            fill_value=0.0
        )

        phi = interp_func(r_grid)

        # ----------------------------------------------------
        # normalize
        # ∫|R(r)|²dr = 1
        # ----------------------------------------------------

        norm = np.sqrt(
            np.sum(np.abs(phi)**2) * dr
        )

        phi /= norm

        return phi

    # ========================================================
    # ground state : 5s
    # ========================================================

    phi_g = build_state(
        n=5,
        l=0,
        j=0.5
    )

    # ========================================================
    # rydberg state
    # ========================================================

    phi_r = build_state(
        n=rydberg_n,
        l=0,
        j=0.5
    )

    return phi_g, phi_r

# ============================================================
# construct total two-electron wavefunction
# ============================================================

def construct_total_wavefunction(
        c_gg,
        c_gr,
        c_rg,
        c_rr
):
    """
    构造总波函数

    Psi(r1,r2)
    =
    c_gg * phi_g(r1)phi_g(r2)
    +
    c_gr * phi_g(r1)phi_r(r2)
    +
    c_rg * phi_r(r1)phi_g(r2)
    +
    c_rr * phi_r(r1)phi_r(r2)

    输入:
        c_gg,c_gr,c_rg,c_rr:
            复系数

    返回:
        Psi_total:
            shape = (Nr,Nr)
    """

    phi_g, phi_r = get_arc_radial_wavefunctions()

    # --------------------------------------------------------
    # basis states
    # --------------------------------------------------------

    phi_gg = np.outer(
        phi_g,
        phi_g
    )

    phi_gr = np.outer(
        phi_g,
        phi_r
    )

    phi_rg = np.outer(
        phi_r,
        phi_g
    )

    phi_rr = np.outer(
        phi_r,
        phi_r
    )

    # --------------------------------------------------------
    # total wavefunction
    # --------------------------------------------------------

    Psi_total = (
        c_gg * phi_gg
        +
        c_gr * phi_gr
        +
        c_rg * phi_rg
        +
        c_rr * phi_rr
    )

    return Psi_total


# ============================================================
# compatible with propagation output
# ============================================================

def construct_from_main_output(psi_step):
    """
    兼容main propagation输出

    输入:
        psi_step:
            shape = (Nr,Nr,4)

    返回:
        total_wave:
            shape = (Nr,Nr)
    """

    phi_g, phi_r = get_arc_radial_wavefunctions()

    # --------------------------------------------------------
    # basis states
    # --------------------------------------------------------

    phi_gg = np.outer(phi_g, phi_g)

    phi_gr = np.outer(phi_g, phi_r)

    phi_rg = np.outer(phi_r, phi_g)

    phi_rr = np.outer(phi_r, phi_r)

    # --------------------------------------------------------
    # channel coefficients
    # --------------------------------------------------------

    c_gg = psi_step[:, :, 0]

    c_gr = psi_step[:, :, 1]

    c_rg = psi_step[:, :, 2]

    c_rr = psi_step[:, :, 3]

    # --------------------------------------------------------
    # total wavefunction
    # --------------------------------------------------------

    total_wave = (
        c_gg * phi_gg
        +
        c_gr * phi_gr
        +
        c_rg * phi_rg
        +
        c_rr * phi_rr
    )

    return total_wave


# ============================================================
# reconstruct all time steps
# ============================================================

def reconstruct_history(history):
    """
    reconstruct all propagated wavefunctions

    输入:
        history:
            propagate输出

    返回:
        total_history:
            list of Psi(r1,r2,t)
    """

    total_history = []

    for psi_step in history:

        total_wave = construct_from_main_output(
            psi_step
        )

        total_history.append(
            total_wave
        )

    return total_history


# ============================================================
# probability density
# ============================================================

def probability_density(Psi):

    return np.abs(Psi)**2

