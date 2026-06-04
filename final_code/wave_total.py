# wave_total.py

import numpy as np
from scipy.interpolate import interp1d
from atom_parameters import params

# ============================================================
# parameters
# ============================================================

r_grid = params['r_grid']
Nr = len(r_grid)

internal_states = params['internal_states']
n_channels = params['n_channels']

# ============================================================
# effective radial basis functions
# ============================================================

def hydrogen_like_ground(r, a0=5.29e-11):
    """
    hydrogen-like 1s radial wavefunction
    """
    return 2.0 * np.exp(-r / a0) / (a0 ** 1.5)


def rydberg_radial(r, n=70, a0=5.29e-11):
    """
    simplified rydberg radial envelope
    """
    r0 = n**2 * a0

    return (
        (r / r0)
        * np.exp(-r / (2 * r0))
        / (r0 ** 1.5)
    )


# ============================================================
# build single-electron basis
# ============================================================

def build_single_particle_basis():

    phi_g = hydrogen_like_ground(r_grid)

    phi_r = rydberg_radial(
        r_grid,
        n=params['rydberg_n']
    )

    # normalization

    dr = r_grid[1] - r_grid[0]

    phi_g /= np.sqrt(np.sum(np.abs(phi_g)**2) * dr)

    phi_r /= np.sqrt(np.sum(np.abs(phi_r)**2) * dr)

    return phi_g, phi_r


# ============================================================
# build two-electron basis
# ============================================================

def build_two_particle_basis():

    phi_g, phi_r = build_single_particle_basis()

    basis = {}

    # --------------------------------------------------------
    # tensor products
    # --------------------------------------------------------

    basis['gg'] = np.outer(phi_g, phi_g)

    basis['gr'] = np.outer(phi_g, phi_r)

    basis['rg'] = np.outer(phi_r, phi_g)

    basis['rr'] = np.outer(phi_r, phi_r)

    return basis


# ============================================================
# reconstruct total wavefunction
# ============================================================

def reconstruct_total_wavefunction(history):
    """
    reconstruct total wavefunction from channel coefficients

    input:
        history:
            list of psi(r1,r2,c)

    return:
        total_history:
            list of Psi_total(r1,r2)
    """

    basis = build_two_particle_basis()

    total_history = []

    for psi_step in history:

        Psi_total = np.zeros(
            (Nr, Nr),
            dtype=np.complex128
        )

        # ----------------------------------------------------
        # sum all channels
        # ----------------------------------------------------

        for c, state in enumerate(internal_states):

            Psi_total += (
                psi_step[:, :, c]
                * basis[state]
            )

        total_history.append(Psi_total)

    return total_history


# ============================================================
# probability density
# ============================================================

def probability_density(Psi):

    return np.abs(Psi)**2


# ============================================================
# radial marginal distributions
# ============================================================

def radial_density_r1(Psi):

    dr = r_grid[1] - r_grid[0]

    return np.sum(
        np.abs(Psi)**2,
        axis=1
    ) * dr


def radial_density_r2(Psi):

    dr = r_grid[1] - r_grid[0]

    return np.sum(
        np.abs(Psi)**2,
        axis=0
    ) * dr


# ============================================================
# example
# ============================================================

if __name__ == "__main__":

    print("wave_total module loaded.")