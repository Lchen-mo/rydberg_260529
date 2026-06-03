# main.py

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from atom_parameters import params

from wavefunction import (
    initialize_wavefunction,
    radial_density
)

from propagator import (
    build_total_hamiltonian,
    propagate
)

# ============================================================
# user parameters
# ============================================================

psi_type = 'rr'

dt = 1e-10

n_steps = 100

save_every = 5

perturb = 0.0

# ============================================================
# system parameters
# ============================================================

r_grid = params['r_grid']

dr = params['dr']

Nr = len(r_grid)

n_channels = params['n_channels']

internal_states = params['internal_states']

# ============================================================
# initialize wavefunction
# ============================================================

print("\n================================================")
print("Initializing wavefunction...")
print("================================================\n")

psi0 = initialize_wavefunction(
    psi_type=psi_type,
    perturb=perturb
)

print("Initial wavefunction shape:")
print(psi0.shape)

# ============================================================
# build Hamiltonian
# ============================================================

print("\n================================================")
print("Building Hamiltonian...")
print("================================================\n")

t0 = time.time()

H = build_total_hamiltonian()

t1 = time.time()

print(f"\nHamiltonian build finished.")
print(f"Time used: {t1 - t0:.2f} s")

# ============================================================
# propagation
# ============================================================

print("\n================================================")
print("Starting propagation...")
print("================================================\n")

t2 = time.time()

history = propagate(
    psi0=psi0,
    H=H,
    dt=dt,
    n_steps=n_steps,
    save_every=save_every
)

t3 = time.time()

print("\nPropagation finished.")
print(f"Propagation time: {t3 - t2:.2f} s")

# ============================================================
# final wavefunction
# ============================================================

psi_final = history[-1]

# ============================================================
# rr population
# ============================================================

idx_rr = internal_states.index('rr')

rr_population = (
    np.sum(
        np.abs(
            psi_final[:, :, idx_rr]
        )**2
    ) * dr * dr
)

print("\n================================================")
print("Final observables")
print("================================================\n")

print(f"Final |rr> population = {rr_population:.6e}")

# ============================================================
# radial density
# ============================================================

rho = radial_density(psi_final)

# ============================================================
# plot radial density
# ============================================================

print("\nPlotting radial density...\n")

plt.figure(figsize=(7,6))

plt.imshow(
    rho.T,
    origin='lower',
    extent=[
        r_grid[0]*1e6,
        r_grid[-1]*1e6,
        r_grid[0]*1e6,
        r_grid[-1]*1e6
    ],
    aspect='auto'
)

plt.xlabel(r"$r_1$ [$\mu$m]")

plt.ylabel(r"$r_2$ [$\mu$m]")

plt.title("Radial Probability Density")

plt.colorbar(label="Probability Density")

plt.tight_layout()

plt.show()

# ============================================================
# rr channel density
# ============================================================

print("\nPlotting rr channel density...\n")

rr_density = np.abs(
    psi_final[:, :, idx_rr]
)**2

plt.figure(figsize=(7,6))

plt.imshow(
    rr_density.T,
    origin='lower',
    extent=[
        r_grid[0]*1e6,
        r_grid[-1]*1e6,
        r_grid[0]*1e6,
        r_grid[-1]*1e6
    ],
    aspect='auto'
)

plt.xlabel(r"$r_1$ [$\mu$m]")

plt.ylabel(r"$r_2$ [$\mu$m]")

plt.title(r"Final $|rr\rangle$ Density")

plt.colorbar(label=r"$| \psi_{rr} |^2$")

plt.tight_layout()

plt.show()

# ============================================================
# norm check
# ============================================================

norm = np.sum(
    np.abs(psi_final)**2
) * dr * dr

print("\n================================================")
print("Norm check")
print("================================================\n")

print(f"Wavefunction norm = {norm:.8f}")

# ============================================================
# runtime summary
# ============================================================

total_runtime = t3 - t0

print("\n================================================")
print("Simulation complete")
print("================================================\n")

print(f"Total runtime: {total_runtime:.2f} s")