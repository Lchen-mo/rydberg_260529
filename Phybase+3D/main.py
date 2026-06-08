# main.py

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

#from wave_total import reconstruct_total_wavefunction
from atom_parameters import params   #物理参数

from wavefunction import (
    initialize_wavefunction,
    radial_density
)                                    #波函数

from propagator import (
    build_total_hamiltonian,
    propagate
)                                    #哈密顿量和传播

# ============================================================
# user parameters
# ============================================================

psi_type = 'gg'

dt = 10e-9

n_steps = 100

save_every = 10

perturb = 0.0

# ============================================================
# system parameters
# ============================================================

r_grid = params['r_grid']

dr = params['dr']

r1_0=params['r1_0']
r2_0=params['r2_0']
sigma=params['sigma']

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
    perturb=perturb,
    r1_0=r1_0,
    r2_0=r2_0,
    sigma=sigma
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

#total_history = reconstruct_total_wavefunction(history)

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
# radial density 表示第一个粒子出现在r1+dr1,同时第二个粒子出现在r2+dr2的概率
# ============================================================

rho = radial_density(psi_final)*1e12  #单位转化为平方微米

# ============================================================
# rr channel density
# ============================================================


rr_density = np.abs(
    psi_final[:, :, idx_rr]
)**2

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



#后续三维可视化没做，留有接口