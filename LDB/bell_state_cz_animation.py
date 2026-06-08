import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================================
# Basis: |00>, |01>, |10>, |11>
# ================================

# Hadamard |0> -> (|0>+|1>)/√2
H = 1/np.sqrt(2) * np.array([[1,1],[1,-1]])

# tensor product H⊗H
H2 = np.kron(H, H)

# Bell target state |Φ+>
psi_bell = np.array([1,1,1,-1], dtype=complex) / 2

# ================================
# Time steps: 0→Hadamard→CZ
# ================================

N = 200
t_split = 80

# ================================
# initial state |00>
# ================================

psi = np.array([1,0,0,0], dtype=complex)

history = []

# ================================
# helper: reduced density entropy
# ================================

def entanglement_entropy(psi):

    # reshape into 2-qubit state
    psi2 = psi.reshape(2,2)

    # reduced density matrix
    rho = psi2 @ psi2.conj().T

    eigs = np.linalg.eigvalsh(rho)

    eigs = eigs[eigs > 1e-12]

    return -np.sum(eigs * np.log2(eigs))

# ================================
# evolution simulation
# ================================

for t in range(N):

    if t < t_split:
        # Hadamard stage
        psi = H2 @ np.array([1,0,0,0], dtype=complex)

    else:
        # CZ stage gradually applied
        phi = (t - t_split) / (N - t_split) * np.pi

        psi_temp = H2 @ np.array([1,0,0,0], dtype=complex)
        psi_temp[3] *= np.exp(1j * phi)

        psi = psi_temp

    history.append(psi)

# ================================
# plotting
# ================================

fig = plt.figure(figsize=(12,6))

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

ax1.set_title("Amplitude evolution")

ax1.set_ylim(-0.6,0.6)
ax1.axhline(0,color='gray')

labels = ["00","01","10","11"]

bars = ax1.bar(labels, np.zeros(4))

ax2.set_title("Fidelity & Entanglement")

ax2.set_xlim(0,N)
ax2.set_ylim(0,1.1)

line_fid, = ax2.plot([],[],label="Fidelity")
line_ent, = ax2.plot([],[],label="Entanglement")

ax2.legend()

fids = []
ents = []

# ================================
# update function
# ================================

def update(frame):

    psi = history[frame]

    probs = np.abs(psi)**2

    for i,b in enumerate(bars):
        b.set_height(probs[i])

    fidelity = np.abs(np.vdot(psi_bell, psi))**2
    ent = entanglement_entropy(psi)

    fids.append(fidelity)
    ents.append(ent)

    line_fid.set_data(range(len(fids)), fids)
    line_ent.set_data(range(len(ents)), ents)

    return bars

ani = FuncAnimation(fig, update, frames=N, interval=40, blit=False)

plt.tight_layout()
plt.show()