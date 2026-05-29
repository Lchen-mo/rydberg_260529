import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from qutip import *

# ==================================
# Parameters
# ==================================

Omega = 2 * np.pi * 1.0

t_max = 5
n_steps = 300

times = np.linspace(0, t_max, n_steps)

# ==================================
# States
# ==================================

g = basis(2,0)
r = basis(2,1)

psi0 = g

# ==================================
# Hamiltonian
# ==================================

H = 0.5 * Omega * sigmax()

result = mesolve(H, psi0, times, [], [])

# ==================================
# Compute Rydberg population
# ==================================

P_r = []

for state in result.states:

    prob = abs(r.overlap(state))**2
    P_r.append(prob)

# ==================================
# Figure
# ==================================

fig, ax = plt.subplots(figsize=(6,6))

ax.set_xlim(-1,1)
ax.set_ylim(-1,1)

atom = plt.Circle((0,0), 0.2, color='blue')

ax.add_patch(atom)

ax.set_aspect('equal')

ax.set_title("Single Atom Rabi Oscillation")

# ==================================
# Animation update
# ==================================

def update(frame):

    p = P_r[frame]

    # Color interpolation
    color = (p, 0, 1-p)

    atom.set_color(color)

    return atom,

# ==================================
# Animation
# ==================================

ani = FuncAnimation(
    fig,
    update,
    frames=n_steps,
    interval=30,
    blit=True
)

plt.show()