import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ================================
# Lattice
# ================================

N = 4
num = N * N

pos = np.array([
    (i, j)
    for i in range(N)
    for j in range(N)
])

# ================================
# Physics parameters
# ================================

C6 = 1.0
Rb = 1.2
drive = 0.03

def dist(i, j):
    return np.linalg.norm(pos[i] - pos[j])

def V(i, j):
    r = dist(i, j)
    return 0 if r == 0 else C6 / r**6

# ================================
# precompute interaction graph
# ================================

Vmat = np.zeros((num, num))
for i in range(num):
    for j in range(num):
        Vmat[i, j] = V(i, j)

# ================================
# state (Rydberg probability)
# ================================

p = np.zeros(num)
p[0] = 0.8

# ================================
# entanglement proxy (fluctuation entropy)
# ================================

def entanglement(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return -np.sum(p*np.log(p)) / num

# ================================
# update dynamics (soft blockade)
# ================================

def step(p):

    new = np.zeros_like(p)

    for i in range(num):

        blockade = 0

        for j in range(num):
            if i != j:
                blockade += Vmat[i, j] * p[j]

        excite = drive * np.exp(-blockade)

        new[i] = p[i] + excite*(1 - p[i]) - 0.05*p[i]

        new[i] = np.clip(new[i], 0, 1)

    return new

# ================================
# figure
# ================================

fig, ax = plt.subplots(figsize=(7,7))

ax.set_xlim(-0.5, N-0.5)
ax.set_ylim(-0.5, N-0.5)

ax.set_aspect("equal")

ax.set_title("4×4 Rydberg CZ Network (Blockade + Entanglement)")

# atoms
sc = ax.scatter(
    pos[:,0],
    pos[:,1],
    c=p,
    cmap="coolwarm",
    s=600,
    vmin=0,
    vmax=1
)

# blockade circles
circles = [
    plt.Circle((x,y), Rb, fill=False, alpha=0.15)
    for x,y in pos
]

for c in circles:
    ax.add_patch(c)

# CZ edges
lines = []

for i in range(num):
    for j in range(i+1, num):
        if Vmat[i,j] > 0.05:
            line, = ax.plot([], [], lw=0.5, alpha=0.2)
            lines.append((i,j,line))

# text
txt = ax.text(
    0.02, 0.98, "",
    transform=ax.transAxes,
    va="top",
    bbox=dict(boxstyle="round")
)

# ================================
# update
# ================================

def update(frame):

    global p

    p = step(p)

    sc.set_array(p)

    # update edges
    for i,j,line in lines:
        if Vmat[i,j] > 0.05:

            alpha = min(1.0, Vmat[i,j]*0.5)

            line.set_data(
                [pos[i,0], pos[j,0]],
                [pos[i,1], pos[j,1]]
            )
            line.set_alpha(alpha)

    E = entanglement(p)

    txt.set_text(
        f"Step: {frame}\n"
        f"<p> = {np.mean(p):.3f}\n"
        f"Entanglement proxy = {E:.3f}\n"
        f"Max V = {np.max(Vmat):.2f}"
    )

    return sc, txt

ani = FuncAnimation(
    fig,
    update,
    frames=200,
    interval=60,
    blit=False
)

plt.show()