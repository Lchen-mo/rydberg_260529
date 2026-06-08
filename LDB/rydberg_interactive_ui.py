import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================
# lattice
# ==========================

N = 4
num = N * N

pos = np.array([
    (i, j)
    for i in range(N)
    for j in range(N)
])

Rb = 1.2

# ==========================
# discrete state (IMPORTANT)
# ==========================

# 0 = ground, 1 = Rydberg
state = np.zeros(num, dtype=int)

drive_mask = np.zeros(num)

pulse = 0

# ==========================
# UI click
# ==========================

def on_click(event):

    if event.inaxes != ax:
        return

    x, y = event.xdata, event.ydata

    d = np.linalg.norm(pos - np.array([x, y]), axis=1)

    i = np.argmin(d)

    drive_mask[i] = 1 - drive_mask[i]

    print("toggle drive:", i)

# ==========================
# HARD blockade dynamics
# ==========================

def step(state):

    new = state.copy()

    for i in range(num):

        if drive_mask[i] == 0:
            continue

        # check blockade condition
        blocked = False

        for j in range(num):

            if state[j] == 1:

                dist = np.linalg.norm(pos[i] - pos[j])

                if dist < Rb:
                    blocked = True
                    break

        # excitation rule
        if not blocked:
            if np.random.rand() < 0.25:
                new[i] = 1
        else:
            new[i] = 0  # forced suppression

        # decay
        if np.random.rand() < 0.05:
            new[i] = 0

    return new

# ==========================
# entanglement proxy
# ==========================

def ent(s):
    return np.var(s)

# ==========================
# figure
# ==========================

fig, ax = plt.subplots(figsize=(6,6))

ax.set_xlim(-0.5, N-0.5)
ax.set_ylim(-0.5, N-0.5)
ax.set_aspect("equal")

ax.set_title("TRUE Rydberg Blockade (hard constraint model)")

sc = ax.scatter(
    pos[:,0],
    pos[:,1],
    c=state,
    cmap="coolwarm",
    s=700,
    vmin=0,
    vmax=1
)

circles = [
    plt.Circle((x,y), Rb, fill=False, alpha=0.25)
    for x,y in pos
]

for c in circles:
    ax.add_patch(c)

txt = ax.text(
    0.02,0.98,"",
    transform=ax.transAxes,
    va="top",
    bbox=dict(boxstyle="round")
)

fig.canvas.mpl_connect("button_press_event", on_click)

# ==========================
# update
# ==========================

def update(frame):

    global state

    state = step(state)

    sc.set_array(state)

    txt.set_text(
        f"Step {frame}\n"
        f"Excitations: {np.sum(state)}\n"
        f"Entanglement proxy: {ent(state):.3f}\n"
        f"Active drives: {np.sum(drive_mask)}"
    )

    return sc, txt

ani = FuncAnimation(
    fig,
    update,
    frames=400,
    interval=60,
    blit=False
)

plt.show()