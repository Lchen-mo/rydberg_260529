import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ==========================================
# Initial State
# ==========================================

amp = 0.5

labels = [
    "|00⟩",
    "|01⟩",
    "|10⟩",
    "|11⟩"
]

# CZ作用时间

N = 180

phases = np.linspace(
    0,
    np.pi,
    N
)

# ==========================================
# Figure
# ==========================================

fig = plt.figure(
    figsize=(12,8)
)

axes = []

for i in range(4):

    ax = plt.subplot(2,2,i+1)

    ax.set_xlim(-0.7,0.7)
    ax.set_ylim(-0.7,0.7)

    ax.axhline(0,color='gray')
    ax.axvline(0,color='gray')

    ax.set_aspect("equal")

    ax.set_title(labels[i])

    axes.append(ax)

# ==========================================
# Arrows
# ==========================================

arrows = []

for ax in axes:

    arrow = ax.arrow(
        0,
        0,
        amp,
        0,
        width=0.01
    )

    arrows.append(arrow)

# ==========================================
# Text Panel
# ==========================================

info = fig.text(
    0.75,
    0.2,
    "",
    fontsize=12,
    bbox=dict(boxstyle="round")
)

# ==========================================
# Update
# ==========================================

def update(frame):

    global arrows

    # 删除旧箭头

    for a in arrows:
        a.remove()

    arrows = []

    phi = phases[frame]

    phase_list = [
        0,
        0,
        0,
        phi
    ]

    for i,ax in enumerate(axes):

        p = phase_list[i]

        x = amp*np.cos(p)
        y = amp*np.sin(p)

        arrow = ax.arrow(
            0,
            0,
            x,
            y,
            width=0.02
        )

        arrows.append(arrow)

    info.set_text(

        f"CZ Gate Evolution\n\n"

        f"phase(|00>) = 0\n"
        f"phase(|01>) = 0\n"
        f"phase(|10>) = 0\n"
        f"phase(|11>) = {phi:.3f}\n\n"

        f"{phi/np.pi:.2f} π"
    )

    return arrows

# ==========================================
# Animation
# ==========================================

ani = FuncAnimation(
    fig,
    update,
    frames=N,
    interval=40,
    blit=False
)

plt.tight_layout()
plt.show()