import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm

# =====================================
# 参数
# =====================================

Omega = 2*np.pi*1.0

# 强Blockade
V = 20*Omega

T = 5
N = 300

times = np.linspace(0, T, N)

# =====================================
# Hamiltonian
# =====================================

H_drive = 0.5 * Omega * np.array(
[
    [0,1,1,0],
    [1,0,0,1],
    [1,0,0,1],
    [0,1,1,0]
],
dtype=complex
)

H_int = np.diag([0,0,0,V])

H = H_drive + H_int

# =====================================
# Initial state |gg>
# =====================================

psi0 = np.array(
    [1,0,0,0],
    dtype=complex
)

# =====================================
# Precompute
# =====================================

Pgg = []
Pgr = []
Prg = []
Prr = []

PA = []
PB = []

for t in times:

    U = expm(-1j * H * t)

    psi = U @ psi0

    pgg = abs(psi[0])**2
    pgr = abs(psi[1])**2
    prg = abs(psi[2])**2
    prr = abs(psi[3])**2

    Pgg.append(pgg)
    Pgr.append(pgr)
    Prg.append(prg)
    Prr.append(prr)

    # 单原子激发概率

    pa = prg + prr
    pb = pgr + prr

    PA.append(pa)
    PB.append(pb)

# =====================================
# Figure
# =====================================

fig = plt.figure(figsize=(12,6))

ax_atoms = plt.subplot(121)
ax_prob = plt.subplot(122)

# =====================================
# Atom view
# =====================================

ax_atoms.set_xlim(0,10)
ax_atoms.set_ylim(0,10)

ax_atoms.set_aspect("equal")

ax_atoms.set_title("Rydberg Blockade")

ax_atoms.axis("off")

# 连线

ax_atoms.plot(
    [3,7],
    [5,5],
    linewidth=2
)

# 原子

atomA = ax_atoms.scatter(
    [3],
    [5],
    s=1000,
    c=[0]
)

atomB = ax_atoms.scatter(
    [7],
    [5],
    s=1000,
    c=[0]
)

# blockade圈

blockade = plt.Circle(
    (3,5),
    3,
    fill=False,
    linestyle='--'
)

ax_atoms.add_patch(blockade)

# 文字

info = ax_atoms.text(
    1,
    1,
    ""
)

# =====================================
# Probability plot
# =====================================

ax_prob.set_xlim(0,T)
ax_prob.set_ylim(0,1)

ax_prob.grid()

line_rr, = ax_prob.plot(
    [],
    [],
    linewidth=3,
    label="P(rr)"
)

line_gr, = ax_prob.plot(
    [],
    [],
    label="P(gr)"
)

line_rg, = ax_prob.plot(
    [],
    [],
    label="P(rg)"
)

ax_prob.legend()

# =====================================
# Animation
# =====================================

def update(frame):

    pa = PA[frame]
    pb = PB[frame]

    # 红色强度表示激发概率

    atomA.set_array(
        np.array([pa])
    )

    atomB.set_array(
        np.array([pb])
    )

    atomA.set_clim(0,1)
    atomB.set_clim(0,1)

    # 概率曲线

    line_rr.set_data(
        times[:frame+1],
        Prr[:frame+1]
    )

    line_gr.set_data(
        times[:frame+1],
        Pgr[:frame+1]
    )

    line_rg.set_data(
        times[:frame+1],
        Prg[:frame+1]
    )

    info.set_text(

        f"t={times[frame]:.2f}\n\n"

        f"P_A(r)={pa:.3f}\n"
        f"P_B(r)={pb:.3f}\n\n"

        f"P(rr)={Prr[frame]:.5f}"
    )

    return (
        atomA,
        atomB,
        line_rr,
        line_gr,
        line_rg,
        info
    )

ani = FuncAnimation(
    fig,
    update,
    frames=N,
    interval=30,
    blit=False
)

plt.tight_layout()
plt.show()