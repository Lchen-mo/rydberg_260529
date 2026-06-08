import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm

# ==========================
# 参数
# ==========================

Omega = 2*np.pi*1.0

T = 5
N = 400

times = np.linspace(0, T, N)

# ==========================
# Hamiltonian
# ==========================

H = 0.5 * Omega * np.array(
    [[0,1],
     [1,0]],
    dtype=complex
)

psi0 = np.array([1,0], dtype=complex)

# ==========================
# 预计算
# ==========================

Pg = []
Pr = []

cg_list = []
cr_list = []

bloch_y = []
bloch_z = []

for t in times:

    U = expm(-1j * H * t)

    psi = U @ psi0

    cg = psi[0]
    cr = psi[1]

    cg_list.append(cg)
    cr_list.append(cr)

    Pg.append(abs(cg)**2)
    Pr.append(abs(cr)**2)

    x = 2*np.real(np.conj(cg)*cr)
    y = 2*np.imag(np.conj(cg)*cr)
    z = abs(cg)**2 - abs(cr)**2

    bloch_y.append(y)
    bloch_z.append(z)

# ==========================
# Figure
# ==========================

fig = plt.figure(figsize=(12,6))

ax_bloch = plt.subplot(121)
ax_prob  = plt.subplot(122)

# ==========================
# Bloch Circle
# ==========================

theta = np.linspace(0, 2*np.pi, 300)

ax_bloch.plot(
    np.cos(theta),
    np.sin(theta),
    linewidth=1
)

ax_bloch.axhline(0)
ax_bloch.axvline(0)

ax_bloch.set_aspect('equal')

ax_bloch.set_xlim(-1.1,1.1)
ax_bloch.set_ylim(-1.1,1.1)

ax_bloch.set_title("Bloch Circle (y-z plane)")

ax_bloch.set_xlabel("y")
ax_bloch.set_ylabel("z")

# 北极和南极

ax_bloch.text(0,1.05,"|g⟩",ha='center')
ax_bloch.text(0,-1.12,"|r⟩",ha='center')

state_point, = ax_bloch.plot(
    [],
    [],
    'o',
    markersize=10
)

trajectory, = ax_bloch.plot(
    [],
    [],
    linewidth=1
)

# ==========================
# Probability Plot
# ==========================

ax_prob.set_xlim(0,T)
ax_prob.set_ylim(0,1.05)

ax_prob.grid()

ax_prob.set_title("Rabi Oscillation")

line_g, = ax_prob.plot(
    [],
    [],
    label="P(g)"
)

line_r, = ax_prob.plot(
    [],
    [],
    label="P(r)"
)

ax_prob.legend()

# ==========================
# Text
# ==========================

text_box = ax_prob.text(
    0.55,
    0.95,
    "",
    transform=ax_prob.transAxes,
    verticalalignment='top',
    bbox=dict(boxstyle="round")
)

# ==========================
# Animation
# ==========================

def update(frame):

    cg = cg_list[frame]
    cr = cr_list[frame]

    pg = Pg[frame]
    pr = Pr[frame]

    # Bloch point

    state_point.set_data(
        [bloch_y[frame]],
        [bloch_z[frame]]
    )

    trajectory.set_data(
        bloch_y[:frame+1],
        bloch_z[:frame+1]
    )

    # probability curves

    line_g.set_data(
        times[:frame+1],
        Pg[:frame+1]
    )

    line_r.set_data(
        times[:frame+1],
        Pr[:frame+1]
    )

    text_box.set_text(

        f"t = {times[frame]:.2f}\n\n"

        f"Pg = {pg:.3f}\n"
        f"Pr = {pr:.3f}\n\n"

        f"cg = {cg.real:.3f}"
        f"{cg.imag:+.3f}i\n\n"

        f"cr = {cr.real:.3f}"
        f"{cr.imag:+.3f}i"
    )

    return (
        state_point,
        trajectory,
        line_g,
        line_r,
        text_box
    )

ani = FuncAnimation(
    fig,
    update,
    frames=N,
    interval=20,
    blit=True
)

plt.tight_layout()
plt.show()