import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm

# ==================================================
# Physical Parameters
# ==================================================

Omega = 2*np.pi

C6 = 1000

Rb = (C6/Omega)**(1/6)

# 距离扫描

R_values = np.linspace(
    12,
    1.5,
    120
)

# 时间演化窗口

Tmax = 5

Nt = 300

times = np.linspace(
    0,
    Tmax,
    Nt
)

# ==================================================
# Compute max Prr
# ==================================================

Prr_max_list = []

for R in R_values:

    V = C6/(R**6)

    H_drive = 0.5*Omega*np.array(
        [
            [0,1,1,0],
            [1,0,0,1],
            [1,0,0,1],
            [0,1,1,0]
        ],
        dtype=complex
    )

    H_int = np.diag(
        [0,0,0,V]
    )

    H = H_drive + H_int

    psi0 = np.array(
        [1,0,0,0],
        dtype=complex
    )

    Prr_vs_time = []

    for t in times:

        U = expm(-1j*H*t)

        psi = U @ psi0

        Prr_vs_time.append(
            abs(psi[3])**2
        )

    Prr_max_list.append(
        np.max(Prr_vs_time)
    )

# ==================================================
# Figure
# ==================================================

fig = plt.figure(figsize=(14,6))

ax_atoms = plt.subplot(121)
ax_curve = plt.subplot(122)

# ==================================================
# Atom Panel
# ==================================================

ax_atoms.set_xlim(0,14)
ax_atoms.set_ylim(0,10)

ax_atoms.set_aspect("equal")
ax_atoms.axis("off")

ax_atoms.set_title(
    "Blockade Radius Scan"
)

bond_line, = ax_atoms.plot(
    [],
    [],
    linewidth=2
)

atomA = plt.Circle(
    (3,5),
    0.25
)

atomB = plt.Circle(
    (10,5),
    0.25
)

ax_atoms.add_patch(atomA)
ax_atoms.add_patch(atomB)

circleA = plt.Circle(
    (3,5),
    Rb,
    fill=False,
    linewidth=2
)

circleB = plt.Circle(
    (10,5),
    Rb,
    fill=False,
    linewidth=2
)

ax_atoms.add_patch(circleA)
ax_atoms.add_patch(circleB)

status_text = ax_atoms.text(
    1,
    1,
    "",
    fontsize=11,
    bbox=dict(boxstyle="round")
)

# ==================================================
# Curve Panel
# ==================================================

ax_curve.set_xlim(
    min(R_values),
    max(R_values)
)

ax_curve.set_ylim(
    0,
    1.05
)

ax_curve.grid()

ax_curve.set_xlabel(
    "Distance R"
)

ax_curve.set_ylabel(
    "max P(rr)"
)

ax_curve.set_title(
    "Maximum Double Excitation Probability"
)

curve_line, = ax_curve.plot(
    [],
    [],
    linewidth=3
)

current_point, = ax_curve.plot(
    [],
    [],
    'o',
    markersize=10
)

# Blockade Radius Marker

ax_curve.axvline(
    Rb,
    linestyle="--",
    linewidth=2
)

ax_curve.text(
    Rb,
    0.95,
    "Rb",
    ha='center'
)

# ==================================================
# Animation
# ==================================================

def update(frame):

    R = R_values[frame]

    V = C6/(R**6)

    Prr_max = Prr_max_list[frame]

    x1 = 3
    x2 = x1 + 0.7*R

    bond_line.set_data(
        [x1,x2],
        [5,5]
    )

    atomA.center = (x1,5)
    atomB.center = (x2,5)

    circleA.center = (x1,5)
    circleB.center = (x2,5)

    # Blockade check

    if abs(x2-x1) < 2*Rb:

        color = "red"
        status = "BLOCKADE ON"

    else:

        color = "green"
        status = "BLOCKADE OFF"

    circleA.set_edgecolor(color)
    circleB.set_edgecolor(color)

    # Atom color reflects blockade strength

    strength = 1-Prr_max

    atomA.set_facecolor(
        (strength,0,1-strength)
    )

    atomB.set_facecolor(
        (strength,0,1-strength)
    )

    # Curve

    curve_line.set_data(
        R_values[:frame+1],
        Prr_max_list[:frame+1]
    )

    current_point.set_data(
        [R],
        [Prr_max]
    )

    status_text.set_text(

        f"R = {R:.2f}\n\n"

        f"Rb = {Rb:.2f}\n\n"

        f"V = {V:.3f}\n\n"

        f"max P(rr) = {Prr_max:.3f}\n\n"

        f"{status}"
    )

    return (
        bond_line,
        curve_line,
        current_point
    )

ani = FuncAnimation(
    fig,
    update,
    frames=len(R_values),
    interval=80,
    blit=False
)

plt.tight_layout()
plt.show()