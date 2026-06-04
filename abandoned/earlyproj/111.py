import numpy as np
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from arc import *

############################################################
# PARAMETERS
############################################################

R_PAIR = 5.0      # um
NKEEP  = 8

GRID_N = 500
GRID_L = 25000    # a0

R2_FIXED = 9000.0 # a0

############################################################
# ATOM
############################################################

atom = Rubidium87()

############################################################
# PAIR STATE
############################################################

pair = PairStateInteractions(
    atom,
    70,0,0.5,
    70,0,0.5,
    0.5,0.5
)

print("Defining basis...")

pair.defineBasis(
    theta=0,
    phi=0,
    nRange=3,
    lrange=3,
    energyDelta=30e9,
    progressOutput=True
)

print("Basis size =", len(pair.basisStates))

############################################################
# BUILD INTERACTION MATRIX
############################################################

pair.diagonalise(
    np.array([R_PAIR]),
    20,
    progressOutput=True
)

############################################################
# FULL DIAGONALIZATION
############################################################

print("Diagonalising full matrix...")

evals, evecs = spla.eigsh(
    pair.matDiagonal,
    k=20,
    which="SA"
)

############################################################
# FIND EIGENSTATE CONNECTED TO 70S70S
############################################################

target = pair.originalPairStateIndex

overlaps = np.abs(
    evecs[target,:]
)**2

state_index = np.argmax(overlaps)

vec = evecs[:,state_index]

print()
print("Selected eigenstate =", state_index)

############################################################
# DOMINANT COMPONENTS
############################################################

weights = np.abs(vec)**2

idx = np.argsort(weights)[::-1]

print()
print("Largest components:")

for k in range(10):

    i = idx[k]

    print(
        k,
        weights[i],
        pair.basisStates[i]
    )

components = []

for k in range(NKEEP):

    i = idx[k]

    components.append(
        (
            vec[i],
            pair.basisStates[i]
        )
    )

############################################################
# GRID
############################################################

x = np.linspace(
    -GRID_L,
    GRID_L,
    GRID_N
)

y = np.linspace(
    -GRID_L,
    GRID_L,
    GRID_N
)

X,Y = np.meshgrid(x,y)

Rxy = np.sqrt(
    X**2 +
    Y**2
)

Theta = np.arctan2(
    Y,
    X
)

############################################################
# CONDITIONAL TWO-ELECTRON WAVEFUNCTION
############################################################

psi_total = np.zeros(
    (GRID_N,GRID_N),
    dtype=complex
)

eps = 1e-12

theta2 = 0.0

############################################################
# LOOP OVER PAIR COMPONENTS
############################################################

for amp,state in components:

    n1,l1,j1,m1,n2,l2,j2,m2 = state

    ########################################################
    # ELECTRON 1
    ########################################################

    wf1 = Wavefunction(
        atom,
        [[
            int(n1),
            int(l1),
            float(j1),
            float(m1)
        ]],
        [1.0]
    )

    radial1 = wf1.basisWavefunctions[0]

    ########################################################
    # ELECTRON 2
    ########################################################

    wf2 = Wavefunction(
        atom,
        [[
            int(n2),
            int(l2),
            float(j2),
            float(m2)
        ]],
        [1.0]
    )

    radial2 = wf2.basisWavefunctions[0]

    ########################################################
    # SECOND ELECTRON FIXED
    ########################################################

    u2 = radial2(R2_FIXED)

    psi2_fixed = (
        u2
        /
        np.sqrt(R2_FIXED**2 + eps)
    )

    ang2 = np.exp(
        1j*m2*theta2
    )

    coeff = (
        amp
        *
        psi2_fixed
        *
        ang2
    )

    ########################################################
    # FIRST ELECTRON MAP
    ########################################################

    u1 = radial1(Rxy)

    psi1 = (
        u1
        /
        np.sqrt(Rxy**2 + eps)
    )

    ang1 = np.exp(
        1j*m1*Theta
    )

    psi_total += (
        coeff
        *
        psi1
        *
        ang1
    )

############################################################
# NORMALIZE
############################################################

maxamp = np.max(
    np.abs(psi_total)
)

if maxamp > 0:
    psi_total /= maxamp

############################################################
# HSV
############################################################

density = np.abs(
    psi_total
)**2

phase = np.angle(
    psi_total
)

hue = (
    phase + np.pi
)/(2*np.pi)

sat = np.ones_like(
    hue
)

val = density**0.35

HSV = np.stack(
    [hue,sat,val],
    axis=-1
)

RGB = hsv_to_rgb(
    HSV
)

############################################################
# PLOT
############################################################

fig,ax = plt.subplots(
    figsize=(8,8)
)

ax.imshow(
    RGB,
    origin="lower",
    extent=[
        -GRID_L,
        GRID_L,
        -GRID_L,
        GRID_L
    ]
)

ax.set_title(
    "Conditional Two-Electron Wavefunction\n"
    f"Rpair={R_PAIR} um, "
    f"r2={R2_FIXED:.0f} a0"
)

ax.set_xlabel(
    "x (Bohr)"
)

ax.set_ylabel(
    "y (Bohr)"
)

plt.tight_layout()

#plt.savefig("conditional_pair_hsv.png",    dpi=300)

plt.show()

print()
print("saved: conditional_pair_hsv.png")