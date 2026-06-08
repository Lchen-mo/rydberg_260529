import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import scipy.sparse.linalg as spla

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

from arc import *

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

############################################################
# BASIS
############################################################

print("Defining basis...")

pair.defineBasis(
    theta=0,
    phi=0,
    nRange=3,
    lrange=3,
    energyDelta=30e9,
    progressOutput=True
)

print()
print("Basis size =", len(pair.basisStates))

############################################################
# BUILD HAMILTONIAN
############################################################

R = np.array([5.0])  # um

print()
print("Building pair Hamiltonian...")

pair.diagonalise(
    R,
    20,
    progressOutput=True
)

############################################################
# FULL MATRIX
############################################################

Hpair = pair.matDiagonal

print()
print("Diagonalising full matrix...")

evals,evecs = spla.eigsh(
    Hpair,
    k=20,
    which="SA"
)

print("evecs shape =", evecs.shape)

############################################################
# FIND STATE CLOSEST TO |70S,70S>
############################################################

target = pair.originalPairStateIndex

overlaps = np.abs(
    evecs[target,:]
)**2

state_index = np.argmax(overlaps)

print()
print("Best eigenstate =", state_index)

vec = evecs[:,state_index]

############################################################
# DOMINANT COMPONENTS
############################################################

weights = np.abs(vec)**2

idx = np.argsort(weights)[::-1]

print()
print("Largest components:")
print()

for k in range(15):

    i = idx[k]

    print(
        k,
        weights[i],
        pair.basisStates[i]
    )

############################################################
# KEEP MAIN COMPONENTS
############################################################

Nkeep = 8

components = []

for k in range(Nkeep):

    i = idx[k]

    amp = vec[i]

    state = pair.basisStates[i]

    components.append(
        (amp,state)
    )

############################################################
# GRID
############################################################

L = 25000

N = 600

x = np.linspace(-L,L,N)
y = np.linspace(-L,L,N)

X,Y = np.meshgrid(x,y)

Rxy = np.sqrt(
    X**2 +
    Y**2
)

Theta = np.arctan2(Y,X)

eps = 1e-12

############################################################
# BUILD TOTAL WAVEFUNCTION
############################################################

psi_total = np.zeros(
    (N,N),
    dtype=complex
)

############################################################
# LOOP OVER COMPONENTS
############################################################

for amp,state in components:

    ########################################################
    # ARC BASIS STATE FORMAT
    ########################################################

    n1,l1,j1,m1,n2,l2,j2,m2 = state

    ########################################################
    # FIRST ATOM ORBITAL
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
    # SECOND ATOM ORBITAL
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
    # RADIAL PART
    ########################################################

    u1 = radial1(Rxy)
    u2 = radial2(Rxy)

    psi1 = u1/np.sqrt(Rxy**2 + eps)
    psi2 = u2/np.sqrt(Rxy**2 + eps)

    ########################################################
    # SIMPLE ANGULAR STRUCTURE
    ########################################################

    ang1 = np.exp(1j*m1*Theta)
    ang2 = np.exp(1j*m2*Theta)

    ########################################################
    # TOTAL CONTRIBUTION
    ########################################################

    psi_component = (
        psi1*ang1
        +
        psi2*ang2
    )

    psi_total += amp*psi_component

############################################################
# NORMALIZE
############################################################

psi_total /= np.max(
    np.abs(psi_total)
)

############################################################
# HSV MAP
############################################################

density = np.abs(psi_total)**2

phase = np.angle(psi_total)

############################################################
# HSV CHANNELS
############################################################

hue = (
    phase + np.pi
)/(2*np.pi)

saturation = np.ones_like(hue)

value = density**0.35

HSV = np.stack(
    [hue,saturation,value],
    axis=-1
)

RGB = hsv_to_rgb(HSV)

############################################################
# PLOT
############################################################

fig,ax = plt.subplots(
    figsize=(8,8)
)

img = ax.imshow(
    RGB,
    origin="lower",
    extent=[-L,L,-L,L]
)

ax.set_title(
    "Pair-State Mixed Wavefunction"
)

ax.set_xlabel(
    "x (Bohr)"
)

ax.set_ylabel(
    "y (Bohr)"
)

############################################################
# SAVE
############################################################

plt.tight_layout()

plt.savefig(
    "pair_state_hsv.png",
    dpi=300
)

plt.show()

print()
print("saved:")
print("pair_state_hsv.png")