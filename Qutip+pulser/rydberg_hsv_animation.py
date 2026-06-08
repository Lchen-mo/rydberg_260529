import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import hsv_to_rgb

from qutip import *
from arc import *

###########################################################
# QUTIP PARAMETERS
###########################################################

Omega = 2*np.pi*2.0      # MHz

Delta = 0.0

C6 = 8.6e5

R_um = 3.0

# 注意统一成角频率
Vrr = 2*np.pi*C6/(R_um**6)

print("Omega =", Omega)
print("Vrr   =", Vrr)
print("Vrr/Omega =", Vrr/Omega)

###########################################################
# TWO-ATOM BASIS
###########################################################

g = basis(2,0)
r = basis(2,1)

I = qeye(2)

sx = sigmax()

nr = r*r.dag()

sx1 = tensor(sx,I)
sx2 = tensor(I,sx)

n1 = tensor(nr,I)
n2 = tensor(I,nr)

###########################################################
# HAMILTONIAN
###########################################################

H = (
      Omega/2*(sx1+sx2)
    - Delta*(n1+n2)
    + Vrr*n1*n2
)

###########################################################
# EVOLUTION
###########################################################

psi0 = tensor(g,g)

tlist = np.linspace(
    0,
    1.0,
    300
)

result = sesolve(
    H,
    psi0,
    tlist
)

###########################################################
# ARC 70S
###########################################################

atom = Rubidium87()

wf = Wavefunction(
    atom,
    [[70,0,0.5,0.5]],
    [1.0]
)

radial = wf.basisWavefunctions[0]

###########################################################
# DISPLAY GEOMETRY
###########################################################

# 为了显示W态干涉
# 不是实验真实距离
Rsep = 12000.0  # Bohr

###########################################################
# XY PLANE
###########################################################

L = 25000

N = 500

x = np.linspace(-L,L,N)
y = np.linspace(-L,L,N)

X,Y = np.meshgrid(x,y)

###########################################################
# DISTANCES
###########################################################

RA = np.sqrt(
    (X + Rsep/2)**2 +
    Y**2
)

RB = np.sqrt(
    (X - Rsep/2)**2 +
    Y**2
)

###########################################################
# 70S ORBITALS
###########################################################

eps = 1e-12

uA = radial(RA)
uB = radial(RB)

psiA = uA/np.sqrt(RA**2 + eps)
psiB = uB/np.sqrt(RB**2 + eps)

###########################################################
# NORMALIZATION
###########################################################

psiA = psiA/np.max(np.abs(psiA))
psiB = psiB/np.max(np.abs(psiB))

###########################################################
# FIGURE
###########################################################

fig,ax = plt.subplots(
    figsize=(8,8)
)

img = ax.imshow(
    np.zeros((N,N,3)),
    origin="lower",
    extent=[-L,L,-L,L]
)

title = ax.set_title("")

ax.set_xlabel("x (Bohr)")
ax.set_ylabel("y (Bohr)")

###########################################################
# HSV FUNCTION
###########################################################

def wavefunction_to_rgb(psi):

    density = np.abs(psi)**2

    if density.max() > 0:
        density = density/density.max()

    phase = np.angle(psi)

    hue = (phase + np.pi)/(2*np.pi)

    saturation = np.ones_like(hue)

    value = density**0.35

    hsv = np.stack(
        [hue,saturation,value],
        axis=-1
    )

    rgb = hsv_to_rgb(hsv)

    return rgb

###########################################################
# UPDATE
###########################################################

def update(frame):

    vec = result.states[frame].full().flatten()

    cgg,cgr,crg,crr = vec

    #######################################################
    # WAVEFUNCTION
    #######################################################

    psi = (
          cgr*psiA
        + crg*psiB
    )

    #######################################################
    # HSV
    #######################################################

    rgb = wavefunction_to_rgb(psi)

    img.set_data(rgb)

    title.set_text(
        (
            f"t={tlist[frame]:.3f} us\n"
            f"Pgr={abs(cgr)**2:.3f}  "
            f"Prg={abs(crg)**2:.3f}  "
            f"Prr={abs(crr)**2:.5f}"
        )
    )

    return [img,title]

###########################################################
# ANIMATION
###########################################################

ani = FuncAnimation(
    fig,
    update,
    frames=len(tlist),
    interval=40,
    blit=True
)

###########################################################
# SAVE GIF
###########################################################

ani.save(
    "rydberg_hsv.gif",
    writer=PillowWriter(fps=25)
)

print()
print("saved:")
print("rydberg_hsv.gif")