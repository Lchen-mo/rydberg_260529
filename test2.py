import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import pyvista as pv
import matplotlib.pyplot as plt

from qutip import *
from arc import *

############################################################
# PARAMETERS
############################################################

Omega = 2*np.pi*5.0      # MHz
Delta = 0.0

C6 = 8.6e5              # MHz um^6

R_um = 3.0              # REAL DISTANCE

Vrr = C6/(R_um**6)

print("Omega =",Omega)
print("Vrr   =",Vrr)

############################################################
# BASIS
############################################################

g = basis(2,0)
r = basis(2,1)

I = qeye(2)

sx = sigmax()

nr = r*r.dag()

sx1 = tensor(sx,I)
sx2 = tensor(I,sx)

n1 = tensor(nr,I)
n2 = tensor(I,nr)

############################################################
# HAMILTONIAN
############################################################

H = (
      Omega/2*(sx1+sx2)
    - Delta*(n1+n2)
    + Vrr*n1*n2
)

############################################################
# TIME EVOLUTION
############################################################

psi0 = tensor(g,g)

tlist = np.linspace(
    0,
    1.0,
    120
)

result = sesolve(
    H,
    psi0,
    tlist
)

############################################################
# POPULATIONS
############################################################

Pgg = []
Pgr = []
Prg = []
Prr = []

for st in result.states:

    vec = st.full().flatten()

    cgg,cgr,crg,crr = vec

    Pgg.append(abs(cgg)**2)
    Pgr.append(abs(cgr)**2)
    Prg.append(abs(crg)**2)
    Prr.append(abs(crr)**2)

Pgg = np.array(Pgg)
Pgr = np.array(Pgr)
Prg = np.array(Prg)
Prr = np.array(Prr)

############################################################
# SAVE POPULATION FIGURE
############################################################

plt.figure(figsize=(8,5))

plt.plot(tlist,Pgg,label="|gg>")
plt.plot(tlist,Pgr,label="|gr>")
plt.plot(tlist,Prg,label="|rg>")
plt.plot(tlist,Prr,label="|rr>")

plt.xlabel("Time (us)")
plt.ylabel("Population")

plt.legend()

plt.tight_layout()

plt.savefig(
    "rydberg_populations.png",
    dpi=300
)

plt.close()

############################################################
# ARC 70S
############################################################

atom = Rubidium87()

wf = Wavefunction(
    atom,
    [[70,0,0.5,0.5]],
    [1.0]
)

radial = wf.basisWavefunctions[0]

############################################################
# REAL DISTANCE
############################################################

bohr_to_nm = 0.0529177

Rsep = (
    8000.0 /
    bohr_to_nm
)

print("Rsep =",Rsep,"Bohr")

############################################################
# GRID
############################################################

L = Rsep/2 + 20000

N = 60

x = np.linspace(-L,L,N)
y = np.linspace(-25000,25000,N)
z = np.linspace(-25000,25000,N)

X,Y,Z = np.meshgrid(
    x,y,z,
    indexing="ij"
)

############################################################
# DISTANCES
############################################################

RA = np.sqrt(
    (X+Rsep/2)**2 +
    Y**2 +
    Z**2
)

RB = np.sqrt(
    (X-Rsep/2)**2 +
    Y**2 +
    Z**2
)

############################################################
# TRUE ARC DENSITY
############################################################

eps = 1e-12

uA = radial(RA)
uB = radial(RB)

rhoA = np.abs(uA)**2/(RA**2+eps)
rhoB = np.abs(uB)**2/(RB**2+eps)

rhoA /= rhoA.max()
rhoB /= rhoB.max()

############################################################
# GLOBAL MAX
############################################################

global_max = 0.0

for k in range(len(tlist)):

    rho = (
          Pgr[k]*rhoB
        + Prg[k]*rhoA
        + Prr[k]*(rhoA+rhoB)
    )

    global_max = max(
        global_max,
        rho.max()
    )

############################################################
# GIF
############################################################

pv.global_theme.allow_empty_mesh = True

plotter = pv.Plotter(
    off_screen=True
)

plotter.open_gif(
    "rydberg_blockade_real.gif"
)

############################################################
# ANIMATION
############################################################

for k in range(len(tlist)):

    rho = (
          Pgr[k]*rhoB
        + Prg[k]*rhoA
        + Prr[k]*(rhoA+rhoB)
    )

    rho /= global_max

    grid = pv.StructuredGrid(
        X,Y,Z
    )

    grid["rho"] = rho.ravel(order="F")

    contour = grid.contour(
        isosurfaces=[
            0.05,
            0.10,
            0.20
        ],
        scalars="rho"
    )

    plotter.clear()

    plotter.add_axes()

    plotter.add_points(
        np.array([
            [-Rsep/2,0,0],
            [ Rsep/2,0,0]
        ]),
        render_points_as_spheres=True,
        point_size=15
    )

    if contour.n_points > 0:

        plotter.add_mesh(
            contour,
            opacity=0.4
        )

    plotter.add_text(
        f"t = {tlist[k]:.3f} us",
        font_size=12
    )

    plotter.write_frame()

    print(
        f"{k+1}/{len(tlist)}"
    )

plotter.close()

print()
print("saved:")
print("rydberg_blockade_real.gif")
print("rydberg_populations.png")