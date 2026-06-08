import numpy as np
import pyvista as pv

from qutip import *
from arc import *

#########################################################
# QUTIP
#########################################################

Omega = 2*np.pi*5.0

Delta = 0.0

C6 = 8.6e5

R_um = 4.0

Vrr = C6/(R_um**6)

print("Omega =", Omega)
print("Vrr   =", Vrr)

#########################################################
# BASIS
#########################################################

g = basis(2,0)
r = basis(2,1)

I = qeye(2)

sx = sigmax()

nr = r*r.dag()

sx1 = tensor(sx,I)
sx2 = tensor(I,sx)

n1 = tensor(nr,I)
n2 = tensor(I,nr)

#########################################################
# HAMILTONIAN
#########################################################

H = (
      Omega/2*(sx1+sx2)
    - Delta*(n1+n2)
    + Vrr*n1*n2
)

#########################################################
# EVOLUTION
#########################################################

psi0 = tensor(g,g)

tlist = np.linspace(
    0,
    1.0,
    100
)

result = sesolve(
    H,
    psi0,
    tlist
)

#########################################################
# FIND MAX RYDBERG POPULATION
#########################################################

ryd_pop = []

for st in result.states:

    vec = st.full().flatten()

    cgg,cgr,crg,crr = vec

    pop = (
        abs(cgr)**2
        + abs(crg)**2
        + abs(crr)**2
    )

    ryd_pop.append(pop)

kmax = np.argmax(ryd_pop)

print("best frame =",kmax)
print("max pop =",ryd_pop[kmax])

#########################################################
# ARC 70S
#########################################################

atom = Rubidium87()

wf = Wavefunction(
    atom,
    [[70,0,0.5,0.5]],
    [1.0]
)

radial = wf.basisWavefunctions[0]

#########################################################
# DISPLAY GEOMETRY
#########################################################

Rsep = 30000.0

#########################################################
# GRID
#########################################################

L = 45000

N = 60

x = np.linspace(-L,L,N)
y = np.linspace(-L,L,N)
z = np.linspace(-L,L,N)

X,Y,Z = np.meshgrid(
    x,
    y,
    z,
    indexing="ij"
)

#########################################################
# DISTANCES
#########################################################

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

#########################################################
# TRUE ARC DENSITY
#########################################################

eps = 1e-12

uA = radial(RA)
uB = radial(RB)

rhoA = np.abs(uA)**2/(RA**2+eps)
rhoB = np.abs(uB)**2/(RB**2+eps)

#########################################################
# INITIAL FRAME
#########################################################

vec = result.states[kmax].full().flatten()

cgg,cgr,crg,crr = vec

rho = (
      abs(cgr)**2*rhoB
    + abs(crg)**2*rhoA
    + abs(crr)**2*(rhoA+rhoB)
)

mx = rho.max()

if mx <= 0:
    raise RuntimeError(
        "Density vanished."
    )

rho /= mx

#########################################################
# GRID
#########################################################

grid = pv.StructuredGrid(
    X,
    Y,
    Z
)

grid["rho"] = rho.ravel(order="F")

#########################################################
# FIRST CONTOUR
#########################################################

contours = grid.contour(
    isosurfaces=[
        0.1,
        0.3,
        0.5
    ],
    scalars="rho"
)

if contours.n_points == 0:
    raise RuntimeError(
        "Initial contour empty."
    )

#########################################################
# PLOTTER
#########################################################

pv.global_theme.allow_empty_mesh = True

plotter = pv.Plotter(
    off_screen=True
)

plotter.open_gif(
    "rydberg_blockade.gif"
)

#########################################################
# STATIC OBJECTS
#########################################################

plotter.add_axes()

plotter.add_points(
    np.array([
        [-Rsep/2,0,0],
        [ Rsep/2,0,0]
    ]),
    render_points_as_spheres=True,
    point_size=20
)

#########################################################
# ANIMATION
#########################################################

for k in range(len(tlist)):

    vec = result.states[k].full().flatten()

    cgg,cgr,crg,crr = vec

    rho = (
          abs(cgr)**2*rhoB
        + abs(crg)**2*rhoA
        + abs(crr)**2*(rhoA+rhoB)
    )

    mx = rho.max()

    if mx < 1e-20:

        continue

    rho /= mx

    grid["rho"] = rho.ravel(order="F")

    contour = grid.contour(
        isosurfaces=[
            0.1,
            0.3,
            0.5
        ],
        scalars="rho"
    )

    if contour.n_points == 0:

        continue

    plotter.clear()

    plotter.add_axes()

    plotter.add_points(
        np.array([
            [-Rsep/2,0,0],
            [ Rsep/2,0,0]
        ]),
        render_points_as_spheres=True,
        point_size=20
    )

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

#########################################################
# SAVE
#########################################################

plotter.close()

print()
print("saved:")
print("rydberg_blockade.gif")