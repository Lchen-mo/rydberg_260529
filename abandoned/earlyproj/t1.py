import numpy as np
import pyvista as pv

from qutip import *
from arc import *

###########################################################
# PARAMETERS
###########################################################

Omega = 2*np.pi*5.0      # MHz
Delta = 0.0

C6 = 8.6e5              # MHz um^6
Rblock_um = 8.0

Vrr = C6/(Rblock_um**6)

###########################################################
# QUTIP BLOCKADE
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

H = (
    Omega/2*(sx1+sx2)
    - Delta*(n1+n2)
    + Vrr*n1*n2
)

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
# CHOOSE TIME
###########################################################

k = 120

state = result.states[k]

vec = state.full().flatten()

cgg = vec[0]
cgr = vec[1]
crg = vec[2]
crr = vec[3]

print("time =",tlist[k])

print("cgg =",cgg)
print("cgr =",cgr)
print("crg =",crg)
print("crr =",crr)

###########################################################
# ARC 70S WAVEFUNCTION
###########################################################

atom = Rubidium87()

wf = Wavefunction(
    atom,
    [[70,0,0.5,0.5]],
    [1.0]
)

radial = wf.basisWavefunctions[0]

###########################################################
# DISPLAY SCALE
#
# 实际8um太大
# 为了显示轨道结构缩小
###########################################################

Rsep_display = 30000.0     # Bohr

###########################################################
# GRID
###########################################################

L = 45000

N = 80

x = np.linspace(-L,L,N)
y = np.linspace(-L,L,N)
z = np.linspace(-L,L,N)

X,Y,Z = np.meshgrid(
    x,
    y,
    z,
    indexing="ij"
)

###########################################################
# DISTANCES
###########################################################

RA = np.sqrt(
    (X+Rsep_display/2)**2 +
    Y**2 +
    Z**2
)

RB = np.sqrt(
    (X-Rsep_display/2)**2 +
    Y**2 +
    Z**2
)

###########################################################
# TRUE ARC DENSITY
###########################################################

eps = 1e-12

uA = radial(RA)
uB = radial(RB)

rhoA = np.abs(uA)**2/(RA**2+eps)
rhoB = np.abs(uB)**2/(RB**2+eps)

###########################################################
# POPULATIONS
###########################################################

Pgg = np.abs(cgg)**2
Pgr = np.abs(cgr)**2
Prg = np.abs(crg)**2
Prr = np.abs(crr)**2

print()

print("Pgg =",Pgg)
print("Pgr =",Pgr)
print("Prg =",Prg)
print("Prr =",Prr)

###########################################################
# BLOCKADE DENSITY
###########################################################

rho_total = (
      Pgr*rhoB
    + Prg*rhoA
    + Prr*(rhoA+rhoB)
)

rho_total /= rho_total.max()

###########################################################
# PYVISTA
###########################################################

grid = pv.StructuredGrid(
    X,
    Y,
    Z
)

grid["rho"] = rho_total.ravel(order="F")

###########################################################
# CONTOUR
###########################################################

contours = grid.contour(
    isosurfaces=[
        0.02,
        0.05,
        0.10
    ],
    scalars="rho"
)

###########################################################
# VISUALIZATION
###########################################################

plotter = pv.Plotter()

plotter.add_mesh(
    contours,
    opacity=0.35
)

plotter.add_axes()

plotter.show_grid()

plotter.add_text(
    f"t={tlist[k]:.3f} us",
    font_size=14
)

###########################################################
# NUCLEI
###########################################################

plotter.add_points(
    np.array([
        [-Rsep_display/2,0,0],
        [ Rsep_display/2,0,0]
    ]),
    render_points_as_spheres=True,
    point_size=20
)

plotter.show()