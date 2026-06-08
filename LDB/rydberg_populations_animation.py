import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
from qutip import *

#################################################
# PARAMETERS
#################################################

Omega = 2*np.pi*5.0

Delta = 0.0

C6 = 8.6e5

R_um = 4.0

Vrr = C6/(R_um**6)

print("Omega =", Omega)
print("Vrr   =", Vrr)

#################################################
# BASIS
#################################################

g = basis(2,0)
r = basis(2,1)

I = qeye(2)

sx = sigmax()

nr = r*r.dag()

sx1 = tensor(sx,I)
sx2 = tensor(I,sx)

n1 = tensor(nr,I)
n2 = tensor(I,nr)

#################################################
# HAMILTONIAN
#################################################

H = (
      Omega/2*(sx1+sx2)
    - Delta*(n1+n2)
    + Vrr*n1*n2
)

#################################################
# TIME EVOLUTION
#################################################

psi0 = tensor(g,g)

tlist = np.linspace(
    0,
    1.0,
    500
)

result = sesolve(
    H,
    psi0,
    tlist
)

#################################################
# POPULATIONS
#################################################

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

#################################################
# FIGURE
#################################################

fig,ax = plt.subplots(
    figsize=(8,5)
)

ax.set_xlim(
    tlist[0],
    tlist[-1]
)

ax.set_ylim(
    0,
    1.05
)

ax.set_xlabel(
    "Time (us)"
)

ax.set_ylabel(
    "Population"
)

ax.set_title(
    "双原子里德堡态布居演化"
)

line_gg, = ax.plot(
    [],
    [],
    label="|gg>"
)

line_gr, = ax.plot(
    [],
    [],
    label="|gr>"
)

line_rg, = ax.plot(
    [],
    [],
    label="|rg>"
)

line_rr, = ax.plot(
    [],
    [],
    label="|rr>"
)

ax.legend()

#################################################
# UPDATE
#################################################

def update(frame):

    line_gg.set_data(
        tlist[:frame],
        Pgg[:frame]
    )

    line_gr.set_data(
        tlist[:frame],
        Pgr[:frame]
    )

    line_rg.set_data(
        tlist[:frame],
        Prg[:frame]
    )

    line_rr.set_data(
        tlist[:frame],
        Prr[:frame]
    )

    return (
        line_gg,
        line_gr,
        line_rg,
        line_rr
    )

#################################################
# ANIMATION
#################################################

ani = FuncAnimation(
    fig,
    update,
    frames=len(tlist),
    interval=20,
    blit=True
)

ani.save(
    "rydberg_blockade_populations.gif",
    writer=PillowWriter(
        fps=25
    )
)

print()
print("saved:")
print("rydberg_blockade_populations.gif")

plt.figure(figsize=(8,5))

plt.plot(
    tlist,
    Pgr+Prg,
    label="single excitation"
)

plt.plot(
    tlist,
    Prr,
    label="double excitation"
)

plt.xlabel("Time (us)")
plt.ylabel("Population")

plt.legend()


plt.show()

#################################################
# AREA-REPRESENTATION GIF
#################################################

from matplotlib.patches import Rectangle

fig2, ax2 = plt.subplots(figsize=(6,6))

ax2.set_xlim(0, 2)
ax2.set_ylim(0, 2)

ax2.set_aspect('equal')

ax2.set_xticks([])
ax2.set_yticks([])

ax2.set_title("各态布居数演化")

# 四个象限边框
ax2.plot([1,1],[0,2],'k--',lw=1)
ax2.plot([0,2],[1,1],'k--',lw=1)

# 固定宽度
w = 0.8

# 初始矩形
rect_gg = Rectangle(
    (0.1,1.0),
    w,
    Pgg[0],
    color="tab:blue",
    alpha=0.8
)

rect_gr = Rectangle(
    (1.1,1.0),
    w,
    Pgr[0],
    color="tab:orange",
    alpha=0.8
)

rect_rg = Rectangle(
    (0.1,0.0),
    w,
    Prg[0],
    color="tab:green",
    alpha=0.8
)

rect_rr = Rectangle(
    (1.1,0.0),
    w,
    Prr[0],
    color="tab:red",
    alpha=0.8
)

ax2.add_patch(rect_gg)
ax2.add_patch(rect_gr)
ax2.add_patch(rect_rg)
ax2.add_patch(rect_rr)

# 标签
txt_gg = ax2.text(
    0.5,
    1.95,
    "|gg>",
    ha='center'
)

txt_gr = ax2.text(
    1.5,
    1.95,
    "|gr>",
    ha='center'
)

txt_rg = ax2.text(
    0.5,
    0.95,
    "|rg>",
    ha='center'
)

txt_rr = ax2.text(
    1.5,
    0.95,
    "|rr>",
    ha='center'
)

#################################################
# UPDATE
#################################################

def update_area(frame):

    rect_gg.set_height(Pgg[frame])
    rect_gr.set_height(Pgr[frame])
    rect_rg.set_height(Prg[frame])
    rect_rr.set_height(Prr[frame])

    return (
        rect_gg,
        rect_gr,
        rect_rg,
        rect_rr
    )

#################################################
# ANIMATION
#################################################

ani2 = FuncAnimation(
    fig2,
    update_area,
    frames=len(tlist),
    interval=20,
    blit=True
)

ani2.save(
    "rydberg_blockade_area.gif",
    writer=PillowWriter(fps=25)
)

print()
print("saved:")
print("rydberg_blockade_area.gif")