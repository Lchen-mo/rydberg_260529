import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from qutip import *

# ==========================
# system size (2–5 atoms)
# ==========================

N = 4  # 改成 2,3,4,5 都可以

# ==========================
# operators
# ==========================

sx = sigmax()
sz = sigmaz()
sp = sigmap()
sm = sigmam()

I = qeye(2)

def op_on(op, i, N):

    ops = []
    for j in range(N):
        ops.append(I)

    ops[i] = op

    return tensor(ops)

# ==========================
# parameters
# ==========================

Omega = 1.0   # Rabi frequency
Delta = 0.0   # detuning

V0 = 8.0      # blockade strength

# atom positions (1D / 2D can extend)
pos = np.arange(N)

def V(i, j):
    if i == j:
        return 0
    return V0 / abs(i - j)**6

# ==========================
# Hamiltonian construction
# ==========================

H = 0

# Rabi drive
for i in range(N):
    H += (Omega/2) * op_on(sx, i, N)

# detuning
for i in range(N):
    H += (Delta/2) * op_on(sz, i, N)

# blockade interaction
for i in range(N):
    for j in range(i+1, N):

        n_i = op_on(sp*sm, i, N)
        n_j = op_on(sp*sm, j, N)

        H += V(i,j) * n_i * n_j

# ==========================
# initial state |000...0>
# ==========================

psi0 = tensor([basis(2,0) for _ in range(N)])

# ==========================
# time evolution
# ==========================

tlist = np.linspace(0, 5, 200)

result = mesolve(
    H,
    psi0,
    tlist,
    c_ops=[],
    e_ops=[]
)

# ==========================
# observables
# ==========================

def excitation_prob(state, i):

    n_i = op_on(sp*sm, i, N)

    return expect(n_i, state)

# ==========================
# plot
# ==========================

plt.figure(figsize=(8,5))

for i in range(N):

    pops = [excitation_prob(state, i) for state in result.states]

    plt.plot(tlist, pops, label=f"Atom {i}")

plt.xlabel("time")
plt.ylabel("excitation probability")
plt.title(f"Rydberg Blockade Dynamics (N={N})")
plt.legend()
plt.show()