import numpy as np
import matplotlib.pyplot as plt
from qutip import *

# =========================
Omega = 2 * np.pi * 1.0   # Rabi frequency
t_max = 5
n_steps = 1000
times = np.linspace(0, t_max, n_steps)

g = basis(2, 0)          # |g>
r = basis(2, 1)          # |r>
psi0 = g                 # 初始态

# =========================
H = 0.5 * Omega * sigmax()

#用 sesolve + store_states（无耗散标准做法）
result = sesolve(H, psi0, times, e_ops=[])


# 手动计算概率（确保 result.states 非空）
P_g = [abs(g.overlap(state))**2 for state in result.states]
P_r = [abs(r.overlap(state))**2 for state in result.states]



# =========================
plt.figure(figsize=(8,5))
plt.plot(times, P_g, label='Ground State |g>')
plt.plot(times, P_r, label='Rydberg State |r>')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Single Atom Rabi Oscillation')
plt.legend()
plt.grid()
plt.show()