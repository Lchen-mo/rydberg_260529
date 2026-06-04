import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
from scipy.linalg import expm

# -----------------------------
# 二原子参数
# -----------------------------
Omega = 2 * np.pi * 1.0  # Rabi 频率 (MHz)
V = 10 * Omega           # 阻塞相互作用 (MHz)
dt = 0.01                # 时间步
t_max = 3                # 总时间
times = np.arange(0, t_max, dt)

# 阻塞半径示意
R_block = 1.0

# 原子位置
atom_pos = np.array([[0, 0], [2, 0]])  # 两个原子相距 2 μm

# -----------------------------
# Hilbert space: |gg>, |gr>, |rg>, |rr>
# -----------------------------
psi = np.array([1, 0, 0, 0], dtype=complex)  # 初始态 |gg>

# 存储激发概率和相位
P_r = np.zeros((len(times), 2))
phase_r = np.zeros((len(times), 2))

# -----------------------------
# 双原子哈密顿量
# -----------------------------
def H_two_atom():
    """
    双原子哈密顿量
    |gg>, |gr>, |rg>, |rr>
    """
    H = np.zeros((4, 4), dtype=complex)
    # 原子0 Rabi
    H[0,2] = H[2,0] = Omega/2
    H[1,3] = H[3,1] = Omega/2
    # 原子1 Rabi
    H[0,1] = H[1,0] = Omega/2
    H[2,3] = H[3,2] = Omega/2
    # 阻塞项 |rr> 能量偏移
    H[3,3] = V
    return H

# -----------------------------
# 时间演化
# -----------------------------
for idx in range(len(times)):
    H = H_two_atom()
    U = expm(-1j * H * dt)  # 正确的矩阵指数
    psi = U @ psi
    # 激发概率
    P_r[idx,0] = np.abs(psi[2])**2 + np.abs(psi[3])**2  # 原子0
    P_r[idx,1] = np.abs(psi[1])**2 + np.abs(psi[3])**2  # 原子1
    # 相位
    phase_r[idx,0] = np.angle(psi[2]+psi[3]+1e-12)
    phase_r[idx,1] = np.angle(psi[1]+psi[3]+1e-12)

# -----------------------------
# 动画可视化
# -----------------------------
fig, ax = plt.subplots(figsize=(6,3))
ax.set_xlim(-1,3)
ax.set_ylim(-1,1)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Two-Atom Rydberg Blockade Simulation")

# 阻塞半径
for pos in atom_pos:
    ax.add_patch(Circle(pos, R_block, color='red', alpha=0.1))

# 原子 scatter
scat = ax.scatter(atom_pos[:,0], atom_pos[:,1], s=500, c=['blue','blue'])

# 动画更新函数
def update(frame):
    colors = plt.cm.twilight_shifted((phase_r[frame]+np.pi)/(2*np.pi))
    sizes = 50000 * np.sqrt(P_r[frame]) + 50  # 最小点大小
    scat.set_sizes(sizes)
    scat.set_color(colors)
    return scat,

ani = FuncAnimation(fig, update, frames=len(times), interval=30, blit=True)

plt.show()
# 如果想保存：
# ani.save("rydberg_blockade.gif", writer='pillow', dpi=150)