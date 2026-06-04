#atom_parameters.py
import numpy as np
from arc import Rubidium87

# ----------------------------
# 基本物理常数
# ----------------------------

hbar = 1.0545718e-34
c = 2.99792458e8
epsilon0 = 8.854187817e-12
amu = 1.66053906660e-27

# 原子位置
r1_0 = 1e-6      # atom1 初始位置
r2_0 = 1e-6      # atom2 初始位置
R_atoms=1e-6     # 原子间距离

sigma = 0.2e-8    # 波包宽度

# ----------------------------
# Rb-87
# ----------------------------

Rb = Rubidium87()

mRb = 86.9091805 * amu

# ----------------------------
# Rydberg态
# ----------------------------

rydberg_n = 70
rydberg_l = 0
rydberg_j = 0.5

# ----------------------------
# C6 系数
# ----------------------------

# 推荐先手动设置
# 70S Rb87 量级约:
# GHz μm^6

C6_GHz_um6 = 860.0

# 转 SI:
# Hz * m^6

C6 = C6_GHz_um6 * 1e9 * (1e-6)**6

# ----------------------------
# 激光参数
# ----------------------------

Omega = 2 * np.pi * 1e7
Delta = 0.0

# ----------------------------
# blockade radius
# ----------------------------

R_blockade = (C6 / (Omega/(2*np.pi)))**(1/6)

# ----------------------------
# 网格
# ----------------------------

r_min = 1e-7
r_max = 2e-6

Nr = 100

r_grid = np.linspace(r_min, r_max, Nr)

dr = r_grid[1] - r_grid[0]

# ----------------------------
# internal channels
# ----------------------------

internal_states = [
    'gg',
    'gr',
    'rg',
    'rr'
]

n_channels = len(internal_states)

# ----------------------------
# parameter dict
# ----------------------------

params = {

    'hbar': hbar,
    'mRb': mRb,

    'r1_0': r1_0,
    'r2_0': r2_0,

    'R_atoms': R_atoms,

    'sigma': sigma,

    'rydberg_n': rydberg_n,
    'rydberg_l': rydberg_l,
    'rydberg_j': rydberg_j,

    'C6': C6,

    'Omega': Omega,
    'Delta': Delta,

    'R_blockade': R_blockade,

    'r_grid': r_grid,
    'dr': dr,

    'internal_states': internal_states,
    'n_channels': n_channels
}