# internal_hamiltonian.py
import numpy as np
from atom_parameters import params

# ----------------------------
# 参数
# ----------------------------
Omega = params['Omega']       # Rabi频率
Delta = params['Delta']       # 激光失谐
C6 = params['C6']             # Rydberg-Rydberg vdW系数
r_grid = params['r_grid']     # 径向网格
internal_states = params['internal_states']
n_channels = params['n_channels']

# ----------------------------
# Rydberg-Rydberg相互作用势
# ----------------------------
def V_rr(r):
    """双激发态 van der Waals 势"""
    r_safe = np.where(r < 1e-9, 1e-9, r)
    return C6 / r_safe**6

# ----------------------------
# 单个r点内部态哈密顿量
# ----------------------------
def internal_hamiltonian_r(r):
    """
    返回单个r点的多通道内部态哈密顿量 (n_channels x n_channels)
    通道顺序由 internal_states 决定
    """
    H = np.zeros((n_channels, n_channels), dtype=np.complex128)

    # 索引
    idx_gg = internal_states.index('gg')
    idx_gr = internal_states.index('gr')
    idx_rg = internal_states.index('rg')
    idx_rr = internal_states.index('rr')

    # 激光耦合
    H[idx_gg, idx_gr] = Omega / 2
    H[idx_gr, idx_gg] = Omega / 2
    H[idx_gg, idx_rg] = Omega / 2
    H[idx_rg, idx_gg] = Omega / 2

    # 单激发 <-> 双激发
    H[idx_gr, idx_rr] = Omega / 2
    H[idx_rr, idx_gr] = Omega / 2
    H[idx_rg, idx_rr] = Omega / 2
    H[idx_rr, idx_rg] = Omega / 2

    # 激光失谐
    H[idx_gr, idx_gr] = -Delta
    H[idx_rg, idx_rg] = -Delta

    # 双激发态加上Rydberg-Rydberg相互作用
    H[idx_rr, idx_rr] = -2*Delta + V_rr(r)

    return H

# ----------------------------
# 径向网格多通道内部态哈密顿量
# ----------------------------
def internal_hamiltonian_grid(r_grid):
    """
    返回shape = (Nr, n_channels, n_channels) 的哈密顿量
    """
    Nr = len(r_grid)
    H_grid = np.zeros((Nr, n_channels, n_channels), dtype=np.complex128)
    for i, r in enumerate(r_grid):
        H_grid[i] = internal_hamiltonian_r(r)
    return H_grid

# ----------------------------
# 使用示例
# ----------------------------
if __name__ == "__main__":
    H0 = internal_hamiltonian_r(1e-6)  # r = 1 μm
    print("单个r点内部态哈密顿量 |rr> 态能量:", H0[-1,-1])

    H_grid = internal_hamiltonian_grid(r_grid)
    print("径向哈密顿量网格形状:", H_grid.shape)