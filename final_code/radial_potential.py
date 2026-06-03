# radial_potential.py
import numpy as np
from atom_parameters import params

# ----------------------------
# 参数
# ----------------------------
C6 = params['C6']
Omega = params['Omega']
Delta = params['Delta']
r_grid = params['r_grid']
dr = params['dr']
internal_states = params['internal_states']
n_channels = params['n_channels']

Nr = len(r_grid)

# ----------------------------
# 双里德堡态van der Waals势
# ----------------------------
def V_rr(r):
    """Rydberg-Rydberg van der Waals作用, 单位与C6一致"""
    # 避免r=0的奇异
    r_safe = np.where(r < 1e-9, 1e-9, r)
    return C6 / r_safe**6

# ----------------------------
# 内部态多通道径向势矩阵
# ----------------------------
def H_internal_radial(r):
    """
    构建径向多通道哈密顿量 H(r) 
    返回 shape = (n_channels, n_channels)
    内部态顺序：['gg','gr','rg','rr']
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

    # 失谐作用在Rydberg态
    H[idx_gr, idx_gr] = -Delta
    H[idx_rg, idx_rg] = -Delta
    H[idx_rr, idx_rr] = -2 * Delta + V_rr(r)

    return H

# ----------------------------
# 生成径向势网格
# ----------------------------
def generate_radial_potential_grid():
    """
    为每个r生成多通道哈密顿量矩阵
    返回 shape = (Nr, n_channels, n_channels)
    """
    H_grid = np.zeros((Nr, n_channels, n_channels), dtype=np.complex128)
    for i, r in enumerate(r_grid):
        H_grid[i] = H_internal_radial(r)
    return H_grid

# ----------------------------
# 使用示例
# ----------------------------
if __name__ == "__main__":
    H_radial = generate_radial_potential_grid()
    print("径向势矩阵形状:", H_radial.shape)  # (Nr, n_channels, n_channels)
    print("r=0.1 um 时 |rr> 态能量:", H_radial[int(Nr*0.1/params['r_grid'][-1]), -1, -1])