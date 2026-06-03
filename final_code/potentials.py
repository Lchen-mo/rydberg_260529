import numpy as np
from atom_parameters import params

# ----------------------------
# 基本参数
# ----------------------------
C6 = params['C6']
Omega = params['Omega']
Delta = params['Delta']
internal_states = params['internal_states']
n_channels = params['n_channels']


# ----------------------------
# 势函数：Rydberg-Rydberg相互作用
# ----------------------------
def V_rr(r12):
    """
    计算双里德堡态的van der Waals相互作用
    r12 : 原子间距离 (m)
    返回: 作用在 |rr> 态的能量 (Hz)
    """
    return C6 / r12**6

# ----------------------------
# 内部态依赖势矩阵
# ----------------------------
def H_internal(r12):
    """
    构建两体内部态哈密顿量，依赖原子间距 r12
    返回形状为 (n_channels, n_channels) 的矩阵
    内部态顺序由 internal_states 决定: ['gg','gr','rg','rr']
    """
    H = np.zeros((n_channels, n_channels), dtype=np.complex128)

    # 激光耦合
    # |gg> <-> |gr> 和 |gg> <-> |rg>
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
    H[idx_rr, idx_rr] = -2 * Delta + V_rr(r12)  # rr态加上C6/r^6势

    return H

# ----------------------------
# 生成空间势函数网格
# ----------------------------
def generate_potential_grid(r1_grid, r2_grid):
    """
    为七自由度波函数生成势函数矩阵
    r1_grid, r2_grid : 三维格点 (Nx, Ny, Nz)
    返回: H_grid，形状 (Nx, Ny, Nz, Nx, Ny, Nz, n_channels, n_channels)
    注意: 内存会很大，适合小网格或使用稀疏化
    """
    Nx1, Ny1, Nz1 = r1_grid.shape
    Nx2, Ny2, Nz2 = r2_grid.shape

    H_grid = np.zeros((Nx1, Ny1, Nz1, Nx2, Ny2, Nz2, n_channels, n_channels),
                      dtype=np.complex128)

    # 遍历所有格点计算原子间距离
    # 使用广播加速
    r1 = r1_grid[..., np.newaxis, np.newaxis, np.newaxis, :]
    r2 = r2_grid[np.newaxis, np.newaxis, np.newaxis, ... , :]

    # 计算原子间距
    r12_vec = r1 - r2  # shape: (Nx1,Ny1,Nz1,Nx2,Ny2,Nz2,3)
    r12 = np.linalg.norm(r12_vec, axis=-1)

    # 对每个格点生成内部态势矩阵
    for i in range(Nx1):
        for j in range(Ny1):
            for k in range(Nz1):
                for p in range(Nx2):
                    for q in range(Ny2):
                        for r in range(Nz2):
                            H_grid[i,j,k,p,q,r,:,:] = H_internal(r12[i,j,k,p,q,r])

    return H_grid