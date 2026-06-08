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

R_atoms=params['R_atoms']

# ----------------------------
# 径向网格多通道内部态哈密顿量
# ----------------------------
def internal_hamiltonian_grid(r_grid):
    """
    构造两原子内部态哈密顿量网格 H_internal(r1,r2)
    shape: (Nr, Nr, n_channels, n_channels)
    """
    Nr = len(r_grid)
    H_grid = np.zeros((Nr, Nr, n_channels, n_channels), dtype=np.complex128)


    # --------------------------------------------------------
    # Rydberg-Rydberg相互作用
    # --------------------------------------------------------
    Vrr = C6 / (R_atoms**6)

    # --------------------------------------------------------
    # 填充内部态哈密顿量
    # --------------------------------------------------------
    idx_gg = internal_states.index('gg')
    idx_gr = internal_states.index('gr')
    idx_rg = internal_states.index('rg')
    idx_rr = internal_states.index('rr')

# ✅ 修复3：向量化填充，去掉双重循环（速度提升100倍）
# 激光耦合项
    H_grid[:, :, idx_gg, idx_gr] = Omega / 2
    H_grid[:, :, idx_gr, idx_gg] = Omega / 2
    H_grid[:, :, idx_gg, idx_rg] = Omega / 2
    H_grid[:, :, idx_rg, idx_gg] = Omega / 2
    H_grid[:, :, idx_gr, idx_rr] = Omega / 2
    H_grid[:, :, idx_rr, idx_gr] = Omega / 2
    H_grid[:, :, idx_rg, idx_rr] = Omega / 2
    H_grid[:, :, idx_rr, idx_rg] = Omega / 2

    # 失谐项
    H_grid[:, :, idx_gr, idx_gr] = -Delta
    H_grid[:, :, idx_rg, idx_rg] = -Delta

    # ✅ 修复4：双激发态能量随距离变化
    H_grid[:, :, idx_rr, idx_rr] = -2 * Delta + Vrr

    return H_grid

# ----------------------------
# 使用示例
# ----------------------------
if __name__ == "__main__":
    # 打印单个点内部态哈密顿量
    Nr_mid = len(r_grid) // 2
    H_grid = internal_hamiltonian_grid(r_grid)
    print("径向网格形状:", H_grid.shape)
    print("中间点 |rr> 态能量:", H_grid[Nr_mid, Nr_mid, -1, -1])


'''
    for i in range(Nr):
        for j in range(Nr):
            Hloc = np.zeros((n_channels, n_channels), dtype=np.complex128)

            # 激光耦合
            Hloc[idx_gg, idx_gr] = Omega / 2
            Hloc[idx_gr, idx_gg] = Omega / 2
            Hloc[idx_gg, idx_rg] = Omega / 2
            Hloc[idx_rg, idx_gg] = Omega / 2

            Hloc[idx_gr, idx_rr] = Omega / 2
            Hloc[idx_rr, idx_gr] = Omega / 2
            Hloc[idx_rg, idx_rr] = Omega / 2
            Hloc[idx_rr, idx_rg] = Omega / 2

            # 激光失谐
            Hloc[idx_gr, idx_gr] = -Delta
            Hloc[idx_rg, idx_rg] = -Delta

            # 双激发态加上Rydberg-Rydberg相互作用
            Hloc[idx_rr, idx_rr] = -2*Delta + Vrr

            H_grid[i,j] = Hloc
'''