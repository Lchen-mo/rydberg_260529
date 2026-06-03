# wavefunction.py
import numpy as np
from atom_parameters import params

r_grid = params['r_grid']
Nr = len(r_grid)
n_channels = params['n_channels']

# ----------------------------
# 初始化径向多通道波函数
# ----------------------------
def initialize_wavefunction(psi_type='gg', perturb=0.0):
    """
    初始化径向多通道波函数 psi(r1,r2)
    
    psi_type: 初始态类型
        'gg' - 两原子都在基态
        'rr' - 两原子都在Rydberg态
        'superposition' - 均匀叠加态
    perturb: 随机扰动幅度，用于测试动力学
    """
    psi = np.zeros((Nr, Nr, n_channels), dtype=np.complex128)

    if psi_type == 'gg':
        psi[:,:,0] = 1.0  # |gg>通道
    elif psi_type == 'rr':
        psi[:,:,-1] = 1.0  # |rr>通道
    elif psi_type == 'superposition':
        psi[:,:,:] = 1.0/np.sqrt(n_channels)
    else:
        raise ValueError("psi_type must be 'gg', 'rr', or 'superposition'")

    # 加入小随机扰动
    if perturb > 0:
        psi += perturb * (np.random.rand(Nr, Nr, n_channels) + 1j*np.random.rand(Nr, Nr, n_channels))

    # 归一化
    norm = np.sqrt(np.sum(np.abs(psi)**2) * (r_grid[1]-r_grid[0])**2)
    psi /= norm

    return psi

# ----------------------------
# 波函数角向积分到径向
# ----------------------------
def radial_density(psi):
    """
    对波函数进行角向积分，得到径向密度
    psi: shape (Nr, Nr, n_channels)
    返回: shape (Nr, Nr) 只保留总概率密度
    """
    rho = np.sum(np.abs(psi)**2, axis=-1)
    return rho

# ----------------------------
# 使用示例
# ----------------------------
if __name__ == "__main__":
    psi0 = initialize_wavefunction('gg', perturb=1e-3)
    rho = radial_density(psi0)
    print("波函数shape:", psi0.shape)
    print("径向密度shape:", rho.shape)
    print("归一化:", np.sum(rho)*(r_grid[1]-r_grid[0])**2)