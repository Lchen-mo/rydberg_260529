# wavefunction.py
import numpy as np
from atom_parameters import params

r_grid = params['r_grid']
Nr = len(r_grid)
n_channels = params['n_channels']



# ----------------------------
# 初始化径向多通道波函数
# ----------------------------
def initialize_wavefunction(
        psi_type='gg', 
        perturb=0.0 ,
        r1_0=0,
        r2_0=0,
        sigma=None        

        ):
    """
    初始化径向多通道波函数 psi(r1,r2)
    
    psi_type: 初始态类型
        'gg' - 两原子都在基态
        'rr' - 两原子都在Rydberg态
        'superposition' - 四通道均匀叠加态
    perturb: 随机扰动幅度，用于测试动力学
    """

    psi = np.zeros((Nr, Nr, n_channels), dtype=np.complex128)

# --------------------------------------------------------
    # Gaussian packet
# --------------------------------------------------------

    R1, R2 = np.meshgrid(
        r_grid,
        r_grid,
        indexing='ij'
    )           #二维坐标网络矩阵

    phi1 = np.exp(
        -(R1-r1_0)**2 / (2*sigma**2)
    )

    phi2 = np.exp(
        -(R2-r2_0)**2 / (2*sigma**2)
    )                   #！！！这个地方似乎仍然有问题，物理层面

    gaussian_packet = phi1 * phi2

    # --------------------------------------------------------
    # internal state
    # --------------------------------------------------------


    if psi_type == 'gg':
        psi[:,:,0] = gaussian_packet  # |gg>通道
    elif psi_type == 'rr':
        psi[:,:,-1] = gaussian_packet  # |rr>通道
    elif psi_type == 'superposition':
        for c in range(n_channels):

            psi[:,:,c] = gaussian_packet / np.sqrt(n_channels)

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
# 波函数求和到径向
# ----------------------------
def radial_density(psi):
    """
    对波函数进行通道求和，得到总径向密度
    psi: shape (Nr, Nr, n_channels)
    返回: shape (Nr, Nr) 只保留总概率密度
    """
    rho = np.sum(np.abs(psi)**2, axis=-1)
    return rho
                    #！！！！这个地方似乎也有问题，不能直接相加