import numpy as np
from arc import Rubidium87, PairStateInteractions
from scipy.constants import hbar, h

#创建 Rb 原子实例，定义计算所需的原子参数和基矢空间
def initialize_atom_system(
    target_state1: tuple,  # (n1, l1, j1, mj1)
    target_state2: tuple,  # (n2, l2, j2, mj2)
    electric_field: float = 0.0,  # DC电场强度(V/m)
    magnetic_field: float = 0.0,  # DC磁场强度(T)
    n_range: int = 4,  # 主量子数范围
    l_range: int = 5,  # 角量子数范围
    energy_cutoff: float = 10e9  # 能量截断(Hz)
) -> tuple:
    """
    初始化两个Rb原子系统和相互作用计算对象
    
    Returns:
        atom: Rubidium87原子实例
        pair_calc: PairStateInteractions计算对象
        basis_states: 基矢列表
    """
    # 创建Rb-87原子实例
    atom = Rubidium87()
    
    # 初始化双原子相互作用计算对象
    n1, l1, j1, mj1 = target_state1
    n2, l2, j2, mj2 = target_state2
    
    pair_calc = PairStateInteractions(
        atom, n1, l1, j1, n2, l2, j2, mj1, mj2,
        interactionsUpTo=2  # 计算到四极-四极相互作用
    )
    
    # 定义基矢空间
    pair_calc.defineBasis(
        electric_field, magnetic_field,
        n_range, l_range, energy_cutoff,
        progressOutput=True
    )
    
    return atom, pair_calc, pair_calc.basisStates



#构建单原子哈密顿量、双原子相互作用哈密顿量和光场耦合哈密顿量
def build_single_atom_hamiltonian(
    atom: Rubidium87,
    basis_states: list
) -> np.ndarray:
    """
    构建单原子哈密顿量的直和（对角矩阵）
    """
    dim = len(basis_states)
    H_single = np.zeros((dim, dim), dtype=np.complex128)
    
    for i, state in enumerate(basis_states):
        # state格式: (n1, l1, j1, mj1, n2, l2, j2, mj2)
        energy1 = atom.getEnergy(state[0], state[1], state[2]) * h  # 转换为焦耳
        energy2 = atom.getEnergy(state[4], state[5], state[6]) * h
        H_single[i, i] = energy1 + energy2
    
    return H_single

def build_interaction_hamiltonian(
    pair_calc: PairStateInteractions,
    R: float  # 原子间距(m)
) -> np.ndarray:
    """
    构建双原子相互作用哈密顿量
    
    Args:
        R: 原子间距(米)
    """
    # matR[0]: 1/R³项(偶极-偶极), matR[1]: 1/R⁴项(偶极-四极), matR[2]: 1/R⁵项(四极-四极)
    H_int = (pair_calc.matR[0] / R**3 + 
             pair_calc.matR[1] / R**4 + 
             pair_calc.matR[2] / R**5) * h  # 转换为焦耳
    
    return H_int

def build_light_coupling_hamiltonian(
    atom: Rubidium87,
    basis_states: list,
    laser_params: dict  # 包含transition, power, waist, polarization, detuning
) -> np.ndarray:
    """
    构建光场耦合哈密顿量（旋转波近似）
    
    laser_params格式:
    {
        "transition1": (n1g, l1g, j1g, mj1g, n1e, l1e, j1e, mj1e),  # 原子1的跃迁
        "transition2": (n2g, l2g, j2g, mj2g, n2e, l2e, j2e, mj2e),  # 原子2的跃迁
        "power": 1e-3,  # 激光功率(W)
        "waist": 1e-6,  # 激光束腰半径(m)
        "polarization": 0,  # -1:σ-, 0:π, 1:σ+
        "detuning": 0.0  # 激光失谐(Hz)
    }
    """
    dim = len(basis_states)
    H_light = np.zeros((dim, dim), dtype=np.complex128)
    
    # 计算原子1的Rabi频率
    t1 = laser_params["transition1"]
    omega1 = atom.getRabiFrequency(
        t1[0], t1[1], t1[2], t1[3],
        t1[4], t1[5], t1[6], t1[7],
        laser_params["polarization"],
        laser_params["power"],
        laser_params["waist"]
    )  # 单位: rad/s
    
    # 计算原子2的Rabi频率
    t2 = laser_params["transition2"]
    omega2 = atom.getRabiFrequency(
        t2[0], t2[1], t2[2], t2[3],
        t2[4], t2[5], t2[6], t2[7],
        laser_params["polarization"],
        laser_params["power"],
        laser_params["waist"]
    )  # 单位: rad/s
    
    # 构建耦合矩阵
    for i, state in enumerate(basis_states):
        # 检查原子1是否在跃迁的基态
        if (state[0], state[1], state[2], state[3]) == (t1[0], t1[1], t1[2], t1[3]):
            # 找到原子1在激发态的对应态
            for j, state_j in enumerate(basis_states):
                if (state_j[0], state_j[1], state_j[2], state_j[3]) == (t1[4], t1[5], t1[6], t1[7]) and \
                   state_j[4:] == state[4:]:
                    H_light[i, j] = 0.5 * hbar * omega1 * np.exp(-1j * laser_params["detuning"] * 2 * np.pi * 0)
                    H_light[j, i] = 0.5 * hbar * omega1 * np.exp(1j * laser_params["detuning"] * 2 * np.pi * 0)
        
        # 检查原子2是否在跃迁的基态
        if (state[4], state[5], state[6], state[7]) == (t2[0], t2[1], t2[2], t2[3]):
            # 找到原子2在激发态的对应态
            for j, state_j in enumerate(basis_states):
                if (state_j[4], state_j[5], state_j[6], state_j[7]) == (t2[4], t2[5], t2[6], t2[7]) and \
                   state_j[:4] == state[:4]:
                    H_light[i, j] += 0.5 * hbar * omega2 * np.exp(-1j * laser_params["detuning"] * 2 * np.pi * 0)
                    H_light[j, i] += 0.5 * hbar * omega2 * np.exp(1j * laser_params["detuning"] * 2 * np.pi * 0)
    
    return H_light

def build_total_hamiltonian(
    H_single: np.ndarray,
    H_int: np.ndarray,
    H_light: np.ndarray
) -> np.ndarray:
    """
    构建总哈密顿量
    """
    return H_single + H_int + H_light


#求解时间依赖的薛定谔方程，计算内部态波函数的演化
from scipy.integrate import solve_ivp

def schrodinger_equation(t: float, psi: np.ndarray, H_func: callable) -> np.ndarray:
    """
    薛定谔方程的右侧: dpsi/dt = -i/hbar * H(t) * psi
    """
    H = H_func(t)
    return -1j / hbar * H.dot(psi)

def evolve_internal_wavefunction(
    initial_psi: np.ndarray,
    t_span: tuple,  # (t_start, t_end)
    t_eval: np.ndarray,
    H_time_dependent: callable  # 返回时刻t的哈密顿量
) -> np.ndarray:
    """
    演化内部态波函数
    
    Args:
        initial_psi: 初始内部态波函数(归一化)
        t_span: 时间范围(秒)
        t_eval: 需要计算的时间点数组
        H_time_dependent: 函数，输入时间t，返回该时刻的哈密顿量
    
    Returns:
        psi_t: 各时刻的内部态波函数，shape=(len(t_eval), dim)
    """
    # 求解薛定谔方程
    sol = solve_ivp(
        schrodinger_equation,
        t_span,
        initial_psi,
        t_eval=t_eval,
        args=(H_time_dependent,),
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    
    return sol.y.T  # 转置为(time, state)格式


#计算原子的空间波函数，并与内部态波函数结合得到总波函数
def gaussian_tweezer_wavefunction(
    r: np.ndarray,  # 位置坐标(m)
    r0: np.ndarray,  # 光镊中心位置(m)
    omega_trap: float  # 囚禁频率(rad/s)
) -> np.ndarray:
    """
    计算光镊中原子的基态高斯空间波函数
    
    基态波函数: ψ(r) = (mω/(πħ))^(3/4) * exp(-mω(r-r0)²/(2ħ))
    """
    m = Rubidium87().mass  # Rb原子质量(kg)
    sigma = np.sqrt(hbar / (m * omega_trap))
    
    # 计算归一化常数
    norm = (m * omega_trap / (np.pi * hbar))**(3/4)
    
    # 计算波函数值
    r_squared = np.sum((r - r0)**2, axis=-1)
    psi = norm * np.exp(-r_squared / (2 * sigma**2))
    
    return psi

def compute_total_wavefunction(
    psi_internal_t: np.ndarray,  # 各时刻的内部态波函数
    basis_states: list,
    r1_grid: np.ndarray,  # 原子1的位置网格
    r2_grid: np.ndarray,  # 原子2的位置网格
    r0_1: np.ndarray,  # 原子1的光镊中心
    r0_2: np.ndarray,  # 原子2的光镊中心
    omega_trap1: float,  # 原子1的囚禁频率
    omega_trap2: float   # 原子2的囚禁频率
) -> np.ndarray:
    """
    计算总波函数φ(r₁,r₂,t) = ψ_internal(t) ⊗ ψ_spatial(r₁,r₂)
    
    Returns:
        phi_t: 总波函数，shape=(len(t_eval), len(r1_grid), len(r2_grid))
    """
    # 计算空间波函数
    psi1 = gaussian_tweezer_wavefunction(r1_grid, r0_1, omega_trap1)
    psi2 = gaussian_tweezer_wavefunction(r2_grid, r0_2, omega_trap2)
    
    # 计算双原子空间波函数(乘积态)
    psi_spatial = np.outer(psi1, psi2)
    
    # 计算总波函数
    n_time = len(psi_internal_t)
    n_r1 = len(r1_grid)
    n_r2 = len(r2_grid)
    
    phi_t = np.zeros((n_time, n_r1, n_r2), dtype=np.complex128)
    
    for t_idx in range(n_time):
        # 内部态波函数与空间波函数的乘积
        for state_idx, state in enumerate(basis_states):
            phi_t[t_idx] += psi_internal_t[t_idx, state_idx] * psi_spatial
    
    return phi_t


#主函数
def main():
    # ======================
    # 1. 设置计算参数
    # ======================
    # 目标态(这里以两个Rb原子都在5S₁/₂基态为例)
    target_state1 = (5, 0, 0.5, 0.5)  # (n, l, j, mj)
    target_state2 = (5, 0, 0.5, 0.5)
    
    # 激光参数(5S₁/₂ → 5P₃/₂跃迁)
    laser_params = {
        "transition1": (5, 0, 0.5, 0.5, 5, 1, 1.5, 0.5),
        "transition2": (5, 0, 0.5, 0.5, 5, 1, 1.5, 0.5),
        "power": 1e-3,  # 1mW
        "waist": 1e-6,  # 1μm
        "polarization": 0,  # π偏振
        "detuning": 0.0  # 共振
    }
    
    # 原子间距
    R = 5e-6  # 5μm
    
    # 光镊参数
    r0_1 = np.array([0, 0, 0])  # 原子1在原点
    r0_2 = np.array([R, 0, 0])  # 原子2在x轴上
    omega_trap = 2 * np.pi * 100e3  # 100kHz囚禁频率
    
    # 时间演化参数
    t_start = 0
    t_end = 1e-6  # 1μs
    n_time_points = 100
    t_eval = np.linspace(t_start, t_end, n_time_points)
    
    # ======================
    # 2. 初始化系统
    # ======================
    atom, pair_calc, basis_states = initialize_atom_system(
        target_state1, target_state2
    )
    dim = len(basis_states)
    print(f"基矢空间维度: {dim}")
    
    # ======================
    # 3. 构建哈密顿量
    # ======================
    H_single = build_single_atom_hamiltonian(atom, basis_states)
    H_int = build_interaction_hamiltonian(pair_calc, R)
    
    # 定义时间依赖的哈密顿量函数(这里激光是恒定的)
    def H_time_dependent(t):
        H_light = build_light_coupling_hamiltonian(atom, basis_states, laser_params)
        return build_total_hamiltonian(H_single, H_int, H_light)
    
    # ======================
    # 4. 设置初始波函数
    # ======================
    # 初始态: 两个原子都在5S₁/₂基态
    initial_psi = np.zeros(dim, dtype=np.complex128)
    for i, state in enumerate(basis_states):
        if state[:4] == target_state1 and state[4:] == target_state2:
            initial_psi[i] = 1.0
            break
    initial_psi /= np.linalg.norm(initial_psi)  # 归一化
    
    # ======================
    # 5. 演化内部态波函数
    # ======================
    print("开始演化内部态波函数...")
    psi_internal_t = evolve_internal_wavefunction(
        initial_psi, (t_start, t_end), t_eval, H_time_dependent
    )
    
    # ======================
    # 6. 计算总波函数
    # ======================
    print("计算总波函数...")
    # 创建位置网格
    grid_size = 20
    x = np.linspace(-2e-6, 2e-6, grid_size)
    y = np.linspace(-2e-6, 2e-6, grid_size)
    z = np.linspace(-2e-6, 2e-6, grid_size)
    r1_grid = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)
    r2_grid = np.array(np.meshgrid(x + R, y, z)).T.reshape(-1, 3)
    
    phi_t = compute_total_wavefunction(
        psi_internal_t, basis_states,
        r1_grid, r2_grid, r0_1, r0_2,
        omega_trap, omega_trap
    )
    
    # ======================
    # 7. 结果分析与保存
    # ======================
    print("计算完成!")
    print(f"总波函数形状: {phi_t.shape}")
    
    # 可以在这里添加结果可视化和保存代码
    # 例如: 绘制布居数随时间的变化
    # 例如: 保存总波函数数据
    
    return phi_t, t_eval, basis_states

if __name__ == "__main__":
    phi_t, t_eval, basis_states = main()
