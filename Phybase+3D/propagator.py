#总哈密顿量 + 薛定谔方程求解
#propagator.py

import numpy as np

from scipy.sparse import (
    diags,
    eye,
    kron,
    csr_matrix
)

from scipy.sparse.linalg import expm_multiply

from tqdm import tqdm

from atom_parameters import params
from internal_hamiltonian import internal_hamiltonian_grid


# ============================================================
# parameters
# ============================================================

hbar = params['hbar']

m = params['mRb']

r_grid = params['r_grid']

dr = params['dr']

Nr = len(r_grid)

n_channels = params['n_channels']

mu = m / 2




# ============================================================
# kinetic operator 动能
# ============================================================
'''
def kinetic_operator_sparse():

    """
    sparse radial kinetic operator
    """

    main_diag = -2.0 * np.ones(Nr)

    off_diag = np.ones(Nr - 1)

    D2 = diags(         
        [off_diag, main_diag, off_diag],
        [-1, 0, 1],
        shape=(Nr, Nr),
        format='csr'
    )               #二阶导数

    T1D = -(hbar**2)/(2*mu*dr**2) * D2 / hbar  #一维动能算符

    I = eye(Nr, format='csr')

    # T(r1,r2)

    T2D = kron(T1D, I) + kron(I, T1D)   #二体动能算符

    return T2D.tocsr()
'''#表示原子核运动，不需要

# ============================================================
# total Hamiltonian   （有问题）
# ============================================================

def build_total_hamiltonian():

    #print("Building kinetic operator...")

    #T2D = kinetic_operator_sparse()

    dim_space = Nr * Nr

    dim_total = dim_space * n_channels  #总维度

    H = csr_matrix((dim_total, dim_total), dtype=np.complex128)

    # --------------------------------------------------------

    print("Building internal Hamiltonian grid...")

    H_internal = internal_hamiltonian_grid(r_grid)

    # --------------------------------------------------------

    print("Assembling sparse Hamiltonian...")

    for c1 in tqdm(range(n_channels), desc="channel row"):

        for c2 in range(n_channels):

            # spatial kinetic term

            #if c1 == c2:

            #    row_start = c1 * dim_space
            #    row_end = (c1 + 1) * dim_space

            #    H[row_start:row_end,
            #      row_start:row_end] += T2D

            # potential term

            V_diag =  H_internal[:,:, c1, c2].flatten()   

            

            V_sparse = diags(
                V_diag,
                0,
                format='csr'
            )           #稀疏矩阵转化

            row_start = c1 * dim_space
            row_end = (c1 + 1) * dim_space

            col_start = c2 * dim_space
            col_end = (c2 + 1) * dim_space

            H[row_start:row_end,
              col_start:col_end] += V_sparse
   # H*=1e-5
    return H.tocsr()




# ============================================================
# propagation  求解
# ============================================================

def propagate(
    psi0,
    H,
    dt,
    n_steps,
    save_every=10
):

    # ✅ 修复1：严格按照哈密顿量的块结构展平（通道优先）
    psi = np.moveaxis(psi0, -1, 0).flatten()  # 核心：(4,20,20)展平，匹配H的分块
    history = []

    print("Starting propagation...")

    for step in tqdm(range(n_steps), desc="time evolution"):
        # ✅ 修复2：删除多余的 /hbar！（你的哈密顿量已约化ħ）
        # ✅ 修复3：用expm_multiply保范传播，永不溢出
        psi = expm_multiply(
            -1j * H * dt,   
            psi
        )                 # 解哈密顿方程

        if step % save_every == 0:
            # 重塑回原始形状
            psi_reshaped = np.moveaxis(psi.reshape(n_channels, Nr, Nr), 0, -1)
            history.append(psi_reshaped.copy())

    return history