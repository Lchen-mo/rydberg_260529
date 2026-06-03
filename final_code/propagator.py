# propagator.py

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
# kinetic operator
# ============================================================

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
    )

    T1D = -(hbar**2)/(2*mu*dr**2) * D2

    I = eye(Nr, format='csr')

    # T(r1,r2)

    T2D = kron(T1D, I) + kron(I, T1D)

    return T2D.tocsr()


# ============================================================
# total Hamiltonian
# ============================================================

def build_total_hamiltonian():

    """
    construct sparse total Hamiltonian
    """

    print("Building kinetic operator...")

    T2D = kinetic_operator_sparse()

    dim_space = Nr * Nr

    dim_total = dim_space * n_channels

    H = csr_matrix((dim_total, dim_total), dtype=np.complex128)

    # --------------------------------------------------------

    print("Building internal Hamiltonian grid...")

    H_internal = internal_hamiltonian_grid(r_grid)

    # --------------------------------------------------------

    print("Assembling sparse Hamiltonian...")

    for c1 in tqdm(range(n_channels), desc="channel row"):

        for c2 in range(n_channels):

            # spatial kinetic term

            if c1 == c2:

                row_start = c1 * dim_space
                row_end = (c1 + 1) * dim_space

                H[row_start:row_end,
                  row_start:row_end] += T2D

            # potential term

            V_diag = np.repeat(
                H_internal[:, c1, c2],
                Nr
            )

            V_sparse = diags(
                V_diag,
                0,
                format='csr'
            )

            row_start = c1 * dim_space
            row_end = (c1 + 1) * dim_space

            col_start = c2 * dim_space
            col_end = (c2 + 1) * dim_space

            H[row_start:row_end,
              col_start:col_end] += V_sparse
    H=H*10e-5
    return H.tocsr()


# ============================================================
# propagation
# ============================================================

# ============================================================
# propagation
# ============================================================
def propagate(
    psi0,
    H,
    dt,
    n_steps,
    save_every=1
):
    """
    修复版：保范、无溢出、维度匹配、无NaN
    """
    # ✅ 修复1：严格按照哈密顿量的块结构展平（通道优先）
    psi = np.moveaxis(psi0, -1, 0).flatten()  # 核心：(4,20,20)展平，匹配H的分块
    history = []

    print("Starting propagation...")

    for step in tqdm(range(n_steps), desc="time evolution"):
        # ✅ 修复2：删除多余的 /hbar！（你的哈密顿量已约化ħ）
        # ✅ 修复3：用expm_multiply保范传播，永不溢出
        psi = expm_multiply(
            -1j * H * dt,  # 关键：删掉 /hbar！
            psi
        )

        if step % save_every == 0:
            # 重塑回原始形状
            psi_reshaped = np.moveaxis(psi.reshape(n_channels, Nr, Nr), 0, -1)
            history.append(psi_reshaped.copy())

    return history