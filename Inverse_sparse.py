from scipy.sparse.linalg import splu, spsolve
from scipy import sparse
import torch
import sys
sys.path.append('/autofs/space/durian_001/users/xh999/regre4regis/proj_supersynth_registration')
import ext.interpol
from time import time

from scipy.sparse.linalg import splu, inv

import torch.sparse

import numpy as np
from scipy.sparse import kron, csc_matrix, csr_matrix
from scipy.sparse.linalg import splu, spsolve
import scipy.sparse as sp

# Example Usage
if __name__ == "__main__":
    start_time = time()
    shape = 10
    control_points_shape = (shape, shape, shape)
    BigX = torch.zeros([control_points_shape[0], 256])
    BigY = torch.zeros([control_points_shape[1], 256])
    BigZ = torch.zeros([control_points_shape[2], 256])
    for i in range(control_points_shape[0]):
        small = torch.zeros(control_points_shape[0])
        small[i] = 1
        BigX[i] = ext.interpol.resize(small, shape = 256, prefilter=False)
    for j in range(control_points_shape[1]):
        small = torch.zeros(control_points_shape[1])
        small[j] = 1
        BigY[j] = ext.interpol.resize(small, shape = 256, prefilter=False)
    for k in range(control_points_shape[2]):
        small = torch.zeros(control_points_shape[2])
        small[k] = 1
        BigZ[k] = ext.interpol.resize(small, shape = 256, prefilter=False)

    W = torch.rand(256, 256, 256).flatten()

    ## m > n

    # Economy QR decomposition of A
    Q_econX, R_econX = np.linalg.qr(BigX.T, mode='reduced')
    Q_econY, R_econY = np.linalg.qr(BigY.T, mode='reduced')
    Q_econZ, R_econZ = np.linalg.qr(BigZ.T, mode='reduced')

    # clip small values
    # Q_econX[abs(Q_econX) < 1e-5] = 0
    # Q_econY[abs(Q_econY) < 1e-5] = 0
    # Q_econZ[abs(Q_econZ) < 1e-5] = 0
    # R_econX[abs(R_econX) < 1e-5] = 0
    # R_econY[abs(R_econY) < 1e-5] = 0
    # R_econZ[abs(R_econZ) < 1e-5] = 0

    # Convert to sparse matrices (use csr_matrix for example)
    Q_econX_sparse = csr_matrix(Q_econX)
    Q_econY_sparse = csr_matrix(Q_econY)
    Q_econZ_sparse = csr_matrix(Q_econZ)

    R_econX_sparse = csr_matrix(R_econX)
    R_econY_sparse = csr_matrix(R_econY)
    R_econZ_sparse = csr_matrix(R_econZ)

    # Compute the sparse Kronecker product
    Q_sparse = kron(Q_econX_sparse, kron(Q_econY_sparse, Q_econZ_sparse))
    R_sparse = kron(R_econX_sparse, kron(R_econY_sparse, R_econZ_sparse))

    # W_sparse = sp.diags(W.numpy()) # Convert to 1D array and then to diagonal matrix

    W_sparse = sp.diags(W.numpy(), offsets=0, format='csr')

    import pdb; pdb.set_trace()
    inverse = inv(Q_sparse.T.multiply(W_sparse) @ Q_sparse)
    M_inv = inv(R_sparse) @ inverse @ Q_sparse.T @ W_sparse
    print("Inverse of (Q^T W Q):\n", M_inv)

    # compute inverse directly
    # M = A.T @ W @ A
    # M_inv_direct = np.linalg.inv(M)
    # print("Inverse of (Q^T W Q) directly:\n", M_inv_direct)

    # error = np.linalg.norm(M_inv - M_inv_direct, ord='fro')
    # print("Inverse error:", error)

    end_time = time()
    print("Inverse took {} seconds.".format(end_time-start_time))
