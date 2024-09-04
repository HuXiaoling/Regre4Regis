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
from scipy.sparse import kron, csc_matrix, csr_matrix, coo_matrix
from scipy.sparse.linalg import splu, spsolve
import scipy.sparse as sp

# Example Usage
if __name__ == "__main__":
    start_time = time()
    shape = 50
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

    # Economy QR decomposition of A
    Q_econX, R_econX = np.linalg.qr(BigX.T, mode='reduced')
    Q_econY, R_econY = np.linalg.qr(BigY.T, mode='reduced')
    Q_econZ, R_econZ = np.linalg.qr(BigZ.T, mode='reduced')

    threshold = 1e-3

    # clip small values
    Q_econX[abs(Q_econX) < threshold] = 0
    Q_econY[abs(Q_econY) < threshold] = 0
    Q_econZ[abs(Q_econZ) < threshold] = 0
    R_econX[abs(R_econX) < threshold] = 0
    R_econY[abs(R_econY) < threshold] = 0
    R_econZ[abs(R_econZ) < threshold] = 0

    # Convert to sparse matrices (use csr_matrix for example)
    Q_econX_sparse = csr_matrix(Q_econX)
    Q_econY_sparse = csr_matrix(Q_econY)
    Q_econZ_sparse = csr_matrix(Q_econZ)

    R_econX_sparse = csr_matrix(R_econX)
    R_econY_sparse = csr_matrix(R_econY)
    R_econZ_sparse = csr_matrix(R_econZ)

    # Compute the sparse Kronecker product
    Q_temp = kron(Q_econY_sparse, Q_econZ_sparse)
    R_temp = kron(R_econZ_sparse, R_econZ_sparse)

    # Clip Q_temp
    data = Q_temp.data
    row = Q_temp.row
    col = Q_temp.col
    clipped_data = np.where(np.abs(data) < threshold, 0, data)
    nonzero_indices = clipped_data != 0
    clipped_data = clipped_data[nonzero_indices]
    clipped_row = row[nonzero_indices]
    clipped_col = col[nonzero_indices]
    Q_temp_clipped = coo_matrix((clipped_data, (clipped_row, clipped_col)), shape=Q_temp.shape)

    # Clip R_temp
    data = R_temp.data
    row = R_temp.row
    col = R_temp.col
    clipped_data = np.where(np.abs(data) < threshold, 0, data)
    nonzero_indices = clipped_data != 0
    clipped_data = clipped_data[nonzero_indices]
    clipped_row = row[nonzero_indices]
    clipped_col = col[nonzero_indices]
    R_temp_clipped = coo_matrix((clipped_data, (clipped_row, clipped_col)), shape=R_temp.shape)

    Q_sparse = kron(Q_econX_sparse, Q_temp_clipped)
    R_sparse = kron(R_econX_sparse, R_temp_clipped)

    # Clip Q_sparse
    data = Q_sparse.data
    row = Q_sparse.row
    col = Q_sparse.col
    clipped_data = np.where(np.abs(data) < threshold, 0, data)
    nonzero_indices = clipped_data != 0
    clipped_data = clipped_data[nonzero_indices]
    clipped_row = row[nonzero_indices]
    clipped_col = col[nonzero_indices]
    Q_sparse_clipped = coo_matrix((clipped_data, (clipped_row, clipped_col)), shape=Q_sparse.shape)

    # Clip R_sparse
    data = R_sparse.data
    row = R_sparse.row
    col = R_sparse.col
    clipped_data = np.where(np.abs(data) < threshold, 0, data)
    nonzero_indices = clipped_data != 0
    clipped_data = clipped_data[nonzero_indices]
    clipped_row = row[nonzero_indices]
    clipped_col = col[nonzero_indices]
    R_sparse_clipped = coo_matrix((clipped_data, (clipped_row, clipped_col)), shape=R_sparse.shape)

    # W_sparse = sp.diags(W.numpy()) # Convert to 1D array and then to diagonal matrix

    W_sparse = sp.diags(W.numpy(), offsets=0, format='csr')

    # Clip inverse
    inverse = inv(Q_sparse_clipped.T @ W_sparse @ Q_sparse_clipped)
    inverse_clipped = np.where(np.abs(inverse.data) < threshold, 0, inverse.data)
    inverse.data = inverse_clipped
    inverse.eliminate_zeros()
    inverse_coo = inverse.tocoo()

    # Clip R_inverse
    R_inverse = inv(R_sparse_clipped) @ inverse_coo
    R_inverse_clipped = np.where(np.abs(R_inverse.data) < threshold, 0, R_inverse.data)
    R_inverse.data = R_inverse_clipped
    R_inverse.eliminate_zeros()
    R_inverse_coo = R_inverse.tocoo()

    # Clip QTW
    QTW = Q_sparse_clipped.T @ W_sparse
    QTW_clipped = np.where(np.abs(QTW.data) < threshold, 0, QTW.data)
    QTW.data = QTW_clipped
    QTW.eliminate_zeros()
    QTW_coo = QTW.tocoo()

    M_inv = R_inverse_coo @ QTW_coo
    print("Shape of output:\n", M_inv.shape)

    # compute inverse directly
    # M = A.T @ W @ A
    # M_inv_direct = np.linalg.inv(M)
    # print("Inverse of (Q^T W Q) directly:\n", M_inv_direct)

    # error = np.linalg.norm(M_inv - M_inv_direct, ord='fro')
    # print("Inverse error:", error)

    end_time = time()
    print("Inverse took {} seconds.".format(end_time-start_time))
