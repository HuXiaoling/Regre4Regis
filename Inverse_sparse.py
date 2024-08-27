# import numpy as np
# from scipy.sparse.linalg import splu, spsolve
# from scipy import sparse
# import torch
# import sys
# sys.path.append('/autofs/space/durian_001/users/xh999/regre4regis/proj_supersynth_registration')
# import ext.interpol
# from time import time
# from sklearn.decomposition import TruncatedSVD

# from scipy.sparse.linalg import splu, inv

# import torch.sparse

# def conjugate_gradient(A, b, x=None, tol=1e-10, max_iter=1000):
#     """
#     Solves Ax = b using the Conjugate Gradient method.
#     A should be symmetric positive definite (e.g., A^T A).
#     """
#     if x is None:
#         x = torch.zeros_like(b)

#     r = b - A @ x
#     p = r.clone()
#     rsold = torch.dot(r, r)

#     for i in range(max_iter):
#         Ap = A @ p
#         alpha = rsold / torch.dot(p, Ap)
#         x = x + alpha * p
#         r = r - alpha * Ap
#         rsnew = torch.dot(r, r)
        
#         if torch.sqrt(rsnew) < tol:
#             break
        
#         p = r + (rsnew / rsold) * p
#         rsold = rsnew

#     return x

# # Example Usage
# if __name__ == "__main__":
#     start_time = time()
#     shape = 50
#     control_points_shape = (shape, shape, shape)
#     BigX = np.zeros([control_points_shape[0], 256])
#     BigY = np.zeros([control_points_shape[1], 256])
#     BigZ = np.zeros([control_points_shape[2], 256])
#     for i in range(control_points_shape[0]):
#         small = torch.zeros(control_points_shape[0])
#         small[i] = 1
#         BigX[i] = ext.interpol.resize(small, shape = 256, prefilter=False)
#     for j in range(control_points_shape[1]):
#         small = torch.zeros(control_points_shape[1])
#         small[j] = 1
#         BigY[j] = ext.interpol.resize(small, shape = 256, prefilter=False)
#     for k in range(control_points_shape[2]):
#         small = torch.zeros(control_points_shape[2])
#         small[k] = 1
#         BigZ[k] = ext.interpol.resize(small, shape = 256, prefilter=False)

#     sBigX = sparse.csr_matrix(BigX)
#     sBigY = sparse.csr_matrix(BigY)
#     sBigZ = sparse.csr_matrix(BigZ)
#     sparse_basis = sparse.kron(sBigX, sparse.kron(sBigY, sBigZ))
#     b = np.random.randn(shape**3)

#     import pdb; pdb.set_trace()
#     end_time = time()
#     print("CG took {} seconds.".format(end_time-start_time))

import torch

# Example matrices B and C (assume they are dense or sparse)
B = torch.rand(256, 50)
C = torch.rand(256, 50)
D = torch.rand(256, 50)

# Compute B^T B and C^T C
BtB = B.T @ B
CtC = C.T @ C
DtD = D.T @ D

# Compute the inverses
BtB_inv = torch.linalg.inv(BtB)
CtC_inv = torch.linalg.inv(CtC)
DtD_inv = torch.linalg.inv(DtD)

# Compute the Kronecker product of the inverses
AtA_inv = torch.kron(BtB_inv, torch.kron(CtC_inv, DtD_inv))
import pdb; pdb.set_trace()
print("Inverse of A^T A (AtA_inv):", AtA_inv)