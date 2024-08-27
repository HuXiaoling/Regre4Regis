import numpy as np
from scipy.sparse.linalg import splu, spsolve
from scipy import sparse
import torch
import sys
sys.path.append('/autofs/space/durian_001/users/xh999/regre4regis/proj_supersynth_registration')
import ext.interpol
from scipy.sparse.linalg import LinearOperator, cg
from time import time

def solve_sparse_cg(A, b, tol=1e-8, max_iter=100000):
    """
    Solves the linear system (A^T A)x = b using Conjugate Gradient method.
    A is assumed to be sparse.

    Parameters:
    - A: sparse matrix (CSR format)
    - b: dense vector (1D numpy array)
    - tol: tolerance for convergence
    - max_iter: maximum number of iterations

    Returns:
    - x: solution vector
    """

    def AtA_mv(v):
        """
        Function to compute the matrix-vector product (A^T A)v
        """
        Av = A.dot(v)  # A @ v
        AtAv = A.transpose().dot(Av)  # A^T @ (A @ v)
        return AtAv

    # Define the LinearOperator that applies (A^T A) to a vector v
    n_cols = A.shape[1]
    AtA_op = LinearOperator((n_cols, n_cols), matvec=AtA_mv)

    # Use the Conjugate Gradient method to solve AtA x = b
    x, info = cg(AtA_op, b, tol=tol, maxiter=max_iter)

    if info != 0:
        print(f"Conjugate Gradient did not converge. Info: {info}")

    return x

# Example Usage
if __name__ == "__main__":
    start_time = time()
    shape = 50
    control_points_shape = (shape, shape, shape)
    BigX = np.zeros([control_points_shape[0], 256])
    BigY = np.zeros([control_points_shape[1], 256])
    BigZ = np.zeros([control_points_shape[2], 256])
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

    sBigX = sparse.csr_matrix(BigX)
    sBigY = sparse.csr_matrix(BigY)
    sBigZ = sparse.csr_matrix(BigZ)
    sparse_basis = sparse.kron(sBigX, sparse.kron(sBigY, sBigZ))
    b = np.random.randn(shape**3)

    # Solve the system using Cholesky decomposition
    import pdb; pdb.set_trace()
    x = solve_sparse_cg(sparse_basis.T, b)
    # import pdb; pdb.set_trace()
    print("Solution x:", x)
    end_time = time()
    print("CG took {} seconds.".format(end_time-start_time))