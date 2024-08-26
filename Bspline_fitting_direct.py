import torch
import torch.nn.functional as F
import numpy as np
from scipy import sparse
import sys
sys.path.append('/autofs/space/durian_001/users/xh999/regre4regis/proj_supersynth_registration')
import ext.interpol
from scipy.sparse.linalg import lsqr
import scipy.sparse as sp

def sparse_kron(A: torch.sparse_coo_tensor, B: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    # Extracting indices and values from matrix A
    A_indices = A._indices()
    A_values = A._values()
    A_rows, A_cols = A.shape

    # Extracting indices and values from matrix B
    B_indices = B._indices()
    B_values = B._values()
    B_rows, B_cols = B.shape

    # Kronecker product for indices
    kron_row_indices = A_indices[0, :].repeat_interleave(B._nnz()) * B_rows + B_indices[0, :].repeat(A._nnz())
    kron_col_indices = A_indices[1, :].repeat_interleave(B._nnz()) * B_cols + B_indices[1, :].repeat(A._nnz())
    
    # Kronecker product for values
    kron_values = A_values.repeat_interleave(B._nnz()) * B_values.repeat(A._nnz())
    
    # Combine the row and column indices
    kron_indices = torch.stack([kron_row_indices, kron_col_indices], dim=0)

    # Calculate the shape of the resulting Kronecker product
    kron_shape = (A_rows * B_rows, A_cols * B_cols)

    # Create the resulting sparse tensor
    return torch.sparse_coo_tensor(kron_indices, kron_values, size=kron_shape)

def weighted_bspline_lsqt(fixed_image, moving_image, moving_weights, control_points_shape):

    BigX = torch.zeros([control_points_shape[0], fixed_image.shape[0]])
    BigY = torch.zeros([control_points_shape[1], fixed_image.shape[1]])
    BigZ = torch.zeros([control_points_shape[2], fixed_image.shape[2]])
    for i in range(control_points_shape[0]):
        small = torch.zeros(control_points_shape[0])
        small[i] = 1
        BigX[i] = ext.interpol.resize(small, shape = fixed_image.shape[0], prefilter=False)
    for j in range(control_points_shape[1]):
        small = torch.zeros(control_points_shape[1])
        small[j] = 1
        BigY[j] = ext.interpol.resize(small, shape = fixed_image.shape[1], prefilter=False)
    for k in range(control_points_shape[2]):
        small = torch.zeros(control_points_shape[2])
        small[k] = 1
        BigZ[k] = ext.interpol.resize(small, shape = fixed_image.shape[2], prefilter=False)

    sBigX = BigX.to_sparse()
    sBigY = BigY.to_sparse()
    sBigZ = BigZ.to_sparse()

    import pdb; pdb.set_trace()
    sparse_basis = sparse_kron(sBigX, sparse_kron(sBigY, sBigZ))

    # numpy ver
    moving_weights = torch.sqrt(moving_weights.flatten().cpu())
    W_prime = moving_weights[:, None] * sparse_basis.T
    b_prime = moving_weights * moving_image.flatten().cpu()
    
    ## issues as torch has no lsqr
    coefficient = lsqr(W_prime, b_prime)[0]
    return coefficient

def weighted_bspline_lsqt_scipy(moving_image, moving_weights, control_points_shape):

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
    import pdb; pdb.set_trace()
    sparse_basis = sparse.kron(sBigX, sparse.kron(sBigY, sBigZ))

    # numpy ver
    moving_weights = np.sqrt(moving_weights.flatten().cpu().numpy())
    W_prime = sp.diags(moving_weights) * sparse_basis.T
    b_prime = moving_weights * moving_image.detach().flatten().cpu().numpy()
    
    coefficient = lsqr(W_prime, b_prime)[0]
    return coefficient

def weighted_bspline_direct(moving_image, moving_weights, control_points_shape):

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
    import pdb; pdb.set_trace()
    sparse_basis = sparse.kron(sBigX, sparse.kron(sBigY, sBigZ))

    # numpy ver
    moving_weights = np.sqrt(moving_weights.flatten().cpu().numpy())
    W_prime = sp.diags(moving_weights) * sparse_basis.T
    b_prime = moving_weights * moving_image.detach().flatten().cpu().numpy()
    
    coefficient = lsqr(W_prime, b_prime)[0]
    return coefficient

# Example Usage
if __name__ == "__main__":

    fixed_image = torch.rand(256, 256, 256).cuda()  # Fixed image with shape (D, H, W)
    moving_image = torch.rand((256, 256, 256), requires_grad=True).cuda().flatten()  # Moving image with shape (D, H, W)
    moving_weights = torch.rand(256, 256, 256).cuda().flatten()  # Uniform weights with shape (D, H, W)

    control_points_shape = (20, 20, 20)

    # coefficient = weighted_bspline_direct(fixed_image, moving_image, moving_weights, control_points_shape)
    coefficient = weighted_bspline_lsqt_scipy(moving_image, moving_weights, control_points_shape)

    print("Registration complete.")
