import torch
import torch.nn.functional as F
import numpy as np 
import sys
sys.path.append('/autofs/space/durian_001/users/xh999/regre4regis/proj_supersynth_registration')
import ext.interpol

def weighted_bspline_direct(fixed_image, control_points_shape):
    # BASIS = torch.zeros((256*256*256, 50*50*50), layout=torch.sparse_csc)
    small = torch.eye(control_points_shape[0])
    # BigX = ext.interpol.resize(small, [1, fixed_image.shape[1]], prefilter=False)
    # BigY = ext.interpol.resize(small, [1, fixed_image.shape[2]], prefilter=False)
    # BigZ = ext.interpol.resize(small, [1, fixed_image.shape[3]], prefilter=False)
    # for i in range(50):
    #     for j in range(50):
    #         for k in range(50):
    #             import pdb; pdb.set_trace()
    #             BASIS[:,c] = torch.sparse(torch.vectorize(BigX[i, :, None, :] * BigY[j, None, :, :], BigZ[k, :, :, None]))
    # c=c+1

    BigX = torch.zeros([control_points_shape[0], fixed_image.shape[1]])
    BigY = torch.zeros([control_points_shape[1], fixed_image.shape[2]])
    BigZ = torch.zeros([control_points_shape[2], fixed_image.shape[3]])
    for i in range(control_points_shape[0]):
        small = torch.zeros(control_points_shape[0])
        small[i] = 1
        # import pdb; pdb.set_trace()
        BigX[i] = ext.interpol.resize(small, shape = fixed_image.shape[1], prefilter=False)
    for j in range(control_points_shape[1]):
        small = torch.zeros(control_points_shape[1])
        small[j] = 1
        BigY[j] = ext.interpol.resize(small, shape = fixed_image.shape[2], prefilter=False)
    for k in range(control_points_shape[2]):
        small = torch.zeros(control_points_shape[2])
        small[k] = 1
        BigZ[k] = ext.interpol.resize(small, shape = fixed_image.shape[3], prefilter=False)

    BigX = BigX.to_sparse_csr()
    BigY = BigY.to_sparse_csr()
    BigZ = BigZ.to_sparse_csr()

    c=0
    indices_x = []
    indices_y = []
    values = []
    for i in range(control_points_shape[0]):
        for j in range(control_points_shape[1]):
            for k in range(control_points_shape[2]):
                # BASIS[:,c] = torch.sparse(torch.vectorize(BigX[i][:, None, None] * BigY[j][None, :, None] * BigZ[k][None, None, :]))
                # import pdb; pdb.set_trace()

                # to do: use sparse kron instead of dense kron
                sparse_basis = torch.kron(BigX[i].to_dense(), torch.kron(BigY[j].to_dense(), BigZ[k].to_dense()))
                # BASIS[:,c] = torch.transpose(sparse_basis[None, :], 0, 1).to_sparse_csc()
                temp = torch.transpose(sparse_basis[None, :], 0, 1).to_sparse_csc()
                indices_x += (temp.row_indices()).tolist()
                indices_y += (torch.ones(temp.row_indices().shape[0]) * c).tolist()
                values += (temp.values()).tolist()

                c += 1
    import pdb; pdb.set_trace()
    basis_functions = torch.sparse_csc_tensor(torch.tensor(indices_x, dtype=torch.int64), torch.tensor(indices_y, dtype=torch.int64), torch.tensor(values), size = (16777216, 125) ,dtype=torch.double)

def weighted_bspline_direct_scipy(fixed_image, control_points_shape):

    BigX = np.zeros([control_points_shape[0], fixed_image.shape[1]])
    BigY = np.zeros([control_points_shape[1], fixed_image.shape[2]])
    BigZ = np.zeros([control_points_shape[2], fixed_image.shape[3]])
    for i in range(control_points_shape[0]):
        small = torch.zeros(control_points_shape[0])
        small[i] = 1
        # import pdb; pdb.set_trace()
        BigX[i] = ext.interpol.resize(small, shape = fixed_image.shape[1], prefilter=False)
    for j in range(control_points_shape[1]):
        small = torch.zeros(control_points_shape[1])
        small[j] = 1
        BigY[j] = ext.interpol.resize(small, shape = fixed_image.shape[2], prefilter=False)
    for k in range(control_points_shape[2]):
        small = torch.zeros(control_points_shape[2])
        small[k] = 1
        BigZ[k] = ext.interpol.resize(small, shape = fixed_image.shape[3], prefilter=False)

    # BigX = BigX.to_sparse_csr()
    # BigY = BigY.to_sparse_csr()
    # BigZ = BigZ.to_sparse_csr()

    from scipy import sparse
    sBigX = sparse.csr_matrix(BigX)
    sBigY = sparse.csr_matrix(BigY)
    sBigZ = sparse.csr_matrix(BigZ)

    sparse_basis = sparse.kron(sBigX, sparse.kron(sBigY, sBigZ))

    import pdb; pdb.set_trace()
# Example Usage
if __name__ == "__main__":
    # Example data (replace these with actual medical images and weights)
    fixed_image = torch.rand(3, 256, 256, 256).cuda()  # Fixed image with shape (D, H, W)
    moving_image = torch.rand((3, 256, 256, 256), requires_grad=True).cuda()  # Moving image with shape (D, H, W)
    moving_weights = torch.ones(256, 256, 256).cuda()  # Uniform weights with shape (D, H, W)

    control_points_shape = (20, 20, 20)

    # Perform registration
    registered_image = weighted_bspline_direct(
        fixed_image, control_points_shape
    )

    print("Registration complete.")
