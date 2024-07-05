import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion
import torch

# Pearson correlation coefficient using NumPy

gt_mni = nib.load('samples/414456.mni_coords.nii.gz').get_fdata()
pred_mni = nib.load('samples/414456_coor_pred.nii.gz').get_fdata()
pred_mni_affine = nib.load('samples/414456_coor_pred.nii.gz').affine
pred_sigma = nib.load('samples/414456_sigma_pred.nii.gz').get_fdata()
pred_sigma = np.nan_to_num(pred_sigma, nan=0.0, posinf=1e10, neginf=-1e10)
M = nib.load('data/414456.mask.nii.gz').get_fdata()

structuring_element = np.ones((6, 7, 6))  # 3x3x3 cube
M = binary_erosion(M, structure=structuring_element)

diff = np.sqrt(np.mean((gt_mni - pred_mni) ** 2, axis=3))
pred_sigma = pred_sigma * M
diff = diff * M

# Flatten the matrices
flattened_matrix1 = diff.flatten()
flattened_matrix2 = pred_sigma.flatten()

pred_sigma = nib.Nifti1Image(pred_sigma, affine = pred_mni_affine)
nib.save(pred_sigma, 'samples/pred_sigma_masked.nii.gz')
diff = nib.Nifti1Image(diff, affine = pred_mni_affine)
nib.save(diff, 'samples/diff_masked.nii.gz')

# Compute the correlation
correlation = np.corrcoef(flattened_matrix1, flattened_matrix2)[0, 1]
print("Correlation (NumPy):", correlation)