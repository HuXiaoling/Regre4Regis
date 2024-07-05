import numpy as np
import nibabel as nib
import torch
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from sklearn.metrics import mean_squared_error

# Read in the atlas and its header

MNIatlas = nib.load('samples/mni.nii.gz')
MNIintensities = MNIatlas.get_fdata()
MNIaffine = MNIatlas.affine

gt_mni = nib.load('samples/414456.mni_coords.nii.gz').get_fdata()

pred_mni = nib.load('samples/414456_coor_pred.nii.gz').get_fdata()
pred_mni_affine = nib.load('samples/414456_coor_pred.nii.gz').affine
M = nib.load('data/414456.mask.nii.gz').get_fdata()

xx = pred_mni[:, :, :, 0] * M # M is the brain mask, mni is the prediction
yy = pred_mni[:, :, :, 1] * M
zz = pred_mni[:, :, :, 2] * M

# xx, yy, zz are in RAS; we need to convert them to vox
A = np.linalg.inv(MNIaffine)
i2 = A[0, 0] * xx + A[0, 1] * yy + A[0, 2] * zz + A[0, 3]
j2 = A[1, 0] * xx + A[1, 1] * yy + A[1, 2] * zz + A[1, 3]
k2 = A[2, 0] * xx + A[2, 1] * yy + A[2, 2] * zz + A[2, 3]

DEFORMED = np.zeros(pred_mni.shape[0:3])

# Version 1: deform yourself with nearest neighbor

i2 = np.minimum(np.maximum(0, np.round(i2)), MNIintensities.shape[0])
j2 = np.minimum(np.maximum(0, np.round(j2)), MNIintensities.shape[1])
k2 = np.minimum(np.maximum(0, np.round(k2)), MNIintensities.shape[2])

# blah blah

for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        for k in range(xx.shape[2]):
            # import pdb; pdb.set_trace()
            DEFORMED[i,j,k] = MNIintensities[int(np.round(i2[i,j,k])), int(np.round(j2[i,j,k])), int(np.round(k2[i,j,k]))]

new_image = nib.Nifti1Image(DEFORMED, affine = pred_mni_affine)
nib.save(new_image, 'samples/414456_direct.nii.gz')
# Version =2: deform yourself with trilinear interpolation (scipy / rgi)
# No need to clip to 0, shape as rgi will just gives you zeros if you're out of bounds

# blah blah


# Write DEFORMED with pred_mni_affine in header

# MNI_flat = MNI.flatten()
# pred_mni_flat = pred_mni.flatten()
# error = mean_squared_error(pred_mni_flat, MNI_flat)

diff = np.sqrt(np.mean((gt_mni - pred_mni) ** 2, axis=3))
diff_mni = nib.Nifti1Image(diff, affine = pred_mni_affine)
nib.save(diff_mni, 'samples/diff.nii.gz')

