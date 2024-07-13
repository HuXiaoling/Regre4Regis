# fitting functions to make the framework end2end
import os, sys
sys.path.append('/autofs/space/durian_001/users/xh999/regre4regis/proj_supersynth_registration')
import numpy as np
import torch
from SuperSynth.generators import fast_3D_interp_torch
from SuperSynth.utils import MRIread, MRIwrite, torch_resize, align_volume_to_ref
from scipy.interpolate import RegularGridInterpolator as rgi
from unet.unet_3d import UNet_3d
from scipy.ndimage import binary_fill_holes, binary_erosion
from time import time
import nibabel as nib
from dataloader_aug_cc import regress
from torch.utils import data

def least_square_fitting(im, pred):
    im = torch.squeeze(im)

    pred_mask_temp = pred[4:5,...] > pred[3:4,...]
    normalizer = torch.median(im[pred_mask_temp[0,...]])

    im /= normalizer    
    pred = 100 * pred
    pred_mni = pred.permute([1, 2, 3, 0])

    y_pred_binary = torch.argmax(pred_mni[...,3:5], dim=-1)
    M = torch.tensor(binary_fill_holes((y_pred_binary > 0.5).detach().cpu().numpy()), device='cuda', dtype=torch.bool)

    MNISeg, Maff2 = MRIread('fitting/mni.seg.nii.gz', im_only=False, dtype='int32')
    MNISeg = torch.tensor(MNISeg, device='cuda', dtype=torch.int16)
    
    MNI, aff2 = MRIread('fitting/mni.nii.gz')
    A = np.linalg.inv(aff2)
    MNI = torch.tensor(MNI, device='cuda', dtype=torch.float32)
    A = torch.tensor(A, device='cuda', dtype=torch.float32)
    xx = pred_mni[:, :, :, 0][M]
    yy = pred_mni[:, :, :, 1][M]
    zz = pred_mni[:, :, :, 2][M]
    ii = A[0, 0] * xx + A[0, 1] * yy + A[0, 2] * zz + A[0, 3]
    jj = A[1, 0] * xx + A[1, 1] * yy + A[1, 2] * zz + A[1, 3]
    kk = A[2, 0] * xx + A[2, 1] * yy + A[2, 2] * zz + A[2, 3]

    ri = np.arange(pred_mni.shape[0]).astype('float'); ri -= np.mean(ri); ri /= 100
    rj = np.arange(pred_mni.shape[1]).astype('float'); rj -= np.mean(rj); rj /= 100
    rk = np.arange(pred_mni.shape[2]).astype('float'); rk -= np.mean(rk); rk /= 100
    i, j, k = np.meshgrid(ri, rj, rk, sparse=False, indexing='ij')
    i = torch.tensor(i, device='cuda', dtype=torch.float)[M]
    j = torch.tensor(j, device='cuda', dtype=torch.float)[M]
    k = torch.tensor(k, device='cuda', dtype=torch.float)[M]
    o = torch.ones_like(k)
    B = torch.stack([i, j, k, o], dim=1)

    P = torch.linalg.pinv(B)
    fit_x = P @ ii
    fit_y = P @ jj
    fit_z = P @ kk
    ii2aff = B @ fit_x
    jj2aff = B @ fit_y
    kk2aff = B @ fit_z

    valsAff = fast_3D_interp_torch(MNI, ii2aff, jj2aff, kk2aff, 'linear', device='cuda')
    DEFaff = torch.zeros_like(pred_mni[..., 0])
    DEFaff[M] = valsAff

    valsAff_seg = fast_3D_interp_torch(MNISeg, ii2aff, jj2aff, kk2aff, 'nearest', device='cuda')
    DEFaffseg = torch.zeros_like(pred_mni[..., 0])
    DEFaffseg[M] = valsAff_seg

    return DEFaff, DEFaffseg

if __name__ == "__main__":
    start_time = time()
    # input_path = '/autofs/vast/lemon/data_original_downloads/OASIS3/sub-0103/anat/sub-0103_T1w.nii.gz'
    # im, aff = MRIread(input_path, im_only=False, dtype='float')
    # im = torch.tensor(im, device='cuda', dtype=torch.float64).unsqueeze(0)
    # aff = torch.tensor(aff, device='cuda', dtype=torch.float64)

    training_set = regress('data_lists/regress/train_list.csv', 'data/', is_training=True)
    trainloader = data.DataLoader(training_set,batch_size=1,shuffle=True, drop_last=True) 

    batch = next(iter(trainloader))
    im, mask, target, seg, aff = batch
    im = im[0,:].to('cuda')
    aff = aff[0,:]

    model = UNet_3d(in_dim=1, out_dim=8, num_filters=4).to('cuda')
    model.load_state_dict(torch.load('experiments/regress/pre_train_l2_01_2/model_best.pth'))
    pred = model(im[None, ...].to(dtype=torch.float)) 
    channels_to_select = [0, 1, 2, 6, 7]
    pred = pred[0, channels_to_select, :, :]
    
    DEFaff, DEFaffseg = least_square_fitting(im, pred)
    end_time = time()
    print("LSF took {} seconds.".format(end_time-start_time))

    new_image = nib.Nifti1Image(DEFaff.cpu().detach().numpy(), affine=aff.cpu().detach().numpy())
    new_image.to_filename('samples/fitting_img.nii.gz')

    new_seg = nib.Nifti1Image(DEFaffseg.cpu().detach().numpy(), affine=aff.cpu().detach().numpy())
    new_seg.to_filename('samples/fitting_seg.nii.gz')