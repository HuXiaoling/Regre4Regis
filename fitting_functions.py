# fitting functions to make the framework end2end
import os, sys
import numpy as np
import torch
from scipy.interpolate import RegularGridInterpolator as rgi
from unet.unet_3d import UNet_3d
from scipy.ndimage import binary_fill_holes
from time import time
import nibabel as nib
from dataloader_aug_cc import regress
from torch.utils import data

def MRIread(filename, dtype=None, im_only=False):

    assert filename.endswith(('.nii', '.nii.gz', '.mgz')), 'Unknown data file: %s' % filename

    x = nib.load(filename)
    volume = x.get_fdata()
    aff = x.affine

    if dtype is not None:
        volume = volume.astype(dtype=dtype)

    if im_only:
        return volume
    else:
        return volume, aff

def fast_3D_interp_torch(X, II, JJ, KK, mode, device, default_value_linear=0.0):
    if mode=='nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        if len(X.shape)==3:
            X = X[..., None]
        Y = torch.zeros([*II.shape, X.shape[3]], dtype=torch.float, device=device)
        for channel in range(X.shape[3]):
            aux = X[:, :, :, channel]
            Y[...,channel] = aux[IIr, JJr, KKr]
        if Y.shape[-1] == 1:
            Y = Y[..., 0]

    elif mode=='linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz

        if len(X.shape)==3:
            X = X[..., None]

        Y = torch.zeros([*II.shape, X.shape[3]], dtype=torch.float, device=device)
        for channel in range(X.shape[3]):
            Xc = X[:, :, :, channel]

            c000 = Xc[fx, fy, fz]
            c100 = Xc[cx, fy, fz]
            c010 = Xc[fx, cy, fz]
            c110 = Xc[cx, cy, fz]
            c001 = Xc[fx, fy, cz]
            c101 = Xc[cx, fy, cz]
            c011 = Xc[fx, cy, cz]
            c111 = Xc[cx, cy, cz]

            c00 = c000 * wfx + c100 * wcx
            c01 = c001 * wfx + c101 * wcx
            c10 = c010 * wfx + c110 * wcx
            c11 = c011 * wfx + c111 * wcx

            c0 = c00 * wfy + c10 * wcy
            c1 = c01 * wfy + c11 * wcy

            c = c0 * wfz + c1 * wcz

            Yc = torch.zeros(II.shape, dtype=torch.float, device=device)
            Yc[ok] = c.float()
            Yc[~ok] = default_value_linear
            Y[...,channel] = Yc

        if Y.shape[-1]==1:
            Y = Y[...,0]

    else:
        raise Exception('mode must be linear or nearest')

    return Y

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

    # nonlinear fitting part (Demons, draft by Eugenio)
    if False:
        ii_res = ii - ii2aff
        jj_res = jj - jj2aff
        kk_res = kk - kk2aff
        ii_nonlin = gaussian_filter_torch(torch.clip(ii_res, min=-clipval, max=clipval), sigma=sigma, device='cuda')
        jj_nonlin = gaussian_filter_torch(torch.clip(jj_res, min=-clipval, max=clipval), sigma=sigma, device='cuda')
        kk_nonlin = gaussian_filter_torch(torch.clip(kk_res, min=-clipval, max=clipval), sigma=sigma, device='cuda')
        ii2aff += ii_nonlin
        jj2aff += jj_nonlin
        kk2aff += kk_nonlin
    
    # deformation
    valsAff = fast_3D_interp_torch(MNI, ii2aff, jj2aff, kk2aff, 'linear', device='cuda')
    DEFimg = torch.zeros_like(pred_mni[..., 0])
    DEFimg[M] = valsAff
    import pdb; pdb.set_trace()
    valsAff_seg = fast_3D_interp_torch(MNISeg, ii2aff, jj2aff, kk2aff, 'linear', device='cuda')
    DEFseg = torch.zeros_like(pred_mni[..., 0])
    DEFseg[M] = valsAff_seg

    return DEFimg, DEFseg

if __name__ == "__main__":
    start_time = time()
    # input_path = '/autofs/vast/lemon/data_original_downloads/OASIS3/sub-0103/anat/sub-0103_T1w.nii.gz'
    # im, aff = MRIread(input_path, im_only=False, dtype='float')
    # im = torch.tensor(im, device='cuda', dtype=torch.float64).unsqueeze(0)
    # aff = torch.tensor(aff, device='cuda', dtype=torch.float64)

    training_set = regress('data_lists/regress/train_list.csv', 'data/', is_training=True)
    trainloader = data.DataLoader(training_set,batch_size=4,shuffle=True, drop_last=True, pin_memory=False) 

    batch = next(iter(trainloader))
    im, mask, target, seg, aff = batch

    ori_image = nib.Nifti1Image(im[0,0,:,:,:].cpu().detach().numpy(), affine=aff[0])
    ori_image.to_filename('samples/ori_image.nii.gz')

    ori_seg = nib.Nifti1Image(seg[0,0,:,:,:].cpu().detach().numpy(), affine=aff[0])
    ori_seg.to_filename('samples/ori_seg.nii.gz')

    # im = im[0,:].to('cuda')
    aff = aff[0,:]

    model = UNet_3d(in_dim=1, out_dim=8, num_filters=4).to('cuda')
    model.load_state_dict(torch.load('experiments/regress/pre_train_l2_01_2/model_best.pth'))
    pred = model(im.to('cuda').to(dtype=torch.float)) 
    channels_to_select = [0, 1, 2, 6, 7]
    # pred = pred[0, channels_to_select, :, :]

    DEFimg, DEFseg = least_square_fitting(im[0,:].to('cuda'), pred[0, channels_to_select, :])
    end_time = time()
    print("LSF took {} seconds.".format(end_time-start_time))

    new_image = nib.Nifti1Image(DEFimg.cpu().detach().numpy(), affine=aff.cpu().detach().numpy())
    new_image.to_filename('samples/fitting_img.nii.gz')

    new_seg = nib.Nifti1Image(DEFseg.cpu().detach().numpy(), affine=aff.cpu().detach().numpy())
    new_seg.to_filename('samples/fitting_seg.nii.gz')
    import pdb; pdb.set_trace()