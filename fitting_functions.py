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
import torch.nn.functional as F
from utilities import onehot_encoding
import sys
sys.path.append('/autofs/space/durian_001/users/xh999/regre4regis/proj_supersynth_registration')
import ext.interpol

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

def make_gaussian_kernel(sigma, device):
    if type(sigma) is torch.Tensor:
        sigma = sigma.cpu()
    sl = int(np.ceil(3 * sigma))
    ts = torch.linspace(-sl, sl, 2*sl+1, dtype=torch.float, device=device)
    gauss = torch.exp((-(ts / sigma)**2 / 2))
    kernel = gauss / gauss.sum()
    return kernel

def gaussian_blur_3d(input, stds, device):
    from torch.nn.functional import conv3d
    blurred = input[None, None, :, :, :]
    if stds[0]>0:
        kx = make_gaussian_kernel(stds[0], device=device)
        blurred = conv3d(blurred, kx[None, None, :, None, None], stride=1, padding=(len(kx) // 2, 0, 0))
    if stds[1]>0:
        ky = make_gaussian_kernel(stds[1], device=device)
        blurred = conv3d(blurred, ky[None, None, None, :, None], stride=1, padding=(0, len(ky) // 2, 0))
    if stds[2]>0:
        kz = make_gaussian_kernel(stds[2], device=device)
        blurred = conv3d(blurred, kz[None, None, None, None, :], stride=1, padding=(0, 0, len(kz) // 2))
    return torch.squeeze(blurred)

def least_square_fitting(pred, aff2, MNISeg, nonlin=False):
 
    pred = 100 * pred
    pred_mni = pred.permute([1, 2, 3, 0])
    if pred_mni.shape[3] == 6:
        y_pred_binary = torch.argmax(pred_mni[...,4:6], dim=-1)
    else:
        y_pred_binary = torch.argmax(pred_mni[...,6:8], dim=-1)

    M = torch.tensor(binary_fill_holes((y_pred_binary > 0.5).detach().cpu().numpy()), device='cuda', dtype=torch.bool)

    A = np.linalg.inv(aff2)
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

    # P = inv(B^T * B) * B^T
    # P = torch.linalg.pinv(B)
    # fit_x = P @ ii
    # fit_y = P @ jj
    # fit_z = P @ kk

    if pred_mni.shape[3] == 6:
        sigma_coor = torch.exp((pred_mni[:, :, :, 3]/100)[M])
        weight = 1 / sigma_coor
        
        P = torch.linalg.inv((torch.transpose(B, 0 ,1) @ (torch.unsqueeze(weight, 1) * B))) @ torch.transpose(B, 0, 1)
        fit_x = P @ (weight * ii)
        fit_y = P @ (weight * jj)
        fit_z = P @ (weight * kk)

    else:
        sigma_coor = torch.exp((pred_mni[:, :, :, 3:6]/100)[M])
        weight = 1 / sigma_coor
        P_ii = torch.linalg.inv((torch.transpose(B, 0 ,1) @ (torch.unsqueeze(weight[:,0], 1) * B))) @ torch.transpose(B, 0, 1)
        P_jj = torch.linalg.inv((torch.transpose(B, 0 ,1) @ (torch.unsqueeze(weight[:,1], 1) * B))) @ torch.transpose(B, 0, 1)
        P_kk = torch.linalg.inv((torch.transpose(B, 0 ,1) @ (torch.unsqueeze(weight[:,2], 1) * B))) @ torch.transpose(B, 0, 1)

        fit_x = P_ii @ (weight[:,0] * ii)
        fit_y = P_jj @ (weight[:,1] * jj)
        fit_z = P_kk @ (weight[:,2] * kk)

    ii2aff = B @ fit_x
    jj2aff = B @ fit_y
    kk2aff = B @ fit_z

    # nonlinear fitting part (Demons, draft by Eugenio)
    # if False:
    #     ii_res = ii - ii2aff
    #     jj_res = jj - jj2aff
    #     kk_res = kk - kk2aff
    #     ii_nonlin = gaussian_filter_torch(torch.clip(ii_res, min=-clipval, max=clipval), sigma=sigma, device='cuda')
    #     jj_nonlin = gaussian_filter_torch(torch.clip(jj_res, min=-clipval, max=clipval), sigma=sigma, device='cuda')
    #     kk_nonlin = gaussian_filter_torch(torch.clip(kk_res, min=-clipval, max=clipval), sigma=sigma, device='cuda')
    #     ii2aff += ii_nonlin
    #     jj2aff += jj_nonlin
    #     kk2aff += kk_nonlin

    if nonlin:
        # Demons

        # sigma = 3
        
        # idef = ii - ii2aff
        # jdef = jj - jj2aff
        # kdef = kk - kk2aff

        # disp = torch.sqrt(torch.square(idef) + torch.square(jdef) + torch.square(kdef))
        # max_disp = torch.tensor(10.0, device='cuda')
        # toofar = disp>max_disp

        # new_idef = idef.clone()
        # new_jdef = jdef.clone()
        # new_kdef = kdef.clone()

        # new_idef[toofar] = (idef[toofar] / disp[toofar]) * max_disp
        # new_jdef[toofar] = (jdef[toofar] / disp[toofar]) * max_disp
        # new_kdef[toofar] = (kdef[toofar] / disp[toofar]) * max_disp

        # aux = torch.zeros_like(pred_mni[..., 0])
        # aux[M] = new_idef
        # num = gaussian_blur_3d(aux, [sigma, sigma, sigma], device='cuda')
        # den = gaussian_blur_3d(M.float(), [sigma, sigma, sigma], device='cuda')
        # new_idef = num[M] / den[M]
        # aux[M] = new_jdef
        # num = gaussian_blur_3d(aux, [sigma, sigma, sigma], device='cuda')
        # new_jdef = num[M] / den[M]
        # aux[M] = new_kdef
        # num = gaussian_blur_3d(aux, [sigma, sigma, sigma], device='cuda')
        # new_kdef = num[M] / den[M]

        # ii2demon = ii2aff + new_idef
        # jj2demon = jj2aff + new_jdef
        # kk2demon = kk2aff + new_kdef

        # valsDemon_seg = fast_3D_interp_torch(MNISeg, ii2demon, jj2demon, kk2demon, 'linear', device='cuda')
        # DEFseg = torch.zeros([pred_mni.shape[0], pred_mni.shape[1], pred_mni.shape[2], 32], device='cuda')
        # DEFseg[M] = valsDemon_seg

        # Bspline

        # clip  outliers
        idef = ii - ii2aff
        jdef = jj - jj2aff
        kdef = kk - kk2aff
        disp = disp = torch.sqrt(torch.square(idef) + torch.square(jdef) + torch.square(kdef))
        max_disp = torch.tensor(10.0, device='cuda')
        toofar = disp > max_disp
        idef[toofar] = (idef[toofar] / disp[toofar]) * max_disp
        jdef[toofar] = (jdef[toofar] / disp[toofar]) * max_disp
        kdef[toofar] = (kdef[toofar] / disp[toofar]) * max_disp
        iifixed = ii2aff + idef
        jjfixed = jj2aff + jdef
        kkfixed = kk2aff + kdef

        # fit bsplines
        small_shape = tuple(np.ceil(np.array(pred_mni.shape[:-1]) / 2.5).astype(int))
        iifixed_matrix = torch.zeros_like(pred_mni[..., 0])
        iifixed_matrix[M] = iifixed
        aux = ext.interpol.resize(iifixed_matrix, shape=small_shape, interpolation=3, prefilter=True)
        aux2 = ext.interpol.resize(aux, shape=pred_mni.shape[:-1], interpolation=3, prefilter=False)
        ii2_bspline = aux2[M]

        jjfixed_matrix = torch.zeros_like(pred_mni[..., 0])
        jjfixed_matrix[M] = jjfixed        
        aux = ext.interpol.resize(jjfixed_matrix, shape=small_shape, interpolation=3, prefilter=True)
        aux2 = ext.interpol.resize(aux, shape=pred_mni.shape[:-1], interpolation=3, prefilter=False)
        jj2_bspline = aux2[M]

        kkfixed_matrix = torch.zeros_like(pred_mni[..., 0])
        kkfixed_matrix[M] = kkfixed
        aux = ext.interpol.resize(kkfixed_matrix, shape=small_shape, interpolation=3, prefilter=True)
        aux2 = ext.interpol.resize(aux, shape=pred_mni.shape[:-1], interpolation=3, prefilter=False)
        kk2_bspline = aux2[M]

        vals_bspline = fast_3D_interp_torch(MNISeg, ii2_bspline, jj2_bspline, kk2_bspline, 'linear', 'cuda')
        DEFseg = torch.zeros([pred_mni.shape[0], pred_mni.shape[1], pred_mni.shape[2], 32], device='cuda')
        DEFseg[M] = vals_bspline

    else:
        # valsAff = fast_3D_interp_torch(MNI, ii2aff, jj2aff, kk2aff, 'linear', device='cuda')
        # DEFimg = torch.zeros_like(pred_mni[..., 0])
        # DEFimg[M] = valsAff

        valsAff_seg = fast_3D_interp_torch(MNISeg, ii2aff, jj2aff, kk2aff, 'linear', device='cuda')
        DEFseg = torch.zeros([pred_mni.shape[0], pred_mni.shape[1], pred_mni.shape[2], 32], device='cuda')
        DEFseg[M] = valsAff_seg

    return DEFseg

if __name__ == "__main__":
    start_time = time()

    training_set = regress('data_lists/regress/test_list.csv', 'data/', is_training=True)
    trainloader = data.DataLoader(training_set,batch_size=1,shuffle=True, drop_last=True, pin_memory=False) 

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

    # model = UNet_3d(in_dim=1, out_dim=6, num_filters=4).to('cuda')
    # model.load_state_dict(torch.load('experiments/regress/pre_train_single_sigma_l2_01_2/model_best.pth'))
    pred = model(im.to('cuda').to(dtype=torch.float)) 
    # channels_to_select = [0, 1, 2, 6, 7]

    # Define one-hot encoding   
    label_list_segmentation = [0, 14, 15, 16,
                2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 
                41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]

    n_labels = len(label_list_segmentation)

    # create look up table
    lut = torch.zeros(10000, dtype=torch.long, device='cuda')
    for l in range(n_labels):
        lut[label_list_segmentation[l]] = l

    onehotmatrix = torch.eye(n_labels, dtype=torch.float, device='cuda')
        
    ## Load MNI and MNI segmentation
    MNISeg, Maff2 = MRIread('fitting/mni.seg.nii.gz', im_only=False, dtype='int32')
    MNISeg = torch.tensor(MNISeg, device='cuda', dtype=torch.int16)
    MNISeg = MNISeg[None, None, ...]
    seg_onehot = onehot_encoding(MNISeg, onehotmatrix, lut)

    DEFseg = least_square_fitting(pred[0, :], Maff2, seg_onehot.squeeze().permute(1, 2, 3, 0), 'true').permute(3, 0, 1, 2)
    end_time = time()
    print("LSF took {} seconds.".format(end_time-start_time))

    deform_discrete_labels = torch.unsqueeze(torch.argmax(DEFseg, dim=0), dim=0).to(dtype=torch.int)
    new_seg = nib.Nifti1Image(deform_discrete_labels[0,:,:,:].cpu().detach().numpy(), affine=aff.cpu().detach().numpy())
    new_seg.to_filename('samples/fitting_seg.nii.gz')
    import pdb; pdb.set_trace()