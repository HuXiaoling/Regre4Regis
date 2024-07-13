import sys
# sys.path.append('/autofs/space/ballarat_001/users/kg149/proj_supersynth_registration')
sys.path.append('/autofs/space/durian_001/users/xh999/regre4regis/proj_supersynth_registration')

import torch
import os
import pandas as pd
from ext.unet3d.model import UNet3D
from SuperSynth.utils import MRIread, MRIwrite, torch_resize, align_volume_to_ref
import numpy as np
from torch.nn import Softmax
from SuperSynth.generators import  super_generator_hemi
from argparse import ArgumentParser
from scipy.ndimage import binary_fill_holes, binary_erosion
from SuperSynth.generators import fast_3D_interp_torch

from xiocode.dataloader_aug import regress
from xiocode.unet.unet_3d import UNet_3d
from xiocode.utilities import DC_and_CE_loss, SoftDiceLoss, softmax_helper
from scipy.interpolate import RegularGridInterpolator as rgi
from time import time
# ================================================================================================
#                                         Main Entrypoint
# ================================================================================================

def main():
    start_time = time()

    # parse arguments
    parser = ArgumentParser(description="Process some integers.")

    # Adding arguments with default values as specified in your JSON structure
    parser.add_argument("--input", type=str, default='/autofs/vast/lemon/data_original_downloads/OASIS3/sub-0103/anat/sub-0103_T1w.nii.gz' ,help="Image to process.")
    parser.add_argument("--input_seg", type=str, default='/autofs/space/rauma_001/users/op035/data/niftyreg_miccai/OASIS3/sub-0103/gt_seg_in_target_space.nii.gz' ,help="Image to process.")
    # parser.add_argument("--input", type=str, default='../proj_keymorph_atlas/data_reg/data/138627.image.nii.gz' ,help="Image to process.")
    # parser.add_argument("--input_seg", type=str, default='../proj_keymorph_atlas/data_reg/data/138627.image.nii.gz' ,help="Image to process.")
    parser.add_argument("--mni_img", type=str, default='./mni.nii.gz' ,help="MNI Image to process.")
    parser.add_argument("--mni_seg", type=str, default='./mni.seg.nii.gz' ,help="MNI Image to process.")
    parser.add_argument("--grid_seg_x", type=str, default='./mni.grid.x.nii.gz' ,help="MNI Image to process.")
    parser.add_argument("--grid_seg_y", type=str, default='./mni.grid.y.nii.gz' ,help="MNI Image to process.")
    parser.add_argument("--grid_seg_z", type=str, default='./mni.grid.z.nii.gz' ,help="MNI Image to process.")
    parser.add_argument("--output_dir", type=str, default='./output_regdir' ,help="MNI registration coordinates")
    # parser.add_argument("--model_file", type=str, default='./output_regdir' ,help="Path to model file")
    parser.add_argument("--cpu", action="store_true", help="Enforce running with CPU rather than GPU.")
    parser.add_argument("--threads", type=int, default=-1, help="Number of threads to be used by PyTorch when running on CPU (-1 for maximum).")

    # Common parameters with defaults
    # parser.add_argument("--activity", type=str, default="validation", help="Activity type")
    # parser.add_argument("--dataset", type=str, default="regress", help="Dataset name")
    # parser.add_argument("--num_classes", type=int, default=2, help="Number of classes")
    # parser.add_argument("--files", type=str, default="../proj_keymorph_atlas/data_reg/data/", help="Image file path")
    parser.add_argument("--checkpoint_restore", type=str, default="../experiments/regress/train_outputs_full_aug_yogurt_2/model_best.pth", help="Path to checkpoint for restoring")
    

    # Validation parameters with defaults
    # parser.add_argument("--validation_datalist", type=str, default="xiocode/data_lists/regress/test_list.csv", help="Validation datalist")
    # parser.add_argument("--output_folder", type=str, default="./train_outputs_full_aug_yogurt_2", help="Output folder for validation")
    # parser.add_argument("--batch_size", type=int, default=1, help="Batch size for validation")

    # Parse the arguments
    # args = parser.parse_args()

    args = vars(parser.parse_args())
    
    csv_file_path = args['output_dir'] + '/metrics.csv'
    results_df = pd.DataFrame(columns=['Algorithm', 'Sigma or spacing or Order', 'Membrane Energy', 'Percentage of Negative Jacobian'])

    # enforce CPU processing if necessary
    if args['cpu']:
        print('using CPU, hiding all CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        device = 'cpu'
    else:
        device = 'cuda'


    # limit the number of threads to be used if running on CPU
    if args['threads'] < 0:
        args['threads'] = os.cpu_count()
        print('using all available threads ( %s )' % args['threads'])
    else:
        print('using %s thread(s)' % args['threads'])
    torch.set_num_threads(args['threads'])

    if os.path.isdir(args['output_dir']) is False:
        os.mkdir(args['output_dir'])

    # down to business
    with torch.no_grad():

        print('Reading, resampling, and padding input image')
        im, aff = MRIread(args['input'], im_only=False, dtype='float')
        im = torch.tensor(np.squeeze(im), dtype=torch.float32, device=device)
        im_native = im.clone()
        aff_native = np.copy(aff)
        im, aff = torch_resize(im, aff, 1.0, device)
        im, aff = align_volume_to_ref(im, aff, aff_ref=np.eye(4), return_aff=True, n_dims=3)
        im_orig = im.clone()
        
        imseg, affseg = MRIread(args['input_seg'], im_only=False, dtype='float')
        imseg = torch.tensor(np.squeeze(imseg), dtype=torch.float32, device=device)
        imseg, affseg = torch_resize(imseg, affseg, 1.0, device)
        imseg, affseg = align_volume_to_ref(imseg, affseg, aff_ref=np.eye(4), return_aff=True, n_dims=3)
        imseg_orig = imseg.clone()

        while len(im.shape) > 3:  # in case it's rgb
            im = torch.mean(im, axis=-1)
        im = im - torch.min(im)
        im = im / torch.max(im)
        W = (np.ceil(np.array(im.shape) / 32.0) * 32).astype('int')
        idx = np.floor((W - im.shape) / 2).astype('int')
        S = torch.zeros(*W, dtype=torch.float32, device=device)
        S[idx[0]:idx[0] + im.shape[0], idx[1]:idx[1] + im.shape[1], idx[2]:idx[2] + im.shape[2]] = im

        print('Preparing model and loading weights')

        model = UNet_3d(in_dim=1, out_dim=5, num_filters=4).to(torch.device(device))        

        if args['checkpoint_restore'] != "":
            model.load_state_dict(torch.load(args['checkpoint_restore']), strict=True)
        else:
            print("No model found!")
            sys.exit()

        print('Pushing data through the CNN')
        pred_temp = model(S[None, None, ...])
        pred_mask_temp = pred_temp[:,4:5,...] > pred_temp[:,3:4,...]
        normalizer = torch.median(S[pred_mask_temp[0,0,...]])
        S /= normalizer
        pred = model(S[None, None, ...])        
        # pred2 = torch.flip(model(torch.flip(S, [0])[None, None, ...]), [2])
        # pred2[:, 0, :, :, :] = -pred2[:, 0, :, :, :]
        # pred = 50 * pred1 + 50 * pred2
        pred = 100 * pred
        pred_mni = pred[0, :, idx[0]:idx[0] + im.shape[0], idx[1]:idx[1] + im.shape[1], idx[2]:idx[2] + im.shape[2]].permute([1, 2, 3, 0])

        if True: # use_native:
            # import pdb; pdb.set_trace()
            TT = np.linalg.inv(aff) @ aff_native
            inter = rgi((np.arange(pred_mni.shape[0]), np.arange(pred_mni.shape[1]), np.arange(pred_mni.shape[2])), pred_mni.detach().cpu().numpy(), bounds_error=False, fill_value=0)
            ig, jg, kg = np.meshgrid(np.arange(im_native.shape[0]), np.arange(im_native.shape[1]), np.arange(im_native.shape[2]), indexing='ij', sparse=False)
            iig = TT[0,0] * ig + TT[0,1] * jg + TT[0,2] * kg + TT[0,3]
            jjg = TT[1,0] * ig + TT[1,1] * jg + TT[1,2] * kg + TT[1,3]
            kkg = TT[2,0] * ig + TT[2,1] * jg + TT[2,2] * kg + TT[2,3]
            pred_mni_resampled = torch.tensor(inter((iig, jjg, kkg)), device=device, dtype=torch.float32)
            pred_mni = pred_mni_resampled
            aff = aff_native
            im = im_native







        MRIwrite(pred_mni.detach().cpu().numpy(), aff, args['output_dir'] + '/predicted_mni_coords.nii.gz')

        print('deforming MNI')
        y_pred_binary = torch.argmax(pred_mni[...,3:5], dim=-1)
        
        M = torch.tensor(binary_fill_holes((y_pred_binary > 0.5).detach().cpu().numpy()), device=device, dtype=torch.bool)
        MRIwrite(M.detach().cpu().numpy(), aff, args['output_dir'] + '/predicted_mask.nii.gz')


        MNISeg, Maff2 = MRIread(args['mni_seg'], im_only=False, dtype='int32')
        MNISeg = torch.tensor(MNISeg, device=device, dtype=torch.int16)
        
        gridSeg_x, graff2 = MRIread(args['grid_seg_x'], im_only=False, dtype='int16')
        gridSeg_x = torch.tensor(gridSeg_x, device=device, dtype=torch.int16)
        
        gridSeg_y, graff2 = MRIread(args['grid_seg_y'], im_only=False, dtype='int16')
        gridSeg_y = torch.tensor(gridSeg_y, device=device, dtype=torch.int16)
        
        gridSeg_z, graff2 = MRIread(args['grid_seg_z'], im_only=False, dtype='int16')
        gridSeg_z = torch.tensor(gridSeg_z, device=device, dtype=torch.int16)
        
        MNI, aff2 = MRIread(args['mni_img'])
        A = np.linalg.inv(aff2)
        MNI = torch.tensor(MNI, device=device, dtype=torch.float32)
        A = torch.tensor(A, device=device, dtype=torch.float32)
        xx = pred_mni[:, :, :, 0][M]
        yy = pred_mni[:, :, :, 1][M]
        zz = pred_mni[:, :, :, 2][M]
        ii = A[0, 0] * xx + A[0, 1] * yy + A[0, 2] * zz + A[0, 3]
        jj = A[1, 0] * xx + A[1, 1] * yy + A[1, 2] * zz + A[1, 3]
        kk = A[2, 0] * xx + A[2, 1] * yy + A[2, 2] * zz + A[2, 3]

        vals = fast_3D_interp_torch(MNI, ii, jj, kk, 'linear', device)
        DEF = torch.zeros_like(pred_mni[..., 0])
        DEF[M] = vals
        MRIwrite(DEF.detach().cpu().numpy(), aff, args['output_dir'] + '/direct_deformation.nii.gz')

        vals_seg = fast_3D_interp_torch(MNISeg, ii, jj, kk, 'nearest', device)
        DEFseg = torch.zeros_like(pred_mni[..., 0])
        DEFseg[M] = vals_seg
        MRIwrite(DEFseg.detach().cpu().numpy(), aff, args['output_dir'] + '/direct_deformation_seg.nii.gz')
        
        vals_seg = fast_3D_interp_torch(gridSeg_x, ii, jj, kk, 'nearest', device)
        DEFseg = torch.zeros_like(pred_mni[..., 0])
        DEFseg[M] = vals_seg
        MRIwrite(DEFseg.detach().cpu().numpy(), aff, args['output_dir'] + '/direct_deformation_gridSeg_x.nii.gz')

        vals_seg = fast_3D_interp_torch(gridSeg_y, ii, jj, kk, 'nearest', device)
        DEFseg = torch.zeros_like(pred_mni[..., 0])
        DEFseg[M] = vals_seg
        MRIwrite(DEFseg.detach().cpu().numpy(), aff, args['output_dir'] + '/direct_deformation_gridSeg_y.nii.gz')

        vals_seg = fast_3D_interp_torch(gridSeg_z, ii, jj, kk, 'nearest', device)
        DEFseg = torch.zeros_like(pred_mni[..., 0])
        DEFseg[M] = vals_seg
        MRIwrite(DEFseg.detach().cpu().numpy(), aff, args['output_dir'] + '/direct_deformation_gridSeg_z.nii.gz')

        CNN_inference = time()
        print("The cnn inference took {} seconds.".format(CNN_inference-start_time))

        print('Fitting affine transform with least squares')
        ri = np.arange(pred_mni.shape[0]).astype('float'); ri -= np.mean(ri); ri /= 100
        rj = np.arange(pred_mni.shape[1]).astype('float'); rj -= np.mean(rj); rj /= 100
        rk = np.arange(pred_mni.shape[2]).astype('float'); rk -= np.mean(rk); rk /= 100
        i, j, k = np.meshgrid(ri, rj, rk, sparse=False, indexing='ij')
        i = torch.tensor(i, device=device, dtype=torch.float)[M]
        j = torch.tensor(j, device=device, dtype=torch.float)[M]
        k = torch.tensor(k, device=device, dtype=torch.float)[M]
        o = torch.ones_like(k)
        B = torch.stack([i, j, k, o], dim=1)

        P = torch.linalg.pinv(B)
        fit_x = P @ ii
        fit_y = P @ jj
        fit_z = P @ kk
        ii2aff = B @ fit_x
        jj2aff = B @ fit_y
        kk2aff = B @ fit_z

        valsAff = fast_3D_interp_torch(MNI, ii2aff, jj2aff, kk2aff, 'linear', device)
        DEFaff = torch.zeros_like(pred_mni[..., 0])
        DEFaff[M] = valsAff
        MRIwrite(DEFaff.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_least_squares.nii.gz')

        valsAff_seg = fast_3D_interp_torch(MNISeg, ii2aff, jj2aff, kk2aff, 'nearest', device)
        DEFaffseg = torch.zeros_like(pred_mni[..., 0])
        DEFaffseg[M] = valsAff_seg
        MRIwrite(DEFaffseg.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_least_squares_seg.nii.gz')
        
        valsAff_seg = fast_3D_interp_torch(gridSeg_x, ii2aff, jj2aff, kk2aff, 'nearest', device)
        DEFaffseg = torch.zeros_like(pred_mni[..., 0])
        DEFaffseg[M] = valsAff_seg
        MRIwrite(DEFaffseg.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_least_squares_gridSeg_x.nii.gz')
        import pdb; pdb.set_trace() 
        valsAff_seg = fast_3D_interp_torch(gridSeg_y, ii2aff, jj2aff, kk2aff, 'nearest', device)
        DEFaffseg = torch.zeros_like(pred_mni[..., 0])
        DEFaffseg[M] = valsAff_seg
        MRIwrite(DEFaffseg.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_least_squares_gridSeg_y.nii.gz')

        valsAff_seg = fast_3D_interp_torch(gridSeg_z, ii2aff, jj2aff, kk2aff, 'nearest', device)
        DEFaffseg = torch.zeros_like(pred_mni[..., 0])
        DEFaffseg[M] = valsAff_seg
        MRIwrite(DEFaffseg.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_least_squares_gridSeg_z.nii.gz')

        least_square = time()
        print("The affine_least_square took {} seconds.".format(least_square-CNN_inference))

        print('Fitting affine transform with RANSAC') # see wikipedia article for nomenclature
        if False: # all samples
            from numpy.random import default_rng
            rng = default_rng()
            K = 100
            N = 1000
            D = 250000 # in voxels
            D2max = 50
            best_fit_x = []
            best_fit_y = []
            best_fit_z = []
            best_err = torch.tensor(10000)
            for its in range(K):
                ids = torch.tensor(rng.permutation(len(i)), device=device)
                maybe_inliers = ids[:N]
                B_maybe = torch.stack([i[maybe_inliers], j[maybe_inliers], k[maybe_inliers], o[maybe_inliers]], dim=1)
                P_maybe = torch.linalg.pinv(B_maybe)
                fit_x_maybe = P_maybe @ ii[maybe_inliers]
                fit_y_maybe = P_maybe @ jj[maybe_inliers]
                fit_z_maybe = P_maybe @ kk[maybe_inliers]

                B_test = torch.stack([i[ids[N:]], j[ids[N:]], k[ids[N:]], o[ids[N:]]], dim=1)
                err_i = (B_test @ fit_x_maybe - ii[ids[N:]])
                err_j = (B_test @ fit_y_maybe - jj[ids[N:]])
                err_k = (B_test @ fit_z_maybe - kk[ids[N:]])
                err_sq = err_i * err_i + err_j * err_j + err_k * err_k
                inlier_ids = ids[N:][err_sq<D2max]
                if inlier_ids.size()[0] > D:
                    inlier_points = torch.concat([maybe_inliers, inlier_ids])
                    B_better = torch.stack([i[inlier_points], j[inlier_points], k[inlier_points], o[inlier_points]], dim=1)
                    P_better = torch.linalg.pinv(B_better)
                    fit_x_better = P_better @ ii[inlier_points]
                    fit_y_better = P_better @ jj[inlier_points]
                    fit_z_better = P_better @ kk[inlier_points]
                    err_i = (B_better @ fit_x_better - ii[inlier_points])
                    err_j = (B_better @ fit_y_better - jj[inlier_points])
                    err_k = (B_better @ fit_z_better - kk[inlier_points])
                    err_sq = err_i * err_i + err_j * err_j + err_k * err_k
                    err_total = torch.mean(err_sq)
                    if err_total<best_err:
                        best_err = err_total
                        best_fit_x = fit_x_better
                        best_fit_y = fit_y_better
                        best_fit_z = fit_z_better
                print('RANSAC iteration ' + str(its+1) + ' of ' + str(K) + ': loss is ' + str(best_err))

            B = torch.stack([i, j, k, o], dim=1)
            ii2aff_ransac = B @ best_fit_x
            jj2aff_ransac = B @ best_fit_y
            kk2aff_ransac = B @ best_fit_z
            valsAff_ransac = fast_3D_interp_torch(MNI, ii2aff_ransac, jj2aff_ransac, kk2aff_ransac, 'linear', device)
            DEFaff_ransac = torch.zeros_like(pred_mni[..., 0])
            DEFaff_ransac[M] = valsAff_ransac
            MRIwrite(DEFaff_ransac.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_ransac.nii.gz')

        else: # subset of S samples
            from numpy.random import default_rng
            rng = default_rng()
            S = 50000 # number of voxels to use
            K = 100 # maximum number
            N = 500 # number of samples in every attempt
            D = 20000  # minimum number of inliers to consider solution (in voxels)
            D2max = 50 # thrshold for inlinear (in squared distance)
            best_fit_x = []
            best_fit_y = []
            best_fit_z = []
            best_err = torch.tensor(10000)
            ids = torch.tensor(rng.permutation(len(i)), device=device)
            i_ransac = i[ids[:S]]
            j_ransac = j[ids[:S]]
            k_ransac = k[ids[:S]]
            o_ransac = o[ids[:S]]
            ii_ransac = ii[ids[:S]]
            jj_ransac = jj[ids[:S]]
            kk_ransac = kk[ids[:S]]
            for its in range(K):
                ids = torch.tensor(rng.permutation(S), device=device)
                maybe_inliers = ids[:N]
                B_maybe = torch.stack([i_ransac[maybe_inliers], j_ransac[maybe_inliers], k_ransac[maybe_inliers], o_ransac[maybe_inliers]], dim=1)
                P_maybe = torch.linalg.pinv(B_maybe)
                fit_x_maybe = P_maybe @ ii_ransac[maybe_inliers]
                fit_y_maybe = P_maybe @ jj_ransac[maybe_inliers]
                fit_z_maybe = P_maybe @ kk_ransac[maybe_inliers]

                B_test = torch.stack([i_ransac[ids[N:]], j_ransac[ids[N:]], k_ransac[ids[N:]], o_ransac[ids[N:]]], dim=1)
                err_i = (B_test @ fit_x_maybe - ii_ransac[ids[N:]])
                err_j = (B_test @ fit_y_maybe - jj_ransac[ids[N:]])
                err_k = (B_test @ fit_z_maybe - kk_ransac[ids[N:]])
                err_sq = err_i * err_i + err_j * err_j + err_k * err_k
                inlier_ids = ids[N:][err_sq < D2max]
                if inlier_ids.size()[0] > D:
                    inlier_points = torch.concat([maybe_inliers, inlier_ids])
                    B_better = torch.stack([i_ransac[inlier_points], j_ransac[inlier_points], k_ransac[inlier_points], o_ransac[inlier_points]], dim=1)
                    P_better = torch.linalg.pinv(B_better)
                    fit_x_better = P_better @ ii_ransac[inlier_points]
                    fit_y_better = P_better @ jj_ransac[inlier_points]
                    fit_z_better = P_better @ kk_ransac[inlier_points]
                    err_i = (B_better @ fit_x_better - ii_ransac[inlier_points])
                    err_j = (B_better @ fit_y_better - jj_ransac[inlier_points])
                    err_k = (B_better @ fit_z_better - kk_ransac[inlier_points])
                    err_sq = err_i * err_i + err_j * err_j + err_k * err_k
                    err_total = torch.mean(err_sq)
                    if err_total < best_err:
                        best_err = err_total
                        best_fit_x = fit_x_better
                        best_fit_y = fit_y_better
                        best_fit_z = fit_z_better
                # print('RANSAC iteration ' + str(its + 1) + ' of ' + str(K) + ': loss is ' + str(best_err))

            B = torch.stack([i, j, k, o], dim=1)
            ii2aff_ransac = B @ best_fit_x
            jj2aff_ransac = B @ best_fit_y
            kk2aff_ransac = B @ best_fit_z
            valsAff_ransac = fast_3D_interp_torch(MNI, ii2aff_ransac, jj2aff_ransac, kk2aff_ransac, 'linear', device)
            DEFaff_ransac = torch.zeros_like(pred_mni[..., 0])
            DEFaff_ransac[M] = valsAff_ransac
            MRIwrite(DEFaff_ransac.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_ransac.nii.gz')

            valsAff_ransac_seg = fast_3D_interp_torch(MNISeg, ii2aff_ransac, jj2aff_ransac, kk2aff_ransac, 'nearest', device)
            DEFaff_ransacseg = torch.zeros_like(pred_mni[..., 0])
            DEFaff_ransacseg[M] = valsAff_ransac_seg
            MRIwrite(DEFaff_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_ransac_seg.nii.gz')
            
            valsAff_ransac_seg = fast_3D_interp_torch(gridSeg_x, ii2aff_ransac, jj2aff_ransac, kk2aff_ransac, 'linear', device)
            DEFaff_ransacseg = torch.zeros_like(pred_mni[..., 0])
            DEFaff_ransacseg[M] = valsAff_ransac_seg
            MRIwrite(DEFaff_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_ransac_gridSeg_x.nii.gz')

            valsAff_ransac_seg = fast_3D_interp_torch(gridSeg_y, ii2aff_ransac, jj2aff_ransac, kk2aff_ransac, 'linear', device)
            DEFaff_ransacseg = torch.zeros_like(pred_mni[..., 0])
            DEFaff_ransacseg[M] = valsAff_ransac_seg
            MRIwrite(DEFaff_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_ransac_gridSeg_y.nii.gz')

            valsAff_ransac_seg = fast_3D_interp_torch(gridSeg_z, ii2aff_ransac, jj2aff_ransac, kk2aff_ransac, 'linear', device)
            DEFaff_ransacseg = torch.zeros_like(pred_mni[..., 0])
            DEFaff_ransacseg[M] = valsAff_ransac_seg
            MRIwrite(DEFaff_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/affine_deformation_ransac_gridSeg_z.nii.gz')

        ransac = time()
        print("The ransac took {} seconds.".format(ransac - least_square))
        # RANSAC done
        print('Fitting nonlinear transform, demons style')
            
        all_sigma = [1, 3, 5]

        for sigma in all_sigma:

            demon_style_start = time()
            
            idef = ii - ii2aff_ransac
            jdef = jj - jj2aff_ransac
            kdef = kk - kk2aff_ransac
            disp = torch.sqrt(idef*idef + jdef * jdef + kdef * kdef)
            max_disp = torch.tensor(10.0, device=device)
            toofar = disp>max_disp

            idef[toofar] = (idef[toofar] / disp[toofar]) * max_disp
            jdef[toofar] = (jdef[toofar] / disp[toofar]) * max_disp
            kdef[toofar] = (kdef[toofar] / disp[toofar]) * max_disp

            aux = torch.zeros_like(pred_mni[..., 0])
            aux[M] = idef
            num = gaussian_blur_3d(aux, [sigma, sigma, sigma], device)
            den = gaussian_blur_3d(M.float(), [sigma, sigma, sigma], device)
            idef = num[M] / den[M]
            aux[M] = jdef
            num = gaussian_blur_3d(aux, [sigma, sigma, sigma], device)
            jdef = num[M] / den[M]
            aux[M] = kdef
            num = gaussian_blur_3d(aux, [sigma, sigma, sigma], device)
            kdef = num[M] / den[M]

            ii2demon = ii2aff_ransac + idef
            jj2demon = jj2aff_ransac + jdef
            kk2demon = kk2aff_ransac + kdef

            valsDemon = fast_3D_interp_torch(MNI, ii2demon, jj2demon, kk2demon, 'linear', device)
            DEFdemon = torch.zeros_like(pred_mni[..., 0])
            DEFdemon[M] = valsDemon
            # MRIwrite(DEFdemon.detach().cpu().numpy(), aff, args['output_dir'] + '/' + str(sigma) + '_demons_like_deformation.nii.gz')
            MRIwrite(DEFdemon.detach().cpu().numpy(), aff, args['output_dir'] + '/demons_like_deformation_' + str(sigma) + '.nii.gz')


            valsDemon_seg = fast_3D_interp_torch(MNISeg, ii2demon, jj2demon, kk2demon, 'nearest', device)
            DEFdemon_ransacseg = torch.zeros_like(pred_mni[..., 0])
            DEFdemon_ransacseg[M] = valsDemon_seg
            # MRIwrite(DEFdemon_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/' + str(sigma) + '_demons_like_deformation_seg.nii.gz')
            MRIwrite(DEFdemon_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/demons_like_deformation_seg_' + str(sigma) + '.nii.gz')
            
            valsDemon_seg = fast_3D_interp_torch(gridSeg_x, ii2demon, jj2demon, kk2demon, 'linear', device)
            DEFdemon_ransacseg = torch.zeros_like(pred_mni[..., 0])
            DEFdemon_ransacseg[M] = valsDemon_seg
            # MRIwrite(DEFdemon_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/' + str(sigma) + '_demons_like_deformation_gridSeg.nii.gz')
            MRIwrite(DEFdemon_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/demons_like_deformation_gridSeg_x_' + str(sigma) + '.nii.gz')

            valsDemon_seg = fast_3D_interp_torch(gridSeg_y, ii2demon, jj2demon, kk2demon, 'linear', device)
            DEFdemon_ransacseg = torch.zeros_like(pred_mni[..., 0])
            DEFdemon_ransacseg[M] = valsDemon_seg
            # MRIwrite(DEFdemon_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/' + str(sigma) + '_demons_like_deformation_gridSeg.nii.gz')
            MRIwrite(DEFdemon_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/demons_like_deformation_gridSeg_y_' + str(sigma) + '.nii.gz')

            valsDemon_seg = fast_3D_interp_torch(gridSeg_z, ii2demon, jj2demon, kk2demon, 'linear', device)
            DEFdemon_ransacseg = torch.zeros_like(pred_mni[..., 0])
            DEFdemon_ransacseg[M] = valsDemon_seg
            # MRIwrite(DEFdemon_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/' + str(sigma) + '_demons_like_deformation_gridSeg.nii.gz')
            MRIwrite(DEFdemon_ransacseg.detach().cpu().numpy(), aff, args['output_dir'] + '/demons_like_deformation_gridSeg_z_' + str(sigma) + '.nii.gz')

            
            pct_neg_J, membrane_energy = get_metrics(ii2demon, ii2aff_ransac, jj2demon, jj2aff_ransac, kk2demon, kk2aff_ransac, M, device)     
            print('Membrane energy for sigma ' + str(sigma) + ' is ' + str(membrane_energy.item()) + ' and pct of negative J is ' + str(pct_neg_J.item()))

            results_df.loc[len(results_df)] = {
                'Algorithm': 'Demons',
                'Sigma or spacing or Order': str(sigma),
                'Membrane Energy': membrane_energy.item(),
                'Percentage of Negative Jacobian': pct_neg_J.item()
            }

            demon_style_end = time()
            print("The demon style took {} seconds.".format(demon_style_end-demon_style_start))

        print('Fitting diffeomorphic nonlinear transform, with log-Euclidean poly-affine')
        from scipy.linalg import logm
        all_spac = [5, 10, 20]
        for spac in all_spac:

            poly_affine_start = time()

            sigma = float(spac) / 2.0
            min_valid = 100 # minimum number of voxels inside mask to make fit (must be at least 4)
            idef = ii - ii2aff_ransac
            jdef = jj - jj2aff_ransac
            kdef = kk - kk2aff_ransac
            disp = torch.sqrt(idef * idef + jdef * jdef + kdef * kdef)
            max_disp = torch.tensor(10.0, device=device)
            toofar = disp > max_disp
            idef[toofar] = (idef[toofar] / disp[toofar]) * max_disp
            jdef[toofar] = (jdef[toofar] / disp[toofar]) * max_disp
            kdef[toofar] = (kdef[toofar] / disp[toofar]) * max_disp
            Idef = torch.zeros(M.shape, device=device, dtype=torch.float)
            Jdef = torch.zeros(M.shape, device=device, dtype=torch.float)
            Kdef = torch.zeros(M.shape, device=device, dtype=torch.float)
            Idef[M] = idef; Jdef[M] = jdef; Kdef[M] = kdef;
            ri = np.arange(pred_mni.shape[0]).astype('float');
            rj = np.arange(pred_mni.shape[1]).astype('float');
            rk = np.arange(pred_mni.shape[2]).astype('float');
            I, J, K = np.meshgrid(ri, rj, rk, sparse=False, indexing='ij')
            I = torch.tensor(I, device=device, dtype=torch.float)
            J = torch.tensor(J, device=device, dtype=torch.float)
            K = torch.tensor(K, device=device, dtype=torch.float)
            ic = torch.mean(I); jc= torch.mean(J); kc = torch.mean(K);
            I -= ic; J-= jc; K-=kc;

            NUM = torch.zeros([*M.shape,3], device=device, dtype=torch.float)
            DEN = 1e-6 + torch.zeros(M.shape, device=device, dtype=torch.float)
            i1 = 0; i2 = spac;
            while i2 <= M.shape[0]:
                j1 = 0; j2 = spac;
                while j2 <= M.shape[1]:
                    k1 = 0; k2 = spac;
                    while k2 <= M.shape[2]:
                        # print(str(i1) + ' ' + str(j1) + ' ' + str(k1))
                        mask = M[i1:i2, j1:j2, k1:k2]
                        nvox = torch.sum(mask)
                        if  nvox > min_valid:
                            im = I[i1:i2, j1:j2, k1:k2][mask]
                            jm = J[i1:i2, j1:j2, k1:k2][mask]
                            km = K[i1:i2, j1:j2, k1:k2][mask]
                            om = torch.ones_like(km, device=device)
                            idefM = im + Idef[i1:i2, j1:j2, k1:k2][mask]
                            jdefM = jm + Jdef[i1:i2, j1:j2, k1:k2][mask]
                            kdefM = km + Kdef[i1:i2, j1:j2, k1:k2][mask]
                            Bm = torch.stack([im, jm, km, om], dim=1)
                            Pm = torch.linalg.pinv(Bm)
                            fit_x_M = Pm @ idefM;
                            fit_y_M = Pm @ jdefM;
                            fit_z_M = Pm @ kdefM;
                            affine = torch.stack([fit_x_M, fit_y_M, fit_z_M, torch.tensor([0,0,0,1], device=device)], axis=0)
                            lm = np.real(logm(affine.cpu(), disp=False)[0]) # TODO: faster in torch?
                            L = lm[:3, :3]
                            v = lm[:3,3]
                            dx = I + (ic - 0.5 * (i1+i2)); dy = J + (jc - 0.5 * (j1+j2)); dz = K + (kc - 0.5 * (k1+k2))
                            weight_map = nvox * torch.exp(-0.5 * (dx*dx+dy*dy+dz*dz) /sigma/sigma)
                            DEN += weight_map
                            for c in range(3):
                                NUM[:,:,:,c] += (weight_map * ( v[c] + L[c, 0] * I + L[c, 1] * J + L[c, 2] * K ))

                        k1 = k2; k2 = k1 + spac
                    j1 = j2; j2 = j1 + spac
                i1 = i2; i2 = i1 + spac

            SVF = NUM / DEN[:, :, :, None]

            # Scale and square
            steps = 8
            FIELD = SVF / (2.0 ** float(steps))
            for s in range(steps):
                FIELD += fast_3D_interp_torch(FIELD, (I+ic) + FIELD[:,:,:,0], (J+jc) + FIELD[:,:,:,1], (K+kc) + FIELD[:,:,:,2], 'linear', device)

            ii2_logpolyaffine = ii2aff_ransac + FIELD[:, :, :, 0][M]
            jj2_logpolyaffine = jj2aff_ransac + FIELD[:, :, :, 1][M]
            kk2_logpolyaffine = kk2aff_ransac + FIELD[:, :, :, 2][M]
            vals_logpolyaffine = fast_3D_interp_torch(MNI, ii2_logpolyaffine, jj2_logpolyaffine, kk2_logpolyaffine, 'linear', device)
            DEF_logpolyaffine = torch.zeros_like(pred_mni[..., 0])
            DEF_logpolyaffine[M] = vals_logpolyaffine
            # MRIwrite(DEF_logpolyaffine.detach().cpu().numpy(), aff, args['output_dir'] + '/' + str(spac) + '_log_polyaffine_deformation.nii.gz')
            MRIwrite(DEF_logpolyaffine.detach().cpu().numpy(), aff, args['output_dir'] + '/log_polyaffine_deformation_' + str(spac) + '.nii.gz')

            vals_logpolyaffine_seg = fast_3D_interp_torch(MNISeg, ii2_logpolyaffine, jj2_logpolyaffine, kk2_logpolyaffine, 'nearest', device)
            DEF_logpolyaffine_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_logpolyaffine_seg[M] = vals_logpolyaffine_seg
            # MRIwrite(DEF_logpolyaffine_seg.detach().cpu().numpy(), aff, args['output_dir'] + '/' + str(spac) + '_log_polyaffine_deformation_seg.nii.gz')
            MRIwrite(DEF_logpolyaffine_seg.detach().cpu().numpy(), aff, args['output_dir'] + '/log_polyaffine_deformation_seg_' + str(spac) + '.nii.gz')
            
            vals_logpolyaffine_seg = fast_3D_interp_torch(gridSeg_x, ii2_logpolyaffine, jj2_logpolyaffine, kk2_logpolyaffine, 'linear', device)
            DEF_logpolyaffine_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_logpolyaffine_seg[M] = vals_logpolyaffine_seg
            # MRIwrite(DEF_logpolyaffine_seg.detach().cpu().numpy(), aff, args['output_dir'] + '/' + str(spac) + '_log_polyaffine_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_logpolyaffine_seg.detach().cpu().numpy(), aff, args['output_dir'] + '/log_polyaffine_deformation_gridSeg_x_' + str(spac) + '.nii.gz')

            vals_logpolyaffine_seg = fast_3D_interp_torch(gridSeg_y, ii2_logpolyaffine, jj2_logpolyaffine, kk2_logpolyaffine, 'linear', device)
            DEF_logpolyaffine_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_logpolyaffine_seg[M] = vals_logpolyaffine_seg
            # MRIwrite(DEF_logpolyaffine_seg.detach().cpu().numpy(), aff, args['output_dir'] + '/' + str(spac) + '_log_polyaffine_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_logpolyaffine_seg.detach().cpu().numpy(), aff, args['output_dir'] + '/log_polyaffine_deformation_gridSeg_y_' + str(spac) + '.nii.gz')

            vals_logpolyaffine_seg = fast_3D_interp_torch(gridSeg_z, ii2_logpolyaffine, jj2_logpolyaffine, kk2_logpolyaffine, 'linear', device)
            DEF_logpolyaffine_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_logpolyaffine_seg[M] = vals_logpolyaffine_seg
            # MRIwrite(DEF_logpolyaffine_seg.detach().cpu().numpy(), aff, args['output_dir'] + '/' + str(spac) + '_log_polyaffine_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_logpolyaffine_seg.detach().cpu().numpy(), aff, args['output_dir'] + '/log_polyaffine_deformation_gridSeg_z_' + str(spac) + '.nii.gz')

            pct_neg_J, membrane_energy = get_metrics(ii2_logpolyaffine, ii2aff_ransac, jj2_logpolyaffine, jj2aff_ransac, kk2_logpolyaffine, kk2aff_ransac, M, device)
            print('Membrane energy for spac ' + str(spac) + ' is ' + str(membrane_energy.item()) + ' and pct of negative J is ' + str(pct_neg_J.item()))

            results_df.loc[len(results_df)] = {
            'Algorithm': 'Log_polyaffine',
            'Sigma or spacing or Order': str(spac),
            'Membrane Energy': membrane_energy.item(),
            'Percentage of Negative Jacobian': pct_neg_J.item()}

            # apply inverse transform
            FIELD = -SVF / (2.0 ** float(steps))
            for s in range(steps):
                FIELD += fast_3D_interp_torch(FIELD, (I+ic) + FIELD[:,:,:,0], (J+jc) + FIELD[:,:,:,1], (K+kc) + FIELD[:,:,:,2], 'linear', device)
            ri = np.arange(MNI.shape[0]).astype('float');
            rj = np.arange(MNI.shape[1]).astype('float');
            rk = np.arange(MNI.shape[2]).astype('float');
            mni_i, mni_j, mni_k = np.meshgrid(ri, rj, rk, sparse=False, indexing='ij')
            mni_i = torch.tensor(mni_i, device=device, dtype=torch.float)
            mni_j = torch.tensor(mni_j, device=device, dtype=torch.float)
            mni_k = torch.tensor(mni_k, device=device, dtype=torch.float)
            T = torch.inverse(torch.stack([best_fit_x, best_fit_y, best_fit_z, torch.tensor([0, 0, 0, 1], device=device, dtype=torch.float)]))
            field_i = (pred_mni.shape[0]-1)/2 + 100 * (T[0, 0] * mni_i + T[0, 1] * mni_j + T[0, 2] * mni_k + T[0, 3])
            field_j = (pred_mni.shape[1]-1)/2 + 100 * (T[1, 0] * mni_i + T[1, 1] * mni_j + T[1, 2] * mni_k + T[1, 3])
            field_k = (pred_mni.shape[2]-1)/2 + 100 * (T[2, 0] * mni_i + T[2, 1] * mni_j + T[2, 2] * mni_k + T[2, 3])
            FIELDinter = fast_3D_interp_torch(FIELD, field_i, field_j, field_k, 'linear', device)
            mni_i += FIELDinter[:, :, :, 0]
            mni_j += FIELDinter[:, :, :, 1]
            mni_k += FIELDinter[:, :, :, 2]
            source_i = (pred_mni.shape[0]-1)/2 + 100 * (T[0, 0] * mni_i + T[0, 1] * mni_j + T[0, 2] * mni_k + T[0, 3])
            source_j = (pred_mni.shape[1]-1)/2 + 100 * (T[1, 0] * mni_i + T[1, 1] * mni_j + T[1, 2] * mni_k + T[1, 3])
            source_k = (pred_mni.shape[2]-1)/2 + 100 * (T[2, 0] * mni_i + T[2, 1] * mni_j + T[2, 2] * mni_k + T[2, 3])
            
            vals_inverse = fast_3D_interp_torch(im_orig, source_i, source_j, source_k, 'linear', device)
            # MRIwrite(vals_inverse.detach().cpu().numpy(), aff2, args['output_dir']  + '/' + str(spac) + '_log_polyaffine_inverse.nii.gz')
            MRIwrite(vals_inverse.detach().cpu().numpy(), aff2, args['output_dir'] + '/log_polyaffine_inverse_' + str(spac) + '.nii.gz')
            
            vals_inverse_seg = fast_3D_interp_torch(imseg_orig, source_i, source_j, source_k, 'nearest', device)
            # MRIwrite(vals_inverse_seg.detach().cpu().numpy(), aff2, args['output_dir']  + '/' + str(spac) + '_log_polyaffine_inverse_seg.nii.gz')
            MRIwrite(vals_inverse_seg.detach().cpu().numpy(), aff2, args['output_dir'] + '/log_polyaffine_inverse_seg_' + str(spac) + '.nii.gz')

            poly_affine_end = time()
            print("The poly affine took {} seconds.".format(poly_affine_end-poly_affine_start))


        print('Bspline basis functions')

        all_spacing = [2.5, 5, 10]
        for spacing in all_spacing:
            Bspline_start = time()

            if False: # too slow...
                idef = ii - ii2aff_ransac
                jdef = jj - jj2aff_ransac
                kdef = kk - kk2aff_ransac
                disp = torch.sqrt(idef * idef + jdef * jdef + kdef * kdef)
                max_disp = torch.tensor(10.0, device=device)
                toofar = disp > max_disp
                idef[toofar] = (idef[toofar] / disp[toofar]) * max_disp
                jdef[toofar] = (jdef[toofar] / disp[toofar]) * max_disp
                kdef[toofar] = (kdef[toofar] / disp[toofar]) * max_disp

                PHI, Gx, Gy, Gz = get_basis_functions_bspline(pred_mni.shape[:-1], spacing=spacing, device=device, mask=M, get_gradients=True)

                term1 = torch.zeros([len(PHI), len(PHI)], device=device, dtype=torch.float)
                term2 = torch.zeros([len(PHI), len(PHI)], device=device, dtype=torch.float)
                term_i = torch.zeros(len(PHI), device=device, dtype=torch.float)
                term_j = torch.zeros(len(PHI), device=device, dtype=torch.float)
                term_k = torch.zeros(len(PHI), device=device, dtype=torch.float)
                for a in range(len(PHI)):
                    for b in range(len(PHI)):
                        term1[a, b] = torch.sum(PHI[a] * PHI[b])
                        term2[a, b] = torch.sum(Gx[a] * Gx[b]) + torch.sum(Gy[a] * Gy[b]) + torch.sum(Gz[a] * Gz[b])
                    term_i[a] =  torch.sum(PHI[a] * idef)
                    term_j[a] = torch.sum(PHI[a] * jdef)
                    term_k[a] = torch.sum(PHI[a] * kdef)
                PHI = torch.stack(PHI, dim=1)

                alpha = 0.001
                aux = term1 + alpha * term2
                A = torch.linalg.inv(aux.to_dense())
                c_i = A @ term_i
                c_j = A @ term_j
                c_k = A @ term_k

                ii2_bspline = ii2aff_ransac + PHI @ c_i
                jj2_bspline = jj2aff_ransac + PHI @ c_j
                kk2_bspline = kk2aff_ransac + PHI @ c_k

            if False: # tons of boundary artifacts
                idef = ii - ii2aff_ransac
                jdef = jj - jj2aff_ransac
                kdef = kk - kk2aff_ransac
                disp = torch.sqrt(idef * idef + jdef * jdef + kdef * kdef)
                max_disp = torch.tensor(10.0, device=device)
                toofar = disp > max_disp
                idef[toofar] = (idef[toofar] / disp[toofar]) * max_disp
                jdef[toofar] = (jdef[toofar] / disp[toofar]) * max_disp
                kdef[toofar] = (kdef[toofar] / disp[toofar]) * max_disp

                import sys
                sys.path.append('/homes/2/iglesias/python/code/ERC_bayesian_segmentation-release/ERC_bayesian_segmentation/ext/interpol')
                import ext.interpol
                small_shape = tuple(np.ceil(np.array(pred_mni.shape[:-1]) / spacing).astype(int))

                aux = torch.zeros(pred_mni.shape[:-1], device=device, dtype=torch.float)
                aux2 = ext.interpol.resize(M.float(), shape=small_shape, interpolation=3, prefilter=True)
                aux3 = ext.interpol.resize(aux2, shape=pred_mni.shape[:-1], interpolation=3, prefilter=False)
                aux[M] = idef
                aux4 = ext.interpol.resize(aux, shape=small_shape, interpolation=3, prefilter=True) / aux2
                aux5 = ext.interpol.resize(aux4, shape=pred_mni.shape[:-1], interpolation=3, prefilter=False) / aux3
                ii2_bspline = ii2aff_ransac + aux5[M]
                aux[M] = jdef
                aux4 = ext.interpol.resize(aux, shape=small_shape, interpolation=3, prefilter=True) / aux2
                aux5 = ext.interpol.resize(aux4, shape=pred_mni.shape[:-1], interpolation=3, prefilter=False) / aux3
                jj2_bspline = jj2aff_ransac + aux5[M]
                aux[M] = kdef
                aux4 = ext.interpol.resize(aux, shape=small_shape, interpolation=3, prefilter=True) / aux2
                aux5 = ext.interpol.resize(aux4, shape=pred_mni.shape[:-1], interpolation=3, prefilter=False) / aux3
                kk2_bspline = kk2aff_ransac + aux5[M]

            # recompute affine transform everywhere
            ri = np.arange(pred_mni.shape[0]).astype('float'); ri -= np.mean(ri); ri /= 100
            rj = np.arange(pred_mni.shape[1]).astype('float'); rj -= np.mean(rj); rj /= 100
            rk = np.arange(pred_mni.shape[2]).astype('float'); rk -= np.mean(rk); rk /= 100
            ifull, jfull, kfull = np.meshgrid(ri, rj, rk, sparse=False, indexing='ij')
            ifull = torch.tensor(ifull, device=device, dtype=torch.float)
            jfull = torch.tensor(jfull, device=device, dtype=torch.float)
            kfull = torch.tensor(kfull, device=device, dtype=torch.float)
            iifull = best_fit_x[0] * ifull + best_fit_x[1] * jfull + best_fit_x[2] * kfull + best_fit_x[3]
            jjfull = best_fit_y[0] * ifull + best_fit_y[1] * jfull + best_fit_y[2] * kfull + best_fit_y[3]
            kkfull = best_fit_z[0] * ifull + best_fit_z[1] * jfull + best_fit_z[2] * kfull + best_fit_z[3]
            xxfull = pred_mni[:, :, :, 0]
            yyfull = pred_mni[:, :, :, 1]
            zzfull = pred_mni[:, :, :, 2]
            iimnifull = A[0, 0] * xxfull + A[0, 1] * yyfull + A[0, 2] * zzfull + A[0, 3]
            jjmnifull = A[1, 0] * xxfull + A[1, 1] * yyfull + A[1, 2] * zzfull + A[1, 3]
            kkmnifull = A[2, 0] * xxfull + A[2, 1] * yyfull + A[2, 2] * zzfull + A[2, 3]

            # clip  outliers
            idef = iimnifull - iifull
            jdef = jjmnifull - jjfull
            kdef = kkmnifull - kkfull
            disp = torch.sqrt(idef * idef + jdef * jdef + kdef * kdef)
            max_disp = torch.tensor(10.0, device=device)
            toofar = disp > max_disp
            idef[toofar] = (idef[toofar] / disp[toofar]) * max_disp
            jdef[toofar] = (jdef[toofar] / disp[toofar]) * max_disp
            kdef[toofar] = (kdef[toofar] / disp[toofar]) * max_disp
            iifixed = iifull + idef
            jjfixed = jjfull + jdef
            kkfixed = kkfull + kdef

            # fit bsplines
            import sys
            sys.path.append('/homes/2/iglesias/python/code/ERC_bayesian_segmentation-release/ERC_bayesian_segmentation/ext/interpol')
            import ext.interpol
            small_shape = tuple(np.ceil(np.array(pred_mni.shape[:-1]) / spacing).astype(int))
            aux = ext.interpol.resize(iifixed, shape=small_shape, interpolation=3, prefilter=True)
            aux2 = ext.interpol.resize(aux, shape=pred_mni.shape[:-1], interpolation=3, prefilter=False)
            ii2_bspline = aux2[M]
            aux = ext.interpol.resize(jjfixed, shape=small_shape, interpolation=3, prefilter=True)
            aux2 = ext.interpol.resize(aux, shape=pred_mni.shape[:-1], interpolation=3, prefilter=False)
            jj2_bspline = aux2[M]
            aux = ext.interpol.resize(kkfixed, shape=small_shape, interpolation=3, prefilter=True)
            aux2 = ext.interpol.resize(aux, shape=pred_mni.shape[:-1], interpolation=3, prefilter=False)
            kk2_bspline = aux2[M]

            vals_bspline = fast_3D_interp_torch(MNI, ii2_bspline, jj2_bspline, kk2_bspline, 'linear', device)
            DEF_bspline = torch.zeros_like(pred_mni[..., 0])
            DEF_bspline[M] = vals_bspline
            # MRIwrite(DEF_bspline.detach().cpu().numpy(), aff, args['output_dir']  + '/' + str(spacing) + '_bspline_deformation.nii.gz')
            MRIwrite(DEF_bspline.detach().cpu().numpy(), aff, args['output_dir']  + '/bspline_deformation_' + str(spacing) + '.nii.gz')
            
            vals_bspline_seg = fast_3D_interp_torch(MNISeg, ii2_bspline, jj2_bspline, kk2_bspline, 'nearest', device)
            DEF_bspline_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_bspline_seg[M] = vals_bspline_seg
            # MRIwrite(DEF_bspline_seg.detach().cpu().numpy(), aff, args['output_dir']  + '/' + str(spacing) + '_bspline_deformation_seg.nii.gz')
            MRIwrite(DEF_bspline_seg.detach().cpu().numpy(), aff, args['output_dir']  + '/bspline_deformation_seg_' + str(spacing) + '.nii.gz')
            
            vals_bspline_seg = fast_3D_interp_torch(gridSeg_x, ii2_bspline, jj2_bspline, kk2_bspline, 'linear', device)
            DEF_bspline_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_bspline_seg[M] = vals_bspline_seg
            # MRIwrite(DEF_bspline_seg.detach().cpu().numpy(), aff, args['output_dir']  + '/' + str(spacing) + '_bspline_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_bspline_seg.detach().cpu().numpy(), aff, args['output_dir']  + '/bspline_deformation_gridSeg_x_' + str(spacing) + '.nii.gz')

            vals_bspline_seg = fast_3D_interp_torch(gridSeg_y, ii2_bspline, jj2_bspline, kk2_bspline, 'linear', device)
            DEF_bspline_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_bspline_seg[M] = vals_bspline_seg
            # MRIwrite(DEF_bspline_seg.detach().cpu().numpy(), aff, args['output_dir']  + '/' + str(spacing) + '_bspline_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_bspline_seg.detach().cpu().numpy(), aff, args['output_dir']  + '/bspline_deformation_gridSeg_y_' + str(spacing) + '.nii.gz')

            vals_bspline_seg = fast_3D_interp_torch(gridSeg_z, ii2_bspline, jj2_bspline, kk2_bspline, 'linear', device)
            DEF_bspline_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_bspline_seg[M] = vals_bspline_seg
            # MRIwrite(DEF_bspline_seg.detach().cpu().numpy(), aff, args['output_dir']  + '/' + str(spacing) + '_bspline_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_bspline_seg.detach().cpu().numpy(), aff, args['output_dir']  + '/bspline_deformation_gridSeg_z_' + str(spacing) + '.nii.gz')

            pct_neg_J, membrane_energy = get_metrics(ii2_bspline, ii2aff_ransac, jj2_bspline, jj2aff_ransac, kk2_bspline, kk2aff_ransac, M, device)
            print('Membrane energy for spacing ' + str(spacing) + ' is ' + str(membrane_energy.item()) + ' and pct of negative J is ' + str(pct_neg_J.item()))

            results_df.loc[len(results_df)] = {
            'Algorithm': 'Bspline',
            'Sigma or spacing or Order': str(spacing),
            'Membrane Energy': membrane_energy.item(),
            'Percentage of Negative Jacobian': pct_neg_J.item()}

            Bspline_end = time()
            print("The Bspline took {} seconds.".format(Bspline_end-Bspline_start))


        print('DCT basis functions')
        
        all_order = [3, 6, 9, 12]

        for order in all_order:

            DCT_start = time()
            
            PHI, Gx, Gy, Gz = get_basis_functions_dct(pred_mni.shape[:-1], order=order, device=device, mask=M, get_gradients=True)
            PHI = torch.stack(PHI, dim=1)
            Gx = torch.stack(Gx, dim=1)
            Gy = torch.stack(Gy, dim=1)
            Gz = torch.stack(Gz, dim=1)


            idef = ii - ii2aff_ransac
            jdef = jj - jj2aff_ransac
            kdef = kk - kk2aff_ransac
            disp = torch.sqrt(idef*idef + jdef * jdef + kdef * kdef)
            max_disp = torch.tensor(10.0, device=device)
            toofar = disp>max_disp
            idef[toofar] = (idef[toofar] / disp[toofar]) * max_disp
            jdef[toofar] = (jdef[toofar] / disp[toofar]) * max_disp
            kdef[toofar] = (kdef[toofar] / disp[toofar]) * max_disp

            term1 = PHI.t() @ PHI
            term2 = Gx.t() @ Gx + Gy.t() @ Gy + Gz.t() @ Gz
            term_i = PHI.t() @ idef
            term_j = PHI.t() @ jdef
            term_k = PHI.t() @ kdef

            alpha = 0.001
            A = torch.linalg.inv(term1 + alpha * term2)
            c_i = A @ term_i
            c_j = A @ term_j
            c_k = A @ term_k

            ii2_dct = ii2aff_ransac + PHI @ c_i
            jj2_dct = jj2aff_ransac + PHI @ c_j
            kk2_dct = kk2aff_ransac + PHI @ c_k

            vals_dct = fast_3D_interp_torch(MNI, ii2_dct, jj2_dct, kk2_dct, 'linear', device)
            DEF_dct = torch.zeros_like(pred_mni[..., 0])
            DEF_dct[M] = vals_dct
            # MRIwrite(DEF_dct.detach().cpu().numpy(), aff, args['output_dir']+ '/' + str(order) + '_dct_deformation.nii.gz')
            MRIwrite(DEF_dct.detach().cpu().numpy(), aff, args['output_dir']+ '/dct_deformation_' + str(order) + '.nii.gz')
            
            vals_dct_seg = fast_3D_interp_torch(MNISeg, ii2_dct, jj2_dct, kk2_dct, 'nearest', device)
            DEF_dct_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_dct_seg[M] = vals_dct_seg
            # MRIwrite(DEF_dct_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/' + str(order) + '_dct_deformation_seg.nii.gz')
            MRIwrite(DEF_dct_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/dct_deformation_seg_' + str(order) + '.nii.gz')
            
            vals_dct_seg = fast_3D_interp_torch(gridSeg_x, ii2_dct, jj2_dct, kk2_dct, 'linear', device)
            DEF_dct_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_dct_seg[M] = vals_dct_seg
            # MRIwrite(DEF_dct_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/' + str(order) + '_dct_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_dct_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/dct_deformation_gridSeg_x_' + str(order) + '.nii.gz')

            vals_dct_seg = fast_3D_interp_torch(gridSeg_y, ii2_dct, jj2_dct, kk2_dct, 'linear', device)
            DEF_dct_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_dct_seg[M] = vals_dct_seg
            # MRIwrite(DEF_dct_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/' + str(order) + '_dct_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_dct_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/dct_deformation_gridSeg_y_' + str(order) + '.nii.gz')

            vals_dct_seg = fast_3D_interp_torch(gridSeg_z, ii2_dct, jj2_dct, kk2_dct, 'linear', device)
            DEF_dct_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_dct_seg[M] = vals_dct_seg
            # MRIwrite(DEF_dct_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/' + str(order) + '_dct_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_dct_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/dct_deformation_gridSeg_z_' + str(order) + '.nii.gz')

            pct_neg_J, membrane_energy = get_metrics(ii2_dct, ii2aff_ransac, jj2_dct, jj2aff_ransac, kk2_dct, kk2aff_ransac, M, device)
            print('Membrane energy for order ' + str(order) + ' is ' + str(membrane_energy.item()) + ' and pct of negative J is ' + str(pct_neg_J.item()))



            results_df.loc[len(results_df)] = {
            'Algorithm': 'DCT basis',
            'Sigma or spacing or Order': str(order),
            'Membrane Energy': membrane_energy.item(),
            'Percentage of Negative Jacobian': pct_neg_J.item()}

            # DEF_dct[M] = (idef - PHI @ c_i) ** 2 + (jdef - PHI @ c_j) ** 2 + (kdef - PHI @ c_k) ** 2
            # MRIwrite(DEF_dct.detach().cpu().numpy(), aff, args['output_dir'] + '/dct_error.nii.gz')

            DCT_end = time()
            print("The DCT took {} seconds.".format(DCT_end-DCT_start))

            # print('Polynomial basis functions')
        print('q basis functions')
        all_order = [3, 5, 7]

        for order in all_order:    

            Polynomial_start = time()

            PHI, Gx, Gy, Gz = get_basis_functions(pred_mni.shape[:-1], order=order, device=device, mask=M, get_gradients=True)
            PHI = torch.stack(PHI, dim=1)
            Gx = torch.stack(Gx, dim=1)
            Gy = torch.stack(Gy, dim=1)
            Gz = torch.stack(Gz, dim=1)

            idef = ii - ii2aff_ransac
            jdef = jj - jj2aff_ransac
            kdef = kk - kk2aff_ransac
            disp = torch.sqrt(idef*idef + jdef * jdef + kdef * kdef)
            max_disp = torch.tensor(20.0, device=device)
            toofar = disp>max_disp
            idef[toofar] = (idef[toofar] / disp[toofar]) * max_disp
            jdef[toofar] = (jdef[toofar] / disp[toofar]) * max_disp
            kdef[toofar] = (kdef[toofar] / disp[toofar]) * max_disp

            term1 = PHI.t() @ PHI
            term2 = Gx.t() @ Gx + Gy.t() @ Gy + Gz.t() @ Gz
            term_i = PHI.t() @ idef
            term_j = PHI.t() @ jdef
            term_k = PHI.t() @ kdef

            alpha = 0.001
            A = torch.linalg.inv(term1 + alpha * term2)
            c_i = A @ term_i
            c_j = A @ term_j
            c_k = A @ term_k

            ii2_polynomial = ii2aff_ransac + PHI @ c_i
            jj2_polynomial = jj2aff_ransac + PHI @ c_j
            kk2_polynomial = kk2aff_ransac + PHI @ c_k

            vals_polynomial = fast_3D_interp_torch(MNI, ii2_polynomial, jj2_polynomial, kk2_polynomial, 'linear', device)
            DEF_polynomial = torch.zeros_like(pred_mni[..., 0])
            DEF_polynomial[M] = vals_polynomial
            # MRIwrite(DEF_polynomial.detach().cpu().numpy(), aff, args['output_dir']+ '/' + str(order) + '_polymonial_deformation.nii.gz')
            MRIwrite(DEF_polynomial.detach().cpu().numpy(), aff, args['output_dir']+ '/polymonial_deformation_' + str(order) + '.nii.gz')
            
            vals_polynomial_seg = fast_3D_interp_torch(MNISeg, ii2_polynomial, jj2_polynomial, kk2_polynomial, 'nearest', device)
            DEF_polynomial_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_polynomial_seg[M] = vals_polynomial_seg
            # MRIwrite(DEF_polynomial_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/' + str(order) + '_polymonial_deformation_seg.nii.gz')
            MRIwrite(DEF_polynomial_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/polymonial_deformation_seg_' + str(order) + '.nii.gz')
            
            vals_polynomial_seg = fast_3D_interp_torch(gridSeg_x, ii2_polynomial, jj2_polynomial, kk2_polynomial, 'linear', device)
            DEF_polynomial_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_polynomial_seg[M] = vals_polynomial_seg
            # MRIwrite(DEF_polynomial_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/' + str(order) + '_polymonial_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_polynomial_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/polymonial_deformation_gridSeg_x_' + str(order) + '.nii.gz')

            vals_polynomial_seg = fast_3D_interp_torch(gridSeg_y, ii2_polynomial, jj2_polynomial, kk2_polynomial, 'linear', device)
            DEF_polynomial_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_polynomial_seg[M] = vals_polynomial_seg
            # MRIwrite(DEF_polynomial_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/' + str(order) + '_polymonial_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_polynomial_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/polymonial_deformation_gridSeg_y_' + str(order) + '.nii.gz')

            vals_polynomial_seg = fast_3D_interp_torch(gridSeg_z, ii2_polynomial, jj2_polynomial, kk2_polynomial, 'linear', device)
            DEF_polynomial_seg = torch.zeros_like(pred_mni[..., 0])
            DEF_polynomial_seg[M] = vals_polynomial_seg
            # MRIwrite(DEF_polynomial_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/' + str(order) + '_polymonial_deformation_gridSeg.nii.gz')
            MRIwrite(DEF_polynomial_seg.detach().cpu().numpy(), aff, args['output_dir']+ '/polymonial_deformation_gridSeg_z_' + str(order) + '.nii.gz')

            pct_neg_J, membrane_energy = get_metrics(ii2_polynomial, ii2aff_ransac, jj2_polynomial, jj2aff_ransac, kk2_polynomial, kk2aff_ransac, M, device)
            print('Membrane energy for order ' + str(order) + ' is ' + str(membrane_energy.item()) + ' and pct of negative J is ' + str(pct_neg_J.item()))

            results_df.loc[len(results_df)] = {
            'Algorithm': 'Polymonial basis',
            'Sigma or spacing or Order': str(order),
            'Membrane Energy': membrane_energy.item(),
            'Percentage of Negative Jacobian': pct_neg_J.item()}

            Polynomial_end = time()
            print("The Polynomial took {} seconds.".format(Polynomial_end-Polynomial_start))

        if not results_df.empty:
            # Determine if we need to write headers (i.e., if file doesn't exist or is empty)
            header = not pd.io.common.file_exists(csv_file_path) or pd.read_csv(csv_file_path).empty
            results_df.to_csv(csv_file_path, mode='a', header=header, index=False)
        print('Results saved to ' + csv_file_path)
        print('freeview ' + args['input'] + ' ' + args['output_dir'] + '/*.nii.gz')
        print('All done')

        end_time = time()
        print("The whole process took {} seconds.".format(end_time-start_time))


# get_metrics
        
def get_metrics(ii2demon, ii2aff_ransac, jj2demon, jj2aff_ransac, kk2demon, kk2aff_ransac, M, device):
    print('Computing membrane energy')
    Meroded = torch.tensor(binary_erosion(M.detach().cpu().numpy(), iterations=3), device=device, dtype=torch.bool)
    #Meroded = erode mask which we need not to include voxels outside the mask when we compute gradients
    aux = torch.zeros(M.shape).to(device)
    G = torch.zeros([*M.shape, 3, 3]).to(device)
    aux[M] = ii2demon - ii2aff_ransac
    G[1:-1, :, :, 0, 0] = aux[2:,:,:] - aux[:-2,:,:]
    G[:, 1:-1, :, 0, 1] = aux[:, 2:,:] - aux[:, :-2,:]
    G[:, :, 1:-1, 0, 2] = aux[:,:,2:] - aux[:, :, :-2]
    aux[M] = jj2demon - jj2aff_ransac
    G[1:-1, :, :, 1, 0] = aux[2:,:,:] - aux[:-2,:,:]
    G[:, 1:-1, :, 1, 1] = aux[:, 2:,:] - aux[:, :-2,:]
    G[:, :, 1:-1, 1, 2] = aux[:,:,2:] - aux[:, :, :-2]
    aux[M] = kk2demon - kk2aff_ransac
    G[1:-1, :, :, 2, 0] = aux[2:,:,:] - aux[:-2,:,:]
    G[:, 1:-1, :, 2, 1] = aux[:, 2:,:] - aux[:, :-2,:]
    G[:, :, 1:-1, 2, 2] = aux[:,:,2:] - aux[:, :, :-2]
    G *= 0.5

    ME = torch.sum(torch.sum(G*G,axis=-1), axis=-1)
    membrane_energy = torch.mean(ME[Meroded>0])

    print('Computing Jacobian determinant')
    aux[M] = ii2demon
    G[1:-1, :, :, 0, 0] = aux[2:,:,:] - aux[:-2,:,:]
    G[:, 1:-1, :, 0, 1] = aux[:, 2:,:] - aux[:, :-2,:]
    G[:, :, 1:-1, 0, 2] = aux[:,:,2:] - aux[:, :, :-2]
    aux[M] = jj2demon
    G[1:-1, :, :, 1, 0] = aux[2:,:,:] - aux[:-2,:,:]
    G[:, 1:-1, :, 1, 1] = aux[:, 2:,:] - aux[:, :-2,:]
    G[:, :, 1:-1, 1, 2] = aux[:,:,2:] - aux[:, :, :-2]
    aux[M] = kk2demon
    G[1:-1, :, :, 2, 0] = aux[2:,:,:] - aux[:-2,:,:]
    G[:, 1:-1, :, 2, 1] = aux[:, 2:,:] - aux[:, :-2,:]
    G[:, :, 1:-1, 2, 2] = aux[:,:,2:] - aux[:, :, :-2]
    G *= 0.5

    def jacobian_det(G):
        a = G[:,:,:,0,0]
        b = G[:,:,:,0,1]
        c = G[:,:,:,0,2]
        d = G[:,:,:,1,0]
        e = G[:,:,:,1,1]
        f = G[:,:,:,1,2]
        g = G[:,:,:,2,0]
        h = G[:,:,:,2,1]
        i = G[:,:,:,2,2]
        J = a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)
        return J

    Jdet = jacobian_det(G)
    pct_neg_J = 100 * torch.sum(Jdet[Meroded>0]<=0) / torch.sum(Meroded>0)
    
    return pct_neg_J, membrane_energy


# make polynomial basis functions
def get_basis_functions(shape, order=3, device='cpu', dtype=torch.float32, mask=None, get_gradients=False):

    G = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]))
    Gnorm = []
    for i in range(3):
        aux = G[i].type(dtype).to(device)
        Gnorm.append(2 * ((aux / (shape[i] - 1)) - 0.5))

    B = []
    Gx = []; Gy = []; Gz = []

    for x in range(order + 1):
        for y in range(order + 1):
            for z in range(order + 1):
                if ((x + y + z) <= order): # and ((x + y + z) > 0):
                    b = torch.ones(shape, device=device, dtype=dtype)
                    for i in range(x):
                        b = b * Gnorm[0]
                    for i in range(y):
                        b = b * Gnorm[1]
                    for i in range(z):
                        b = b * Gnorm[2]

                    if get_gradients:
                        gx = torch.zeros_like(b)
                        gx[0, :, :] = b[1, :, :] - b[0, :, :]
                        gx[1:-1, :, :] = 0.5 * (b[2:, :, :] - b[:-2, :, :])
                        gx[-1, :, :] = b[-1, :, :] - b[-2, :, :]

                        gy = torch.zeros_like(b)
                        gy[:, 0, :] = b[:, 1, :] - b[:, 0, :]
                        gy[:, 1:-1, :] = 0.5 * (b[:, 2:, :] - b[:, :-2, :])
                        gy[:, -1, :] = b[:, -1, :] - b[:, -2, :]

                        gz = torch.zeros_like(b)
                        gz[:, :, 0] = b[:, :, 1] - b[:, :, 0]
                        gz[:, :, 1:-1] = 0.5 * (b[:, :, 2:] - b[:, :, :-2])
                        gz[:, :, -1] = b[:, :, -1] - b[:, :, -2]

                    if mask is None:
                        B.append(b)
                        if get_gradients:
                            Gx.append(gx); Gy.append(gy); Gz.append(gz)
                    else:
                        B.append(b[mask])
                        if get_gradients:
                            Gx.append(gx[mask]); Gy.append(gy[mask]); Gz.append(gz[mask])

    if get_gradients:
        return B, Gx, Gy, Gz
    else:
        return B

# make dct basis functions
def get_basis_functions_dct(shape, order=3, device='cpu', dtype=torch.float32, mask=None, get_gradients=False):

    one_d_basis_x = []
    one_d_basis_y = []
    one_d_basis_z = []
    for i in range(order):
        one_d_basis_x.append(torch.tensor(np.cos((2.0 * np.arange(shape[0]) + 1) * np.pi * (i + 1) / (2.0 * shape[0])), device=device, dtype=dtype))
        one_d_basis_y.append(torch.tensor(np.cos((2.0 * np.arange(shape[1]) + 1) * np.pi * (i + 1) / (2.0 * shape[1])), device=device, dtype=dtype))
        one_d_basis_z.append(torch.tensor(np.cos((2.0 * np.arange(shape[2]) + 1) * np.pi * (i + 1) / (2.0 * shape[2])), device=device, dtype=dtype))

    B = []
    Gx = []; Gy = []; Gz = []

    for x in range(order + 1):
        for y in range(order + 1):
            for z in range(order + 1):
                if ((x + y + z) <= order):  # and ((x + y + z) > 0):
                    b = torch.ones(shape, device=device, dtype=dtype)
                    for i in range(x):
                        b *= one_d_basis_x[i][:, None, None]
                    for i in range(y):
                        b *= one_d_basis_y[i][None, :, None]
                    for i in range(z):
                        b *= one_d_basis_z[i][None, None, :]

                    if get_gradients:
                        gx = torch.zeros_like(b)
                        gx[0, :, :] = b[1, :, :] - b[0, :, :]
                        gx[1:-1, :, :] = 0.5 * (b[2:, :, :] - b[:-2, :, :])
                        gx[-1, :, :] = b[-1, :, :] - b[-2, :, :]

                        gy = torch.zeros_like(b)
                        gy[:, 0, :] = b[:, 1, :] - b[:, 0, :]
                        gy[:, 1:-1, :] = 0.5 * (b[:, 2:, :] - b[:, :-2, :])
                        gy[:, -1, :] = b[:, -1, :] - b[:, -2, :]

                        gz = torch.zeros_like(b)
                        gz[:, :, 0] = b[:, :, 1] - b[:, :, 0]
                        gz[:, :, 1:-1] = 0.5 * (b[:, :, 2:] - b[:, :, :-2])
                        gz[:, :, -1] = b[:, :, -1] - b[:, :, -2]

                    if mask is None:
                        B.append(b)
                        if get_gradients:
                            Gx.append(gx); Gy.append(gy); Gz.append(gz)
                    else:
                        B.append(b[mask])
                        if get_gradients:
                            Gx.append(gx[mask]); Gy.append(gy[mask]); Gz.append(gz[mask])



    if get_gradients:
        return B, Gx, Gy, Gz
    else:
        return B



# make dct basis functions
def get_basis_functions_bspline(shape, spacing=10, device='cpu', dtype=torch.float32, mask=None, get_gradients=False):

    # ugly as hell and works but super slow...
    import sys
    sys.path.append('/homes/2/iglesias/python/code/ERC_bayesian_segmentation-release/ERC_bayesian_segmentation/ext/interpol')
    import ext.interpol

    B = []
    Gx = []; Gy = []; Gz = []
    small_shape = np.ceil(np.array(shape) / spacing).astype(int)
    X = torch.zeros(tuple(small_shape), device=device, dtype=dtype)
    gx = torch.zeros(np.array(shape).tolist(), device=device, dtype=dtype)
    gy = torch.zeros(np.array(shape).tolist(), device=device, dtype=dtype)
    gz = torch.zeros(np.array(shape).tolist(), device=device, dtype=dtype)
    if mask is None:
        mask = X < 1

    for i in range(small_shape[0]):
        for j in range(small_shape[1]):
            for k in range(small_shape[2]):
                X[:] = 0
                X[i, j, k] = 1
                b = ext.interpol.resize(X, shape=np.array(shape).tolist(), anchor='e', interpolation=3, prefilter=False)
                nz = torch.nonzero(b)
                mini = torch.min(nz, axis=0)[0]
                maxi = torch.max(nz, axis=0)[0]
                if get_gradients:
                    gx[:] = 0
                    gx[0, mini[1]:maxi[1]+1, mini[2]:maxi[2]+1] = b[1,  mini[1]:maxi[1]+1, mini[2]:maxi[2]+1] - b[0,  mini[1]:maxi[1]+1, mini[2]:maxi[2]+1]
                    gx[1:-1, mini[1]:maxi[1]+1, mini[2]:maxi[2]+1] = 0.5 * (b[2:, mini[1]:maxi[1]+1, mini[2]:maxi[2]+1] - b[:-2, mini[1]:maxi[1]+1, mini[2]:maxi[2]+1])
                    gx[-1, mini[1]:maxi[1]+1, mini[2]:maxi[2]+1] = b[-1, mini[1]:maxi[1]+1, mini[2]:maxi[2]+1] - b[-2, mini[1]:maxi[1]+1, mini[2]:maxi[2]+1]

                    gy[:] = 0
                    gy[mini[0]:maxi[0]+1, 0, mini[2]:maxi[2]+1] = b[mini[0]:maxi[0]+1, 1, mini[2]:maxi[2]+1] - b[mini[0]:maxi[0]+1, 0, mini[2]:maxi[2]+1]
                    gy[mini[0]:maxi[0]+1, 1:-1, mini[2]:maxi[2]+1] = 0.5 * (b[mini[0]:maxi[0]+1, 2:, mini[2]:maxi[2]+1] - b[mini[0]:maxi[0]+1, :-2, mini[2]:maxi[2]+1])
                    gy[mini[0]:maxi[0]+1, -1, mini[2]:maxi[2]+1] = b[mini[0]:maxi[0]+1, -1, mini[2]:maxi[2]+1] - b[mini[0]:maxi[0]+1, -2, mini[2]:maxi[2]+1]

                    gz[:] = 0
                    gz[mini[0]:maxi[0]+1, mini[1]:maxi[1]+1, 0] = b[mini[0]:maxi[0]+1, mini[1]:maxi[1]+1, 1] - b[mini[0]:maxi[0]+1, mini[1]:maxi[1]+1, 0]
                    gz[mini[0]:maxi[0]+1, mini[1]:maxi[1]+1, 1:-1] = 0.5 * (b[mini[0]:maxi[0]+1, mini[1]:maxi[1]+1, 2:] - b[mini[0]:maxi[0]+1, mini[1]:maxi[1]+1, :-2])
                    gz[mini[0]:maxi[0]+1, mini[1]:maxi[1]+1, -1] = b[mini[0]:maxi[0]+1, mini[1]:maxi[1]+1, -1] - b[mini[0]:maxi[0]+1, mini[1]:maxi[1]+1, -2]

                bmasked = b[mask]
                if torch.any(bmasked > 0):
                    B.append(bmasked.to_sparse())
                    if get_gradients:
                        Gx.append(gx[mask].to_sparse())
                        Gy.append(gy[mask].to_sparse())
                        Gz.append(gz[mask].to_sparse())





    if get_gradients:
        return B, Gx, Gy, Gz
    else:
        return B


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

# execute script
if __name__ == '__main__':
    main()