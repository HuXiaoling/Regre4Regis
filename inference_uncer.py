# train loop
# use validation to save checkpoint ---> take 2-3 vals only, OR take patch of all the vals
# lr schedule
import torch
import numpy as np
import nibabel as nib

import argparse, json
import os, glob, sys
from time import time

from dataloader_aug import regress
from unet.unet_3d import UNet_3d
from utilities import DC_and_CE_loss, SoftDiceLoss, softmax_helper
import torch
from torchvision.utils import save_image
import torch.nn as nn
torch.cuda.empty_cache()

def parse_func(args):
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    activity = params['common']['activity']
    mydict = {}
    mydict['dataset'] = params['common']['dataset']
    mydict['num_classes'] = int(params['common']['num_classes'])
    mydict['files'] = params['common']['img_file']
    mydict["checkpoint_restore"] = params['common']['checkpoint_restore']

    mydict['validation_datalist'] = params['validation']['validation_datalist']
    mydict['output_folder'] = params['validation']['output_folder']
    mydict['batch_size'] = params['validation']['batch_size']

    print(activity, mydict)
    return activity, mydict

def validation_func(mydict):
    print("Inference!")
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    # network = UNet(n_channels=3, n_classes=mydict['num_classes']).to(device)
    network = UNet_3d(in_dim=1, out_dim=8, num_filters=4).to(device)

    soft_dice_args = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
    ce_loss = nn.CrossEntropyLoss()
    
    sdl = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_args)
    l1_loss = nn.L1Loss()

    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
    else:
        print("No model found!")
        sys.exit()

    # Test Data

    if mydict['dataset'] == 'regress':

        # Validation Data
        validation_set = regress(mydict['validation_datalist'], mydict['files'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['batch_size'],shuffle=False,num_workers=1, drop_last=False)
    else:
        print ('Wrong dataloader!')

    validation_start_time = time()
    with torch.no_grad():
        network.eval()
        validation_iterator = iter(validation_generator)
        avg_dice = 0.0
        for i in range(len(validation_generator)):
            x, mask, y_gt, affine = next(validation_iterator)
            # x, mask, y_gt = next(validation_iterator)

            x = x.to(device, non_blocking=True)
            x = x.type(torch.cuda.FloatTensor)
            mask = mask.to(device, non_blocking=True)
            mask = mask.type(torch.cuda.FloatTensor) # make mask logic
            y_gt = y_gt.to(device, non_blocking=True)
            y_gt = y_gt.type(torch.cuda.FloatTensor)

            y_pred = network(x)
            pre_coor = y_pred[:,0:3,][0]
            pre_coor = pre_coor.permute(1, 2, 3, 0)
            pre_coor = pre_coor * 100

            pre_coor_nii = nib.Nifti1Image(pre_coor.cpu().detach().numpy(), affine=affine[0])
            pre_coor_nii.to_filename('samples/coor_pred.nii.gz')

            pre_sigma = torch.exp(y_pred[:,3:6,][0])
            pre_sigma = pre_sigma.permute(1, 2, 3, 0)
            pre_sigma_nii = nib.Nifti1Image(torch.exp(pre_sigma).cpu().detach().numpy(), affine=affine[0])
            pre_sigma_nii.to_filename('samples/sigma_pred.nii.gz')

            y_pred_binary = softmax_helper(y_pred[:,6:8,:])
            y_pred_binary = torch.argmax(y_pred[:,6:8,:], dim=1)
            y_pred_binary_nii = nib.Nifti1Image(y_pred_binary[0].to(torch.float).cpu().detach().numpy(), affine=affine[0])
            y_pred_binary_nii.to_filename('samples/mask_pred.nii.gz')

            avg_dice += sdl(y_pred[:,6:8,:], mask)

            regress_loss = l1_loss(y_pred[:,0:3,] * mask, y_gt*mask) / (1e-6 + torch.mean(mask))
            mask_loss = 0.75 * sdl(y_pred[:,6:8,:], mask) + 0.25 * ce_loss(y_pred[:,6:8,:], mask[:,0,:].type(torch.LongTensor).to(device))
            uncer_loss = 0.5 * torch.mean(y_pred[:,3,] * mask + (y_pred[:,0,] * mask - y_gt[:,0,:] * mask) ** 2/(1e-3 * torch.exp(y_pred[:,3,])) + \
                                          y_pred[:,4,] * mask + (y_pred[:,1,] * mask - y_gt[:,1,:] * mask) ** 2/(1e-3 * torch.exp(y_pred[:,4,])) + \
                                          y_pred[:,5,] * mask + (y_pred[:,2,] * mask - y_gt[:,2,:] * mask) ** 2/(1e-3 * torch.exp(y_pred[:,5,]))) / (1e-6 + torch.mean(mask))
            
            print("Validation: regress_loss: {:.4f}, mask_loss: {:.4f}, uncer_loss: {:.4f}".format(regress_loss, mask_loss, uncer_loss))
            import pdb; pdb.set_trace()

        avg_dice = -avg_dice # because SoftDice returns negative dice
        avg_dice /= len(validation_generator)
    validation_end_time = time()
    print("End of epoch validation took {} seconds.\nAverage dice: {}".format(validation_end_time - validation_start_time, avg_dice))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        activity, mydict = parse_func(args)

    # call train
    validation_func(mydict)
