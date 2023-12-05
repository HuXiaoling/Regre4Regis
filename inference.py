# train loop
# use validation to save checkpoint ---> take 2-3 vals only, OR take patch of all the vals
# lr schedule
import torch
import numpy as np

import argparse, json
import os, glob, sys
from time import time

from dataloader import CREMI
from unet.unet_3d import UNet_3d
from utilities import DC_and_CE_loss, SoftDiceLoss, softmax_helper, IOU
import torch
torch.cuda.empty_cache()
from torchvision.utils import save_image


def parse_func(args):
    ### Reading the parameters json file
    print("Reading params file {}...".format(args.params))
    with open(args.params, 'r') as f:
        params = json.load(f)

    activity = params['common']['activity']
    mydict = {}
    mydict['dataset'] = params['common']['dataset']
    mydict['num_classes'] = int(params['common']['num_classes'])
    mydict['files'] = [params['common']['img_file'], params['common']['gt_file']]
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
    network = UNet_3d(in_dim=3, out_dim=2, num_filters=4).to(device)

    soft_dice_args = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
    val_dice_func = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_args)
    val_iou_func = IOU(apply_nonlin=softmax_helper, **soft_dice_args)

    if mydict['checkpoint_restore'] != "":
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
    else:
        print("No model found!")
        sys.exit()

    # Test Data

    if mydict['dataset'] == 'CREMI':

        # Validation Data
        validation_set = CREMI(mydict['validation_datalist'], mydict['files'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['batch_size'],shuffle=False,num_workers=1, drop_last=False)
    else:
        print ('Wrong dataloader!')

    validation_start_time = time()
    with torch.no_grad():
        network.eval()
        validation_iterator = iter(validation_generator)
        avg_dice = 0.0
        avg_iou = 0.0
        for i in range(len(validation_generator)):
            x, y_gt = next(validation_iterator)
            x = x.to(device, non_blocking=True)
            x = x.type(torch.cuda.FloatTensor)
            y_gt = y_gt.to(device, non_blocking=True)
            y_gt = y_gt.type(torch.cuda.FloatTensor)

            y_pred = network(x)

            avg_dice += val_dice_func(y_pred, y_gt)
            avg_iou += val_iou_func(y_pred, y_gt)
            y_pred_binary = softmax_helper(y_pred)
            y_pred_binary = torch.argmax(y_pred, dim=1)

            for j in range(x.shape[4]):
                save_image(x[0,:,:,:,j], os.path.join(mydict['output_folder'], 'img' + str(mydict['batch_size'] * i + j) + '.png'))
                save_image(y_gt[0,:,:,:,j], os.path.join(mydict['output_folder'], 'img' + str(mydict['batch_size'] * i + j) + '_gt.png'))
                save_image(y_pred[0,:,:,j].float(), os.path.join(mydict['output_folder'], 'img' + str(mydict['batch_size'] * i + j) + '_pred.png'))
                save_image(y_pred_binary[0,:,:,j].float(), os.path.join(mydict['output_folder'], 'img' + str(mydict['batch_size'] * i + j) + '_pred_binary.png'))

        avg_dice = -avg_dice # because SoftDice returns negative dice
        avg_dice /= len(validation_generator)
        avg_iou /= len(validation_generator)
    validation_end_time = time()
    print("End of epoch validation took {} seconds.\nAverage dice: {}".format(validation_end_time - validation_start_time, avg_dice))
    print("End of epoch validation took {} seconds.\nAverage iou: {}".format(validation_end_time - validation_start_time, avg_iou))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    parser.add_argument('--warp', type=float, default = 1e-2, help="the weight for warping loss")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        activity, mydict = parse_func(args)

    mydict['checkpoint_restore'] = mydict['checkpoint_restore'] + '_' + str(args.warp) + '/model_best.pth'
    mydict['output_folder'] = mydict['output_folder'] + '_' + str(args.warp)
    # call train
    validation_func(mydict)
