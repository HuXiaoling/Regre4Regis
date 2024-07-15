# train loop
# use validation to save checkpoint ---> take 2-3 vals only, OR take patch of all the vals
# lr schedule
import torch
import numpy as np

import argparse, json
import os, glob, sys
from time import time
import cornucopia as cc
from cornucopia import (
    RandomGaussianNoiseTransform,
    RandomSmoothTransform,
    RandomMulFieldTransform,
    RandomAffineElasticTransform,
    SequentialTransform,
)

# from dataloader_aug import regress
from dataloader_aug_cc import regress
from unet.unet_3d import UNet_3d
from utilities import SoftDiceLoss, softmax_helper
import torch
import pdb
import torch.nn as nn
from collections import OrderedDict
from fitting_functions import least_square_fitting
from torch.utils.tensorboard import SummaryWriter

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

    mydict['train_datalist'] = params['train']['train_datalist']
    mydict['validation_datalist'] = params['train']['validation_datalist']
    mydict['mode'] = params['train']['mode']
    mydict['regress_loss'] = params['train']['regress_loss']
    mydict['uncer'] = params['train']['uncer']
    mydict['output_folder'] = params['train']['output_folder']
    mydict['loss_weight'] = params['train']['loss_weight']
    mydict['loss_weight_uncer'] = params['train']['loss_weight_uncer']
    mydict['num_workers'] = params['train']['num_workers']
    mydict['train_batch_size'] = int(params['train']['train_batch_size'])
    mydict['validation_batch_size'] = int(params['train']['validation_batch_size'])
    mydict['learning_rate'] = float(params['train']['learning_rate'])
    mydict['num_epochs'] = int(params['train']['num_epochs'])
    mydict['save_every'] = params['train']['save_every']

    print(activity, mydict)
    return activity, mydict


def set_seed(): # reproductibility 
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def train_func(mydict):
    # Reproducibility, and Cuda setup
    set_seed()
    device = torch.device("cuda")
    print("CUDA device: {}".format(device))

    if not torch.cuda.is_available():
        print("WARNING!!! You are attempting to run training on a CPU (torch.cuda.is_available() is False). This can be VERY slow!")

    # Train Data

    if mydict['dataset'] == 'regress':
        training_set = regress(mydict['train_datalist'], mydict['files'], is_training= True)
        training_generator = torch.utils.data.DataLoader(training_set,batch_size=mydict['train_batch_size'],\
                                                         shuffle=True, drop_last=True, num_workers=mydict['num_workers'], pin_memory=False)

        # Validation Data
        validation_set = regress(mydict['validation_datalist'], mydict['files'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],\
                                                           shuffle=False, drop_last=False, num_workers=mydict['num_workers'], pin_memory=False)
    else:
        print ('Wrong dataloader!')

    # Network
    # network = UNet_3d(n_channels=3, n_classes=mydict['num_classes']).to(device)
    network = UNet_3d(in_dim=1, out_dim=8, num_filters=4).to(device)
    # network = torch.nn.DataParallel(network, device_ids=range(torch.cuda.device_count()))

    # Optimizer
    optimizer = torch.optim.SGD(network.parameters(), mydict['learning_rate'], weight_decay=0.00003, momentum=0.99, nesterov=True)

    # Load checkpoint (if specified)
    if os.path.exists(mydict['output_folder'] + '/model_best.pth'):
        print('Finetune the best model!')
        network.load_state_dict(torch.load(mydict['output_folder'] + '/model_best.pth'), strict=True)
    elif mydict['checkpoint_restore'] != "" and os.path.exists(mydict['checkpoint_restore']):
        print('Load the best baseline model!')
        network.load_state_dict(torch.load(mydict['checkpoint_restore']), strict=True)
    else:
        print('Train from scratch!')


    # Losses
    soft_dice_args = {'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}
    ce_loss = nn.CrossEntropyLoss()

    sdl = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_args)
    l1_loss = nn.L1Loss()
    l2_loss = nn.MSELoss()

    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    logfile = os.path.join(mydict['output_folder'], 'parameters.txt')
    log_file = open(logfile, 'w')
    p = OrderedDict()
    p['mode'] = mydict['mode']
    p['regress_loss'] = mydict['regress_loss']
    p['uncer'] = mydict['uncer']
    p['loss_weight_seg'] = mydict['loss_weight']
    p['loss_weight_uncer'] = mydict['loss_weight_uncer']
    p['num_workers'] = mydict['num_workers']
    p['learning_rate'] = mydict['learning_rate']
    p['epochs'] = mydict['num_epochs']
    p['checkpoint_restore'] = mydict["checkpoint_restore"]
    p['output_folder'] = mydict["output_folder"]

    for key, val in p.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()

    # Train loop
    best_dict = {}
    best_dict['epoch'] = 0
    best_dict['avg_val_loss'] = 1000.0
    print("Let the training begin!")

    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter(mydict['output_folder'])
    
    num_batches = len(training_generator)
    for epoch in range(mydict['num_epochs']):

        network.to(device).train() # after .eval() in validation

        avg_train_loss = 0.0
        epoch_start_time = time()

        training_iterator = iter(training_generator)
        for step in range(num_batches):
            optimizer.zero_grad()

            print("Step {}.".format(step))
            x, mask, y_gt, seg, affine = next(training_iterator)
            x = x.to(device, non_blocking=True)
            x = x.type(torch.cuda.FloatTensor)
            mask = mask.to(device, non_blocking=True)
            # mask = mask.type(torch.cuda.FloatTensor) # make mask logic
            y_gt = y_gt.to(device, non_blocking=True)
            y_gt = y_gt.type(torch.cuda.FloatTensor)
            seg = seg.to(device, non_blocking=True)

            # Data Augmentation
            transform_intensity = cc.ctx.batch(SequentialTransform([
                cc.ctx.maybe(RandomSmoothTransform(include=x), 0.5, shared=True),
                cc.ctx.maybe(RandomMulFieldTransform(include=x, order=1), 0.5, shared=True),
                cc.ctx.maybe(RandomGaussianNoiseTransform(include=x), 0.5, shared=True),
            ]))

            transform_spatial = cc.ctx.batch(SequentialTransform([
                cc.ctx.maybe(RandomAffineElasticTransform(order=1), 0.5, shared=True),
            ]))

            x = transform_intensity(x)
            x, mask, y_gt, seg = transform_spatial(x, mask, y_gt, seg)
            mask = mask.type(torch.cuda.FloatTensor)
            seg = seg.type(torch.cuda.FloatTensor) 

            # one hot encoding
            label_list_segmentation = [0, 14, 15, 16, 24, 77, 85, 
                            2, 3, 4, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 
                            41, 42, 43, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]

            n_labels = len(label_list_segmentation)

            # create look up table
            lut = torch.zeros(10000, dtype=torch.long, device=device)
            for l in range(n_labels):
                lut[label_list_segmentation[l]] = l

            onehotmatrix = torch.eye(n_labels, dtype=torch.float64, device=device)
            label = np.squeeze(seg)
            seg_onehot = onehotmatrix[lut[label.long()]]
            seg_onehot = seg_onehot.permute(0, 4, 1, 2, 3)

            # test dataloader_aug_cc.py
            # import nibabel as nib
            # new_image = nib.Nifti1Image(x[0,0,:,:,:].cpu().detach().numpy(), affine=affine[0])
            # new_image.to_filename('samples/aug_image.nii.gz')

            # new_mask = nib.Nifti1Image(mask[0,0,:,:,:].cpu().detach().numpy(), affine=affine[0])
            # new_mask.to_filename('samples/aug_mask.nii.gz')

            # y_gt = y_gt[0,:,:,:,:]
            # y_gt = y_gt.permute(1, 2, 3, 0)
            # new_target = nib.Nifti1Image(y_gt.cpu().detach().numpy(), affine=affine[0])
            # new_target.to_filename('samples/aug_target.nii.gz')

            # discrete_labels = torch.unsqueeze(torch.argmax(seg_onehot, dim=1), dim=1).to(dtype=torch.int)
            # new_seg = nib.Nifti1Image(discrete_labels[0,0,:,:,:].cpu().detach().numpy(), affine=affine[0])
            # new_seg.to_filename('samples/aug_seg.nii.gz')

            DEFaff = torch.empty_like(x)
            DEFaffseg = torch.empty_like(mask)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                y_pred = network(x)
                for i in range(x.shape[0]):
                    channels_to_select = [0, 1, 2, 6, 7]
                    DEFaff[i], DEFaffseg[i] = least_square_fitting(x[i,:], y_pred[i, channels_to_select, :].to(dtype=torch.float))

                seg_loss = 0.75 * sdl(y_pred[:,6:8,:], mask) + 0.25 * ce_loss(y_pred[:,6:8,:], mask[:,0,:].type(torch.LongTensor).to(device))

                if mydict['mode'] == 'pre': 
                    if mydict['regress_loss'] == 'l1':
                        print("We are using L1 loss for regression!")
                        regress_loss = l1_loss(y_pred[:,0:3,] * mask, y_gt * mask) / (1e-6 + torch.mean(mask))
                    else:
                        print("We are using L2 loss for regression!")
                        regress_loss = l2_loss(y_pred[:,0:3,] * mask, y_gt * mask) / (1e-6 + torch.mean(mask))

                    train_loss = regress_loss + mydict['loss_weight'] * seg_loss
                else:
                    if mydict['uncer'] == 'gaussian':
                        print("We are using Gaussian distribution to model uncertainty for three sigmas!")
                        
                        uncer_loss = 0.5 * torch.mean(y_pred[:,3,] * mask + (y_pred[:,0,] * mask - y_gt[:,0,:] * mask) ** 2/(1e-3 * torch.exp(y_pred[:,3,])) + \
                                                      y_pred[:,4,] * mask + (y_pred[:,1,] * mask - y_gt[:,1,:] * mask) ** 2/(1e-3 * torch.exp(y_pred[:,4,])) + \
                                                      y_pred[:,5,] * mask + (y_pred[:,2,] * mask - y_gt[:,2,:] * mask) ** 2/(1e-3 * torch.exp(y_pred[:,5,]))) / (1e-6 + torch.mean(mask))

                        train_loss = mydict['loss_weight'] * seg_loss + mydict['loss_weight_uncer'] * uncer_loss

                    else:
                        print("We are using Laplacian distribution to model uncertainty for three sigmas!")

                        uncer_loss = torch.mean(y_pred[:,3,] * mask + l1_loss(y_pred[:,0,] * mask / (0.03 * torch.exp(y_pred[:,3,])), y_gt[:,0,:] * mask / (0.03 * torch.exp(y_pred[:,3,]))) + \
                                                y_pred[:,4,] * mask + l1_loss(y_pred[:,1,] * mask / (0.03 * torch.exp(y_pred[:,4,])), y_gt[:,1,:] * mask / (0.03 * torch.exp(y_pred[:,4,]))) + \
                                                y_pred[:,5,] * mask + l1_loss(y_pred[:,2,] * mask / (0.03 * torch.exp(y_pred[:,5,])), y_gt[:,2,:] * mask / (0.03 * torch.exp(y_pred[:,5,])))) / (1e-6 + torch.mean(mask))

                        train_loss = mydict['loss_weight'] * seg_loss + mydict['loss_weight_uncer'] * uncer_loss

                if torch.isnan(train_loss):
                    print("NaN detected in training loss. Exiting...")
                    sys.exit(1)
                avg_train_loss += train_loss

            scaler.scale(train_loss).backward()
            scaler.unscale_(optimizer)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()

        avg_train_loss /= num_batches
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        epoch_end_time = time()
        print("Epoch {} took {} seconds.\nAverage training loss: {}".format(epoch, epoch_end_time-epoch_start_time, avg_train_loss))

        validation_start_time = time()
        if epoch % 2 == 0:
            with torch.no_grad():
                network.eval()
                validation_iterator = iter(validation_generator)
                avg_val_loss = 1000.0
                seg_dice = 0.0
                for validation_step in range(len(validation_generator)):
                    print("Validation Step {}.".format(validation_step))
                    x, mask, y_gt, seg, affine = next(validation_iterator)
                    x = x.to(device, non_blocking=True)
                    x = x.type(torch.cuda.FloatTensor)
                    mask = mask.to(device, non_blocking=True)
                    mask = mask.type(torch.cuda.FloatTensor)
                    y_gt = y_gt.to(device, non_blocking=True)
                    y_gt = y_gt.type(torch.cuda.FloatTensor)

                    y_pred = network(x)
                    
                    regress_loss = l1_loss(y_pred[:,0:3,] * mask, y_gt * mask) / (1e-6 + torch.mean(mask))
                    seg_loss = 0.75 * sdl(y_pred[:,6:8,:], mask) + 0.25 * ce_loss(y_pred[:,6:8,:], mask[:,0,:].type(torch.LongTensor).to(device))

                    if mydict['uncer'] == 'gaussian':
                        uncer_loss = 0.5 * torch.mean(y_pred[:,3,] * mask + (y_pred[:,0,] * mask - y_gt[:,0,:] * mask) ** 2/(1e-3 * torch.exp(y_pred[:,3,])) + \
                                                      y_pred[:,4,] * mask + (y_pred[:,1,] * mask - y_gt[:,1,:] * mask) ** 2/(1e-3 * torch.exp(y_pred[:,4,])) + \
                                                      y_pred[:,5,] * mask + (y_pred[:,2,] * mask - y_gt[:,2,:] * mask) ** 2/(1e-3 * torch.exp(y_pred[:,5,]))) / (1e-6 + torch.mean(mask))

                    else:
                        uncer_loss = torch.mean(y_pred[:,3,] * mask + l1_loss(y_pred[:,0,] * mask / (0.03 * torch.exp(y_pred[:,3,])), y_gt[:,0,:] * mask / (0.03 * torch.exp(y_pred[:,3,]))) + \
                                                y_pred[:,4,] * mask + l1_loss(y_pred[:,1,] * mask / (0.03 * torch.exp(y_pred[:,4,])), y_gt[:,1,:] * mask / (0.03 * torch.exp(y_pred[:,4,]))) + \
                                                y_pred[:,5,] * mask + l1_loss(y_pred[:,2,] * mask / (0.03 * torch.exp(y_pred[:,5,])), y_gt[:,2,:] * mask / (0.03 * torch.exp(y_pred[:,5,])))) / (1e-6 + torch.mean(mask))

                    if mydict['mode'] == 'pre':
                        val_loss = regress_loss + mydict['loss_weight'] * seg_loss
                    else:
                        val_loss = regress_loss + mydict['loss_weight'] * seg_loss + mydict['loss_weight_uncer'] * uncer_loss
                    
                    avg_val_loss += val_loss
                    seg_dice += sdl(y_pred[:,6:8,:], mask)
                seg_dice = -seg_dice # because SoftDice returns negative dice
                seg_dice /= len(validation_generator)
                avg_val_loss /= len(validation_generator)
            validation_end_time = time()
            writer.add_scalar('Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Dice/val', seg_dice, epoch)
            print("End of epoch validation took {} seconds.\nAverage validation loss: {}.\nAverage dice: {}"
                  .format(validation_end_time - validation_start_time, avg_val_loss, seg_dice))

            # check for best epoch and save it if it is and print
            if epoch == 0:
                best_dict['epoch'] = epoch
                best_dict['avg_val_loss'] = avg_val_loss
            else:
                if best_dict['avg_val_loss'] > avg_val_loss:
                    best_dict['avg_val_loss'] = avg_val_loss
                    best_dict['epoch'] = epoch
            if epoch == best_dict['epoch']:
                torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_best.pth"))
                # torch.save({'epoch': epoch, 'dice': seg_dice, 'model_state_dict': network.state_dict(),}, 
                #            os.path.join(mydict['output_folder'], "model_best.pth"))
            print("Best epoch so far: {}\n".format(best_dict))

            # save checkpoint for save_every
            if epoch % mydict['save_every'] == 0:
                torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_epoch" + str(epoch) + ".pth"))
                torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_last.pth"))
                # torch.save({'epoch': epoch, 'dice': seg_dice, 'model_state_dict': network.state_dict(),}, 
                #            os.path.join(mydict['output_folder'], "model_epoch" + str(epoch) + ".pth"))
                # torch.save({'epoch': epoch, 'dice': seg_dice, 'model_state_dict': network.state_dict(),}, 
                #            os.path.join(mydict['output_folder'], "model_last.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--params', type=str, help="Path to the parameters file")
    parser.add_argument('--train_batch', type=int, default = 1, help="batch size for training")
    
    if len(sys.argv) == 1:
        print("Path to parameters file not provided. Exiting...")

    else:
        args = parser.parse_args()
        activity, mydict = parse_func(args)

    # mydict['train_batch_size'] = args.train_batch

    with open(args.params, 'r') as f:
        params = json.load(f)
    # mydict['output_folder'] = params['train']['output_folder']
    mydict['output_folder'] = params['train']['output_folder']
    # call train
    train_func(mydict)
