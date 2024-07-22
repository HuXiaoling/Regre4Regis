# train loop
# use validation to save checkpoint ---> take 2-3 vals only, OR take patch of all the vals
# lr schedule
import torch
import numpy as np

import argparse, json
import os, glob, sys
from time import time

from dataloader_aug import regress
from unet.unet_3d import UNet_3d
from utilities import SoftDiceLoss, softmax_helper
import torch
import pdb
import torch.nn as nn
from collections import OrderedDict

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
    mydict['output_folder'] = params['train']['output_folder']
    mydict['loss_weight_mask'] = params['train']['loss_weight_mask']
    mydict['loss_weight_uncer'] = params['train']['loss_weight_uncer']
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
        training_generator = torch.utils.data.DataLoader(training_set,batch_size=mydict['train_batch_size'],shuffle=True, drop_last=True)

        # Validation Data
        validation_set = regress(mydict['validation_datalist'], mydict['files'])
        validation_generator = torch.utils.data.DataLoader(validation_set,batch_size=mydict['validation_batch_size'],shuffle=False, drop_last=False)
    else:
        print ('Wrong dataloader!')

    # Network
    # network = UNet_3d(n_channels=3, n_classes=mydict['num_classes']).to(device)
    network = UNet_3d(in_dim=1, out_dim=5, num_filters=4).to(device)
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

    if not os.path.exists(mydict['output_folder']):
        os.makedirs(mydict['output_folder'])

    logfile = os.path.join(mydict['output_folder'], 'parameters.txt')
    log_file = open(logfile, 'w')
    p = OrderedDict()
    p['loss_weight_mask'] = mydict['loss_weight_mask']
    p['loss_weight_uncer'] = mydict['loss_weight_uncer']
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
    best_dict['mask_dice'] = 1000.0
    print("Let the training begin!")

    num_batches = len(training_generator)
    for epoch in range(mydict['num_epochs']):

        network.to(device).train() # after .eval() in validation

        avg_train_loss = 0.0
        epoch_start_time = time()

        training_iterator = iter(training_generator)
        for step in range(num_batches):
            optimizer.zero_grad()

            print("Step {}.".format(step))
            x, mask, y_gt, affine = next(training_iterator)
            x = x.to(device, non_blocking=True)
            x = x.type(torch.cuda.FloatTensor)
            mask = mask.to(device, non_blocking=True)
            mask = mask.type(torch.cuda.FloatTensor) # make mask logic
            y_gt = y_gt.to(device, non_blocking=True)
            y_gt = y_gt.type(torch.cuda.FloatTensor)

            y_pred = network(x)

            # import pdb; pdb.set_trace()
            # regress_loss: divide by number of foreground pixel (inefficient)
            # mask_logic = torch.cat([mask, mask, mask], dim=1)
            # regress_loss = l1_loss(y_pred[:,0:3,][mask], y_gt[mask]) # make mask logic

            train_loss = l1_loss(y_pred[:,0:3,] * mask, y_gt*mask) / (1e-6 + torch.mean(mask)) +\
                mydict['loss_weight_mask']* (0.75 * sdl(y_pred[:,3:5,:], mask) + 0.25 * ce_loss(y_pred[:,3:5,:], mask[:,0,:].type(torch.LongTensor).to(device)))
            avg_train_loss += train_loss

            train_loss.backward()
            optimizer.step()

        avg_train_loss /= num_batches
        epoch_end_time = time()
        print("Epoch {} took {} seconds.\nAverage training loss: {}".format(epoch, epoch_end_time-epoch_start_time, avg_train_loss))

        validation_start_time = time()
        if epoch % 2 == 0:
            with torch.no_grad():
                network.eval()
                validation_iterator = iter(validation_generator)
                avg_val_loss = 0.0
                mask_dice = 0.0
                for validation_step in range(len(validation_generator)):
                    print("Validation Step {}.".format(validation_step))
                    x, mask, y_gt, _ = next(validation_iterator)
                    x = x.to(device, non_blocking=True)
                    x = x.type(torch.cuda.FloatTensor)
                    mask = mask.to(device, non_blocking=True)
                    mask = mask.type(torch.cuda.FloatTensor)
                    y_gt = y_gt.to(device, non_blocking=True)
                    y_gt = y_gt.type(torch.cuda.FloatTensor)

                    y_pred = network(x)

                    val_loss = l1_loss(y_pred[:,0:3,] * mask, y_gt*mask)/ (1e-6 + torch.mean(mask)) +\
                          mydict['loss_weight_mask']* (0.75 * sdl(y_pred[:,3:5,:], mask) + 0.25 * ce_loss(y_pred[:,3:5,:], mask[:,0,:].type(torch.LongTensor).to(device)))
                    avg_val_loss += val_loss
                    mask_dice += sdl(y_pred[:,3:5,:], mask)
                mask_dice = -mask_dice # because SoftDice returns negative dice
                mask_dice /= len(validation_generator)
                avg_val_loss /= len(validation_generator)
            validation_end_time = time()
            print("End of epoch validation took {} seconds.\nAverage validation loss: {}".format(validation_end_time - validation_start_time, avg_val_loss))

            # check for best epoch and save it if it is and print
            if epoch == 0:
                best_dict['epoch'] = epoch
                best_dict['mask_dice'] = mask_dice
            else:
                if best_dict['mask_dice'] < mask_dice:
                    best_dict['mask_dice'] = mask_dice
                    best_dict['epoch'] = epoch
            if epoch == best_dict['epoch']:
                torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_best.pth"))
            print("Best epoch so far: {}\n".format(best_dict))

            # save checkpoint for save_every
            if epoch % mydict['save_every'] == 0:
                torch.save(network.state_dict(), os.path.join(mydict['output_folder'], "model_epoch" + str(epoch) + ".pth"))

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
