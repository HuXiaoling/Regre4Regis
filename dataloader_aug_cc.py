# load all into cpu
# do cropping to patch size
# restrict number of slices (see null slices)
# normalize range
# test code by outputting few patches
# training, testing, val
import torch
import os, glob, sys
import numpy as np
from PIL import Image
from os.path import join as pjoin
from torchvision import transforms
from torch.utils import data
from skimage import io
import nibabel as nib
import cornucopia as cc
import random
from time import time

class regress(data.Dataset):
    def __init__(self, listpath, folderpaths, is_training=False):

        self.listpath = listpath
        self.imgfolder = folderpaths
        self.gtfolder = folderpaths

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['mask'] = []
        self.dataCPU['seg'] = []
        self.dataCPU['coord'] = []
        self.dataCPU['affine'] = []

        self.indices = []
        self.to_tensor = transforms.ToTensor()
        self.is_training = is_training

        self.loadCPU()

    def loadCPU(self):
        with open(self.listpath, 'r') as f:
            mylist = f.readlines()
        mylist = [x.rstrip('\n') for x in mylist]

        for i, entry in enumerate(mylist):

            components = entry.split('.')
            filename = components[0]

            im_path = pjoin(self.imgfolder, filename) + '.image.nii.gz'
            mask_path = pjoin(self.imgfolder, filename) + '.mask.nii.gz'
            seg_path = pjoin(self.imgfolder, filename) + '.seg.nii.gz'
            gt_path = pjoin(self.gtfolder, filename) + '.mni_coords.nii.gz'

            self.indices.append((i))
            self.dataCPU['image'].append(im_path)
            self.dataCPU['mask'].append(mask_path)
            self.dataCPU['seg'].append(seg_path)
            self.dataCPU['coord'].append(gt_path)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]

        im_path = self.dataCPU['image'][index] #HW
        mask_path = self.dataCPU['mask'][index] #HW
        seg_path = self.dataCPU['seg'][index] #HW
        gt_path = self.dataCPU['coord'][index] #HW

        img = nib.load(im_path).get_fdata()
        affine = nib.load(im_path).affine
        mask = nib.load(mask_path).get_fdata()
        seg = nib.load(seg_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()
        
        img = torch.from_numpy(img)
        seg = torch.from_numpy(seg)
        seg[seg == 24] = 0
        mask = torch.from_numpy(mask)
        
        valid_value = img * mask
        non_zero_values = valid_value[valid_value != 0]
        median_non_zero = torch.median(non_zero_values)
        img = img/median_non_zero
        gt = torch.from_numpy(gt/100)

        torch_img = torch.unsqueeze(img, dim=0).to(dtype=torch.float)
        torch_mask = torch.unsqueeze(mask, dim=0).to(dtype=torch.int)
        torch_seg = torch.unsqueeze(seg, dim=0).to(dtype=torch.int)
        torch_gt = gt.permute(3, 0, 1, 2).to(dtype=torch.float)

        return torch_img, torch_mask, torch_gt, torch_seg, affine

if __name__ == "__main__":
    start_time = time()
    training_set = regress('data_lists/regress/train_list.csv', 'data/', is_training=True)
    trainloader = data.DataLoader(training_set,batch_size=1,shuffle=True, drop_last=True) 

    batch = next(iter(trainloader))
    input, mask, target, seg, affine = batch

    # padded_input = torch.zeros(1,1,256, 256, 256)
    # padded_input[0,0,0:input.shape[2], 0:input.shape[3], 0:input.shape[4]] = input[0,0,:,:,:]

    new_image = nib.Nifti1Image(input[0,0,:,:,:].cpu().detach().numpy(), affine=affine[0])
    new_image.to_filename('samples/aug_image.nii.gz')

    new_mask = nib.Nifti1Image(mask[0,0,:,:,:].cpu().detach().numpy(), affine=affine[0])
    new_mask.to_filename('samples/aug_mask.nii.gz')

    target = target[0,:,:,:,:]
    target = target.permute(1, 2, 3, 0)
    new_target = nib.Nifti1Image(target.cpu().detach().numpy(), affine=affine[0])
    new_target.to_filename('samples/aug_target.nii.gz')
    
    new_seg = nib.Nifti1Image(seg[0,0,:,:,:].cpu().detach().numpy(), affine=affine[0])
    new_seg.to_filename('samples/aug_seg.nii.gz')

    end_time = time()
    print("Dataloader took {} seconds.".format(end_time-start_time))
    import pdb; pdb.set_trace()