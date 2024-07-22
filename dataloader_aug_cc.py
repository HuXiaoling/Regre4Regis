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

from cornucopia import (
    RandomGaussianNoiseTransform,
    RandomSmoothTransform,
    RandomMulFieldTransform,
    RandomAffineElasticTransform,
    SequentialTransform,
)

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

        torch_img = torch.unsqueeze(img, dim=0).to(dtype=torch.float32)
        torch_mask = torch.unsqueeze(mask, dim=0).to(dtype=torch.int)
        torch_seg = torch.unsqueeze(seg, dim=0).to(dtype=torch.int)
        torch_gt = gt.permute(3, 0, 1, 2).to(dtype=torch.float32)

        if self.is_training:

            transform_intensity = SequentialTransform([
                cc.ctx.maybe(RandomSmoothTransform(include=torch_img), 1, shared=True),
                cc.ctx.maybe(RandomMulFieldTransform(include=torch_img, order=1), 1, shared=True),
                cc.ctx.maybe(RandomGaussianNoiseTransform(include=torch_img), 1, shared=True),
            ])

            transform_spatial = cc.ctx.maybe(RandomAffineElasticTransform(order=1), 1, shared=True)

            torch_img = transform_intensity(torch_img)
            torch_img, torch_mask, torch_gt, torch_seg = transform_spatial(torch_img, torch_mask, torch_gt, torch_seg)

            # torch_mask[torch_mask >= 0.5] = 1.0
            # torch_mask[torch_mask < 0.5] = 0.0
        
        label_list_segmentation = [0, 14, 15, 16,
                        2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 17, 18, 26, 28, 
                        41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60]

        n_labels = len(label_list_segmentation)

        # create look up table
        lut = torch.zeros(10000, dtype=torch.long)
        for l in range(n_labels):
            lut[label_list_segmentation[l]] = l

        onehotmatrix = torch.eye(n_labels, dtype=torch.float16)
        label = np.squeeze(torch_seg)
        torch_seg_onehot = onehotmatrix[lut[label]]
        # torch_label = torch.argmax(torch_onehot, axis=0).to(dtype=torch.int)
        # onehot_matrix = torch.eye(n_labels)[torch_label]
        torch_seg_onehot = torch_seg_onehot.permute(3, 0, 1, 2)

        return torch_img, torch_mask, torch_gt, torch_seg_onehot, affine

if __name__ == "__main__":
    start_time = time()
    training_set = regress('data_lists/regress/train_list.csv', 'data/', is_training=True)
    trainloader = data.DataLoader(training_set,batch_size=1,shuffle=True, drop_last=True) 

    batch = next(iter(trainloader))
    input, mask, target, seg_onehot, affine = batch

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
    
    import pdb; pdb.set_trace()
    discrete_labels = torch.argmax(seg_onehot, axis=1).to(dtype=torch.int)
    new_seg = nib.Nifti1Image(discrete_labels[0,:,:,:].cpu().detach().numpy(), affine=affine[0])
    new_seg.to_filename('samples/aug_seg.nii.gz')

    end_time = time()
    print("Dataloader took {} seconds.".format(end_time-start_time))
    import pdb; pdb.set_trace()