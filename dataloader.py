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

class regress(data.Dataset):
    def __init__(self, listpath, folderpaths, is_training=False):

        self.listpath = listpath
        self.imgfolder = folderpaths
        self.gtfolder = folderpaths

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['mask'] = []
        self.dataCPU['label'] = []

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
            gt_path = pjoin(self.gtfolder, filename) + '.mni_coords.nii.gz'
            img = nib.load(im_path).get_fdata()
            mask = nib.load(mask_path).get_fdata()
            gt = nib.load(gt_path).get_fdata()

            # img = self.to_tensor(img)
            # mask = self.to_tensor(mask)
            # gt = self.to_tensor(gt)
            img = torch.from_numpy(img)
            mask = torch.from_numpy(mask)
            gt = torch.from_numpy(gt)

            #normalize within a channel
            # meanval = img.mean()
            # stdval = img.std()
            # img = (img - meanval) / stdval

            self.indices.append((i))

            #cpu store
            self.dataCPU['image'].append(img)
            self.dataCPU['mask'].append(mask)
            self.dataCPU['label'].append(gt)


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]

        torch_img = self.dataCPU['image'][index] #HW
        torch_mask = self.dataCPU['mask'][index] #HW
        torch_gt = self.dataCPU['label'][index] #HW

        torch_img = torch.unsqueeze(torch_img, dim=0)
        torch_mask = torch.unsqueeze(torch_mask, dim=0)
        torch_gt = torch_gt.permute(3, 0, 1, 2)
        return torch_img, torch_mask, torch_gt


if __name__ == "__main__":
    flag = "training"

    dst = regress('data_lists/regress/debug_list.csv', 'data/', is_training= True)
    
    trainloader = data.DataLoader(dst, batch_size=1, num_workers=1)
    ## dataloader check
    batch = next(iter(trainloader))
    input, mask, target = batch
    import pdb; pdb.set_trace()