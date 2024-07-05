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
import torchio as tio
import random
from time import time

class regress(data.Dataset):
    def __init__(self, listpath, folderpaths, is_training=False):

        self.listpath = listpath
        self.imgfolder = folderpaths
        self.gtfolder = folderpaths

        self.tio_transform = tio.Compose([
        tio.RandomBlur(p=0.5),
        tio.RandomBiasField(p=0.5),
        tio.RandomNoise(p=0.5),
        # tio.RandomAffine(p=0.5) ,
        # tio.RandomElasticDeformation(p=0.5)
        ])

        self.dataCPU = {}
        self.dataCPU['image'] = []
        self.dataCPU['mask'] = []
        self.dataCPU['label'] = []
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
            gt_path = pjoin(self.gtfolder, filename) + '.mni_coords.nii.gz'

            self.indices.append((i))
            self.dataCPU['image'].append(im_path)
            self.dataCPU['mask'].append(mask_path)
            self.dataCPU['label'].append(gt_path)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index): # return CHW torch tensor
        index = self.indices[index]

        im_path = self.dataCPU['image'][index] #HW
        mask_path = self.dataCPU['mask'][index] #HW
        gt_path = self.dataCPU['label'][index] #HW

        img = nib.load(im_path).get_fdata()
        affine = nib.load(im_path).affine
        mask = nib.load(mask_path).get_fdata()
        gt = nib.load(gt_path).get_fdata()
        
        img = torch.from_numpy(img)
        mask = torch.from_numpy(mask)
        
        valid_value = img * mask
        non_zero_values = valid_value[valid_value != 0]
        median_non_zero = np.median(non_zero_values)
        img = img/median_non_zero
        gt = torch.from_numpy(gt/100)

        torch_img = torch.unsqueeze(img, dim=0)
        torch_mask = torch.unsqueeze(mask, dim=0)
        torch_gt = gt.permute(3, 0, 1, 2)

        if self.is_training:
            torch_img = self.tio_transform(torch_img)

            # # random flip augmentation
            # sigma = random.uniform(0, 1)
            # if sigma >= 0.5:
            #     random_flip = tio.RandomFlip(p=1)
            #     torch_img = random_flip(torch_img)
            #     torch_mask = random_flip(torch_mask)
            #     torch_gt[0,:,:,:] = -torch_gt[0,:,:,:]

            # random affine transformation 

            sigma_affine = random.uniform(0, 1)
            if sigma_affine >= 0.5:
                random_affine = tio.RandomAffine(p=1)

                scaling_params, rotation_params, translation_params = random_affine.get_params(
                    random_affine.scales,
                    random_affine.degrees,
                    random_affine.translation,
                    random_affine.isotropic,
                )

                arguments_affine = {
                    'scales': scaling_params.tolist(),
                    'degrees': rotation_params.tolist(),
                    'translation': translation_params.tolist(),
                    'center': random_affine.center,
                    'default_pad_value': random_affine.default_pad_value,
                    'image_interpolation': random_affine.image_interpolation,
                    'label_interpolation': random_affine.label_interpolation,
                    'check_shape': random_affine.check_shape,
                }

                transform_affine = tio.Affine(**(arguments_affine))
                torch_img = transform_affine(torch_img)
                torch_mask = transform_affine(torch_mask)
                torch_gt = transform_affine(torch_gt)

            # random elastic deformation 
            sigma_deformation = random.uniform(0, 1)
            if sigma_deformation >= 0.5:        
                elastic_deformation = tio.RandomElasticDeformation(p=1)

                control_points = elastic_deformation.get_params(
                    elastic_deformation.num_control_points,
                    elastic_deformation.max_displacement,
                    elastic_deformation.num_locked_borders,
                )

                arguments_deformation = {
                    'control_points': control_points,
                    'max_displacement': elastic_deformation.max_displacement,
                    'image_interpolation': elastic_deformation.image_interpolation,
                    'label_interpolation': elastic_deformation.label_interpolation,
                }

                transform_deformation = tio.ElasticDeformation(**arguments_deformation)
                torch_img = transform_deformation(torch_img)
                torch_mask = transform_deformation(torch_mask)
                torch_gt = transform_deformation(torch_gt)

            torch_mask[torch_mask >= 0.5] = 1.0
            torch_mask[torch_mask < 0.5] = 0.0

        return torch_img, torch_mask, torch_gt, affine

if __name__ == "__main__":
    start_time = time()
    training_set = regress('data_lists/regress/train_list.csv', 'data/', is_training=True)
    trainloader = data.DataLoader(training_set,batch_size=1,shuffle=True, drop_last=True) 

    batch = next(iter(trainloader))
    input, mask, target, affine = batch

    new_image = nib.Nifti1Image(input.squeeze().cpu().detach().numpy(), affine=affine[0])
    new_image.to_filename('samples/aug_image.nii.gz')

    new_mask = nib.Nifti1Image(mask.squeeze().cpu().detach().numpy(), affine=affine[0])
    new_mask.to_filename('samples/aug_mask.nii.gz')

    target = target.squeeze()
    target = target.permute(1, 2, 3, 0)
    new_target = nib.Nifti1Image(target.cpu().detach().numpy(), affine=affine[0])
    new_target.to_filename('samples/aug_target.nii.gz')

    end_time = time()
    print("Dataloader took {} seconds.".format(end_time-start_time))
    import pdb; pdb.set_trace()