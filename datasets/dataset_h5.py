import time

import h5py
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class Whole_Slide_Bag(Dataset):
    def __init__(self,
                 file_path,
                 img_transforms=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            roi_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.roi_transforms = img_transforms
        self.file_path = file_path

        with h5py.File(self.file_path, "r") as f:
            dset = f['imgs']
            self.length = len(dset)

        self.summary()

    def __len__(self):
        return self.length

    def summary(self):
        with h5py.File(self.file_path, "r") as hdf5_file:
            dset = hdf5_file['imgs']
            for name, value in dset.attrs.items():
                print(name, value)

        print('transformations:', self.roi_transforms)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as hdf5_file:
            img = hdf5_file['imgs'][idx]
            coord = hdf5_file['coords'][idx]

        img = Image.fromarray(img)
        img = self.roi_transforms(img)
        return {'img': img, 'coord': coord}


class Whole_Slide_Bag_FP(Dataset):


    def __init__(self,
                 file_path,
                 wsi,
                 img_transforms=None,
                 patch_level_low_res=None,
                 patch_level_high_res=None,
                 extract_high_res_features=None,
                 dataset_name=None):
        """
        Args:
            file_path (string): Path to the .h5 file containing patched data.
            img_transforms (callable, optional): Optional transform to be applied on a sample
        """
        self.wsi = wsi
        self.roi_transforms = img_transforms

        self.file_path = file_path
        self.dataset_name = dataset_name

        # Adding more parameters for initialization
        self.patch_level_low_res = patch_level_low_res
        self.patch_level_high_res = patch_level_high_res
        self.extract_high_res_features = extract_high_res_features

        with h5py.File(self.file_path, "r") as f:
            dset = f['coords']
            self.patch_level = f['coords'].attrs['patch_level']
            self.patch_size = f['coords'].attrs['patch_size']
            self.length = len(dset)

    def __len__(self):
        return self.length

    def summary(self):
        hdf5_file = h5py.File(self.file_path, "r")
        dset = hdf5_file['coords']
        for name, value in dset.attrs.items():
            print(name, value)

        print('\nfeature extraction settings')
        print('transformations: ', self.roi_transforms)

    def __getitem__(self, idx):

        with h5py.File(self.file_path, 'r') as hdf5_file:
            coord = hdf5_file['coords'][idx]

        if self.extract_high_res_features:

            if self.dataset_name == 'camelyon16':

                high_resolution_imgs = []
                high_resolution_coords = []

                x_start = coord[0]
                y_start = coord[1]

                scale_factor = 2 ** (self.patch_level_low_res - self.patch_level_high_res)
                step_size = 256
                cnt = 0

                start_time_hr = time.time()

                # Step 1 - At High Resolution
                for x_step_idx in range(scale_factor):
                    for y_step_idx in range(scale_factor):
                        x_curr = x_start + x_step_idx  * 2  *  step_size
                        y_curr = y_start + y_step_idx  * 2  *  step_size
                        patch = self.wsi.read_region((x_curr, y_curr), self.patch_level_high_res,
                                                     (self.patch_size, self.patch_size)).convert("RGB")
                        cnt += 1
                        patch = self.roi_transforms(patch)
                        high_resolution_imgs.append(patch)
                        high_resolution_coords.append(np.array([x_curr, y_curr]))

                # Combining all logic to get the output
                high_resolution_imgs = torch.stack(high_resolution_imgs)
                high_resolution_coords = np.stack(high_resolution_coords)
                end_time_hr = time.time()

                # Step 2 - At Low Resolution
                start_time_lr = time.time()
                patch = self.wsi.read_region((x_start, y_start), self.patch_level_low_res,
                                             (self.patch_size, self.patch_size)).convert("RGB")
                low_resolution_imgs = self.roi_transforms(patch)
                low_resolution_coords = coord
                end_time_lr = time.time()

                return {
                    'hr_img': high_resolution_imgs,  # Op : K, 3, 224, 224
                    'hr_coords': high_resolution_coords,  # Op : K, 2
                    'hr_time': end_time_hr - start_time_hr,  # Op : 1
                    'lr_img': low_resolution_imgs,  # Op : 3, 224, 224
                    'lr_coords': low_resolution_coords,  # Op : 2
                    'lr_time': end_time_lr - start_time_lr  # Op : 1
                }

            elif self.dataset_name == 'tcga':

                high_resolution_imgs = []
                high_resolution_coords = []

                x_start = coord[0]
                y_start = coord[1]

                scale_factor = 4 ** (self.patch_level_low_res - self.patch_level_high_res)
                step_size = 256
                cnt = 0

                start_time_hr = time.time()

                # Step 1 - At High Resolution
                for x_step_idx in range(scale_factor):
                    for y_step_idx in range(scale_factor):
                        x_curr = x_start + x_step_idx * 4 * step_size
                        y_curr = y_start + y_step_idx * 4 * step_size
                        patch = self.wsi.read_region((x_curr, y_curr), self.patch_level_high_res,
                                                     (self.patch_size, self.patch_size)).convert("RGB")
                        cnt += 1
                        patch = self.roi_transforms(patch)
                        high_resolution_imgs.append(patch)
                        high_resolution_coords.append(np.array([x_curr, y_curr]))

                # Combining all logic to get the output
                high_resolution_imgs = torch.stack(high_resolution_imgs)
                high_resolution_coords = np.stack(high_resolution_coords)
                end_time_hr = time.time()

                # Step 2 - At Low Resolution
                start_time_lr = time.time()
                patch = self.wsi.read_region((x_start, y_start), self.patch_level_low_res,
                                             (self.patch_size, self.patch_size)).convert("RGB")
                low_resolution_imgs = self.roi_transforms(patch)
                low_resolution_coords = coord
                end_time_lr = time.time()

                return {
                    'hr_img': high_resolution_imgs,  # Op : K, 3, 224, 224
                    'hr_coords': high_resolution_coords,  # Op : K, 2
                    'hr_time': end_time_hr - start_time_hr,  # Op : 1
                    'lr_img': low_resolution_imgs,  # Op : 3, 224, 224
                    'lr_coords': low_resolution_coords,  # Op : 2
                    'lr_time': end_time_lr - start_time_lr  # Op : 1
                }

        else:

            start_time = time.time()
            img = self.wsi.read_region(coord, self.patch_level_low_res, (self.patch_size, self.patch_size)).convert('RGB')
            img = self.roi_transforms(img)
            end_time = time.time()
            return {'img': img, 'coord': coord, 'time': end_time - start_time}

    def get_high_res_img(self, coord):

        if self.dataset_name == 'camelyon16':

            # Second Logic
            high_resolution_imgs = []
            high_resolution_coords = []
            high_resolution_time = []

            x_start = coord[0]
            y_start = coord[1]

            scale_factor = 2 ** (self.patch_level_low_res - self.patch_level_high_res)
            step_size = 256
            cnt = 0

            start_time_hr = time.time()

            # Step 1 - At High Resolution
            for x_step_idx in range(scale_factor):
                for y_step_idx in range(scale_factor):
                    x_curr = x_start + x_step_idx * 2 * step_size
                    y_curr = y_start + y_step_idx * 2 * step_size
                    patch = self.wsi.read_region((x_curr, y_curr), self.patch_level_high_res,
                                                 (self.patch_size, self.patch_size)).convert("RGB")
                    cnt += 1
                    patch = self.roi_transforms(patch)
                    high_resolution_imgs.append(patch)
                    high_resolution_coords.append(np.array([x_curr, y_curr]))

            # Combining all logic to get the output
            high_resolution_imgs = torch.stack(high_resolution_imgs)
            high_resolution_coords = np.stack(high_resolution_coords)
            end_time_hr = time.time()

            return {
                'hr_img': high_resolution_imgs,  # Op : K, 3, 224, 224
                'hr_coords': high_resolution_coords,  # Op : K, 2
                'hr_time': end_time_hr - start_time_hr,  # Op : 1
            }

        elif self.dataset_name == 'tcga':

            print("TCGA")

class Dataset_All_Bags(Dataset):

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df['slide_id'][idx]