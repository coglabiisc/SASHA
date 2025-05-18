"""
This step handles feature extraction.

There are two modes of operation:

1. If `extract_high_res_features` is set to `True`, the feature extractor will generate features for both high-resolution and low-resolution patches.

2. If `extract_high_res_features` is set to `False`, features will be extracted only for low-resolution patches.

Adjust the `extract_high_res_features` flag based on the desired resolution level for feature extraction.

CAMELYON16 -
For extraction both high resolution and low resolution together
python step2_extract_features.py --dataset_name camelyon16 --data_h5_dir SAVE_DIR_PATH_FROM_PATCH_CREATION --data_slide_dir WSI_IMAGES_DIR --slide_ext .tif --csv_path dataset_csv/camelyon16/camelyon16.csv --feat_dir FEAT_DIR_TO_SAVE --batch_size 32 --extract_high_res_features True --patch_level_low_res 3 --patch_level_high_res 1

For only low resolution
python step2_extract_features.py --dataset_name camelyon16 --data_h5_dir SAVE_DIR_PATH_FROM_PATCH_CREATION --data_slide_dir WSI_IMAGES_DIR --slide_ext .tif --csv_path dataset_csv/camelyon16/camelyon16.csv --feat_dir FEAT_DIR_TO_SAVE --batch_size 512 --extract_high_res_features False --patch_level_low_res 3 --patch_level_high_res 1


TCGA -
For extraction both high and low resolution together
python step2_extract_features.py --dataset_name tcga --data_h5_dir SAVE_DIR_PATH_FROM_PATCH_CREATION --data_slide_dir WSI_IMAGES_DIR --slide_ext .tif --csv_path dataset_csv/tcga/tcga.csv --feat_dir FEAT_DIR_TO_SAVE --batch_size 32 --extract_high_res_features True --patch_level_low_res 2 --patch_level_high_res 1

For only low resolution
python step2_extract_features.py --dataset_name tcga --data_h5_dir SAVE_DIR_PATH_FROM_PATCH_CREATION --data_slide_dir WSI_IMAGES_DIR --slide_ext .tif --csv_path dataset_csv/tcga/tcga.csv --feat_dir FEAT_DIR_TO_SAVE --batch_size 512 --extract_high_res_features False --patch_level_low_res 2 --patch_level_high_res 1

"""

import argparse
import os
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from models_features_extraction import get_encoder
from step1_create_patches import create_time_df


def plot_tensor_images(images, num_rows=1, num_cols=1, file_name=None, figsize=(10, 10)):
    """
    Plots a batch of images in a tensor using subplots.

    Args:
        images (torch.Tensor): Tensor of shape (B, 3, H, W).
        num_rows (int): Number of rows in the subplot.
        num_cols (int): Number of columns in the subplot.
        figsize (tuple): Size of the figure.
    """
    # Ensure the tensor is on the CPU and convert it to numpy
    images = images.cpu().numpy()

    # Create a figure
    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.flatten()  # Flatten in case of a grid of subplots

    for idx, ax in enumerate(axs):
        if idx < images.shape[0]:
            # Convert the image from (3, H, W) to (H, W, 3) for plotting
            img = images[idx].transpose(1, 2, 0)
            # Clip values to the range [0, 1] if necessary
            img = img.clip(0, 1)
            ax.imshow(img)
            ax.axis('off')
        else:
            ax.axis('off')  # Turn off axes for empty subplots

    plt.tight_layout()
    plt.savefig(file_name)

def plot_single_image(image, file_name=None, figsize=(6, 6)):
    """
    Plots a single image tensor of shape (3, H, W).

    Args:
        image (torch.Tensor): Tensor of shape (3, H, W).
        figsize (tuple): Size of the figure.
    """
    # Ensure the tensor is on the CPU and convert it to numpy
    image = image.cpu().numpy()

    # Convert the image from (3, H, W) to (H, W, 3) for plotting
    img = image.transpose(1, 2, 0)

    # Clip values to the range [0, 1] if necessary
    img = img.clip(0, 1)

    # Plot the image
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis('off')  # Turn off axis labels
    plt.savefig(file_name)


def compute_w_loader(loader, model, verbose=0, extract_high_res_features = True, device = 'cpu' ):
    """
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""

    if verbose > 0:
        print(f'processing a total of {len(loader)} batches'.format(len(loader)))

    mode = 'w'
    hr_features_list = []
    lr_features_list = []
    hr_coords_list = []
    lr_coords_list = []
    hr_total_time_list = []
    lr_total_time_list = []

    features_list = []
    coords_list = []
    total_time_list = []

    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():

            if extract_high_res_features:
                # Need to write Logic for this part

                hr_img = data['hr_img']
                hr_coords = data['hr_coords']
                hr_time = data['hr_time']
                lr_img = data['lr_img']
                lr_coords = data['lr_coords']
                lr_time = data['lr_time']

                # Step 1 -  Obtaining H.R. Image Features
                start_time_hr = time.time()
                mapping_factor = hr_img.shape[1]
                hr_batch = hr_img
                hr_batch = torch.reshape(hr_batch, (-1, hr_batch.shape[-3], hr_batch.shape[-2], hr_batch.shape[-1]))
                hr_batch = hr_batch.to(device, non_blocking=True)
                hr_features = model(hr_batch)
                hr_features = torch.reshape(hr_features, (-1, mapping_factor, hr_features.shape[-1]))
                hr_features = hr_features.cpu()  # Op : (k, 1024)

                hr_features_list.append(hr_features)
                hr_coords_list.append(hr_coords)
                end_time_hr = time.time()

                # Adding them to list
                total_time_hr = hr_time.sum().item() + end_time_hr - start_time_hr
                hr_total_time_list.append(total_time_hr)

                # Step 2 - Obtaining L.R Image Features
                start_time_lr = time.time()
                lr_batch = lr_img
                lr_batch = torch.reshape(lr_batch, (-1, lr_batch.shape[-3], lr_batch.shape[-2], lr_batch.shape[-1]))
                lr_batch = lr_batch.to(device, non_blocking=True)
                lr_features = model(lr_batch)
                lr_features = lr_features.cpu()  # Op : (1, 1024)


                lr_features_list.append(lr_features)
                lr_coords_list.append(lr_coords)
                end_time_lr = time.time()

                # Adding them to list
                total_time_lr = lr_time.sum().item() + end_time_lr - start_time_lr
                lr_total_time_list.append(total_time_lr)

            else:
                state_time_feat = time.time()

                batch = data['img']  # Op : (B, 3, 224, 224)
                coords = data['coord'].numpy().astype(np.int32)  # Op : (B, 2)
                time_1 = data['time'] # Op : (B, 1)
                batch = torch.reshape(batch,(-1, batch.shape[-3], batch.shape[-2], batch.shape[-1]))  # Op : ( B, 3, 224, 224)
                coords = np.reshape(coords, (-1, coords.shape[-1]))  # Op : (B, 2)
                batch = batch.to(device, non_blocking=True)

                features = model(batch)  # Ip : (B, 3, 224, 224)
                features = features.cpu()
                features_list.append(features)
                coords_list.append(coords)

                end_time_feat = time.time()

                total_time_feat =  time_1.sum().item() + end_time_feat - state_time_feat
                total_time_list.append(total_time_feat)

    if extract_high_res_features:
        hr_features = torch.cat(hr_features_list, dim=0).numpy()
        lr_features = torch.cat(lr_features_list, dim=0).numpy()
        hr_coords = np.concatenate(hr_coords_list, axis=0)
        lr_coords = np.concatenate(lr_coords_list, axis=0)
        hr_total_time = sum(hr_total_time_list)
        lr_total_time = sum(lr_total_time_list)

        return hr_features, hr_coords, hr_total_time, lr_features, lr_coords, lr_total_time

    else :

        features = torch.cat(features_list, dim=0).numpy()
        coords = np.concatenate(coords_list, axis=0)
        total_time = sum(total_time_list)

        return features, coords, total_time

def update_arguments(args) :

    if args.time_csv is None :
        args.time_csv = os.path.join('outputs/time', 'time_feat.csv')

    # Initializing the column name for storing time
    if args.lr_time_col_name is None :
        args.lr_time_col_name = f"fe_b_{args.batch_size}_lr_level_{args.patch_level_low_res}"

    if args.hr_time_col_name is None :
        args.hr_time_col_name = f"fe_b_{args.batch_size}_hr_level_{args.patch_level_high_res}"

    return args

def get_arguments() :
    parser = argparse.ArgumentParser(description='Feature Extraction')

    # Primary Arguments 
    parser.add_argument('--dataset_name', type=str, default='tcga', choices=['camelyon16', 'tcga'])
    parser.add_argument('--data_h5_dir', type=str, default="/media/internal_8T/naman/test/tcga")
    parser.add_argument('--data_slide_dir', type=str, default='/media/internal_8T/naman/tcga_sample_files')
    parser.add_argument('--slide_ext', type=str, default=".svs",help="we have two options *.tif, *.svs, or any other compatible can work")
    parser.add_argument('--csv_path', type=str, default="dataset_csv/tcga/tcga.csv")
    parser.add_argument('--feat_dir', type=str, default="/media/internal_8T/naman/test/tcga/features")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--extract_high_res_features', type=bool, default=False, help="To create a mapping from high resolution to low resolution")
    parser.add_argument('--patch_level_low_res', type=int, default=2)  # Low  represents the magnified level [ Just Make sure that patch level should match from create patches ]
    parser.add_argument('--patch_level_high_res', type=int, default=1)  # High represents the scanning level
    parser.add_argument('--time_csv', type=str, default=None, help='store the features time per slide')
    parser.add_argument('--lr_time_col_name', type=str, default=None, help='column name for time_csv_col_name')
    parser.add_argument('--hr_time_col_name', type=str, default=None, help='column name for time_csv_col_name')


    # Secondary Arguments ---> These arguments are default used for feature extraction 
    parser.add_argument('--model_name', type=str, default='ViT-S/16', choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'ViT-S/16', 'Resnet50'])
    parser.add_argument('--pretrain', type=str, default='medical_ssl', choices=['medical_ssl'])
    parser.add_argument('--no_auto_skip', default=True, action='store_true')
    parser.add_argument('--target_patch_size', type=int, default=224)
    
    return parser
    

if __name__ == '__main__':

    # Initialize cuda device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    parser = get_arguments()
    args = parser.parse_args()
    args = update_arguments(args= args)

    print('Initializing dataset')

    print('Step 1 : Loading CSV Path')
    csv_path = args.csv_path
    if csv_path is None:
        raise NotImplementedError

    # Loading the slide name from csv file
    bags_dataset = Dataset_All_Bags(csv_path)
    df = bags_dataset.df.set_index('slide_id')
    print(f"Total slides : {len(bags_dataset)}")

    # Creating the Necessary Directories
    os.makedirs(args.feat_dir, exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'lr', 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'lr', 'h5_files'), exist_ok=True)
    h5_path_lr = os.path.join(args.feat_dir, 'lr', 'h5_files', 'patch_feats_pretrain_%s.h5' % args.pretrain)
    h5file_lr = h5py.File(h5_path_lr, "w")
    
    os.makedirs(os.path.join(args.feat_dir, 'hr', 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(args.feat_dir, 'hr', 'h5_files'), exist_ok=True)
    h5_path_hr = os.path.join(args.feat_dir, 'hr', 'h5_files', 'patch_feats_pretrain_%s.h5' % args.pretrain)
    h5file_hr = h5py.File(h5_path_hr, "w")
    
    # Loading the pretrained encoder ----> For now we are working with resnet-50 architecture
    model, img_transforms = get_encoder(args.model_name, pretrain= args.pretrain, target_img_size=args.target_patch_size)
    model.eval()
    model = model.to(device)

    loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}

    # Creating dataframe to store the time for feature extraction of each slide 
    time_df = create_time_df(csv_file_path= args.time_csv, column_name_ls = ['slide_name', args.lr_time_col_name, args.hr_time_col_name] )
    
    for bag_candidate_idx in tqdm(range(len(bags_dataset))) :

        # Step 1 ----> Loading the h5 file path in high resolution and corresponding file path

        slide_id_with_ext = bags_dataset[bag_candidate_idx]
        slide_id = slide_id_with_ext.split(args.slide_ext)[0]
        bag_name = slide_id + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)
        if not os.path.exists(h5_file_path):
            continue
            
        slide_file_path = os.path.join(args.data_slide_dir, slide_id + args.slide_ext)
        if os.path.exists(slide_file_path):
            print("Slide exists")
        else:
            raise FileNotFoundError(f"None of the paths exist for slide_id: {slide_id}")

        print('\nprogress: {}/{}'.format(bag_candidate_idx, len(bags_dataset)))
        print(f"Slide name : {slide_id}")

        if not args.no_auto_skip:
            print('skipped {}'.format(slide_id))
            continue

        # Step 3 - Initializing the result path
        output_path = os.path.join(args.feat_dir, 'h5_files', bag_name)
        time_start = time.time()
        wsi = openslide.open_slide(slide_file_path)

        # Step 4 - Main Function to read the file **** IMP *** Here focus on __getitem__ function is important
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path,
                                     wsi=wsi,
                                     img_transforms=img_transforms,
                                     extract_high_res_features=args.extract_high_res_features,
                                     patch_level_low_res=args.patch_level_low_res,
                                     patch_level_high_res=args.patch_level_high_res,
                                     dataset_name=args.dataset_name)

        loader = DataLoader(dataset=dataset, batch_size=args.batch_size, **loader_kwargs)

        if args.extract_high_res_features:

            hr_features, hr_coords, hr_total_time, lr_features, lr_coords, lr_total_time = compute_w_loader(
                loader=loader, model=model, verbose=1)

            time_elapsed = time.time() - time_start
            print('\ncomputing features for {} took {} s'.format(slide_id, time_elapsed))

            # Storing time details -->
            # Check if slide_id exists
            if slide_id in time_df['slide_name'].values:
                time_df.loc[time_df['slide_name'] == slide_id, args.lr_time_col_name] = lr_total_time
                time_df.loc[time_df['slide_name'] == slide_id, args.hr_time_col_name] = hr_total_time
            else:
                # Create new row
                new_row = {col: "" for col in time_df.columns}
                new_row["slide_name"] = slide_id
                new_row[args.lr_time_col_name] = lr_total_time
                new_row[args.hr_time_col_name] = hr_total_time
                time_df = pd.concat([time_df, pd.DataFrame([new_row])], ignore_index=True)

        else:

            features, coords, total_time = compute_w_loader(loader=loader, model=model, verbose=1)

            # Storing time details -->
            # Check if slide_id exists
            if slide_id in time_df['slide_name'].values:
                time_df.loc[time_df['slide_name'] == slide_id, args.lr_time_col_name] = total_time
            else:
                # Create new row
                new_row = {col: "" for col in time_df.columns}
                new_row["slide_name"] = slide_id
                new_row[args.lr_time_col_name] = total_time
                time_df = pd.concat([time_df, pd.DataFrame([new_row])], ignore_index=True)

        # Converting df to csv  ---> Save back
        time_df.to_csv(args.time_csv, index=False)


        if args.extract_high_res_features :

            # Storing features
            slide_grp_hr = h5file_hr.create_group(slide_id)
            slide_grp_hr.create_dataset('feat', data=hr_features.astype(np.float16))
            slide_grp_hr.create_dataset('coords', data=hr_coords)
            slide_grp_hr.attrs['label'] = df.loc[slide_id_with_ext]['label']

            slide_grp_lr = h5file_lr.create_group(slide_id)
            slide_grp_lr.create_dataset('feat', data=lr_features.astype(np.float16))
            slide_grp_lr.create_dataset('coords', data=lr_coords)
            slide_grp_lr.attrs['label'] = df.loc[slide_id_with_ext]['label']

            torch.save(torch.from_numpy(hr_features), os.path.join(args.feat_dir, 'hr', 'pt_files', slide_id + '.pt'))
            torch.save(torch.from_numpy(lr_features), os.path.join(args.feat_dir, 'lr', 'pt_files', slide_id + '.pt'))

        else :

            # Storing features
            slide_grp = h5file_lr.create_group(slide_id)
            slide_grp.create_dataset('feat', data=features.astype(np.float16))
            slide_grp.create_dataset('coords', data=coords)
            slide_grp.attrs['label'] = df.loc[slide_id_with_ext]['label']

            torch.save(torch.from_numpy(features), os.path.join(args.feat_dir, 'lr', 'pt_files', slide_id + '.pt'))
