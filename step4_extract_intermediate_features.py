"""
Once the HAFED model has been trained using all high-resolution patches,
the workflow proceeds as follows:

- Input: (N × k × d)
- Intermediate output (after feature aggregation): (N × d)
- Final output: Predicted class label

The next task involves extracting these intermediate features (N × d),
which are subsequently used in the pipeline for training downstream models.

Config files
CAMELYON16 ---> config/camelyon_config.yml
TCGA ----> config/tcga_config.yml

Cmd -
CAMELYON16
python step4_extract_intermediate_features.py --config config/camelyon_config.yml --seed 4 --arch hga --ckpt_path CKPT_PATH --output_path OUTPUT_PATH

TCGA
python step4_extract_intermediate_features.py --config config/tcga_config.yml --seed 1 --arch hga --ckpt_path CKPT_PATH --output_path OUTPUT_PATH

"""

import argparse
import os

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from architecture.transformer import ACMIL_GA
from architecture.transformer import HAFED
from datasets.datasets import build_HDF5_feat_dataset
from utils.gpu_utils import check_gpu_availability
from utils.utils import MetricLogger
from utils.utils import Struct, set_seed


def get_arguments():
    parser = argparse.ArgumentParser('WSI classification training', add_help=False)

    # Primary Arguments for HAFED Feature extraction
    parser.add_argument('--config', dest='config', default='config/camelyon_config.yml', help='settings of dataset in yaml format')
    parser.add_argument("--seed", type=int, default=4, help="set the random seed to ensure reproducibility")
    parser.add_argument("--arch", type=str, default='hga', choices=['ga', 'hga'], help="choice of architecture type")
    parser.add_argument('--ckpt_path', default= None, type=str, help='Load checkpoint path for HAFED')
    parser.add_argument('--output_path', type=str, default= None, help='directory path to save the intermediate features')

    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args

@torch.no_grad()
def load_model(ckpt_path, args):
    dict = torch.load(ckpt_path, map_location=args.device)
    curr_epoch = dict['epoch']
    config = dict['config']
    model_dict = dict['model']
    optimizer_dict = dict['optimizer']
    return model_dict, optimizer_dict, config, curr_epoch


def main():

    # Load config file
    args = get_arguments()

    # get parameter details from configuration file
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)


    # Set different seed for dataset
    set_seed(args.seed)

    # define datasets and dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # Define Network
    if conf.arch == 'ga':
        model = ACMIL_GA(conf,
                         n_token=conf.n_token,
                         n_masked_patch=conf.n_masked_patch,
                         mask_drop=conf.mask_drop)

    elif conf.arch == 'hga':
        model = HAFED(conf,
                      n_token_1=conf.n_token_1,
                      n_token_2=conf.n_token_2,
                      n_masked_patch_1=conf.n_masked_patch_1,
                      n_masked_patch_2=conf.n_masked_patch_2,
                      mask_drop=conf.mask_drop)
    else:
        raise Exception(f"Enter a valid model architecture name e.g. ga, hga")

    model.to(conf.device)

    # Loading model weights
    model_dict, _, _, _ = load_model(ckpt_path= args.ckpt_path, args= args)
    model.load_state_dict(model_dict)

    # Creating h5 file
    output_path = os.path.join(args.output_path, 'patch_feats_pretrain_%s.h5'% conf.pretrain)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    h5_file = h5py.File(output_path, "w")

    # Saving intermediate weights for data loaders
    extract_features(model, train_loader, conf.device, conf, 'Train', h5_file)
    extract_features(model, val_loader, conf.device, conf, 'Val', h5_file)
    extract_features(model, test_loader, conf.device, conf, 'Test', h5_file)

    h5_file.close()


# Disable gradient calculation during extract features
@torch.no_grad()
def extract_features(net, data_loader, device, conf, header, h5_file, extract_feature=True):

    # Set the network to evaluation mode
    net.eval()

    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):

        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)
        slide_id = data['slide_name'][0]

        if conf.arch == 'hga':
            _, _, _, _, features, _ = net(image_patches, extract_feature=extract_feature)

        slide_grp = h5_file.create_group(slide_id)
        slide_grp.create_dataset('feat', data=features.cpu().numpy().astype(np.float32))
        slide_grp.attrs['label'] = labels.item()
        pt_file_path = os.path.join(conf.output_path, 'patch_feats_pretrain_%s'%conf.pretrain, 'pt_files', )
        os.makedirs(pt_file_path, exist_ok=True)
        torch.save(features.cpu().float(), os.path.join(pt_file_path, f"{slide_id}.pt"))

    return


if __name__ == '__main__':
    main()