"""
This script is responsible for loading the training model and updating patch-level features based on similarity.

Core Logic:
-----------
The primary objective is to propagate feature information from selected patches to their similar counterparts,
based on a cosine similarity threshold.

Key Variables:
--------------
- v_at : Intermediate feature aggregator for the selected (high-attention) patch.
- z_at : Corresponding patch feature at a lower resolution.
- high_cosine_indices : Indices of patches at low resolution that are highly similar (based on cosine similarity)
                        and should inherit or update their feature representation using the predicted feature
                        aggregator (v_at).


Command -
CAMELYON16
python step5_tsu_training.py --config config/camelyon_tsu_config.yml --seed 4 --arch hga --log_dir LOG_DIR

TCGA
python step5_tsu_training.py --config config/tcga_tsu_config.yml --seed 4 --arch hga --log_dir LOG_DIR

"""

import argparse
import os
import random
from pprint import pprint

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.datasets import build_HDF5_feat_dataset_2
from modules.fglobal_mlp import FGlobal
from utils.gpu_utils import check_gpu_availability
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from utils.utils import save_model, Struct, set_seed


def get_arguments():
    parser = argparse.ArgumentParser('TSU Training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/camelyon_tsu_config.yml', help='settings of dataset in yaml format')
    parser.add_argument("--seed", type=int, default=4, help="set the random seed to ensure reproducibility")
    parser.add_argument("--arch", type=str, default='hga', choices=['ga', 'hga'], help="choice of architecture type e.e. hga or ga")
    parser.add_argument("--exp_name", type=str, default="DEBUG", help="Experiment name")
    parser.add_argument('--logs', default='enabled', choices=['enabled', 'disabled'], help='tensorboard logging')
    parser.add_argument("--log_dir", type=str, default= '/mnt/hdd/naman/naman/results_2025/results/camleyon16/extra', help="Path to logs folder")

    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")
    
    return args

def set_seed(seed=42):
    random.seed(seed)  # Python RNG
    np.random.seed(seed)  # NumPy RNG
    torch.manual_seed(seed)  # PyTorch CPU
    torch.cuda.manual_seed(seed)  # PyTorch GPU
    torch.cuda.manual_seed_all(seed)  # Multi-GPU (if applicable)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = get_arguments()

    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    conf.in_features = 384
    ff_dim = 512

    conf.writer = SummaryWriter(log_dir=os.path.join(conf.log_dir, "logs", conf.exp_name))

    hyparams = {
        'lr': conf.lr,
        'seed': conf.seed,
        'reg constant': conf.reg,
        'cosine_threshold': conf.cosine_threshold,
    }
    
    hyparams_text = "\n".join([f"**{key}**: {value}" for key, value in hyparams.items()])
    conf.writer.add_text("Hyperparameters", hyparams_text)
    
    ckpt_dir = os.path.join(conf.log_dir, "models", conf.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Used config:");
    pprint(vars(conf));

    set_seed(conf.seed)

    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    model = FGlobal().to(conf.device)
    criterion = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.reg)

    best_state = {'epoch': -1, 'val_mse_loss': 1e9, 'test_mse_loss': 1e9}
    
    for epoch in range(conf.train_epoch):
        
        train_one_epoch(model, criterion, train_loader, optimizer, conf.device, epoch, conf)
        val_mse_loss = evaluate(model, criterion, val_loader, conf.device, conf, 'Val', epoch)
        test_mse_loss = evaluate(model, criterion, test_loader, conf.device, conf, 'Test', epoch)

        if val_mse_loss < best_state['val_mse_loss']:
            best_state['epoch'] = epoch
            best_state['val_mse_loss'] = val_mse_loss
            best_state['test_mse_loss'] = test_mse_loss
            save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch, save_path=os.path.join(ckpt_dir, 'checkpoint-best.pt'))
        
        print('\n')

    save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch, save_path=os.path.join(ckpt_dir, 'checkpoint-last.pt'))
    
    print("Results on best epoch:")
    print(best_state)
    best_state_text = "\n".join([f"{key}: {value}" for key, value in best_state.items()])
    conf.writer.add_text("Best Model State", best_state_text, global_step=best_state["epoch"])
    conf.writer.close()


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    epoch_loss = 0

    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        lr_features = data['lr'][0].to(device, dtype=torch.float32)
        
        # Calculate and set new learning rate
        adjust_learning_rate(optimizer, epoch + data_it / len(data_loader), conf)
        N = lr_features.shape[0]
        choices = list(range(N))
        a_t = np.random.choice(choices)
        v_at = hr_features[a_t].to(device)
        z_at = lr_features[a_t].to(device)

        # Similar patches 
        cosine_vector = torch.cosine_similarity(lr_features, lr_features[a_t])
        high_cosine_indices = (torch.abs(cosine_vector) >= conf.cosine_threshold).nonzero()[:, 0]
        input_f_global = torch.cat((v_at.repeat(len(high_cosine_indices), 1),
                                    z_at.repeat(len(high_cosine_indices), 1),
                                    lr_features[high_cosine_indices]), dim=1)
        output = model(input_f_global)

        loss = criterion(output, hr_features[high_cosine_indices])
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        epoch_loss += loss.item()
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(mse_loss=loss.item())

        if conf.logs != 'disabled':
            conf.writer.add_scalar("Step_Loss/mse loss", loss.item(), data_it + (epoch * len(data_loader)))
            
    if conf.logs != 'disabled':
        conf.writer.add_scalar("Epoch_Loss/mse loss", epoch_loss / len(data_loader), epoch)


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, conf, header, epoch):
    
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    mse_loss = 0
    for data in metric_logger.log_every(data_loader, 100, header):
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        lr_features = data['lr'][0].to(device, dtype=torch.float32)
        
        
        # Load basic details
        N = lr_features.shape[0]
        choices = list(range(N))
        a_t = np.random.choice(choices)
        v_at = hr_features[a_t].to(device)
        z_at = lr_features[a_t].to(device)

        # Similar patches details
        cosine_vector = torch.cosine_similarity(lr_features, lr_features[a_t])
        high_cosine_indices = (torch.abs(cosine_vector) >= conf.cosine_threshold).nonzero()[:, 0]

        input_f_global = torch.cat((v_at.repeat(len(high_cosine_indices), 1), z_at.repeat(len(high_cosine_indices), 1), lr_features[high_cosine_indices]), dim=1)
        output = model(input_f_global)
        loss = criterion(output, hr_features[high_cosine_indices])

        mse_loss += loss.item()
        metric_logger.update(loss=loss.item())
        
    mse_loss /= len(data_loader)
    
    if conf.logs != 'disabled':
        conf.writer.add_scalar(f"{header}/loss", metric_logger.loss.global_avg, epoch)
        
    print(f"{header}: loss={mse_loss}")
    
    return mse_loss


if __name__ == '__main__':
    main()