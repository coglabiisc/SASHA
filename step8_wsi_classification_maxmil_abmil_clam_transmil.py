'''
Commands -
python step8_wsi_classification_maxmil_abmil_clam_transmil.py --arch maxmil --config config/camelyon_config.yml --log_dir LOG_DIR
--arch abmil
--arch transmil

'''

import argparse
import os
import sys
from pprint import pprint

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from architecture.bmil import probabilistic_MIL_Bayes_spvis
from architecture.clam import CLAM_SB, CLAM_MB
from architecture.dsmil import MILNet, FCLayer, BClassifier
from architecture.ilra import ILRA
from architecture.transMIL import TransMIL
from architecture.transformer import MHA, ABMIL
from datasets.datasets import build_HDF5_feat_dataset
from engine import train_one_epoch, evaluate
from modules import mean_max
from utils.gpu_utils import check_gpu_availability
from utils.utils import save_model, Struct, set_seed


def get_arguments():
    parser = argparse.ArgumentParser('Patch classification training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/camelyon_config.yml', help='settings of dataset in yaml format')
    parser.add_argument("--seed", type=int, default=4, help="set the random seed to ensure reproducibility")
    parser.add_argument('--logs_mode', default='disabled', choices=['offline', 'online', 'disabled'], help='the model of wandb')
    parser.add_argument("--w_loss", type=float, default=1.0, help="number of query token")
    parser.add_argument("--arch", type=str, default='abmil', choices=['transmil', 'clam_sb', 'clam_mb', 'abmil', 'ilra',
                                                 'mha', 'dsmil', 'bmil_spvis', 'meanmil', 'maxmil', 'acmil'], help="number of query token")
    parser.add_argument('--pretrain', default='medical_ssl', choices=['natural_supervsied', 'medical_ssl', 'plip', 'path-clip-B-AAAI'
                                                            'path-clip-B', 'path-clip-L-336', 'openai-clip-B', 'openai-clip-L-336', 'quilt-net', 
                                                            'biomedclip', 'path-clip-L-768', 'UNI', 'GigaPath'],
                                                            help='settings of Tip-Adapter in yaml format')
    parser.add_argument("--exp_name", type=str, default="DEBUG", help="Experiment name")
    parser.add_argument("--log_dir", type=str, default= None, help="Path to logs folder")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    
    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")
    
    return args

def main():
    # Load config file
    args = get_arguments()

    # get config
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    # Initialized the config directory
    conf.writer = SummaryWriter(log_dir=os.path.join(conf.log_dir, "logs"))

    # Initializing model directory to save weights
    ckpt_dir = os.path.join(conf.log_dir, "models", conf.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    print("Used config:");
    pprint(vars(conf));

    # Prepare dataset
    set_seed(args.seed)

    # define datasets and dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset(conf.data_dir, conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # Define model architecture
    
    if conf.arch == 'maxmil':
        net = mean_max.MaxMIL(conf).to(conf.device)
    elif conf.arch == 'abmil':
        net = ABMIL(conf)
    elif conf.arch == 'clam_sb':
        net = CLAM_SB(conf).to(conf.device)
    elif conf.arch == 'transmil':
        net = TransMIL(conf)
        
    elif conf.arch == 'mha':
        net = MHA(conf)
    elif conf.arch == 'clam_mb':
        net = CLAM_MB(conf).to(conf.device)
    elif conf.arch == 'dsmil':
        i_classifier = FCLayer(conf.D_feat, conf.n_class)
        b_classifier = BClassifier(conf, nonlinear=False)
        net = MILNet(i_classifier, b_classifier)
    elif conf.arch == 'bmil_spvis':
        net = probabilistic_MIL_Bayes_spvis(conf)
        net.relocate()
    elif conf.arch == 'meanmil':
        net = mean_max.MeanMIL(conf).to(conf.device)
    elif conf.arch == 'ilra':
        net = ILRA(feat_dim=conf.D_feat, n_classes=conf.n_class, ln=True)
    else:
        print("architecture %s is not exist."%conf.arch)
        sys.exit(1)

    # define loss function and optimizer
    net.to(conf.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=conf.lr, weight_decay=conf.wd)

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    
    for epoch in range(conf.train_epoch):

        train_one_epoch(net, criterion, train_loader, optimizer, conf.device, epoch, conf)
        val_auc, val_acc, val_f1, val_loss = evaluate(net, criterion, val_loader, conf.device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss = evaluate(net, criterion, test_loader, conf.device, conf, 'Test')

        if conf.logs_mode != 'disabled':

            conf.writer.add_scalar('test/test_acc1', test_acc, epoch)
            conf.writer.add_scalar('test/test_auc', test_auc, epoch)
            conf.writer.add_scalar('test/test_f1', test_f1, epoch)
            conf.writer.add_scalar('test/test_loss', test_loss, epoch)
            conf.writer.add_scalar('val/val_acc1', val_acc, epoch)
            conf.writer.add_scalar('val/val_auc', val_auc, epoch)
            conf.writer.add_scalar('val/val_f1', val_f1, epoch)
            conf.writer.add_scalar('val/val_loss', val_loss, epoch)

        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            save_model(conf=conf, model=net, optimizer=optimizer, epoch=epoch, save_path=os.path.join(ckpt_dir, 'checkpoint-best.pth'))

        print('\n')

    save_model(conf=conf, model=net, optimizer=optimizer, epoch=epoch, save_path=os.path.join(ckpt_dir, 'checkpoint-last.pth'))
    print("Results on best epoch:")
    print(best_state)


if __name__ == '__main__':
    main()
