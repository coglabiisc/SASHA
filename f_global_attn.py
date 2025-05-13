"""
CUDA_VISIBLE_DEVICES=2 python f_global_attn.py --seed 4 --logs enabled --log_dir /mnt/hdd/tarung/project/test/rlogist/ACMIL/tarun_fglobal --config config/fglobal_config.yml --exp_name "fglobal_1"
"""

from csv import writer
from pyexpat import features
import sys
import os

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from sympy import false
import yaml
from pprint import pprint
from types import SimpleNamespace
import h5py
import numpy as np

import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader

from utils.utils import save_model, Struct, set_seed
from datasets.datasets import build_HDF5_feat_dataset_2
from architecture.transformer import ACMIL_GA
from architecture.transformer import ACMIL_MHA
from architecture.transformer import HACMIL_GA
from architecture.fglobal import f_global
import torch.nn.functional as F

from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from timm.utils import accuracy
import torchmetrics
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_arguments():
    parser = argparse.ArgumentParser('flobal training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/fglobal.yml',
                        help='settings of dataset in yaml format')
    parser.add_argument(
        "--seed", type=int, default=1, help="set the random seed to ensure reproducibility"
    )
    parser.add_argument('--logs', default='disabled', choices=['enabled', 'disabled'],
                        help='tensorboard logging')
    parser.add_argument(
        "--log_dir", type=str, default=f"/mnt/hdd/tarung/project/test/rlogist/ACMIL/tarun_logs",
        help="Path to logs folder"
    )
    parser.add_argument(
        "--exp_name", type=str, default="High_Res_Features", help="Experiment name"
    )
    args = parser.parse_args()
    return args



@torch.no_grad()
def load_model(ckpt_path):
    dict = torch.load(ckpt_path, map_location=device)
    curr_epoch = dict['epoch']
    config = dict['config']
    model_dict = dict['model']
    optimizer_dict = dict['optimizer']
    return model_dict, optimizer_dict, config, curr_epoch


def main():
    args = get_arguments()

    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)
    
    if conf.pretrain == "medical_ssl" and conf.backbone == "ViT-S/16":
        conf.in_features = 384
        ff_dim = 512
    else:
        conf.in_features = 2048
        conf.ff_dim = 4096
        
    conf.writer = SummaryWriter(log_dir=os.path.join(conf.log_dir, "logs", conf.exp_name))

    hyparams = {
        'nhead': conf.nhead,
        'nlayer': conf.nlayer,
        'lr': conf.lr,
        'seed': conf.seed,
        'reg constant': conf.reg
    }
    hyparams_text = "\n".join([f"**{key}**: {value}" for key, value in hyparams.items()])
    conf.writer.add_text("Hyperparameters", hyparams_text)
    ckpt_dir = os.path.join(conf.log_dir, "models", conf.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True) 

    print("Used config:");
    pprint(vars(conf));

    set_seed(args.seed)

    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    
    model = f_global(
        in_features=conf.in_features,
        nhead=conf.nhead,
        num_layers=conf.nlayer,
        ff_dim=conf.ff_dim,
        dropout=conf.dropout
    )
    model.to(device)
    classifier_dict, _, config, curr_epoch = load_model(ckpt_path=conf.ckpt_path)
    classifier_conf = SimpleNamespace(**config)
    if classifier_conf.arch == 'ga':
        classifier = ACMIL_GA(classifier_conf, n_token=classifier_conf.n_token, n_masked_patch=classifier_conf.n_masked_patch, mask_drop=classifier_conf.mask_drop)
    elif classifier_conf.arch == 'hga':
        classifier = HACMIL_GA(classifier_conf, n_token_1=classifier_conf.n_token_1, n_token_2=classifier_conf.n_token_2, n_masked_patch_1=classifier_conf.n_masked_patch_1, n_masked_patch_2=classifier_conf.n_masked_patch_2, mask_drop=classifier_conf.mask_drop)
    else:
        classifier = ACMIL_MHA(classifier_conf, n_token=classifier_conf.n_token, n_masked_patch=classifier_conf.n_masked_patch, mask_drop=classifier_conf.mask_drop)
    classifier.to(device)
    classifier.load_state_dict(classifier_dict)
    classifier.eval()

    criterion1 = nn.MSELoss(reduction="mean")
    criterion2 = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.reg)

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'val_mse_loss':1e9, 'test_acc':0, 'test_auc':0, 'test_f1':0, 'test_mse_loss':1e9}
    train_epoch = conf.train_epoch
    for epoch in range(train_epoch):
        train_one_epoch(model, classifier, criterion1, criterion2, train_loader, optimizer, device, epoch, conf)

        val_auc, val_acc, val_f1, val_loss, val_mse_loss = evaluate(model, classifier, criterion1, criterion2, val_loader, device, conf, 'Val', epoch)
        test_auc, test_acc, test_f1, test_loss, test_mse_loss = evaluate(model, classifier, criterion1, criterion2, test_loader, device, conf, 'Test', epoch)

        if (1-val_f1) + (1-val_auc) + val_mse_loss < (1-best_state['val_f1']) + (1-best_state['val_auc']) + best_state['val_mse_loss']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['val_mse_loss'] = val_mse_loss
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            best_state['test_mse_loss'] = test_mse_loss
            save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
                save_path=os.path.join(ckpt_dir, 'checkpoint-best.pt'))
        print('\n')

    
    save_model(conf=conf, model=model, optimizer=optimizer, epoch=epoch,
        save_path=os.path.join(ckpt_dir, 'checkpoint-last.pt'))
    print("Results on best epoch:")
    print(best_state)
    best_state_text = "\n".join([f"{key}: {value}" for key, value in best_state.items()])
    conf.writer.add_text("Best Model State", best_state_text, global_step=best_state["epoch"])
    conf.writer.close()




def train_one_epoch(model, classifier, criterion1, criterion2, data_loader, optimizer, device, epoch, conf):
    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    epoch_loss_1 = 0
    epoch_loss_2 = 0

    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
        slide_name = data['slide_name'][0]
        label = data['label'].to(device)
        # # Calculate and set new learning rate
        adjust_learning_rate(optimizer, epoch + data_it/len(data_loader), conf)
        N = state.shape[1]
        T = min(N, int(conf.frac * N))
        choices = list(range(N))
        visited_patch_id = []
        for t in range(T):
            a_t = np.random.choice(choices)
            choices.remove(a_t)
            visited_patch_id.append(a_t)
            v_at = hr_features[a_t].to(device)
            new_state = model(v_at, state.clone())
            new_state[0][visited_patch_id] = hr_features[visited_patch_id]
            slide_preds, attn = classifier.classify(new_state)
            pred = torch.softmax(slide_preds, dim=-1)
            loss1 = criterion1(new_state, hr_features.unsqueeze(0))
            loss2 = criterion2(pred, label)
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            state = new_state.detach()
        epoch_loss_1 += loss1.item()
        epoch_loss_2 += loss2.item()
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        metric_logger.update(mse_loss=loss1.item())
        metric_logger.update(ce_loss=loss2.item())
        metric_logger.update(slide_loss=loss.item())

        if conf.logs != 'disabled':
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            conf.writer.add_scalar("Step_Loss/ce loss", loss2.item(), data_it + (epoch * len(data_loader)))
            conf.writer.add_scalar("Step_Loss/mse loss", loss1.item(), data_it + (epoch * len(data_loader)))
    if conf.logs != 'disabled':
        conf.writer.add_scalar("Epoch_Loss/mse loss", epoch_loss_1/len(data_loader), epoch)
        conf.writer.add_scalar("Epoch_Loss/ce loss", epoch_loss_2/len(data_loader), epoch)
        


@torch.no_grad()
def evaluate(model, classifier, criterion1, criterion2, data_loader, device, conf, header, epoch):
    model.eval()
    y_pred = []
    y_true = []
    metric_logger = MetricLogger(delimiter="  ")
    mse_loss = 0
    for data in metric_logger.log_every(data_loader, 100, header):
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
        slide_name = data['slide_name'][0]
        label = data['label'].to(device)
        N = state.shape[1]
        T = min(N, int(conf.frac * N))
        choices = list(range(N))
        visited_patch_id = []
        for t in range(T):
            a_t = np.random.choice(choices)
            choices.remove(a_t)
            visited_patch_id.append(a_t)
            v_at = hr_features[a_t].to(device)
            new_state = model(v_at, state.clone())
            new_state[0][visited_patch_id] = hr_features[visited_patch_id]
            state = new_state.detach()
        
        slide_preds, attn = classifier.classify(new_state)
        pred = torch.softmax(slide_preds, dim=-1)
        loss1 = criterion1(new_state, hr_features.unsqueeze(0))
        loss2 = criterion2(pred, label)
        loss = loss1 + loss2
        mse_loss += loss1.item()
        acc1 = accuracy(pred, label, topk=(1,))[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=label.shape[0])
        y_pred.append(pred)
        y_true.append(label)
    mse_loss /= len(data_loader)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred_labels = torch.argmax(y_pred, dim=-1)
    AUROC_metric = torchmetrics.AUROC(task='binary').to(device)
    AUROC_metric(y_pred[:, 1], y_true)
    auroc = AUROC_metric.compute().item()
    F1_metric = torchmetrics.F1Score(task='binary').to(device)
    F1_metric(y_pred_labels, y_true)
    f1_score = F1_metric.compute().item()

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))
    if conf.logs != 'disabled':
        conf.writer.add_scalar(f"{header}/accuracy", metric_logger.acc1.global_avg, epoch)
        conf.writer.add_scalar(f"{header}/auroc", auroc, epoch)
        conf.writer.add_scalar(f"{header}/f1", f1_score, epoch)
        conf.writer.add_scalar(f"{header}/loss", metric_logger.loss.global_avg, epoch)
    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg, mse_loss


if __name__ == '__main__':
    main()