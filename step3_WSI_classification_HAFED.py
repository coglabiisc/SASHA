"""
This script is used to train the models required to obtain the Feature Aggregator and Classifier components from the HAFED framework.

These trained models are utilized in the subsequent stage of the pipeline.

Model Architecture:
- Feature Aggregator: Input shape (k × d) → Output shape (d)
- Classifier: Input shape (N × d) → Output: predicted class probabilities (ŷ)

For training with all high-resolution patches, the input shape becomes:
- Input: (N × k × d) → Output: ŷ
This corresponds to using both the Feature Aggregator and the Classifier sequentially.

Config files 
CAMELYON16 ---> config/camelyon_config.yml
TCGA ----> config/tcga_config.yml

Cmd -
CAMELYON16
python step3_WSI_classification_HAFED.py --config config/camelyon_config.yml --seed 4 --arch hga --exp_name DEBUG --log_dir LOG_DIR

TCGA
python step3_WSI_classification_HAFED.py --config config/tcga_config.yml --seed 1 --arch hga --exp_name DEBUG --log_dir LOG_DIR

"""


import argparse
import os
import time
from pprint import pprint

import torch
import torch.nn.functional as F
import torchmetrics
import yaml
from timm.utils import accuracy
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from architecture.transformer import ACMIL_GA
from architecture.transformer import HACMIL_GA
from datasets.datasets import build_HDF5_feat_dataset
from utils.gpu_utils import check_gpu_availability
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from utils.utils import save_model, Struct, set_seed


def get_arguments():
    parser = argparse.ArgumentParser('WSI classification training', add_help=False)

    # Primary Arguments for HAFED Training
    parser.add_argument('--config', dest='config', default='config/camelyon_config.yml', help='settings of dataset in yaml format')
    parser.add_argument("--seed", type=int, default=4, help="set the random seed to ensure reproducibility")
    parser.add_argument("--arch", type=str, default='hga', choices=['ga', 'hga'], help="choice of architecture type")
    parser.add_argument("--exp_name", type=str, default="DEBUG", help="Experiment name")
    parser.add_argument('--ckpt_path', type=str, default=None, help='path to checkpoint file')
    parser.add_argument("--log_dir", type=str, default= ' /mnt/hdd/naman/naman/results_2025/results/camleyon16/extra', help="Path to logs folder")


    # Secondary Arguments
    parser.add_argument('--logs', default='enabled', choices=['enabled', 'disabled'], help='tensorboard logging')
    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args


def main():

    # Load config file
    args = get_arguments()

    # get parameter details from configuration file
    with open(args.config, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    if args.arch == "hga" :  # HAFED model is used here
        hyparams = {
            'dataset': conf.dataset,
            'pretrain': conf.pretrain,
            'arch': conf.arch,
            'num_tokens_1': conf.n_token_1,
            'num_tokens_2': conf.n_token_2,
            'num_masked_instances_1': conf.n_masked_patch_1,
            'num_masked_instances_2': conf.n_masked_patch_2,
            'mask_drop': conf.mask_drop,
            'lr': conf.lr,
            'seed': conf.seed,
        }

    else:
        hyparams = {
            'dataset': conf.dataset,
            'pretrain': conf.pretrain,
            'arch': conf.arch,
            'num_tokens': conf.n_token,
            'num_masked_instances': conf.n_masked_patch,
            'mask_drop': conf.mask_drop,
            'lr': conf.lr,
            'seed': conf.seed

        }

    # Initialized the config directory
    conf.writer = SummaryWriter(log_dir= os.path.join(conf.log_dir, "logs"))

    # Storing hyper-parameter details
    hyparams_text = "\n".join([f"**{key}**: {value}" for key, value in hyparams.items()])
    conf.writer.add_text("Hyperparameters", hyparams_text)

    # Initializing model directory to save weights
    ckpt_dir = os.path.join(conf.log_dir, "models", conf.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)  #

    # Print Configuration details
    print("Used config:");
    pprint(vars(conf));

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
        model = HACMIL_GA(conf,
                          n_token_1=conf.n_token_1,
                          n_token_2=conf.n_token_2,
                          n_masked_patch_1=conf.n_masked_patch_1,
                          n_masked_patch_2=conf.n_masked_patch_2,
                          mask_drop=conf.mask_drop)
    else :
        raise Exception(f"Enter a valid model architecture name e.g. ga, hga")

    model.to(conf.device)
    
    # Define criterion
    criterion = nn.CrossEntropyLoss()

    # Define optimizer, lr not important at this point
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001, weight_decay=conf.wd)


    # Default initialization to store the best parameters based on f1 + auc
    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}

    for epoch in range(conf.train_epoch):

        train_one_epoch(model, criterion, train_loader, optimizer, conf.device, epoch, conf)
        val_auc, val_acc, val_f1, val_loss = evaluate(model, criterion, val_loader, conf.device, conf, 'Val', epoch)
        test_auc, test_acc, test_f1, test_loss = evaluate(model, criterion, test_loader, conf.device, conf, 'Test', epoch)


        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
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

    # Set the network to training mode
    model.train()

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    epoch_loss_0 = 0
    epoch_loss_1 = 0
    epoch_loss_2 = 0

    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)
        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)

        # Calculate and set new learning rate
        adjust_learning_rate(optimizer, epoch + data_it/len(data_loader), conf)

        # Compute loss

        if conf.arch == 'ga':
            sub_preds, slide_preds, attn = model(image_patches)
            loss1 = criterion(slide_preds, labels)

            if conf.n_token > 1:
                loss0 = criterion(sub_preds, labels.repeat_interleave(conf.n_token))
            else:
                loss0 = torch.tensor(0.0)

            diff_loss = torch.tensor(0.).to(device, dtype=torch.float)
            for i in range(conf.n_token):
                for j in range(i + 1, conf.n_token):
                    diff_loss += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                                conf.n_token * (conf.n_token - 1) / 2)
                    
            loss = diff_loss + loss0 + loss1

            epoch_loss_0 += loss0.item()
            epoch_loss_1 += loss1.item()
            epoch_loss_2 += diff_loss.item()
            metric_logger.update(lr=optimizer.param_groups[0]['lr'])
            metric_logger.update(sub_loss=loss0.item())
            metric_logger.update(diff_loss=diff_loss.item())
            metric_logger.update(slide_loss=loss1.item())


        if conf.arch == 'hga':
            sub_preds, slide_preds, attn_1, attn_2, features, attn_raw = model(image_patches)

            loss1 = criterion(slide_preds, labels)

            if conf.n_token_2 > 1:
                loss0 = criterion(sub_preds, labels.repeat_interleave(conf.n_token_2))
            else:
                loss0 = torch.tensor(0.)

            diff_loss_1 = torch.tensor(0).to(device, dtype=torch.float)
            diff_loss_2 = torch.tensor(0).to(device, dtype=torch.float)

            for i in range(conf.n_token_1):
                for j in range(i + 1, conf.n_token_1):
                    diff_loss_1 += torch.cosine_similarity(attn_1[:, i], attn_1[:, j], dim=-1).mean() / (
                                conf.n_token_1 * (conf.n_token_1 - 1) / 2)

            for i in range(conf.n_token_2):
                for j in range(i + 1, conf.n_token_2):
                    diff_loss_2 += torch.cosine_similarity(attn_2[:, i], attn_2[:, j], dim=-1).mean() / (
                                conf.n_token_2 * (conf.n_token_2 - 1) / 2)

            loss = diff_loss_1 + diff_loss_2 + loss0 + loss1 
            diff_loss = diff_loss_1 + diff_loss_2
            epoch_loss_0 += loss0.item()
            epoch_loss_1 += loss1.item()
            epoch_loss_2 += diff_loss_1.item() + diff_loss_2.item()

            metric_logger.update(lr=optimizer.param_groups[0]['lr'])
            metric_logger.update(sub_loss=loss0.item())
            metric_logger.update(diff_loss=diff_loss.item())
            metric_logger.update(slide_loss=loss1.item())


        optimizer.zero_grad()

        # Back-propagate error and update parameters
        loss.backward()
        optimizer.step()

        if conf.logs != 'disabled':
            """ 
                We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
            """

            conf.writer.add_scalar("Step_Loss/combined ce loss", loss0.item(), data_it + (epoch * len(data_loader)))
            conf.writer.add_scalar("Step_Loss/similarity loss of network-I", diff_loss.item(), data_it + (epoch * len(data_loader)))
            conf.writer.add_scalar("Step_Loss/Final ce loss", loss1.item(), data_it + (epoch * len(data_loader)))

    if conf.logs != 'disabled':
        conf.writer.add_scalar("Epoch_Loss/combined ce loss", epoch_loss_0/len(data_loader), epoch)
        conf.writer.add_scalar("Epoch_Loss/similarity loss", epoch_loss_2/len(data_loader), epoch)
        conf.writer.add_scalar("Epoch_Loss/Final ce loss", epoch_loss_1/len(data_loader), epoch)


# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(net, criterion, data_loader, device, conf, header, epoch):

    # Set the network to evaluation mode
    net.eval()

    y_pred = []
    y_true = []
    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):

        start_time = time.time()

        image_patches = data['input'].to(device, dtype=torch.float32)
        labels = data['label'].to(device)
        slide_id = data['slide_name'][0]


        if conf.arch == 'ga':
            sub_preds, slide_preds, attn = net(image_patches)
            div_loss = torch.sum(F.softmax(attn, dim=-1) * F.log_softmax(attn, dim=-1)) / attn.shape[1]
            loss = criterion(slide_preds, labels)

        else:
            sub_preds, slide_preds, attn_1, attn_2, features, attn_raw = net(image_patches)
            div_loss = torch.sum(F.softmax(attn_2, dim=-1) * F.log_softmax(attn_2, dim=-1)) / attn_2.shape[1]
            loss = criterion(slide_preds, labels)
        
        pred = torch.softmax(slide_preds, dim=-1)
        acc1 = accuracy(pred, labels, topk=(1,))[0]

        end_time = time.time()
        total_time = end_time - start_time

        metric_logger.update(loss=loss.item())
        metric_logger.update(div_loss=div_loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=labels.shape[0])

        y_pred.append(pred)
        y_true.append(labels)

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

    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg


if __name__ == '__main__':
    main()
