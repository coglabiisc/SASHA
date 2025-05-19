'''
CAMELYON16
python step8_wsi_classification_dtfd.py --config config/camelyon_config.yml --log_dir LOG_DIR

'''
import argparse
import os
from pprint import pprint

import torch
import torchmetrics
import yaml
from timm.utils import accuracy
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from architecture.Attention import Attention_Gated as Attention
from architecture.Attention import Attention_with_Classifier
from architecture.network import Classifier_1fc, DimReduction
from datasets.datasets import build_HDF5_feat_dataset
from utils.gpu_utils import check_gpu_availability
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from utils.utils import Struct, set_seed
from utils.utils import get_cam_1d


def get_arguments():
    parser = argparse.ArgumentParser('Patch classification training', add_help=False)
    parser.add_argument('--config', dest='config', default= None, help='settings of Tip-Adapter in yaml format')
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--seed", type=int, default=4, help="set the random seed to ensure reproducibility")
    parser.add_argument('--logs_mode', default='disabled', choices=['offline', 'online', 'disabled'], help='the model of wandb')
    parser.add_argument("--n_shot", type=int, default=-1, help="number of wsi images")
    parser.add_argument("--w_loss", type=float, default=1.0, help="number of query token")
    parser.add_argument('--numGroup', default=4, type=int)
    parser.add_argument('--total_instance', default=4, type=int)
    parser.add_argument('--numGroup_test', default=4, type=int)
    parser.add_argument('--total_instance_test', default=4, type=int)
    parser.add_argument('--grad_clipping', default=5, type=float)
    parser.add_argument('--pretrain', default='medical_ssl', choices=['natural_supervised', 'medical_ssl', 'path-clip-L-336'], help='settings of Tip-Adapter in yaml format')
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")

    parser.add_argument("--exp_name", type=str, default="DEBUG", help="Experiment name")
    parser.add_argument("--log_dir", type=str, default= None, help="Path to logs folder")

    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args

def train_one_epoch(classifier, attention, dimReduction, UClassifier, criterion, data_loader, optimizer0, optimizer1, device, epoch, conf, distill='MaxMinS'):

    """
    Trains the given network for one epoch according to given criterions (loss functions)
    """

    # Set the network to training mode
    classifier.train()
    dimReduction.train()
    attention.train()
    UClassifier.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # Calculate and set new learning rate
        adjust_learning_rate(optimizer0, epoch + data_it/len(data_loader), conf)
        adjust_learning_rate(optimizer1, epoch + data_it/len(data_loader), conf)


        # For data_it, data in enumerate(data_loader, start=epoch * len(data_loader)):
        # Move input batch onto GPU if eager execution is enabled (default), else leave it on CPU
        # Data is a dict with keys `input` (patches) and `{task_name}` (labels for given task)

        tfeat_tensor = data['input'].to(device, dtype=torch.float32)
        tfeat_tensor = tfeat_tensor[0]
        if tfeat_tensor.ndim == 3 :
            tfeat_tensor = tfeat_tensor.reshape(tfeat_tensor.shape[0] * tfeat_tensor.shape[1], tfeat_tensor.shape[2])
        tslideLabel = data['label'].to(device)

        instance_per_group = conf.total_instance // conf.numGroup
        feat_index = torch.randperm(tfeat_tensor.shape[0]).to(device)
        index_chunk_list = torch.tensor_split(feat_index, conf.numGroup)

        slide_pseudo_feat = []
        slide_sub_preds = []
        slide_sub_labels = []

        for tindex in index_chunk_list:
            slide_sub_labels.append(tslideLabel)
            subFeat_tensor = torch.index_select(tfeat_tensor, dim=0, index=tindex)
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            tPredict = classifier(tattFeat_tensor)  ### 1 x 2
            slide_sub_preds.append(tPredict)

            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)

            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)  ##########################
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)  ### numGroup x fs

        ## optimization for the first tier
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  ### numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  ### numGroup
        loss0 = criterion(slide_sub_preds, slide_sub_labels).mean()
        optimizer0.zero_grad()
        loss0.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), conf.grad_clipping)
        torch.nn.utils.clip_grad_norm_(attention.parameters(), conf.grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), conf.grad_clipping)
        optimizer0.step()

        ## optimization for the second tier
        gSlidePred = UClassifier(slide_pseudo_feat)
        loss1 = criterion(gSlidePred, tslideLabel).mean()
        optimizer1.zero_grad()
        loss1.backward()
        torch.nn.utils.clip_grad_norm_(UClassifier.parameters(), conf.grad_clipping)
        optimizer1.step()


        metric_logger.update(lr=optimizer0.param_groups[0]['lr'])
        metric_logger.update(loss0=loss0.item())
        metric_logger.update(loss1=loss1.item())

        if conf.logs_mode != 'disabled':
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            conf.writer.add_scalar('loss/loss0', loss0, epoch)
            conf.writer.add_scalar('loss/loss1', loss1, epoch)

# Disable gradient calculation during evaluation
@torch.no_grad()
def evaluate(classifier, attention, dimReduction, UClassifier, criterion, data_loader, device, conf, header, distill='MaxMinS'):

    # Set the network to evaluation mode
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    UClassifier.eval()

    y_pred = []
    y_true = []
    instance_per_group = conf.total_instance // conf.numGroup

    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):
        tfeat = data['input'].to(device, dtype=torch.float32)
        tfeat = tfeat[0]
        if tfeat.ndim == 3 :
            tfeat = tfeat.reshape(tfeat.shape[0] * tfeat.shape[1], tfeat.shape[2])
        tslideLabel = data['label'].to(device)

        midFeat = dimReduction(tfeat)

        AA = attention(midFeat, isNorm=False).squeeze(0)  ## N

        feat_index = torch.randperm(tfeat.shape[0]).to(device)
        index_chunk_list = torch.tensor_split(feat_index, conf.numGroup)

        slide_d_feat = []

        for tindex in index_chunk_list:
            tmidFeat = midFeat.index_select(dim=0, index=tindex)

            tAA = AA.index_select(dim=0, index=tindex)
            tAA = torch.softmax(tAA, dim=0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs

            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:, -1], descending=True)

            if distill == 'MaxMinS':
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx_min = sort_idx[-instance_per_group:].long()
                topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                slide_d_feat.append(d_inst_feat)
            elif distill == 'MaxS':
                topk_idx_max = sort_idx[:instance_per_group].long()
                topk_idx = topk_idx_max
                d_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)
                slide_d_feat.append(d_inst_feat)
            elif distill == 'AFS':
                slide_d_feat.append(tattFeat_tensor)

        slide_d_feat = torch.cat(slide_d_feat, dim=0)

        gSlidePred = UClassifier(slide_d_feat)
        allSlide_pred_softmax = torch.softmax(gSlidePred, dim=1)

        loss = criterion(allSlide_pred_softmax, tslideLabel)
        acc1 = accuracy(allSlide_pred_softmax, tslideLabel, topk=(1,))[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=1)

        y_pred.append(allSlide_pred_softmax)
        y_true.append(tslideLabel)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)

    AUROC_metric = torchmetrics.AUROC(num_classes = conf.n_class, task='multiclass').to(device)
    AUROC_metric(y_pred, y_true)
    auroc = AUROC_metric.compute().item()
    F1_metric = torchmetrics.F1Score(num_classes = conf.n_class, task='multiclass').to(device)
    F1_metric(y_pred, y_true)
    f1_score = F1_metric.compute().item()

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}' .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))

    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg


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
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # define network
    classifier = Classifier_1fc(conf.D_inner, conf.n_class, 0).to(conf.device)
    attention = Attention(conf.D_inner).to(conf.device)
    dimReduction = DimReduction(conf.D_feat, conf.D_inner).to(conf.device)
    attCls = Attention_with_Classifier(L=conf.D_inner, num_cls=conf.n_class, droprate=0).to(conf.device)


    trainable_parameters = []
    trainable_parameters += list(classifier.parameters())
    trainable_parameters += list(attention.parameters())
    trainable_parameters += list(dimReduction.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer_adam0 = torch.optim.Adam(trainable_parameters, lr=conf.lr,  weight_decay=conf.wd)
    optimizer_adam1 = torch.optim.Adam(attCls.parameters(), lr=conf.lr,  weight_decay=conf.wd)

    best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}

    for epoch in range(conf.train_epoch):

        train_one_epoch(classifier, attention, dimReduction, attCls, criterion, train_loader, optimizer_adam0, optimizer_adam1, conf.device, epoch, conf)
        val_auc, val_acc, val_f1, val_loss = evaluate(classifier, attention, dimReduction, attCls, criterion, val_loader, conf.device, conf, 'Val')
        test_auc, test_acc, test_f1, test_loss = evaluate(classifier, attention, dimReduction, attCls, criterion, test_loader, conf.device, conf, 'Test')

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

        print('\n')

    print("Results on best epoch:")
    print(best_state)

if __name__ == '__main__':
    main()
