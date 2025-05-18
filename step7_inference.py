"""
Here we are not going with the feature extraction at each iteration. We are utilizing the intermediate feature we have saved

Description:
This script is used to run inference using the SASHA pipeline (versions 0.1 and 0.2).
It assumes that all the required components — HAFED, TSU, and RL — have been trained beforehand.

Specifically, this script loads the trained models, performs inference over the target dataset,
and generates predictions according to the specified SASHA configuration.

Supported SASHA Variants:
- SASHA-0.1
- SASHA-0.2

Make sure to configure the required paths to checkpoints and datasets before execution.

"""

import argparse
import os
from pprint import pprint
from types import SimpleNamespace

import torch
import torchmetrics
import yaml
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from WSI_cosine_env import WSICosineObservationEnv
from WSI_env import WSIObservationEnv
from architecture.transformer import ACMIL_GA
from architecture.transformer import HACMIL_GA
from datasets.datasets import build_HDF5_feat_dataset_2
from modules.fglobal_mlp import FGlobal
from rl_algorithms.ppo import Agent, Actor, Critic
from step4_extract_intermediate_features import load_model
from utils.gpu_utils import check_gpu_availability
from utils.utils import MetricLogger
from utils.utils import Struct, set_seed


def get_arguments():
    parser = argparse.ArgumentParser('SASHA inference', add_help=False)
    parser.add_argument( '--config', dest='config', default='config/camelyon_sasha_inference.yml', help='path to config file')
    parser.add_argument('--seed', type=int, default= 4, help='set the random seed')
    parser.add_argument('--classifier_arch', default='hga', choices=['ga', 'hga', 'mha'], help='choice of architecture for HACMIL')
    parser.add_argument('--exp_name', type=str, default='DEBUG', help='name of the exp')
    parser.add_argument('--logs', default='enabled', choices=['enabled', 'disabled'], type=str, help='flag to save logs')
    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args

def load_policy_model(model, actor_optimizer, critic_optimizer, load_path, device="cpu"):

    # Load the checkpoint
    checkpoint = torch.load(load_path, map_location=device)

    # Load model weights
    model.load_state_dict(checkpoint['model'])

    # Load optimizer states
    actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
    critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

    # Get epoch number and config
    epoch = checkpoint['epoch']
    config = checkpoint['config']

    print(f"Model loaded from {load_path} at epoch {epoch}")

    return model, actor_optimizer, critic_optimizer, epoch, config

def main():
    # getting and config file
    args = get_arguments()

    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    conf.writer = SummaryWriter(log_dir=os.path.join(conf.log_dir, "logs", conf.exp_name))

    hyparams = {
        'dataset': conf.dataset,
        'pretrain': conf.pretrain,
        'classifier_arch': conf.classifier_arch,
        'seed': conf.seed,
        'frac_visit': conf.frac_visit,
        'only_ce_as_reward': conf.only_ce_as_reward,
    }
    hyparams['fraction of visit'] = conf.frac_visit

    hyparams_text = "\n".join([f"**{key}**: {value}" for key, value in hyparams.items()])
    conf.writer.add_text("Hyperparameters", hyparams_text)

    print("Used config:")
    pprint(vars(conf))

    # Loading seed
    set_seed(args.seed)

    # create dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)
    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # loading classifier
    classifier_dict, _, config, _ = load_model(conf.classifier_ckpt_path, args)
    classifier_conf = SimpleNamespace(**config)

    if classifier_conf.arch == 'ga':
        classifier = ACMIL_GA(classifier_conf, n_token=classifier_conf.n_token,
                              n_masked_patch=classifier_conf.n_masked_patch, mask_drop=classifier_conf.mask_drop)
    elif classifier_conf.arch == 'hga':
        classifier = HACMIL_GA(classifier_conf, n_token_1=classifier_conf.n_token_1,
                               n_token_2=classifier_conf.n_token_2, n_masked_patch_1=classifier_conf.n_masked_patch_1,
                               n_masked_patch_2=classifier_conf.n_masked_patch_2, mask_drop=classifier_conf.mask_drop)
    else :
        raise Exception("Select a valid classifier architecture.")

    classifier.to(conf.device)
    classifier.load_state_dict(classifier_dict)
    classifier.eval()

    # Loading TSU
    fglobal_dict = torch.load(conf.mlp_fglobal_ckpt, map_location=conf.device)
    fglobal = FGlobal(ip_dim=384 * 3, op_dim=384).to(conf.device)
    fglobal.load_state_dict(fglobal_dict['model'])
    fglobal.eval()

    # Loading RL Agent
    actor = Actor(conf=conf)
    critic = Critic(conf=conf)
    model = Agent(actor, critic, conf).to(conf.device)
    actor_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=0.001)
    critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, critic.parameters()), lr=0.001)
    model, actor_optimizer, critic_optimizer, epoch, rl_config = load_policy_model(model, actor_optimizer, critic_optimizer, conf.rl_ckpt_path, conf.device)


    # Now Evaluation Starts
    # Step 1 ----> We are evaluating the HAFED model [ ALl patches in H.R.]
    print("HAFED Inference")
    train_acc, train_auroc, train_f1_score, train_precision, train_recall, train_balance_acc, train_tumor_correctly_classified, train_tumor_mis_classified = evaluate_hafed(classifier, train_loader, 'Train', conf.device)
    val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_balance_acc, val_tumor_correctly_classified, val_tumor_mis_classified = evaluate_hafed(classifier, val_loader, 'Val', conf.device)
    test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_balance_acc, test_tumor_correctly_classified, test_tumor_mis_classified = evaluate_hafed(classifier, test_loader,'Test', conf.device)

    print(f"{'Phase':<6} | {'Acc':<6} | {'AUROC':<6} | {'F1':<6} | {'Precision':<9} | {'Recall':<6} | {'Balanced Acc':<13} | {'Tumor Correct':<15} | {'Tumor Misclass':<15}")
    print("-" * 110)
    print(f"{'Train':<6} | {train_acc:.4f} | {train_auroc:.4f} | {train_f1_score:.4f} | {train_precision:.4f} | {train_recall:.4f} | {train_balance_acc:.4f} | {len(train_tumor_correctly_classified):<5} | {len(train_tumor_mis_classified):<5}")
    print(f"{'Val':<6}   | {val_acc:.4f}   | {val_auroc:.4f}   | {val_f1_score:.4f}   | {val_precision:.4f}   | {val_recall:.4f}   | {val_balance_acc:.4f}   | {len(val_tumor_correctly_classified):<5}   | {len(val_tumor_mis_classified):<5}")
    print(f"{'Test':<6}  | {test_acc:.4f}  | {test_auroc:.4f}  | {test_f1_score:.4f}  | {test_precision:.4f}  | {test_recall:.4f}  | {test_balance_acc:.4f}  | {len(test_tumor_correctly_classified):<5}  | {len(test_tumor_mis_classified):<5}")


    ################# Breaker

    print("--- * 50")
    print("SASHA - Deterministic Policy {Pick Max}")
    train_acc, train_auroc, train_f1_score, train_precision, train_recall, train_balance_acc, train_tumor_correctly_classified, train_tumor_mis_classified = evaluate_policy(model, fglobal, classifier, train_loader,'Train', conf.device, epoch, conf)
    val_acc, val_auroc, val_f1_score, val_precision, val_recall, val_balance_acc, val_tumor_correctly_classified, val_tumor_mis_classified = evaluate_policy(model, fglobal, classifier, val_loader,'Val', conf.device, epoch, conf)
    test_acc, test_auroc, test_f1_score, test_precision, test_recall, test_balance_acc, test_tumor_correctly_classified, test_tumor_mis_classified = evaluate_policy(model, fglobal, classifier, test_loader,'Test', conf.device, epoch, conf)

    print(f"{'Phase':<6} | {'Acc':<6} | {'AUROC':<6} | {'F1':<6} | {'Precision':<9} | {'Recall':<6} | {'Balanced Acc':<13} | {'Tumor Correct':<15} | {'Tumor Misclass':<15}")
    print("-" * 110)
    print(f"{'Train':<6} | {train_acc:.4f} | {train_auroc:.4f} | {train_f1_score:.4f} | {train_precision:.4f} | {train_recall:.4f} | {train_balance_acc:.4f} | {len(train_tumor_correctly_classified):<5} | {len(train_tumor_mis_classified):<5}")
    print(f"{'Val':<6}   | {val_acc:.4f}   | {val_auroc:.4f}   | {val_f1_score:.4f}   | {val_precision:.4f}   | {val_recall:.4f}   | {val_balance_acc:.4f}   | {len(val_tumor_correctly_classified):<5}   | {len(val_tumor_mis_classified):<5}")
    print(f"{'Test':<6}  | {test_acc:.4f}  | {test_auroc:.4f}  | {test_f1_score:.4f}  | {test_precision:.4f}  | {test_recall:.4f}  | {test_balance_acc:.4f}  | {len(test_tumor_correctly_classified):<5}  | {len(test_tumor_mis_classified):<5}")


@torch.no_grad()
def evaluate_policy(model, fglobal, classifier, data_loader, header, device, epoch, conf, is_eval = True, is_top_k = False, is_top_p = False, seed = 1):

    model.eval()

    y_pred = []
    y_true = []
    slide_names = []
    metric_logger = MetricLogger(delimiter=" ")
    final_reward = 0
    patches_idx = {}

    for data in metric_logger.log_every(data_loader, 100, header):
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
        slide_name = data['slide_name'][0]
        label = data['label'].to(device)

        if conf.fglobal == 'attn':
            env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)
        else:
            env = WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)  # By default this is the environment creation

        N = state.shape[1]
        visited_patch_id = []
        done = False

        while not done:
            action, _, _ = model.get_action(state, visited_patch_id, is_eval = is_eval, is_top_k= is_top_k, is_top_p = is_top_p)
            new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier, device=device)
            state = new_state
            visited_patch_id.append(action.item())

        final_reward += reward
        loss = -1 * reward
        slide_preds, attn = classifier.classify(state)
        pred = torch.softmax(slide_preds, dim=-1)

        y_pred.append(pred)
        y_true.append(label)
        slide_names.append(slide_name)
        patches_idx[slide_name] = visited_patch_id.copy()

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred_labels = torch.argmax(y_pred, dim=-1)

    Accuracy_metric = torchmetrics.Accuracy(task='binary').to(device)
    Accuracy_metric(y_pred_labels, y_true)
    accuracy = Accuracy_metric.compute().item()

    AUROC_metric = torchmetrics.AUROC(task='binary').to(device)
    AUROC_metric(y_pred[:, 1], y_true)
    auroc = AUROC_metric.compute().item()

    F1_metric = torchmetrics.F1Score(task='binary').to(device)
    F1_metric(y_pred_labels, y_true)
    f1_score = F1_metric.compute().item()

    Precision_metric = torchmetrics.Precision(task='binary').to(device)
    Precision_metric(y_pred_labels, y_true)
    precision = Precision_metric.compute().item()

    Recall_metric = torchmetrics.Recall(task='binary').to(device)
    Recall_metric(y_pred_labels, y_true)
    recall = Recall_metric.compute().item()

    y_pred_np = y_pred_labels.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)

    # Get slide names where true label is 1
    true_label_1_indices = [i for i, val in enumerate(y_true_np) if val == 1]

    # Lists to hold categorized slide names
    correctly_classified = []
    misclassified = []

    for idx in true_label_1_indices:
        if y_pred_np[idx] == 1:
            correctly_classified.append(slide_names[idx])
        else:
            misclassified.append(slide_names[idx])

    return accuracy, auroc, f1_score, precision, recall, balanced_acc, correctly_classified, misclassified

@torch.no_grad()
def evaluate_hafed(model, data_loader, header, device):

    # Set the network to evaluation mode
    model.eval()

    y_pred = []
    y_true = []
    slide_names = []
    metric_logger = MetricLogger(delimiter="  ")

    for data in metric_logger.log_every(data_loader, 100, header):

        hr_features = data['hr'].to(device, dtype=torch.float32) # Op : N x d
        slide_name = data['slide_name'][0]
        label = data['label'].to(device)

        slide_preds, attn = model.classify(hr_features)
        pred = torch.softmax(slide_preds, dim=-1)

        y_pred.append(pred)
        y_true.append(label)
        slide_names.append(slide_name)

    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred_labels = torch.argmax(y_pred, dim=-1)


    Accuracy_metric = torchmetrics.Accuracy(task='binary').to(device)
    Accuracy_metric(y_pred_labels, y_true)
    accuracy = Accuracy_metric.compute().item()

    AUROC_metric = torchmetrics.AUROC(task='binary').to(device)
    AUROC_metric(y_pred[:, 1], y_true)
    auroc = AUROC_metric.compute().item()

    F1_metric = torchmetrics.F1Score(task='binary').to(device)
    F1_metric(y_pred_labels, y_true)
    f1_score = F1_metric.compute().item()

    Precision_metric = torchmetrics.Precision(task='binary').to(device)
    Precision_metric(y_pred_labels, y_true)
    precision = Precision_metric.compute().item()

    Recall_metric = torchmetrics.Recall(task='binary').to(device)
    Recall_metric(y_pred_labels, y_true)
    recall = Recall_metric.compute().item()

    y_pred_np = y_pred_labels.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)

    # Get slide names where true label is 1
    true_label_1_indices = [i for i, val in enumerate(y_true_np) if val == 1]

    # Lists to hold categorized slide names
    correctly_classified = []
    misclassified = []

    for idx in true_label_1_indices:
        if y_pred_np[idx] == 1:
            correctly_classified.append(slide_names[idx])
        else:
            misclassified.append(slide_names[idx])

    return accuracy, auroc, f1_score, precision, recall, balanced_acc, correctly_classified, misclassified


if __name__ == '__main__':
    main()