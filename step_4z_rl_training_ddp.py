"""
CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc_per_node=2 step4_rl_training_ddp.py --exp_name 'rl_training_ddp_2'
"""

import argparse
from doctest import master
import os
import numpy as np
import pandas as pd
import yaml
from pprint import pprint
from types import SimpleNamespace


from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from yaml import Loader

from WSI_env import WSIObservationEnv
from utils.utils import save_policy_model, Struct, set_seed
from datasets.datasets import build_HDF5_feat_dataset_2
from architecture.transformer import ACMIL_GA
from architecture.transformer import ACMIL_MHA
from architecture.transformer import HACMIL_GA
from architecture.fglobal import f_global
from rl_algorithms.ppo import Agent, Actor, Critic
from utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from timm.utils import accuracy
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

def get_arguments():
    parser = argparse.ArgumentParser('RL training', add_help=False)
    parser.add_argument(
        '--config', dest='config',
        default='config/rl_config.yml', help='path to config file'
    )
    parser.add_argument(
        '--seed', type=int, default=4, help='set the random seed'
    )
    parser.add_argument(
        '--classifier_arch', default='hga', choices=['ga', 'hga', 'mha'],
        help='choice of architecture for HACMIL'
    )
    parser.add_argument(
        '--data_dir', type=str, help='path to h5 files of lr and hr features',
        default='/mnt/hdd/tarung/project/test/rlogist/ACMIL/camelyon16_features_VIT_med_level_1_3'
    )
    parser.add_argument(
        '--exp_name', type=str, default='DEBUG', help='name of the exp'
    )

    parser.add_argument(
        '--logs', default='enabled', choices=['enabled', 'disabled'], type=str, help='flag to save logs'
    )
    args = parser.parse_args()
    return args


@torch.no_grad()
def load_model(ckpt_path, device):
    dict = torch.load(ckpt_path, map_location=device)
    curr_epoch = dict['epoch']
    config = dict['config']
    model_dict = dict['model']
    optimizer_dict = dict['optimizer']
    return model_dict, optimizer_dict, config, curr_epoch



def compute_rtgs(batch_rews, conf):
    """
        Compute the Reward-To-Go of each timestep in a batch given the rewards.

        Parameters:
            batch_rews - the rewards in a batch, Shape: (number of episodes, number of timesteps per episode)

        Return:
            batch_rtgs - the rewards to go, Shape: (number of timesteps in batch)
    """
    # The rewards-to-go (rtg) per episode per batch to return.
    # The shape will be (num timesteps per episode)
    batch_rtgs = []

    # Iterate through each episode
    for ep_rews in reversed(batch_rews):

        discounted_reward = 0 # The discounted reward so far

        # Iterate through all rewards in the episode. We go backwards for smoother calculation of each
        # discounted return (think about why it would be harder starting from the beginning)
        for rew in reversed(ep_rews):
            discounted_reward = rew + discounted_reward * conf.gamma
            batch_rtgs.insert(0, discounted_reward)

    # Convert the rewards-to-go into a tensor
    batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

    return batch_rtgs



def main():

    #setting up DDP--------------------------
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        assert torch.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
        if master_process:
            print(f"----- DDP {ddp} and Master node is {ddp_local_rank}")
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        master_process = 0


    #getting and config file
    args = get_arguments()
    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    if conf.pretrain == 'medical_ssl' and conf.backbone == 'ViT-S/16':
        conf.D_feat = 384
        conf.D_inner = 128
    elif conf.pretrain == 'medical_ssl' and conf.backbone == 'ResNet_50':
        conf.D_feat = 2048
        conf.D_inner = 512
    elif conf.pretrain == 'natural_supervised' and conf.backbone == 'ViT-B/16':
        conf.D_feat = 768
        conf.D_inner = 384
    elif conf.pretrain == 'path-clip-B' or conf.pretrain == 'openai-clip-B' or conf.pretrain == 'plip'\
            or conf.pretrain == 'quilt-net'  or conf.pretrain == 'path-clip-B-AAAI'  or conf.pretrain == 'biomedclip':
        conf.D_feat = 512
        conf.D_inner = 256
    elif conf.pretrain == 'path-clip-L-336' or conf.pretrain == 'openai-clip-L-336':
        conf.D_feat = 768
        conf.D_inner = 384
    elif conf.pretrain == 'UNI':
        conf.D_feat = 1024
        conf.D_inner = 512
    elif conf.pretrain == 'GigaPath':
        conf.D_feat = 1536
        conf.D_inner = 768    
    conf.device = device
    if master_process:
        conf.writer = SummaryWriter(log_dir=os.path.join(conf.log_dir, "logs", conf.exp_name))
        hyparams = {
        'dataset': conf.dataset,
        'pretrain': conf.pretrain,
        'classifier_arch': conf.classifier_arch,
        'lr': conf.lr,
        'seed': conf.seed,
        'frac_visit': conf.frac_visit,
        'gamma': conf.gamma,
        'clip': conf.clip,
        'num_envs': conf.num_envs,
        'num_epochs_on_single_roll_out': conf.num_epochs_on_single_roll_out,
        'only_ce_as_reward': conf.only_ce_as_reward
    }

        hyparams_text = "\n".join([f"**{key}**: {value}" for key, value in hyparams.items()])
        conf.writer.add_text("Hyperparameters", hyparams_text)
        ckpt_dir = os.path.join(conf.log_dir, "models", conf.exp_name)
        os.makedirs(ckpt_dir, exist_ok=True)  # Create the 'ckpt' directory if it doesn't exist

        print("Used config:");
        pprint(vars(conf));

    set_seed(args.seed)




    #dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    # loading classifier
    classifier_dict, _, config, _ = load_model(ckpt_path=conf.classifier_ckpt_path, device=device)
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
    #loading state update model
    fglobal_dict, _, fg_config, _ = load_model(ckpt_path=conf.fglobal_ckpt_path, device=device)
    fg_config = SimpleNamespace(**fg_config)
    fglobal = f_global(
        in_features=fg_config.in_features,
        nhead=fg_config.nhead,
        num_layers=fg_config.nlayer,
        ff_dim=fg_config.ff_dim,
        dropout=fg_config.dropout
    ).to(device)
    fglobal.load_state_dict(fglobal_dict)
    fglobal.eval()


    #creating agent 
    actor = Actor(conf=conf)
    critic  = Critic(conf=conf)

    model = Agent(actor, critic, conf).to(device)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        fglobal = DDP(fglobal, device_ids=[ddp_local_rank])
        classifier = DDP(classifier, device_ids=[ddp_local_rank])

    raw_model = model.module if ddp else model
    raw_fglobal = fglobal.module if ddp else fglobal
    raw_classifer = classifier.module if ddp else classifier

    actor_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, raw_model.actor.parameters()), lr=0.001, weight_decay=conf.wd)
    critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, raw_model.critic.parameters()), lr=0.001, weight_decay=conf.wd)
    if master_process:
        best_state = {'epoch':-1, 'val_acc':0, 'val_auc':0, 'val_f1':0, 'test_acc':0, 'test_auc':0, 'test_f1':0}
    train_epoch = conf.train_epoch
    for epoch in range(train_epoch):
        train_one_epoch(model, raw_fglobal, raw_classifer, train_loader, actor_optimizer, critic_optimizer, device, epoch, conf, master_process)
        dist.barrier()
        if master_process:
            val_auc, val_acc, val_f1, val_loss = evaluate_policy(model, raw_fglobal, raw_classifer, val_loader, 'Val', device, epoch, conf)
            test_auc, test_acc, test_f1, test_loss = evaluate_policy(model, raw_fglobal, raw_classifer, test_loader, 'Test', device, epoch, conf)

            if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
                best_state['epoch'] = epoch
                best_state['val_auc'] = val_auc
                best_state['val_acc'] = val_acc
                best_state['val_f1'] = val_f1
                best_state['test_auc'] = test_auc
                best_state['test_acc'] = test_acc
                best_state['test_f1'] = test_f1
                save_policy_model(conf=conf, model=raw_model, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, epoch=epoch,
                    save_path=os.path.join(ckpt_dir, 'checkpoint-best.pt'))
        dist.barrier()

    if master_process:
        print('\n')
        save_policy_model(conf=conf, model=raw_model, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer, epoch=epoch,
            save_path=os.path.join(ckpt_dir, 'checkpoint-last.pt'))
        print("Results on best epoch:")
        print(best_state)
        best_state_text = "\n".join([f"{key}: {value}" for key, value in best_state.items()])
        conf.writer.add_text("Best Model State", best_state_text, global_step=best_state["epoch"])
        conf.writer.close()


def train_one_epoch(model, fglobal, classifier, data_loader, actor_optimizer, critic_optimizer, device, epoch, conf, master_process):
    model.train()
    
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('actor_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('critic_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}] - device:{}'.format(epoch, device)
    print_freq = 100

    total_reward = 0
    epoch_actor_loss = 0
    epoch_critic_loss = 0
    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
        slide_name = data['slide_name'][0]
        label = data['label'].to(device)

        env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)
        
        adjust_learning_rate(actor_optimizer, epoch + data_it/len(data_loader), conf)
        adjust_learning_rate(critic_optimizer, epoch + data_it/len(data_loader), conf)
        
        #collectiong data as PPO is a onpolicy algorithm
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        N = state.shape[1]
        T = min(N, int(conf.frac_visit * N))
        with torch.no_grad():
            for i in range(conf.num_envs):
                ep_rews = []
                visited_patch_id = []
                state = env.reset()
                done = False
                ep_t = 0
                while not done:
                    #track observations in this batch
                    batch_obs.append(state)
                    action, log_prob = model.module.get_action(state, visited_patch_id)
                    new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier, device=device)
                    state = new_state
                    ep_rews.append(reward)
                    batch_acts.append(action.item())
                    batch_log_probs.append(log_prob.item())
                    visited_patch_id.append(action.item())
                    ep_t += 1
                batch_lens.append(ep_t + 1)
                batch_rews.append(ep_rews)

            batch_obs = torch.stack(batch_obs).squeeze(1)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float, device=conf.device)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=conf.device)
            batch_rtgs = compute_rtgs(batch_rews, conf).to(conf.device) 
        V, _ = model.module.evaluate(batch_obs, batch_acts)
        A_k = batch_rtgs - V.detach()  
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)

        #updating parameters for n epochs
        for _ in range(conf.num_epochs_on_single_roll_out):
            torch.cuda.empty_cache()
            V, curr_log_probs = model.module.evaluate(batch_obs, batch_acts)
            ratios = torch.exp(curr_log_probs - batch_log_probs)
            #calculate surrogate loss
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - conf.clip, 1 + conf.clip) * A_k
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)
            # Synchronize losses across GPUs
            dist.all_reduce(actor_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(critic_loss, op=dist.ReduceOp.AVG)
            #updating actor
            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            actor_optimizer.step()
            #updating critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()
            if master_process:
                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()

        metric_logger.update(actor_lr=actor_optimizer.param_groups[0]['lr'])
        metric_logger.update(critic_lr=critic_optimizer.param_groups[0]['lr'])
        metric_logger.update(actor_loss=actor_loss.item())
        metric_logger.update(critic_loss=critic_loss.item())

        if conf.logs != 'disabled' and master_process:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            conf.writer.add_scalar("Loss/critic_loss", critic_loss.item(), data_it + (epoch * len(data_loader)))
            conf.writer.add_scalar("Loss/actor_loss", actor_loss.item(), data_it + (epoch * len(data_loader)))
        
        del batch_obs, batch_acts, batch_log_probs, batch_rtgs, V, curr_log_probs, ratios, surr1, surr2, A_k

        # Clear CUDA cache
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    if conf.logs != 'disabled' and master_process:
        conf.writer.add_scalar("Epoch_Loss/critic_loss", epoch_critic_loss/len(data_loader), epoch)
        conf.writer.add_scalar("Epoch_Loss/actor_loss", epoch_actor_loss/len(data_loader), epoch)
        



@torch.no_grad()
def evaluate_policy(model, fglobal, classifier, data_loader, header, device, epoch, conf):
    model.eval()

    y_pred = []
    y_true = []
    metric_logger = MetricLogger(delimiter=" ")
    final_reward = 0
    for data in metric_logger.log_every(data_loader, 100, header):
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
        slide_name = data['slide_name'][0]
        label = data['label'].to(device)

        env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

        N = state.shape[1]
        T = min(N, int(conf.frac_visit * N))
        visited_patch_id = []
        done = False
        while not done:
            action, log_prob = model.module.get_action(state, visited_patch_id)
            new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier, device=device)
            state = new_state
        final_reward += reward
        loss = -1 * reward
        slide_preds, attn = classifier.classify(state)
        pred = torch.softmax(slide_preds, dim=-1)
        acc1 = accuracy(pred, label, topk=(1,))[0]
        metric_logger.update(loss=loss)
        metric_logger.meters['acc1'].update(acc1.item(), n=label.shape[0])

        y_pred.append(pred)
        y_true.append(label)
    y_pred = torch.cat(y_pred, dim=0)
    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred_labels = torch.argmax(y_pred, dim=-1).cpu().numpy()
    y_pred_probs = y_pred[:, 1].cpu().numpy()
    auroc = roc_auc_score(y_true, y_pred_probs)

    f1score = f1_score(y_true, y_pred_labels)

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
        .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1score))
    
    if conf.logs != 'disabled':
        conf.writer.add_scalar(f"{header}/accuracy", metric_logger.acc1.global_avg, epoch)
        conf.writer.add_scalar(f"{header}/auroc", auroc, epoch)
        conf.writer.add_scalar(f"{header}/f1", f1score, epoch)
        conf.writer.add_scalar(f"{header}/loss", metric_logger.loss.global_avg, epoch)
    return auroc, metric_logger.acc1.global_avg, f1score, metric_logger.loss.global_avg



                
if __name__ == '__main__':
    main()

