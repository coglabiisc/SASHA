"""
CUDA_VISIBLE_DEVICES=1 python step4_rl_training.py --exp_name 'Rl_training_1'
"""

import argparse
import os
from pprint import pprint
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import yaml
from sklearn.metrics import balanced_accuracy_score
from timm.utils import accuracy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sasha.WSI_cosine_env import WSICosineObservationEnv
from sasha.WSI_env import WSIObservationEnv
from sasha.architecture.transformer import ACMIL_GA
from sasha.architecture.transformer import ACMIL_MHA
from sasha.architecture.transformer import HACMIL_GA
from sasha.architecture.transformer_sparse import HACMIL_GA_Sparse
from sasha.datasets.datasets import build_HDF5_feat_dataset_2
from sasha.rl_algorithms.ppo import Agent, Actor, Critic
from sasha.utils.utils import MetricLogger, SmoothedValue, adjust_learning_rate
from sasha.utils.utils import save_policy_model, Struct, set_seed
from rlogist_histo.models.f_global_mlp import FGlobal
from rlogist_histo.utils.gpu_utils import check_gpu_availability

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Adding Device Details
gpus = check_gpu_availability(5, 1, [0,1,2,5])
print(f"occupied {gpus}")
device = torch.device(f"cuda:{gpus[0]}")

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
        default= None
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
def load_model(ckpt_path):
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


def gae(rewards, values, episode_ends, gamma, lam):
    """Compute generalized advantage estimate.
        rewards: a list of rewards at each step.
        values: the value estimate of the state at each step.
        episode_ends: an array of the same shape as rewards, with a 1 if the
            episode ended at that step and a 0 otherwise.
        gamma: the discount factor.
        lam: the GAE lambda parameter.
    """
    # Invert episode_ends to have 0 if the episode ended and 1 otherwise
    episode_ends = (episode_ends * -1) + 1

    N = rewards.shape[0]
    T = rewards.shape[1]
    gae_step = np.zeros((N, ))
    advantages = np.zeros((N, T))
    for t in reversed(range(T - 1)):
        # First compute delta, which is the one-step TD error
        delta = rewards[:, t] + gamma * values[:, t + 1] * episode_ends[:, t] - values[:, t]
        # Then compute the current step's GAE by discounting the previous step
        # of GAE, resetting it to zero if the episode ended, and adding this
        # step's delta
        gae_step = delta + gamma * lam * episode_ends[:, t] * gae_step
        # And store it
        advantages[:, t] = gae_step
    return advantages


def main():
    # getting and config file
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
    elif conf.pretrain == 'imagenet' and conf.backbone == 'Resnet':
        conf.D_feat = 1024
        conf.D_inner = 512
    elif conf.pretrain == 'natural_supervised' and conf.backbone == 'ViT-B/16':
        conf.D_feat = 768
        conf.D_inner = 384
    elif conf.pretrain == 'path-clip-B' or conf.pretrain == 'openai-clip-B' or conf.pretrain == 'plip' \
            or conf.pretrain == 'quilt-net' or conf.pretrain == 'path-clip-B-AAAI' or conf.pretrain == 'biomedclip':
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
        'only_ce_as_reward': conf.only_ce_as_reward,
        'lambda': conf.lam,
        'use_gae': conf.use_gae,
        'max_gradient_norm': conf.max_grad_norm,
        'entropy_coef': conf.entropy_coef,
        'use_entropy_loss': conf.use_entropy_loss,
    }

    hyparams['fraction of visit'] = conf.frac_visit
    hyparams['num_envs'] = conf.num_envs

    hyparams_text = "\n".join([f"**{key}**: {value}" for key, value in hyparams.items()])
    conf.writer.add_text("Hyperparameters", hyparams_text)
    ckpt_dir = os.path.join(conf.log_dir, "models", conf.exp_name)
    os.makedirs(ckpt_dir, exist_ok=True)  # Create the 'ckpt' directory if it doesn't exist

    print("Used config:")
    pprint(vars(conf))

    set_seed(args.seed)
    # dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                            num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    # loading classifier
    classifier_dict, _, config, _ = load_model(ckpt_path=conf.classifier_ckpt_path)
    classifier_conf = SimpleNamespace(**config)
    if classifier_conf.arch == 'ga':
        classifier = ACMIL_GA(classifier_conf, n_token=classifier_conf.n_token,
                              n_masked_patch=classifier_conf.n_masked_patch, mask_drop=classifier_conf.mask_drop)
    elif classifier_conf.arch == 'hga':
        classifier = HACMIL_GA(classifier_conf, n_token_1=classifier_conf.n_token_1,
                               n_token_2=classifier_conf.n_token_2, n_masked_patch_1=classifier_conf.n_masked_patch_1,
                               n_masked_patch_2=classifier_conf.n_masked_patch_2, mask_drop=classifier_conf.mask_drop)
    elif classifier_conf.arch == 'hga_sparse' :
        classifier = HACMIL_GA_Sparse(classifier_conf, n_token_1=classifier_conf.n_token_1,
                               n_token_2=classifier_conf.n_token_2, n_masked_patch_1=classifier_conf.n_masked_patch_1,
                               n_masked_patch_2=classifier_conf.n_masked_patch_2, mask_drop=classifier_conf.mask_drop)
    else:
        classifier = ACMIL_MHA(classifier_conf, n_token=classifier_conf.n_token,
                               n_masked_patch=classifier_conf.n_masked_patch, mask_drop=classifier_conf.mask_drop)
    classifier.to(device)
    classifier.load_state_dict(classifier_dict)
    classifier.eval()


    # loading state update model
    if conf.fglobal == 'attn':
        # fglobal_dict, _, fg_config, _ = load_model(ckpt_path=conf.fglobal_ckpt_path)
        # fg_config = SimpleNamespace(**fg_config)
        # fglobal = f_global(
        #     in_features=fg_config.in_features,
        #     nhead=fg_config.nhead,
        #     num_layers=fg_config.nlayer,
        #     ff_dim=fg_config.ff_dim,
        #     dropout=fg_config.dropout
        # ).to(device)
        # fglobal.load_state_dict(fglobal_dict)

        fglobal = FGlobal(ip_dim= 384 * 3, op_dim= 384).to(device)
        fglobal.load_state_dict(torch.load(conf.fglobal_ckpt_path, weights_only=True, map_location = device))

    else:
        fglobal_dict = torch.load(conf.mlp_fglobal_ckpt, map_location=device)
        fglobal = FGlobal(ip_dim= 384 * 3, op_dim= 384).to(device)
        fglobal.load_state_dict(fglobal_dict)
    fglobal.eval()

    # creating agent
    actor = Actor(conf=conf)
    critic = Critic(conf=conf)

    model = Agent(actor, critic, conf).to(device)

    actor_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, actor.parameters()), lr=0.001,
                                        weight_decay=conf.wd)
    critic_optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, critic.parameters()), lr=0.001,
                                         weight_decay=conf.wd)

    best_state = {'epoch': -1, 'val_acc': 0, 'val_auc': 0, 'val_f1': 0, 'test_acc': 0, 'test_auc': 0, 'test_f1': 0,
                  'test_bal_acc': 0, 'test_precision': 0, 'test_recall': 0}
    train_epoch = conf.train_epoch
    for epoch in range(train_epoch):
        train_one_epoch(model, fglobal, classifier, train_loader, actor_optimizer, critic_optimizer, device, epoch,
                        conf)

        val_auc, val_acc, val_f1, val_loss, val_balanced_acc, val_precision, val_recall = evaluate_policy(model,
                                                                                                          fglobal,
                                                                                                          classifier,
                                                                                                          val_loader,
                                                                                                          'Val', device,
                                                                                                          epoch, conf)
        test_auc, test_acc, test_f1, test_loss, test_balanced_acc, test_precision, test_recall = evaluate_policy(model,
                                                                                                                 fglobal,
                                                                                                                 classifier,
                                                                                                                 test_loader,
                                                                                                                 'Test',
                                                                                                                 device,
                                                                                                                 epoch,
                                                                                                                 conf)

        if val_f1 + val_auc > best_state['val_f1'] + best_state['val_auc']:
            best_state['epoch'] = epoch
            best_state['val_auc'] = val_auc
            best_state['val_acc'] = val_acc
            best_state['val_f1'] = val_f1
            best_state['test_auc'] = test_auc
            best_state['test_acc'] = test_acc
            best_state['test_f1'] = test_f1
            best_state['test_bal_acc'] = test_balanced_acc
            best_state['test_precision'] = test_precision
            best_state['test_recall'] = test_recall
            save_policy_model(conf=conf, model=model, actor_optimizer=actor_optimizer,
                              critic_optimizer=critic_optimizer, epoch=epoch,
                              save_path=os.path.join(ckpt_dir, 'checkpoint-best.pt'))
        print('\n')

    save_policy_model(conf=conf, model=model, actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
                      epoch=epoch,
                      save_path=os.path.join(ckpt_dir, 'checkpoint-last.pt'))
    print("Results on best epoch:")
    print(best_state)
    best_state_text = "\n".join([f"{key}: {value}" for key, value in best_state.items()])
    conf.writer.add_text("Best Model State", best_state_text, global_step=best_state["epoch"])
    conf.writer.close()


def train_one_epoch(model, fglobal, classifier, data_loader, actor_optimizer, critic_optimizer, device, epoch, conf):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('actor_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('critic_lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    total_reward = 0
    epoch_actor_loss = 0
    epoch_critic_loss = 0
    for data_it, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        torch.cuda.empty_cache()
        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32).clone()
        slide_name = data['slide_name'][0]
        label = data['label'].to(device)
        if conf.fglobal == 'attn':
            env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)
        else:
            env = WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

        adjust_learning_rate(actor_optimizer, epoch + data_it / len(data_loader), conf)
        adjust_learning_rate(critic_optimizer, epoch + data_it / len(data_loader), conf)

        # collectiong data as PPO is a onpolicy algorithm
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_lens = []
        batch_visited_patch_indices = [] # Added by Naman
        dones = []
        N = state.shape[1]
        T = min(N, int(conf.frac_visit * N))
        with torch.no_grad():
            for i in range(conf.num_envs):
                ep_rews = []
                visited_patch_id = []
                env.reset()
                done = False
                ep_t = 0
                while not done:
                    # track observations in this batch
                    batch_obs.append(state)
                    action, log_prob, entropy = model.get_action(state, visited_patch_id[0] if len(visited_patch_id) > 0 else [], is_eval = False)
                    state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier,
                                                   device=device)
                    curr_visited_patches = env.get_visited_patches()
                    # state = new_state.clone()
                    ep_rews.append(reward)
                    batch_acts.append(action.item())
                    batch_log_probs.append(log_prob.item())
                    visited_patch_id.append(curr_visited_patches.copy())
                    batch_visited_patch_indices.append(curr_visited_patches.copy())
                    dones.append(done)
                    ep_t += 1
                state = data['lr'].to(device, dtype=torch.float32).clone()
                batch_lens.append(ep_t + 1)
                batch_rews.append(ep_rews)

            batch_obs = torch.stack(batch_obs).squeeze(1)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float, device=conf.device)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=conf.device)
            batch_visited_patch_indices = torch.tensor(batch_visited_patch_indices, dtype=torch.bool, device= conf.device)
            batch_rtgs = compute_rtgs(batch_rews, conf).to(conf.device)
            V, _, _ = model.evaluate(batch_obs, batch_acts, batch_visited_patch_indices)
            if conf.use_gae:
                V = V.view(conf.num_envs, -1).cpu().numpy()
                dones = np.array(dones).reshape(conf.num_envs, -1)
                batch_rews = np.array(batch_rews).reshape(conf.num_envs, -1)
                A_k = gae(rewards=batch_rews, values=V, episode_ends=dones, gamma=conf.gamma, lam=conf.lam)
                A_k = torch.tensor(A_k, device=conf.device).flatten()
            else:
                A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-8)

        # updating parameters for n epochs
        for _ in range(conf.num_epochs_on_single_roll_out):
            torch.cuda.empty_cache()
            V, curr_log_probs, entropy = model.evaluate(batch_obs, batch_acts, batch_visited_patch_indices)
            ratios = torch.exp(curr_log_probs - batch_log_probs)
            # calculate surrogate loss
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - conf.clip, 1 + conf.clip) * A_k
            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = nn.MSELoss()(V, batch_rtgs)
            entropy_loss = entropy.mean()
            if conf.use_entropy_loss:
                actor_loss += conf.entropy_coef * entropy_loss
            # updating actor
            actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(model.actor.parameters(), conf.max_grad_norm)
            actor_optimizer.step()
            # updating critic
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), conf.max_grad_norm)
            critic_optimizer.step()

            epoch_actor_loss += actor_loss.item()
            epoch_critic_loss += critic_loss.item()

            torch.cuda.empty_cache()
            import gc
            gc.collect()
        metric_logger.update(actor_lr=actor_optimizer.param_groups[0]['lr'])
        metric_logger.update(critic_lr=critic_optimizer.param_groups[0]['lr'])
        metric_logger.update(actor_loss=actor_loss.item())
        metric_logger.update(critic_loss=critic_loss.item())

        if conf.logs != 'disabled':
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

    if conf.logs != 'disabled':
        conf.writer.add_scalar("Epoch_Loss/critic_loss", epoch_critic_loss / len(data_loader), epoch)
        conf.writer.add_scalar("Epoch_Loss/actor_loss", epoch_actor_loss / len(data_loader), epoch)


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

        if conf.fglobal == 'attn':
            env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)
        else:
            env = WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

        N = state.shape[1]
        T = min(N, int(conf.frac_visit * N))
        visited_patch_id = []
        done = False
        while not done:
            action, log_prob, entropy = model.get_action(state, visited_patch_id[0] if len(visited_patch_id) > 0 else [], is_eval = True)
            new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier,
                                               device=device)

            curr_visited_patches = env.get_visited_patches()
            visited_patch_id.append(curr_visited_patches.copy())
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
    y_true = torch.cat(y_true, dim=0)
    y_pred_labels = torch.argmax(y_pred, dim=-1)
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

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss, AUROC=auroc, F1=f1_score))
    if conf.logs != 'disabled':
        conf.writer.add_scalar(f"{header}/accuracy", metric_logger.acc1.global_avg, epoch)
        conf.writer.add_scalar(f"{header}/auroc", auroc, epoch)
        conf.writer.add_scalar(f"{header}/f1", f1_score, epoch)
        conf.writer.add_scalar(f"{header}/loss", metric_logger.loss.global_avg, epoch)
    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg, balanced_acc, precision, recall


if __name__ == '__main__':
    main()