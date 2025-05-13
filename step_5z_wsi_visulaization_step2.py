"""
CUDA_VISIBLE_DEVICES=1 python step4_rl_training.py --exp_name 'Rl_training_1'
"""

import argparse
import os
import time
from pprint import pprint
from types import SimpleNamespace

import pandas as pd
import torchmetrics
import yaml
from sklearn.metrics import balanced_accuracy_score
from timm.utils import accuracy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from rlogist_histo.models.f_global_mlp import FGlobal
from rlogist_histo.utils.gpu_utils import check_gpu_availability
from sasha.WSI_cosine_env import WSICosineObservationEnv
from sasha.WSI_env import WSIObservationEnv
from sasha.architecture.fglobal import f_global
from sasha.architecture.transformer import ACMIL_GA
from sasha.architecture.transformer import ACMIL_MHA
from sasha.architecture.transformer import HACMIL_GA
from sasha.architecture.transformer_sparse import HACMIL_GA_Sparse
from sasha.datasets.datasets import build_HDF5_feat_dataset_2
from sasha.rl_algorithms.ppo import Agent, Actor, Critic
from sasha.step4_rl_training import load_model
from sasha.utils.utils import MetricLogger
from sasha.utils.utils import Struct, set_seed
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import math
import csv

def get_arguments():
    parser = argparse.ArgumentParser('RL training', add_help=False)
    parser.add_argument( '--config', dest='config', default='config/rl_config_vis_dgx.yml', help='path to config file')
    parser.add_argument('--seed', type=int, default=4, help='set the random seed')
    parser.add_argument('--classifier_arch', default='hga', choices=['ga', 'hga', 'mha'], help='choice of architecture for HACMIL')
    parser.add_argument('--exp_name', type=str, default='DEBUG', help='name of the exp')
    parser.add_argument('--logs', default='enabled', choices=['enabled', 'disabled'], type=str, help='flag to save logs')
    parser.add_argument('--time_csv', type=str, default="/media/internal_8T/naman/rlogist/champ.csv",help='path to csv file with time of segmentation')
    args = parser.parse_args()
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

    # create dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)

    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True,
                              num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False,
                            num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)

    # loading classifier
    classifier_dict, _, config, _ = load_model(ckpt_path=conf.classifier_ckpt_path, device= device)
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


    # loading State Update Model
    if conf.fglobal == 'attn':
        fglobal_dict, _, fg_config, _ = load_model(ckpt_path=conf.fglobal_ckpt_path)
        fg_config = SimpleNamespace(**fg_config)
        fglobal = f_global(
            in_features=fg_config.in_features,
            nhead=fg_config.nhead,
            num_layers=fg_config.nlayer,
            ff_dim=fg_config.ff_dim,
            dropout=fg_config.dropout
        ).to(device)
        fglobal.load_state_dict(fglobal_dict)

        # fglobal = FGlobal(ip_dim= 384 * 3, op_dim= 384).to(device)
        # fglobal.load_state_dict(torch.load(conf.mlp_fglobal_ckpt, weights_only=True, map_location = device))

    else:
        data = torch.load(conf.mlp_fglobal_ckpt, map_location=device)
        fglobal_dict = data['model']
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

    model, actor_optimizer, critic_optimizer, epoch, rl_config = load_policy_model(model, actor_optimizer, critic_optimizer,
                                                                               conf.rl_ckpt_path, device)

    # Loading the attention weights
    attention_data = torch.load(conf.attention_path, map_location= 'cpu')


    # Now Evaluation Starts
    print("HACMIL")
    val_tumor_correctly_classified_hacmil, val_tumor_misclassified_hacmil = evaluate_hacmil(classifier, val_loader, 'Val', device)
    test_tumor_correctly_classified_hacmil, test_tumor_misclassified_hacmil = evaluate_hacmil(classifier, test_loader,'Test', device)

    # For loading time
    for i in range(1) :

        args.time_csv_col_name = f'sasha_iter_{i + 1}'

        # Loading details for the csv file

        # Dataframe
        csv_file = args.time_csv
        required_columns = ['slide_name', args.time_csv_col_name]

        # Step 1: Create the file if it doesn't exist
        if not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=required_columns)
                writer.writeheader()
            print(f"File '{csv_file}' created with headers: {required_columns}")

        # Step 2: File exists – check and add missing columns
        else:
            with open(csv_file, mode='r', newline='', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                existing_columns = reader.fieldnames if reader.fieldnames else []
                data = list(reader)

            # Find missing columns
            missing = [col for col in required_columns if col not in existing_columns]

            if missing:
                print(f"Missing columns found: {missing}. Adding them...")
                # Add missing columns with empty values
                updated_columns = existing_columns + missing
                for row in data:
                    for col in missing:
                        row[col] = ''

                # Write back the updated data with new headers
                with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
                    writer = csv.DictWriter(file, fieldnames=updated_columns)
                    writer.writeheader()
                    writer.writerows(data)
                print(f"File '{csv_file}' updated with missing columns: {missing}")
            else:
                print(f"'{csv_file}' already contains required columns: {required_columns}")

        time_df = pd.read_csv(args.time_csv)


        print("SASHA - Deterministic Policy {Pick Max}")

        val_auc, val_acc, val_f1, val_loss, val_balanced_acc, val_precision, val_recall, patch_indicies_val, val_tumor_correctly_classified_sasha, val_tumor_misclassified_sasha, time_df = evaluate_policy(model, fglobal, classifier, val_loader,'Val', device, epoch, conf,   time_csv = time_df, time_csv_column_name= args.time_csv_col_name)
        test_auc, test_acc, test_f1, test_loss, test_balanced_acc, test_precision, test_recall, patch_indicies_test, test_tumor_correctly_classified_sasha, test_tumor_misclassified_sasha, time_df = evaluate_policy(model, fglobal, classifier, test_loader,'Test', device, epoch, conf, time_csv = time_df, time_csv_column_name = args.time_csv_col_name)

        # Save back
        # time_df.to_csv(args.time_csv, index=False)

    # print("SASHA - Stochastic Policy {Pick from distribution}")
    # val_auc_stoch_ls , val_acc_stoch_ls, val_f1_stoch_ls, val_loss_stoch_ls, val_balanced_acc_stoch_ls, val_precision_stoch_ls, val_recall_stoch_ls = [], [], [], [], [], [], []
    # test_auc_stoch_ls, test_acc_stoch_ls, test_f1_stoch_ls, test_loss_stoch_ls, test_balanced_acc_stoch_ls, test_precision_stoch_ls, test_recall_stoch_ls = [], [], [], [], [], [], []
    #
    # for i in range(1) : # TODO : Need to change it to 10
    #     val_auc_stoch, val_acc_stoch, val_f1_stoch, val_loss_stoch, val_balanced_acc_stoch, val_precision_stoch, val_recall_stoch, _, _, _, _ = evaluate_policy(model, fglobal, classifier, val_loader, 'Val', device, epoch, conf, is_eval= True, is_top_k= True)
    #     test_auc_stoch, test_acc_stoch, test_f1_stoch, test_loss_stoch, test_balanced_acc_stoch, test_precision_stoch, test_recall_stoch, _, _, _, _ = evaluate_policy(model, fglobal, classifier, test_loader, 'Test', device, epoch, conf, is_eval= True, is_top_k= True)
    #
    #     val_auc_stoch_ls.append(val_auc_stoch)
    #     val_acc_stoch_ls.append(val_acc_stoch)
    #     val_f1_stoch_ls.append(val_f1_stoch)
    #     val_loss_stoch_ls.append(val_loss_stoch)
    #     val_balanced_acc_stoch_ls.append(val_balanced_acc_stoch)
    #     val_precision_stoch_ls.append(val_precision_stoch)
    #     val_recall_stoch_ls.append(val_recall_stoch)
    #
    #     test_auc_stoch_ls.append(test_auc_stoch)
    #     test_acc_stoch_ls.append(test_acc_stoch)
    #     test_f1_stoch_ls.append(test_f1_stoch)
    #     test_loss_stoch_ls.append(test_loss_stoch)
    #     test_balanced_acc_stoch_ls.append(test_balanced_acc_stoch)
    #     test_precision_stoch_ls.append(test_precision_stoch)
    #     test_recall_stoch_ls.append(test_recall_stoch)
    #
    # combined_patches = patch_indicies_val | patch_indicies_test
    #
    # # Print Final Evaluation Table
    # print("Stochastic Policy top_k = 0.5 ")
    # print("\nEvaluation Metrics:")
    # print("-" * 85)
    # print(
    #     f"{'Metric':<20} {'Val (Deterministic)':<20} {'Test (Deterministic)':<20} {'Val (Stochastic)':<20} {'Test (Stochastic)':<20}")
    # print("-" * 85)
    #
    # def fmt(metric_list):
    #     return f"{np.mean(metric_list):.4f} ± {np.std(metric_list):.4f}"
    #
    # metrics = [
    #     ("AUC", val_auc, test_auc, fmt(val_auc_stoch_ls), fmt(test_auc_stoch_ls)),
    #     ("Accuracy", val_acc, test_acc, fmt(val_acc_stoch_ls), fmt(test_acc_stoch_ls)),
    #     ("F1 Score", val_f1, test_f1, fmt(val_f1_stoch_ls), fmt(test_f1_stoch_ls)),
    #     ("Loss", val_loss, test_loss, fmt(val_loss_stoch_ls), fmt(test_loss_stoch_ls)),
    #     ("Balanced Acc", val_balanced_acc, test_balanced_acc, fmt(val_balanced_acc_stoch_ls),
    #      fmt(test_balanced_acc_stoch_ls)),
    #     ("Precision", val_precision, test_precision, fmt(val_precision_stoch_ls), fmt(test_precision_stoch_ls)),
    #     ("Recall", val_recall, test_recall, fmt(val_recall_stoch_ls), fmt(test_recall_stoch_ls)),
    # ]
    #
    # for name, val_det, test_det, val_stoch, test_stoch in metrics:
    #     print(f"{name:<20} {val_det:<20.4f} {test_det:<20.4f} {val_stoch:<20} {test_stoch:<20}")
    # print("-" * 85)


    # Now next step is getting when diagram
    classification_mapping = {
        "hacmil_val_c" : set(val_tumor_correctly_classified_hacmil) ,
        "hacmil_val_m" : set(val_tumor_misclassified_hacmil) ,
        "sasha_val_c" : set(val_tumor_correctly_classified_sasha) ,
        "sasha_val_c" : set(val_tumor_misclassified_sasha) ,
        "hacmil_test_c" : set(test_tumor_correctly_classified_hacmil) ,
        "hacmil_test_m" : set(test_tumor_misclassified_hacmil) ,
        "sasha_test_c" : set(test_tumor_correctly_classified_sasha) ,
        "sasha_test_m" : set(test_tumor_misclassified_sasha)
    }

    intersection_results = analyze_tumor_classification_sets(hacmil_test_c= classification_mapping['hacmil_test_c'],
                                      hacmil_test_m= classification_mapping['hacmil_test_m'],
                                      sasha_test_c= classification_mapping['sasha_test_c'],
                                      sasha_test_m= classification_mapping['sasha_test_m'])

    # Print results
    for name, val in intersection_results.items():
        print(f"{name}: {val}")


    # Now Next question is to divide these into different blocks
    # First consider for correctly classification by sasha_test
    correct_fraction_mapping = {}
    misclassified_fraction_mapping = {}

    for slide_name in classification_mapping["sasha_test_c"] & classification_mapping["hacmil_test_c"] :
        correct_fraction_mapping[slide_name] = attention_data[slide_name]['tumor_fraction_side']

    for slide_name in classification_mapping["sasha_test_m"] - classification_mapping["hacmil_test_m"] :
        misclassified_fraction_mapping[slide_name] = attention_data[slide_name]['tumor_fraction_side']

    plot_combined_tumor_fraction_histogram(correct_fraction_mapping, misclassified_fraction_mapping, title = "tumor_fraction_comparison")

    # Initialize bags
    bag1, bag2, bag3 = [], [], []

    # Combine slides from both classifications
    intersect_slides = classification_mapping["sasha_test_c"] & classification_mapping["hacmil_test_c"]

    # Split into bags based on tumor_fraction_slide
    for slide_name in intersect_slides:
        frac = attention_data[slide_name]['tumor_fraction_side']
        print(f"Slide name : {slide_name} and fraction : {frac}")

        if frac < 0.1:
            bag1.append(slide_name)
        elif frac < 0.6:
            bag2.append(slide_name)
        else:
            bag3.append(slide_name)

    print(bag1)
    print(bag2)
    print(bag3)
    exit()

    # Plotting attention weights distribution across slide
    evaluate_attention_weights_new_2(model, fglobal, classifier, test_loader, 'Attention Weights for Correct Tumor Fraction < 0.1', bag1, conf, attention_data)
    evaluate_attention_weights_new_2(model, fglobal, classifier, test_loader, 'Attention Weights for Correct Tumor Fraction 0.1-0.6', bag2, conf, attention_data)
    evaluate_attention_weights_new_2(model, fglobal, classifier, test_loader, 'Attention Weights for Correct Tumor Fraction ≥ 0.6', bag3, conf, attention_data)
    evaluate_attention_weights_new_2(model, fglobal, classifier, test_loader, 'Attention Weights for Mis-classified Tumor Fraction', classification_mapping["sasha_test_m"] - classification_mapping["hacmil_test_m"], conf, attention_data)

    # Run evaluation with appropriate headers
    print("Deterministic Policy")
    evaluate_policy_combined_plot(model, fglobal, classifier, test_loader, 'Deterministic Correct Tumor Fraction < 0.1', bag1, epoch, conf,attention_data)
    evaluate_policy_combined_plot(model, fglobal, classifier, test_loader, 'Deterministic Correct Tumor Fraction 0.1–0.6', bag2, epoch, conf, attention_data)
    evaluate_policy_combined_plot(model, fglobal, classifier, test_loader, 'Deterministic Correct Tumor Fraction ≥ 0.6', bag3, epoch, conf, attention_data)
    evaluate_policy_combined_plot(model, fglobal, classifier, test_loader, 'Deterministic Mis-classified', classification_mapping["sasha_test_m"] - classification_mapping["hacmil_test_m"], epoch, conf, attention_data)


    print("Stochastic Policy")
    evaluate_policy_combined_plot(model, fglobal, classifier, test_loader, 'Stochastic Correct Tumor Fraction < 0.1', bag1, epoch, conf, attention_data, is_eval= False)
    evaluate_policy_combined_plot(model, fglobal, classifier, test_loader,'Stochastic Correct Tumor Fraction 0.1–0.6', bag2, epoch, conf, attention_data, is_eval = False)
    evaluate_policy_combined_plot(model, fglobal, classifier, test_loader, 'Stochastic Correct Tumor Fraction ≥ 0.6', bag3, epoch, conf, attention_data, is_eval = False)
    evaluate_policy_combined_plot(model, fglobal, classifier, test_loader, 'Stochastic Mis-classified', classification_mapping["hacmil_test_m"] - classification_mapping["hacmil_test_m"], epoch, conf, attention_data, is_eval = False)



    conf.writer.close()

def plot_predictions_loss_entropy(prediction, loss, is_tumor, slide_name, save_path=None):
    """
    Plot prediction (class 1), loss, and entropy over time steps.
    Highlights time steps where tumor is True with vertical lines.

    Args:
        prediction (list of list): Each inner list has [class_0_prob, class_1_prob]
        loss (list of float): Loss values per time step
        is_tumor (list of bool): Whether the time step contains tumor or not
        slide_name (str): Name of the slide for saving the plot
        save_path (str, optional): If provided, saves the plot to this path
    """
    time_steps = list(range(len(prediction)))
    second_class_pred = [p[0][1].cpu() for p in prediction]

    # Calculate entropy for each prediction
    def calc_entropy(p):
        p_tensor = torch.tensor(p[0])
        return -(p_tensor * torch.log(p_tensor + 1e-9)).sum().item()

    entropy_vals = [calc_entropy(p) for p in prediction]

    plt.figure(figsize=(18, 5))

    # --- Plot 1: Prediction ---
    plt.subplot(1, 3, 1)
    plt.plot(time_steps, second_class_pred, marker='o', label='Prediction (Class 1)', color='orange')
    for t, tumor in enumerate(is_tumor):
        if tumor:
            plt.axvline(x=t, color='blue', linestyle='--', alpha=0.5,
                        label='Tumor' if t == is_tumor.index(True) else "")
    plt.xlabel("Time Step")
    plt.ylabel("Prediction Probability")
    plt.title("Prediction vs Time Step")
    plt.legend()
    plt.grid(True)

    # --- Plot 2: Loss ---
    plt.subplot(1, 3, 2)
    plt.plot(time_steps, loss, marker='s', color='red', label='Loss')
    for t, tumor in enumerate(is_tumor):
        if tumor:
            plt.axvline(x=t, color='blue', linestyle='--', alpha=0.5,
                        label='Tumor' if t == is_tumor.index(True) else "")
    plt.xlabel("Time Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Time Step")
    plt.legend()
    plt.grid(True)

    # --- Plot 3: Entropy ---
    plt.subplot(1, 3, 3)
    plt.plot(time_steps, entropy_vals, marker='^', color='green', label='Entropy')
    for t, tumor in enumerate(is_tumor):
        if tumor:
            plt.axvline(x=t, color='blue', linestyle='--', alpha=0.5,
                        label='Tumor' if t == is_tumor.index(True) else "")
    plt.xlabel("Time Step")
    plt.ylabel("Entropy")
    plt.title("Entropy vs Time Step")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    final_path = save_path or f"/media/internal_8T/naman/rlogist_dgx/images/{slide_name}_tr2.png"
    plt.savefig(final_path, dpi=300)
    print(f"Plot saved to {final_path}")

def plot_predictions_and_loss(prediction, loss, is_tumor, slide_name, save_path=None):
    """
    Plot prediction for second class and loss over time steps.
    Highlights time steps where tumor is True with vertical lines.

    Args:
        prediction (list of list): Each inner list has [class_0_prob, class_1_prob]
        loss (list of float): Loss values per time step
        is_tumor (list of bool): Whether the time step contains tumor or not
        save_path (str, optional): If provided, saves the plot to this path
    """
    time_steps = list(range(len(prediction)))
    second_class_pred = [p[0][1].cpu() for p in prediction]

    plt.figure(figsize=(12, 5))

    # --- Plot 1: Prediction ---
    plt.subplot(1, 2, 1)
    plt.plot(time_steps, second_class_pred, marker='o', label='Prediction (Class 1)', color='orange')
    for t, tumor in enumerate(is_tumor):
        if tumor:
            plt.axvline(x=t, color='blue', linestyle='--', alpha=0.5,
                        label='Tumor' if t == is_tumor.index(True) else "")

    plt.xlabel("Time Step")
    plt.ylabel("Prediction Probability")
    plt.title("Prediction vs Time Step")
    plt.legend()
    plt.grid(True)

    # --- Plot 2: Loss ---
    plt.subplot(1, 2, 2)
    plt.plot(time_steps, loss, marker='s', color='red', label='Loss')
    for t, tumor in enumerate(is_tumor):
        if tumor:
            plt.axvline(x=t, color='blue', linestyle='--', alpha=0.5,
                        label='Tumor' if t == is_tumor.index(True) else "")

    plt.xlabel("Time Step")
    plt.ylabel("Loss")
    plt.title("Loss vs Time Step")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    save_path = f"/media/internal_8T/naman/rlogist_dgx/images/{slide_name}_bce2"
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

@torch.no_grad()
def evaluate_policy_combined_plot(model, fglobal, classifier, data_loader, header, slide_names, epoch, conf, attention_data, is_eval = True):
    model.eval()

    fraction_loss_map = defaultdict(list)
    fraction_pred_map = defaultdict(list)
    fraction_hit_map = defaultdict(list)
    fraction_entropy_map = defaultdict(list)
    fraction_log_prob = defaultdict(list)

    if is_eval :
        iter = 1
    else :
        iter = 10

    bin_width = 0.005
    def bin_fraction(x): return round(x / bin_width) * bin_width

    for _ in range(iter) :
        for slide_name in slide_names:
            metric_logger = MetricLogger(delimiter=" ")
            for data in metric_logger.log_every(data_loader, 100, header):
                if slide_name != data['slide_name'][0]:
                    continue

                hr_features = data['hr'][0].to(device, dtype=torch.float32)
                state = data['lr'].to(device, dtype=torch.float32)
                label = data['label'].to(device)

                env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf) \
                    if conf.fglobal == 'attn' else \
                    WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

                N = state.shape[1]
                visited_patch_id = []
                done = False
                step = 0

                while not done:
                    action, log_prob, entropy = model.get_action(state, visited_patch_id, is_eval= is_eval)
                    new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier, device=device)

                    slide_preds, _ = classifier.classify(state)
                    probs = F.softmax(slide_preds, dim=1)[0]
                    pred_prob = F.softmax(slide_preds, dim=1)[0][1].item()  # Tumor probability
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

                    fraction = step / N
                    binned_frac = bin_fraction(fraction)

                    fraction_loss_map[binned_frac].append(-1 * reward)
                    fraction_pred_map[binned_frac].append(pred_prob)
                    fraction_entropy_map[binned_frac].append(entropy)
                    fraction_log_prob[binned_frac].append(log_prob)

                    is_tumor = attention_data[slide_name]['lr_patch_tumor_area_percentage'][int(action)] > 0
                    fraction_hit_map[binned_frac].append(1 if is_tumor else 0)

                    visited_patch_id.append(action.item())
                    state = new_state
                    step += 1
                break

    # Compute stats
    def get_stats(fmap):
        keys = sorted(fmap.keys())
        means = [np.mean(fmap[k]) for k in keys]
        stds = [np.std(fmap[k]) for k in keys]
        return keys, means, stds

    frac_keys, avg_losses, std_losses = get_stats(fraction_loss_map)
    _, avg_preds, std_preds = get_stats(fraction_pred_map)
    _, avg_hits, _ = get_stats(fraction_hit_map)
    _, avg_entropy, std_entropy = get_stats(fraction_entropy_map)
    _, avg_prob_actions, std_prob_actions = get_stats(fraction_log_prob)

    # ---- Plot 1: Loss ----
    plt.figure(figsize=(10, 5))
    plt.plot(frac_keys, avg_losses, label="Avg Loss", color='red')
    plt.fill_between(frac_keys,
                     np.clip(np.array(avg_losses) - np.array(std_losses), 0, None),
                     np.array(avg_losses) + np.array(std_losses),
                     color='red', alpha=0.2)
    plt.xlabel("Visited Fraction of Slide (step / N)")
    plt.ylabel("Loss")
    plt.title("Loss vs Visited Fraction")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/loss_fraction_epoch_{header}.png", dpi=300)
    plt.show()

    # ---- Plot 2: Combined Entropy + Tumor Prediction + Tumor Patch Hits ----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot tumor prediction probability
    ax1.plot(frac_keys, avg_preds, color='blue', label="Average Tumor Prediction Class Probability", linewidth=2)
    ax1.fill_between(frac_keys,
                     np.array(avg_preds) - np.array(std_preds),
                     np.array(avg_preds) + np.array(std_preds),
                     color='blue', alpha=0.2)

    # Plot entropy on same axis
    ax1.plot(frac_keys, avg_entropy, color='green', label="Average Entropy", linewidth=2)
    ax1.fill_between(frac_keys,
                     np.clip(np.array(avg_entropy) - np.array(std_entropy), 0, None),
                     np.clip(np.array(avg_entropy) + np.array(std_entropy), 0, None),
                     color='green', alpha=0.2)

    ax1.set_xlabel("Visited Fraction of Slide (step / N)")
    ax1.set_ylabel("Tumor Class Prediction Probability / Entropy")
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc="upper left")

    # Plot tumor patch frequency as bars on secondary y-axis
    ax2 = ax1.twinx()
    ax2.bar(frac_keys, avg_hits, width=bin_width * 0.8, color='orange', alpha=0.3, label='Tumor Patch Hits')
    ax2.set_ylabel("Tumor Patch Hit Ratio", color='orange')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='y', labelcolor='orange')

    fig.suptitle("Tumor Prediction Probability, Entropy & Tumor Patch Hit Ratio", fontsize=14)
    fig.tight_layout()
    plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/combined_pred_entropy_hits_epoch_{header}.png", dpi=300)
    plt.show()

    # # ---- Plot 2: Prediction + Tumor Patch Hits ----
    # fig, ax1 = plt.subplots(figsize=(10, 5))
    #
    # ax1.plot(frac_keys, avg_preds, color='blue', label="Avg Tumor Prediction")
    # ax1.fill_between(frac_keys,
    #                  np.array(avg_preds) - np.array(std_preds),
    #                  np.array(avg_preds) + np.array(std_preds),
    #                  color='blue', alpha=0.2)
    # ax1.set_xlabel("Visited Fraction of Slide (step / N)")
    # ax1.set_ylabel("Tumor Prediction", color='blue')
    # ax1.set_ylim(0, 1)
    # ax1.tick_params(axis='y', labelcolor='blue')
    #
    # ax2 = ax1.twinx()
    # ax2.bar(frac_keys, avg_hits, width=bin_width * 0.8, color='orange', alpha=0.3, label='Tumor Patch Frequency')
    # ax2.set_ylabel("Tumor Patch Hits", color='orange')
    # ax2.set_ylim(0, 1)
    # ax2.tick_params(axis='y', labelcolor='orange')
    #
    # fig.suptitle("Tumor Prediction and Patch Hit Frequency")
    # fig.tight_layout()
    # plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/prediction_tumor_hits_epoch_{header}.png", dpi=300)
    # plt.show()
    #
    # # ---- Plot 3: Entropy + Tumor Patch Hits ----
    # fig, ax1 = plt.subplots(figsize=(10, 5))
    #
    # # Entropy curve
    # frac_keys, avg_entropy, std_entropy = get_stats(fraction_entropy_map)
    # ax1.plot(frac_keys, avg_entropy, color='green', label="Avg Entropy")
    # ax1.fill_between(frac_keys,
    #                  np.clip(np.array(avg_entropy) - np.array(std_entropy), 0, None),
    #                  np.clip(np.array(avg_entropy) + np.array(std_entropy), 0, None),
    #                  color='green', alpha=0.2)
    # ax1.set_ylim(0, 0.7)
    # ax1.set_xlabel("Visited Fraction of Slide (step / N)")
    # ax1.set_ylabel("Entropy", color='green')
    # ax1.tick_params(axis='y', labelcolor='green')
    #
    # # Tumor patch hit bars
    # ax2 = ax1.twinx()
    # ax2.bar(frac_keys, avg_hits, width=bin_width * 0.8, color='orange', alpha=0.3, label='Tumor Patch Frequency')
    # ax2.set_ylabel("Tumor Patch Hits", color='orange')
    # ax2.set_ylim(0, 1)
    # ax2.tick_params(axis='y', labelcolor='orange')
    #
    # fig.suptitle("Prediction Entropy and Patch Hit Frequency")
    # fig.tight_layout()
    # plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/entropy_tumor_hits_epoch_{header}.png", dpi=300)
    # plt.show()


@torch.no_grad()
def evaluate_policy_combined_plot_2(model, fglobal, classifier, data_loader, header, slide_names, epoch, conf, attention_data, is_eval=True):
    model.eval()

    fraction_loss_map = defaultdict(list)
    fraction_pred_map = defaultdict(list)
    fraction_hit_map = defaultdict(list)
    fraction_entropy_map = defaultdict(list)

    per_slide_pred_map = {}
    per_slide_entropy_map = {}

    iter = 1 if is_eval else 10
    bin_width = 0.005
    def bin_fraction(x): return round(x / bin_width) * bin_width

    for _ in range(iter):
        for slide_name in slide_names:
            metric_logger = MetricLogger(delimiter=" ")
            for data in metric_logger.log_every(data_loader, 100, header):
                if slide_name != data['slide_name'][0]:
                    continue

                hr_features = data['hr'][0].to(device, dtype=torch.float32)
                state = data['lr'].to(device, dtype=torch.float32)
                label = data['label'].to(device)

                env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf) \
                    if conf.fglobal == 'attn' else \
                    WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

                N = state.shape[1]
                visited_patch_id = []
                done = False
                step = 0

                slide_fraction_preds = {}
                slide_fraction_entropy = {}

                while not done:
                    action, log_prob, entropy_val = model.get_action(state, visited_patch_id, is_eval=is_eval)
                    new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier, device=device)

                    slide_preds, _ = classifier.classify(state)
                    probs = F.softmax(slide_preds, dim=1)[0]
                    pred_prob = probs[1].item()  # Tumor probability
                    entropy = -torch.sum(probs * torch.log(probs + 1e-8)).item()

                    fraction = step / N
                    binned_frac = bin_fraction(fraction)

                    # Store for averages
                    fraction_loss_map[binned_frac].append(-1 * reward)
                    fraction_pred_map[binned_frac].append(pred_prob)
                    fraction_entropy_map[binned_frac].append(entropy)

                    is_tumor = attention_data[slide_name]['lr_patch_tumor_area_percentage'][int(action)] > 0
                    fraction_hit_map[binned_frac].append(1 if is_tumor else 0)

                    # Store per-slide data
                    slide_fraction_preds.setdefault(binned_frac, []).append(pred_prob)
                    slide_fraction_entropy.setdefault(binned_frac, []).append(entropy)

                    visited_patch_id.append(action.item())
                    state = new_state
                    step += 1

                # Average per-slide entropy and prediction for this slide
                per_slide_pred_map[slide_name] = {k: np.mean(v) for k, v in slide_fraction_preds.items()}
                per_slide_entropy_map[slide_name] = {k: np.mean(v) for k, v in slide_fraction_entropy.items()}

                break  # Process only one matching slide per iteration

    # Compute stats
    def get_stats(fmap):
        keys = sorted(fmap.keys())
        means = [np.mean(fmap[k]) for k in keys]
        stds = [np.std(fmap[k]) for k in keys]
        return keys, means, stds

    frac_keys, avg_losses, std_losses = get_stats(fraction_loss_map)
    _, avg_preds, std_preds = get_stats(fraction_pred_map)
    _, avg_hits, _ = get_stats(fraction_hit_map)
    _, avg_entropy, std_entropy = get_stats(fraction_entropy_map)

    # ---- Plot 1: Loss ----
    plt.figure(figsize=(10, 5))
    plt.plot(frac_keys, avg_losses, label="Avg Loss", color='red')
    plt.fill_between(frac_keys,
                     np.clip(np.array(avg_losses) - np.array(std_losses), 0, None),
                     np.array(avg_losses) + np.array(std_losses),
                     color='red', alpha=0.2)
    plt.xlabel("Visited Fraction of Slide (step / N)")
    plt.ylabel("Loss")
    plt.title("Loss vs Visited Fraction")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0, 8)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/loss_fraction_epoch_{header}.png", dpi=300)
    plt.show()

    # ---- Plot 2: Combined Entropy + Tumor Prediction + Tumor Patch Hits ----
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Light blue per-slide tumor prediction lines
    for slide_data in per_slide_pred_map.values():
        x = sorted(slide_data.keys())
        y = [slide_data[k] for k in x]
        ax1.plot(
            x, y,
            color='#1f77b4',  # darker, matplotlib blue
            alpha=0.8,
            linewidth=1.8,
            linestyle='--'
        )

    # Improved green for per-slide entropy lines
    for slide_data in per_slide_entropy_map.values():
        x = sorted(slide_data.keys())
        y = [slide_data[k] for k in x]
        ax1.plot(
            x, y,
            color='#2ca02c',  # darker, matplotlib green
            alpha=0.8,
            linewidth=1.8,
            linestyle='--'
        )

    # Bold average tumor prediction
    ax1.plot(frac_keys, avg_preds, color='blue', label="Average Tumor Prediction Class Probability", linewidth=2)
    ax1.fill_between(frac_keys,
                     np.array(avg_preds) - np.array(std_preds),
                     np.array(avg_preds) + np.array(std_preds),
                     color='blue', alpha=0.2)

    # Bold average entropy
    ax1.plot(frac_keys, avg_entropy, color='green', label="Average Entropy", linewidth=2)
    ax1.fill_between(frac_keys,
                     np.clip(np.array(avg_entropy) - np.array(std_entropy), 0, None),
                     np.clip(np.array(avg_entropy) + np.array(std_entropy), 0, None),
                     color='green', alpha=0.2)

    ax1.set_xlabel("Visited Fraction of Slide (step / N)")
    ax1.set_ylabel("Tumor Class Prediction Probability / Entropy")
    ax1.set_ylim(0, 1.0)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend(loc="upper left")

    # Tumor patch hit ratio bar plot
    ax2 = ax1.twinx()
    ax2.bar(frac_keys, avg_hits, width=bin_width * 0.8, color='orange', alpha=0.3, label='Tumor Patch Hits')
    ax2.set_ylabel("Tumor Patch Hit Ratio", color='orange')
    ax2.set_ylim(0, 1.0)
    ax2.tick_params(axis='y', labelcolor='orange')

    fig.suptitle("Tumor Prediction Probability, Entropy & Tumor Patch Hit Ratio", fontsize=14)
    fig.tight_layout()
    plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/combined_pred_entropy_hits_epoch_2_{header}.png", dpi=300)
    plt.show()


@torch.no_grad()
def evaluate_policy_2(model, fglobal, classifier, data_loader, header, slide_name, epoch, conf, attention_data):

    model.eval()

    # At each time step
    loss = []
    prediction = []
    is_tumor = []

    metric_logger = MetricLogger(delimiter=" ")
    for data in metric_logger.log_every(data_loader, 100, header):

        if slide_name != data['slide_name'][0] :
            continue

        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
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
            action, log_prob, entropy = model.get_action(state, visited_patch_id, is_eval = True)
            new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier,
                                               device=device)
            slide_preds, _ = classifier.classify(state)
            # visited_patch_id.append(action)

            # Storing details at each time step --->
            loss.append(-1 * reward)
            prediction.append(F.softmax(slide_preds, dim=1))
            # Now storing whether the patch is tumor or normal
            if attention_data[slide_name]['lr_patch_tumor_area_percentage'][int(action)] > 0 :
                is_tumor.append(True)
            else :
                is_tumor.append(False)
            state = new_state

        plot_predictions_loss_entropy(prediction, loss, is_tumor, slide_name)

        # Compute the hit ratio
        print(f"Slide Name - {slide_name} , Hit Ratio : {(sum(is_tumor) / min(math.ceil(0.2 * N), len([i for i in attention_data[slide_name]['lr_patch_tumor_area_percentage'] if i > 0  ]) )) * 100}")

@torch.no_grad()
def evaluate_policy(model, fglobal, classifier, data_loader, header, device, epoch, conf, is_eval = True, is_top_k = False, is_top_p = False, time_csv = None, time_csv_column_name = None ):

    print("SASHA")
    print("--" * 50)

    model.eval()

    y_pred = []
    y_true = []
    slide_names = []
    metric_logger = MetricLogger(delimiter=" ")
    final_reward = 0
    patches_idx = {}

    for data in metric_logger.log_every(data_loader, 100, header):

        start_time = time.time()

        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
        slide_id = data['slide_name'][0]
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
            action, log_prob, entropy = model.get_action(state, visited_patch_id, is_eval = is_eval, is_top_k= is_top_k, is_top_p= is_top_p)
            new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier,
                                               device=device)
            state = new_state
            visited_patch_id.append(action.item())

        final_reward += reward
        loss = -1 * reward
        slide_preds, attn = classifier.classify(state)
        pred = torch.softmax(slide_preds, dim=-1)
        acc1 = accuracy(pred, label, topk=(1,))[0]

        end_time = time.time()

        total_time = end_time - start_time

        if time_csv is not None :
            # Check if slide_id exists
            if slide_id in time_csv['slide_name'].values:
                time_csv.loc[time_csv['slide_name'] == slide_id, time_csv_column_name] = total_time
            else:
                # Create new row
                new_row = {col: "" for col in time_csv.columns}
                new_row["slide_name"] = slide_id
                new_row[time_csv_column_name] = total_time
                time_csv = pd.concat([time_csv, pd.DataFrame([new_row])], ignore_index=True)

        metric_logger.update(loss=loss)
        metric_logger.meters['acc1'].update(acc1.item(), n=label.shape[0])

        y_pred.append(pred)
        y_true.append(label)
        slide_names.append(slide_id)
        patches_idx[slide_id] = visited_patch_id.copy()

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


    # Get slide names where the slide is correctly predicted or falsely predicted

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

    # Print results
    print(f"Total slides with true label 1: {len(true_label_1_indices)}")
    print(f"Correctly classified: {len(correctly_classified)}")
    print(f"Misclassified: {len(misclassified)}")

    # Optionally print names
    print("\n✅ Correctly Classified Slides:")
    print(correctly_classified)

    print("\n❌ Misclassified Slides:")
    print(misclassified)

    return auroc, metric_logger.acc1.global_avg, f1_score, metric_logger.loss.global_avg, balanced_acc, precision, recall, patches_idx, correctly_classified, misclassified, time_csv

@torch.no_grad()
def evaluate_hacmil(model, data_loader, header, device):

    print("HACMIL")
    print("--" * 50)
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
        acc1 = accuracy(pred, label, topk=(1,))[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=label.shape[0])

        y_pred.append(pred)
        y_true.append(label)
        slide_names.append(slide_name)

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

    print('* Acc@1 {top1.global_avg:.3f}  auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1, AUROC=auroc, F1=f1_score))

    # Get slide names where the slide is correctly predicted or falsely predicted

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

    # Print results
    print(f"Total slides with true label 1: {len(true_label_1_indices)}")
    print(f"Correctly classified: {len(correctly_classified)}")
    print(f"Misclassified: {len(misclassified)}")

    # Optionally print names
    print("\n✅ Correctly Classified Slides:")
    print(correctly_classified)

    print("\n❌ Misclassified Slides:")
    print(misclassified)

    return correctly_classified, misclassified


@torch.no_grad()
def evaluate_attention_weights(model,  data_loader, header,  slide_names, conf, attention_data):

    # === Transform X so that 0.0–0.8 -> 0.0–0.2 and 0.8–1.0 -> 0.2–1.0 ===
    def compress_tail_x(x):
        return np.where(x < 0.8, x * (0.2 / 0.8), 0.2 + ((x - 0.8) * (0.8 / 0.2)))

    # === Collect all slide curves ===
    slide_curves = []

    model.eval()
    for slide_name in slide_names:
        metric_logger = MetricLogger(delimiter=" ")
        for data in metric_logger.log_every(data_loader, 100, header):
            if slide_name != data['slide_name'][0]:
                continue

            hr_features = data['hr'].to(device, dtype=torch.float32)
            attention_weights = model.classify(hr_features, average_block2_weights=True)
            attention_weights = attention_weights.squeeze().detach().cpu().numpy()

            sorted_weights = np.sort(attention_weights)
            n = len(sorted_weights)
            x_fraction = np.linspace(0, 1, n)

            slide_curves.append((x_fraction, sorted_weights))

    # === Plot with compressed X scale ===
    plt.figure(figsize=(10, 6))

    for x, y in slide_curves:
        plt.plot(compress_tail_x(x), y, alpha=0.5, linewidth=1)

    # Customize X ticks to look meaningful
    xticks_orig = [0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0]
    xtick_labels = [str(v) for v in xticks_orig]
    xtick_positions = compress_tail_x(np.array(xticks_orig))
    plt.xticks(xtick_positions, xtick_labels)

    plt.xlabel('Fraction of patches (nonlinear scale)')
    plt.ylabel('Softmax attention weight')
    plt.title('Attention Curves Per Slide (High Attention Region Expanded)')
    plt.ylim(0, 0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/attention_weights_focus_tail_{header}.png")
    plt.show()


@torch.no_grad()
def evaluate_attention_weights_new(model, fglobal, classifier, data_loader, header,  slide_names, conf, attention_data):

    # === Transform X so that 0.0–0.8 -> 0.0–0.2 and 0.8–1.0 -> 0.2–1.0 ===
    def compress_tail_x(x):
        return np.where(x < 0.8, x * (0.2 / 0.8), 0.2 + ((x - 0.8) * (0.8 / 0.2)))

    # === Collect all slide curves ===
    slide_curves = []

    model.eval()
    for slide_name in slide_names:
        metric_logger = MetricLogger(delimiter=" ")
        for data in metric_logger.log_every(data_loader, 100, header):
            if slide_name != data['slide_name'][0]:
                continue

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
                action, log_prob, entropy = model.get_action(state, visited_patch_id, is_eval=True)
                new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier,
                                                   device=device)
                state = new_state
                visited_patch_id.append(action.item())

            attention_weights = classifier.classify(state, average_block2_weights=True)
            attention_weights = attention_weights.squeeze().detach().cpu().numpy()
            lr_path_tumor_area_percentage = attention_data[slide_name]['lr_path_tumor_area_percentage']


            sorted_weights = np.sort(attention_weights)
            n = len(sorted_weights)
            x_fraction = np.linspace(0, 1, n)

            slide_curves.append((x_fraction, sorted_weights))

    # === Plot with compressed X scale ===
    plt.figure(figsize=(10, 6))

    for x, y in slide_curves:
        plt.plot(compress_tail_x(x), y, alpha=0.5, linewidth=1)

    # Customize X ticks to look meaningful
    xticks_orig = [0.0, 0.2, 0.4, 0.6, 0.8, 0.85, 0.9, 0.95, 1.0]
    xtick_labels = [str(v) for v in xticks_orig]
    xtick_positions = compress_tail_x(np.array(xticks_orig))
    plt.xticks(xtick_positions, xtick_labels)

    plt.xlabel('Fraction of patches (nonlinear scale)')
    plt.ylabel('Softmax attention weight')
    plt.title('Attention Curves Per Slide (High Attention Region Expanded)')
    plt.ylim(0, 0.5)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/attention_weights_focus_tail_2_{header}.png")
    plt.show()

@torch.no_grad()
def evaluate_attention_weights_new_2(model, fglobal, classifier, data_loader, header, slide_names, conf, attention_data):
    # === Collect all slide curves ===
    slide_curves = []

    model.eval()
    for slide_name in slide_names:
        metric_logger = MetricLogger(delimiter=" ")
        for data in metric_logger.log_every(data_loader, 100, header):
            if slide_name != data['slide_name'][0]:
                continue

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
                action, log_prob, entropy = model.get_action(state, visited_patch_id, is_eval=True)
                new_state, reward, done = env.step(
                    action=action,
                    state_update_net=fglobal,
                    classifier_net=classifier,
                    device=device
                )
                state = new_state
                visited_patch_id.append(action.item())

            attention_weights = classifier.classify(state, average_block2_weights=True)
            attention_weights = attention_weights.squeeze().detach().cpu().numpy()

            lr_path_tumor_area_percentage = attention_data[slide_name]['lr_patch_tumor_area_percentage']
            tumor_area_percentages = np.array(lr_path_tumor_area_percentage)

            slide_curves.append((tumor_area_percentages, attention_weights))

    # === Plot Scatter: Attention Weight vs Tumor Area Percentage ===
    plt.figure(figsize=(10, 6))

    for tumor_areas, att_weights in slide_curves:
        colors = ['red' if t > 0 else 'blue' for t in tumor_areas]
        plt.scatter(
            tumor_areas,
            att_weights,
            c=colors,
            alpha=0.5,
            edgecolors='k',
            linewidths=0.2,
            s=20
        )

    plt.xlabel('Tumor Area Percentage (per patch)')
    plt.ylabel('Softmax Attention Weight')
    plt.title('Attention Weight vs Tumor Area Percentage')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 100)
    plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/attention_scatter_tumor_vs_weight_{header}.png")
    plt.show()

    # === Plot Scatter: Attention Weight vs Tumor Area Percentage (Tumor Only) ===
    plt.figure(figsize=(10, 6))
    for tumor_areas, att_weights in slide_curves:
        tumor_only = np.array(tumor_areas) > 0
        if np.any(tumor_only):
            filtered_areas = np.array(tumor_areas)[tumor_only]
            filtered_weights = np.array(att_weights)[tumor_only]
            plt.scatter(
                filtered_areas,
                filtered_weights,
                c='red',
                alpha=0.6,
                edgecolors='k',
                linewidths=0.2,
                s=20
            )

    plt.xlabel('Tumor Area Percentage (per patch)')
    plt.ylabel('Softmax Attention Weight')
    plt.title('Attention Weight vs Tumor Area Percentage (Tumor Patches Only)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 100)
    plt.ylim(0, 0.5)
    plt.tight_layout()
    plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/attention_scatter_tumor_only_{header}.png")
    plt.show()

from matplotlib import pyplot as plt

def analyze_tumor_classification_sets(
    hacmil_test_c, hacmil_test_m, sasha_test_c, sasha_test_m
):
    """
    Plots Venn diagrams and returns intersections of tumor classification results.

    Parameters:
        hacmil_test_c (set): Correctly classified by HACMIL
        hacmil_test_m (set): Misclassified by HACMIL
        sasha_test_c (set): Correctly classified by SASHA
        sasha_test_m (set): Misclassified by SASHA

    Returns:
        dict: Dictionary of intersection sets
    """

    # Return intersections
    intersections = {
        "hacmil_correct ∩ sasha_correct": hacmil_test_c & sasha_test_c,
        "hacmil_correct ∩ sasha_mis": hacmil_test_c & sasha_test_m,
        "hacmil_mis ∩ sasha_correct": hacmil_test_m & sasha_test_c,
        "hacmil_mis ∩ sasha_mis": hacmil_test_m & sasha_test_m,
    }

    return intersections

def plot_combined_tumor_fraction_histogram(correct_mapping, misclassified_mapping, bins=20, title=None):
    """
    Plots a combined histogram of tumor fractions for correctly and misclassified slides.

    Parameters:
        correct_mapping (dict): slide_name -> tumor_fraction for correct predictions.
        misclassified_mapping (dict): slide_name -> tumor_fraction for misclassifications.
        bins (int): Number of bins in the histogram.
        title (str): Title for the plot.
    """
    correct_fractions = list(correct_mapping.values())
    misclassified_fractions = list(misclassified_mapping.values())

    plt.figure(figsize=(10, 6))
    plt.hist(
        [correct_fractions, misclassified_fractions],
        bins=bins,
        range=(0.0, 1.0),
        color=['green', 'red'],
        label=['Correctly Classified', 'Misclassified'],
        alpha=0.7,
        edgecolor='black',
    )

    plt.xlabel('Tumor Fraction')
    plt.ylabel('Number of Slides')
    plt.title('Tumor Fraction Histogram for Test Slides')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = title.replace(" ", "_") if title else "tumor_fraction_comparison"
    plt.savefig(f"/media/internal_8T/naman/rlogist_dgx/images/{filename}_bins_{bins}.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    # Adding Device Details
    gpus = check_gpu_availability(5, 1, [])
    print(f"occupied {gpus}")
    device = torch.device(f"cuda:{gpus[0]}")
    main()