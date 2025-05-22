"""
This is the complete loop, where we are doing feature extraction for selective zooming, and running the inference loop
Description:
This script is used to run inference using the SASHA pipeline (versions 0.1 and 0.2).
It assumes that all the required components — HAFED, TSU, and RL — have been trained beforehand.

Specifically, this script loads the trained models, performs inference over the target dataset,
and generates predictions according to the specified SASHA configuration.

Supported SASHA Variants:
- SASHA-0.1
- SASHA-0.2

Make sure to configure the required paths to checkpoints and datasets before execution.


Command
python step7_inference_with_fe.py --config config/camelyon_sasha_inference_with_fe.yml --seed 4 --save_dir SAVE_DIR

"""

import argparse
import json
import os
import time
from collections import defaultdict
from types import SimpleNamespace

import torch
import yaml
from sklearn.metrics import balanced_accuracy_score

from architecture.transformer import HAFED
from envs.WSI_cosine_env_inference import WSICosineObservationEnv_inference
from tqdm import tqdm

from datasets.dataset_h5 import Dataset_All_Bags
from modules.fglobal_mlp import FGlobal
from rl_algorithms.ppo import Agent, Actor, Critic
from utils.gpu_utils import check_gpu_availability
from utils.inference_utils import Helper
from utils.utils import Struct
import torchmetrics


def get_arguments():

    parser = argparse.ArgumentParser('Inference with Feature Extraction', add_help=False)

    # Patching arguments
    parser.add_argument('--step_size', type=int, default=256, help='step_size')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--extension', default='tif', help='extension to processes data type, e.g. *.svs, *.tif')

    # Feature Extraction arguments
    parser.add_argument('--batch_size_hr', type=int, default=1)
    parser.add_argument('--batch_size_lr', type=int, default=512)
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--slide_ext', type=str, default="tif", help="we have two options tif, svs, or any other compatible can work")
    parser.add_argument('--extract_high_res_features', type=bool, default=True, help="To create a mapping from high resolution to low resolution")
    parser.add_argument('--patch_level_low_res', type=int, default=3)  # Low  represents the magnified level [ Just Make sure that patch level should match from create patches ]
    parser.add_argument('--patch_level_high_res', type=int, default=1)  # High represents the scanning level

    # RL Models
    parser.add_argument('--classifier_arch', default='hafed', help='choice of architecture for HAFED')
    parser.add_argument('--config', default=None, type=str, help='config file path')
    parser.add_argument('--seed', type=int, default=4, help='set the random seed')
    parser.add_argument('--save_dir', default=None, help= 'folder path to save intermediate steps')


    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args


@torch.no_grad()
def load_agent(conf):
    dict = torch.load(conf.rl_ckpt_path, map_location=conf.device)
    config = dict['config']
    model_dict = dict['model']
    return model_dict, config


@torch.no_grad()
def load_model(ckpt_path, device):
    dict = torch.load(ckpt_path, map_location=device)
    config = dict['config']
    model_dict = dict['model']
    return model_dict, config

@torch.no_grad()
def get_classifier(conf):
    classifier_dict, config = load_model(conf.classifier_ckpt_path, conf.device)
    classifier_conf = SimpleNamespace(**config)
    classifier = HAFED(classifier_conf, n_token_1=classifier_conf.n_token_1, n_token_2=classifier_conf.n_token_2, n_masked_patch_1=classifier_conf.n_masked_patch_1, n_masked_patch_2=classifier_conf.n_masked_patch_2, mask_drop=classifier_conf.mask_drop)
    classifier.to(conf.device)
    classifier.load_state_dict(classifier_dict)
    classifier.eval()
    return classifier

@torch.no_grad()
def get_tsu(conf):
    fglobal_dict = torch.load(conf.mlp_fglobal_ckpt, map_location=conf.device)['model']
    fglobal = FGlobal().to(conf.device)
    fglobal.load_state_dict(fglobal_dict)
    fglobal.eval()
    return fglobal

@torch.no_grad()
def evaluate(conf):

    #df to get the true label
    bags_dataset = Dataset_All_Bags(conf.csv_path)
    df = bags_dataset.df.set_index('slide_id')
    
    # Loading RL agent
    agent_dict, config = load_agent(conf)
    agent_conf = SimpleNamespace(**config)
    actor = Actor(conf=agent_conf)
    critic  = Critic(conf=agent_conf)
    agent = Agent(actor, critic, agent_conf).to(conf.device)
    agent.load_state_dict(agent_dict)
    agent.to(conf.device)
    agent.eval()

    # Loading Classifier (feature aggregator + classifier)
    classifier = get_classifier(conf)

    # Loading cosine similarity based target state updater
    tsu = get_tsu(conf)

    # Initializing helper class for inference
    helper = Helper(conf, classifier)


    # Now load the train, validation, test slides
    split_file_path = './dataset_csv/%s/splits/split_%s.json' % (conf.dataset, conf.seed)
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']

    else:
        raise Exception(f"Please enter a valid split seed for dataset - {conf.dataset} ")

    time_dict = defaultdict(list)
    y_pred = []
    y_true = []
    y_prob = []

    for slide in test_names:

        start_time = time.time()

        # STEP1 PRE-PROCESSING AND CREATING PATCHES
        print(f"Evaluating slide: {slide}")
        print(f"Started Step1: Creating Patches")
        helper.create_patches(slide, conf.patch_level_low_res, conf.step_size, conf.patch_size)
        print(f"Completed Step1: Creating Patches")


        #STEP2 FEATURE EXTRACTION AT LOW RESOLUTION
        true_label = df.loc[slide]['label']
        print(f"Started Step2: Extraction Features")
        feature, coords, wsi = helper.extract_lr_features(slide, true_label, conf.slide_ext, conf.patch_level_low_res, conf.patch_level_high_res)
        print(f"Completed Step2: Extraction Features")


        # Step3 RL agent evaluation
        print(f"Started Step3: Sampling relevant patches using RL")
        state = torch.tensor(feature, dtype=torch.float32, device=conf.device).unsqueeze(0)
        true_label = torch.tensor([true_label]).to(conf.device)
        env = WSICosineObservationEnv_inference(lr_features=state, device=conf.device, frac_visit = conf.frac_visit, cosine_threshold= conf.cosine_threshold)
        N = state.shape[1]
        visited_patch_id = []
        visited_patch_coords = []
        done = False
        pbar = tqdm(total=int(conf.frac_visit*N), desc=f"{slide} RL Sampling Steps", leave=False)
        while not done:
            action, log_prob, entropy = agent.get_action(state, visited_patch_id, is_eval=True)
            visited_patch_id.append(action.item())

            # Zoom in to the selected action and extract featues for k patches in high resolution
            selected_coords = coords[action.item()]
            visited_patch_coords.append(selected_coords)
            v_at = helper.get_embedding(selected_coords, wsi, conf.patch_level_low_res, conf.patch_level_high_res, conf.step_size, conf.patch_size)
            state, done = env.step(action=action.item(),
                                       v_at=v_at,
                                       state_update_net=tsu,
                                       classifier_net=classifier,
                                       device=conf.device)
            pbar.update(1)

        pbar.close()
        slide_preds, attn = classifier.classify(state)
        pred = torch.softmax(slide_preds, dim=-1)
        y_hat = torch.argmax(pred)
        y_pred.append(y_hat.item())
        y_true.append(true_label)
        y_prob.append(pred)

        # Just to verify the values
        # print(y_pred)
        # print([y.item() for y in y_true])
        
        end_time = time.time()

        time_dict[f'{slide}'].append(end_time - start_time)
        print(f"Slide time : {time_dict[f'{slide}']}")

    y_pred = torch.cat(y_prob, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_pred_labels = torch.argmax(y_pred, dim=-1)

    Accuracy_metric = torchmetrics.Accuracy(task='binary').to(conf.device)
    Accuracy_metric(y_pred_labels, y_true)
    accuracy = Accuracy_metric.compute().item()

    AUROC_metric = torchmetrics.AUROC(task='binary').to(conf.device)
    AUROC_metric(y_pred[:, 1], y_true)
    auroc = AUROC_metric.compute().item()

    F1_metric = torchmetrics.F1Score(task='binary').to(conf.device)
    F1_metric(y_pred_labels, y_true)
    f1_score = F1_metric.compute().item()

    Precision_metric = torchmetrics.Precision(task='binary').to(conf.device)
    Precision_metric(y_pred_labels, y_true)
    precision = Precision_metric.compute().item()

    Recall_metric = torchmetrics.Recall(task='binary').to(conf.device)
    Recall_metric(y_pred_labels, y_true)
    recall = Recall_metric.compute().item()

    y_pred_np = y_pred_labels.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    balanced_acc = balanced_accuracy_score(y_true_np, y_pred_np)

    print(f"{'Phase':<6} | {'Acc':<6} | {'AUROC':<6} | {'F1':<6} | {'Precision':<9} | {'Recall':<6} | {'Balanced Acc':<13} ")
    print("-" * 110)
    print(f"{'Test':<6}  | {accuracy:.4f}  | {auroc:.4f}  | {f1_score:.4f}  | {precision:.4f}  | {recall:.4f}  | {balanced_acc:.4f}")

    total_time = []
    for key in time_dict.keys():
        total_time.append(sum(time_dict[key]))
    print(f"Average : {sum(total_time) / len(total_time)}")

    torch.save(time_dict, os.path.join(conf.save_dir, f"time_dict_sasha_{conf.frac_visit}.pt"))


if __name__ == '__main__':

    args = get_arguments()

    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    evaluate(conf)