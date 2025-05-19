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

"""

import argparse
import json
import os
import time
from collections import defaultdict
from pprint import pprint
from types import SimpleNamespace

import openslide
import pandas as pd
import torch
import torchmetrics
import yaml
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from envs.WSI_cosine_env_inference import WSICosineObservationEnv
from architecture.transformer import ACMIL_GA
from architecture.transformer import HAFED
from datasets.dataset_h5 import Whole_Slide_Bag_FP
from models_features_extraction import get_encoder
from modules.fglobal_mlp import FGlobal
from rl_algorithms.ppo import Agent, Actor, Critic
from step4_extract_intermediate_features import load_model
from step7_inference import load_policy_model
from utils.gpu_utils import check_gpu_availability
from utils.utils import Struct, set_seed
from step2_extract_features import compute_w_loader


def get_arguments():
    parser = argparse.ArgumentParser('SASHA inference with feature extraction', add_help=False)

    # Patching arguments ---->
    parser.add_argument('--step_size', type=int, default=256, help='step_size')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--extension', default='tif', help='extension to processes data type, e.g. *.svs, *.tif')
    parser.add_argument('--patch_level', type=int, default=3, help='downsample level at which to patch')

    parser.add_argument('--patch', default=True, action='store_true')
    parser.add_argument('--seg', default=True, action='store_true')
    parser.add_argument('--stitch', default=True, action='store_true')
    parser.add_argument('--no_auto_skip', default=True, action='store_false')
    parser.add_argument('--process_list', type=str, default=None, help='name of list of images to process with parameters (.csv)')
    parser.add_argument('--preset', default=None, type=str, help='predefined profile of default segmentation and filter parameters (.csv)')

    # Feature extraction arguments ---->
    parser.add_argument('--batch_size_hr', type=int, default=1)
    parser.add_argument('--batch_size_lr', type=int, default=512)
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--extract_high_res_features', type=bool, default=True, help="To create a mapping from high resolution to low resolution")
    parser.add_argument('--patch_level_low_res', type=int, default=3)  # Low  represents the magnified level [ Just Make sure that patch level should match from create patches ]
    parser.add_argument('--patch_level_high_res', type=int, default=1)  # High represents the scanning level

    # RL Models
    parser.add_argument('--config', dest='config', default='config/camelyon_sasha_inference_with_fe.yml', help='path to config file')
    parser.add_argument('--seed', type=int, default=4, help='set the random seed')
    parser.add_argument('--classifier_arch', default='hga', choices=['ga', 'hga', 'mha'], help='choice of architecture for HACMIL')
    parser.add_argument('--exp_name', type=str, default='DEBUG', help='name of the exp')
    parser.add_argument('--logs', default='enabled', choices=['enabled', 'disabled'], type=str, help='flag to save logs')

    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args



def compute_w_loader_2(model, data, device):
    """
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""

    with torch.inference_mode():

        hr_img = data['hr_img']
        hr_coords = data['hr_coords']
        hr_time = data['hr_time']

        hr_img = hr_img.unsqueeze(0)

        # Step 1 -  Obtaining H.R. Image Features
        start_time_hr = time.time()
        mapping_factor = hr_img.shape[1]
        hr_batch = hr_img
        hr_batch = torch.reshape(hr_batch, (-1, hr_batch.shape[-3], hr_batch.shape[-2], hr_batch.shape[-1]))
        hr_batch = hr_batch.to(device, non_blocking=True)
        hr_features = model(hr_batch)
        hr_features = torch.reshape(hr_features, (-1, mapping_factor, hr_features.shape[-1]))
        hr_features = hr_features.cpu()  # Op : (k, 1024)
        end_time_hr = time.time()

        # Adding them to list
        total_time_hr = hr_time + end_time_hr - start_time_hr

    return hr_features, hr_coords, total_time_hr


def main():

    # Load arguments
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

    # Introducing variable to store the details
    time_dict = defaultdict(list)
    y_pred = []
    y_true = []

    start_time = time.time()

    # Loading the pretrained encoder ----> For now we are working with ViT pretrained on medical ssl
    feature_extraction, img_transforms = get_encoder(conf.backbone, pretrain=conf.pretrain, target_img_size=conf.target_patch_size)
    feature_extraction.eval()
    feat_extractor = feature_extraction.to(conf.device)

    loader_kwargs = {'pin_memory': True} if conf.device.type == "cuda" else {}

    # loading classifier
    classifier_dict, _, config, _ = load_model(conf.classifier_ckpt_path, args)
    classifier_conf = SimpleNamespace(**config)

    if classifier_conf.arch == 'ga':
        classifier = ACMIL_GA(classifier_conf, n_token=classifier_conf.n_token,
                              n_masked_patch=classifier_conf.n_masked_patch, mask_drop=classifier_conf.mask_drop)
    elif classifier_conf.arch == 'hga':
        classifier = HAFED(classifier_conf, n_token_1=classifier_conf.n_token_1,
                           n_token_2=classifier_conf.n_token_2, n_masked_patch_1=classifier_conf.n_masked_patch_1,
                           n_masked_patch_2=classifier_conf.n_masked_patch_2, mask_drop=classifier_conf.mask_drop)
    else:
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
    model.eval()

    end_time = time.time()
    time_dict['model_load_time'].append(end_time - start_time)

    # Get the train, validation, test slide names
    slide_df = pd.read_csv(conf.csv_path)

    # Now load the train, validation, test slides
    split_file_path = './dataset_csv/%s/splits/split_%s.json' % (conf.dataset, conf.seed)
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']

    else :
        raise Exception(f"Please enter a valid split seed for dataset - {conf.dataset} ")


    for slide_name in test_names :
        
        start_time = time.time()

        # Loading the .h5 file path for slide name 
        bag_name = slide_name + '.h5'
        h5_file_path = os.path.join(conf.data_h5_dir, 'patches', bag_name)
        
        if not os.path.exists(h5_file_path):
            raise Exception("Slide name not found")  # This line can be commented out if you want to just continue the process
            continue
            
        slide_file_path = os.path.join(conf.source, slide_name + conf.slide_ext)
        if os.path.exists(slide_file_path):
            print("Slide exists")
        else:
            raise FileNotFoundError(f"None of the paths exist for slide_id: {slide_name}")
        
        # Step 3 - Initializing the result path
        wsi = openslide.open_slide(slide_file_path)

        # Step 4 - Main Function to read the file **** IMP *** Here focus on __getitem__ function is important
        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path,
                                     wsi=wsi,
                                     img_transforms=img_transforms,
                                     extract_high_res_features=False,
                                     patch_level_low_res=args.patch_level_low_res,
                                     patch_level_high_res=args.patch_level_high_res,
                                     dataset_name=conf.dataset)

        loader = DataLoader(dataset=dataset, batch_size=args.batch_size_lr, **loader_kwargs)

        # Extract low resolution images
        state, coords, low_res_feature_time = compute_w_loader(loader=loader, model=feat_extractor, verbose=0, extract_high_res_features = False, device = conf.device)
        state = torch.from_numpy(state)
        state = state.to(conf.device)
        state = state.unsqueeze(0)

        # Load the slide label
        label = slide_df[slide_df['slide_id'] == slide_name]['label'].item()
        label = torch.tensor(label, device= conf.device)
        label = label.unsqueeze(0)

        # Load RL Agent Environment
        env = WSICosineObservationEnv(lr_features=state, label=label, conf=conf)

        visited_patch_id = []
        done = False
        while not done:
            action, _, _ = model.get_action(state, visited_patch_id, is_eval=True)

            # Based on action load the high resolution feature
            hr_feature, hr_coords, hr_total_time = compute_w_loader_2(feat_extractor, dataset.get_high_res_img(coords[action.item()]), conf.device)
            hr_feature = hr_feature.to(conf.device)

            # Now based on the classification model will go from k x d ----> 1 x d
            hr_feature = classifier.get_hr_fa(hr_feature)

            new_state, _, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier, device=conf.device, hr_feature=hr_feature)
            state = new_state
            visited_patch_id.append(action.item())

        slide_preds, attn = classifier.classify(state)
        pred = torch.softmax(slide_preds, dim=-1)

        y_pred.append(pred)
        y_true.append(label)

        end_time = time.time()

        time_dict[f'{slide_name}'].append(end_time - start_time)
        print(f"Slide time : {time_dict[f'{slide_name}']}")


    y_pred = torch.cat(y_pred, dim=0)
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

    torch.save(time_dict, os.path.join(conf.log_dir, f"time_dict_sasha_{conf.frac_visit}.pt"))

    print(f"{'Phase':<6} | {'Acc':<6} | {'AUROC':<6} | {'F1':<6} | {'Precision':<9} | {'Recall':<6} | {'Balanced Acc':<13} ")
    print("-" * 110)
    print(f"{'Test':<6}  | {accuracy:.4f}  | {auroc:.4f}  | {f1_score:.4f}  | {precision:.4f}  | {recall:.4f}  | {balanced_acc:.4f}")

    total_time = []
    for key in time_dict.keys():
        total_time.append(sum(time_dict[key]))
    print(f"Average : {sum(total_time) / len(total_time)}")


if __name__ == '__main__':
    main()

