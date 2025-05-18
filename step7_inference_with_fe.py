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
import os
import time
from collections import defaultdict
from pprint import pprint
from types import SimpleNamespace

import numpy as np
import openslide
import torch
import torchmetrics
import yaml
from sklearn.metrics import balanced_accuracy_score
from timm.utils import accuracy
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models_features_extraction import get_encoder
from step7_inference import load_policy_model
from utils.gpu_utils import check_gpu_availability
from utils.utils import Struct, set_seed
from step4_extract_intermediate_features import load_model
from architecture.transformer import ACMIL_GA
from architecture.transformer import HACMIL_GA
from modules.fglobal_mlp import FGlobal
from rl_algorithms.ppo import Agent, Actor, Critic



def get_arguments():
    parser = argparse.ArgumentParser('SASHA inference with feature extraction', add_help=False)

    # Patching arguments ---->
    parser.add_argument('--source', type=str, default='/media/internal_8T/karm/karm_8T_backup/camelyon16/images/all',help='path to folder containing raw wsi image files')

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
    parser.add_argument('--dataset_name', type=str, default='camelyon16', choices=['camelyon16', 'tcga'])
    parser.add_argument('--data_h5_dir', type=str, default="/media/internal_8T/naman/test/camelyon16/level3_all")
    parser.add_argument('--data_slide_dir', type=str, default='/media/internal_8T/karm/karm_8T_backup/camelyon16/images/all')
    parser.add_argument('--slide_ext', type=str, default=".tif", help="we have two options *.tif, *.svs, or any other compatible can work")
    parser.add_argument('--csv_path', type=str, default="/media/internal_8T/naman/rlogist/sasha/dataset_csv/camelyon16.csv")
    parser.add_argument('--feat_dir', type=str, default="/media/internal_8T/naman/test/camelyon16/level3_all/features")
    parser.add_argument('--model_name', type=str, default='ViT-S/16', choices=['resnet50_trunc', 'uni_v1', 'conch_v1', 'ViT-S/16', 'Resnet50'])
    parser.add_argument('--pretrain', type=str, default='medical_ssl', choices=['medical_ssl'])
    parser.add_argument('--batch_size_hr', type=int, default=1)
    parser.add_argument('--batch_size_lr', type=int, default=512)
    parser.add_argument('--target_patch_size', type=int, default=224)
    parser.add_argument('--extract_high_res_features', type=bool, default=True, help="To create a mapping from high resolution to low resolution")
    parser.add_argument('--patch_level_low_res', type=int, default=3)  # Low  represents the magnified level [ Just Make sure that patch level should match from create patches ]
    parser.add_argument('--patch_level_high_res', type=int, default=1)  # High represents the scanning level

    # RL Models
    parser.add_argument('--config', dest='config', default='camelyon_sasha_inference_with_fe.yml', help='path to config file')
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


def compute_w_loader(loader, model, args, verbose=0, extract_high_res_features=False):
    """
	args:
		output_path: directory to save computed features (.h5 file)
		model: pytorch model
		verbose: level of feedback
	"""

    if verbose > 0:
        print(f'processing a total of {len(loader)} batches'.format(len(loader)))

    mode = 'w'
    hr_features_list = []
    lr_features_list = []
    hr_coords_list = []
    lr_coords_list = []
    hr_total_time_list = []
    lr_total_time_list = []

    features_list = []
    coords_list = []
    total_time_list = []

    for count, data in enumerate(tqdm(loader)):
        with torch.inference_mode():

            if extract_high_res_features:
                # Need to write Logic for this part

                hr_img = data['hr_img']
                hr_coords = data['hr_coords']
                hr_time = data['hr_time']
                lr_img = data['lr_img']
                lr_coords = data['lr_coords']
                lr_time = data['lr_time']

                # Step 1 -  Obtaining H.R. Image Features
                start_time_hr = time.time()
                mapping_factor = hr_img.shape[1]
                hr_batch = hr_img
                hr_batch = torch.reshape(hr_batch, (-1, hr_batch.shape[-3], hr_batch.shape[-2], hr_batch.shape[-1]))
                hr_batch = hr_batch.to(args.device, non_blocking=True)
                hr_features = model(hr_batch)
                hr_features = torch.reshape(hr_features, (-1, mapping_factor, hr_features.shape[-1]))
                hr_features = hr_features.cpu()  # Op : (k, 1024)

                hr_features_list.append(hr_features)
                hr_coords_list.append(hr_coords)
                end_time_hr = time.time()

                # Adding them to list
                total_time_hr = hr_time.sum().item() + end_time_hr - start_time_hr
                hr_total_time_list.append(total_time_hr)

                # Step 2 - Obtaining L.R Image Features
                start_time_lr = time.time()
                lr_batch = lr_img
                lr_batch = torch.reshape(lr_batch, (-1, lr_batch.shape[-3], lr_batch.shape[-2], lr_batch.shape[-1]))
                lr_batch = lr_batch.to(args.device, non_blocking=True)
                lr_features = model(lr_batch)
                lr_features = lr_features.cpu()  # Op : (1, 1024)

                lr_features_list.append(lr_features)
                lr_coords_list.append(lr_coords)
                end_time_lr = time.time()

                # Adding them to list
                total_time_lr = lr_time.sum().item() + end_time_lr - start_time_lr
                lr_total_time_list.append(total_time_lr)

            else:
                state_time_feat = time.time()

                batch = data['img']  # Op : (B, 3, 224, 224)
                coords = data['coord'].numpy().astype(np.int32)  # Op : (B, 2)
                time_1 = data['time']  # Op : (B, 1)
                batch = torch.reshape(batch,
                                      (-1, batch.shape[-3], batch.shape[-2], batch.shape[-1]))  # Op : ( B, 3, 224, 224)
                coords = np.reshape(coords, (-1, coords.shape[-1]))  # Op : (B, 2)
                batch = batch.to(args.device, non_blocking=True)

                features = model(batch)  # Ip : (B, 3, 224, 224)
                features = features.cpu()
                features_list.append(features)
                coords_list.append(coords)

                end_time_feat = time.time()

                total_time_feat = time_1.sum().item() + end_time_feat - state_time_feat
                total_time_list.append(total_time_feat)

    if extract_high_res_features:
        hr_features = torch.cat(hr_features_list, dim=0).numpy()
        lr_features = torch.cat(lr_features_list, dim=0).numpy()
        hr_coords = np.concatenate(hr_coords_list, axis=0)
        lr_coords = np.concatenate(lr_coords_list, axis=0)
        hr_total_time = sum(hr_total_time_list)
        lr_total_time = sum(lr_total_time_list)

        return hr_features, hr_coords, hr_total_time, lr_features, lr_coords, lr_total_time

    else:

        features = torch.cat(features_list, dim=0).numpy()
        coords = np.concatenate(coords_list, axis=0)
        total_time = sum(total_time_list)

        return features, coords, total_time


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
    feature_extraction, img_transforms = get_encoder(args.model_name, pretrain=args.pretrain, target_img_size=args.target_patch_size)
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
        classifier = HACMIL_GA(classifier_conf, n_token_1=classifier_conf.n_token_1,
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

    # create dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)

    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False,
                             num_workers=conf.n_worker, pin_memory=conf.pin_memory, drop_last=False)


    exit()


    metric_logger = MetricLogger(delimiter=" ")
    for data in metric_logger.log_every(test_loader, 100, 'Test'):

        start_time = time.time()

        # slide name
        slide_name = data['slide_name'][0]
        bag_name = slide_name + '.h5'
        h5_file_path = os.path.join(args.data_h5_dir, 'patches', bag_name)

        if not os.path.exists(h5_file_path):
            continue
        slide_file_path = os.path.join(args.data_slide_dir, slide_name + args.slide_ext)

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
                                     dataset_name=args.dataset_name)

        loader = DataLoader(dataset=dataset, batch_size=args.batch_size_lr, **loader_kwargs)

        # Extract low resolution images
        state, coords, low_res_feature_time = compute_w_loader(loader=loader, model=feat_extractor, args=args,
                                                               verbose=1)
        state = torch.from_numpy(state)
        state = state.to(device)
        state = state.unsqueeze(0)
        time_dict[slide_name].append(low_res_feature_time)

        # RL Agent Environment
        label = data['label'].to(device)
        env = WSICosineObservationEnv(lr_features=state, label=label, conf=conf)

        N = state.shape[1]
        visited_patch_id = []
        done = False
        while not done:
            action, _, _ = model.get_action(state, visited_patch_id, is_eval=True)

            # Based on action load the high resolution feature
            hr_feature, hr_coords, hr_total_time = compute_w_loader_2(feat_extractor,
                                                                      dataset.get_high_res_img(coords[action.item()]),
                                                                      args.device)
            hr_feature = hr_feature.to(device)

            # Now based on the classification model will go from k x d ----> 1 x d
            hr_feature = classifier.get_hr_fa(hr_feature)

            new_state, _, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier,
                                          device=device, hr_feature=hr_feature)
            state = new_state
            visited_patch_id.append(action.item())

        slide_preds, attn = classifier.classify(state)
        pred = torch.softmax(slide_preds, dim=-1)
        acc1 = accuracy(pred, label, topk=(1,))[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=label.shape[0])

        y_pred.append(pred)
        y_true.append(label)

        end_time = time.time()
        time_dict[f'{slide_name}_complete'].append(end_time - start_time)
        print(f"Slide time : {time_dict[f'{slide_name}_complete']}")

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

    torch.save(time_dict, "time_dict_lr_comp_sasha01_1.pt")
    total_time = []
    for key in time_dict.keys():
        total_time.append(sum(time_dict[key]))

    print('* Acc@1 {top1.global_avg:.3f} auroc {AUROC:.3f} f1_score {F1:.3f}'
          .format(top1=metric_logger.acc1, AUROC=auroc, F1=f1_score))

    # Get slide names where true label is 1
    print(f"Average : {sum(total_time) / len(total_time)}")


if __name__ == '__main__':
    main()

