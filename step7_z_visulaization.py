import argparse
import os
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import cv2
import h5py
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from torch.utils.data import DataLoader

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
from sasha.step5_wsi_visulaization_step2 import load_policy_model
from sasha.utils.utils import MetricLogger
from sasha.utils.utils import Struct, set_seed


def get_arguments() :
    parser = argparse.ArgumentParser('RL training', add_help=False)
    parser.add_argument('--config', dest='config', default='config/rl_config_vis_dgx.yml', help='path to config file')
    parser.add_argument('--slide_name', type=str, default='test_068', help='Get the slide name for visualization')
    parser.add_argument('--ext', type=str, default = 'tif', help = 'tif, svs')
    parser.add_argument('--wsi_images_dir_path', type=str, default='/media/internal_8T/karm/karm_8T_backup/camelyon16/images/all', help='Get the path where all the raw *.tif / *.svs images are present')
    parser.add_argument('--annotation_dir_path', type=str, default = '/media/internal_8T/karm/karm_8T_backup/camelyon16/annotations', help= 'Get the path where all the annotation are there to form the boundary over the region')
    parser.add_argument('--output_dir_path', type= str, default='/media/internal_8T/naman/rlogist/images')
    parser.add_argument('--level', type= int, default = 6, help= 'this will determine the downsample factor to save the image')
    parser.add_argument('--seed', type=int, default = 4, help = 'this will help to determine which seed to take for further analysis')
    parser.add_argument('--classifier_arch', default='hga', choices=['ga', 'hga', 'mha'], help='choice of architecture for HACMIL')
    parser.add_argument('--patch_level_base', default= 2048, help = 'determine the patch size at the highest resolution present in wsi, to downscale properly' )
    parser.add_argument('--text_font_size', default = 48, help = 'determine the size of the text font in pixels')
    args = parser.parse_args()
    return args
    

def main() : 
    
    args = get_arguments() # Load all the arguments from here

    # Now load the configuration file for execution
    conf, train_loader, val_loader, test_loader, classifier, fglobal, model, actor_optimizer, critic_optimizer, epoch, rl_config, attention_data = load_configuration_file()

    print("SASHA - Deterministic Policy {Pick Max}")
    # val_auc, val_acc, val_f1, val_loss, val_balanced_acc, val_precision, val_recall, patch_indicies_val, val_tumor_correctly_classified_sasha, val_tumor_misclassified_sasha, time_df = evaluate_policy(model, fglobal, classifier, val_loader, 'Val', device, epoch, conf)
    # test_auc, test_acc, test_f1, test_loss, test_balanced_acc, test_precision, test_recall, patch_indicies_test, test_tumor_correctly_classified_sasha, test_tumor_misclassified_sasha, time_df = evaluate_policy(model, fglobal, classifier, test_loader, 'Test', device, epoch, conf)

    (coords_patches_select_by_agent_ls, coords_similar_patches_selected_by_agent_ls, reward_ls, loss_ls,
     binary_tumor_non_tumor_patches_selected_by_agent_ls, binary_tumor_non_tumor_similar_patches_ls,
     coords, attention_weights) = evaluate_policy_per_slide(model, fglobal, classifier, test_loader, 'Test', device, conf, attention_data, args.slide_name)

    wsi_image_file_path = os.path.join(args.wsi_images_dir_path, f"{args.slide_name}.{args.ext}")
    wsi_annotation_file_path = os.path.join(args.annotation_dir_path, f"{args.slide_name}.xml")

    print(f"WSI Image file path : {wsi_image_file_path}")
    print(f"WSI Annotations file path : {wsi_annotation_file_path}")

    wsi_np, downscale_factor = draw_annotation_contours(wsi_path = wsi_image_file_path,
                                      xml_path = wsi_annotation_file_path,
                                      save_path = os.path.join(args.output_dir_path, f"{args.slide_name}_v1.png"),
                                      level = args.level)

    _ = draw_patches_selected_by_rl_agent(
        wsi_np = wsi_np.copy(),
        save_path = os.path.join(args.output_dir_path, f"{args.slide_name}_v2.png"),
        coords_patches_selected_by_agent_ls=coords_patches_select_by_agent_ls,
        patch_size_level0= args.patch_level_base,
        downscale_factor = downscale_factor,
        binary_tumor_non_tumor_patches_selected_by_agent_ls = binary_tumor_non_tumor_patches_selected_by_agent_ls,
        args = args
    )

    _ = draw_patches_updated_by_ssu(
        wsi_np=wsi_np.copy(),
        save_path=os.path.join(args.output_dir_path, f"{args.slide_name}_v3.png"),
        coords_similar_patches_selected_by_agent_ls=coords_similar_patches_selected_by_agent_ls,
        binary_tumor_non_tumor_similar_patches_ls = binary_tumor_non_tumor_similar_patches_ls,
        patch_size_level0=args.patch_level_base,
        downscale_factor=downscale_factor, args=  args
    )

    # Now take all the images -
    images_path = [os.path.join(args.output_dir_path, f"{args.slide_name}_v1.png"),
                   os.path.join(args.output_dir_path, f"{args.slide_name}_v2.png"),
                   os.path.join(args.output_dir_path, f"{args.slide_name}_v3.png")]

    combine_images_row_with_titles(images_path, os.path.join(args.output_dir_path, f"{args.slide_name}_v4.png"))

    crop_top_30_percent(image_path = os.path.join(args.output_dir_path, f"{args.slide_name}_v4.png") ,
                        output_path = os.path.join(args.output_dir_path, f"{args.slide_name}_v4_cropped.png")
                        )

    exit()


def crop_top_30_percent(image_path, output_path):
    # Open the image
    img = Image.open(image_path)

    # Calculate 30% of the height
    crop_height = int(img.height * 0.30)

    # Crop (left, upper, right, lower)
    cropped_img = img.crop((0, crop_height, img.width, img.height))

    # Save the cropped image
    cropped_img.save(output_path)
    print(f"Cropped image saved at: {output_path}")




@torch.no_grad()
def evaluate_policy_per_slide(model, fglobal, classifier, data_loader, header, device, conf, attention_data, slide_name, is_eval = True, is_top_k = False, is_top_p = False ):

    if slide_name is None :
        raise Exception("Enter a valid slide_name in test loader")

    model.eval()

    patches_selected_by_agent_ls = []
    similar_patches_selected_by_agent_ls = []
    entropy_changing_with_time_ls = []
    hit_ratio_ls = []
    binary_tumor_non_tumor_patches_selected_by_agent_ls = []
    binary_tumor_non_tumor_similar_patches_ls = []
    reward_ls = []
    attention_weights = None
    loss_ls = []

    lr_patch_tumor_area_percentage_ls = attention_data[slide_name]['lr_patch_tumor_area_percentage']

    metric_logger = MetricLogger(delimiter=" ")

    for data in metric_logger.log_every(data_loader, 100, header):

        if slide_name != data['slide_name'][0] :
            continue

        hr_features = data['hr'][0].to(device, dtype=torch.float32)
        state = data['lr'].to(device, dtype=torch.float32)
        slide_id = data['slide_name'][0]
        label = data['label'].to(device)

        if conf.fglobal == 'attn':
            env = WSIObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)
        else:
            env = WSICosineObservationEnv(lr_features=state, hr_features=hr_features, label=label, conf=conf)

        N = state.shape[1]
        done = False
        visited_patch_id = []


        while not done:
            action, log_prob, entropy = model.get_action(state, visited_patch_id, is_eval = is_eval, is_top_k= is_top_k, is_top_p= is_top_p)
            new_state, reward, done = env.step(action=action, state_update_net=fglobal, classifier_net=classifier,
                                               device=device)
            state = new_state
            visited_patch_id.append(action.item())

            # Store details at each time step
            patches_selected_by_agent_ls.append(action.item())
            entropy_changing_with_time_ls.append(entropy.item())
            similar_patches_selected_by_agent_ls.append(env.get_similar_patches())
            reward_ls.append(reward)
            loss_ls.append(-1 * reward)

        # Final state ---->
        _, attn = classifier.classify(state)

    attention_weights = attn
    binary_tumor_non_tumor_patches_selected_by_agent_ls = [1 if lr_patch_tumor_area_percentage_ls[idx] > 0 else 0 for idx in patches_selected_by_agent_ls]
    for sub_ls in similar_patches_selected_by_agent_ls :
        intermediate_ls = [1 if lr_patch_tumor_area_percentage_ls[idx] > 0 else 0 for idx in sub_ls]
        binary_tumor_non_tumor_similar_patches_ls.append(intermediate_ls)


    # Loading coordinates --->
    coords = get_slide_coords(conf.level3_h5_path, slide_name)

    assert coords.shape[0] == lr_patch_tumor_area_percentage_ls.shape[0]
    assert coords.shape[0] == state.shape[1]

    coords_patches_select_by_agent_ls = coords[patches_selected_by_agent_ls]
    coords_similar_patches_selected_by_agent_ls = []
    for sub_ls in similar_patches_selected_by_agent_ls :
        sub_ls = sub_ls.tolist()
        intermediate_ls = coords[sub_ls]
        coords_similar_patches_selected_by_agent_ls.append(intermediate_ls)


    print("OK")
    return coords_patches_select_by_agent_ls, coords_similar_patches_selected_by_agent_ls, reward_ls, loss_ls, binary_tumor_non_tumor_patches_selected_by_agent_ls, binary_tumor_non_tumor_similar_patches_ls, coords, attention_weights


def get_slide_coords(features_path, slide_name):
    """
    Returns the coords array (N, 2) from a nested group in the HDF5 file.

    Parameters:
        features_path (str): Path to the HDF5 file.
        slide_name (str): Group name inside the HDF5 file (e.g., 'test_016').

    Returns:
        np.ndarray: The coordinates array of shape (N, 2) if found.

    Raises:
        KeyError: If the slide or 'coords' key is not found.
    """
    with h5py.File(features_path, 'r') as f:
        if slide_name not in f:
            raise KeyError(f"Slide '{slide_name}' not found in HDF5 file.")

        slide_group = f[slide_name]
        if 'coords' not in slide_group:
            raise KeyError(f"'coords' not found under slide '{slide_name}'.")

        coords = slide_group['coords'][:]
        return coords


def load_configuration_file() :

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
        'fraction of visit' : conf.frac_visit,
        'num_envs' : conf.num_envs
    }

    # hyparams_text = "\n".join([f"{key}: {value}" for key, value in hyparams.items()])
    # print(f"Hyparams: {hyparams_text}")

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
    classifier_dict, _, config, _ = load_model(ckpt_path=conf.classifier_ckpt_path, device=device)
    classifier_conf = SimpleNamespace(**config)
    if classifier_conf.arch == 'ga':
        classifier = ACMIL_GA(classifier_conf, n_token=classifier_conf.n_token,
                              n_masked_patch=classifier_conf.n_masked_patch, mask_drop=classifier_conf.mask_drop)
    elif classifier_conf.arch == 'hga':
        classifier = HACMIL_GA(classifier_conf, n_token_1=classifier_conf.n_token_1,
                               n_token_2=classifier_conf.n_token_2, n_masked_patch_1=classifier_conf.n_masked_patch_1,
                               n_masked_patch_2=classifier_conf.n_masked_patch_2, mask_drop=classifier_conf.mask_drop)
    elif classifier_conf.arch == 'hga_sparse':
        classifier = HACMIL_GA_Sparse(classifier_conf, n_token_1=classifier_conf.n_token_1,
                                      n_token_2=classifier_conf.n_token_2,
                                      n_masked_patch_1=classifier_conf.n_masked_patch_1,
                                      n_masked_patch_2=classifier_conf.n_masked_patch_2,
                                      mask_drop=classifier_conf.mask_drop)
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
        fglobal = FGlobal(ip_dim=384 * 3, op_dim=384).to(device)
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

    model, actor_optimizer, critic_optimizer, epoch, rl_config = load_policy_model(model, actor_optimizer,
                                                                                   critic_optimizer,
                                                                                   conf.rl_ckpt_path, device)

    # Loading the attention weights
    attention_data = torch.load(conf.attention_path, map_location='cpu')

    return conf, train_loader, val_loader, test_loader, classifier, fglobal, model, actor_optimizer, critic_optimizer, epoch, rl_config, attention_data


def draw_annotation_contours(
    wsi_path,
    xml_path,
    save_path,
    level,
    is_save = True
):
    # Load WSI
    slide = openslide.OpenSlide(wsi_path)

    # Level info
    downscale_factor = slide.level_downsamples[level]
    wsi_size = slide.level_dimensions[level]
    wsi_img = slide.read_region((0, 0), level, wsi_size).convert("RGB")
    wsi_np = np.array(wsi_img)

    # Parse XML annotations
    tree = ET.parse(xml_path)
    root = tree.getroot()

    all_polygons = []
    for annotation in root.findall(".//Annotation"):
        coords = []
        for coord in annotation.findall(".//Coordinate"):
            x = float(coord.get("X")) / downscale_factor
            y = float(coord.get("Y")) / downscale_factor
            coords.append((x, y))
        all_polygons.append(Polygon(coords))

    outer_polys = []
    inner_polys = []
    for i, poly in enumerate(all_polygons):
        is_inner = any(other.contains(poly) for j, other in enumerate(all_polygons) if i != j)
        (inner_polys if is_inner else outer_polys).append(poly)

    # Draw annotation polygons
    for poly in outer_polys:
        pts = np.array(poly.exterior.coords, np.int32).reshape((-1, 1, 2))
        cv2.drawContours(wsi_np, [pts], -1, (255, 0, 0), thickness=5)  # Red

    for poly in inner_polys:
        pts = np.array(poly.exterior.coords, np.int32).reshape((-1, 1, 2))
        cv2.drawContours(wsi_np, [pts], -1, (0, 0, 255), thickness=5)  # Blue

    # Save final result if required
    if is_save:
        annotated_img = Image.fromarray(wsi_np)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        annotated_img.save(save_path)
        print(f"Annotated image saved at: {save_path}")

    # Return the annotated image array
    return wsi_np, downscale_factor


def draw_patches_selected_by_rl_agent(
    wsi_np,
    save_path,
    coords_patches_selected_by_agent_ls,
    binary_tumor_non_tumor_patches_selected_by_agent_ls,
    patch_size_level0,
    downscale_factor,
    args= None,
    is_save=True
):
    if len(coords_patches_selected_by_agent_ls) != len(binary_tumor_non_tumor_patches_selected_by_agent_ls):
        raise ValueError("Mismatch between patch coordinates and tumor flags.")

    # Overlay RL Agent selected patches (light blue → dark blue)
    if len(coords_patches_selected_by_agent_ls) > 0:
        norm = plt.Normalize(0.0, 0.2)
        colormap = cm.get_cmap("Blues_r")  # Reverse for dark-to-light blue
        box_w = int(patch_size_level0 / downscale_factor)

        for i, ((x_lvl0, y_lvl0), tumor_flag) in enumerate(zip(coords_patches_selected_by_agent_ls, binary_tumor_non_tumor_patches_selected_by_agent_ls)):
            x_scaled = int(x_lvl0 / downscale_factor)
            y_scaled = int(y_lvl0 / downscale_factor)
            frac = i / max(1, len(coords_patches_selected_by_agent_ls) - 1) * 0.2
            color = tuple(int(255 * c) for c in colormap(norm(frac))[:3])

            # Draw the RL-selected patch rectangle
            cv2.rectangle(wsi_np, (x_scaled, y_scaled), (x_scaled + box_w, y_scaled + box_w), color, thickness=3)

    # Create combined PIL image
    # pil_img = Image.fromarray(wsi_np)
    # draw = ImageDraw.Draw(pil_img)
    # text_font_size = args.text_font_size
    # try:
    #     font = ImageFont.truetype("arial.ttf", text_font_size)
    # except:
    #     font = ImageFont.load_default()
    #
    # # Draw RL Agent legend (Blues gradient)
    # legend_height = 50
    # legend_width = 350
    # spacing = 80
    # bar_spacing = 30
    # x_start = 30
    # y_start_rl = pil_img.height - (3 * legend_height + 2 * spacing + 80)  # Increased vertical space
    #
    # norm = plt.Normalize(0.0, 0.2)
    # colormap_blues = cm.get_cmap("Blues_r")
    # gradient_rl = np.linspace(0.0, 0.2, legend_width)
    # gradient_img_rl = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    # for i in range(legend_width):
    #     color = tuple(int(255 * c) for c in colormap_blues(norm(gradient_rl[i]))[:3])
    #     gradient_img_rl[:, i, :] = color
    # pil_img.paste(Image.fromarray(gradient_img_rl), (x_start, y_start_rl))
    # # Draw black boundary around the gradient bar
    # draw.rectangle(
    #     [x_start, y_start_rl, x_start + legend_width, y_start_rl + legend_height],
    #     outline="black",
    #     width=5
    # )
    #
    # # Text label above the gradient bar
    # draw.text((x_start, y_start_rl - 50), "Patches selected by RL Agent", fill="black", font=font)
    #
    # # Fraction labels below the gradient bar
    # draw.text((x_start, y_start_rl + legend_height + 10), "Frac 0.0", fill="black", font=font)
    # draw.text((x_start + legend_width - 40, y_start_rl + legend_height + 10), "0.2", fill="black", font=font)
    #
    # # Tumor/Non-tumor barcode strip (aligned with RL gradient)
    # if len(binary_tumor_non_tumor_patches_selected_by_agent_ls) > 0:
    #     bar_height = legend_height
    #     bar_img = np.ones((bar_height, legend_width, 3), dtype=np.uint8) * 255  # white = normal
    #     for i in range(legend_width):
    #         patch_index = int(i / legend_width * len(binary_tumor_non_tumor_patches_selected_by_agent_ls))
    #         patch_index = min(patch_index, len(binary_tumor_non_tumor_patches_selected_by_agent_ls) - 1)
    #         if binary_tumor_non_tumor_patches_selected_by_agent_ls[patch_index] == 1:
    #             bar_img[:, i, :] = (0, 0, 139)  # dark blue = tumor
    #     y_bar = y_start_rl + legend_height + 80  # More vertical gap between gradient and barcode
    #     pil_img.paste(Image.fromarray(bar_img), (x_start, y_bar))
    #
    #     draw.rectangle(
    #         [x_start, y_bar, x_start + legend_width, y_bar + legend_height],
    #         outline="black",
    #         width=5
    #     )
    #
    #     # Adjusted text label Y positions (below barcode with extra space)
    #     label_y = y_bar + bar_height + 30
    #     box_w, box_h = 30, 20
    #     spacing_x = 350  # Space between the two boxes and labels
    #
    #     # Tumor Patch = Blue box
    #     tumor_box_x = x_start
    #     draw.rectangle(
    #         [tumor_box_x, label_y, tumor_box_x + box_w, label_y + box_h],
    #         fill=(0, 0, 255), outline="black", width=2
    #     )
    #     draw.text((tumor_box_x + box_w + 10, label_y), "Tumor Patch", fill="black", font=font)
    #
    #     # Normal Patch = White box
    #     normal_box_x = tumor_box_x + box_w + spacing_x
    #     draw.rectangle(
    #         [normal_box_x, label_y, normal_box_x + box_w, label_y + box_h],
    #         fill=(255, 255, 255), outline="black", width=2
    #     )
    #     draw.text((normal_box_x + box_w + 10, label_y), "Normal Patch", fill="black", font=font)

    # Convert back to np array for return
    wsi_np = np.array(wsi_np)

    if is_save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(wsi_np).save(save_path)
        print(f"Annotated image saved at: {save_path}")

    return wsi_np


def draw_patches_updated_by_ssu(
    wsi_np,
    save_path,
    coords_similar_patches_selected_by_agent_ls,
    binary_tumor_non_tumor_similar_patches_ls,
    patch_size_level0,
    downscale_factor,
    args = None,
    is_save=True
):
    if len(coords_similar_patches_selected_by_agent_ls) != len(binary_tumor_non_tumor_similar_patches_ls):
        raise ValueError("Mismatch between patch coordinates and tumor flags.")

    # Overlay Similar patches (light orange → dark orange)
    if coords_similar_patches_selected_by_agent_ls:
        norm = plt.Normalize(0.0, 0.2)
        colormap = cm.get_cmap("Oranges_r")
        box_w = int(patch_size_level0 / downscale_factor)

        for i, (patch_group, tumor_flags) in enumerate(zip(coords_similar_patches_selected_by_agent_ls, binary_tumor_non_tumor_similar_patches_ls)):
            frac = i / max(1, len(coords_similar_patches_selected_by_agent_ls) - 1) * 0.2
            color = tuple(int(255 * c) for c in colormap(norm(frac))[:3])
            for (x_lvl0, y_lvl0), tumor_flag in zip(patch_group, tumor_flags):
                x_scaled = int(x_lvl0 / downscale_factor)
                y_scaled = int(y_lvl0 / downscale_factor)
                cv2.rectangle(
                    wsi_np,
                    (x_scaled, y_scaled),
                    (x_scaled + box_w, y_scaled + box_w),
                    color=color,
                    thickness=3
                )

    # Create combined PIL image to draw on
    # pil_img = Image.fromarray(wsi_np)
    # draw = ImageDraw.Draw(pil_img)
    # text_font_size = args.text_font_size
    # try:
    #     font = ImageFont.truetype("arial.ttf", text_font_size)
    # except:
    #     font = ImageFont.load_default()
    #
    # # Legend setup
    # legend_height = 50
    # legend_width = 350
    # spacing = 80
    # bar_spacing = 30
    # x_start = 30
    #
    # # Compute y positions
    # y_start_sim = pil_img.height - (3 * legend_height + 2 * spacing + 40)
    #
    # # Similar Patches Gradient
    # colormap_oranges = cm.get_cmap("Oranges_r")
    # gradient_sim = np.linspace(0.0, 0.2, legend_width)
    # gradient_img_sim = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    # for i in range(legend_width):
    #     color = tuple(int(255 * c) for c in colormap_oranges(norm(gradient_sim[i]))[:3])
    #     gradient_img_sim[:, i, :] = color
    # pil_img.paste(Image.fromarray(gradient_img_sim), (x_start, y_start_sim))
    # # Draw black boundary around the gradient bar
    # draw.rectangle(
    #     [x_start, y_start_sim, x_start + legend_width, y_start_sim + legend_height],
    #     outline="black",
    #     width=5
    # )
    #
    # # Text label above the gradient bar
    # draw.text((x_start, y_start_sim - 50), "Similar patches updated", fill="black", font=font)
    #
    # # Fraction labels below the gradient bar
    # draw.text((x_start, y_start_sim + legend_height + 10), "Fraction 0.0", fill="black", font=font)
    # draw.text((x_start + legend_width - 40, y_start_sim + legend_height + 10), "0.2", fill="black", font=font)
    #
    # # Tumor/Normal patch barcode strip
    # if binary_tumor_non_tumor_similar_patches_ls:
    #     bar_height = legend_height
    #     bar_img = np.ones((bar_height, legend_width, 3), dtype=np.uint8) * 255  # white = normal
    #     all_flags = [flag for sublist in binary_tumor_non_tumor_similar_patches_ls for flag in sublist]
    #     for i in range(legend_width):
    #         patch_index = int(i / legend_width * len(all_flags))
    #         patch_index = min(patch_index, len(all_flags) - 1)
    #         if all_flags[patch_index] == 1:
    #             bar_img[:, i, :] = (255, 140, 0)  # orange = tumor
    #     y_bar = y_start_sim + legend_height + 80
    #     pil_img.paste(Image.fromarray(bar_img), (x_start, y_bar))
    #
    #     draw.rectangle(
    #         [x_start, y_bar, x_start + legend_width, y_bar + legend_height],
    #         outline="black",
    #         width=5
    #     )
    #
    #     # Legend boxes below barcode
    #     label_y = y_bar + bar_height + 30
    #     box_w, box_h = 30, 20
    #     spacing_x = 350
    #
    #     # Tumor Patch = Orange box
    #     tumor_box_x = x_start
    #     draw.rectangle(
    #         [tumor_box_x, label_y, tumor_box_x + box_w, label_y + box_h],
    #         fill=(255, 140, 0), outline="black", width=2
    #     )
    #     draw.text((tumor_box_x + box_w + 10, label_y), "Tumor Patch", fill="black", font=font)
    #
    #     # Normal Patch = White box
    #     normal_box_x = tumor_box_x + box_w + spacing_x
    #     draw.rectangle(
    #         [normal_box_x, label_y, normal_box_x + box_w, label_y + box_h],
    #         fill=(255, 255, 255), outline="black", width=2
    #     )
    #     draw.text((normal_box_x + box_w + 10, label_y), "Normal Patch", fill="black", font=font)

    # Save final result
    wsi_np = np.array(wsi_np)
    if is_save:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(wsi_np).save(save_path)
        print(f"Annotated image saved at: {save_path}")

    return wsi_np


def combine_images_row_with_titles(images_path, save_path_v4):
    titles = [
        "WSI Image",
        "Patches selected by RL Agent",
        "Similar patches updated"
    ]

    # Load images
    images = [Image.open(path) for path in images_path]

    # Ensure all images have the same height
    min_height = min(img.height for img in images)
    images = [img.resize((int(img.width * min_height / img.height), min_height)) for img in images]

    # Create new image with extra space for titles
    font_size = 36
    spacing = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
    title_height = font_size + 20
    combined_img = Image.new("RGB", (total_width, min_height + title_height), "white")
    draw = ImageDraw.Draw(combined_img)

    # Paste images and draw titles
    x_offset = 0
    for img, title in zip(images, titles):
        combined_img.paste(img, (x_offset, title_height))
        text_width = draw.textlength(title, font=font)
        text_x = x_offset + (img.width - text_width) // 2
        draw.text((text_x, 10), title, fill="black", font=font)
        x_offset += img.width + spacing

    # Save the final v4 image
    combined_img.save(save_path_v4)
    print(f"Combined image saved at: {save_path_v4}")


def draw_attention_weighted_patches(
    wsi_path,
    save_path,
    level,
    coords,
    attention_weights,
    patch_size_level0=2048,
    colormap_name="RdYlGn_r"  # Red=high, Green=low
):


    # Load WSI
    slide = openslide.OpenSlide(wsi_path)
    downscale_factor = slide.level_downsamples[level]
    wsi_size = slide.level_dimensions[level]
    wsi_img = slide.read_region((0, 0), level, wsi_size).convert("RGB")
    wsi_np = np.array(wsi_img)

    attention_weights = np.array(attention_weights)
    if attention_weights.max() > 1.0 or attention_weights.min() < 0.0:
        attention_weights = (attention_weights - attention_weights.min()) / (
                attention_weights.max() - attention_weights.min() + 1e-8
        )

    # Normalize attention weights
    min_w, max_w = min(attention_weights), max(attention_weights)
    norm = plt.Normalize(min_w, max_w)
    colormap = cm.get_cmap(colormap_name)

    box_w = int(patch_size_level0 / downscale_factor)

    # Draw each patch with attention-weighted color
    for (x_lvl0, y_lvl0), weight in zip(coords, attention_weights):
        x_scaled = int(x_lvl0 / downscale_factor)
        y_scaled = int(y_lvl0 / downscale_factor)
        color = tuple(int(255 * c) for c in colormap(norm(weight))[:3])
        cv2.rectangle(
            wsi_np,
            (x_scaled, y_scaled),
            (x_scaled + box_w, y_scaled + box_w),
            color=color,
            thickness=3
        )

    # Add gradient legend
    legend_width = 350
    legend_height = 20
    spacing = 60
    text_font_size = args.text_font_size

    pil_img = Image.fromarray(wsi_np)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("arial.ttf", text_font_size)
    except:
        font = ImageFont.load_default()

    y_start = pil_img.height - (legend_height + spacing)
    x_start = 30

    gradient_vals = np.linspace(min_w, max_w, legend_width)
    gradient_img = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    for i in range(legend_width):
        color = tuple(int(255 * c) for c in colormap(norm(gradient_vals[i]))[:3])
        gradient_img[:, i, :] = color

    pil_img.paste(Image.fromarray(gradient_img), (x_start, y_start))
    draw.text((x_start, y_start - 35), f"Attention {min_w:.2f}", fill="black", font=font)
    draw.text((x_start + legend_width - 100, y_start - 35), f"{max_w:.2f}", fill="black", font=font)
    draw.text((x_start, y_start + legend_height + 10), "Attention-weighted patches", fill="black", font=font)

    # Save
    final_img = np.array(pil_img)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(final_img).save(save_path)
    print(f"Image with attention-weighted patches saved at: {save_path}")

if __name__ == "__main__":
    # Adding Device Details
    gpus = check_gpu_availability(5, 1, [])
    print(f"occupied {gpus}")
    device = torch.device(f"cuda:{gpus[0]}")
    main()




# def draw_patches_updated_by_ssu(wsi_np, save_path, coords_similar_patches_selected_by_agent_ls, patch_size_level0,
#                                       downscale_factor, is_save = True):
#
#     if coords_similar_patches_selected_by_agent_ls:
#         norm = plt.Normalize(0.0, 0.2)
#         colormap = cm.get_cmap("Oranges_r")
#         box_w = int(patch_size_level0 / downscale_factor)
#
#         for i, patch_group in enumerate(coords_similar_patches_selected_by_agent_ls):
#             frac = i / max(1, len(coords_similar_patches_selected_by_agent_ls) - 1) * 0.2
#             color = tuple(int(255 * c) for c in colormap(norm(frac))[:3])
#             for x_lvl0, y_lvl0 in patch_group:
#                 x_scaled = int(x_lvl0 / downscale_factor)
#                 y_scaled = int(y_lvl0 / downscale_factor)
#                 cv2.rectangle(
#                     wsi_np,
#                     (x_scaled, y_scaled),
#                     (x_scaled + box_w, y_scaled + box_w),
#                     color=color,
#                     thickness=3
#                 )
#
#     # Add both gradient legends at the bottom
#     legend_height = 20
#     legend_width = 350
#     spacing = 80  # Space between two legends
#     text_font_size = 32
#
#     # Create combined PIL image to draw on
#     pil_img = Image.fromarray(wsi_np)
#     draw = ImageDraw.Draw(pil_img)
#     try:
#         font = ImageFont.truetype("arial.ttf", text_font_size)
#     except:
#         font = ImageFont.load_default()
#
#     # Starting y coordinate from bottom
#     y_start_rl = pil_img.height - 2 * (legend_height + spacing)
#     y_start_sim = pil_img.height - (legend_height + spacing)
#     x_start = 30
#
#     # Similar Patches Gradient (Oranges)
#     colormap_oranges = cm.get_cmap("Oranges_r")
#     gradient_sim = np.linspace(0.0, 0.2, legend_width)
#     gradient_img_sim = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
#     for i in range(legend_width):
#         color = tuple(int(255 * c) for c in colormap_oranges(norm(gradient_sim[i]))[:3])
#         gradient_img_sim[:, i, :] = color
#     pil_img.paste(Image.fromarray(gradient_img_sim), (x_start, y_start_sim))
#     draw.text((x_start, y_start_sim + legend_height + 10), "Similar patches updated", fill="black", font=font)
#     draw.text((x_start, y_start_sim - 35), "Fraction 0.0", fill="black", font=font)
#     draw.text((x_start + legend_width - 40, y_start_sim - 25), "0.2", fill="black", font=font)
#
#     # Save final result
#     wsi_np = np.array(pil_img)
#
#     if is_save :
#
#         # Save final image
#         annotated_img = Image.fromarray(wsi_np)
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         annotated_img.save(save_path)
#         print(f"Annotated image saved at: {save_path}")
#
#     return wsi_np

# def draw_patches_selected_by_rl_agent(wsi_np, save_path, coords_patches_selected_by_agent_ls, patch_size_level0, downscale_factor, is_save = True) :
#
#     # Overlay RL Agent selected patches (light blue → dark blue)
#     if coords_patches_selected_by_agent_ls.any() :
#         norm = plt.Normalize(0.0, 0.2)
#         colormap = cm.get_cmap("Blues_r")  # Reverse the colormap for dark-to-light
#         box_w = int(patch_size_level0 / downscale_factor)
#
#         for i, (x_lvl0, y_lvl0) in enumerate(coords_patches_selected_by_agent_ls):
#             x_scaled = int(x_lvl0 / downscale_factor)
#             y_scaled = int(y_lvl0 / downscale_factor)
#             frac = i / max(1, len(coords_patches_selected_by_agent_ls) - 1) * 0.2
#             color = tuple(int(255 * c) for c in colormap(norm(frac))[:3])
#             cv2.rectangle(wsi_np, (x_scaled, y_scaled), (x_scaled + box_w, y_scaled + box_w), color, thickness=3)
#
#     # Add both gradient legends at the bottom
#     legend_height = 20
#     legend_width = 350
#     spacing = 80  # Space between two legends
#     text_font_size = 32
#
#     # Create combined PIL image to draw on
#     pil_img = Image.fromarray(wsi_np)
#     draw = ImageDraw.Draw(pil_img)
#     try:
#         font = ImageFont.truetype("arial.ttf", text_font_size)
#     except:
#         font = ImageFont.load_default()
#
#     # Starting y coordinate from bottom
#     y_start_rl = pil_img.height - 2 * (legend_height + spacing)
#     y_start_sim = pil_img.height - (legend_height + spacing)
#     x_start = 30
#
#     # RL Agent Gradient (Blues)
#     norm = plt.Normalize(0.0, 0.2)
#     colormap_blues = cm.get_cmap("Blues_r")
#     gradient_rl = np.linspace(0.0, 0.2, legend_width)
#     gradient_img_rl = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
#     for i in range(legend_width):
#         color = tuple(int(255 * c) for c in colormap_blues(norm(gradient_rl[i]))[:3])
#         gradient_img_rl[:, i, :] = color
#     pil_img.paste(Image.fromarray(gradient_img_rl), (x_start, y_start_rl))
#     draw.text((x_start, y_start_rl + legend_height + 10), "Patches selected by RL Agent", fill="black", font=font)
#     draw.text((x_start, y_start_rl - 35), "Fraction 0.0", fill="black", font=font)
#     draw.text((x_start + legend_width - 40, y_start_rl - 25), "0.2", fill="black", font=font)
#
#     # Save final result
#     wsi_np = np.array(pil_img)
#
#     if is_save :
#
#         # Save final image
#         annotated_img = Image.fromarray(wsi_np)
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         annotated_img.save(save_path)
#         print(f"Annotated image saved at: {save_path}")
#
#     return wsi_np