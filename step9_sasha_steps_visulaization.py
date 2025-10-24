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

from architecture.transformer import HAFED
from datasets.datasets import build_HDF5_feat_dataset_2
from envs.WSI_cosine_env import WSICosineObservationEnv
from envs.WSI_env import WSIObservationEnv
from modules.fglobal_mlp import FGlobal
from rl_algorithms.ppo import Agent, Actor, Critic
from step4_extract_intermediate_features import load_model
from step7_inference import load_policy_model
from utils.gpu_utils import check_gpu_availability
from utils.utils import MetricLogger
from utils.utils import Struct, set_seed


def get_arguments() :
    parser = argparse.ArgumentParser('RL training', add_help=False)
    parser.add_argument('--config', default=None, help='path to config file')
    parser.add_argument('--slide_name', type=str, default='test_068', help='Get the slide name for visualization')
    parser.add_argument('--ext', type=str, default = 'tif', help = 'tif, svs')
    parser.add_argument('--wsi_images_dir_path', type=str, default=None, help='Get the path where all the raw *.tif / *.svs images are present')
    parser.add_argument('--annotation_dir_path', type=str, default = None, help= 'Get the path where all the annotation are there to form the boundary over the region')
    parser.add_argument('--output_dir_path', type= str, default=None)
    parser.add_argument('--level', type= int, default = 6, help= 'this will determine the downsample factor to save the image')
    parser.add_argument('--seed', type=int, default = 4, help = 'this will help to determine which seed to take for further analysis')
    parser.add_argument('--classifier_arch', default='hafed', choices=['hafed'], help='choice of architecture for HAFED')
    parser.add_argument('--patch_level_base', default= 2048, help = 'determine the patch size at the highest resolution present in wsi, to downscale properly' )
    parser.add_argument('--text_font_size', default = 48, help = 'determine the size of the text font in pixels')
    args = parser.parse_args()

    # Adding Device Details
    gpus = check_gpu_availability(3, 1, [])
    print(f"occupied {gpus}")
    args.device = torch.device(f"cuda:{gpus[0]}")

    return args


def load_configuration_file():
    # getting and config file
    args = get_arguments()

    with open(args.config, 'r') as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        c.update(vars(args))
        conf = Struct(**c)

    # Loading seed
    set_seed(args.seed)

    # create dataloaders
    train_data, val_data, test_data = build_HDF5_feat_dataset_2(conf.level1_path, conf.level3_path, conf)
    train_loader = DataLoader(train_data, batch_size=conf.B, shuffle=True, num_workers=conf.n_worker,
                              pin_memory=conf.pin_memory, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker,
                            pin_memory=conf.pin_memory, drop_last=False)
    test_loader = DataLoader(test_data, batch_size=conf.B, shuffle=False, num_workers=conf.n_worker,
                             pin_memory=conf.pin_memory, drop_last=False)

    # loading classifier
    classifier_dict, _, config, _ = load_model(conf.classifier_ckpt_path, args)
    classifier_conf = SimpleNamespace(**config)


    if conf.classifier_arch == 'hafed':
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
    model, actor_optimizer, critic_optimizer, epoch, rl_config = load_policy_model(model, actor_optimizer,
                                                                                   critic_optimizer, conf.rl_ckpt_path,
                                                                                   conf.device)

    return conf, train_loader, val_loader, test_loader, classifier, fglobal, model, actor_optimizer, critic_optimizer, epoch, rl_config


def main() : 
    
    args = get_arguments() # Load all the arguments from here

    # Now load the configuration file for execution
    conf, train_loader, val_loader, test_loader, classifier, fglobal, model, actor_optimizer, critic_optimizer, epoch, rl_config = load_configuration_file()

    (coords_patches_select_by_agent_ls, coords_similar_patches_selected_by_agent_ls, reward_ls, loss_ls,
     coords, attention_weights) = evaluate_policy_per_slide(model, fglobal, classifier, test_loader, 'Test', conf.device, conf, args.slide_name)


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
        args = args
    )

    _ = draw_patches_updated_by_ssu(
        wsi_np=wsi_np.copy(),
        save_path=os.path.join(args.output_dir_path, f"{args.slide_name}_v3.png"),
        coords_similar_patches_selected_by_agent_ls=coords_similar_patches_selected_by_agent_ls,
        patch_size_level0=args.patch_level_base,
        downscale_factor=downscale_factor, args=  args
    )

    # Now take all the images -
    images_path = [os.path.join(args.output_dir_path, f"{args.slide_name}_v1.png"),
                   os.path.join(args.output_dir_path, f"{args.slide_name}_v2.png"),
                   os.path.join(args.output_dir_path, f"{args.slide_name}_v3.png")]

    combine_images_row_with_titles(images_path, os.path.join(args.output_dir_path, f"{args.slide_name}_v4.png"))


    exit()


@torch.no_grad()
def evaluate_policy_per_slide(model, fglobal, classifier, data_loader, header, device, conf, slide_name, is_eval = True, is_top_k = False, is_top_p = False ):

    if slide_name is None :
        raise Exception("Enter a valid slide_name in test loader")

    model.eval()

    patches_selected_by_agent_ls = []
    similar_patches_selected_by_agent_ls = []
    entropy_changing_with_time_ls = []
    reward_ls = []
    loss_ls = []


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

    # Loading coordinates --->
    coords = get_slide_coords(conf.level3_path, slide_name)

    assert coords.shape[0] == state.shape[1]

    coords_patches_select_by_agent_ls = coords[patches_selected_by_agent_ls]
    coords_similar_patches_selected_by_agent_ls = []
    for sub_ls in similar_patches_selected_by_agent_ls :
        sub_ls = sub_ls.tolist()
        intermediate_ls = coords[sub_ls]
        coords_similar_patches_selected_by_agent_ls.append(intermediate_ls)

    return coords_patches_select_by_agent_ls, coords_similar_patches_selected_by_agent_ls, reward_ls, loss_ls, coords, attention_weights


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

    # List .h5 files
    h5_files = [f for f in os.listdir(features_path) if f.endswith('.h5')]

    # Get the single .h5 file path
    if len(h5_files) == 1:
        h5_file_path = os.path.join(features_path, h5_files[0])
        print("Found .h5 file:", h5_file_path)
    else:
        print(f"Expected 1 .h5 file, but found {len(h5_files)}.")

    features_path = h5_file_path

    with h5py.File(features_path, 'r') as f:
        if slide_name not in f:
            raise KeyError(f"Slide '{slide_name}' not found in HDF5 file.")

        slide_group = f[slide_name]
        if 'coords' not in slide_group:
            raise KeyError(f"'coords' not found under slide '{slide_name}'.")

        coords = slide_group['coords'][:]
        return coords


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
    patch_size_level0,
    downscale_factor,
    args= None,
    is_save=True
):

    # Overlay RL Agent selected patches (light blue → dark blue)
    if len(coords_patches_selected_by_agent_ls) > 0:
        norm = plt.Normalize(0.0, 0.2)
        colormap = cm.get_cmap("Blues_r")  # Reverse for dark-to-light blue
        box_w = int(patch_size_level0 / downscale_factor)

        for i, ((x_lvl0, y_lvl0)) in enumerate(coords_patches_selected_by_agent_ls):
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
    patch_size_level0,
    downscale_factor,
    args = None,
    is_save=True
):

    # Overlay Similar patches (light orange → dark orange)
    if coords_similar_patches_selected_by_agent_ls:
        norm = plt.Normalize(0.0, 0.2)
        colormap = cm.get_cmap("Oranges_r")
        box_w = int(patch_size_level0 / downscale_factor)

        for i, (patch_group) in enumerate(coords_similar_patches_selected_by_agent_ls):
            frac = i / max(1, len(coords_similar_patches_selected_by_agent_ls) - 1) * 0.2
            color = tuple(int(255 * c) for c in colormap(norm(frac))[:3])
            for (x_lvl0, y_lvl0) in patch_group:
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


if __name__ == "__main__":

    main()