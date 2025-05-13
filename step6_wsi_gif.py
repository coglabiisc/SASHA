import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
import openslide
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon
from tqdm import tqdm
import h5py
import openslide
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from shapely.geometry import Polygon
from matplotlib.patches import Polygon as MplPolygon, Rectangle
import imageio
import torch
from matplotlib.lines import Line2D


# Paths
slide_name = 'test_021'
wsi_path = f"/media/internal_8T/karm/karm_8T_backup/camelyon16/images/all/{slide_name}.tif"
xml_path = f"/media/internal_8T/karm/karm_8T_backup/camelyon16/annotations/{slide_name}.xml"  # Assuming annotations are in XML
output_gif_path = f"/media/internal_8T/naman/rlogist/images/{slide_name}_tr.gif"
action_sequences_path = '/media/internal_8T/naman/rlogist/images/combined_patches_frac_0.2_terminal_reward.pt'
h5_file_path = "/home/tarun/rlogist/ACMIL/CAMELYON16_features/VIT_medical_ssl_level_1_3/lr/h5_files/patch_feats_pretrain_medical_ssl.h5"
attention_path = "/media/internal_8T/naman/results_2025/results/camleyon16/results_vit_hacmil_level_1_3/wsi_tumor_frac/attention_1_3.pt"
level = 3

action_sequences_data = torch.load(action_sequences_path, map_location= 'cpu')
attention_data = torch.load(attention_path, map_location= 'cpu')


# Loading Coordinates for the selected file

# 1. Specify the WSI filename whose coordinates you want
with h5py.File(h5_file_path, "r") as h5_file:

    # Check available keys (optional, to understand the structure)
    print("Available keys:", list(h5_file.keys()))

    # Assuming the file structure has keys corresponding to WSI filenames
    if slide_name in h5_file:
        # Extract coordinates
        coordinates = h5_file[slide_name]["coords"][:]
        print(f"Coordinates for {slide_name}:")
        print(coordinates)
    else:
        print(f"WSI {slide_name} not found in the dataset.")

print(len(coordinates), len(coordinates[0]), len(action_sequences_data[slide_name]))


# Creating GIF

# Load WSI
slide = openslide.OpenSlide(wsi_path)

# Get Level 3 (5X) properties
level = 3
downscale_factor = slide.level_downsamples[level]  # Scaling factor from level 0 to level 3
wsi_size = slide.level_dimensions[level]  # (width, height) at level 3

# Read WSI at level 3
wsi_img = slide.read_region((0, 0), level, wsi_size).convert("RGB")

# Parse XML for annotation coordinates
tree = ET.parse(xml_path)
root = tree.getroot()

annotations = []
for annotation in root.findall(".//Annotation"):
    coords = []
    for coord in annotation.findall(".//Coordinate"):
        x = float(coord.get("X")) / downscale_factor  # Scale to level 3
        y = float(coord.get("Y")) / downscale_factor  # Scale to level 3
        coords.append((x, y))
    annotations.append(Polygon(coords))  # Store as Shapely Polygon

# Example sequence of patches (list of coordinates in (x, y, width, height) format at level 0)
action = action_sequences_data[slide_name]
patch_sequence = []
for id in action:
    patch_sequence.append(coordinates[id])


# Convert patch coordinates to Level 3
patch_sequence = [
    (x / downscale_factor, y / downscale_factor)
    for x, y in patch_sequence
]

# Create frames for the GIF
frames = []
durations = []

for i, (x, y) in tqdm(enumerate(patch_sequence), desc="Processing Patches", total=len(patch_sequence)):
    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(wsi_img)

    # Draw existing annotations (red outlines)
    for polygon in annotations:
        patch = MplPolygon(np.array(polygon.exterior.coords), edgecolor='r', fill=False, linewidth=2)
        ax.add_patch(patch)

    # Get the tumor fraction for the current patch
    patch_area = attention_data[slide_name]['lr_patch_tumor_area_percentage'][action_sequences_data[slide_name][i]]

    # Choose color based on tumor area %
    if patch_area == 0:
        edgecolor = 'lightgreen'
        label = "No Tumor"
    elif 0 < patch_area < 10:
        edgecolor = 'orange'
        label = "Low Tumor (0–10%)"
    elif 10 <= patch_area < 50:
        edgecolor = 'purple'
        label = "Moderate Tumor (10–50%)"
    else:  # patch_area >= 50
        edgecolor = 'red'
        label = "High Tumor (50–100%)"

    # Highlight the patch
    rect = Rectangle((x, y), 512, 512, edgecolor=edgecolor, facecolor='none', linewidth=3)
    ax.add_patch(rect)

    # Add title with tumor information
    ax.set_title(f"Patch {i+1} | Tumor Fraction: {patch_area:.2f}% - {label}", fontsize=16)
    ax.axis("off")

    # Add legend only in the first frame
    legend_elements = [
        Line2D([0], [0], color='lightgreen', lw=4, label='No Tumor'),
        Line2D([0], [0], color='orange', lw=4, label='Low Tumor (0–10%)'),
        Line2D([0], [0], color='purple', lw=4, label='Moderate Tumor (10–50%)'),
        Line2D([0], [0], color='red', lw=4, label='High Tumor (50–100%)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

    # Convert plot to image and save
    fig.canvas.draw()
    frame = np.array(fig.canvas.renderer.buffer_rgba())
    frames.append(frame)

    if patch_area == 0 :
        durations.append(0.5)
    else :
        durations.append(1.5)

    plt.close(fig)

# Save as GIF
imageio.mimsave(output_gif_path, frames, duration= durations)
print(f"GIF saved as {output_gif_path}")