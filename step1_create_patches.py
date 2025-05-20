'''
First step to process is to remove WSI is to segment the WSI and remove the region where the tissue sample for
a patch is less than some threshold.

For CAMELYON16 dataset
python step1_create_patches.py --source SOURCE_DIR --save_dir SAVE_DIR --extension tif --patch_level 3

FOR TCGA-NSCLC dataset
python step1_create_patches.py --source SOURCE_DIR --save_dir SAVE_DIR --extension svs --patch_level 2

'''

import argparse
import csv
import os
import time
from glob import glob

import numpy as np
import pandas as pd

from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.batch_process_utils import initialize_df
from wsi_core.wsi_utils import StitchCoords


def get_arguments() :

    parser = argparse.ArgumentParser(description='seg and patch')
    parser.add_argument('--source', type=str, default= None, help='path to folder containing raw wsi image files')
    parser.add_argument('--step_size', type=int, default=256, help='step_size')
    parser.add_argument('--patch_size', type=int, default=256, help='patch_size')
    parser.add_argument('--patch', default=True, action='store_true')
    parser.add_argument('--seg', default=True, action='store_true')
    parser.add_argument('--stitch', default=True, action='store_true')
    parser.add_argument('--no_auto_skip', default=True, action='store_false')
    parser.add_argument('--save_dir', type=str, default= None, help='directory to save processed data')
    parser.add_argument('--extension', default='tif', help='extension to processes data type, e.g. *.svs, *.tif')
    parser.add_argument('--patch_level', type=int, default=3, help='downsample level at which to patch')
    parser.add_argument('--time_csv', type=str, default=None, help='store the patch time per slide')
    parser.add_argument('--time_csv_col_name', type=str, default=None, help='column name for time_csv_col_name')
    parser.add_argument('--preset', default=None, type=str, help='predefined profile of default segmentation and filter parameters (.csv)')
    parser.add_argument('--process_list', type=str, default=None, help='name of list of images to process with parameters (.csv)')

    return parser


def update_arguments(args) :

    if args.time_csv is None :
        args.time_csv = os.path.join('outputs/time', 'time_patch.csv')
    if args.time_csv_col_name is None :
        args.time_csv_col_name = 'patch_time'

    return args

def create_time_df(csv_file_path = None, column_name_ls = None) :

    # Required columns -
    required_columns = column_name_ls

    # Step 1: Create the file if it doesn't exist
    if not os.path.exists(csv_file_path) :
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=required_columns)
            writer.writeheader()
        print(f"File '{csv_file_path}' created with headers: {required_columns}")

    # Step 2: File exists – check and add missing columns
    else:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
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
            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.DictWriter(file, fieldnames=updated_columns)
                writer.writeheader()
                writer.writerows(data)
            print(f"File '{csv_file_path}' updated with missing columns: {missing}")
        else:
            print(f"'{csv_file_path}' already contains required columns: {required_columns}")

    time_df = pd.read_csv(csv_file_path)
    return time_df

def stitching(file_path, wsi_object, downscale=64):
    start = time.time()
    heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0, 0, 0), alpha=-1, draw_grid=False)
    total_time = time.time() - start

    return heatmap, total_time


def segment(WSI_object, seg_params, filter_params):
    ### Start Seg Timer
    start_time = time.time()

    # Segment
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time
    return WSI_object, seg_time_elapsed


def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()
    # Patch
    file_path = WSI_object.process_contours(**kwargs)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def walk_dir(data_dir,
             file_types=['.kfb', '.tif', '.svs', '.ndpi', '.mrxs', '.hdx', '.sdpc', '.mdsx', '.tiff', '.tmap']):
    path_list = []
    for dirpath, dirnames, files in os.walk(data_dir):
        for f in files:
            for this_type in file_types:
                if f.lower().endswith(this_type):
                    path_list.append(os.path.join(dirpath, f))
                    break
    return path_list


def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir,
                  patch_size=256, step_size=256,
                  seg_params={'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                              'keep_ids': 'none', 'exclude_ids': 'none'},
                  filter_params={'a_t': 100, 'a_h': 16, 'max_n_holes': 8},
                  vis_params={'vis_level': -1, 'line_thickness': 500},
                  patch_params={'use_padding': True, 'contour_fn': 'four_pt'},
                  patch_level=1,
                  use_default_params=False,
                  seg=False, save_mask=True,
                  stitch=False,
                  patch=False, auto_skip=True,
                  process_list=None,
                  time_df = None,
                  time_df_column_name = None ):


    slides = glob(source + f'/*.{args.extension}')
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
    print("Slides: ", len(slides))

    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)

    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)

    legacy_support = 'a' in df.keys()
    if legacy_support:
        print('detected legacy segmentation csv file, legacy support enabled')
        df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
                          'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
                          'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
                          'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
                          'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i / total, i, total))
        print('processing {}'.format(slide))

        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide.split('/')[-1])

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Initialize WSI
        full_path = slide
        try:
            WSI_object = WholeSlideImage(full_path)
        except:
            print('cannot open file', full_path)
            continue
        try:
            WSI_object.initXML(os.path.splitext(full_path)[0] + '.xml')
        except:
            print('no xml annos found')
            pass

        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()

        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}

            for key in vis_params.keys():
                if legacy_support and key == 'vis_level':
                    df.loc[idx, key] = -1
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                if legacy_support and key == 'a_t':
                    old_area = df.loc[idx, 'a']
                    seg_level = df.loc[idx, 'seg_level']
                    scale = WSI_object.level_downsamples[seg_level]
                    adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
                    current_filter_params.update({key: adjusted_area})
                    df.loc[idx, key] = adjusted_area
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                if legacy_support and key == 'seg_level':
                    df.loc[idx, key] = -1
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        if current_vis_params['vis_level'] < 0:

            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0

            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        w, h = WSI_object.level_dim[current_seg_params['seg_level']]
        if w * h > 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        seg_time_elapsed = -1
        if seg:
            try:
                WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params)
            except:
                continue
        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id + '.jpg')
            mask.save(mask_path)

        patch_time_elapsed = -1  # Default time

        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size,
                                         'save_path': patch_save_dir})
            file_path, patch_time_elapsed = patching(WSI_object=WSI_object, **current_patch_params, )

        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id + '.h5')
            if os.path.isfile(file_path):
                heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
                stitch_path = os.path.join(stitch_save_dir, slide_id + '.jpg')
                heatmap.save(stitch_path)

        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

        total_time_per_slide = seg_time_elapsed + patch_time_elapsed + stitch_time_elapsed
        time_df.loc[time_df['slide_name'] == slide_id, time_df_column_name] = total_time_per_slide

        # Check if slide_id exists
        if slide_id in time_df['slide_name'].values:
            time_df.loc[time_df['slide_name'] == slide_id, time_df_column_name] = total_time_per_slide
        else:
            # Create new row
            new_row = {col: "" for col in time_df.columns}
            new_row["slide_name"] = slide_id
            new_row[time_df_column_name] = total_time_per_slide
            time_df = pd.concat([time_df, pd.DataFrame([new_row])], ignore_index=True)

    seg_times /= total
    patch_times /= total
    stitch_times /= total

    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)

    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))

    return seg_times, patch_times, time_df


if __name__ == '__main__':

    parser = get_arguments()
    args = parser.parse_args()
    args = update_arguments(args= args)

    # Create the required directories
    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)
    else:
        process_list = None


    # Storing them in dictionary
    directories = {'source': args.source,
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   'mask_save_dir': mask_save_dir,
                   'stitch_save_dir': stitch_save_dir}

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    # Define parameters for segmentation, visulalization and patching [ Default ----> TRY not to modify this]
    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    if args.preset:  # For custom parameters for segmentation, filters, visualization and patching

        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]

    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}


    # Creating dataframe to store the time for patching for each slide
    time_df = create_time_df(csv_file_path= args.time_csv, column_name_ls = ['slide_name', args.time_csv_col_name])

    seg_times, patch_times, time_df = seg_and_patch(**directories, **parameters,
                                                    patch_size=args.patch_size,
                                                    step_size=args.step_size,
                                                    seg=args.seg,
                                                    use_default_params=False,
                                                    save_mask=True,
                                                    stitch=args.stitch,
                                                    patch_level=args.patch_level,
                                                    patch=args.patch,
                                                    process_list=process_list,
                                                    auto_skip=args.no_auto_skip,
                                                    time_df = time_df,
                                                    time_df_column_name= args.time_csv_col_name)

    # Save csv file
    time_df.to_csv(args.time_csv, index=False)
