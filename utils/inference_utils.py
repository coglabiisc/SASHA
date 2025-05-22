import os
import h5py
import openslide
import torch
import numpy as np
from torch.utils.data import DataLoader


from datasets.dataset_h5 import Whole_Slide_Bag_FP
from models_features_extraction import get_encoder
from step1_create_patches import create_time_df, seg_and_patch
from step2_extract_features import compute_w_loader


class Helper():
    def __init__(self, conf, classifier):
        self.conf = conf
        # declaring necessary variables for create patch and segmentation ( STEP1)
        self.step1_save_dir = os.path.join(self.conf.save_dir, 'STEP1')
        os.makedirs(self.step1_save_dir, exist_ok=True)
        self.patch_save_dir = os.path.join(self.step1_save_dir, 'patches')
        self.mask_save_dir = os.path.join(self.step1_save_dir, 'masks')
        self.stitch_save_dir = os.path.join(self.step1_save_dir, 'stiches')
        os.makedirs(self.patch_save_dir, exist_ok=True)
        os.makedirs(self.mask_save_dir, exist_ok=True)
        os.makedirs(self.stitch_save_dir, exist_ok=True)

        self.directories = {'source': self.conf.source,
                'save_dir': self.step1_save_dir,
                'patch_save_dir': self.patch_save_dir,
                'mask_save_dir': self.mask_save_dir,
                'stitch_save_dir': self.stitch_save_dir,}
        
        seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                    'keep_ids': 'none', 'exclude_ids': 'none'}
        filter_params = {'a_t': 100, 'a_h': 16, 'max_n_holes': 8}
        vis_params = {'vis_level': -1, 'line_thickness': 250}
        patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}
        self.parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}
        

        #declaring the necessary variables for feature extraction
        self.step2_save_dir = os.path.join(self.conf.save_dir, 'STEP2')
        os.makedirs(self.step2_save_dir, exist_ok=True)
        self.feat_dir = os.path.join(self.step2_save_dir, f'features')
        self.h5_dir = os.path.join(self.feat_dir, 'lr/h5_files')
        self.pt_dir = os.path.join(self.feat_dir, 'lr/pt_files')
        os.makedirs(self.h5_dir, exist_ok=True)
        os.makedirs(self.pt_dir, exist_ok=True)

        self.feature_extractor, self.img_transforms =  get_encoder(model_name=conf.backbone,
                                                     pretrain=conf.pretrain, 
                                                     )
        self.feature_extractor.eval()
        self.feature_extractor.to(conf.device)


        # declaring the necessary variables for STEP3 - feature aggregation of zoomed in patch
        self.feature_aggregator = classifier
        

    def create_patches(self, slide, patch_level, step_size, patch_size):
        time_csv = os.path.join(f'{self.step1_save_dir}', 'time_patch.csv')
        time_csv_col_name = 'patch_time'

        time_df = create_time_df(csv_file_path=time_csv, column_name_ls=['slide_name', time_csv_col_name])
        seg_times, patch_times, time_df = seg_and_patch(**self.directories, **self.parameters,
                                                        patch_size=256,
                                                        step_size=256,
                                                        seg=True,
                                                        use_default_params=False,
                                                        save_mask=True,
                                                        stitch=True,
                                                        patch_level=patch_level,
                                                        patch=True,
                                                        process_list=None,
                                                        auto_skip=True,
                                                        time_df=time_df,
                                                        time_df_column_name=time_csv_col_name,
                                                        slide_name = slide
                                                        )
        time_df.to_csv(time_csv, index=False)



    def extract_lr_features(self, slide, label, slide_ext, patch_level_low_res, patch_level_high_res):

        h5_path_lr = os.path.join(self.h5_dir, f'{slide}_patch_feats_pretrain_medical.h5')
        h5file_lr = h5py.File(h5_path_lr, "w")

        slide_path = os.path.join(self.conf.source, f'{slide}.{slide_ext}')
        h5_file_path = os.path.join(self.step1_save_dir, f'patches/{slide}.h5')

        loader_kwargs = {'num_workers': 8, 'pin_memory': True} if self.conf.device.type == "cuda" else {}

        wsi = openslide.open_slide(slide_path)

        dataset = Whole_Slide_Bag_FP(file_path=h5_file_path,
                                     wsi=wsi,
                                     img_transforms=self.img_transforms,
                                     extract_high_res_features=False,
                                     patch_level_high_res=patch_level_high_res,
                                     patch_level_low_res=patch_level_low_res,
                                     dataset_name=self.conf.dataset)

        loader = DataLoader(dataset=dataset, batch_size=self.conf.batch_size_lr, **loader_kwargs)

        feature, coords, total_time = compute_w_loader(loader=loader, model=self.feature_extractor, verbose=1, extract_high_res_features=False, device=self.conf.device)
        
        slide_grp = h5file_lr.create_group(slide)
        slide_grp.create_dataset('feat', data=feature.astype(np.float16))
        slide_grp.create_dataset('coords', data=coords)
        slide_grp.attrs['label'] = label
        torch.save(torch.from_numpy(feature), os.path.join(self.pt_dir, slide + '.pt'))

        return feature, coords, wsi
    


    def get_embedding(self, coords, wsi, patch_level_low_res, patch_level_high_res, step_size, patch_size):
        high_res_patches = self.get_patches_for_selected_action(coords, wsi, patch_level_low_res, patch_level_high_res, step_size, patch_size)
        high_res_patches = torch.stack(high_res_patches).to(self.conf.device)
        hr_features = self.feature_extractor(high_res_patches)
        hr_features = hr_features.unsqueeze(0)
        agg_feat = self.feature_aggregator.get_hr_fa(hr_features)
        return agg_feat
    

    def get_patches_for_selected_action(self, coords, wsi, patch_level_low_res, patch_level_high_res, step_size, patch_size):

        low_res_level= patch_level_low_res
        high_res_level= patch_level_high_res
        x = coords[0]
        y = coords[1]
        scale_factor = 2 ** (low_res_level - high_res_level)
        step_size = 256
        patch_size = 256
        cnt = 0
        high_resolution_imgs = []
        for x_step_idx in range(scale_factor):
            for y_step_idx in range(scale_factor):
                x_curr = x + x_step_idx  * 2  *  step_size
                y_curr = y + y_step_idx  * 2  *  step_size
                patch = wsi.read_region((x_curr, y_curr), high_res_level,
                                                (patch_size, patch_size)).convert("RGB")
                cnt += 1
                patch = self.img_transforms(patch)
                high_resolution_imgs.append(patch)
        return high_resolution_imgs

    
