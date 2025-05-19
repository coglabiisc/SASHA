import json
import os
import random

import h5py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# If we have a single h5 file for all the slides

def split_dataset_camelyon(file_path, conf):

    h5_data = h5py.File(os.path.join(file_path, f'patch_feats_pretrain_{conf.pretrain}.h5'), 'r')
    split_file_path = './dataset_csv/%s/splits/split_%s.json'%(conf.dataset, conf.seed)

    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']
    else:
        print(f"Enter a valid split path : {split_file_path}")
        exit()

    train_split, val_split, test_split = {}, {}, {}
    for (names, split) in [(train_names, train_split), (val_names, val_split), (test_names, test_split)]:
        for name in names:
            slide = h5_data[name]

            label = slide.attrs['label']
            feat = slide['feat'][:]
            coords = slide['coords'][:]

            split[name] = {'input': feat, 'coords': coords, 'label': label}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names


def split_dataset_fglobal_camelyon(file_path_level1, file_path_level3, conf):
    h5_data_level1 = h5py.File(os.path.join(file_path_level1, f'patch_feats_pretrain_{conf.pretrain}.h5'), 'r')
    h5_data_level3 = h5py.File(os.path.join(file_path_level3, f'patch_feats_pretrain_{conf.pretrain}.h5'), 'r')

    split_file_path = './splits/%s/split_%s.json' % (conf.dataset, conf.seed)

    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']
    else:
        print(f"Enter a valid split path : {split_file_path}")
        exit()

    train_split1, val_split1, test_split1 = {}, {}, {}
    train_split3, val_split3, test_split3 = {}, {}, {}
    for (names, split1, split3) in [(train_names, train_split1, train_split3), (val_names, val_split1, val_split3),
                                    (test_names, test_split1, test_split3)]:
        for name in names:
            slide1 = h5_data_level1[name]

            label = slide1.attrs['label']
            feat1 = slide1['feat'][:]
            # coords = slide1['coords'][:]

            slide3 = h5_data_level3[name]
            feat3 = slide3['feat'][:]

            split1[name] = {'input': feat1, 'label': label}
            split3[name] = {'input': feat3, 'label': label}
    h5_data_level1.close()
    h5_data_level3.close()
    return train_split1, train_split3, train_names, val_split1, val_split3, val_names, test_split1, test_split3, test_names


def split_dataset_tcga(file_path, conf):
    h5_data = h5py.File(os.path.join(file_path, f'patch_feats_pretrain_{conf.pretrain}.h5'), 'r')

    # Loading the tcga.csv file for complete dataset
    df = pd.read_csv(conf.data_csv)
    name_mapping = dict(zip(df["slide_id_short"], df["slide_file_name"]))

    # Loading splits
    split_file_path = './dataset_csv/%s/splits/split_%s.json'%(conf.dataset, conf.seed)

    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']
    else :
        print(f"Enter a valid split path : {split_file_path}")
        exit()

    train_split, val_split, test_split = {}, {}, {}
    for (names, split) in [(train_names, train_split), (val_names, val_split), (test_names, test_split)]:
        for name in names:
            # Loading value from h5_data
            slide = h5_data[name]  # Adding this to remove the .svs extension from name

            label = slide.attrs['label']
            feat = slide['feat'][:]
            coords = slide['coords'][:]

            split[name] = {'input': feat, 'coords': coords, 'label': label}

    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names


def split_dataset_fglobal_tcga(file_path_level1, file_path_level3, conf):

    h5_data_level1 = h5py.File(os.path.join(file_path_level1, f'patch_feats_pretrain_{conf.pretrain}.h5'), 'r')
    h5_data_level3 = h5py.File(os.path.join(file_path_level3, f'patch_feats_pretrain_{conf.pretrain}.h5'), 'r')

    # Loading the tcga.csv file for complete dataset
    df = pd.read_csv(conf.data_csv)
    name_mapping = dict(zip(df["slide_id_short"], df["slide_file_name"]))

    # Loading splits
    split_file_path = './dataset_csv/%s/splits/split_%s.json' % (conf.dataset, conf.seed)

    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']
    else:
        print(f"Enter a valid split path : {split_file_path}")
        exit()

    train_split1, val_split1, test_split1 = {}, {}, {}
    train_split3, val_split3, test_split3 = {}, {}, {}
    for (names, split1, split3) in [(train_names, train_split1, train_split3), (val_names, val_split1, val_split3),
                                    (test_names, test_split1, test_split3)]:
        for name in names:
            slide1 = h5_data_level1[name]

            label = slide1.attrs['label']
            feat1 = slide1['feat'][:]

            slide3 = h5_data_level3[name]
            feat3 = slide3['feat'][:]

            split1[name] = {'input': feat1, 'label': label}
            split3[name] = {'input': feat3, 'label': label}

    h5_data_level1.close()
    h5_data_level3.close()
    return train_split1, train_split3, train_names, val_split1, val_split3, val_names, test_split1, test_split3, test_names


def split_dataset_bracs(file_path, conf):

    csv_path = './dataset_csv/bracs.csv'
    slide_info = pd.read_csv(csv_path).set_index('slide_id')
    class_transfer_dict_3class = {0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2}
    class_transfer_dict_2class = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}

    h5_data = h5py.File(file_path, 'r')
    slide_names = list(h5_data.keys())
    train_split, val_split, test_split = {}, {}, {}
    train_names, val_names, test_names = [], [], []
    for slide_id in slide_names:
        slide = h5_data[slide_id]

        label = slide.attrs['label']
        if conf.n_class == 3:
            label = class_transfer_dict_3class[label]
        elif conf.n_class == 2:
            label = class_transfer_dict_2class[label]

        feat = slide['feat'][:]
        coords = slide['coords'][:]

        split_info = slide_info.loc[slide_id]['split_info']
        if split_info == 'train':
            train_names.append(slide_id)
            train_split[slide_id] = {'input': feat, 'coords': coords, 'label': label}
        elif split_info == 'val':
            val_names.append(slide_id)
            val_split[slide_id] = {'input': feat, 'coords': coords, 'label': label}
        else:
            test_names.append(slide_id)
            test_split[slide_id] = {'input': feat, 'coords': coords, 'label': label}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names


def split_dataset_lct(file_path, conf):


    class_transfer_dict_4class = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}
    class_transfer_dict_2class = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

    h5_data = h5py.File(file_path, 'r')
    split_file_path = './splits/%s/split_%s.json' % (conf.dataset, conf.seed)
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']
    else:
        slide_names = list(h5_data.keys())
        train_val_names, test_names = train_test_split(slide_names, test_size=0.2)
        train_names, val_names = train_test_split(train_val_names, test_size=0.25)

    train_split, val_split, test_split = {}, {}, {}
    for (names, split) in [(train_names, train_split), (val_names, val_split), (test_names, test_split)]:
        for name in names:
            slide = h5_data[name]
            label = slide.attrs['label']

            if conf.n_class == 4:
                label = class_transfer_dict_4class[label]
            elif conf.n_class == 2:
                label = class_transfer_dict_2class[label]

            if conf.B > 1:
                feat = np.zeros([conf.n_patch, conf.feat_d])
                coords = 0
                n = min(slide['feat'][:].shape[0], conf.n_patch)
                feat[:n] = slide['feat'][:]

            else:

                feat = slide['feat'][:]
                coords = 0

            split[name] = {'input': feat, 'coords': coords, 'label': label}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names


class HDF5_feat_dataset2(object):
    def __init__(self, data_dict, data_names):
        self.data_dict = data_dict
        self.data_names = data_names

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        slide_dict = self.data_dict[self.data_names[index]]
        slide_dict['slide_name'] = self.data_names[index]
        return slide_dict


class HDF5_feat_dataset4(object):
    def __init__(self, data_dict1, data_dict3, data_names):
        self.data_dict1 = data_dict1
        self.data_dict3 = data_dict3
        self.data_names = data_names

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        slide_dict = {}
        slide_dict['hr'] = self.data_dict1[self.data_names[index]]['input']
        slide_dict['lr'] = self.data_dict3[self.data_names[index]]['input']
        slide_dict['slide_name'] = self.data_names[index]
        slide_dict['label'] = self.data_dict1[self.data_names[index]]['label']
        return slide_dict


def generate_fewshot_dataset(train_split, train_names, num_shots):
    if num_shots < len(train_names) and num_shots > 0:
        labels = [it['label'] for it in train_split.values()]
        train_split_ = {}
        train_names_ = []
        for l in set(labels):
            indices = [index for index, element in enumerate(labels) if element == l]
            selected_indices = random.sample(indices, num_shots)
            names = [train_names[index] for index in selected_indices]
            train_names_ += names
            split = {name: train_split[name] for name in names}
            train_split_.update(split)
        return train_split_, train_names_
    else:
        return train_split, train_names


def build_HDF5_feat_dataset(file_path, conf):

    if conf.dataset == 'camelyon16':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_camelyon(file_path, conf)
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)

    elif conf.dataset == 'tcga' :
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_tcga(file_path, conf)
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split,val_names), HDF5_feat_dataset2(test_split, test_names)

    elif conf.dataset == 'bracs':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_bracs(file_path, conf)
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)

    elif conf.dataset == 'lct':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_lct(file_path, conf)
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)


def build_HDF5_feat_dataset_2(file_path_level1, file_path_level3, conf):

    if conf.dataset == 'camelyon16' :
        train_split1, train_split3, train_names, val_split1, val_split3, val_names, test_split1, test_split3, test_names = split_dataset_fglobal_camelyon(file_path_level1, file_path_level3, conf)
        return HDF5_feat_dataset4(train_split1, train_split3, train_names), HDF5_feat_dataset4(val_split1, val_split3, val_names), HDF5_feat_dataset4(test_split1, test_split3, test_names)

    elif conf.dataset == 'tcga' :
        train_split1, train_split3, train_names, val_split1, val_split3, val_names, test_split1, test_split3, test_names = split_dataset_fglobal_tcga(file_path_level1, file_path_level3, conf)
        return HDF5_feat_dataset4(train_split1, train_split3, train_names), HDF5_feat_dataset4(val_split1, val_split3,val_names), HDF5_feat_dataset4(test_split1, test_split3, test_names)



if __name__ == '__main__':
    print("K")