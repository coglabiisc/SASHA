import h5py

file_path = '/media/internal_8T/naman/results_2025/camelyon16_features/vit_medical_ssl_1_3_final/level1_path/camelyon16_features_VIT_med_level_1_3/hr/patch_feats_pretrain_medical_ssl.h5'

with h5py.File(file_path, 'r') as h5_file:
    group_name = '/normal_001'

    if group_name in h5_file:
        group = h5_file[group_name]
        print(f"Group: {group_name} has {len(group)} members:")

        for name, item in group.items():
            print(f"  - Name: {name}")
            print(f"    Type: {'Group' if isinstance(item, h5py.Group) else 'Dataset'}")

            if isinstance(item, h5py.Dataset):
                print(f"    Shape: {item.shape}")
                print(f"    Dtype: {item.dtype}")
                print(f"    Sample data (first row): {item[0] if item.shape[0] > 0 else 'Empty'}")
            elif isinstance(item, h5py.Group):
                print(f"    Subgroup with {len(item)} members")
    else:
        print(f"Group '{group_name}' not found.")
