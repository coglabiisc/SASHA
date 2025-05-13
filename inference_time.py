import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Inference Time Evaluation

file_path_patch = "/media/internal_8T/naman/rlogist/time_patch.csv"
df_patch = pd.read_csv(file_path_patch)

file_path_feat = "/media/internal_8T/naman/rlogist/time_feat.csv"
df_feat = pd.read_csv(file_path_feat)

file_path_hacmil = "/media/internal_8T/naman/rlogist/time_hacmil.csv"
df_hacmil = pd.read_csv(file_path_hacmil)

file_path_rl = "/media/internal_8T/naman/rlogist/time_rl.csv"
df_rl = pd.read_csv(file_path_rl)

file_path_rl_01 = "/media/internal_8T/naman/rlogist/time_rl_01.csv"
df_rl_01 = pd.read_csv(file_path_rl_01)

# Step 1 - Loading all the test slide names
slide_names = df_patch['slide_name']
test_slides = slide_names[slide_names.str.startswith('test')].tolist()
print(test_slides)
print("Total Test Slides  : ", len(test_slides))

# Step 2 - Get Patch Time
df_patch_test = df_patch[df_patch['slide_name'].str.startswith('test')]
patch_columns = [col for col in df_patch.columns if col.startswith('patch_level_')]
df_patch_test['patching_mean_time_per_slide'] = df_patch_test[patch_columns].mean(axis=1)
df_patch_test['patching_std_time_per_slide'] = df_patch_test[patch_columns].std(axis=1)

# Step 3 - Get Feature Extraction Time
df_feat_test = df_feat[df_feat['slide_name'].str.startswith('test')]

hr_feat_columns = [col for col in df_feat.columns if col.startswith('feature_extraction_b_512_level_1_iter_')]
df_feat_test['feat_hr_mean_time_per_slide'] = df_feat_test[hr_feat_columns].mean(axis=1)
df_feat_test['feat_hr_std_time_per_slide'] = df_feat_test[hr_feat_columns].std(axis=1)

lr_feat_columns = [col for col in df_feat.columns if col.startswith('feature_extraction_b_512_level_3_iter_')]
df_feat_test['feat_lr_mean_time_per_slide'] = df_feat_test[lr_feat_columns].mean(axis=1)
df_feat_test['feat_lr_std_time_per_slide'] = df_feat_test[lr_feat_columns].std(axis=1)

hr_feat_columns = [col for col in df_feat.columns if col.startswith('feature_extraction_b_16_level_1_iter_')]
df_feat_test['feat_hr_b16__mean_time_per_slide'] = df_feat_test[hr_feat_columns].mean(axis=1)
df_feat_test['feat_hr_b16_std_time_per_slide'] = df_feat_test[hr_feat_columns].std(axis=1)

# Step 4 - HACMIL time
df_hacmil_test = df_hacmil[df_hacmil['slide_name'].str.startswith('test')]
patch_columns = [col for col in df_hacmil.columns if col.startswith('hacmil_iter_')]
df_hacmil_test['hacmil_mean_time_per_slide'] = df_hacmil_test[patch_columns].mean(axis=1)
df_hacmil_test['hacmil_std_time_per_slide'] = df_hacmil_test[patch_columns].std(axis=1)

# Step 5 - SASHA - 0.2 time
df_rl_test = df_rl[df_rl['slide_name'].str.startswith('test')]
patch_columns = [col for col in df_rl.columns if col.startswith('sasha_iter_')]
df_rl_test['sasha_mean_time_per_slide'] = df_rl_test[patch_columns].mean(axis=1)
df_rl_test['sasha_std_time_per_slide'] = df_rl_test[patch_columns].std(axis=1)

# SASHA - 0.1 time
df_rl_01_test = df_rl_01[df_rl_01['slide_name'].str.startswith('test')]
patch_columns = [col for col in df_rl.columns if col.startswith('sasha_iter_')]
df_rl_01_test['sasha_mean_time_per_slide_01'] = df_rl_01_test[patch_columns].mean(axis=1)
df_rl_01_test['sasha_std_time_per_slide_01'] = df_rl_01_test[patch_columns].std(axis=1)

# Now based on the test slide name need to combine this final column in 1 table

# Important columns from each table
df_patch_final = df_patch_test[['slide_name', 'patching_mean_time_per_slide', 'patching_std_time_per_slide']]
df_feat_final = df_feat_test[['slide_name',
                              'feat_hr_mean_time_per_slide', 'feat_hr_std_time_per_slide',
                              'feat_lr_mean_time_per_slide', 'feat_lr_std_time_per_slide',
                              'feat_hr_b16__mean_time_per_slide', 'feat_hr_b16_std_time_per_slide']]
df_hacmil_final = df_hacmil_test[['slide_name', 'hacmil_mean_time_per_slide', 'hacmil_std_time_per_slide']]
df_rl_final = df_rl_test[['slide_name', 'sasha_mean_time_per_slide', 'sasha_std_time_per_slide']]
df_rl_01_final = df_rl_01_test[['slide_name', 'sasha_mean_time_per_slide_01', 'sasha_std_time_per_slide_01']]

# Merge everything on 'slide_name'
df_final = df_patch_final.merge(df_feat_final, on='slide_name', how='inner')
df_final = df_final.merge(df_hacmil_final, on='slide_name', how='inner')
df_final = df_final.merge(df_rl_final, on='slide_name', how='inner')
df_final = df_final.merge(df_rl_01_final, on='slide_name', how='inner')

# Save dataframe
output_path = "/media/internal_8T/naman/rlogist/time_test_final.csv"
df_final.to_csv(output_path, index=False)
print(f"✅ Saved combined dataframe successfully at: {output_path}")


# Now based on the columns in final table need to combine following -

# For H.R. ---> Columns total_time = patching_mean_time_per_slide + feat_hr_mean_time_per_slide + hacmil_mean_time_per_slide
# For SASHA ---> Columns total_time = patching_mean_time_per_slide + feat_lr_mean_time_per_slide + 0.2 * feat_hr_b16__mean_time_per_slide + sasha_mean_time_per_slide

# -----------------------------------------------------------------------------
# Now based on the columns in final table, we need to combine for total_time:
# -----------------------------------------------------------------------------

# For Fraction 0.2 --->

# 1. For H.R. ---->
# total_time_hr = patching + HR feature extraction + HACMIL
df_final['total_time_hr'] = (
    df_final['patching_mean_time_per_slide'] +
    df_final['feat_hr_mean_time_per_slide'] +
    df_final['hacmil_mean_time_per_slide']
)

# 2. For SASHA (with 0.2 fraction on B16 feature) ---->
# total_time_sasha = patching + LR feature extraction + (0.2 * B16 HR feature) + SASHA
df_final['total_time_sasha_02'] = (
    df_final['patching_mean_time_per_slide'] +
    df_final['feat_lr_mean_time_per_slide'] +
    0.2 * df_final['feat_hr_b16__mean_time_per_slide'] +
    df_final['sasha_mean_time_per_slide']
)

df_final['total_time_sasha_01'] = (
    df_final['patching_mean_time_per_slide'] +
    df_final['feat_lr_mean_time_per_slide'] +
    0.1 * df_final['feat_hr_b16__mean_time_per_slide'] +
    df_final['sasha_mean_time_per_slide_01']
)


df_final['speed_up_02'] = (df_final['total_time_hr'] / df_final['total_time_sasha_02'])
df_final['speed_up_01'] = (df_final['total_time_hr'] / df_final['total_time_sasha_01'])

# ------------------------
# 1. Print Average Times
# ------------------------

# Calculate mean times across all test slides
average_times = {
    'patching_mean_time_per_slide': df_final['patching_mean_time_per_slide'].mean(),
    'feat_hr_mean_time_per_slide': df_final['feat_hr_mean_time_per_slide'].mean(),
    'feat_lr_mean_time_per_slide': df_final['feat_lr_mean_time_per_slide'].mean(),
    'feat_hr_b16__mean_time_per_slide': df_final['feat_hr_b16__mean_time_per_slide'].mean(),
    'hacmil_mean_time_per_slide': df_final['hacmil_mean_time_per_slide'].mean(),
    'sasha_mean_time_per_slide': df_final['sasha_mean_time_per_slide'].mean(),
    'total_time_hr': df_final['total_time_hr'].mean(),
    'total_time_sasha_02': df_final['total_time_sasha_02'].mean(),
    'total_time_sasha_01': df_final['total_time_sasha_01'].mean(),
    'speed_up_02' : df_final['speed_up_02'].mean(),
    'speed_up_01' : df_final['speed_up_01'].mean()
}

# Calculate standard deviation times
std_times = {
    'patching': df_final['patching_mean_time_per_slide'].std(),
    'feat_hr': df_final['feat_hr_mean_time_per_slide'].std(),
    'feat_lr': df_final['feat_lr_mean_time_per_slide'].std(),
    'feat_hr_b16': df_final['feat_hr_b16__mean_time_per_slide'].std(),
    'hacmil': df_final['hacmil_mean_time_per_slide'].std(),
    'sasha': df_final['sasha_mean_time_per_slide'].std(),
    'total_hr': df_final['total_time_hr'].std(),
    'total_time_sasha_02': df_final['total_time_sasha_02'].std(),
    'total_time_sasha_01': df_final['total_time_sasha_01'].std(),
    'speed_up_02' : df_final['speed_up_02'].std(),
    'speed_up_01' : df_final['speed_up_01'].std()
}


# Print average and std
print("\n====== Average Times Across Test Slides ======")
for step, avg_time in average_times.items():
    print(f"{step}: {avg_time:.4f} seconds")

print("\n====== Standard Deviation of Times Across Test Slides ======")
for step, std_time in std_times.items():
    print(f"{step}: {std_time:.4f} seconds")

output_path = "/media/internal_8T/naman/rlogist/time_test_final_2.csv"
df_final.to_csv(output_path, index=False)
print(f"✅ Saved combined dataframe successfully at: {output_path}")


# Average values from your dictionary
patching_avg = average_times['patching_mean_time_per_slide']
feat_hr_avg = average_times['feat_hr_mean_time_per_slide']
feat_lr_avg = average_times['feat_lr_mean_time_per_slide']
feat_hr_b16_avg = 0.2 * average_times['feat_hr_b16__mean_time_per_slide']  # scaled by 0.2
hacmil_avg = average_times['hacmil_mean_time_per_slide']
sasha_avg = average_times['sasha_mean_time_per_slide']

# Plot Graph

# Data for HR and SASHA
hr_components = [patching_avg, feat_hr_avg, hacmil_avg]
sasha_components = [patching_avg, feat_lr_avg, feat_hr_b16_avg, sasha_avg]

# Labels for components
hr_labels = ['Patching', 'Feature Extraction High Resolution', 'HACMIL']
sasha_labels = ['Patching', 'Feature Extraction Low Resolution ', 'Feature Extraction High Resolution', 'SASHA']

# Plotting
fig, ax = plt.subplots(figsize=(10, 7))

x = np.array([0, 1])  # positions for HR and SASHA bars
width = 0.4

# Plot HR stacked bar
bottom = 0
for value, label in zip(hr_components, hr_labels):
    ax.bar(x[0], value, width, bottom=bottom, label=label if x[0] == 0 else "", color=None)
    bottom += value

# Plot SASHA stacked bar
bottom = 0
for value, label in zip(sasha_components, sasha_labels):
    ax.bar(x[1], value, width, bottom=bottom, label=label if x[1] == 1 else "", hatch='//')
    bottom += value

# Set labels and title
ax.set_ylabel('Average Time (seconds)')
ax.set_title('Average Time Composition Comparison: HR vs SASHA')
ax.set_xticks(x)
ax.set_xticklabels(['HACMIL', 'SASHA'])
ax.legend(loc='upper center')
ax.grid(True, linestyle='--', linewidth=0.5)
ax.set_ylim([0, 400])

# Add value labels on bars
for i, method in enumerate(['HACMIL', 'SASHA']):
    total = sum(hr_components) if method == 'HACMIL' else sum(sasha_components)
    ax.text(x[i], total + 0.5, f'{total:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()