
################################################################################
# %%

import leafstats_analysis as lsa
    # import importlib; importlib.reload(lsa)

import os

################################################################################
# %%

# Paths below are relative, and are interpreted relative to your working
# directory, so run this script from the root of the repository.
# (Use absolute paths instead when your own data lives elsewhere.)

# Where to put plots
OUTPUTDIR = 'Synthetic_data/OUTPUT1'

# Where to find synthetic images
SYNTHETIC_IMAGE_PATH = 'Synthetic_data/'

# 1) Ensure base output directory exists
os.makedirs(OUTPUTDIR, exist_ok=True)

# 2) Define channel configuration (channel index per role; Reference can be None)
config_channels = {
    'Leaf': 1,
    'Damage': 2,
    'Reference': 0
}

# 3) Load synthetic example images and run synthetic sanity-check analysis/plots
img_leafs_syn, img_damages_syn, img_disk = lsa.load_synthetic_data(
    SYNTHETIC_IMAGE_PATH,
    config_channels
)
lsa.run_synthetic_analysis(
    img_leafs = img_leafs_syn, 
    img_damages = img_damages_syn,
    img_disk = img_disk, 
    config_channels = config_channels,
    outputdir = OUTPUTDIR
)
# img_leafs = img_leafs_syn; img_damages = img_damages_syn; img_disk = img_disk
# config_channels = config_channels
# outputdir = OUTPUTDIR
# %% 






# %% Now also run the usual analysis


# Paths below are relative, and are interpreted relative to your working
# directory, so run this script from the root of the repository.
# (Use absolute paths instead when your own data lives elsewhere.)
OUTPUTDIR = 'Synthetic_data/OUTPUT2'

# 1) Tell script where data is and which channels should be used
# Conditions and paths to images for that condition
condition_path_map = {
    'noise': 'Synthetic_data/images/noise/',
    'disk': 'Synthetic_data/images/disk/',
    'spots': 'Synthetic_data/images/spots/',
    'donut': 'Synthetic_data/images/donut/',
    'dualspot': 'Synthetic_data/images/dualspot/'
}


# Channel configuration (channel index per role; Reference can be None)
config_channels = {
    'Leaf': 1,
    'Damage': 2,
    'Reference': 0 # optional
}
# Optional conversion from pixel area to cm^2 (set to e.g. 0.0004 if known)
pixel_to_cm2_factor = None
# obtain 
data_file_paths = lsa.get_data_file_paths(condition_path_map)


# 2) Run the complete analysis pipeline
df_samples, array_data = lsa.run_complete_analysis(
    data_file_paths = data_file_paths, 
    config_channels = config_channels,   
    # optional parameters 
    leaf_threshold_method = 'otsu',
    leaf_roundness_threshold=0,
    apply_smooth_leafmask=False,
    pixel_to_cm2_factor=pixel_to_cm2_factor
)

# 3) Generate summary plots for radial ACF, nearest-island distances, and radial PDFs
lsa.plot_acf_norms_avgrs(df_samples, array_data, OUTPUTDIR)
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR, remove_zerocnt=False)
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR, remove_zerocnt=True)
lsa.plot_radial_pdfs(df_samples, array_data, OUTPUTDIR)
lsa.plot_damaged_area(df_samples, OUTPUTDIR)
lsa.plot_damaged_percentage(df_samples, OUTPUTDIR)
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="threshold_val_dmg", 
                              y_label = "Intensity threshold for damage", 
                              title=f"Threshold consistency\nDamage threshold should not\nshow trend per condition.")
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="background_dmg", 
                              y_label = "Estimated background intensity", 
                              title=f"Background consistency\nBackground should not\nshow trend per condition.")


    # import importlib; importlib.reload(lsa)

# 4) Export per-image mask overlays to output folders
lsa.run_plot_and_save(
    df_samples,
    array_data,
    OUTPUTDIR,
    config_channels
)

# 5) Export single-value metrics to CSV and Excel
df_samples.to_csv(OUTPUTDIR + '/data_leaf_damage_singlemetrics.csv', index=False)
df_samples.to_excel(OUTPUTDIR + '/data_leaf_damage_singlemetrics.xlsx', index=False)

# %%
