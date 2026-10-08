
################################################################################
# %%

import leafstats_analysis as lsa
    # import importlib; importlib.reload(lsa)

################################################################################
# %%

# Paths below are relative, and are interpreted relative to your working
# directory, so run this script from the root of the repository.
# (Use absolute paths instead when your own data lives elsewhere.)
OUTPUTDIR = 'Synthetic_data/OUTPUT'

# 1) Tell script where data is and which channels should be used
# Conditions and paths to images for that condition
condition_path_map = {
    'noise': 'Synthetic_data/images/noise/',
    'disk': 'Synthetic_data/images/disk/',
    'spots': 'Synthetic_data/images/spots/',
    'donut': 'Synthetic_data/images/donut/',
    'dualspot': 'Synthetic_data/images/dualspot/'
}


# Reference condition, used to derive the damage threshold for method B
# (damage mask based on reference condition, see step 3)
REFERENCE_CONDITION = 'noise'

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

# 3) Alternative damage mask: threshold derived from a reference condition
# (The per-leaf threshold from step 2 assumes the true base level of the damage
# signal is equal across conditions; the reference threshold allows it to
# differ, but assumes identical imaging conditions for all images. See readme.)
# This adds the columns with suffix "_refthr" to df_samples (next to the
# "_leafthr" columns of the per-leaf threshold), and "mask_damage_refthr"
# to array_data.
df_samples, array_data = lsa.apply_reference_threshold(
    df_samples, array_data,
    reference_condition = REFERENCE_CONDITION,
    pixel_to_cm2_factor = pixel_to_cm2_factor
)

# 4) Plots that do not depend on the damage mask (same for both methods)
# Plots are saved to OUTPUTDIR/plots_general_stats/.
lsa.plot_acf_norms_avgrs(df_samples, array_data, OUTPUTDIR)
lsa.plot_radial_pdfs(df_samples, array_data, OUTPUTDIR)
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="baselvl_dmg", 
                              y_label = "Estimated base level (damage channel)", 
                              title="Base level per condition\n(damage channel, within leaf)")
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="mean_dmg_signal", 
                              y_label = "Mean damage signal per leaf pixel", 
                              title="Mean damage signal per condition")

# 5) Damage mask METHOD A: per-leaf threshold (damage_mask_method='leafthr')
# Assumes equal true base level across conditions; robust to intensity
# differences between images.
# Plots are saved to OUTPUTDIR/plots_damageregionstats_leafthr/ and
# OUTPUTDIR/plots_segmasks_leafthr/ (per-image segmentation plots).
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR, remove_zerocnt=False, damage_mask_method='leafthr')
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR, remove_zerocnt=True, damage_mask_method='leafthr')
lsa.plot_damaged_area(df_samples, OUTPUTDIR, damage_mask_method='leafthr')
lsa.plot_damaged_percentage(df_samples, OUTPUTDIR, damage_mask_method='leafthr')
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="threshold_val_dmg", 
                              y_label = "Intensity threshold for damage", 
                              title=f"Threshold consistency\nDamage threshold should not\nshow trend per condition.",
                              damage_mask_method='leafthr')
lsa.plot_damage_overview(df_samples, array_data, OUTPUTDIR, pixel_to_cm2_factor=pixel_to_cm2_factor, damage_mask_method='leafthr')
lsa.run_plot_and_save(df_samples, array_data, OUTPUTDIR, config_channels, damage_mask_method='leafthr')

# 6) Damage mask METHOD B: reference-condition threshold (damage_mask_method='refthr')
# Allows the base level to differ per condition; assumes identical imaging
# conditions for all images.
# Plots are saved to OUTPUTDIR/plots_damageregionstats_refthr/ and
# OUTPUTDIR/plots_segmasks_refthr/ (per-image segmentation plots).
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR, remove_zerocnt=False, damage_mask_method='refthr')
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR, remove_zerocnt=True, damage_mask_method='refthr')
lsa.plot_damaged_area(df_samples, OUTPUTDIR, damage_mask_method='refthr')
lsa.plot_damaged_percentage(df_samples, OUTPUTDIR, damage_mask_method='refthr')
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="threshold_val_dmg", 
                              y_label = "Intensity threshold for damage", 
                              title=f"Threshold",
                              damage_mask_method='refthr')
lsa.plot_damage_overview(df_samples, array_data, OUTPUTDIR, pixel_to_cm2_factor=pixel_to_cm2_factor, damage_mask_method='refthr')
lsa.run_plot_and_save(df_samples, array_data, OUTPUTDIR, config_channels, damage_mask_method='refthr')

# 7) Export single-value metrics (both methods) to CSV and Excel
df_samples.to_csv(OUTPUTDIR + '/data_leaf_damage_singlemetrics.csv', index=False)
df_samples.to_excel(OUTPUTDIR + '/data_leaf_damage_singlemetrics.xlsx', index=False)

# %%
