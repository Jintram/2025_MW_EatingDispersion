
# %%

import leafstats_analysis as lsa
    # import importlib; importlib.reload(lsa)


# %%

# Paths below are relative, and are interpreted relative to your working
# directory, so run this script from the root of the repository.
# (Use absolute paths instead when your own data lives elsewhere.)
OUTPUTDIR = 'Example_data/OUTPUT-1channel/'

# 1) Tell script where data is and which channels should be used
# Conditions and paths to images for that condition
condition_path_map = {
    'Ctrl': 'Example_data/DATA/condition_Control',
    'Edited': 'Example_data/DATA/condition_Photoshopped'
}

# Reference condition, used to derive the damage threshold for method B
# (damage mask based on reference condition, see step 3)
REFERENCE_CONDITION = 'Ctrl'
# Output folders for the damage-mask dependent output, one per method
OUTPUTDIR_PERLEAF = OUTPUTDIR + '/damage_threshold_per_leaf/'
OUTPUTDIR_REF = OUTPUTDIR + f'/damage_threshold_ref_{REFERENCE_CONDITION}/'

# Channel configuration (channel index per role; Reference can be None)
config_channels = {
    'Leaf': 2,      # Set to same channel for illustratory purposes
    'Damage': 2,    # Set to same channel for illustratory purposes 
    'Reference': None
}
# Optional conversion from pixel area to cm^2 (set to e.g. 0.0004 if known)
pixel_to_cm2_factor = 1/(131**2)
# obtain
data_file_paths = lsa.get_data_file_paths(condition_path_map)

# 2) Run the complete analysis pipeline
df_samples, array_data = lsa.run_complete_analysis(
    data_file_paths = data_file_paths,
    config_channels = config_channels,
    # optional parameters
    leaf_threshold_method='triangle',
    leaf_roundness_threshold=0.8,
    apply_smooth_leafmask=True,
    pixel_to_cm2_factor=pixel_to_cm2_factor
)

# 3) Alternative damage mask: threshold derived from a reference condition
# (The per-leaf threshold from step 2 assumes the true base level of the damage
# signal is equal across conditions; the reference threshold allows it to
# differ, but assumes identical imaging conditions for all images. See readme.)
df_samples_ref, array_data_ref = lsa.apply_reference_threshold(
    df_samples, array_data,
    reference_condition = REFERENCE_CONDITION,
    pixel_to_cm2_factor = pixel_to_cm2_factor
)

# 4) Plots that do not depend on the damage mask (same for both methods)
lsa.plot_acf_norms_avgrs(df_samples, array_data, OUTPUTDIR)
lsa.plot_radial_pdfs(df_samples, array_data, OUTPUTDIR)
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="baselvl_dmg", 
                              y_label = "Estimated base level (damage channel)", 
                              title="Base level per condition\n(damage channel, within leaf)")
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="mean_dmg_signal", 
                              y_label = "Mean damage signal per leaf pixel", 
                              title="Mean damage signal per condition")

# 5) Damage mask METHOD A: per-leaf threshold
# Assumes equal true base level across conditions; robust to intensity
# differences between images.
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR_PERLEAF, remove_zerocnt=False)
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR_PERLEAF, remove_zerocnt=True)
lsa.plot_damaged_area(df_samples, OUTPUTDIR_PERLEAF)
lsa.plot_damaged_percentage(df_samples, OUTPUTDIR_PERLEAF)
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR_PERLEAF, metric_key="threshold_val_dmg", 
                              y_label = "Intensity threshold for damage", 
                              title=f"Threshold consistency\nDamage threshold should not\nshow trend per condition.")
lsa.plot_damage_overview(df_samples, array_data, OUTPUTDIR_PERLEAF, pixel_to_cm2_factor=pixel_to_cm2_factor)
lsa.run_plot_and_save(df_samples, array_data, OUTPUTDIR_PERLEAF, config_channels)
df_samples.to_csv(OUTPUTDIR_PERLEAF + 'data_leaf_damage_singlemetrics.csv', index=False)
df_samples.to_excel(OUTPUTDIR_PERLEAF + 'data_leaf_damage_singlemetrics.xlsx', index=False)

# 6) Damage mask METHOD B: reference-condition threshold
# Allows the base level to differ per condition; assumes identical imaging
# conditions for all images.
lsa.plot_nearest_island_distances(df_samples_ref, OUTPUTDIR_REF, remove_zerocnt=False)
lsa.plot_nearest_island_distances(df_samples_ref, OUTPUTDIR_REF, remove_zerocnt=True)
lsa.plot_damaged_area(df_samples_ref, OUTPUTDIR_REF)
lsa.plot_damaged_percentage(df_samples_ref, OUTPUTDIR_REF)
lsa.plot_metric_per_condition(df_samples_ref, OUTPUTDIR_REF, metric_key="threshold_val_dmg", 
                              y_label = "Intensity threshold for damage", 
                              title=f"Threshold (reference condition: {REFERENCE_CONDITION})")
lsa.plot_damage_overview(df_samples_ref, array_data_ref, OUTPUTDIR_REF, pixel_to_cm2_factor=pixel_to_cm2_factor)
lsa.run_plot_and_save(df_samples_ref, array_data_ref, OUTPUTDIR_REF, config_channels)
df_samples_ref.to_csv(OUTPUTDIR_REF + 'data_leaf_damage_singlemetrics.csv', index=False)
df_samples_ref.to_excel(OUTPUTDIR_REF + 'data_leaf_damage_singlemetrics.xlsx', index=False)

# %%
