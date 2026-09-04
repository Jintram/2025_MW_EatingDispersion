"""
Looking at some data where things go wrong.
"""


# %%




# %%

import leafstats_analysis as lsa
    # import importlib; importlib.reload(lsa)


# %%

OUTPUTDIR = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20260305_EdgeCases/OUTPUT/'

# 1) Tell script where data is and which channels should be used
# Conditions and paths to images for that condition
condition_path_map = {
    'damageedgecase': '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20260305_EdgeCases/undetecteddamage'
}
# Channel configuration (channel index per role; Reference can be None)
config_channels = {
    'Leaf': 1,
    'Damage': 2,
    'Reference': 0 # optional
}
# Optional conversion from pixel area to cm^2 (set to e.g. 0.0004 if known)
pixel_to_cm2_factor = None
# obtain list of all files
data_file_paths = lsa.get_data_file_paths(condition_path_map)

# 2) Run the complete analysis pipeline
df_samples, array_data = lsa.run_complete_analysis(
    data_file_paths = data_file_paths,
    config_channels = config_channels,
    pixel_to_cm2_factor=pixel_to_cm2_factor
)

# 3) Generate summary plots for radial ACF, nearest-island distances, and radial PDFs
lsa.plot_acf_norms_avgrs(df_samples, array_data, OUTPUTDIR)
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR, remove_zerocnt=False)
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR, remove_zerocnt=True)
lsa.plot_radial_pdfs(df_samples, array_data, OUTPUTDIR)
lsa.plot_damaged_area(df_samples, OUTPUTDIR)

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
