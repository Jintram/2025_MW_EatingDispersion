
# %%

import leafstats_analysis as lsa
    # import importlib; importlib.reload(lsa)


# %%

OUTPUTDIR = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20260218_NewData_Nina/OUTPUT202602/KOvsWT/'

# PART B, real data

# 1) Tell script where data is and which channels should be used
# Conditions and paths to images for that condition
condition_path_map = {
    'WT': '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20260218_NewData_Nina/KOvsWT/WT/',
    'KO': '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20260218_NewData_Nina/KOvsWT/KO/'
}
# Channel configuration (channel index per role; Reference can be None)
config_channels = {
    'Leaf': 0,      # channel 1 often
    'Damage': 0,    # channel 2 often
    'Reference': None
}
# obtain 
data_file_paths = lsa.get_data_file_paths(condition_path_map)

# 2) Run the complete analysis pipeline
data_all = lsa.run_complete_analysis(data_file_paths, config_channels,
                                     leaf_threshold_method='triangle',
                                     leaf_roundness_threshold=0.8, apply_smooth_leafmask=True,
                                     pixel_to_cm2_factor=1/(131**2))

# 3) Generate summary plots for radial ACF, inter-island distances, and radial PDFs
lsa.plot_acf_norms_avgrs(data_all, OUTPUTDIR)
lsa.plot_interisland_distances(data_all, OUTPUTDIR, remove_zerocnt=False)
lsa.plot_interisland_distances(data_all, OUTPUTDIR, remove_zerocnt=True)
lsa.plot_radial_pdfs(data_all, OUTPUTDIR)
lsa.plot_damaged_area(data_all, OUTPUTDIR)

# 4) Export per-image mask overlays to output folders
lsa.run_plot_and_save(
    data_all,
    data_file_paths,
    OUTPUTDIR,
    config_channels
)

# 5) Export single-value metrics to CSV and Excel
df_singledata = lsa.export_singledatapoints(
    data_all,
    data_file_paths
)
df_singledata.to_csv(OUTPUTDIR + '/leaf_damage_singlemetrics.csv', index=False)
df_singledata.to_excel(OUTPUTDIR + '/leaf_damage_singlemetrics.xlsx', index=False)
# %%
