
# %%

import leafstats_analysis as lsa
    # import importlib; importlib.reload(lsa)


# %%

OUTPUTDIR = '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/OUTPUT202602/'

# 1) Collect real-data file paths and run the complete analysis pipeline
condition_path_map = {
    'infected': '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Infected',
    'noninfected': '/Users/m.wehrens/Data_UVA/2024_small-analyses/2025_Nina_LeafDamage/20250709_PartialData_Nina/Non infected'
}
data_file_paths = lsa.get_data_file_paths(condition_path_map)
data_all = lsa.run_complete_analysis(data_file_paths)

# 2) Generate summary plots for radial ACF, inter-island distances, and radial PDFs
lsa.plot_acf_norms_avgrs(data_all, OUTPUTDIR)
lsa.plot_interisland_distances(data_all, OUTPUTDIR, remove_zerocnt=False)
lsa.plot_interisland_distances(data_all, OUTPUTDIR, remove_zerocnt=True)
lsa.plot_radial_pdfs(data_all, OUTPUTDIR)

# 3) Export per-image mask overlays to output folders
lsa.run_plot_and_save(data_all, data_file_paths, img_disk, OUTPUTDIR)

# 4) Export single-value metrics to CSV and Excel
df_singledata = lsa.export_singledatapoints(
    data_all,
    data_file_paths,
    data_singledatapoint=['total_interisland_distances', 'island_counts']
)
df_singledata.to_csv(OUTPUTDIR + '/leaf_damage_singlemetrics.csv', index=False)
df_singledata.to_excel(OUTPUTDIR + '/leaf_damage_singlemetrics.xlsx', index=False)
# %%
