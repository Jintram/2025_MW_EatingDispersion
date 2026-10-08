

## Notes on running the script

#### Set up file structure and configuration

Before running the actual analysis, information is constructed about which files to use
and what configuration these files are.

This is based on choosing different directories with images 
that each correspond to a specific condition. This can be set as follows:
```{python}
# 1) Tell script where data is and which channels should be used
# Conditions and paths to images for that condition
condition_path_map = {
    'Ctrl': 'Example_data/DATA/condition_Control',
    'Edited': 'Example_data/DATA/condition_Photoshopped'
}
```
Note that a so-called `dict` is used to link each condition (e.g. `'Ctrl'`)
to a specific folder.

These folder paths can be absolute, or relative to your working directory,
as in the example above. The same holds for `OUTPUTDIR`.
The condition names are also used to organize the exported per-image plots,
which end up in `OUTPUTDIR/plots_segmasks_<leafthr|refthr>/<condition>/`.

Additionally, the script needs to know in which channel to look for the
leaf data and where to look for the damage. A third channel can be displayed
and is called the reference channel.
```{python}
# Channel configuration (channel index per role; Reference can be None)
config_channels = {
    'Leaf': 1,
    'Damage': 2,
    'Reference': 0 # optional
}
```
Again a `dict` is used. Each entry links a role (`'Leaf'`, `'Damage'` or
`'Reference'`) to the index of the channel that should be used for that role
(e.g. `0`, the first channel). The reference channel is only displayed in the
per-image plots, and can be set to `None` when it isn't needed. To analyze a
single-channel dataset, simply point both `'Leaf'` and `'Damage'` to the same
channel (see [leafstats_example_1channel.py](leafstats_example_1channel.py)).

A list of files is then collected by calling the following function:
```{python}
# obtain 
data_file_paths = lsa.get_data_file_paths(condition_path_map)
```

#### Running the analysis

The code 

```{python}
df_samples, array_data = lsa.run_complete_analysis(
    data_file_paths = data_file_paths, 
    config_channels = config_channels,   
    # optional parameters 
    leaf_threshold_method = 'bg10',
    leaf_roundness_threshold=0,
    apply_smooth_leafmask=False,
    pixel_to_cm2_factor=pixel_to_cm2_factor
)
```

will run all analyses, and returns the results in two objects:

- `df_samples`, a pandas dataframe with one row per image, holding all
single-value metrics (e.g. `island_counts_leafthr`, `total_damage_area_px_leafthr`,
`total_damage_percentage_leafthr`, `threshold_val_dmg_leafthr`, `baselvl_dmg`, 
`mean_dmg_signal`), plus the
condition, the file path, and status fields (`leaf_found`, `damage_found_leafthr`,
`analysis_status_leafthr`) that record whether the analysis succeeded for that image.
The suffix `_leafthr` marks metrics that depend on the damage mask, as 
determined with the per-leaf threshold (method (i)); the corresponding 
`_refthr` columns (method (ii)) are filled by `apply_reference_threshold` (see 
below), and are empty until then (`analysis_status_refthr` is 
`'not_calculated'`).
When a leaf is found but no damage, `analysis_status_leafthr` is `'no_damage_mask'`;
metrics that depend on the damage mask (areas, island statistics) are then set
to 0, whereas metrics that only need the leaf mask (autocorrelation, radial
distribution, damage threshold, base level and mean damage signal) are still 
calculated.
- `array_data`, a `dict` keyed by file path, holding the array-like results
per image (the images themselves, the leaf mask, the damage masks 
`mask_damage_leafthr` and `mask_damage_refthr`, the centroid, the
autocorrelation, and the radial distribution).

Both are needed for the plotting functions below.

See above for how to set the optional parameters.

When `pixel_to_cm2_factor` is set, areas in pixels will be multiplied
with this factor to determine the area in square centimeters.

#### Damage threshold based on a reference condition

`run_complete_analysis` determines the damage mask with the per-leaf damage 
threshold (method (i), see "Two methods to determine the damage threshold" 
above). The results for method (ii), which uses a threshold derived from a 
reference condition, are added with:

```python
df_samples, array_data = lsa.apply_reference_threshold(
    df_samples, array_data,
    reference_condition = REFERENCE_CONDITION,
    pixel_to_cm2_factor = pixel_to_cm2_factor
)
```

where `REFERENCE_CONDITION` is one of the keys of `condition_path_map` 
(e.g. `'Ctrl'`). This fills the `_refthr` columns of `df_samples` (and the
column `reference_condition`), and adds `mask_damage_refthr` to `array_data`.
(Note that `array_data` is extended in place.)

#### Generating plots

To generate each of the plots, the following functions can be used.
Functions that plot damage-mask dependent data (`plot_nearest_island_distances`, 
`plot_damaged_area`, `plot_damaged_percentage`, `plot_damage_overview`, 
`run_plot_and_save`, and `plot_metric_per_condition` for damage-mask 
dependent metrics such as `"threshold_val_dmg"`) take the argument 
`damage_mask_method='leafthr'` (default) or `damage_mask_method='refthr'`, which selects 
the method, and save their output to `OUTPUTDIR/plots_damageregionstats_<leafthr|refthr>/`
(or `OUTPUTDIR/plots_segmasks_<leafthr|refthr>/` for `run_plot_and_save`). 
Other plots are saved to `OUTPUTDIR/plots_general_stats/`. For brevity, the 
examples below show the default; in the example scripts, these plots are made
for both methods. (The damage-mask dependent figures shown below are those 
of the per-leaf threshold method, `damage_mask_method='leafthr'`.)

```{python}
lsa.plot_acf_norms_avgrs(df_samples, array_data, OUTPUTDIR)
```

<img src="Example_data/OUTPUT-3channels_frozen/plots_general_stats/Radial_acf_samples.png" width=50%><br>
<img src="Example_data/OUTPUT-3channels_frozen/plots_general_stats/Radial_acf_averages.png" width=50%>

```{python}
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR, remove_zerocnt=False)
lsa.plot_nearest_island_distances(df_samples, OUTPUTDIR, remove_zerocnt=True)
```

<img src="Example_data/OUTPUT-3channels_frozen/plots_damageregionstats_leafthr/nearest_island_distances.png" width=100%>

```{python}
lsa.plot_radial_pdfs(df_samples, array_data, OUTPUTDIR)
```
<img src="Example_data/OUTPUT-3channels_frozen/plots_general_stats/radial_pdfs_samples.png" width=50%><br>
<img src="Example_data/OUTPUT-3channels_frozen/plots_general_stats/radial_pdfs_averages.png" width=50%>

```{python}
lsa.plot_damaged_area(df_samples, OUTPUTDIR)
```
<img src="Example_data/OUTPUT-3channels_frozen/plots_damageregionstats_leafthr/damaged_area_px.png" width=50%>

(This plot is exported as `damaged_area_px.png` when no `pixel_to_cm2_factor`
was given, and as `damaged_area_cm2.png` when it was.)

```python
lsa.plot_damaged_percentage(df_samples, OUTPUTDIR)
```

<img src="Example_data/OUTPUT-3channels_frozen/plots_damageregionstats_leafthr/damaged_percentage.png" width=50%>

The function `lsa.plot_metric_per_condition` can be used to plot any of the
single-value metrics in `df_samples` per condition; the plot is exported using
the metric name as file name.

The following code plots the threshold that 
was used for something to be considered damaged leaf (`"threshold_val_dmg"`),
this should not show a dependency on the condition (see also above).

```python
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="threshold_val_dmg", 
                              y_label = "Intensity threshold for damage", 
                              title=f"Threshold consistency\nDamage threshold should not\nshow trend per condition.")
```

<img src="Example_data/OUTPUT-3channels_frozen/plots_damageregionstats_leafthr/threshold_val_dmg.png" width=50%>

Likewise, the estimated base level of the damage channel within the leaf
(`"baselvl_dmg"`) can be plotted, which ideally shouldn't show a trend per 
condition either (this is the plot that was also shown above):

```python
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="baselvl_dmg", 
                              y_label = "Estimated base level (damage channel)", 
                              title="Base level per condition\n(damage channel, within leaf)")
```

Note that currently, the `threshold_val_dmg` is simply twice the `baselvl_dmg`.

<img src="Example_data/OUTPUT-3channels_frozen/plots_general_stats/baselvl_dmg.png" width=50%>

The mean damage signal per leaf pixel (`"mean_dmg_signal"`, ie the mean 
intensity of the damage channel within the leaf mask) can be plotted in the
same way. In contrast to the damaged area, this metric does not depend on 
the damage threshold, and is therefore not affected by changes in the base 
level (see assumptions above). 

```python
lsa.plot_metric_per_condition(df_samples, OUTPUTDIR, metric_key="mean_dmg_signal", 
                              y_label = "Mean damage signal per leaf pixel", 
                              title="Mean damage signal per condition")
```

<img src="Example_data/OUTPUT-3channels_frozen/plots_general_stats/mean_dmg_signal.png" width=50%>

Set `OUTPUTDIR` to a directory where you want the plots to be exported.

To inspect single segmentation and damage area segmentation, run the following function:

```{python}
# 4) Export per-image mask overlays to output folders
lsa.run_plot_and_save(
    df_samples,
    array_data,
    OUTPUTDIR,
    config_channels
)
```

<img src="Example_data/OUTPUT-3channels_frozen/plots_segmasks_leafthr/Ctrl/Example_A_1.png">

These figures are exported to 
`OUTPUTDIR/plots_segmasks_<leafthr|refthr>/<condition>/`,
one per input image, whilst the summary plots are placed in the other 
`plots_*` folders (see the output structure above). The segmentation shown here is the first analysis step, on
which all other results depend: the damaged area, the pattern statistics, and
every value in the exported tables are all derived from these masks. It is
therefore recommended to inspect these figures manually for artifacts (e.g. a
mask that captured background instead of the leaf) before interpreting the
summary plots.

#### Exporting data to excel/csv

Finally, the following lines export data to csv and excel files.

```{python}
df_samples.to_csv(OUTPUTDIR + '/data_leaf_damage_singlemetrics.csv', index=False)
df_samples.to_excel(OUTPUTDIR + '/data_leaf_damage_singlemetrics.xlsx', index=False)
```

(All single-value metrics, for both damage threshold methods, are collected 
in the `df_samples` dataframe (see above), so they can be written
out directly using the standard pandas export functions. Note that
`.to_excel` requires the `openpyxl` library, see the installation instructions
above.)