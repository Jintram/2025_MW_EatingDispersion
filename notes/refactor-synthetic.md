# Removing the synthetic-specific code

(Written by Claude, 2026-09-24. Originally written as an analysis before the refactor; line numbers refer to the code before it.)

**Status (24/9/2026): done** — the refactor below has been carried out; see the changelog in [readme.md](readme.md). The `noise` ACF/radial PDF is kept, by computing these in the pipeline whenever a leaf is found.

## Synthetic-specific code

In [leafstats_analysis.py](../leafstats_analysis.py):

| Lines | Item | Used by |
|---|---|---|
| 15 | `from skimage import io` | only `load_synthetic_data` (the pipeline uses `iio`) |
| 289–325 | `plot_images()` | only `run_synthetic_analysis` |
| 519–567 | `load_synthetic_data()` + section header | only `leafstats_syntheticdata.py` |
| 570–591 | `plot_img_n_acf()` + section header | only `run_synthetic_analysis` |
| 593–725 | `run_synthetic_analysis()` | only `leafstats_syntheticdata.py` |
| 876 | docstring "(as for synthetic data)" in `run_complete_analysis` | wording only |

In [leafstats_syntheticdata.py](../leafstats_syntheticdata.py): lines 13–47 (the `OUTPUT1` block).

Things that point to the old outputs:
- `Synthetic_data/OUTPUT1_frozen/` (the `synthdata_*` figures) would no longer be regenerated.
- [readme.md](../readme.md) line 25 says the script "generates the synthetic-data figures shown below, and additionally runs the regular analysis pipeline"; line 208 is a commented-out link to `OUTPUT1_frozen`.
- [tests/test_examples_smoke.py](../tests/test_examples_smoke.py) checks only `OUTPUT2`, so it is not affected.

## Checked: the numbers are the same

I ran both paths on the 5 synthetic images. Leaf masks, damage masks, the radial ACF average, island counts and damage % were identical for all samples. The pipeline was run with `leaf_threshold_method='otsu'`, as in the script.

## What each synthetic output maps to

| Old output (`OUTPUT1`) | Pipeline output (`OUTPUT2`) | Lost? |
|---|---|---|
| `synthdata_img_<key>` (1×3: reference, leaf+mask+centroid, damage+mask) | `plots/segmentation_masks/<cond>/*_images.png` (plus a version with histograms) | No. Note: the old version always showed the *disk* image in the reference panel (`img0=img_disk`), which was a bug. |
| `synthdata_summary_damage` (bar) | `damaged_percentage` | No |
| `synthdata_summary_nearestisland`, `_mean`, `_islandcount` | `nearest_island_distances` (3 panels) | No |
| `synthdata_radialpdf_<key>` (image + PDF, one figure per sample) | `radial_pdfs_samples` / `_averages` (all samples in one panel) + `overview_damage` | Only the one-figure-per-sample layout |
| `synthdata_acf_<key>` (image + ACF, one figure per sample) | `Radial_acf_samples` / `_averages` | **Partly**, see below |

## Functionality that would be lost

1. **ACF and radial PDF for the `noise` sample.** With the `bg2` threshold, no damage is found in the noise image. The pipeline then marks it `no_damage_mask` and skips the ACF and radial PDF, so `noise` does not appear in `Radial_acf_*` or `radial_pdfs_*`. The synthetic code computes and plots these anyway, from the raw intensities, with an empty damage mask. That gives a "no spatial structure" negative control. The readme's ACF section doesn't discuss noise, so no text breaks. Decide whether to keep this control, e.g. by computing the ACF from the intensity image even when no damage mask is found.
2. **1D centre-line ACF profile.** `plot_img_n_acf` also plots the horizontal slice through the ACF centre (dotted), not just the radial average. The pipeline doesn't plot this. The data is still available (`array_data[...]['acf_norm']` and `['acf_center']`).
3. **Per-sample image + curve figures** (ACF and radial PDF side by side with the damage image). Only the layout is lost, since `overview_damage` shows the images.

Everything else (loading, masks, metrics, plots) is covered by `get_data_file_paths` + `run_complete_analysis` + the `plot_*` functions. The pipeline also does more: roundness check, thresholds/backgrounds, and CSV/XLSX export.
