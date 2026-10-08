(Written by Claude, based on the git log of 2 June – 8 October 2026.)

Here's a summary of the ~124 commits from June 2 to October 8, 2026, grouped by topic. Activity came in four bursts: early June, Aug 31 to Sep 4, Sep 15 to 24, and Oct 6 to 8.

### 1. Damage threshold and damaged-area method (the main thread)
- **June:** Refactored the channel configuration. Added leaf size, damage as a percentage of leaf area, storage of the threshold values found, and a "background" estimate for the damage channel.
- **Oct 6:** Renamed "background" to **base level** (`background_dmg` → `baselvl_dmg`) for the damage channel, because the mode inside the leaf is real signal rather than background. The assumptions in the readme were revised to match.
- **Oct 8, "v1.0 of the improved pipeline":** Added a second threshold method, a **reference-condition threshold** (`apply_reference_threshold`), next to the per-leaf one. All mask-dependent metrics now carry `_leafthr` / `_refthr` suffixes in a single dataframe. This breaks the old column names. Part of this work was committed as "unchecked Claude edits".

### 2. New and corrected metrics
- **ACF bug fix (Sep 3):** `get_autocorrelation()` now computes a proper Pearson correlation instead of the raw second moment, which makes differences in the example data visible.
- **Nearest-island distance (Sep 4):** Added a mean nearest-island distance and renamed "inter-island" to "nearest-island" throughout.
- **Mean damage signal (Oct 6):** New metric `mean_dmg_signal`, the mean damage-channel intensity within the leaf, independent of the threshold.
- **Behaviour change (Sep 24):** Leaves without a damage mask now still get their ACF, radial PDF and threshold metrics.

### 3. Code structure and robustness
- **Sep 2:** Removed the `cv2` dependency, allowed relative paths, and saved segmentation masks per condition.
- **Sep 15:** Split the large `run_complete_analysis()`, moving the per-image work into `analyse_sample()` with the `SampleMetrics` / `SampleArrays` dataclasses.
- **Sep 24:** Removed the synthetic-data-specific code, so the synthetic data now runs through the regular pipeline.
- **Oct 6:**
  - Added `check_bool_mask()`.
  - Fixed a mask-dtype bug in `background_leaf`.
  - Fixed the conda line in the regeneration script.

### 4. Plotting and outputs
- New plots:
  - a damage overview grid (`plot_damage_overview`)
  - separate per-sample and condition-average versions of the ACF and radial PDF plots
  - image-only versions of the per-image plots
  - plots for the base level and the mean signal
- Introduced the `_frozen` reference output folders for the example and synthetic data, and stopped tracking generated output.
- **Oct 8:** Flattened the output folder layout into `plots_general_stats/`, `plots_damageregionstats_<method>/` and `plots_segmasks_<method>/`.
- Added a `regenerate_example_outputs.sh` script.

### 5. Synthetic data
- Reorganised the images into per-pattern folders and added a "noise" pattern.
- Switched to relative paths inside the repo.

### 6. Tests
- Added a `tests/` folder containing:
  - a smoke test for the examples
  - tests for no-damage samples, the bool-mask check, the reference threshold, and `get_mask`

### 7. Documentation (most of the commits by count)
- **Aug 31:** Moved the changelog to its own file. Renamed the example scripts (`leafstats_example_*.py`) and added example figures to the readme.
- **Sep:** Wrote up the ACF maths (both human and Claude versions) and started `notes/` as a to-do list and changelog. Added `notes/ACF.md` and `notes/refactor-synthetic.md`.
- **Oct 8:** Restructured the readme around the two threshold methods and their assumptions. The older sections moved to `readme_old.md`.

The open item in `notes/readme.md` is how to determine the damaged area when the leaf base level isn't constant across conditions.
