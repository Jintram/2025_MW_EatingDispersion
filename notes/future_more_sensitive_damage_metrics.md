# Future idea: more sensitive damage metrics

(Written by Claude, 18/11/2026. Shelved idea: nothing described here has been implemented.
The numbers come from a one-off exploratory analysis, and were not human-checked.)

## Question

Thrips damage affects a small part of the leaf, so the mean damage signal changes little,
because most of the leaf is unaffected. Is there a better single metric to summarize the
damage? The existing damaged percentage is a good candidate, since it rises when the
bright tail of the intensity distribution rises.

## Conclusion so far

- **The existing whole-leaf damaged percentage (`total_damage_percentage_leafthr`) holds up
  well, and there is no strong reason to replace it.**
- Removing the leaf rim gives a much larger thrips:control *ratio*, but this mainly comes
  from removing an offset that is present in the controls. It does not make the effect
  larger: the absolute thrips-control difference becomes smaller.
- The thrips-control difference is largest in the rim itself. Whether this reflects
  feeding near the edge, or a stronger response at the cut edge, can't be told from this
  analysis.

## Data and definitions

- Time-series data (`projects/analyze_202611_timeseries.py`, back side, Cy7), with the
  settings of that script: 29 valid leaves, 10 control and 19 thrips, 2 to 5 per group
  per time point (3H, 6H, 9H, 12H).
- **Base level:** `baselvl_dmg` of the leaf. **Reference threshold:** intensity 76, from
  `Ctrl_3H`.
- **Interior:** leaf pixels more than 20 px (about 1.5 mm) from the leaf mask edge; on
  average 74% of the leaf. **Rim:** the remaining leaf pixels.
- **% >2x base / % >ref thr:** percentage of pixels in the region above the threshold.
- **Excess over 2x base:** mean of `max(0, intensity/base - 2)` over the region.
- **Excess over ref thr:** mean of `max(0, intensity - threshold)` over the region.
- **P99:** 99th percentile of the region (raw, or as fold over base).
- **Separation:** the chance that a thrips leaf scores higher than a control leaf from the
  same time point (0.5 = no difference, 1 = perfect separation).
- **Ratio:** median of all thrips leaves / median of all control leaves.

## Sensitivity comparison

Sorted by overall separation.

| Metric | Separation, all | 3H | 6H | 9H | 12H | Ratio T/C |
|---|---:|---:|---:|---:|---:|---:|
| Interior P99, fold over base | 0.98 | 0.93 | 1.0 | 1.0 | 1.00 | 1.5 |
| Interior P99, raw | 0.98 | 0.93 | 1.0 | 1.0 | 1.00 | 2.2 |
| Whole leaf, mean fold over base | 0.96 | 1.00 | 0.8 | 1.0 | 1.00 | 1.1 |
| Whole leaf, % >2x base | 0.96 | 0.87 | 1.0 | 1.0 | 1.00 | 1.8 |
| Interior, % >2x base | 0.96 | 0.87 | 1.0 | 1.0 | 1.00 | 13.4 |
| Whole leaf, excess over ref thr | 0.91 | 0.93 | 0.7 | 1.0 | 1.00 | 3.8 |
| Whole leaf, P99 raw | 0.91 | 0.93 | 0.7 | 1.0 | 1.00 | 1.7 |
| Interior, excess over ref thr | 0.91 | 0.93 | 0.7 | 1.0 | 1.00 | 86.4 |
| Whole leaf, excess over 2x base | 0.89 | 0.87 | 0.8 | 1.0 | 0.92 | 2.2 |
| Interior, % >ref thr | 0.89 | 0.93 | 0.6 | 1.0 | 1.00 | 81.9 |
| Interior, excess over 2x base | 0.89 | 0.67 | 1.0 | 1.0 | 1.00 | 22.3 |
| Whole leaf, mean signal | 0.85 | 0.87 | 0.6 | 0.9 | 1.00 | 1.5 |
| Whole leaf, % >ref thr | 0.85 | 0.80 | 0.6 | 1.0 | 1.00 | 3.4 |
| Interior, mean fold over base | 0.85 | 0.67 | 0.8 | 1.0 | 1.00 | 1.1 |
| Whole leaf, P99 fold over base | 0.81 | 0.73 | 0.8 | 1.0 | 0.75 | 1.4 |
| Interior, mean signal | 0.81 | 0.80 | 0.6 | 0.8 | 1.00 | 1.4 |

## Important: the rim effect on the ratio is mainly an offset

The large interior ratios look impressive, but controls have almost no signal above the
threshold in the interior, so the ratio divides by a number close to zero. Bright rims in
the controls form an offset in the whole-leaf metric, and removing the rim removes it.

The absolute differences show what happens to the effect itself:

| Metric | Control median | Thrips median | Difference of medians | Mean difference within time points | Within-group SD | Difference / SD |
|---|---:|---:|---:|---:|---:|---:|
| Whole leaf, % >2x base | 4.69 | 8.38 | 3.69 | 5.82 | 4.47 | 1.30 |
| Interior, % >2x base | 0.10 | 1.32 | 1.22 | 2.73 | 4.02 | 0.68 |
| Rim, % >2x base | 18.49 | 28.66 | 10.17 | 14.22 | 6.67 | 2.13 |
| Whole leaf, % >ref thr | 5.11 | 17.20 | 12.09 | 14.77 | 16.34 | 0.90 |
| Interior, % >ref thr | 0.07 | 5.67 | 5.60 | 10.72 | 17.56 | 0.61 |
| Rim, % >ref thr | 20.02 | 47.01 | 26.99 | 25.91 | 15.44 | 1.68 |
| Whole leaf, excess over 2x base | 0.046 | 0.102 | 0.056 | 0.084 | 0.083 | 1.02 |
| Interior, excess over 2x base | 0.000 | 0.009 | 0.009 | 0.023 | 0.048 | 0.48 |
| Rim, excess over 2x base | 0.180 | 0.377 | 0.197 | 0.253 | 0.202 | 1.25 |

(Within-time differences are weighted by the smaller group size at each time point. The
SD is pooled within the 8 conditions.)

Interpretation:

- Removing the rim does remove an offset: the control median of % >2x base drops from 4.69
  to 0.10.
- The expectation that the absolute differences would stay similar does **not** hold: the
  difference shrinks by about 25% to 85%, depending on the metric, and relative to the leaf-to-leaf variation the effect
  also becomes smaller (1.30 to 0.68 for % >2x base).
- The rim covers about 26% of the leaf, but contributes about two thirds of the whole-leaf
  difference in % >2x base.
- The interior metric does have a clean baseline, and it still ranks thrips above control
  well, which may be useful for interpretation. It is not the more sensitive metric.

## Limitations

- Small groups, and many metric variants compared on the same data, so small differences
  in separation mean little. Nothing was tested on an independent dataset.
- The 20 px rim width was chosen by hand; other widths were not tested.
- Separation only reflects ranking, and the ratios were pooled over time points.
- The signal is 8-bit and saturates at about 254, which caps percentiles and excess signal.
- The biological source of the rim signal is unknown. If it is mostly a response to
  cutting, including the rim may mix that response with thrips damage, even though it
  strengthens the difference here.

## Shelved implementation plan

The plan was to add three metrics, each for both threshold methods (`_leafthr`, `_refthr`).
Given the analysis above, these should be reconsidered before implementing; a rim metric
may be the more informative addition.

| Metric | Meaning |
|---|---|
| `interior_damage_percentage` | % of the leaf interior above the damage threshold |
| `excess_signal` | Mean excess above the threshold, over the whole leaf |
| `interior_excess_signal` | Same, over the leaf interior only |

Definitions:

- Interior: leaf pixels more than `leaf_rim_width_px` from the mask edge (default 20), with
  the image border counting as an edge.
- Excess signal: mean of `max(0, intensity - threshold) / threshold`. Dividing by the
  threshold gives one unitless formula for both methods.
- Metrics are 0 when no damage is found. Interior metrics are NaN if a leaf has no interior.

How it would fit the code:

1. `SampleMetrics`: add `leaf_rim_width_px` and the three metrics per method. Storing the
   width there adds it to the output tables, and lets `apply_reference_threshold` reuse it.
2. `DAMAGE_MASK_METRICS`: add the three names, so `plot_metric_per_condition` handles them
   without new plotting code.
3. `analyse_sample` and `run_complete_analysis`: add `leaf_rim_width_px=20`.
4. `fill_damage_mask_metrics`: add `img_damage`, read the threshold from
   `threshold_val_dmg_<suffix>`, determine the interior inline with
   `ndi.distance_transform_edt`, and convert the image to float to avoid integer wrap-around.
5. `apply_reference_threshold`: pass `img_damage` in the existing call.

Estimated size:

| Part | Lines |
|---|---:|
| `leafstats_analysis.py` | ~37 |
| Plot calls in the 4 runner scripts | ~73 |
| New test file | ~70 |
| Smoke test expected outputs | ~6 |
| `readme.md` and `notes/readme.md` | ~15 |
| **Total** | **~200** |

Verification would use a synthetic leaf with known values (both methods, no damage, rim of
0, rim wider than the leaf, damage only on the rim), a check that existing columns do not
change on the real data, inspection of the new plots, and all test scripts.
