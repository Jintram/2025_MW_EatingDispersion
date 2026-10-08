"""
Written by Claude, and not human-checked.

Tests apply_reference_threshold(), which derives one damage threshold from the
leaves of a reference condition and applies it to all samples, filling the
"_refthr" columns of df_samples and adding "mask_damage_refthr" to array_data.
Also tests the damage_mask_method argument of the plotting functions (output folder
selection).

Uses small generated 3-channel images (leaf channel 1, damage channel 2),
written to a temporary folder:
- 'Ctrl' leaves: damage base level 10, with a 10x10 damage spot of 25.
  Per-leaf threshold = 2x10 = 20, so the spot is detected.
- 'Raised' leaves: damage base level 15 (raised, e.g. by thrips activity), with
  the same 10x10 spot of 25. Per-leaf threshold = 2x15 = 30, so the spot is
  missed; with the reference threshold (20) it is detected.

Run from the root of the repository:
    python tests/test_reference_threshold.py
"""

import os
import sys
import tempfile
import warnings

import imageio.v3 as iio
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import leafstats_analysis as lsa

CONFIG_CHANNELS = {'Leaf': 1, 'Damage': 2, 'Reference': None}
DAMAGE_METRICS_TO_COMPARE = ['damage_found', 'analysis_status', 'island_counts',
                             'total_damage_area_px', 'total_damage_percentage',
                             'mean_nearest_island_distance', 'threshold_val_dmg']


def make_leaf_image(base_level, spot_value=25):
    """80x80 image with a 40x40 leaf; damage channel has base level + one spot."""
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    img[20:60, 20:60, 1] = 100                  # leaf
    img[20:60, 20:60, 2] = base_level           # damage channel base level
    img[30:40, 30:40, 2] = spot_value           # damage spot
    return img


def run_analysis(base_levels_per_condition):
    """Write images per condition to a temp dir and run run_complete_analysis."""
    tmpdir = tempfile.mkdtemp(prefix='leafstats_refthr_')
    data_file_paths = {}
    for condition, base_levels in base_levels_per_condition.items():
        data_file_paths[condition] = []
        for i, base_level in enumerate(base_levels):
            path = os.path.join(tmpdir, f'{condition}_{i}.tif')
            iio.imwrite(path, make_leaf_image(base_level))
            data_file_paths[condition].append(path)
    return lsa.run_complete_analysis(data_file_paths, CONFIG_CHANNELS)


def test_refthr_equals_leafthr_when_thresholds_equal():
    """Reference condition with identical leaves: refthr result == leafthr result."""
    df, arr = run_analysis({'Ctrl': [10, 10]})
    df, arr = lsa.apply_reference_threshold(df, arr, 'Ctrl')

    for name in DAMAGE_METRICS_TO_COMPARE:
        # (compare values, not dtypes; int32 vs int64 may differ)
        assert df[f'{name}_refthr'].tolist() == df[f'{name}_leafthr'].tolist(), \
            f"{name}: refthr differs from leafthr"
    for fp in arr:
        assert np.array_equal(arr[fp]['mask_damage_refthr'], arr[fp]['mask_damage_leafthr'])
    assert (df['reference_condition'] == 'Ctrl').all()


def test_raised_base_level_detected_with_refthr():
    """Damage hidden by a raised base level is found with the reference threshold."""
    df, arr = run_analysis({'Ctrl': [10, 10], 'Raised': [15, 15]})
    df, arr = lsa.apply_reference_threshold(df, arr, 'Ctrl')

    raised = df['condition'] == 'Raised'
    assert (df.loc[raised, 'total_damage_area_px_leafthr'] == 0).all(), \
        "expected per-leaf threshold to miss the spot for raised base level"
    assert (df.loc[raised, 'total_damage_area_px_refthr'] == 100).all(), \
        "expected reference threshold to detect the 10x10 spot"
    assert (df['threshold_val_dmg_refthr'] == 20).all()
    assert (df.loc[raised, 'threshold_val_dmg_leafthr'] == 30).all()


def test_refthr_columns_empty_before_apply():
    df, arr = run_analysis({'Ctrl': [10]})
    assert df['threshold_val_dmg_refthr'].isna().all()
    assert (df['analysis_status_refthr'] == 'not_calculated').all()
    assert arr[df['file_path'][0]]['mask_damage_refthr'] is None


def test_warning_when_reference_thresholds_spread():
    """Median 20, mean ~33: should warn (>20% difference)."""
    df, arr = run_analysis({'Ctrl': [10, 10, 30]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        df, _ = lsa.apply_reference_threshold(df, arr, 'Ctrl')
    assert any('spread' in str(w.message) for w in caught), \
        "expected a warning about spread reference thresholds"
    assert (df['threshold_val_dmg_refthr'] == 20).all(), "expected the median (20)"


def test_no_warning_when_reference_thresholds_close():
    df, arr = run_analysis({'Ctrl': [10, 10, 11]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        lsa.apply_reference_threshold(df, arr, 'Ctrl')
    assert not any('spread' in str(w.message) for w in caught)


def test_unknown_reference_condition_raises():
    df, arr = run_analysis({'Ctrl': [10]})
    try:
        lsa.apply_reference_threshold(df, arr, 'DoesNotExist')
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown reference condition")


def test_outputdir_selection():
    """damage_mask_method selects the output subfolder; refthr requires apply_reference_threshold."""
    df, arr = run_analysis({'Ctrl': [10, 10]})

    assert lsa.get_damage_mask_outputdir(df, 'OUT', 'leafthr') == os.path.join('OUT', 'damage_mask_leafthr')
    for bad_damage_mask in ['refthr', 'something_else']:
        try:
            lsa.get_damage_mask_outputdir(df, 'OUT', bad_damage_mask)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for damage_mask_method='{bad_damage_mask}'")

    df, arr = lsa.apply_reference_threshold(df, arr, 'Ctrl')
    assert lsa.get_damage_mask_outputdir(df, 'OUT', 'refthr') == os.path.join('OUT', 'damage_mask_refthr')


def test_plot_metric_per_condition_routing():
    """Damage-mask dependent metrics go to the method subfolder, others to outputdir."""
    df, arr = run_analysis({'Ctrl': [10, 10], 'Raised': [15, 15]})
    df, arr = lsa.apply_reference_threshold(df, arr, 'Ctrl')
    outdir = tempfile.mkdtemp(prefix='leafstats_refthr_plots_')

    lsa.plot_metric_per_condition(df, outdir, metric_key='threshold_val_dmg', damage_mask_method='refthr')
    lsa.plot_metric_per_condition(df, outdir, metric_key='baselvl_dmg', damage_mask_method='refthr')

    assert os.path.isfile(os.path.join(outdir, 'damage_mask_refthr', 'plots', 'threshold_val_dmg.png'))
    assert os.path.isfile(os.path.join(outdir, 'plots', 'baselvl_dmg.png'))
    assert not os.path.exists(os.path.join(outdir, 'damage_mask_refthr', 'plots', 'baselvl_dmg.png'))


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    for name, func in sorted(list(globals().items())):
        if name.startswith('test_') and callable(func):
            func()
            print(f'PASSED: {name}')
    print('\nAll tests passed.')
