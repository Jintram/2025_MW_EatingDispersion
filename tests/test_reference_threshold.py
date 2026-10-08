"""
Written by Claude, and not human-checked.

Tests apply_reference_threshold(), which derives one damage threshold from the
leaves of a reference condition and applies it to all samples.

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
DAMAGE_COLUMNS = ['damage_found', 'analysis_status', 'island_counts',
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


def test_reference_equals_per_leaf_when_thresholds_equal():
    """Reference condition with identical leaves: ref result == per-leaf result."""
    df, arr = run_analysis({'Ctrl': [10, 10]})
    df_ref, arr_ref = lsa.apply_reference_threshold(df, arr, 'Ctrl')

    assert df_ref[DAMAGE_COLUMNS].equals(df[DAMAGE_COLUMNS]), \
        "reference-threshold metrics differ from per-leaf metrics"
    for fp in arr:
        assert np.array_equal(arr_ref[fp]['mask_damage'], arr[fp]['mask_damage'])
    assert (df['damage_threshold_method'] == 'per_leaf').all()
    assert (df_ref['damage_threshold_method'] == 'ref_Ctrl').all()


def test_raised_base_level_detected_with_reference():
    """Damage hidden by a raised base level is found with the reference threshold."""
    df, arr = run_analysis({'Ctrl': [10, 10], 'Raised': [15, 15]})
    df_ref, arr_ref = lsa.apply_reference_threshold(df, arr, 'Ctrl')

    raised = df['condition'] == 'Raised'
    assert (df.loc[raised, 'total_damage_area_px'] == 0).all(), \
        "expected per-leaf threshold to miss the spot for raised base level"
    assert (df_ref.loc[raised, 'total_damage_area_px'] == 100).all(), \
        "expected reference threshold to detect the 10x10 spot"
    assert (df_ref['threshold_val_dmg'] == 20).all()
    # non-damage-mask metrics are untouched
    assert df_ref['baselvl_dmg'].equals(df['baselvl_dmg'])
    assert df_ref['mean_dmg_signal'].equals(df['mean_dmg_signal'])
    # images are shared, not copied
    for fp in arr:
        assert arr_ref[fp]['img_damage'] is arr[fp]['img_damage']


def test_warning_when_reference_thresholds_spread():
    """Median 20, mean ~33: should warn (>20% difference)."""
    df, arr = run_analysis({'Ctrl': [10, 10, 30]})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        df_ref, _ = lsa.apply_reference_threshold(df, arr, 'Ctrl')
    assert any('spread' in str(w.message) for w in caught), \
        "expected a warning about spread reference thresholds"
    assert (df_ref['threshold_val_dmg'] == 20).all(), "expected the median (20)"


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


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    for name, func in sorted(list(globals().items())):
        if name.startswith('test_') and callable(func):
            func()
            print(f'PASSED: {name}')
    print('\nAll tests passed.')
