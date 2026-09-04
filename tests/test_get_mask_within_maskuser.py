"""
Written by Claude, and not human-checked.

Tests that get_mask() only returns pixels inside mask_user, ie that the damage
mask cannot contain signal from outside the leaf.

Run from the root of the repository:
    python tests/test_get_mask_within_maskuser.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import leafstats_analysis as lsa


def make_test_image():
    """
    Image with a dim background, a leaf region containing one bright spot, and
    a second bright spot that lies *outside* the leaf region.
    """
    img = np.full((60, 120), 10, dtype=np.uint8)
    mask_leaf = np.zeros(img.shape, dtype=bool)
    mask_leaf[10:50, 10:50] = True

    img[20:30, 20:30] = 200   # damage on the leaf
    img[20:30, 80:90] = 200   # bright spot off the leaf

    return img, mask_leaf


def test_mask_is_within_mask_user():
    img, mask_leaf = make_test_image()

    mask_damage, threshold_val = lsa.get_mask(img, mask_leaf, method='bg2')

    assert np.all(mask_damage <= mask_leaf), \
        "damage mask contains pixels outside mask_user"
    assert np.sum(mask_damage) == 100, \
        f"expected only the 10x10 on-leaf spot, got {np.sum(mask_damage)} px"
    assert not np.any(mask_damage[20:30, 80:90]), \
        "off-leaf bright spot was included in the damage mask"
    assert threshold_val == 20, f"expected 2x the mode (10), got {threshold_val}"


def test_no_mask_user_keeps_everything():
    """Without mask_user the behaviour should be unchanged (both spots found)."""
    img, _ = make_test_image()

    mask_damage, _ = lsa.get_mask(img, method='bg2')

    assert np.sum(mask_damage) == 200, \
        f"expected both 10x10 spots, got {np.sum(mask_damage)} px"


def test_run_complete_analysis_area_within_leaf():
    """The damage area reported per image should never exceed the leaf area."""
    condition_path_map = {'Ctrl': 'Example_data/DATA/condition_Control',
                          'Edited': 'Example_data/DATA/condition_Photoshopped'}
    config_channels = {'Leaf': 1, 'Damage': 2, 'Reference': 0}

    data_file_paths = lsa.get_data_file_paths(condition_path_map)
    df_samples, array_data = lsa.run_complete_analysis(
        data_file_paths=data_file_paths,
        config_channels=config_channels,
        leaf_threshold_method='bg10'
    )

    for _, row in df_samples.iterrows():
        assert row['total_damage_area_px'] <= row['total_leaf_size_px'], \
            f"damage area exceeds leaf area for {row['file_path']}"
        assert 0 <= row['total_damage_percentage'] <= 100, \
            f"damage percentage out of range for {row['file_path']}"

        arrays = array_data[row['file_path']]
        assert np.all(arrays['mask_damage'] <= arrays['mask_leaf']), \
            f"damage mask extends beyond leaf mask for {row['file_path']}"


def test_empty_mask_user_returns_expected_number_of_values():
    """An empty mask_user should still return the documented number of values."""
    img, _ = make_test_image()
    empty_mask = np.zeros(img.shape, dtype=bool)

    mask_damage, threshold_val = lsa.get_mask(img, empty_mask, method='bg2')
    assert not np.any(mask_damage)
    assert np.isnan(threshold_val)

    mask_damage, threshold_val, found = lsa.get_mask(
        img, empty_mask, method='bg2', return_status=True)
    assert not np.any(mask_damage)
    assert np.isnan(threshold_val)
    assert found is False


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')

    for name, func in sorted(list(globals().items())):
        if name.startswith('test_') and callable(func):
            func()
            print(f'PASSED: {name}')
    print('\nAll tests passed.')
