"""
Written by Claude, and not human-checked.

Tests that analyse_sample() only skips damage-mask dependent metrics when no
damage mask is found, and still calculates the metrics that only need the leaf
mask (ACF, radial PDF, damage threshold, base level and mean damage signal).
Uses the synthetic "noise" image, for which no damage is found.

Also tests that mean_dmg_signal equals the mean damage-channel intensity
within the leaf mask, using a small generated image.

Run from the root of the repository:
    python tests/test_analyse_sample_nodamage.py
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import leafstats_analysis as lsa

NOISE_IMAGE = os.path.join(REPO_ROOT, 'Synthetic_data/images/noise/synthetic_noise.tif')
CONFIG_CHANNELS = {'Leaf': 1, 'Damage': 2, 'Reference': 0}


def test_nodamage_still_computes_leafmask_metrics():
    metrics, arrays = lsa.analyse_sample(NOISE_IMAGE, 'noise', CONFIG_CHANNELS,
                                         leaf_threshold_method='otsu')

    # status: leaf found, damage not found
    assert metrics.leaf_found
    assert not metrics.damage_found
    assert metrics.analysis_status == 'no_damage_mask'

    # metrics that only need the leaf mask are calculated
    assert arrays.acf_norm is not None
    assert arrays.acf_norm_avgr is not None
    assert arrays.radial_pdf is not None
    assert not np.isnan(metrics.threshold_val_dmg)
    assert not np.isnan(metrics.baselvl_dmg)
    assert not np.isnan(metrics.mean_dmg_signal)

    # damage-mask dependent metrics are valid zeros
    assert metrics.island_counts == 0
    assert metrics.total_nearest_island_distances == 0
    assert metrics.mean_nearest_island_distance == 0
    assert metrics.total_damage_area_px == 0
    assert metrics.total_damage_percentage == 0


def test_mean_dmg_signal_value():
    """
    Leaf of 20x20 px in a 60x60 image; the damage channel is 10 in one half
    of the leaf and 30 in the other half, and 255 outside the leaf (which
    should be ignored). Expected mean_dmg_signal: 20.
    """
    import tempfile
    import imageio.v3 as iio

    img = np.zeros((60, 60, 3), dtype=np.uint8)
    img[20:40, 20:40, 1] = 100  # leaf channel
    img[:, :, 2] = 255          # damage channel, outside leaf
    img[20:40, 20:30, 2] = 10   # damage channel, leaf half 1
    img[20:40, 30:40, 2] = 30   # damage channel, leaf half 2

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, 'test_leaf.tif')
        iio.imwrite(file_path, img)
        metrics, _ = lsa.analyse_sample(file_path, 'test', CONFIG_CHANNELS,
                                        leaf_threshold_method='otsu')

    assert metrics.leaf_found
    assert metrics.total_leaf_size_px == 400
    assert metrics.mean_dmg_signal == 20, \
        f"expected 20, got {metrics.mean_dmg_signal}"


if __name__ == '__main__':
    test_nodamage_still_computes_leafmask_metrics()
    test_mean_dmg_signal_value()
    print('All tests passed.')
