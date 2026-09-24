"""
Written by Claude, and not human-checked.

Tests that analyse_sample() only skips damage-mask dependent metrics when no
damage mask is found, and still calculates the metrics that only need the leaf
mask (ACF, radial PDF, damage threshold and background). Uses the synthetic
"noise" image, for which no damage is found.

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
    assert not np.isnan(metrics.background_dmg)

    # damage-mask dependent metrics are valid zeros
    assert metrics.island_counts == 0
    assert metrics.total_nearest_island_distances == 0
    assert metrics.mean_nearest_island_distance == 0
    assert metrics.total_damage_area_px == 0
    assert metrics.total_damage_percentage == 0


if __name__ == '__main__':
    test_nodamage_still_computes_leafmask_metrics()
    print('All tests passed.')
