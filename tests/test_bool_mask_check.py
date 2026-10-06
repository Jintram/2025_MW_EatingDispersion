"""
Written by Claude, and not human-checked.

Tests that functions which index an image with a mask raise a TypeError when
the mask is not boolean. (An integer mask would otherwise be interpreted as
row indices, silently giving wrong results; see the background_leaf bug fixed
on 6/10/2026.) Also checks that boolean masks are still accepted.

Run from the root of the repository:
    python tests/test_bool_mask_check.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import leafstats_analysis as lsa


def make_test_image():
    img = np.full((40, 40), 10, dtype=np.uint8)
    img[15:25, 15:25] = 200
    mask_bool = np.zeros(img.shape, dtype=bool)
    mask_bool[5:35, 5:35] = True
    return img, mask_bool


def calls_with_mask(img, mask):
    """All functions that should check the mask, called with the given mask."""
    return {
        'get_mask': lambda: lsa.get_mask(img, mask, method='baselvl2'),
        'calculate_mode_in_mask': lambda: lsa.calculate_mode_in_mask(img, mask),
        'get_radial_pdf': lambda: lsa.get_radial_pdf(img, (20, 20), mask_user=mask),
        'get_autocorrelation': lambda: lsa.get_autocorrelation(img, mask_user=mask),
    }


def test_integer_mask_raises():
    img, mask_bool = make_test_image()
    mask_int = mask_bool.astype(np.uint8)

    for name, call in calls_with_mask(img, mask_int).items():
        try:
            call()
        except TypeError:
            continue
        raise AssertionError(f"{name} accepted an integer mask without error")


def test_bool_mask_accepted():
    img, mask_bool = make_test_image()

    for name, call in calls_with_mask(img, mask_bool).items():
        call()  # should not raise

    assert lsa.calculate_mode_in_mask(img, mask_bool) == 10


def test_whole_image_mode_uses_all_pixels():
    """Regression test for the background_leaf bug: with a boolean all-ones
    mask the mode is taken over the whole image, not just over row 1."""
    img = np.full((40, 40), 10, dtype=np.uint8)
    img[1, :] = 99   # row 1 differs from the rest of the image

    mode = lsa.calculate_mode_in_mask(img, np.ones_like(img, dtype=bool))
    assert mode == 10, f"expected mode of whole image (10), got {mode}"


if __name__ == '__main__':
    for name, func in sorted(list(globals().items())):
        if name.startswith('test_') and callable(func):
            func()
            print(f'PASSED: {name}')
    print('\nAll tests passed.')
