#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Tests for set_hawaii_border_pixels_to_nan"""

import numpy as np
import pytest

from fridadrp.core import FRIDA_NAXIS1_HAWAII, FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import set_hawaii_border_pixels_to_nan

# The HAWAII detector has 4 reference pixels on each border. The useful region
# is written here explicitly, as 0-based array indices, so that the test does
# not depend on how the 1-based (FITS convention) constants are converted.
EXPECTED_FIRST_USEFUL_INDEX = 4  # 0-based
EXPECTED_LAST_USEFUL_INDEX = 2043  # 0-based


def test_useful_pixel_constants_fits_convention():
    # The constants follow the FITS convention (1-based)
    assert FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL.value == EXPECTED_FIRST_USEFUL_INDEX + 1
    assert FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL.value == EXPECTED_LAST_USEFUL_INDEX + 1
    assert FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value == EXPECTED_FIRST_USEFUL_INDEX + 1
    assert FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value == EXPECTED_LAST_USEFUL_INDEX + 1


def test_set_hawaii_border_pixels_to_nan():
    image = np.ones((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), dtype=float)
    result = set_hawaii_border_pixels_to_nan(image)

    # Expected mask of finite pixels: True only within the useful region
    i1, i2 = EXPECTED_FIRST_USEFUL_INDEX, EXPECTED_LAST_USEFUL_INDEX
    expected_finite = np.zeros_like(image, dtype=bool)
    expected_finite[i1 : i2 + 1, i1 : i2 + 1] = True

    np.testing.assert_array_equal(np.isfinite(result), expected_finite)
    # Useful pixels keep their original value
    assert np.all(result[expected_finite] == 1.0)
    # Explicit checks at the borders of the useful region (row, column)
    for idx in [i1, i2]:
        assert np.isfinite(result[idx, idx])
    for idx in [i1 - 1, i2 + 1]:
        assert np.isnan(result[idx, (i1 + i2) // 2])  # row outside the useful region
        assert np.isnan(result[(i1 + i2) // 2, idx])  # column outside the useful region
    # Number of useful pixels along each axis
    assert np.sum(np.isfinite(result).any(axis=0)) == 2040
    assert np.sum(np.isfinite(result).any(axis=1)) == 2040


def test_set_hawaii_border_pixels_to_nan_wrong_shape():
    with pytest.raises(ValueError):
        set_hawaii_border_pixels_to_nan(np.ones((100, 100), dtype=float))


def test_set_hawaii_border_pixels_to_nan_inplace():
    shape = (FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value)

    # inplace=False (default): the input is unchanged and a new array is returned
    image = np.ones(shape, dtype=float)
    result = set_hawaii_border_pixels_to_nan(image)
    assert result is not image
    assert np.all(image == 1.0)
    assert np.isnan(result).any()

    # inplace=True: the input itself is modified and returned
    image = np.ones(shape, dtype=float)
    result = set_hawaii_border_pixels_to_nan(image, inplace=True)
    assert result is image
    assert np.isnan(image[0, 0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex128])
def test_set_hawaii_border_pixels_to_nan_valid_dtypes(dtype):
    image = np.ones((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), dtype=dtype)
    result = set_hawaii_border_pixels_to_nan(image)
    assert result.dtype == dtype
    assert np.isnan(result[0, 0])


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.uint16, bool])
def test_set_hawaii_border_pixels_to_nan_invalid_dtypes(dtype):
    image = np.ones((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), dtype=dtype)
    for inplace in [False, True]:
        with pytest.raises(TypeError):
            set_hawaii_border_pixels_to_nan(image, inplace=inplace)


def test_set_hawaii_border_pixels_to_nan_not_ndarray():
    image = np.ones((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), dtype=float).tolist()
    with pytest.raises(TypeError):
        set_hawaii_border_pixels_to_nan(image)
