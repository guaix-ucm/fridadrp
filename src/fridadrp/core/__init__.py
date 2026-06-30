#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.units import Quantity
import astropy.units as u
import numpy as np

# use Quantity(...) to define integer parameters
# (otherwise integers are converted to float by default)
FRIDA_NAXIS1_HAWAII = Quantity(value=2048, unit=u.pix, dtype=int)  # dispersion direction
FRIDA_NAXIS2_HAWAII = Quantity(value=2048, unit=u.pix, dtype=int)  # spatial direction (slices)

FRIDA_NAXIS1_IFU = Quantity(value=64, unit=u.pix, dtype=int)  # parallel to the slices
FRIDA_NAXIS2_IFU = Quantity(value=60, unit=u.pix, dtype=int)  # perpendicular to the slices

FRIDA_NSLICES = 30

FRIDA_DEFAULT_WAVELENGTH_UNIT = u.m

FRIDA_VALID_GRATINGS = ["LOW-ZJ", "LOW-JH", "MEDIUM-Z", "MEDIUM-J", "MEDIUM-H", "MEDIUM-K", "HIGH-H", "HIGH-K"]

FRIDA_VALID_SPATIAL_SCALES = ["FINE", "MEDIUM", "COARSE"]

FRIDA_SPATIAL_SCALE = {
    "FINE": 0.01 * u.arcsec / u.pix,
    "MEDIUM": 0.02 * u.arcsec / u.pix,
    "COARSE": 0.04 * u.arcsec / u.pix,
}

# Define array for slice number from index conversions
DEF_SLICENUM_FROM_INDEX = np.array(
    [30, 1, 29, 2, 28, 3, 27, 4, 26, 5, 25, 6, 24, 7, 23, 8, 22, 9, 21, 10, 20, 11, 19, 12, 18, 13, 17, 14, 16, 15],
    dtype=int,
)  # slice number from index (0-29)
if len(DEF_SLICENUM_FROM_INDEX) != FRIDA_NSLICES:
    raise ValueError(
        f"Length of DEF_SLICENUM_FROM_INDEX ({len(DEF_SLICENUM_FROM_INDEX)}) does not match FRIDA_NSLICES ({FRIDA_NSLICES})"
    )


def slicenum_from_index(slice_index):
    """Convert slice index (0-29) to slice number (1-30)

    Parameters
    ----------
    slice_index : int or array-like of int
        Slice index (0-29).

    Returns
    -------
    slice_number : int or array-like of int
        Slice number (1-30).

    """
    if isinstance(slice_index, int):
        if slice_index < 0 or slice_index >= FRIDA_NSLICES:
            raise ValueError(f"Slice index must be in the range [0, {FRIDA_NSLICES - 1}]")
        return DEF_SLICENUM_FROM_INDEX[slice_index]
    elif isinstance(slice_index, (list, np.ndarray, range)):
        slice_index = np.asarray(slice_index)
        if np.any((slice_index < 0) | (slice_index >= FRIDA_NSLICES)):
            raise ValueError(f"Slice index must be in the range [0, {FRIDA_NSLICES - 1}]")
        return DEF_SLICENUM_FROM_INDEX[slice_index]
    else:
        raise TypeError("slice_index must be an int or array-like of int")


# Define array for slice index from slice number conversions
# (note: slice number is 1-30, slice index is 0-29; the first element of the array is a dummy value for non-existent slice number 0)
DEF_SLICEINDEX_FROM_NUM = np.array(
    [-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0],
    dtype=int,
)  # slice index (0-29) from slice number (1-30)
if len(DEF_SLICEINDEX_FROM_NUM) != FRIDA_NSLICES + 1:
    raise ValueError(
        f"Length of DEF_SLICEINDEX_FROM_NUM ({len(DEF_SLICEINDEX_FROM_NUM)}) does not match FRIDA_NSLICES + 1 ({FRIDA_NSLICES + 1})"
    )


def sliceindex_from_num(slice_number):
    """Convert slice number (1-30) to slice index (0-29)

    Parameters
    ----------
    slice_number : int or array-like of int
        Slice number (1-30).

    Returns
    -------
    slice_index : int or array-like of int
        Slice index (0-29).

    """
    if isinstance(slice_number, int):
        if slice_number < 1 or slice_number > FRIDA_NSLICES:
            raise ValueError(f"Slice number must be in the range [1, {FRIDA_NSLICES}]")
        return DEF_SLICEINDEX_FROM_NUM[slice_number]
    elif isinstance(slice_number, (list, np.ndarray, range)):
        slice_number = np.asarray(slice_number)
        if np.any((slice_number < 1) | (slice_number > FRIDA_NSLICES)):
            raise ValueError(f"Slice number must be in the range [1, {FRIDA_NSLICES}]")
        return DEF_SLICEINDEX_FROM_NUM[slice_number]
    else:
        raise TypeError("slice_number must be an int or array-like of int")
