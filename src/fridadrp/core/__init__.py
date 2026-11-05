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
FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL = Quantity(
    value=5, unit=u.pix, dtype=int
)  # first useful pixel in the dispersion direction
FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL = Quantity(
    value=2044, unit=u.pix, dtype=int
)  # last useful pixel in the dispersion direction
FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL = Quantity(
    value=5, unit=u.pix, dtype=int
)  # first useful pixel in the spatial direction
FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL = Quantity(
    value=2044, unit=u.pix, dtype=int
)  # last useful pixel in the spatial direction

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
DEF_SLICEID_FROM_SLICEINDEX = np.array(
    [1, 30, 2, 29, 3, 28, 4, 27, 5, 26, 6, 25, 7, 24, 8, 23, 9, 22, 10, 21, 11, 20, 12, 19, 13, 18, 14, 17, 15, 16],
    dtype=int,
)  # slice number from index (0-29)
if len(DEF_SLICEID_FROM_SLICEINDEX) != FRIDA_NSLICES:
    raise ValueError(
        f"Length of DEF_SLICENUM_FROM_INDEX ({len(DEF_SLICEID_FROM_SLICEINDEX)}) "
        f"does not match FRIDA_NSLICES ({FRIDA_NSLICES})"
    )


def sliceid_from_sliceindex(slice_index):
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
    if isinstance(slice_index, (int, np.integer)):
        if slice_index < 0 or slice_index >= FRIDA_NSLICES:
            raise ValueError(f"Slice index must be in the range [0, {FRIDA_NSLICES - 1}]")
        return DEF_SLICEID_FROM_SLICEINDEX[slice_index]
    elif isinstance(slice_index, (list, np.ndarray, range)):
        slice_index = np.asarray(slice_index)
        if np.any((slice_index < 0) | (slice_index >= FRIDA_NSLICES)):
            raise ValueError(f"Slice index must be in the range [0, {FRIDA_NSLICES - 1}]")
        return DEF_SLICEID_FROM_SLICEINDEX[slice_index]
    else:
        raise TypeError(f"slice_index of type {type(slice_index)} must be an int or array-like of int")


# Define array for slice index from slice number conversions
# (note: slice number is 1-30, slice index is 0-29; the first element of the array
# is a dummy value for non-existent slice number 0)
# Make use of the sliceid_from_sliceindex function to generate the array
DEF_SLICEINDEX_FROM_SLICEID = np.full(FRIDA_NSLICES + 1, -1, dtype=int)
for i in range(FRIDA_NSLICES):
    slice_number = sliceid_from_sliceindex(i)
    DEF_SLICEINDEX_FROM_SLICEID[slice_number] = i
if len(DEF_SLICEINDEX_FROM_SLICEID) != FRIDA_NSLICES + 1:
    raise ValueError(
        f"Length of DEF_SLICEINDEX_FROM_NUM ({len(DEF_SLICEINDEX_FROM_SLICEID)}) "
        f"does not match FRIDA_NSLICES + 1 ({FRIDA_NSLICES + 1})"
    )


def sliceindex_from_sliceid(slice_number):
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
    if isinstance(slice_number, (int, np.integer)):
        if slice_number < 1 or slice_number > FRIDA_NSLICES:
            raise ValueError(f"Slice number must be in the range [1, {FRIDA_NSLICES}]")
        return DEF_SLICEINDEX_FROM_SLICEID[slice_number]
    elif isinstance(slice_number, (list, np.ndarray, range)):
        slice_number = np.asarray(slice_number)
        if np.any((slice_number < 1) | (slice_number > FRIDA_NSLICES)):
            raise ValueError(f"Slice number must be in the range [1, {FRIDA_NSLICES}]")
        return DEF_SLICEINDEX_FROM_SLICEID[slice_number]
    else:
        raise TypeError(f"slice_number of type {type(slice_number)} must be an int or array-like of int")


def set_hawaii_border_pixels_to_nan(image_data, inplace=False):
    """Set the pixels outside the useful detector region to NaN.

    The useful region is defined by the constants
    FRIDA_NAXIS[12]_HAWAII_FIRST_USEFUL_PIXEL and
    FRIDA_NAXIS[12]_HAWAII_LAST_USEFUL_PIXEL, which follow the FITS
    convention (1-based). In 0-based array indices, the useful region
    spans FIRST-1 ... LAST-1 along each axis.

    Parameters
    ----------
    image_data : numpy.ndarray
        Input image data, with shape (FRIDA_NAXIS2_HAWAII, FRIDA_NAXIS1_HAWAII)
        and a floating point (or complex) dtype, so that NaN can be stored.
    inplace : bool, optional
        If True, `image_data` is modified in place and returned. If False
        (default), a modified copy is returned and `image_data` is left
        unchanged.

    Returns
    -------
    image_data : numpy.ndarray
        Image data with pixels outside the useful detector region set to NaN.
        It is the input array itself if `inplace` is True, and a copy
        otherwise.

    Raises
    ------
    TypeError
        If `image_data` is not a numpy.ndarray, or if its dtype cannot store
        NaN (e.g. an integer or boolean image).
    ValueError
        If `image_data` does not have the shape of the HAWAII detector.
    """
    # Check that the input is a numpy array whose dtype can store NaN
    if not isinstance(image_data, np.ndarray):
        raise TypeError(f"Input image data must be a numpy.ndarray, but is of type {type(image_data)}")
    if not np.issubdtype(image_data.dtype, np.inexact):
        raise TypeError(
            f"Input image data must have a floating point dtype to store NaN, but has dtype {image_data.dtype}"
        )

    # Check that the input image data has the correct shape
    if image_data.shape != (FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value):
        raise ValueError(
            f"Input image data must have shape ({FRIDA_NAXIS2_HAWAII.value}, {FRIDA_NAXIS1_HAWAII.value}), "
            f"but has shape {image_data.shape}"
        )

    if not inplace:
        image_data = image_data.copy()

    # Set to NaN the pixels outside the useful detector region.
    # The constants follow the FITS convention (1-based): the useful region
    # spans the 0-based array indices FIRST-1 ... LAST-1
    image_data[:, : FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL.value - 1] = np.nan
    image_data[:, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL.value :] = np.nan
    image_data[: FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value - 1, :] = np.nan
    image_data[FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value :, :] = np.nan

    return image_data
