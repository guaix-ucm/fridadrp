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

# Define array for slice ID from index conversions
DEF_SLICEID_FROM_SLICEINDEX = np.array(
    [1, 30, 2, 29, 3, 28, 4, 27, 5, 26, 6, 25, 7, 24, 8, 23, 9, 22, 10, 21, 11, 20, 12, 19, 13, 18, 14, 17, 15, 16],
    dtype=int,
)  # slice ID from index (0-29)
if len(DEF_SLICEID_FROM_SLICEINDEX) != FRIDA_NSLICES:
    raise ValueError(
        f"Length of DEF_SLICEID_FROM_SLICEINDEX ({len(DEF_SLICEID_FROM_SLICEINDEX)}) "
        f"does not match FRIDA_NSLICES ({FRIDA_NSLICES})"
    )


def sliceid_from_sliceindex(slice_index):
    """Convert slice index (0-29) to slice ID (1-30)

    Parameters
    ----------
    slice_index : int or array-like of int
        Slice index (0-29).

    Returns
    -------
    slice_id : int or array-like of int
        Slice ID (1-30).

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


# Define array for slice index from slice ID conversions
# (note: slice ID is 1-30, slice index is 0-29; the first element of the array
# is a dummy value for non-existent slice ID 0)
# Make use of the sliceid_from_sliceindex function to generate the array
DEF_SLICEINDEX_FROM_SLICEID = np.full(FRIDA_NSLICES + 1, -1, dtype=int)
for i in range(FRIDA_NSLICES):
    slice_id = sliceid_from_sliceindex(i)
    DEF_SLICEINDEX_FROM_SLICEID[slice_id] = i
if len(DEF_SLICEINDEX_FROM_SLICEID) != FRIDA_NSLICES + 1:
    raise ValueError(
        f"Length of DEF_SLICEINDEX_FROM_SLICEID ({len(DEF_SLICEINDEX_FROM_SLICEID)}) "
        f"does not match FRIDA_NSLICES + 1 ({FRIDA_NSLICES + 1})"
    )


def sliceindex_from_sliceid(slice_id):
    """Convert slice ID (1-30) to slice index (0-29)

    Parameters
    ----------
    slice_id : int or array-like of int
        Slice ID (1-30).

    Returns
    -------
    slice_index : int or array-like of int
        Slice index (0-29).

    """
    if isinstance(slice_id, (int, np.integer)):
        if slice_id < 1 or slice_id > FRIDA_NSLICES:
            raise ValueError(f"Slice ID must be in the range [1, {FRIDA_NSLICES}]")
        return DEF_SLICEINDEX_FROM_SLICEID[slice_id]
    elif isinstance(slice_id, (list, np.ndarray, range)):
        slice_id = np.asarray(slice_id)
        if np.any((slice_id < 1) | (slice_id > FRIDA_NSLICES)):
            raise ValueError(f"Slice ID must be in the range [1, {FRIDA_NSLICES}]")
        return DEF_SLICEINDEX_FROM_SLICEID[slice_id]
    else:
        raise TypeError(f"slice_id of type {type(slice_id)} must be an int or array-like of int")


def slicenum_list_from_slice_lists(slicenum_list, sliceid_list):
    """Return a list of slice numbers from the provided slicenum_list and sliceid_list

    This function facilitates the selection of slices to process based
    on either slice numbers or slice IDs. It checks for valid input and ensures
    that the provided lists are compatible. If both lists are provided, an error is raised.

    Parameters
    ----------
    slicenum_list : list of int
        List of slice numbers (1-30) to process. Use [0] to indicate all slices.
    sliceid_list : list of int
        List of slice IDs (1-30) to process. Use [0] to indicate all slices.

    Returns
    -------
    slicenum_to_process : list of int
        List of slice numbers to process.

    Raises
    ------
    TypeError
        If either `slicenum_list` or `sliceid_list` is not a list.
        If any element in `slicenum_list` or `sliceid_list` is not an integer
        or is a boolean.
    ValueError
        If either `slicenum_list` or `sliceid_list` is empty, contains
        duplicated or out-of-range values, contains 0 together with other
        values, or if both lists are provided simultaneously.

    Notes
    -----
    Slices are identified in two different ways:

    - Slice number (1-30): position of the slice in the detector,
      i.e. slice index (0-29) + 1, following the order in which the
      slices appear along the spatial direction of the HAWAII detector.
    - Slice ID (1-30): physical identifier of the slice in the IFU,
      given by `DEF_SLICEID_FROM_SLICEINDEX`.

    The two numberings do not coincide. For example, slice number 2
    (slice index 1) corresponds to slice ID 30.

    """

    # Check that the input lists are valid
    if not isinstance(slicenum_list, list):
        raise TypeError(f"slicenum_list must be a list, but is of type {type(slicenum_list)}")
    if not isinstance(sliceid_list, list):
        raise TypeError(f"sliceid_list must be a list, but is of type {type(sliceid_list)}")

    # Check that the input lists are not empty
    if len(slicenum_list) == 0:
        raise ValueError("slicenum_list cannot be empty")
    if len(sliceid_list) == 0:
        raise ValueError("sliceid_list cannot be empty")

    # Check that the input lists contain valid slice numbers and slice IDs
    for slicenum in slicenum_list:
        if not isinstance(slicenum, (int, np.integer)) or isinstance(slicenum, bool):
            raise TypeError(f"Each element of slicenum_list must be an int, but found {type(slicenum)}")
        if slicenum < 0 or slicenum > FRIDA_NSLICES:
            raise ValueError(f"Slice number {slicenum} in slicenum_list is out of range [0, {FRIDA_NSLICES}]")
    for sliceid in sliceid_list:
        if not isinstance(sliceid, (int, np.integer)) or isinstance(sliceid, bool):
            raise TypeError(f"Each element of sliceid_list must be an int, but found {type(sliceid)}")
        if sliceid < 0 or sliceid > FRIDA_NSLICES:
            raise ValueError(f"Slice ID {sliceid} in sliceid_list is out of range [0, {FRIDA_NSLICES}]")

    # Check that the input lists do not contain duplicates
    if len(slicenum_list) != len(set(slicenum_list)):
        raise ValueError("slicenum_list contains duplicate values")
    if len(sliceid_list) != len(set(sliceid_list)):
        raise ValueError("sliceid_list contains duplicate values")

    # Avoid incompatible choices of slicenum and sliceid
    if slicenum_list != [0] and sliceid_list != [0]:
        raise ValueError("Cannot specify both slicenum_list and sliceid_list. Please choose one.")
    if slicenum_list != [0] and 0 in slicenum_list:
        raise ValueError("Invalid value in slicenum_list: 0. Use 0 only to indicate all slices.")
    if sliceid_list != [0] and 0 in sliceid_list:
        raise ValueError("Invalid value in sliceid_list: 0. Use 0 only to indicate all slices.")
    if sliceid_list == [0]:
        # Convert np.int64 to int (and return a copy of the list)
        slicenum_to_process = [int(i) for i in slicenum_list]
    else:
        slicenum_to_process = [int(sliceindex_from_sliceid(i) + 1) for i in sliceid_list]
    if slicenum_to_process == [0]:
        slicenum_to_process = list(range(1, FRIDA_NSLICES + 1))  # All slices

    # Return the list of slice numbers to process
    return slicenum_to_process


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
