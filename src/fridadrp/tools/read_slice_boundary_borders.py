#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Fit the slice boundaries determined from the flats"""

from astropy.io import fits
import logging
import numpy as np

from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NSLICES


def read_slice_boundary_borders(input_file):
    """Read the slice boundary borders from a FITS file

    The FITS file is expected to contain two extensions:
    "L-BORDER" and "R-BORDER", each with shape
    (FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value).

    Parameters
    ----------
    input_file : str
        Path to the FITS file containing the slice boundary borders.

    Returns
    -------
    array_left_border : np.ndarray
        Array containing the left slice boundaries.
    array_right_border : np.ndarray
        Array containing the right slice boundaries.
    ibad : np.ndarray
        Boolean array indicating the positions of NaN values in the collapsed borders.
    """
    logger = logging.getLogger(__name__)

    # Check input file corresponds to the expected FITS file
    with fits.open(input_file) as hdul:
        if "KEYCODE" not in hdul[0].header:
            raise ValueError(f"Input file {input_file} does not contain a KEYCODE header keyword.")
        keycode = hdul[0].header["KEYCODE"]
        if keycode != "SLICE_BOUNDARY_BORDERS_FROM_FLAT":
            raise ValueError(f"Invalid KEYCODE={keycode}.\nExpected value is 'SLICE_BOUNDARY_BORDERS_FROM_FLAT'.")
        for extname in ["L-BORDER", "R-BORDER"]:
            if extname not in hdul:
                raise ValueError(f"Input file {input_file} does not contain the expected extension '{extname}'.")
        array_left_border = hdul["L-BORDER"].data
        if array_left_border.shape != (FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value):
            raise ValueError(
                f"Input file {input_file} has unexpected shape for L-BORDER extension: {array_left_border.shape}. "
                f"Expected shape is ({FRIDA_NSLICES}, {FRIDA_NAXIS1_HAWAII.value})."
            )
        collapsed_left_border = np.sum(array_left_border, axis=0)
        ibad_left = np.isnan(collapsed_left_border)
        logger.info(f"Number of NaN values in collapsed left border : {np.sum(ibad_left)}")
        array_right_border = hdul["R-BORDER"].data
        if array_right_border.shape != (FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value):
            raise ValueError(
                f"Input file {input_file} has unexpected shape for R-BORDER extension: {array_right_border.shape}. "
                f"Expected shape is ({FRIDA_NSLICES}, {FRIDA_NAXIS1_HAWAII.value})."
            )
        collapsed_right_border = np.sum(array_right_border, axis=0)
        ibad_right = np.isnan(collapsed_right_border)
        logger.info(f"Number of NaN values in collapsed right border: {np.sum(ibad_right)}")
        if not np.array_equal(ibad_left, ibad_right):
            raise ValueError(
                "Mismatch between NaN values in collapsed left and right borders. "
                "The NaN values should be in the same positions."
            )
        ibad = ibad_left  # Use either ibad_left or ibad_right, they are the same

    return array_left_border, array_right_border, ibad
