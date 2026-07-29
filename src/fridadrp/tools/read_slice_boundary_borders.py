#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Read the slice boundary borders from a FITS file"""

from astropy.io import fits
import logging
import numpy as np

from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NSLICES


def check_is_a_valid_slice_boundary_border_file(hdul):
    """Check if the input FITS file is a valid slice boundary border file

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        HDUList object containing the FITS file to be checked.

    Raises
    ------
    ValueError
        If the input FITS file is not a valid slice boundary border file.
    """
    list_required_keywords = ["KEYCODE", "UUID"]
    for keyword in list_required_keywords:
        if keyword not in hdul[0].header:
            raise ValueError(f"Input file does not contain a {keyword} header keyword.")
    if hdul[0].header["KEYCODE"] != "SLICE_BOUNDARY_BORDERS_FROM_FLAT":
        raise ValueError(
            f"Invalid KEYCODE={hdul[0].header['KEYCODE']}.\nExpected value is 'SLICE_BOUNDARY_BORDERS_FROM_FLAT'."
        )
    list_required_extensions = ["L-BORDER", "R-BORDER", "SLIWIDTH"]
    for extname in list_required_extensions:
        if extname not in hdul:
            raise ValueError(f"Input file does not contain a {extname} extension.")


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
        Boolean array indicating the columns with NaN values
        in the collapsed borders (0-based).
    uuid_borders : str
        UUID of the slice boundary borders.
    islice_ok : np.ndarray
        Array containing the indices of the slices to be analyzed (0-based).
    """
    logger = logging.getLogger(__name__)

    # Check input file corresponds to the expected FITS file
    with fits.open(input_file) as hdul:
        check_is_a_valid_slice_boundary_border_file(hdul)
        uuid_borders = hdul[0].header["UUID"]
        islice_ok = []  # indices of slices to be analyzed (0-based index)
        for i in range(1, FRIDA_NSLICES + 1):
            kw = f"SLCNUM{i:02d}"
            if kw not in hdul[0].header:
                raise ValueError(f"Input file {input_file} does not contain the expected header keyword '{kw}'.")
            if hdul[0].header[kw]:
                islice_ok.append(i - 1)  # Store 0-based index of the slice
        islice_ok = np.array(islice_ok, dtype=int)
        slcnumt = hdul[0].header["SLCNUMT"]
        if slcnumt == len(islice_ok):
            logger.info(f"Number of slices defined: SLCNUMT={slcnumt}")
        else:
            raise ValueError(
                f"Input file {input_file} has inconsistent number of slices defined: SLCNUMT={slcnumt}, "
                f"but found {len(islice_ok)} slices with SLCNUMxx=True in the header."
            )
        for extname in ["L-BORDER", "R-BORDER"]:
            if extname not in hdul:
                raise ValueError(f"Input file {input_file} does not contain the expected extension '{extname}'.")
        array_left_border = hdul["L-BORDER"].data
        if array_left_border.shape != (FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value):
            raise ValueError(
                f"Input file {input_file} has unexpected shape for L-BORDER extension: {array_left_border.shape}. "
                f"Expected shape is ({FRIDA_NSLICES}, {FRIDA_NAXIS1_HAWAII.value})."
            )
        collapsed_left_border = np.sum(array_left_border[islice_ok, :], axis=0)
        ibad_left = np.isnan(collapsed_left_border)
        logger.info(f"Number of NaN values in collapsed left border : {np.sum(ibad_left)}")
        array_right_border = hdul["R-BORDER"].data
        if array_right_border.shape != (FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value):
            raise ValueError(
                f"Input file {input_file} has unexpected shape for R-BORDER extension: {array_right_border.shape}. "
                f"Expected shape is ({FRIDA_NSLICES}, {FRIDA_NAXIS1_HAWAII.value})."
            )
        collapsed_right_border = np.sum(array_right_border[islice_ok, :], axis=0)
        ibad_right = np.isnan(collapsed_right_border)
        logger.info(f"Number of NaN values in collapsed right border: {np.sum(ibad_right)}")
        if not np.array_equal(ibad_left, ibad_right):
            raise ValueError(
                "Mismatch between NaN values in collapsed left and right borders. "
                "The NaN values should be in the same positions."
            )
        ibad = ibad_left  # Use either ibad_left or ibad_right, they are the same

    return array_left_border, array_right_border, ibad, uuid_borders, islice_ok
