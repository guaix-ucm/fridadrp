#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Overplot the slice boundary polynomials on image"""

from astropy.io import fits
import logging
import numpy as np
from numpy.polynomial import Polynomial

from fridadrp.core import FRIDA_NSLICES

def check_is_a_valid_slice_boundary_polynomial_file(hdul):
    """Check if the input FITS file is a valid slice boundary polynomial file

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        HDUList object containing the FITS file to be checked.

    Raises
    ------
    ValueError
        If the input FITS file is not a valid slice boundary polynomial file.
    """
    list_required_keywords = ["KEYCODE", "UUID", "POLDEG"]
    for keyword in list_required_keywords:
        if keyword not in hdul[0].header:
            raise ValueError(f"Input file does not contain a {keyword} header keyword.")
    if hdul[0].header["KEYCODE"] != "SLICE_BOUNDARY_POLYNOMIALS":
        raise ValueError(
            f"Invalid KEYCODE={hdul[0].header['KEYCODE']}.\nExpected value is 'SLICE_BOUNDARY_POLYNOMIALS'."
        )
    list_required_extensions = ["L-BORDER", "R-BORDER"]
    for extname in list_required_extensions:
        if extname not in hdul:
            raise ValueError(f"Input file does not contain a {extname} extension.")


def read_slice_boundary_polynomials(input_polynomial):
    """Read the slice boundary polynomials from a FITS file

    The polynomials are assumed to be fitted using as independent variable
    the array index along the NAXIS1 axis, which ranges from 0 to FRIDA_NAXIS1_HAWAII-1,
    and as dependent variable the array index along the NAXIS2 axis,
    which ranges from 0 to FRIDA_NAXIS2_HAWAII-1.

    The keywords SLICEINI and SLICEEND in the primary header of the input FITS file
    indicate the range of slices to be analyzed (1-based index). For the slices
    outside this range, the polynomial coefficients are expected to be NaN, and
    the returned list of Polynomial objects will contain None for those slices.

    Parameters
    ----------
    input_polynomial : str
        Path to the FITS file containing the slice boundary polynomials.

    Returns
    -------
    list_poly_left : list of Polynomial
        List of Polynomial objects for the left slice boundaries.
    list_poly_right : list of Polynomial
        List of Polynomial objects for the right slice boundaries.
    poldeg : int
        Degree of the polynomials.
    """
    logger = logging.getLogger(__name__)

    with fits.open(input_polynomial) as hdul:
        check_is_a_valid_slice_boundary_polynomial_file(hdul)
        poldeg = hdul[0].header["POLDEG"]
        islice_ok = []  # indices of slices to be analyzed (0-based index)
        for i in range(1, FRIDA_NSLICES + 1):
            kw = f"SLCNUM{i:02d}"
            if kw not in hdul[0].header:
                raise ValueError(f"Input file {input_polynomial} does not contain the expected header keyword '{kw}'.")
            if hdul[0].header[kw]:
                islice_ok.append(i - 1)  # Store 0-based index of the slice
        islice_ok = np.array(islice_ok, dtype=int)

        array_coefs_left = hdul["L-BORDER"].data
        naxis2_left, ncoeff_left = array_coefs_left.shape
        if naxis2_left != FRIDA_NSLICES:
            raise ValueError(
                f"Input file {input_polynomial} has {naxis2_left} slices, but FRIDA_NSLICES is {FRIDA_NSLICES}."
            )
        if ncoeff_left != poldeg + 1:
            raise ValueError(
                f"Input file {input_polynomial} has POLDEG={hdul[0].header['POLDEG']}, "
                f"but the L-BORDER extension has polynomial degree {ncoeff_left-1}."
            )
        logger.info(f"Reading {naxis2_left} slices with polynomial degree {ncoeff_left-1}.")

        list_poly_left = []
        for islice in range(FRIDA_NSLICES):
            if islice not in islice_ok:
                if not np.all(np.isnan(array_coefs_left[islice])):
                    raise ValueError(
                        f"Slice {islice+1} has valid polynomial coefficients in the input file {input_polynomial}, "
                        f"but SLCNUM{islice+1:02d} is False in the primary header."
                    )
                list_poly_left.append(None)
            else:
                if np.any(np.isnan(array_coefs_left[islice])):
                    raise ValueError(
                        f"Slice {islice+1} has invalid polynomial coefficients in the input file {input_polynomial}, "
                        f"but SLCNUM{islice+1:02d} is True in the primary header."
                    )
                list_poly_left.append(Polynomial(array_coefs_left[islice]))

        array_coefs_right = hdul["R-BORDER"].data
        naxis2_right, ncoeff_right = array_coefs_right.shape
        if ncoeff_right != poldeg + 1:
            raise ValueError(
                f"Input file {input_polynomial} has POLDEG={hdul[0].header['POLDEG']}, "
                f"but the R-BORDER extension has polynomial degree {ncoeff_right-1}."
            )
        if naxis2_right != FRIDA_NSLICES:
            raise ValueError(
                f"Input file {input_polynomial} has {naxis2_right} slices, but FRIDA_NSLICES is {FRIDA_NSLICES}."
            )
        logger.info(f"Reading {naxis2_right} slices with polynomial degree {ncoeff_right-1}.")

        list_poly_right = []
        for islice in range(FRIDA_NSLICES):
            if islice not in islice_ok:
                if not np.all(np.isnan(array_coefs_right[islice])):
                    raise ValueError(
                        f"Slice {islice+1} has valid polynomial coefficients in the input file {input_polynomial}, "
                        f"but SLCNUM{islice+1:02d} is False in the primary header."
                    )
                list_poly_right.append(None)
            else:
                if np.any(np.isnan(array_coefs_right[islice])):
                    raise ValueError(
                        f"Slice {islice+1} has invalid polynomial coefficients in the input file {input_polynomial}."
                    )
                list_poly_right.append(Polynomial(array_coefs_right[islice]))

    return list_poly_left, list_poly_right, poldeg
