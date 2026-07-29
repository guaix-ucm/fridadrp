#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Read the polynomials that define the traces of the slices from a FITS file"""

from astropy.io import fits
import logging
import numpy as np

from fridadrp.core import FRIDA_NSLICES


def check_is_a_valid_slice_trace_polynomial_file(hdul):
    """Check if the input FITS file is a valid slice trace polynomial file

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        HDUList object containing the FITS file to be checked.

    Raises
    ------
    ValueError
        If the input FITS file is not a valid slice trace polynomial file.
    """
    list_required_keywords = ["KEYCODE", "UUID", "POLDEG", "TRACDEG", "TRACESLC", "SLCNUMT"]
    for keyword in list_required_keywords:
        if keyword not in hdul[0].header:
            raise ValueError(f"Input file does not contain a {keyword} header keyword.")
    expected_keycode_values = ["SLICE_TRACES_POLYNOMIALS"]
    if hdul[0].header["KEYCODE"] not in expected_keycode_values:
        raise ValueError(
            f"Invalid KEYCODE={hdul[0].header['KEYCODE']}.\nExpected value is one of {expected_keycode_values}."
        )
    list_required_extensions = ["L-BORDER", "R-BORDER", "SLIWIDTH"]
    for i in range(1, FRIDA_NSLICES + 1):
        kw = f"SLCNUM{i:02d}"
        list_required_extensions.append(kw)
    for extname in list_required_extensions:
        if extname not in hdul:
            raise ValueError(f"Input file does not contain a {extname} extension.")


def read_slice_trace_polynomials(input_polynomial):
    """Read the slice trace polynomials from a FITS file

    The polynomials are assumed to be fitted using as independent variable
    the array index along the NAXIS1 axis, which ranges from 0 to FRIDA_NAXIS1_HAWAII-1,
    and as dependent variable the array index along the NAXIS2 axis,
    which ranges from 0 to FRIDA_NAXIS2_HAWAII-1.

    Parameters
    ----------
    input_polynomial : str
        Path to the FITS file containing the slice trace polynomials.

    Returns
    -------
    list_poly_traces_all_slices : list of list of Polynomial
        List of lists of Polynomial objects for the traces in each slice.
    """
    logger = logging.getLogger(__name__)

    with fits.open(input_polynomial) as hdul:
        check_is_a_valid_slice_trace_polynomial_file(hdul)
    
        ntraces_per_slice = hdul[0].header["TRACESLC"]
        tracdeg = hdul[0].header["TRACDEG"]
        if hdul[0].header["SLCNUMT"] != FRIDA_NSLICES:
            raise ValueError(
                f"Invalid SLCNUMT={hdul[0].header['SLCNUMT']}.\nExpected value is {FRIDA_NSLICES}."
            )

        list_poly_traces_all_slices = []
        for islice in range(FRIDA_NSLICES):
            extname = f"SLCNUM{islice+1:02d}"
            array2d_coeffs = hdul[extname].data
            if array2d_coeffs.shape[0] != ntraces_per_slice:
                raise ValueError(
                    f"Invalid number of traces in extension {extname}.\nExpected value is {ntraces_per_slice}."
                )
            if array2d_coeffs.shape[1] != tracdeg + 1:
                raise ValueError(
                    f"Invalid number of polynomial coefficients in extension {extname}.\nExpected value is {tracdeg + 1}."
                )
            list_poly_traces_slice = []
            for itrace in range(ntraces_per_slice):
                coeffs = array2d_coeffs[itrace]
                if np.any(np.isnan(coeffs)):
                    raise ValueError(
                        f"NaN values found in polynomial coefficients for trace {itrace} in extension {extname}."
                    )
                poly = np.polynomial.Polynomial(coeffs)
                list_poly_traces_slice.append(poly)
            list_poly_traces_all_slices.append(list_poly_traces_slice)

    return list_poly_traces_all_slices
