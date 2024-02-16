#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Wavelength calibration parameters for each grating configuration"""

import astropy.units as u
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_VALID_GRATINGS


def set_wv_parameters(grating):
    """Set wavelength calibration parameters for rectified images.

    Parameters
    ----------
    grating : str
        Grating name.

    Returns
    -------
    wv_parameters : dictionary
    Python dictionary containing relevant wavelength calibration
    parameters:
    - crpix1_wavecal
    - crval1_wavecal
    - cdelt1_wavecal
    - naxis1_wavecal

    """

    if grating not in FRIDA_VALID_GRATINGS:
        raise ValueError(f'Unexpected grating name: {grating}')

    # intialize output
    wv_parameters = {'crpix1_wavecal': 1.0 * u.pix}
    # set parameters
    if grating == 'medium-K':
        wv_parameters['crval1_wavecal'] = 1.9344 * u.micrometer
        wv_parameters['cdelt1_wavecal'] = 0.000285 * u.micrometer / u.pix
        wv_parameters['naxis1_waveval'] = FRIDA_NAXIS1_HAWAII
    else:
        raise ValueError(f"Invalid grating {grating}")

    return wv_parameters
