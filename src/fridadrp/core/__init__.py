#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.units import Quantity
import astropy.units as u

# use Quantity(...) to define integer parameters
# (otherwise integers are converted to float by default)
FRIDA_NAXIS1_HAWAII = Quantity(value=2048, unit=u.pix, dtype=int)  # dispersion direction
FRIDA_NAXIS2_HAWAII = Quantity(value=2048, unit=u.pix, dtype=int)  # spatial direction (slices)

FRIDA_NAXIS1_IFU = Quantity(value=64, unit=u.pix, dtype=int)  # parallel to the slices
FRIDA_NAXIS2_IFU = Quantity(value=60, unit=u.pix, dtype=int)  # perpendicular to the slices

FRIDA_NSLICES = 30

FRIDA_DEFAULT_WAVELENGTH_UNIT = u.m

FRIDA_VALID_GRATINGS = ['LOW-ZJ', 'LOW-JH',
                        'MEDIUM-Z', 'MEDIUM-J', 'MEDIUM-H', 'MEDIUM-K',
                        'HIGH-H', 'HIGH-K']

FRIDA_VALID_SPATIAL_SCALES = ['FINE', 'MEDIUM', 'COARSE']

FRIDA_SPATIAL_SCALE = {
    'FINE': 0.01 * u.arcsec / u.pix,
    'MEDIUM': 0.02 * u.arcsec / u.pix,
    'COARSE': 0.04 * u.arcsec / u.pix
}