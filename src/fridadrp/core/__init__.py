#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import astropy.units as u

FRIDA_NAXIS1_HAWAII = 2048 * u.pix # dispersion direction
FRIDA_NAXIS2_HAWAII = 2048 * u.pix # spatial direction (slices)

FRIDA_NAXIS1_IFU = 64 * u.pix # parallel to the slices
FRIDA_NAXIS2_IFU = 60 * u.pix # perpendicular to the slices

FRIDA_NSLICES = 30

FRIDA_DEFAULT_WAVELENGTH_UNIT = u.m

FRIDA_VALID_GRATINGS = ['low-zJ', 'low-JH',
                        'medium-z', 'medium-J', 'medium-H', 'medium-K',
                        'high-H', 'high-K']

FRIDA_VALID_SPATIAL_SCALES = ['fine', 'medium', 'coarse']

FRIDA_SPATIAL_SCALE = {
    'fine': 0.01 * u.arcsec / u.pix,
    'medium': 0.02 * u.arcsec / u.pix,
    'coarse': 0.04 * u.arcsec / u.pix
}