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

FRIDA_VALID_GRATINGS = ['low-zJ', 'low-JH',
                        'medium-z', 'medium-J', 'medium-H', 'medium-K',
                        'high-H', 'high-K']
