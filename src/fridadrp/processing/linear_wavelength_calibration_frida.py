#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Auxiliary class for linear wavelength calibration for FRIDA"""

import astropy.units as u
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_VALID_GRATINGS
from fridadrp.core import FRIDA_DEFAULT_WAVELENGTH_UNIT

from .linear_wavelength_calibration import LinearWaveCal


class LinearWaveCalFRIDA(LinearWaveCal):
    """Define a LinearWaveCal class for a particular grating.

    Instantiates a LinearWaveCal object for the particular
    linear wavelength calibration parameters corresponding
    to the provided grating.

    Parameters
    ----------
    grating : str
        Grating name.

    """

    def __init__(self, grating):
        if grating not in FRIDA_VALID_GRATINGS:
            raise ValueError(f'Unexpected grating name: {grating}')

        crpix1 = 1.0 * u.pix
        if grating == 'medium-K':
            crval1 = 1.9344 * u.micrometer
            cdelt1 = 0.000285 * u.micrometer / u.pix
            naxis1 = FRIDA_NAXIS1_HAWAII
        else:
            raise ValueError(f"Invalid grating {grating}")

        super().__init__(
            crpix1_wavecal=crpix1,
            crval1_wavecal=crval1,
            cdelt1_wavecal=cdelt1,
            naxis1_wavecal=naxis1,
            default_wavelength_unit=FRIDA_DEFAULT_WAVELENGTH_UNIT
        )

        self.grating = grating

    def __str__(self):
        output = super().__str__()
        output += f'\n<LinearWaveCalFRIDA(LinearWaveCal) instance> for grating: {self.grating}'
        return output

    def __repr__(self):
        output = f"LinearWaveCalFRIDA(grating='{self.grating}')"
        return output

    def __eq__(self, other):
        if isinstance(other, LinearWaveCalFRIDA):
            return self.__dict__ == other.__dict__
        else:
            return False
