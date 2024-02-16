#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Auxiliary class for linear wavelength calibration"""

import astropy.units as u
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_VALID_GRATINGS


class LinearWaveCal(object):
    """Class to store a linear wavelength calibration.

    The parameters are stored making use of astropy units.

    Parameters
    ----------
    crpix1_wavecal : `~astropy.units.Quantity`
        CRPIX1 value of the wavelength calibrated spectrum.
    crval1_wavecal : `~astropy.units.Quantigy`
        CRVAL1 value of the wavelength calibrated spectrum.
    naxis1_wavecal : `~astropy.units.Quantity`
        NAXIS1 value of the wavelength calibrated spectrum.

    Attributes
    ----------
    crpix1_wavecal : `~astropy.units.Quantity`
        CRPIX1 value of the wavelength calibrated spectrum.
    crval1_wavecal : `~astropy.units.Quantigy`
        CRVAL1 value of the wavelength calibrated spectrum.
    naxis1_wavecal : `~astropy.units.Quantity`
        NAXIS1 value of the wavelength calibrated spectrum.

    Methods
    -------
    wave_at_pixel(pixel):
        Compute wavelength(s) at the pixel coordinate(s).
    pixel_at_wave(wavelength)
        Compute pixel coordinate(s) at the wavelength(s).

    """

    def __init__(self, crpix1_wavecal, crval1_wavecal, cdelt1_wavecal, naxis1_wavecal):
        # check parameters are astropy quantities
        if not isinstance(crpix1_wavecal, u.Quantity):
            raise ValueError(f"Object 'crpix1_wavecal' is not a Quantity instance")
        if not isinstance(crval1_wavecal, u.Quantity):
            raise ValueError(f"Object 'crval1_wavecal' is not a Quantity instance")
        if not isinstance(cdelt1_wavecal, u.Quantity):
            raise ValueError(f"Object 'cdelt1_wavecal' is not a Quantity instance")
        if not isinstance(naxis1_wavecal, u.Quantity):
            raise ValueError(f"Object 'naxis1_wavecal' is not a Quantity instance")

        # check expected units
        if crpix1_wavecal.unit != u.pix:
            raise ValueError(f"Unexpected unit for 'crpix1_wavecal': {crpix1_wavecal.unit}")
        if naxis1_wavecal.unit != u.pix:
            raise ValueError(f"Unexpected unit for 'naxis1_wavecal': {crpix1_wavecal.unit}")

        # check compatibility of units
        if cdelt1_wavecal.unit * u.pix != crval1_wavecal.unit:
            raise ValueError(
                f"Different wavelength units used for:\n"
                f"crval1_wavecal --> {crval1_wavecal.unit}\n"
                f"cdelt1_wavecal --> {cdelt1_wavecal.unit}\n"
                f"Employ the same wavelength unit in both cases to set default."
            )

        # define attributes
        self.crpix1_wavecal = crpix1_wavecal
        self.crval1_wavecal = crval1_wavecal
        self.cdelt1_wavecal = cdelt1_wavecal
        self.naxis1_wavecal = naxis1_wavecal

    @classmethod
    def define_from_grating(cls, grating):
        """Define class for a particular grating.

        Instantiates a LinearWaveCal object for the particular
        linear wavelength calibration parameters corresponding
        to the provided grating.

        Parameters
        ----------
        grating : str
            Grating name.

        """

        if grating not in FRIDA_VALID_GRATINGS:
            raise ValueError(f'Unexpected grating name: {grating}')

        crpix1 = 1.0 * u.pix
        if grating == 'medium-K':
            crval1 = 1.9344 * u.micrometer
            cdelt1 = 0.000285 * u.micrometer / u.pix
            naxis1 = FRIDA_NAXIS1_HAWAII
        else:
            raise ValueError(f"Invalid grating {grating}")

        self = LinearWaveCal(
            crpix1_wavecal=crpix1,
            crval1_wavecal=crval1,
            cdelt1_wavecal=cdelt1,
            naxis1_wavecal=naxis1
        )

        self.grating = grating

        return self

    def __str__(self):
        output = '<LinearWaveCal instance>'
        if hasattr(self, 'grating'):
            output += f' for grating: {self.grating}'
        output += '\n'
        output += f'crpix1_wavecal: {self.crpix1_wavecal}\n'
        output += f'crval1_wavecal: {self.crval1_wavecal}\n'
        output += f'cdelt1_wavecal: {self.cdelt1_wavecal}\n'
        output += f'naxis1_wavecal: {self.naxis1_wavecal}'
        return output

    def __repr__(self):
        output = f'LinearWaveCal(\n'
        output += f'    crpix1_wavecal={self.crpix1_wavecal.value} * {self.crpix1_wavecal.unit.__repr__()},\n'
        output += f'    crval1_wavecal={self.crval1_wavecal.value} * {self.crval1_wavecal.unit.__repr__()},\n'
        output += f'    cdelt1_wavecal={self.cdelt1_wavecal.value} * {self.cdelt1_wavecal.unit.__repr__()},\n'
        output += f'    naxis1_wavecal={self.naxis1_wavecal.value} * {self.naxis1_wavecal.unit.__repr__()}\n'
        output += ')'
        return output

    def __eq__(self, other):
        if isinstance(other, LinearWaveCal):
            return self.__dict__ == other.__dict__
        else:
            return False

    def wave_at_pixel(self, pixel):
        """Compute wavelength(s) at the pixel coordinate(s).

        Parameters
        ----------
        pixel : `~astropy.units.Quantity`
            A single number or a numpy array with pixel coordinates.
            The units used serve to decide the criterion used to
            indicate the coordinates: u.pix for the FITS system
            (which starts at 1) and u.dimensionless_unscaled to
            indicate that the positions correspond to indices of
            an array (which starts at 0).

        Returns
        -------
        wave : `~astropy.units.Quantity`
            Wavelength computed at the considered pixel value(s).

        """

        if not isinstance(pixel, u.Quantity):
            raise ValueError(f"Object 'pixel' is not a Quantity instance")

        if pixel.unit == u.pix:
            fitspixel = pixel
        elif pixel.unit == u.dimensionless_unscaled:
            fitspixel = (pixel.value + 1) * u.pix
        else:
            raise ValueError(f"Unexpected 'pixel' units: {pixel.unit}")

        wave = self.crval1_wavecal + (fitspixel - self.crpix1_wavecal) * self.cdelt1_wavecal

        return wave

    def pixel_at_wave(self, wave, return_units):
        """Compute pixel coordinate(s) at the wavelength(s).

        Parameters
        ----------
        wave : `~astropy.units.Quantity`
            A single number or a numpy array with wavelengths.
        return_units : `astropy.units.core.Unit`
            The return units serve to decide the criterion used to
            indicate the pixel coordinates: u.pix for the FITS system
            (which starts at 1) and u.dimensionless_unscaled to
            indicate that the positions correspond to indices of
            an array (which starts at 0).

        Returns
        -------
        pixel : `~astropy.units.Quantity`
            Pixel coordinates computed at the considered wavelength(s).

        """

        if not isinstance(wave, u.Quantity):
            raise ValueError(f"Object 'wave' is not an Quantity instance")

        if return_units not in [u.pix, u.dimensionless_unscaled]:
            raise ValueError(f'Unexpected return_units: {return_units}')

        waveunit = self.crval1_wavecal.unit
        fitspixel = (wave.to(waveunit) - self.crval1_wavecal) / self.cdelt1_wavecal + self.crpix1_wavecal

        if return_units == u.pix:
            return fitspixel
        else:
            return (fitspixel.value - 1) * u.dimensionless_unscaled
