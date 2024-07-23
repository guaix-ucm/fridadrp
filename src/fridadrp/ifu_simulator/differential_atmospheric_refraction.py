#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import astropy.units as u
import numpy as np

from numina.tools.ctext import ctext


def air_refractive_index_15_760(wave_vacuum):
    """Air refractive index (dry air at sea level)

    Equation (1) from Filippenko (1982): refractive index of dry air
    at sea level (P=760 mm Hg, T=15 degree Celsius).

    Parameters
    ----------
    wave_vacuum : `~astropy.units.Quantity`
        Wavelength (vacuum).

    Returns
    -------
    n_air : float
        Refractive index of dray air.

    """

    wave_vacuum_micron = wave_vacuum.to(u.micron).value

    n_air = 64.328 + 29498.1 / (146 - (1 / wave_vacuum_micron) ** 2) + 255.4 / (41 - (1 / wave_vacuum_micron) ** 2)
    n_air = 1 + n_air * 1E-6

    return n_air


def air_refractive_index(wave_vacuum, temperature, pressure_mm, pressure_water_mm):
    """Air refractive index (general case)

    Equations (2) and (3) from Filippenko (1982).

    Parameters
    ----------
    wave_vacuum : `~astropy.units.Quantity`
        Wavelength (vacuum).
    temperature : `~astropy.units.Quantity`
        Temperature.
    pressure_mm : float
        Pressure (mm Hg).
    pressure_water_mm : float
        Water vapour pressure (mm Hg).

    Returns
    -------
    n_air : float
        Refractive index of air.

    """

    wave_vacuum_micron = wave_vacuum.to(u.micron).value
    temperature_value = temperature.to(u.Celsius).value

    n_air = (pressure_mm * (1 + (1.049 - 0.0157 * temperature_value) * 1E-6 * pressure_mm)) / \
            (720.833 * (1 + 0.003661 * temperature_value))
    n_air = 1 + (air_refractive_index_15_760(wave_vacuum) - 1) * n_air

    water_factor = (0.0624 - 0.000680/wave_vacuum_micron**2)/(1 + 0.003661 * temperature_value) * pressure_water_mm
    n_air -= water_factor * 1E-6

    return n_air


def compute_differential_atmospheric_refraction(
        airmass,
        reference_wave_differential_refraction,
        simulated_wave,
        verbose=False
):
    """Compute differential atmospheric refraction

    Parameters
    ----------
    airmass : float
        Airmass.
    reference_wave_differential_refraction : `~astropy.units.Quantity`
        Reference wavelength to compute the differential refraction
        correction. This wavelength corresponds to a correction of
        zero.
    simulated_wave : `~astropy.units.Quantity`
        Array containing `nphotons` simulated photons with the
        spectrum requested in the scene block. These values are
        required to compute the differential refraction correction.
    verbose : bool
        If True, display additional information.

    Returns
    -------
    differential_refraction : `~astropy.units.Quantity`
        Differential refraction at each simulated wavelength.

    """

    if airmass < 1.0:
        raise ValueError(f'Unexpected {airmass=}')

    # zenith distance
    zenith_distance = np.rad2deg(np.arccos(1/airmass)) * u.deg
    if verbose:
        print(ctext(f'{airmass=} --> {zenith_distance=}', faint=True))

    # air refractive index for reference wavelength, at the conditions
    # employed by Filippenko (1982)
    n_air_reference = air_refractive_index(
        wave_vacuum=reference_wave_differential_refraction,
        temperature=7*u.Celsius,
        pressure_mm=600,
        pressure_water_mm=8
    )
    if verbose:
        print(ctext(f'{reference_wave_differential_refraction=}', faint=True))
        print(ctext(f'{n_air_reference=}', faint=True))

    # air refractive index for all the simulated wavelengths
    n_air = air_refractive_index(
        wave_vacuum=simulated_wave,
        temperature=7*u.Celsius,
        pressure_mm=600,
        pressure_water_mm=8
    )

    # refraction (plane-parallel atmosphere)
    refraction_reference = (n_air_reference - 1) * np.tan(zenith_distance)
    refraction = (n_air - 1) * np.tan(zenith_distance)
    factor_arcsec_per_radian = 206264.80624709636
    differential_refraction = (refraction - refraction_reference) * factor_arcsec_per_radian * u.arcsec
    if verbose:
        imin = np.argmin(differential_refraction)
        imax = np.argmax(differential_refraction)
        print(ctext(f'Minimum differential refraction: {differential_refraction[imin]:+.4f} ' +
                    f'at wavelength: {simulated_wave[imin]}', faint=True))
        print(ctext(f'Maximum differential refraction: {differential_refraction[imax]:+.4f} ' +
                    f'at wavelength: {simulated_wave[imax]}', faint=True))

    return differential_refraction
