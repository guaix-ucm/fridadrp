#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import astropy.units as u

from .raise_valueerror import raise_ValueError


def simulate_constant_photlam(wmin, wmax, nphotons, rng):
    """Simulate spectrum with constant flux (in PHOTLAM units).

    Parameters
    ----------
    wmin : `~astropy.units.Quantity`
        Minimum wavelength to be considered.
    wmax : `~astropy.units.Quantity`
        Maximum wavelength to be considered.
    nphotons : int
        Number of photons to be simulated.
    rng : `~numpy.random._generator.Generator`
        Random number generator.

    """

    if not isinstance(wmin, u.Quantity):
        raise_ValueError(f"Object 'wmin': {wmin} is not a Quantity instance")
    if not isinstance(wmax, u.Quantity):
        raise_ValueError(f"Object 'wmax': {wmax} is not a Quantity instance")
    if wmin.unit != wmax.unit:
        raise_ValueError(f"Different units used for 'wmin' and 'wmax': {wmin.unit}, {wmax.unit}.\n" +
                         "Employ the same unit to unambiguously define the output result.")

    simulated_wave = rng.uniform(low=wmin.value, high=wmax.value, size=nphotons)
    simulated_wave *= wmin.unit
    return simulated_wave
