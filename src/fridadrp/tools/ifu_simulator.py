#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.io import fits
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np


def display_skycalc(faux_skycalc):
    """
    Display sky radiance and transmission.

    Data generated with the SKYCALC Sky Model Calculator tool
    provided by ESO.
    See https://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html

    Parameters
    ----------
    faux_skycalc : str
        FITS file name with SKYCALC predictions.
    """

    with fits.open(faux_skycalc) as hdul:
        skycalc_table = hdul[1].data
    wave = skycalc_table['lam']
    flux = skycalc_table['flux']
    trans = skycalc_table['trans']

    # plot radiance
    fig, ax = plt.subplots()
    ax.plot(wave, flux, '-', linewidth=1)
    ax.set_xlabel('Vacuum Wavelength (nm)')
    ax.set_ylabel('ph/s/m2/micron/arcsec2')
    ax.set_title('SKYCALC prediction')
    plt.tight_layout()
    plt.show()

    # plot transmission
    fig, ax = plt.subplots()
    ax.plot(wave, trans, '-', linewidth=1)
    ax.set_xlabel('Vacuum Wavelength (nm)')
    ax.set_ylabel('Transmission fraction')
    ax.set_title('SKYCALC prediction')
    plt.tight_layout()
    plt.show()


def simulate_constant_flux(wmin, wmax, nphotons, rng):
    """Simulate spectrum with constant flux (per unit wavelength).

    Parameters
    ----------
    wmin : `~astroppy.units.Quantity`
        Minimum wavelength to be considered.
    wmax : `~astroppy.units.Quantity`
        Maximum wavelength to be considered.
    nphotons : int
        Number of photons to be simulated.
    rng : `~numpy.random._generator.Generator`
        Random number generator.

    """

    if not isinstance(wmin, u.Quantity):
        raise ValueError(f"Object 'wmin': {wmin} is not a Quantity instance")
    if not isinstance(wmax, u.Quantity):
        raise ValueError(f"Object 'wmax': {wmax} is not a Quantity instance")
    if wmin.unit != wmax.unit:
        raise ValueError(f"Different units used for 'wmin' and 'wmax': {wmin.unit}, {wmax.unit}.\n"
                         "Employ the same unit to unambiguously define the output result.")

    simulated_wave = rng.uniform(low=wmin.value, high=wmax.value, size=nphotons)
    simulated_wave *= wmin.unit
    return simulated_wave


def simulate_delta_lines(line_wave, line_flux, nphotons, rng, wmin=None, wmax=None, plots=False):
    """Simulate spectrum defined from isolated wavelengths.

    Parameters
    ----------
    line_wave : `~astropy.units.Quantity`
        Numpy array (with astropy units) containing the individual
        wavelength of each line.
    line_flux : array_like
        Array-like object containing the individual flux of each line.
    nphotons : int
        Number of photons to be simulated
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    wmin : `~astroppy.units.Quantity`
        Minimum wavelength to be considered.
    wmax : `~astroppy.units.Quantity`
        Maximum wavelength to be considered.
    plots : bool
        If True, plot input and output results.

    Returns
    -------
    simulated_wave : `~astropy.units.Quantity`
        Wavelength of simulated photons.

    """

    line_flux = np.asarray(line_flux)
    if len(line_wave) != len(line_flux):
        raise ValueError(f"Incompatible array length: 'line_wave' ({len(line_wave)}), 'line_flux' ({len(line_flux)})")

    if not isinstance(line_wave, u.Quantity):
        raise ValueError(f"Object 'line_wave': {line_wave} is not a Quantity instance")
    wave_unit = line_wave.unit
    if not wave_unit.is_equivalent(u.m):
        raise ValueError(f"Unexpected unit for 'line_wave': {wave_unit}")

    # lower wavelength limit
    if wmin is not None:
        if not isinstance(wmin, u.Quantity):
            raise ValueError(f"Object 'wmin':{wmin}  is not a Quantity instance")
        if not wmin.unit.is_equivalent(u.m):
            raise ValueError(f"Unexpected unit for 'wmin': {wmin}")
        wmin = wmin.to(wave_unit)
        lower_index =np.searchsorted(line_wave.value, wmin.value, side='left')
    else:
        lower_index = 0

    # upper wavelength limit
    if wmax is not None:
        if not isinstance(wmax, u.Quantity):
            raise ValueError(f"Object 'wmax': {wmax} is not a Quantity instance")
        if not wmax.unit.is_equivalent(u.m):
            raise ValueError(f"Unexpected unit for 'wmax': {wmin}")
        wmax = wmax.to(wave_unit)
        upper_index =np.searchsorted(line_wave.value, wmax.value, side='right')
    else:
        upper_index = len(line_wave)

    print(lower_index, upper_index)

    if plots:
        fig, ax = plt.subplots()
        ax.stem(line_wave.value, line_flux, markerfmt=' ', basefmt=' ')
        if wmin is not None:
            ax.axvline(wmin.value, linestyle='--', color='gray')
        if wmax is not None:
            ax.axvline(wmax.value, linestyle='--', color='gray')
        ax.set_xlabel(f'Wavelength ({wave_unit})')
        ax.set_ylabel('Intensity (arbitrary units)')
        plt.tight_layout()
        plt.show()

    line_wave = line_wave[lower_index:upper_index]
    line_flux = line_flux[lower_index:upper_index]

    # normalized cumulative sum
    cumsum = np.cumsum(line_flux)
    cumsum /= cumsum[-1]

    if plots:
        fig, ax = plt.subplots()
        ax.plot(line_wave.value, cumsum, '-')
        if wmin is not None:
            ax.axvline(wmin.value, linestyle='--', color='gray')
        if wmax is not None:
            ax.axvline(wmax.value, linestyle='--', color='gray')
        ax.set_xlabel(f'Wavelength ({wave_unit})')
        ax.set_ylabel('Cumulative sum')
        plt.tight_layout()
        plt.show()

    # samples following a uniform distribution
    unisamples = rng.uniform(low=0, high=1, size=nphotons)

    # closest array indices in sorted array
    closest_indices = np.searchsorted(cumsum, unisamples, side='right')

    # simulated wavelengths
    simulated_wave = line_wave.value[closest_indices]
    simulated_wave *= wave_unit

    if plots:
        # count number of photons at each tabulated wavelength value
        x_spectrum, y_spectrum = np.unique(simulated_wave, return_counts=True)

        # scale factor to overplot expected spectrum with same total number
        # of photons as the simulated dataset
        factor = np.sum(line_flux) / nphotons

        # overplot expected and simulated spectrum
        fig, ax = plt.subplots()
        ax.stem(line_wave.value, line_flux / factor, markerfmt=' ', basefmt=' ')
        ax.plot(x_spectrum, y_spectrum, '.')
        if wmin is not None:
            ax.axvline(wmin.value, linestyle='--', color='gray')
        if wmax is not None:
            ax.axvline(wmax.value, linestyle='--', color='gray')
        ax.set_xlabel(f'Wavelength ({wave_unit})')
        ax.set_ylabel('Intensity (number of photons)')
        plt.tight_layout()
        plt.show()

    return simulated_wave


def ifu_simulator(faux_dict, wv_lincal, rng, verbose):
    """IFU simulator.

    Parameters
    ----------
    faux_dict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation
          x_ifu, y_ify, wavelength -> x_detector, y_detector
    wv_lincal : `fridadrp.processing.linear_wavelength_calibration.LinearWaveCal`
        Object that stores the linear wavelength calibration
        parameters: CRPIX1, CRVAL1, CDELT1 and NAXIS1.
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    verbose : bool
        If True, display/plot additional information.

    Returns
    -------
    """

    if verbose:
        for item in faux_dict:
            print(f'- Required file for item {item}:\n  {faux_dict[item]}')
        # display SKYCALC predictions for sky radiance and transmission
        display_skycalc(faux_skycalc=faux_dict['skycalc'])

    catlines = np.genfromtxt('lines_argon_neon_xenon_empirical_EMIR.dat')
    cat_wave = catlines[:, 0] / 10000 * u.micrometer
    cat_flux = catlines[:, 1]
    result = simulate_delta_lines(cat_wave, cat_flux, int(1E7), rng=rng,
                                  wmin=wv_lincal.wmin, wmax=wv_lincal.wmax, plots=True)
    print(type(result))
    print(len(result))
    print(result)

    result = simulate_constant_flux(wmin=wv_lincal.wmin, wmax=wv_lincal.wmax, nphotons=int(1E7), rng=rng)
    print(type(result))
    print(len(result))
    print(result)
