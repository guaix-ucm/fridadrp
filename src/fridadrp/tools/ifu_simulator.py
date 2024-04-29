#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
from astropy import wcs
import astropy.constants as constants
from astropy.io import fits
from astropy.units import Quantity
import astropy.units as u
from astropy.units import Unit
from astropy.visualization import ZScaleInterval
from joblib import Parallel, delayed
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import os
import pprint
from scipy.signal import convolve2d
import time
import yaml

from fridadrp.processing.define_3d_wcs import get_wvparam_from_wcs3d
from numina.array.distortion import fmap
from numina.array.distortion import compute_distortion, rectify2d
from numina.array.display.polfit_residuals import polfit_residuals
from numina.tools.ctext import ctext


pp = pprint.PrettyPrinter(indent=1, sort_dicts=False)


def raise_ValueError(msg):
    """Raise exception showing a coloured message."""
    raise ValueError(ctext(msg, fg='red'))


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
        skycalc_header = hdul[1].header
        skycalc_table = hdul[1].data

    if skycalc_header['TTYPE1'] != 'lam':
        raise_ValueError(f"Unexpected TTYPE1: {skycalc_header['TTYPE1']}")
    if skycalc_header['TTYPE2'] != 'flux':
        raise_ValueError(f"Unexpected TTYPE2: {skycalc_header['TTYPE2']}")

    wave = skycalc_table['lam']
    flux = skycalc_table['flux']
    trans = skycalc_table['trans']

    # plot radiance
    fig, ax = plt.subplots()
    ax.plot(wave, flux, '-', linewidth=1)
    cwave_unit = skycalc_header['TUNIT1']
    cflux_unit = skycalc_header['TUNIT2']
    ax.set_xlabel(f'Vacuum Wavelength ({cwave_unit})')
    ax.set_ylabel(f'{cflux_unit}')
    ax.set_title('SKYCALC prediction')
    plt.tight_layout()
    plt.show()

    # plot transmission
    fig, ax = plt.subplots()
    ax.plot(wave, trans, '-', linewidth=1)
    ax.set_xlabel(f'Vacuum Wavelength ({cwave_unit})')
    ax.set_ylabel('Transmission fraction')
    ax.set_title('SKYCALC prediction')
    plt.tight_layout()
    plt.show()


def load_atmosphere_transmission_curve(atmosphere_transmission, wmin, wmax, wv_cunit1, faux_dict, verbose):
    """Load atmosphere transmission curve.

    Parameters
    ----------
    atmosphere_transmission : str
        String indicating whether the atmosphere transmission of
        the atmosphere is applied or not. Two possible values are:
        - 'default': use default curve defined in 'faux_dict'
        - 'none': do not apply atmosphere transmission
    wmin : `~astroppy.units.Quantity`
        Minimum wavelength to be considered.
    wmax : `~astroppy.units.Quantity`
        Maximum wavelength to be considered.
    wv_cunit1 : `~astropy.units.core.Unit`
        Default wavelength unit to be employed in the wavelength scale.
    faux_dict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation
          x_ifu, y_ify, wavelength -> x_detector, y_detector
    verbose : bool
        If True, display additional information.

    Returns
    -------
    wave_transmission : `~astropy.units.Quantity`
        Wavelength column of the tabulated transmission curve.
    curve_transmission : `~astropy.units.Quantity`
        Transmission values for the wavelengths given in
        'wave_transmission'.

    """

    if atmosphere_transmission == "default":
        infile = faux_dict['skycalc']
        if verbose:
            print(f'\nLoading atmosphere transmission curve {os.path.basename(infile)}')
        with fits.open(infile) as hdul:
            skycalc_header = hdul[1].header
            skycalc_table = hdul[1].data
        if skycalc_header['TTYPE1'] != 'lam':
            raise_ValueError(f"Unexpected TTYPE1: {skycalc_header['TTYPE1']}")
        cwave_unit = skycalc_header['TUNIT1']
        wave_transmission = skycalc_table['lam'] * Unit(cwave_unit)
        curve_transmission = skycalc_table['trans']
        if wmin < np.min(wave_transmission) or wmax > np.max(wave_transmission):
            print(f'{wmin=} (simulated photons)')
            print(f'{wmax=} (simulated photons)')
            print(f'{np.min(wave_transmission.to(wv_cunit1))=} (transmission curve)')
            print(f'{np.max(wave_transmission.to(wv_cunit1))=} (transmission curve)')
            raise_ValueError('Wavelength range covered by the tabulated transmission curve is insufficient')
    elif atmosphere_transmission == "none":
        wave_transmission = None
        curve_transmission = None
        if verbose:
            print('Skipping application of the atmosphere transmission')
    else:
        wave_transmission = None   # avoid PyCharm warning (not aware of raise ValueError)
        curve_transmission = None  # avoid PyCharm warning (not aware of raise ValueError)
        raise_ValueError(f'Unexpected {atmosphere_transmission=}')

    return wave_transmission, curve_transmission


def simulate_constant_photlam(wmin, wmax, nphotons, rng):
    """Simulate spectrum with constant flux (in PHOTLAM units).

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
        raise_ValueError(f"Object 'wmin': {wmin} is not a Quantity instance")
    if not isinstance(wmax, u.Quantity):
        raise_ValueError(f"Object 'wmax': {wmax} is not a Quantity instance")
    if wmin.unit != wmax.unit:
        raise_ValueError(f"Different units used for 'wmin' and 'wmax': {wmin.unit}, {wmax.unit}.\n" +
                         "Employ the same unit to unambiguously define the output result.")

    simulated_wave = rng.uniform(low=wmin.value, high=wmax.value, size=nphotons)
    simulated_wave *= wmin.unit
    return simulated_wave


def simulate_delta_lines(line_wave, line_flux, nphotons, rng, wmin=None, wmax=None, plots=False, plot_title=None):
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
    plot_title : str or None
        Plot title. Used only when 'plots' is True.

    Returns
    -------
    simulated_wave : `~astropy.units.Quantity`
        Wavelength of simulated photons.

    """

    line_flux = np.asarray(line_flux)
    if len(line_wave) != len(line_flux):
        raise_ValueError(f"Incompatible array length: 'line_wave' ({len(line_wave)}), 'line_flux' ({len(line_flux)})")

    if np.any(line_flux < 0):
        raise_ValueError(f'Negative line fluxes cannot be handled')

    if not isinstance(line_wave, u.Quantity):
        raise_ValueError(f"Object 'line_wave': {line_wave} is not a Quantity instance")
    wave_unit = line_wave.unit
    if not wave_unit.is_equivalent(u.m):
        raise_ValueError(f"Unexpected unit for 'line_wave': {wave_unit}")

    # lower wavelength limit
    if wmin is not None:
        if not isinstance(wmin, u.Quantity):
            raise_ValueError(f"Object 'wmin':{wmin}  is not a Quantity instance")
        if not wmin.unit.is_equivalent(u.m):
            raise_ValueError(f"Unexpected unit for 'wmin': {wmin}")
        wmin = wmin.to(wave_unit)
        lower_index = np.searchsorted(line_wave.value, wmin.value, side='left')
    else:
        lower_index = 0

    # upper wavelength limit
    if wmax is not None:
        if not isinstance(wmax, u.Quantity):
            raise_ValueError(f"Object 'wmax': {wmax} is not a Quantity instance")
        if not wmax.unit.is_equivalent(u.m):
            raise_ValueError(f"Unexpected unit for 'wmax': {wmin}")
        wmax = wmax.to(wave_unit)
        upper_index = np.searchsorted(line_wave.value, wmax.value, side='right')
    else:
        upper_index = len(line_wave)

    if plots:
        fig, ax = plt.subplots()
        ax.stem(line_wave.value, line_flux, markerfmt=' ', basefmt=' ')
        if wmin is not None:
            ax.axvline(wmin.value, linestyle='--', color='gray')
        if wmax is not None:
            ax.axvline(wmax.value, linestyle='--', color='gray')
        ax.set_xlabel(f'Wavelength ({wave_unit})')
        ax.set_ylabel('Intensity (arbitrary units)')
        if plot_title is not None:
            ax.set_title(plot_title)
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
        if plot_title is not None:
            ax.set_title(plot_title)
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
        if plot_title is not None:
            ax.set_title(plot_title)
        plt.tight_layout()
        plt.show()

    return simulated_wave


def simulate_spectrum(wave, flux, flux_type, nphotons, rng, wmin, wmax, convolve_sigma_km_s,
                      nbins_histo, plots, plot_title, verbose):
    """Simulate spectrum defined by tabulated wave and flux data.

    Parameters
    ----------
    wave : `~astropy.units.Quantity`
        Numpy array (with astropy units) containing the tabulated
        wavelength.
    flux : array_like
        Array-like object containing the tabulated flux.
    flux_type : str
        Relative flux unit. Valid options are:
        - flam: proportional to erg s^-1 cm^-2 A^-1
        - photlam: proportional to photon s^-1 cm^-2 A^-1
    nphotons : int
        Number of photons to be simulated
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    wmin : `~astroppy.units.Quantity`
        Minimum wavelength to be considered.
    wmax : `~astroppy.units.Quantity`
        Maximum wavelength to be considered.
    convolve_sigma_km_s : `~astropy.units.Quantity`
        Gaussian broadening (sigma) in km/s to be applied.
    nbins_histo : int
        Number of bins for histogram plot.
    plots : bool
        If True, plot input and output results.
    plot_title : str or None
        Plot title. Used only when 'plots' is True.
    verbose : bool
        If True, display additional information.

    Returns
    -------
    simulated_wave : `~astropy.units.Quantity`
        Wavelength of simulated photons.
    """

    flux = np.asarray(flux)
    if len(wave) != len(flux):
        raise_ValueError(f"Incompatible array length: 'wave' ({len(wave)}), 'flux' ({len(flux)})")

    if np.any(flux < 0):
        raise_ValueError(f'Negative flux values cannot be handled')

    if flux_type.lower() not in ['flam', 'photlam']:
        raise_ValueError(f"Flux type: {flux_type} is not any of the valid values: 'flam', 'photlam'")

    if not isinstance(wave, u.Quantity):
        raise_ValueError(f"Object {wave=} is not a Quantity instance")
    wave_unit = wave.unit
    if not wave_unit.is_equivalent(u.m):
        raise_ValueError(f"Unexpected unit for 'wave': {wave_unit}")

    # lower wavelength limit
    if wmin is not None:
        if not isinstance(wmin, u.Quantity):
            raise_ValueError(f"Object {wmin=} is not a Quantity instance")
        if not wmin.unit.is_equivalent(u.m):
            raise_ValueError(f"Unexpected unit for 'wmin': {wmin}")
        wmin = wmin.to(wave_unit)
        lower_index = np.searchsorted(wave.value, wmin.value, side='left')
    else:
        lower_index = 0

    # upper wavelength limit
    if wmax is not None:
        if not isinstance(wmax, u.Quantity):
            raise_ValueError(f"Object {wmax=} is not a Quantity instance")
        if not wmax.unit.is_equivalent(u.m):
            raise_ValueError(f"Unexpected unit for 'wmax': {wmin}")
        wmax = wmax.to(wave_unit)
        upper_index = np.searchsorted(wave.value, wmax.value, side='right')
    else:
        upper_index = len(wave)

    if lower_index == upper_index:
        if plot_title is not None:
            print(f'Working with data from: {plot_title}')
        print(f'Tabulated wavelength range: {wave[0]} - {wave[-1]}')
        print(f'Requested wavelength range: {wmin} - {wmax}')
        raise_ValueError('Wavelength ranges without intersection')

    if not isinstance(convolve_sigma_km_s, u.Quantity):
        raise_ValueError(f'Object {convolve_sigma_km_s=} is not a Quantity instance')
    if convolve_sigma_km_s.unit != u.km / u.s:
        raise_ValueError(f'Unexpected unit for {convolve_sigma_km_s}')
    if convolve_sigma_km_s.value < 0:
        raise_ValueError(f'Unexpected negative value for {convolve_sigma_km_s}')

    if plots:
        fig, ax = plt.subplots()
        ax.plot(wave.value, flux, '-')
        if wmin is not None:
            ax.axvline(wmin.value, linestyle='--', color='gray')
        if wmax is not None:
            ax.axvline(wmax.value, linestyle='--', color='gray')
        ax.set_xlabel(f'Wavelength ({wave_unit})')
        ax.set_ylabel('Flux (arbitrary units)')
        if plot_title is not None:
            ax.set_title(plot_title)
        plt.tight_layout()
        plt.show()

    wave = wave[lower_index:upper_index]
    flux = flux[lower_index:upper_index]

    # convert FLAM to PHOTLAM
    if flux_type.lower() == 'flam':
        if verbose:
            print('Converting FLAM to PHOTLAM')
        flux_conversion = wave.to(u.m) / (constants.h * constants.c)
        flux *= flux_conversion.value

    wmin_eff = wave[0]
    wmax_eff = wave[-1]

    # normalized cumulative area
    # (area under the polygons defined by the tabulated data)
    cumulative_area = np.concatenate((
        [0],
        np.cumsum((flux[:-1] + flux[1:])/2 * (wave[1:] - wave[:-1]))
    ))
    normalized_cumulative_area = cumulative_area / cumulative_area[-1]

    if plots:
        fig, ax = plt.subplots()
        ax.plot(wave.value, normalized_cumulative_area, '.')
        ax.axvline(wmin_eff.value, linestyle='--', color='gray')
        ax.axvline(wmax_eff.value, linestyle='--', color='gray')
        ax.set_xlabel(f'Wavelength ({wave_unit})')
        ax.set_ylabel('Normalized cumulative area')
        if plot_title is not None:
            ax.set_title(plot_title)
        plt.tight_layout()
        plt.show()

    # samples following a uniform distribution
    unisamples = rng.uniform(low=0, high=1, size=nphotons)
    simulated_wave = np.interp(x=unisamples, xp=normalized_cumulative_area, fp=wave.value)

    # apply Gaussian broadening
    if convolve_sigma_km_s.value > 0:
        if verbose:
            print(f'Applying {convolve_sigma_km_s=}')
        sigma_wave = convolve_sigma_km_s / constants.c.to(u.km / u.s) * simulated_wave
        simulated_wave = rng.normal(loc=simulated_wave, scale=sigma_wave)

    # add units
    simulated_wave *= wave_unit

    if plots:
        fig, ax = plt.subplots()
        hist_sim, bin_edges_sim = np.histogram(simulated_wave.value, bins=nbins_histo)
        xhist_sim = (bin_edges_sim[:-1] + bin_edges_sim[1:]) / 2
        fscale = np.median(hist_sim / np.interp(x=xhist_sim, xp=wave.value, fp=flux))
        ax.plot(wave.value, flux*fscale, 'k-', linewidth=1, label='rescaled input spectrum')
        hist_dum = np.diff(np.interp(x=bin_edges_sim, xp=wave.value, fp=normalized_cumulative_area)) * nphotons
        ax.plot(xhist_sim, hist_dum, '-', linewidth=3, label='binned input spectrum')
        ax.plot(xhist_sim, hist_sim, '-', linewidth=1, label='binned simulated spectrum')
        ax.axvline(wmin_eff.value, linestyle='--', color='gray')
        ax.axvline(wmax_eff.value, linestyle='--', color='gray')
        ax.set_xlabel(f'Wavelength ({wave_unit})')
        ax.set_ylabel('Number of simulated photons')
        if plot_title is not None:
            ax.set_title(plot_title)
        ax.legend()
        plt.tight_layout()
        plt.show()

    return simulated_wave


def fapply_atmosphere_transmission(simulated_wave, wave_transmission, curve_transmission, rng,
                                   plots=False, verbose=False):
    """Apply atmosphere transmission.

    The input wavelength of each photon is converted into -1
    if the photon is absorbed. These photons are discarded later
    when the code removes those outside [wmin, vmax].

    Parameters
    ----------
    simulated_wave : `~astropy.units.Quantity`
        Array containing the simulated wavelengths. If the photon is
        absorbed, the wavelength is changed to -1. Note that this
        input array is also the output of this function.
    wave_transmission : `~astropy.units.Quantity`
        Wavelength column of the tabulated transmission curve.
    curve_transmission : `~astropy.units.Quantity`
        Transmission values for the wavelengths given in
        'wave_transmission'.
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    plots : bool
        If True, plot input and output results.
    verbose : bool
        If True, display additional information.

    """

    wave_unit = simulated_wave.unit

    # compute transmission at the wavelengths of the simulated photons
    transmission_values = np.interp(
        x=simulated_wave.value,
        xp=wave_transmission.to(wave_unit).value,
        fp=curve_transmission
    )

    if plots:
        fig, ax = plt.subplots()
        ax.plot(wave_transmission.to(wave_unit), curve_transmission, '-', label='SKYCALC curve')
        ax.plot(simulated_wave, transmission_values, ',', alpha=0.5, label='interpolated values')
        ax.set_xlabel(f'Wavelength ({wave_unit})')
        ax.set_ylabel('Transmission fraction')
        ax.legend(loc=3)  # loc="best" can be slow with large amounts of data
        plt.tight_layout()
        plt.show()

    # generate random values in the interval [0, 1] and discard photons whose
    # transmission value is lower than the random value
    nphotons = len(simulated_wave)
    survival_probability = rng.uniform(low=0, high=1, size=nphotons)
    iremove = np.argwhere(transmission_values < survival_probability)
    simulated_wave[iremove] = -1 * wave_unit

    if verbose:
        print('Applying atmosphere transmission:')
        print(f'- initial number of photons: {nphotons}')
        textwidth_nphotons_number = len(str(nphotons))
        percentage = np.round(100 * len(iremove) / nphotons, 2)
        print(f'- number of photons removed: {len(iremove):>{textwidth_nphotons_number}}  ({percentage}%)')


def simulate_image2d_from_fitsfile(
        infile,
        diagonal_fov_arcsec,
        plate_scale_x,
        plate_scale_y,
        nphotons,
        rng,
        background_to_subtract=None,
        image_threshold=0.0,
        plots=False,
        verbose=False
):
    """Simulate photons mimicking a 2D image from FITS file.

    Parameters
    ----------
    infile : str
        Input file containing the FITS image to be simulated.
    diagonal_fov_arcsec : `~astropy.units.Quantity`
        Desired field of View (arcsec) corresponding to the diagonal
        of the FITS image.
    plate_scale_x : `~astropy.units.Quantity`
        Plate scale of the IFU in the X direction.
    plate_scale_y : `~astropy.units.Quantity`
        Plate scale of the IFU in the Y direction.
    nphotons : int
        Number of photons to be simulated.
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    background_to_subtract : str or None
        If not None, this parameters indicates how to computed
        the background to be subtracted.
    plots : bool
        If True, plot intermediate results.
    verbose : bool
        If True, display additional information.

    Returns
    -------
    simulated_x_ifu : `~numpy.ndarray`
        Array of simulated photon X coordinates in the IFU.
    simulated_y_ify : `~numpy.ndarray`
        Array of simulated photon Y coordinates in the IFU.

    """

    # read input FITS file
    if verbose:
        print(f'Reading {infile=}')
    with fits.open(infile) as hdul:
        image2d_ini = hdul[0].data
    image2d_reference = image2d_ini.astype(float)
    naxis2, naxis1 = image2d_reference.shape
    npixels = naxis1 * naxis2

    # subtract background
    if background_to_subtract is not None:
        if background_to_subtract == 'mode':
            nbins = int(np.sqrt(npixels) + 0.5)
            h, bin_edges = np.histogram(image2d_reference.flatten(), bins=nbins)
            imax = np.argmax(h)
            skylevel = (bin_edges[imax] + bin_edges[imax+1]) / 2
            if verbose:
                print(f'Subtracting {skylevel=} (image mode)')
        elif background_to_subtract == 'median':
            skylevel = np.median(image2d_reference.flatten())
            if verbose:
                print(f'Subtracting {skylevel=} (image median)')
        else:
            skylevel = None   # avoid PyCharm warning (not aware of raise ValueError)
            raise_ValueError(f'Invalid {background_to_subtract=}')
        image2d_reference -= skylevel
    else:
        if verbose:
            print('Skipping background subtraction')

    # impose image threshold
    if verbose:
        print(f'Applying {image_threshold=}')
    image2d_reference[image2d_reference <= image_threshold] = 0
    if np.min(image2d_reference) < 0.0:
        raise_ValueError(f'{np.min(image2d_reference)=} must be >= 0.0')

    # flatten image to be simulated
    image1d = image2d_reference.flatten()
    # compute normalized cumulative area
    xpixel = 1 + np.arange(npixels)
    cumulative_area = np.concatenate((
        [0],
        np.cumsum((image1d[:-1] + image1d[1:]) / 2 * (xpixel[1:] - xpixel[:-1]))
    ))
    normalized_cumulative_area = cumulative_area / cumulative_area[-1]
    if plots:
        fig, ax = plt.subplots()
        ax.plot(xpixel, normalized_cumulative_area, '.')
        ax.set_xlabel(f'xpixel')
        ax.set_ylabel('Normalized cumulative area')
        ax.set_title(os.path.basename(infile))
        plt.tight_layout()
        plt.show()
    # invert normalized cumulative area using random uniform samples
    unisamples = rng.uniform(low=0, high=1, size=nphotons)
    simulated_pixel = np.interp(x=unisamples, xp=normalized_cumulative_area, fp=xpixel)
    # compute histogram of 1D data
    bins_pixel = 0.5 + np.arange(npixels + 1)
    int_simulated_pixel, bin_edges = np.histogram(simulated_pixel, bins=bins_pixel)
    # reshape 1D into 2D image
    image2d_simulated = int_simulated_pixel.reshape((naxis2, naxis1))
    # scale factors to insert simulated image in requested field of view
    plate_scale = diagonal_fov_arcsec / (np.sqrt(naxis1**2 + naxis2**2) * u.pix)
    factor_x = abs((plate_scale / plate_scale_x.to(u.arcsec / u.pix)).value)
    factor_y = abs((plate_scale / plate_scale_y.to(u.arcsec / u.pix)).value)
    # redistribute photons in each pixel of the simulated image using a
    # random distribution within the considered pixel
    jcenter = naxis1 / 2
    icenter = naxis2 / 2
    simulated_x_ifu = []
    simulated_y_ifu = []
    for i, j in np.ndindex(naxis2, naxis1):
        nphotons_in_pixel = image2d_simulated[i, j]
        if nphotons_in_pixel > 0:
            jmin = j - jcenter - 0.5
            jmax = j - jcenter + 0.5
            simulated_x_ifu += (rng.uniform(low=jmin, high=jmax, size=nphotons_in_pixel) * factor_x).tolist()
            imin = i - icenter - 0.5
            imax = i - icenter + 0.5
            simulated_y_ifu += (rng.uniform(low=imin, high=imax, size=nphotons_in_pixel) * factor_y).tolist()

    # return result
    return np.array(simulated_x_ifu), np.array(simulated_y_ifu)


def generate_image2d_method0_ifu(
        wcs3d,
        noversampling_whitelight,
        simulated_x_ifu_all,
        simulated_y_ifu_all,
        prefix_intermediate_fits,
        instname,
        subtitle,
        scene,
        plots
):
    """Compute image2d IFU (white image), method0

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    noversampling_whitelight : int
        Oversampling factor (integer number) to generate the method0
        white image.
    simulated_x_ifu_all : `~astropy.units.Quantity`
        Simulated X coordinates of the photons in the IFU.
    simulated_y_ifu_all : `~astropy.units.Quantity`
        Simulated Y coordinates of the photons in the IFU.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    instname : str or None
        Instrument name.
    subtitle : str or None
        Plot subtitle.
    scene : str
        YAML scene file name.
    plots : bool
        If True, plot intermediate results.

    """

    # select the 2D spatial info of the 3D WCS
    wcs2d = wcs3d.sub(axes=[1, 2])

    naxis1_ifu_oversampled = Quantity(
        value=wcs2d.array_shape[1] * noversampling_whitelight,
        unit=u.pix,
        dtype=int
    )
    naxis2_ifu_oversampled = Quantity(
        value=wcs2d.array_shape[0] * noversampling_whitelight,
        unit=u.pix,
        dtype=int
    )

    bins_x_ifu_oversampled = (0.5 + np.arange(naxis1_ifu_oversampled.value + 1)) * u.pix
    bins_y_ifu_oversampled = (0.5 + np.arange(naxis2_ifu_oversampled.value + 1)) * u.pix

    crpix1_orig, crpix2_orig = wcs2d.wcs.crpix
    crpix1_oversampled = (naxis1_ifu_oversampled.value + 1) / 2
    crpix2_oversampled = (naxis2_ifu_oversampled.value + 1) / 2

    wcs2d.wcs.crpix = crpix1_oversampled, crpix2_oversampled

    # (important: reverse X <-> Y)
    image2d_method0_ifu, xedges, yedges = np.histogram2d(
        x=(simulated_y_ifu_all.value - crpix2_orig) * noversampling_whitelight + crpix2_oversampled,
        y=(simulated_x_ifu_all.value - crpix1_orig) * noversampling_whitelight + crpix1_oversampled,
        bins=(bins_y_ifu_oversampled.value, bins_x_ifu_oversampled.value)
    )

    wcs2d.wcs.cd /= noversampling_whitelight

    # save FITS file
    if len(prefix_intermediate_fits) > 0:
        hdu = fits.PrimaryHDU(image2d_method0_ifu.astype(np.float32))
        hdu.header.extend(wcs2d.to_header(), update=True)
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_ifu_white2D_method0_os{noversampling_whitelight:d}.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(f'{outfile}', overwrite='yes')

    # display result
    if plots:
        fig, ax = plt.subplots(figsize=(6.4, 6.4))
        img = ax.imshow(image2d_method0_ifu, origin='lower', interpolation='None')
        ax.set_xlabel('X axis (array index)  [parallel to the slices]')
        ax.set_ylabel('Y axis (array index)  [perpendicular to the slices]')
        if instname is not None:
            title = f'{instname} '
        else:
            title = ''
        title += f'IFU image, method0 (oversampling={noversampling_whitelight})'
        if subtitle is not None:
            title += f'\n{subtitle}'
        title += f'\nscene: {scene}'
        ax.set_title(title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(img, cax=cax, label='Number of photons')
        plt.tight_layout()
        plt.show()


def generate_image3d_method0_ifu(
        wcs3d,
        simulated_x_ifu_all,
        simulated_y_ifu_all,
        simulated_wave_all,
        bins_x_ifu,
        bins_y_ifu,
        bins_wave,
        prefix_intermediate_fits
):
    """Compute 3D image3 IFU, method 0

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    simulated_x_ifu_all : `~astropy.units.Quantity`
        Simulated X coordinates of the photons in the IFU.
    simulated_y_ifu_all : `~astropy.units.Quantity`
        Simulated Y coordinates of the photons in the IFU.
    simulated_wave_all : `~astropy.units.Quantity`
        Simulated wavelengths of the photons in the IFU.
    bins_x_ifu : `~numpy.ndarray`
        Bin edges in the naxis1_ifu direction
        (along the slice).
    bins_y_ifu : `~numpy.ndarray`
        Bin edges in the naxis2_ifu direction
        (perpendicular to the slice).
    bins_wave : `~numpy.ndarray`
        Bin edges in the wavelength direction.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.

    """

    # generate image
    image3d_method0_ifu, edges = np.histogramdd(
        sample=(simulated_wave_all.value, simulated_y_ifu_all.value, simulated_x_ifu_all.value),
        bins=(bins_wave.value, bins_y_ifu.value, bins_x_ifu.value)
    )

    # save FITS file
    if len(prefix_intermediate_fits) > 0:
        hdu = fits.PrimaryHDU(image3d_method0_ifu.astype(np.float32))
        hdu.header.extend(wcs3d.to_header(), update=True)
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_ifu_3D_method0.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(f'{outfile}', overwrite='yes')


def update_image2d_rss_detector_method0(
        islice,
        simulated_x_ifu_all,
        simulated_y_ifu_all,
        simulated_wave_all,
        naxis1_ifu,
        bins_x_ifu,
        bins_wave,
        bins_x_detector,
        bins_y_detector,
        wv_cdelt1,
        extra_degradation_spectral_direction,
        dict_ifu2detector,
        image2d_rss_method0,
        image2d_detector_method0
):
    """Update the two 2D images: RSS and detector.

    The function updates the following 2D arrays:
    - image2d_rss_method0,
    - image2d_detector_method0
    with the photons observed through the slice 'islice'.

    Note that both arrays are generated simultaneously in
    order to make use of the same value of
    'extra_degradation_spectral_direction'.

    This function can be executed in parallel.

    Parameters
    ----------
    islice : int
        Slice number.
    simulated_x_ifu_all : `~astropy.units.Quantity`
        Simulated X coordinates of the photons in the IFU.
    simulated_y_ifu_all : `~astropy.units.Quantity`
        Simulated Y coordinates of the photons in the IFU.
    simulated_wave_all : `~astropy.units.Quantity`
        Simulated wavelengths of the photons in the IFU.
    naxis1_ifu : `~astropy.units.Quantity`
        IFU NAXIS1 (along the slice)
    bins_x_ifu : `~numpy.ndarray`
        Bin edges in the naxis1_ifu direction
        (along the slice).
    bins_wave : `~numpy.ndarray`
        Bin edges in the wavelength direction.
    bins_x_detector : `~numpy.ndarray`
        Bin edges in the naxis1_detector direction
        (spectral direction).
    bins_y_detector : `~numpy.ndarray`
        Bin edges in the naxis2_detector direction
        (slices direction).
    wv_cdelt1 : `~astropy.units.Quantity`
        CDELT1 value along the spectral direction.
    extra_degradation_spectral_direction : `~astropy.units.Quantity`
        Additional degradation in the spectral direction, in units of
        the detector pixels, for each simulated photon.
    dict_ifu2detector : dict
        A Python dictionary containing the 2D polynomials that allow
        to transform (X, Y, wavelength) coordinates in the IFU focal
        plane to (X, Y) coordinates in the detector.
    image2d_rss_method0 : `~numpy.ndarray`
        2D array containing the RSS image. This array is
        updated by this function.
    image2d_detector_method0 : `~numpy.ndarray`
        2D array containing the detector image. This array is
        updated by this function.

    """

    # determine photons that pass through the considered slice
    y_ifu_expected = 1.5 + 2 * islice
    condition = np.abs(simulated_y_ifu_all.value - y_ifu_expected) < 1
    iok = np.where(condition)[0]
    nphotons_slice = len(iok)

    if nphotons_slice > 0:
        # -------------------------------------------------
        # 1) spectroscopic 2D image with continguous slices
        # -------------------------------------------------
        h, xedges, yedges = np.histogram2d(
            x=simulated_x_ifu_all.value[iok],
            y=simulated_wave_all.value[iok] +
              (simulated_y_ifu_all.value[iok] - y_ifu_expected) * wv_cdelt1.value +
              extra_degradation_spectral_direction.value[iok] * wv_cdelt1.value,
            bins=(bins_x_ifu.value, bins_wave.value)
        )
        j1 = islice * naxis1_ifu.value
        j2 = j1 + naxis1_ifu.value
        image2d_rss_method0[j1:j2, :] += h

        # -----------------------------------------
        # 2) spectroscopic 2D image in the detector
        # -----------------------------------------
        # use models to predict location in Hawaii detector
        # important: reverse here X <-> Y
        wavelength_unit = Unit(dict_ifu2detector['wavelength-unit'])
        dumdict = dict_ifu2detector['contents'][islice]
        order = dumdict['order']
        aij = np.array(dumdict['aij'])
        bij = np.array(dumdict['bij'])
        y_hawaii, x_hawaii = fmap(
            order=order,
            aij=aij,
            bij=bij,
            x=simulated_x_ifu_all.value[iok],
            # important: use the wavelength unit employed to determine
            # the polynomial transformation
            y=simulated_wave_all.to(wavelength_unit).value[iok]
        )
        # disperse photons along the spectral direction according to their
        # location within the slice in the vertical direction
        x_hawaii += simulated_y_ifu_all[iok].value - y_ifu_expected
        # include additional degradation in spectral resolution
        x_hawaii += extra_degradation_spectral_direction.value[iok]
        # compute 2D histogram
        # important: reverse X <-> Y
        h, xedges, yedges = np.histogram2d(
            x=y_hawaii,
            y=x_hawaii,
            bins=(bins_y_detector, bins_x_detector)
        )
        image2d_detector_method0 += h


def update_image2d_rss_method1(
        islice,
        image2d_detector_method0,
        dict_ifu2detector,
        naxis1_detector,
        naxis1_ifu,
        wv_crpix1, wv_crval1, wv_cdelt1,
        image2d_rss_method1,
        debug=False
):
    """Update the RSS image from the detector image.

    The function updates the following 2D array:
    - image2d_rss_method1,
    with the data of the slice 'islice' in the detector image.

    This function can be executed in parallel.

    Parameters
    ----------
    islice : int
        Slice number.
    image2d_detector_method0 : `~numpy.ndarray`
        2D array containing the detector image.
    dict_ifu2detector : dict
        A Python dictionary containing the 2D polynomials that allow
        to transform (X, Y, wavelength) coordinates in the IFU focal
        plane to (X, Y) coordinates in the detector.
    naxis1_detector : `~astropy.units.Quantity`
        Detector NAXIS1 (along the spectral direction).
    naxis1_ifu : `~astropy.units.Quantity`
        IFU NAXIS1 (along the slice).
    wv_crpix1 : `~astropy.units.Quantity`
        CRPIX1 value along the spectral direction.
    wv_crval1 : `~astropy.units.Quantity`
        CRVAL1 value along the spectral direction.
    wv_cdelt1 : `~astropy.units.Quantity`
        CDELT1 value along the spectral direction.
    image2d_rss_method1 : `~numpy.ndarray`
        2D array containing the RSS image. This array is
        updated by this function.
    debug : bool
        If True, show debugging information/plots.

    """

    slice_id = islice + 1

    # minimum and maximum pixel X coordinate defining the IFU focal plane
    min_x_ifu = 0.5 * u.pix
    max_x_ifu = naxis1_ifu + 0.5 * u.pix

    # determine upper and lower frontiers of each slice in the detector
    x_ifu_lower = np.repeat([min_x_ifu.value], naxis1_detector.value)
    x_ifu_upper = np.repeat([max_x_ifu.value], naxis1_detector.value)

    # wavelength values at each pixel in the spectral direction of the detector
    wavelength = wv_crval1 + ((np.arange(naxis1_detector.value) + 1) * u.pix - wv_crpix1) * wv_cdelt1

    # minimum and maximum wavelengths to be considered
    wmin = wv_crval1 + (0.5 * u.pix - wv_crpix1) * wv_cdelt1
    wmax = wv_crval1 + (naxis1_detector + 0.5 * u.pix - wv_crpix1) * wv_cdelt1

    # use model to predict location in detector
    # important: reverse here X <-> Y
    wavelength_unit = Unit(dict_ifu2detector['wavelength-unit'])
    dumdict = dict_ifu2detector['contents'][islice]
    order = dumdict['order']
    aij = np.array(dumdict['aij'])
    bij = np.array(dumdict['bij'])

    y_detector_lower_index, x_detector_lower_index = fmap(
        order=order,
        aij=aij,
        bij=bij,
        x=x_ifu_lower,
        y=wavelength.to(wavelength_unit).value
    )
    # subtract 1 to work with array indices
    x_detector_lower_index -= 1
    y_detector_lower_index -= 1

    y_detector_upper_index, x_detector_upper_index = fmap(
        order=order,
        aij=aij,
        bij=bij,
        x=x_ifu_upper,
        y=wavelength.to(wavelength_unit).value
    )
    # subtract 1 to work with array indices
    x_detector_upper_index -= 1
    y_detector_upper_index -= 1

    if debug:
        debugplot = 1
    else:
        debugplot = 0
    poly_lower_index, residuals = polfit_residuals(
        x=x_detector_lower_index,
        y=y_detector_lower_index,
        deg=order,
        xlabel='x_detector_lower_index',
        ylabel='y_detector_lower_index',
        title=f'slice_id #{slice_id}',
        debugplot=debugplot
    )
    if debug:
        plt.tight_layout()
        plt.show()
    poly_upper_index, residuals = polfit_residuals(
        x=x_detector_upper_index,
        y=y_detector_upper_index,
        deg=order,
        xlabel='x_detector_upper_index',
        ylabel='y_detector_upper_index',
        title=f'slice_id #{slice_id}',
        debugplot=debugplot
    )
    if debug:
        plt.tight_layout()
        plt.show()

    # full image containing only the slice data (zero elsewhere)
    image2d_detector_slice = np.zeros_like(image2d_detector_method0)
    xdum = np.arange(naxis1_detector.value)
    ypoly_lower_index = (poly_lower_index(xdum) + 0.5).astype(int)
    ypoly_upper_index = (poly_upper_index(xdum) + 0.5).astype(int)
    for j in range(naxis1_detector.value):
        i1 = ypoly_lower_index[j]
        i2 = ypoly_upper_index[j]
        image2d_detector_slice[i1:(i2 + 1), j] = image2d_detector_method0[i1:(i2 + 1), j]

    if debug:
        xmin = np.min(np.concatenate((x_detector_lower_index, x_detector_upper_index)))
        xmax = np.max(np.concatenate((x_detector_lower_index, x_detector_upper_index)))
        ymin = np.min(np.concatenate((y_detector_lower_index, y_detector_upper_index)))
        ymax = np.max(np.concatenate((y_detector_lower_index, y_detector_upper_index)))
        print(f'{xmin=}, {xmax=}, {ymin=}, {ymax=}')
        dy = ymax - ymin
        yminplot = ymin - dy / 5
        ymaxplot = ymax + dy / 5
        fig, axarr = plt.subplots(nrows=2, ncols=1, figsize=(6.4*2, 4.8*2))
        vmin, vmax = ZScaleInterval().get_limits(image2d_detector_method0)
        for iplot in range(2):
            ax = axarr[iplot]
            if iplot == 0:
                ax.imshow(image2d_detector_method0, vmin=vmin, vmax=vmax, aspect='auto')
            else:
                ax.imshow(image2d_detector_slice, vmin=vmin, vmax=vmax, aspect='auto')
            ax.plot(x_detector_lower_index, y_detector_lower_index, 'C1--')
            ax.plot(x_detector_upper_index, y_detector_upper_index, 'C1--')
            ax.plot(xdum, ypoly_lower_index, 'w:')
            ax.plot(xdum, ypoly_upper_index, 'w:')
            ax.set_ylim(yminplot, ymaxplot)
        plt.tight_layout()
        plt.show()

    # mathematical transformation for the considered slice
    i1_rss_index = islice * naxis1_ifu.value
    i2_rss_index = i1_rss_index + naxis1_ifu.value
    if debug:
        print(f'{i1_rss_index=}, {i2_rss_index=}, {i2_rss_index - i1_rss_index=}')
    # generate a grid to compute the 2D transformation
    nx_grid_rss = 20
    ny_grid_rss = 20
    # points along the slice in the IFU focal pline
    x_ifu_grid = np.tile(np.linspace(1, naxis1_ifu.value, num=ny_grid_rss), nx_grid_rss)
    wavelength_grid = np.repeat(np.linspace(wmin, wmax, num=nx_grid_rss), ny_grid_rss)
    # pixel in the RSS image
    wavelength_grid_pixel_rss = (wavelength_grid - wv_crval1) / wv_cdelt1 + wv_crpix1
    if debug:
        fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8))
        ax.plot(wavelength_grid_pixel_rss.value, x_ifu_grid + i1_rss_index, 'r.')
        ax.set_xlabel('RSS pixel in wavelength direction')
        ax.set_ylabel('RSS pixel in spatial direction')
        plt.tight_layout()
        plt.show()
    # project the previous points in the detector
    y_hawaii_grid_index, x_hawaii_grid_index = fmap(
        order=order,
        aij=aij,
        bij=bij,
        x=x_ifu_grid,
        y=wavelength_grid.to(wavelength_unit).value   # ignore PyCharm warning here
    )
    x_hawaii_grid_index -= 1
    y_hawaii_grid_index -= 1

    y_hawaii_grid_index, x_hawaii_grid_index = fmap(
        order=order,
        aij=aij,
        bij=bij,
        x=x_ifu_grid,
        y=wavelength_grid.to(wavelength_unit).value  # ignore PyCharm warning here
    )
    x_hawaii_grid_index -= 1
    y_hawaii_grid_index -= 1
    if debug:
        fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8))
        ax.imshow(image2d_detector_slice, vmin=vmin, vmax=vmax, aspect='auto')  # ignore PyCharm warning here
        ax.plot(x_hawaii_grid_index, y_hawaii_grid_index, 'r.')
        ax.set_ylim(yminplot, ymaxplot)  # ignore PyCharm warning here
        plt.tight_layout()
        plt.show()
    # compute distortion transformation
    aij_resample, bij_resample = compute_distortion(
        x_orig=x_hawaii_grid_index + 1,
        y_orig=y_hawaii_grid_index + 1,
        x_rect=wavelength_grid_pixel_rss.value,
        y_rect=x_ifu_grid,
        order=order,
        debugplot=0
    )

    # rectify image
    image2d_slice_rss = rectify2d(
        image2d=image2d_detector_slice,
        aij=aij_resample,
        bij=bij_resample,
        resampling=2,  # 2: flux preserving interpolation
        naxis2out=naxis1_ifu.value
    )
    if debug:
        fig, ax = plt.subplots(figsize=(6.4 * 2, 4.8))
        ax.imshow(image2d_slice_rss, vmin=vmin, vmax=vmax, aspect='auto')
        plt.tight_layout()
        plt.show()

    # insert result in final image
    image2d_rss_method1[i1_rss_index:i2_rss_index, :] = image2d_slice_rss[:, :]


def save_image2d_detector_method0(
        wcs3d,
        image2d_detector_method0,
        prefix_intermediate_fits
):
    """Save the two 2D images: RSS and detector.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    image2d_detector_method0 : `~numpy.ndarray`
        2D array containing the detector image.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    """

    if len(prefix_intermediate_fits) > 0:
        # --------------------------------------
        # spectroscopic 2D image in the detector
        # --------------------------------------
        hdu = fits.PrimaryHDU(image2d_detector_method0.astype(np.float32))
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_detector_2D_method0.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(outfile, overwrite='yes')


def save_image2d_rss(
        wcs3d,
        image2d_rss,
        method,
        prefix_intermediate_fits
):
    """Save the RSS image.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    image2d_rss : `~numpy.ndarray`
        2D array containing the RSS image.
    method : int
        Integer indicating the method: 0 or 1.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    """

    if len(prefix_intermediate_fits) > 0:
        # ------------------------------------------------
        # 1) spectroscopic 2D image with contiguous slices
        # ------------------------------------------------
        # ToDo: compute properly the parameters corresponding to the spatial axis
        # Note that using: wcs2d = wcs3d.sub(axes=[0, 1])
        # selecting the 1D spectral and one of the 1D spatial info of the 3D WCS
        # does not work:
        # "astropy.wcs._wcs.InconsistentAxisTypesError: ERROR 4 in wcs_types()
        #  Unmatched celestial axes."
        # For that reason we try a different approach:
        wv_cunit1, wv_crpix1, wv_crval1, wv_cdelt1 = get_wvparam_from_wcs3d(wcs3d)
        wcs2d = wcs.WCS(naxis=2)
        wcs2d.wcs.crpix = [wv_crpix1.value, 1]  # reference pixel coordinate
        wcs2d.wcs.crval = [wv_crval1.value, 0]  # world coordinate at reference pixel
        wcs2d.wcs.cdelt = [wv_cdelt1.value, 1]
        wcs2d.wcs.ctype = ["WAVE", ""]   # ToDo: fix this
        wcs2d.wcs.cunit = [wv_cunit1, u.pix]
        hdu = fits.PrimaryHDU(image2d_rss.astype(np.float32))
        hdu.header.extend(wcs2d.to_header(), update=True)
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_rss_2D_method{method}.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(outfile, overwrite='yes')


def set_wavelength_unit_and_range(scene_fname, scene_block, wmin, wmax, verbose):
    """Set the wavelength unit and range for a scene block.

    Parameters
    ----------
    scene_fname : str
        YAML scene file name.
    scene_block : dict
        Dictonary storing the scene block.
    wmin : `~astropy.units.Quantity`
        Minimum wavelength covered by the detector.
    wmax : `~astropy.units.Quantity`
        Maximum wavelength covered by the detector.
    verbose : bool
       If True, display additional information.

    Returns
    -------
    wave_unit : `~astropy.units.core.Unit`
        Wavelength unit to be used in the scene block.
    wave_min : `~astropy.units.Quantity`
        Minimum wavelength to be used in the scene block.
    wave_max : `~astropy.units.Quantity`
        Maximum wavelength to be used in the scene block.

    """

    expected_keys_in_spectrum = {'type'}

    spectrum_keys = set(scene_block['spectrum'].keys())
    if not expected_keys_in_spectrum.issubset(spectrum_keys):
        print(ctext(f'ERROR while processing: {scene_fname}', fg='red'))
        print(ctext('expected keys..: ', fg='blue') + f'{expected_keys_in_spectrum}')
        print(ctext('keys found.....: ', fg='blue') + f'{spectrum_keys}')
        list_unexpected_keys = list(spectrum_keys.difference(expected_keys_in_spectrum))
        if len(list_unexpected_keys) > 0:
            print(ctext('unexpected keys: ', fg='red') + f'{list_unexpected_keys}')
        list_missing_keys = list(expected_keys_in_spectrum.difference(spectrum_keys))
        if len(list_missing_keys) > 0:
            print(ctext('missing keys:..: ', fg='red') + f'{list_missing_keys}')
        pp.pprint(scene_block)
        raise_ValueError(f'Invalid format in file: {scene_fname}')
    if 'wave_unit' in scene_block['spectrum']:
        wave_unit = scene_block['spectrum']['wave_unit']
    else:
        wave_unit = wmin.unit
        if verbose:
            print(ctext(f'Assuming wave_unit: {wave_unit}', faint=True))
    if wave_unit is None:  # useful for type="constant-flux"
        wave_min = wmin
        wave_max = wmax
    else:
        if 'wave_min' in scene_block['spectrum']:
            wave_min = scene_block['spectrum']['wave_min']
        else:
            if verbose:
                print(ctext('Assuming wave_min: null', faint=True))
            wave_min = None
        if wave_min is None:
            wave_min = wmin.to(wave_unit)
        else:
            wave_min *= Unit(wave_unit)
        if 'wave_max' in scene_block['spectrum']:
            wave_max = scene_block['spectrum']['wave_max']
        else:
            if verbose:
                print(ctext('Assuming wave_max: null', faint=True))
            wave_max = None
        if wave_max is None:
            wave_max = wmax.to(wave_unit)
        else:
            wave_max *= Unit(wave_unit)

        return wave_unit, wave_min, wave_max


def generate_spectrum_for_scene_blok(scene_fname, scene_block, faux_dict, wave_unit,
                                     wave_min, wave_max, nphotons,
                                     apply_atmosphere_transmission, wave_transmission, curve_transmission,
                                     rng, naxis1_detector,
                                     verbose, plots):
    """Generate photons for the scene block.

    Parameters
    ----------
    scene_fname : str
        YAML scene file name.
    scene_block : dict
        Dictonary storing a scene block.
    faux_dict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation
          x_ifu, y_ify, wavelength -> x_detector, y_detector
    wave_unit : `~astropy.units.core.Unit`
        Wavelength unit to be used in the scene block.
    wave_min : `~astropy.units.Quantity`
        Minimum wavelength to be used in the scene block.
    wave_max : `~astropy.units.Quantity`
        Maximum wavelength to be used in the scene block.
    nphotons : int
        Number of photons to be generated in the scene block.
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    apply_atmosphere_transmission : bool
        If True, apply atmosphere transmission to simulated photons.
    wave_transmission : `~astropy.units.Quantity`
        Wavelength column of the tabulated transmission curve.
    curve_transmission : `~astropy.units.Quantity`
        Transmission values for the wavelengths given in
        'wave_transmission'.
    naxis1_detector : `~astropy.units.Quantity`
        Detector NAXIS1, dispersion direction.
    verbose : bool
        If True, display additional information.
    plots : bool
        If True, plot intermediate results.

    Returns
    -------
    simulated_wave : '~numpy.ndarray'
        Array containint `nphotons` simulated photons with the
        spectrum requested in the scene block.

    """

    spectrum_type = scene_block['spectrum']['type']
    if spectrum_type == 'delta-lines':
        filename = scene_block['spectrum']['filename']
        wave_column = scene_block['spectrum']['wave_column'] - 1
        flux_column = scene_block['spectrum']['flux_column'] - 1
        if filename[0] == '@':
            # retrieve file name from dictionary of auxiliary
            # file names for the considered instrument
            filename = faux_dict[filename[1:]]
        catlines = np.genfromtxt(filename)
        line_wave = catlines[:, wave_column] * Unit(wave_unit)
        if not np.all(np.diff(line_wave.value) > 0):
            raise_ValueError(f'Wavelength array {line_wave=} is not sorted!')
        line_flux = catlines[:, flux_column]
        simulated_wave = simulate_delta_lines(
            line_wave=line_wave,
            line_flux=line_flux,
            nphotons=nphotons,
            rng=rng,
            wmin=wave_min,
            wmax=wave_max,
            plots=plots,
            plot_title=filename
        )
    elif spectrum_type == 'skycalc-radiance':
        faux_skycalc = faux_dict['skycalc']
        with fits.open(faux_skycalc) as hdul:
            skycalc_table = hdul[1].data
        wave = skycalc_table['lam'] * Unit(wave_unit)
        if not np.all(np.diff(wave.value) > 0):
            raise_ValueError(f'Wavelength array {wave=} is not sorted!')
        flux = skycalc_table['flux']
        flux_type = 'photlam'
        simulated_wave = simulate_spectrum(
            wave=wave,
            flux=flux,
            flux_type=flux_type,
            nphotons=nphotons,
            rng=rng,
            wmin=wave_min,
            wmax=wave_max,
            convolve_sigma_km_s=0 * u.km / u.s,
            nbins_histo=naxis1_detector.value,
            plots=plots,
            plot_title=os.path.basename(faux_skycalc),
            verbose=verbose
        )
    elif spectrum_type == 'tabulated-spectrum':
        filename = scene_block['spectrum']['filename']
        wave_column = scene_block['spectrum']['wave_column'] - 1
        flux_column = scene_block['spectrum']['flux_column'] - 1
        flux_type = scene_block['spectrum']['flux_type']
        if 'redshift' in scene_block['spectrum']:
            redshift = scene_block['spectrum']['redshift']
        else:
            if verbose:
                print(ctext('Assuming redshift: 0', faint=True))
            redshift = 0.0
        if 'convolve_sigma_km_s' in scene_block['spectrum']:
            convolve_sigma_km_s = scene_block['spectrum']['convolve_sigma_km_s']
        else:
            if verbose:
                print(ctext('Assuming convolve_sigma_km_s: 0', faint=True))
            convolve_sigma_km_s = 0.0
        convolve_sigma_km_s *= u.km / u.s
        # read data
        table_data = np.genfromtxt(filename)
        wave = table_data[:, wave_column] * (1 + redshift) * Unit(wave_unit)
        if not np.all(np.diff(wave.value) > 0):
            raise_ValueError(f'Wavelength array {wave=} is not sorted!')
        flux = table_data[:, flux_column]
        simulated_wave = simulate_spectrum(
            wave=wave,
            flux=flux,
            flux_type=flux_type,
            nphotons=nphotons,
            rng=rng,
            wmin=wave_min,
            wmax=wave_max,
            convolve_sigma_km_s=convolve_sigma_km_s,
            nbins_histo=naxis1_detector.value,
            plots=plots,
            plot_title=os.path.basename(filename),
            verbose=verbose
        )
    elif spectrum_type == 'constant-flux':
        simulated_wave = simulate_constant_photlam(
            wmin=wave_min,
            wmax=wave_max,
            nphotons=nphotons,
            rng=rng
        )
    else:
        simulated_wave = None  # avoid PyCharm warning (not aware of raise ValueError)
        raise_ValueError(f'Unexpected {spectrum_type=} in file {scene_fname}')

    # apply atmosphere transmission
    if apply_atmosphere_transmission:
        fapply_atmosphere_transmission(
            simulated_wave=simulated_wave,
            wave_transmission=wave_transmission,
            curve_transmission=curve_transmission,
            rng=rng,
            verbose=verbose,
            plots=plots
        )

    return simulated_wave


def generate_geometry_for_scene_block(scene_fname, scene_block, nphotons,
                                      apply_seeing, seeing_fwhm_arcsec, seeing_psf,
                                      wcs3d,
                                      min_x_ifu, max_x_ifu, min_y_ifu, max_y_ifu,
                                      rng,
                                      verbose, plots):
    """Distribute photons in the IFU focal plane for the scene block.

    Parameters
    ----------
    scene_fname : str
        YAML scene file name.
    scene_block : dict
        Dictonary storing a scene block.
    nphotons : int
        Number of photons to be generated in the scene block.
    apply_seeing : bool
        If True, apply seeing to simulated photons.
    seeing_fwhm_arcsec : TBD
    seeing_psf : TBD
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    min_x_ifu : `~astropy.units.Quantity`
        Minimum pixel X coordinate defining the IFU focal plane.
    max_x_ifu : `~astropy.units.Quantity`
        Maximum pixel X coordinate defining the IFU focal plane.
    min_y_ifu : `~astropy.units.Quantity`
        Minimum pixel Y coordinate defining the IFU focal plane.
    max_y_ifu : `~astropy.units.Quantity`
        Maximum pixel Y coordinate defining the IFU focal plane.
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    verbose : bool
        If True, display additional information.
    plots : bool
        If True, plot intermediate results.

    Returns
    -------
    simulated_x_ifu : `~astropy.units.Quantity`
        Array containing the X coordinate of the 'nphotons' photons
        in the focal plane of the IFU.
    simulated_y_ifu : `~astropy.units.Quantity`
        Array containing the X coordinate of the 'nphotons' photons
        in the focal plane of the IFU.

    """

    factor_fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))

    # plate scale
    plate_scale_x = wcs3d.wcs.cd[0, 0] * u.deg / u.pix
    plate_scale_y = wcs3d.wcs.cd[1, 1] * u.deg / u.pix
    if verbose:
        print(f'{plate_scale_x=}')
        print(f'{plate_scale_y=}')

    # define geometry type for scene block
    geometry_type = scene_block['geometry']['type']

    # simulate photons following the selected geometry
    if geometry_type == 'flatfield':
        simulated_x_ifu = rng.uniform(low=min_x_ifu.value, high=max_x_ifu.value, size=nphotons)
        simulated_y_ifu = rng.uniform(low=min_y_ifu.value, high=max_y_ifu.value, size=nphotons)
    elif geometry_type in ['gaussian', 'point-like', 'from-FITS-image']:
        if 'ra_deg' in scene_block['geometry']:
            ra_deg = scene_block['geometry']['ra_deg']
        else:
            if verbose:
                print(ctext('Assuming ra_deg: 0', faint=True))
            ra_deg = 0.0
        ra_deg *= u.deg
        if 'dec_deg' in scene_block['geometry']:
            dec_deg = scene_block['geometry']['dec_deg']
        else:
            if verbose:
                print(ctext('Assuming dec_deg: 0', faint=True))
            dec_deg = 0.0
        dec_deg *= u.deg
        if 'delta_ra_arcsec' in scene_block['geometry']:
            delta_ra_arcsec = scene_block['geometry']['delta_ra_arcsec']
        else:
            if verbose:
                print(ctext('Assuming delta_ra_deg: 0', faint=True))
            delta_ra_arcsec = 0.0
        delta_ra_arcsec *= u.arcsec
        if 'delta_dec_arcsec' in scene_block['geometry']:
            delta_dec_arcsec = scene_block['geometry']['delta_dec_arcsec']
        else:
            if verbose:
                print(ctext('Assuming delta_dec_deg: 0', faint=True))
            delta_dec_arcsec = 0.0
        delta_dec_arcsec *= u.arcsec
        x_center, y_center, w_center = wcs3d.world_to_pixel_values(
            ra_deg + delta_ra_arcsec.to(u.deg),
            dec_deg + delta_dec_arcsec.to(u.deg),
            wcs3d.wcs.crval[2]
        )
        # the previous pixel coordinates are assumed to be 0 at the center
        # of the first pixel in each dimension
        x_center += 1
        y_center += 1
        if geometry_type == 'point-like':
            simulated_x_ifu = np.repeat(x_center, nphotons)
            simulated_y_ifu = np.repeat(y_center, nphotons)
        elif geometry_type == 'gaussian':
            fwhm_ra_arcsec = scene_block['geometry']['fwhm_ra_arcsec'] * u.arcsec
            fwhm_dec_arcsec = scene_block['geometry']['fwhm_dec_arcsec'] * u.arcsec
            position_angle_deg = scene_block['geometry']['position_angle_deg'] * u.deg
            # covariance matrix for the multivariate normal
            std_x = fwhm_ra_arcsec * factor_fwhm_to_sigma / plate_scale_x.to(u.arcsec / u.pix)
            std_y = fwhm_dec_arcsec * factor_fwhm_to_sigma / plate_scale_y.to(u.arcsec / u.pix)
            rotation_matrix = np.array(  # note the sign to rotate N -> E -> S -> W
                [
                    [np.cos(position_angle_deg), np.sin(position_angle_deg)],
                    [-np.sin(position_angle_deg), np.cos(position_angle_deg)]
                ]
            )
            covariance = np.diag([std_x.value ** 2, std_y.value ** 2])
            rotated_covariance = np.dot(rotation_matrix.T, np.dot(covariance, rotation_matrix))
            # simulate X, Y values
            simulated_xy_ifu = rng.multivariate_normal(
                mean=[x_center, y_center],
                cov=rotated_covariance,
                size=nphotons
            )
            simulated_x_ifu = simulated_xy_ifu[:, 0]
            simulated_y_ifu = simulated_xy_ifu[:, 1]
        elif geometry_type == 'from-FITS-image':
            # read reference FITS file
            infile = scene_block['geometry']['filename']
            diagonal_fov_arcsec = scene_block['geometry']['diagonal_fov_arcsec'] * u.arcsec
            # generate simulated locations in the IFU
            simulated_x_ifu, simulated_y_ifu = simulate_image2d_from_fitsfile(
                infile=infile,
                diagonal_fov_arcsec=diagonal_fov_arcsec,
                plate_scale_x=plate_scale_x,
                plate_scale_y=plate_scale_y,
                nphotons=nphotons,
                rng=rng,
                background_to_subtract='mode',
                plots=plots,
                verbose=verbose
            )
            # shift image center
            simulated_x_ifu += x_center
            simulated_y_ifu += y_center
        else:
            simulated_x_ifu = None  # avoid PyCharm warning (not aware of raise ValueError)
            simulated_y_ifu = None  # avoid PyCharm warning (not aware of raise ValueError)
            raise_ValueError(f'Unexpected {geometry_type=}')
    else:
        simulated_x_ifu = None  # avoid PyCharm warning (not aware of raise ValueError)
        simulated_y_ifu = None  # avoid PyCharm warning (not aware of raise ValueError)
        raise_ValueError(f'Unexpected {geometry_type=} in file {scene_fname}')

    # apply seeing
    if apply_seeing:
        if seeing_psf == "gaussian":
            if verbose:
                print(f'Applying Gaussian PSF with {seeing_fwhm_arcsec=}')
            std_x = seeing_fwhm_arcsec * factor_fwhm_to_sigma / plate_scale_x.to(u.arcsec / u.pix)
            simulated_x_ifu += rng.normal(loc=0, scale=abs(std_x.value), size=nphotons)
            std_y = seeing_fwhm_arcsec * factor_fwhm_to_sigma / plate_scale_y.to(u.arcsec / u.pix)
            simulated_y_ifu += rng.normal(loc=0, scale=abs(std_y.value), size=nphotons)
        else:
            raise_ValueError(f'Unexpected {seeing_psf=}')

    # add units
    simulated_x_ifu *= u.pix
    simulated_y_ifu *= u.pix

    return simulated_x_ifu, simulated_y_ifu


def ifu_simulator(wcs3d, naxis1_detector, naxis2_detector, nslices,
                  noversampling_whitelight,
                  scene_fname,
                  seeing_fwhm_arcsec, seeing_psf,
                  flatpix2pix,
                  atmosphere_transmission,
                  rnoise,
                  faux_dict, rng,
                  prefix_intermediate_fits,
                  verbose=False, instname=None, subtitle=None, plots=False):
    """IFU simulator.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    naxis1_detector : `~astropy.units.Quantity`
        Detector NAXIS1, dispersion direction.
    naxis2_detector : `~astropy.units.Quantity`
        Detector NAXIS2, spatial direction (slices).
    nslices : int
        Number of IFU slices.
    noversampling_whitelight : int
        Oversampling factor (integer number) to generate the white
        light image.
    scene_fname : str
        YAML scene file name.
    seeing_fwhm_arcsec : `~astropy.units.Quantity`
        Seeing FWHM (arcsec).
    seeing_psf : str
        Seeing PSF.
    flatpix2pix : str
        String indicating whether a pixel-to-pixel flatfield is
        applied or not. Two possible values:
        - 'default': use default flatfield defined in 'faux_dict'
        - 'none': do not apply flatfield
    atmosphere_transmission : str
        String indicating whether the atmosphere transmission of
        the atmosphere is applied or not. Two possible values are:
        - 'default': use default curve defined in 'faux_dict'
        - 'none': do not apply atmosphere transmission
    rnoise : `~astropy.units.Quantity`
        Readout noise standard deviation (in ADU). Assumed to be
        Gaussian.
    faux_dict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation
          x_ifu, y_ify, wavelength -> x_detector, y_detector
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    verbose : bool
        If True, display additional information.
    instname : str or None
        Instrument name.
    subtitle : str or None
        Plot subtitle.
    plots : bool
        If True, plot intermediate results.

    Returns
    -------
    """

    if verbose:
        print(' ')
        for item in faux_dict:
            print(ctext(f'- Required file for item {item}:\n  {faux_dict[item]}', faint=True))

    if plots:
        # display SKYCALC predictions for sky radiance and transmission
        display_skycalc(faux_skycalc=faux_dict['skycalc'])

    # spatial IFU limits
    naxis1_ifu = Quantity(value=wcs3d.array_shape[2], unit=u.pix, dtype=int)
    naxis2_ifu = Quantity(value=wcs3d.array_shape[1], unit=u.pix, dtype=int)
    min_x_ifu = 0.5 * u.pix
    max_x_ifu = naxis1_ifu + 0.5 * u.pix
    min_y_ifu = 0.5 * u.pix
    max_y_ifu = naxis2_ifu + 0.5 * u.pix

    # wavelength limits
    wv_cunit1, wv_crpix1, wv_crval1, wv_cdelt1 = get_wvparam_from_wcs3d(wcs3d)
    wmin = wv_crval1 + (0.5 * u.pix - wv_crpix1) * wv_cdelt1
    wmax = wv_crval1 + (naxis1_detector + 0.5 * u.pix - wv_crpix1) * wv_cdelt1

    # load atmosphere transmission curve
    wave_transmission, curve_transmission = load_atmosphere_transmission_curve(
        atmosphere_transmission=atmosphere_transmission,
        wmin=wmin,
        wmax=wmax,
        wv_cunit1=wv_cunit1,
        faux_dict=faux_dict,
        verbose=verbose
    )

    required_keys_in_scene_block = {
        'scene_block_name',
        'spectrum',
        'geometry',
        'nphotons',
        'apply_atmosphere_transmission',
        'apply_seeing',
        'render'
    }

    nphotons_all = 0
    simulated_wave_all = None
    simulated_x_ifu_all = None
    simulated_y_ifu_all = None

    # main loop (rendering of scene blocks)
    with open(scene_fname, 'rt') as fstream:
        scene_dict = yaml.safe_load_all(fstream)
        for scene_block in scene_dict:
            scene_block_keys = set(scene_block.keys())
            if scene_block_keys != required_keys_in_scene_block:
                print(ctext(f'ERROR while processing: {scene_fname}', fg='red'))
                print(ctext('expected keys..: ', fg='blue') + f'{required_keys_in_scene_block}')
                print(ctext('keys found.....: ', fg='blue') + f'{scene_block_keys}')
                list_unexpected_keys = list(scene_block_keys.difference(required_keys_in_scene_block))
                if len(list_unexpected_keys) > 0:
                    print(ctext('unexpected keys: ', fg='red') + f'{list_unexpected_keys}')
                list_missing_keys = list(required_keys_in_scene_block.difference(scene_block_keys))
                if len(list_missing_keys) > 0:
                    print(ctext('missing keys...: ', fg='red') + f'{list_missing_keys}')
                pp.pprint(scene_block)
                raise_ValueError(f'Invalid format in file: {scene_fname}')
            scene_block_name = scene_block['scene_block_name']
            if verbose:
                print(ctext(f'\n* Processing: {scene_block_name}', fg='green'))
                pp.pprint(scene_block)
            else:
                print(ctext(f'* Processing: {scene_block_name}', fg='green'))

            nphotons = int(float(scene_block['nphotons']))
            apply_atmosphere_transmission = scene_block['apply_atmosphere_transmission']
            if atmosphere_transmission == "none" and apply_atmosphere_transmission:
                print(ctext(f'WARNING: {apply_atmosphere_transmission=} when {atmosphere_transmission=}', fg='cyan'))
                print(f'{atmosphere_transmission=} overrides {apply_atmosphere_transmission=}')
                print(f'The atmosphere transmission will not be applied!')
                apply_atmosphere_transmission = False
            apply_seeing = scene_block['apply_seeing']
            if apply_seeing:
                if seeing_fwhm_arcsec.value < 0:
                    raise_ValueError(f'Unexpected {seeing_fwhm_arcsec=}')
                elif seeing_fwhm_arcsec == 0:
                    print(ctext(f'WARNING: {apply_seeing=} when {seeing_fwhm_arcsec=}', fg='cyan'))
                    print('Seeing effect will not be applied!')
                    apply_seeing = False
            render = scene_block['render']
            if nphotons > 0 and render:
                # set wavelength unit and range
                wave_unit, wave_min, wave_max = set_wavelength_unit_and_range(
                    scene_fname=scene_fname,
                    scene_block=scene_block,
                    wmin=wmin,
                    wmax=wmax,
                    verbose=verbose
                )
                # generate spectrum
                simulated_wave = generate_spectrum_for_scene_blok(
                    scene_fname=scene_fname,
                    scene_block=scene_block,
                    faux_dict=faux_dict,
                    wave_unit=wave_unit,
                    wave_min=wave_min,
                    wave_max=wave_max,
                    nphotons=nphotons,
                    apply_atmosphere_transmission=apply_atmosphere_transmission,
                    wave_transmission=wave_transmission,
                    curve_transmission=curve_transmission,
                    rng=rng,
                    naxis1_detector=naxis1_detector,
                    verbose=verbose,
                    plots=plots
                )
                # convert to default wavelength_unit
                simulated_wave = simulated_wave.to(wv_cunit1)
                # distribute photons in the IFU focal plane
                simulated_x_ifu, simulated_y_ifu = generate_geometry_for_scene_block(
                    scene_fname=scene_fname,
                    scene_block=scene_block,
                    nphotons=nphotons,
                    apply_seeing=apply_seeing,
                    seeing_fwhm_arcsec=seeing_fwhm_arcsec,
                    seeing_psf=seeing_psf,
                    wcs3d=wcs3d,
                    min_x_ifu=min_x_ifu,
                    max_x_ifu=max_x_ifu,
                    min_y_ifu=min_y_ifu,
                    max_y_ifu=max_y_ifu,
                    rng=rng,
                    verbose=verbose,
                    plots=plots
                )
                # store all simulated photons
                if nphotons_all == 0:
                    simulated_wave_all = simulated_wave
                    simulated_x_ifu_all = simulated_x_ifu
                    simulated_y_ifu_all = simulated_y_ifu
                else:
                    simulated_wave_all = np.concatenate((simulated_wave_all, simulated_wave))
                    simulated_x_ifu_all = np.concatenate((simulated_x_ifu_all, simulated_x_ifu))
                    simulated_y_ifu_all = np.concatenate((simulated_y_ifu_all, simulated_y_ifu))
                # ---
                # update nphotons
                if verbose:
                    print(ctext(f'--> {nphotons} photons simulated', fg='blue'))
                if nphotons_all == 0:
                    nphotons_all = nphotons
                else:
                    nphotons_all += nphotons
                if len({nphotons_all,
                        len(simulated_wave_all),
                        len(simulated_x_ifu_all),
                        len(simulated_y_ifu_all)
                        }) != 1:
                    print(ctext('ERROR: check the following numbers:', fg='red'))
                    print(f'{nphotons_all=}')
                    print(f'{len(simulated_wave_all)=}')
                    print(f'{len(simulated_x_ifu_all)=}')
                    print(f'{len(simulated_y_ifu_all)=}')
                    raise_ValueError('Unexpected differences found in the previous numbers')
            else:
                if verbose:
                    if nphotons == 0:
                        print(ctext('WARNING -> nphotons: 0', fg='cyan'))
                    else:
                        print(ctext('WARNING -> render: False', fg='cyan'))

    # filter simulated photons to keep only those that fall within
    # the IFU field of view and within the expected spectral range
    # (note that this step also removes simulated photons with
    # negative wavelength value corresponding to those absorbed by
    # the atmosphere when applying the transmission curve)
    textwidth_nphotons_number = len(str(nphotons_all))
    if verbose:
        print('\nFiltering photons within IFU field of view and spectral range...')
        print(f'Initial number of simulated photons: {nphotons_all:>{textwidth_nphotons_number}}')
    cond1 = simulated_x_ifu_all >= min_x_ifu
    cond2 = simulated_x_ifu_all <= max_x_ifu
    cond3 = simulated_y_ifu_all >= min_y_ifu
    cond4 = simulated_y_ifu_all <= max_y_ifu
    cond5 = simulated_wave_all >= wmin
    cond6 = simulated_wave_all <= wmax
    iok = np.where(cond1 & cond2 & cond3 & cond4 & cond5 & cond6)[0]

    if len(iok) == 0:
        print(ctext(f'Final number of simulated photons..: {len(iok):>{textwidth_nphotons_number}}', fg='red'))
        raise SystemExit

    if len(iok) < nphotons_all:
        simulated_x_ifu_all = simulated_x_ifu_all[iok]
        simulated_y_ifu_all = simulated_y_ifu_all[iok]
        simulated_wave_all = simulated_wave_all[iok]
        nphotons_all = len(iok)
    if verbose:
        print(ctext(f'Final number of simulated photons..: {nphotons_all:>{textwidth_nphotons_number}}', fg='blue'))

    # ---------------------------------------------------------------
    # compute image2d IFU, white image, with and without oversampling
    # ---------------------------------------------------------------
    if verbose:
        print(ctext('\n* Computing image2d IFU (method 0) with and without oversampling', fg='green'))
    for noversampling in [noversampling_whitelight, 1]:
        generate_image2d_method0_ifu(
            wcs3d=wcs3d,
            noversampling_whitelight=noversampling,
            simulated_x_ifu_all=simulated_x_ifu_all,
            simulated_y_ifu_all=simulated_y_ifu_all,
            prefix_intermediate_fits=prefix_intermediate_fits,
            instname=instname,
            subtitle=subtitle,
            scene=scene_fname,
            plots=plots
        )

    # ----------------------------
    # compute image3d IFU, method0
    # ----------------------------
    if verbose:
        print(ctext('\n* Computing image3d IFU (method 0)', fg='green'))
    bins_x_ifu = (0.5 + np.arange(naxis1_ifu.value + 1)) * u.pix
    bins_y_ifu = (0.5 + np.arange(naxis2_ifu.value + 1)) * u.pix
    bins_wave = wv_crval1 + \
                ((np.arange(naxis2_detector.value + 1) + 1) * u.pix - wv_crpix1) * wv_cdelt1 - 0.5 * u.pix * wv_cdelt1
    generate_image3d_method0_ifu(
        wcs3d=wcs3d,
        simulated_x_ifu_all=simulated_x_ifu_all,
        simulated_y_ifu_all=simulated_y_ifu_all,
        simulated_wave_all=simulated_wave_all,
        bins_x_ifu=bins_x_ifu,
        bins_y_ifu=bins_y_ifu,
        bins_wave=bins_wave,
        prefix_intermediate_fits=prefix_intermediate_fits
    )

    # --------------------------------------------
    # compute image2d RSS and in detector, method0
    # --------------------------------------------
    if verbose:
        print(ctext('\n* Computing image2d RSS and detector (method 0)', fg='green'))
    bins_x_detector = np.linspace(start=0.5, stop=naxis1_detector.value + 0.5, num=naxis1_detector.value + 1)
    bins_y_detector = np.linspace(start=0.5, stop=naxis2_detector.value + 0.5, num=naxis2_detector.value + 1)

    # read ifu2detector transformations
    dict_ifu2detector = json.loads(open(faux_dict['model_ifu2detector'], mode='rt').read())

    # additional degradation in the spectral direction
    # (in units of detector pixels)
    extra_degradation_spectral_direction = rng.normal(loc=0.0, scale=1, size=nphotons_all) * u.pix

    # initialize images
    image2d_rss_method0 = np.zeros((naxis1_ifu.value * nslices, naxis1_detector.value))
    image2d_detector_method0 = np.zeros((naxis2_detector.value, naxis1_detector.value))

    # update images
    # (accelerate computation using joblib.Parallel)
    t0 = time.time()
    """
    for islice in range(nslices):
        print(f'{islice=}')
        update_image2d_rss_detector_method0(...)
    """
    Parallel(n_jobs=-1, prefer="threads")(
        delayed(update_image2d_rss_detector_method0)(
            islice=islice,
            simulated_x_ifu_all=simulated_x_ifu_all,
            simulated_y_ifu_all=simulated_y_ifu_all,
            simulated_wave_all=simulated_wave_all,
            naxis1_ifu=naxis1_ifu,
            bins_x_ifu=bins_x_ifu,
            bins_wave=bins_wave,
            bins_x_detector=bins_x_detector,
            bins_y_detector=bins_y_detector,
            wv_cdelt1=wv_cdelt1,
            extra_degradation_spectral_direction=extra_degradation_spectral_direction,
            dict_ifu2detector=dict_ifu2detector,
            image2d_rss_method0=image2d_rss_method0,
            image2d_detector_method0=image2d_detector_method0
        ) for islice in range(nslices))
    t1 = time.time()
    if verbose:
        print(f'Delta time: {t1 - t0}')

    # save RSS image (note that the flatfield effect is not included!)
    save_image2d_rss(
        wcs3d=wcs3d,
        image2d_rss=image2d_rss_method0,
        method=0,
        prefix_intermediate_fits=prefix_intermediate_fits
    )

    # apply flatpix2pix to detector image
    if flatpix2pix not in ['default', 'none']:
        raise_ValueError(f'Invalid {flatpix2pix=}')
    if flatpix2pix == 'default':
        infile = faux_dict['flatpix2pix']
        with fits.open(infile) as hdul:
            image2d_flatpix2pix = hdul[0].data
        if np.min(image2d_flatpix2pix) <= 0:
            print(f'- minimum flatpix2pix value: {np.min(image2d_flatpix2pix)}')
            raise_ValueError(f'Unexpected signal in flatpix2pix <= 0')
        naxis2_flatpix2pix, naxis1_flatpix2pix = image2d_flatpix2pix.shape
        naxis1_flatpix2pix *= u.pix
        naxis2_flatpix2pix *= u.pix
        if (naxis1_flatpix2pix != naxis1_detector) or (naxis2_flatpix2pix != naxis2_detector):
            raise_ValueError(f'Unexpected flatpix2pix shape: naxis1={naxis1_flatpix2pix}, naxis2={naxis2_flatpix2pix}')
        if verbose:
            print(f'Applying flatpix2pix: {os.path.basename(infile)} to detector image')
            print(f'- minimum flatpix2pix value: {np.min(image2d_flatpix2pix):.6f}')
            print(f'- maximum flatpix2pix value: {np.max(image2d_flatpix2pix):.6f}')
        image2d_detector_method0 /= image2d_flatpix2pix
    else:
        if verbose:
            print('Skipping applying flatpix2pix')

    # apply Gaussian readout noise to detector image
    if rnoise.value > 0:
        if verbose:
            print(f'Applying Gaussian {rnoise=} to detector image')
        ntot_pixels = naxis1_detector.value * naxis2_detector.value
        image2d_rnoise_flatten = rng.normal(loc=0.0, scale=rnoise.value, size=ntot_pixels)
        image2d_detector_method0 += image2d_rnoise_flatten.reshape((naxis2_detector.value, naxis1_detector.value))
    else:
        if verbose:
            print('Skipping adding Gaussian readout noise')

    save_image2d_detector_method0(
        wcs3d=wcs3d,
        image2d_detector_method0=image2d_detector_method0,
        prefix_intermediate_fits=prefix_intermediate_fits
    )

    # ---------------------------------------------------
    # compute image2d RSS from image in detector, method1
    # ---------------------------------------------------
    if verbose:
        print(ctext('\n* Computing image2d RSS (method 1)', fg='green'))

    # initialize image
    image2d_rss_method1 = np.zeros((naxis1_ifu.value * nslices, naxis1_detector.value))

    if verbose:
        print('Rectifying...')
    t0 = time.time()
    """
    # loop in slices
    for islice in range(nslices):
        update_image2d_rss_method1(
            islice=islice,
            image2d_detector_method0=image2d_detector_method0,
            dict_ifu2detector=dict_ifu2detector,
            naxis1_detector=naxis1_detector,
            naxis1_ifu=naxis1_ifu,
            wv_crpix1=wv_crpix1,
            wv_crval1=wv_crval1,
            wv_cdelt1=wv_cdelt1,
            image2d_rss_method1=image2d_rss_method1,
            debug=False
        )
    """
    Parallel(n_jobs=-1, prefer="threads")(
        delayed(update_image2d_rss_method1)(
            islice=islice,
            image2d_detector_method0=image2d_detector_method0,
            dict_ifu2detector=dict_ifu2detector,
            naxis1_detector=naxis1_detector,
            naxis1_ifu=naxis1_ifu,
            wv_crpix1=wv_crpix1,
            wv_crval1=wv_crval1,
            wv_cdelt1=wv_cdelt1,
            image2d_rss_method1=image2d_rss_method1,
            debug=False
        ) for islice in range(nslices))
    t1 = time.time()
    if verbose:
        print(f'Delta time: {t1 - t0}')

    save_image2d_rss(
        wcs3d=wcs3d,
        image2d_rss=image2d_rss_method1,
        method=1,
        prefix_intermediate_fits=prefix_intermediate_fits
    )

    # ------------------------------------
    # compute image3d IFU from RSS method1
    # ------------------------------------
    if verbose:
        print(ctext('\n* Computing image3d IFU from image2d RSS method 1', fg='green'))

    # kernel in the spectral direction
    # (bidimensional to be applied to a bidimensional image)
    kernel = np.array([[0.25, 0.50, 0.25]])

    # convolve RSS image
    convolved_data = convolve2d(image2d_rss_method1, kernel, boundary='fill', fillvalue=0, mode='same')

    # ToDo: the second dimension in the following array should be 2*nslices
    # (check what to do for another IFU, like TARSIS)
    image3d_ifu_method1 = np.zeros((naxis1_detector.value, naxis2_ifu.value, naxis1_ifu.value))
    if verbose:
        print(f'(debug): {image3d_ifu_method1.shape=}')

    for islice in range(nslices):
        i1 = islice * 2
        j1 = islice * naxis1_ifu.value
        j2 = j1 + naxis1_ifu.value
        image3d_ifu_method1[:, i1, :] = convolved_data[j1:j2, :].T
        image3d_ifu_method1[:, i1+1, :] = convolved_data[j1:j2, :].T

    image3d_ifu_method1 /= 2
    if verbose:
        print(f'(debug): {np.sum(image2d_rss_method1)=}')
        print(f'(debug):      {np.sum(convolved_data)=}')
        print(f'(debug): {np.sum(image3d_ifu_method1)=}')

    # save FITS file
    if len(prefix_intermediate_fits) > 0:
        hdu = fits.PrimaryHDU(image3d_ifu_method1.astype(np.float32))
        hdu.header.extend(wcs3d.to_header(), update=True)
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_ifu_3D_method1.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(f'{outfile}', overwrite='yes')
