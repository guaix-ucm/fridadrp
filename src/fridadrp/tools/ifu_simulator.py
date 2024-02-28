#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
from astropy import wcs
from astropy.io import fits
from astropy.units import Quantity
import astropy.units as u
from astropy.units import Unit
from joblib import Parallel, delayed
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pprint
import yaml

from fridadrp.processing.define_3d_wcs import get_wvparam_from_wcs3d
from numina.array.distortion import fmap


pp = pprint.PrettyPrinter(indent=1, sort_dicts=False)


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

    if np.any(line_flux < 0):
        raise ValueError(f'Negative line fluxes cannot be handled')

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
        lower_index = np.searchsorted(line_wave.value, wmin.value, side='left')
    else:
        lower_index = 0

    # upper wavelength limit
    if wmax is not None:
        if not isinstance(wmax, u.Quantity):
            raise ValueError(f"Object 'wmax': {wmax} is not a Quantity instance")
        if not wmax.unit.is_equivalent(u.m):
            raise ValueError(f"Unexpected unit for 'wmax': {wmin}")
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


def simulate_spectrum(wave, flux, nphotons, rng, wmin, wmax, nbins_histo, plots):
    """Simulate spectrum defined by tabulated wave and flux data.

    Parameters
    ----------
    wave : `~astropy.units.Quantity`
        Numpy array (with astropy units) containing the tabulated
        wavelength.
    flux : array_like
        Array-like object containing the tabulated flux.
    nphotons : int
        Number of photons to be simulated
    rng : `~numpy.random._generator.Generator`
        Random number generator.
    wmin : `~astroppy.units.Quantity`
        Minimum wavelength to be considered.
    wmax : `~astroppy.units.Quantity`
        Maximum wavelength to be considered.
    nbins_histo : int
        Number of bins for histogram plot.
    plots : bool
        If True, plot input and output results.

    Returns
    -------
    simulated_wave : `~astropy.units.Quantity`
        Wavelength of simulated photons.
    """

    flux = np.asarray(flux)
    if len(wave) != len(flux):
        raise ValueError(f"Incompatible array length: 'wave' ({len(wave)}), 'flux' ({len(flux)})")

    if np.any(flux < 0):
        raise ValueError(f'Negative flux values cannot be handled')

    if not isinstance(wave, u.Quantity):
        raise ValueError(f"Object 'wave': {wave} is not a Quantity instance")
    wave_unit = wave.unit
    if not wave_unit.is_equivalent(u.m):
        raise ValueError(f"Unexpected unit for 'wave': {wave_unit}")

    # lower wavelength limit
    if wmin is not None:
        if not isinstance(wmin, u.Quantity):
            raise ValueError(f"Object 'wmin':{wmin}  is not a Quantity instance")
        if not wmin.unit.is_equivalent(u.m):
            raise ValueError(f"Unexpected unit for 'wmin': {wmin}")
        wmin = wmin.to(wave_unit)
        lower_index = np.searchsorted(wave.value, wmin.value, side='left')
    else:
        lower_index = 0

    # upper wavelength limit
    if wmax is not None:
        if not isinstance(wmax, u.Quantity):
            raise ValueError(f"Object 'wmax': {wmax} is not a Quantity instance")
        if not wmax.unit.is_equivalent(u.m):
            raise ValueError(f"Unexpected unit for 'wmax': {wmin}")
        wmax = wmax.to(wave_unit)
        upper_index = np.searchsorted(wave.value, wmax.value, side='right')
    else:
        upper_index = len(wave)

    if plots:
        fig, ax = plt.subplots()
        ax.plot(wave.value, flux, '-')
        if wmin is not None:
            ax.axvline(wmin.value, linestyle='--', color='gray')
        if wmax is not None:
            ax.axvline(wmax.value, linestyle='--', color='gray')
        ax.set_xlabel(f'Wavelength ({wave_unit})')
        ax.set_ylabel('Flux (arbitrary units)')
        plt.tight_layout()
        plt.show()

    wave = wave[lower_index:upper_index]
    flux = flux[lower_index:upper_index]

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
        plt.tight_layout()
        plt.show()

    # samples following a uniform distribution
    unisamples = rng.uniform(low=0, high=1, size=nphotons)
    simulated_wave = np.interp(x=unisamples, xp=normalized_cumulative_area, fp=wave.value)
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
        ax.legend()
        plt.tight_layout()
        plt.show()

    return simulated_wave


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
    wv_cdelt1 : ~astropy.units.Quantity`
        CDELT1 value along the spectral direction.
    extra_degradation_spectral_direction : `~astropy.units.Quantity`
        Additional degradation in the spectral direction, in units of
        the detector pixels, for each simulated photon.
    dict_ifu2detector : dict
        A Python dictionary containing the 2D polynomials that allow
        to transform (X, Y) coordinates in the IFU focal plane to
        (X, Y) coordinates in the detector.
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
            y=simulated_wave_all.value[iok] + \
              (simulated_y_ifu_all.value[iok] - y_ifu_expected) * wv_cdelt1.value + \
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


def save_image2d_rss_detector_method0(
        wcs3d,
        image2d_rss_method0,
        image2d_detector_method0,
        prefix_intermediate_fits
):
    """Save the two 2D images: RSS and detector.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    image2d_rss_method0 : `~numpy.ndarray`
        2D array containing the RSS image. This array is
        updated by this function.
    image2d_detector_method0 : `~numpy.ndarray`
        2D array containing the detector image. This array is
        updated by this function.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    """

    if len(prefix_intermediate_fits) > 0:
        # -------------------------------------------------
        # 1) spectroscopic 2D image with continguous slices
        # -------------------------------------------------
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
        hdu = fits.PrimaryHDU(image2d_rss_method0.astype(np.float32))
        hdu.header.extend(wcs2d.to_header(), update=True)
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_rss_2D_method0.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(outfile, overwrite='yes')

        # -----------------------------------------
        # 2) spectroscopic 2D image in the detector
        # -----------------------------------------
        hdu = fits.PrimaryHDU(image2d_detector_method0.astype(np.float32))
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_detector_2D_method0.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(outfile, overwrite='yes')


def ifu_simulator(wcs3d, naxis1_detector, naxis2_detector, nslices,
                  noversampling_whitelight,
                  scene, faux_dict, rng,
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
    scene : str
        YAML scene file name.
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
            print(f'- Required file for item {item}:\n  {faux_dict[item]}')

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

    # render scene
    required_keys_in_scene = {'spectrum', 'geometry', 'nphotons', 'render'}
    expected_keys_in_spectrum = {'type', 'wave_unit', 'wave_min', 'wave_max'}

    nphotons_all = 0
    simulated_wave_all = None
    simulated_x_ifu_all = None
    simulated_y_ifu_all = None
    # main loop
    with open(scene, 'rt') as fstream:
        scene_dict = yaml.safe_load_all(fstream)
        for document in scene_dict:
            document_keys = set(document.keys())
            if document_keys != required_keys_in_scene:
                print(f'ERROR while processing: {scene}')
                print(f'expected keys: {required_keys_in_scene}')
                print(f'keys found...: {document_keys}')
                pp.pprint(document)
                raise ValueError(f'Invalid format in file: {scene}')
            if verbose:
                print('\n* Processing:')
                pp.pprint(document)
            nphotons = int(float(document['nphotons']))
            render = document['render']
            if nphotons > 0 and render:
                # ---
                # wavelength range and units
                spectrum_keys = set(document['spectrum'].keys())
                if not expected_keys_in_spectrum.issubset(spectrum_keys) :
                    print(f'ERROR while processing: {scene}')
                    print(f'expected keys: {expected_keys_in_spectrum}')
                    print(f'keys found...: {spectrum_keys}')
                    pp.pprint(document)
                    raise ValueError(f'Invalid format in file: {scene}')
                wave_unit = document['spectrum']['wave_unit']
                if wave_unit is None:
                    wave_min = wmin
                    wave_max = wmax
                else:
                    wave_min = document['spectrum']['wave_min']
                    if wave_min is None:
                        wave_min = wmin.to(wave_unit)
                    else:
                        wave_min *= Unit(wave_unit)
                    wave_max = document['spectrum']['wave_max']
                    if wave_max is None:
                        wave_max = wmax.to(wave_unit)
                    else:
                        wave_max *= Unit(wave_unit)
                # ---
                # spectrum type
                spectrum_type = document['spectrum']['type']
                if spectrum_type == 'delta-lines':
                    filename = document['spectrum']['filename']
                    wave_column = document['spectrum']['wave_column'] - 1
                    flux_column = document['spectrum']['flux_column'] - 1
                    if filename[0] == '@':
                        # retrieve file name from dictionary of auxiliary
                        # file names for the considered instrument
                        filename = faux_dict[filename[1:]]
                    catlines = np.genfromtxt(filename)
                    line_wave = catlines[:, wave_column] * Unit(wave_unit)
                    if not np.all(np.diff(line_wave.value) > 0):
                        raise ValueError(f"Wavelength array 'line_wave'={line_wave} is not sorted!")
                    line_flux = catlines[:, flux_column]
                    simulated_wave = simulate_delta_lines(
                        line_wave=line_wave,
                        line_flux=line_flux,
                        nphotons=nphotons,
                        rng=rng,
                        wmin=wave_min,
                        wmax=wave_max,
                        plots=plots
                    )
                elif spectrum_type == 'skycalc-radiance':
                    faux_skycalc = faux_dict['skycalc']
                    with fits.open(faux_skycalc) as hdul:
                        skycalc_table = hdul[1].data
                    wave = skycalc_table['lam'] * Unit(wave_unit)
                    if not np.all(np.diff(wave.value) > 0):
                        raise ValueError(f"Wavelength array 'wave'={wave} is not sorted!")
                    flux = skycalc_table['flux']
                    simulated_wave = simulate_spectrum(
                        wave=wave,
                        flux=flux,
                        nphotons=nphotons,
                        rng=rng,
                        wmin=wave_min,
                        wmax=wave_max,
                        nbins_histo=naxis1_detector.value,
                        plots=plots
                    )
                elif spectrum_type == 'tabulated-spectrum':
                    filename = document['spectrum']['filename']
                    wave_column = document['spectrum']['wave_column'] - 1
                    flux_column = document['spectrum']['flux_column'] - 1
                    if 'redshift' in document['spectrum']:
                        redshift = document['spectrum']['redshift']
                    else:
                        redshift = 0.0
                    table_data = np.genfromtxt(filename)
                    wave = table_data[:, wave_column] * (1 + redshift) * Unit(wave_unit)
                    if not np.all(np.diff(wave.value) > 0):
                        raise ValueError(f"Wavelength array 'wave'={wave} is not sorted!")
                    flux = table_data[:, flux_column]
                    simulated_wave = simulate_spectrum(
                        wave=wave,
                        flux=flux,
                        nphotons=nphotons,
                        rng=rng,
                        wmin=wave_min,
                        wmax=wave_max,
                        nbins_histo=naxis1_detector.value,
                        plots=plots
                    )
                elif spectrum_type == 'constant-flux':
                    simulated_wave = simulate_constant_flux(
                        wmin=wave_min,
                        wmax=wave_max,
                        nphotons=nphotons,
                        rng=rng
                    )
                else:
                    raise ValueError(f'Unexpected spectrum type: "{spectrum_type}" '
                                     f'in file "{scene}"')
                # convert to default wavelength_unit
                simulated_wave = simulated_wave.to(wv_cunit1)
                if nphotons_all == 0:
                    simulated_wave_all = simulated_wave
                else:
                    simulated_wave_all = np.concatenate((simulated_wave_all, simulated_wave))
                # ---
                # geometry
                geometry_type = document['geometry']['type']
                if geometry_type == 'flatfield':
                    simulated_x_ifu = rng.uniform(low=min_x_ifu.value, high=max_x_ifu.value, size=nphotons)
                    simulated_x_ifu *= u.pix
                    simulated_y_ifu = rng.uniform(low=min_y_ifu.value, high=max_y_ifu.value, size=nphotons)
                    simulated_y_ifu *= u.pix
                elif geometry_type == 'gaussian':
                    ra_deg = document['geometry']['ra_deg'] * u.deg
                    dec_deg = document['geometry']['dec_deg'] * u.deg
                    fwhm_ra_arcsec = document['geometry']['fwhm_ra_arcsec'] * u.arcsec
                    fwhm_dec_arcsec = document['geometry']['fwhm_dec_arcsec'] * u.arcsec
                    position_angle_deg = document['geometry']['position_angle_deg'] * u.deg
                    x_center, y_center, w_center = wcs3d.world_to_pixel_values(ra_deg, dec_deg, wave_min)
                    # the previous pixel coordinates are assumed to be 0 at the center
                    # of the first pixel in each dimension
                    x_center += 1
                    y_center += 1
                    # plate scale
                    plate_scale_x = wcs3d.wcs.cd[0, 0] * u.deg / u.pix
                    plate_scale_y = wcs3d.wcs.cd[1, 1] * u.deg / u.pix
                    # covariance matrix for the multivariate normal
                    factor_fwhm_to_sigma = 1 / (2 * np.sqrt(2 * np.log(2)))
                    std_x = fwhm_ra_arcsec * factor_fwhm_to_sigma / plate_scale_x.to(u.arcsec / u.pix)
                    std_y = fwhm_dec_arcsec * factor_fwhm_to_sigma / plate_scale_y.to(u.arcsec / u.pix)
                    rotation_matrix = np.array(  # note the sign to rotate N -> E -> S -> W
                        [
                            [np.cos(position_angle_deg), np.sin(position_angle_deg)],
                            [-np.sin(position_angle_deg), np.cos(position_angle_deg)]
                        ]
                    )
                    covariance = np.diag([std_x.value**2, std_y.value**2])
                    rotated_covariance = np.dot(rotation_matrix.T, np.dot(covariance, rotation_matrix))
                    # simulate X, Y values
                    simulated_xy_ifu = rng.multivariate_normal(
                        mean=[x_center, y_center],
                        cov=rotated_covariance,
                        size=nphotons
                    )
                    simulated_x_ifu = simulated_xy_ifu[:, 0] * u.pix
                    simulated_y_ifu = simulated_xy_ifu[:, 1] * u.pix
                else:
                    raise ValueError(f'Unexpected geometry type: "{geometry_type}" '
                                     f'in file "{scene}"')
                if nphotons_all == 0:
                    simulated_x_ifu_all = simulated_x_ifu
                    simulated_y_ifu_all = simulated_y_ifu
                else:
                    simulated_x_ifu_all = np.concatenate((simulated_x_ifu_all, simulated_x_ifu))
                    simulated_y_ifu_all = np.concatenate((simulated_y_ifu_all, simulated_y_ifu))
                # ---
                # update nphotons
                if verbose:
                    print(f'--> {nphotons} simulated')
                if nphotons_all == 0:
                    nphotons_all = nphotons
                else:
                    nphotons_all += nphotons
                if len({nphotons_all,
                        len(simulated_wave_all),
                        len(simulated_x_ifu_all),
                        len(simulated_y_ifu_all)
                        }) != 1:
                    print('ERROR: check the following numbers:')
                    print(f'{nphotons_all=}')
                    print(f'{len(simulated_wave_all)=}')
                    print(f'{len(simulated_x_ifu_all)=}')
                    print(f'{len(simulated_y_ifu_all)=}')
                    raise ValueError('Unexpected differences found in the previous numbers')
            else:
                if verbose:
                    if nphotons == 0:
                        print('WARNING -> nphotons: 0')
                    else:
                        print('WARNING -> render: False')

    # filter simulated photons to keep only those that fall within
    # the IFU field of view and within the expected spectral range
    textwidth_nphotons_number = len(str(nphotons_all))
    if verbose:
        print('Filtering photons within IFU field of view and spectral range...')
        print(f'Initial number of simulated photons: {nphotons_all:>{textwidth_nphotons_number}}')
    cond1 = simulated_x_ifu_all >= min_x_ifu
    cond2 = simulated_x_ifu_all <= max_x_ifu
    cond3 = simulated_y_ifu_all >= min_y_ifu
    cond4 = simulated_y_ifu_all <= max_y_ifu
    cond5 = simulated_wave_all >= wmin
    cond6 = simulated_wave_all <= wmax
    iok = np.where(cond1 & cond2 & cond3 & cond4 & cond5 & cond6)[0]

    if len(iok) < nphotons_all:
        simulated_x_ifu_all = simulated_x_ifu_all[iok]
        simulated_y_ifu_all = simulated_y_ifu_all[iok]
        simulated_wave_all = simulated_wave_all[iok]
        nphotons_all = len(iok)
    if verbose:
        print(f'Final number of simulated photons..: {nphotons_all:>{textwidth_nphotons_number}}')

    # ---------------------------------------------------------------
    # compute image2d IFU, white image, with and without oversampling
    # ---------------------------------------------------------------
    for noversampling in [noversampling_whitelight, 1]:
        generate_image2d_method0_ifu(
            wcs3d=wcs3d,
            noversampling_whitelight=noversampling,
            simulated_x_ifu_all=simulated_x_ifu_all,
            simulated_y_ifu_all=simulated_y_ifu_all,
            prefix_intermediate_fits=prefix_intermediate_fits,
            instname=instname,
            subtitle=subtitle,
            scene=scene,
            plots=plots
        )

    # ----------------------------
    # compute image3d IFU, method0
    # ----------------------------
    bins_x_ifu = (0.5 + np.arange(naxis1_ifu.value + 1)) * u.pix
    bins_y_ifu = (0.5 + np.arange(naxis2_ifu.value + 1)) * u.pix
    bins_wave = wv_crval1 + \
                (np.arange(naxis2_detector.value + 1) * u.pix - wv_crpix1) * wv_cdelt1 - 0.5 * u.pix * wv_cdelt1
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

    save_image2d_rss_detector_method0(
        wcs3d=wcs3d,
        image2d_rss_method0=image2d_rss_method0,
        image2d_detector_method0=image2d_detector_method0,
        prefix_intermediate_fits=prefix_intermediate_fits
    )
