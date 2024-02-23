#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#
import yaml
from astropy.io import fits
import astropy.units as u
from astropy.units import Unit
import matplotlib.pyplot as plt
import numpy as np
import pprint


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


def ifu_simulator(wcs, wv_lincal, naxis1_detector, naxis2_detector,
                  scene, faux_dict, rng, verbose, plots):
    """IFU simulator.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    wv_lincal : `~fridadrp.processing.linear_wavelength_calibration_frida.LinearWaveCalFRIDA`
        Linear wavelength calibration object.
    naxis1_detector : `~astropy.units.Quantity`
        Detector NAXIS1, dispersion direction.
    naxis2_detector : `~astropy.units.Quantity`
        Detector NAXIS2, spatial direction (slices).
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
    verbose : bool
        If True, display additional information.
    plots : bool
        If True, plot intermediate results.

    Returns
    -------
    """

    if verbose:
        for item in faux_dict:
            print(f'- Required file for item {item}:\n  {faux_dict[item]}')

    if plots:
        # display SKYCALC predictions for sky radiance and transmission
        display_skycalc(faux_skycalc=faux_dict['skycalc'])

    if int(naxis1_detector.value) != wcs.array_shape[0]:
        print(wcs)
        raise ValueError(f'naxis1_detector: {int(naxis1_detector.value)} != NAXIS3: {wcs.array_shape[0]} in wcs object')
    naxis1_ifu = wcs.array_shape[2] * u.pix
    naxis2_ifu = wcs.array_shape[1] * u.pix

    min_x_ifu = 0.5 * u.pix
    max_x_ifu = naxis1_ifu + 0.5 * u.pix
    min_y_ifu = 0.5 * u.pix
    max_y_ifu = naxis2_ifu + 0.5 * u.pix

    # render scene
    required_keys = ['spectrum', 'geometry', 'nphotons', 'render']
    required_keys.sort()  # in place

    nphotons_all = 0
    simulated_wave_all = None
    simulated_x_ifu_all = None
    simulated_y_ifu_all = None
    # main loop
    with open(scene, 'rt') as fstream:
        scene_dict = yaml.safe_load_all(fstream)
        for document in scene_dict:
            document_keys = list(document.keys())
            document_keys.sort()  # in place
            if document_keys == required_keys:
                if verbose:
                    print('\n* Processing:')
                    pp.pprint(document)
                nphotons = int(float(document['nphotons']))
                render = document['render']
                if nphotons > 0 and render:
                    # ---
                    # wavelength range and units
                    wave_unit = document['spectrum']['wave_unit']
                    if wave_unit is None:
                        wave_min = wv_lincal.wmin
                        wave_max = wv_lincal.wmax
                    else:
                        wave_min = document['spectrum']['wave_min']
                        if wave_min is None:
                            wave_min = wv_lincal.wmin.to(wave_unit)
                        else:
                            wave_min *= Unit(wave_unit)
                        wave_max = document['spectrum']['wave_max']
                        if wave_max is None:
                            wave_max = wv_lincal.wmax.to(wave_unit)
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
                        pass
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
                print('ERROR while processing:')
                pp.pprint(document)
                raise ValueError(f'Invalid format in file: {scene}')
