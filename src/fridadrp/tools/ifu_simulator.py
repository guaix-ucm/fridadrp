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

from fridadrp.processing.linear_wavelength_calibration_frida import LinearWaveCalFRIDA


def display_skycalc(grating, faux_skycalc):
    """
    Display sky radiance and transmission.

    Data generated with the SKYCALC Sky Model Calculator tool
    provided by ESO.
    See https://www.eso.org/observing/etc/doc/skycalc/helpskycalc.html

    Parameters
    ----------
    grating : str
        Grating name.
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


def ifu_simulator(grating, faux_dict, verbose):
    """IFU simulator.

    Parameters
    ----------
    grating : str
        Grating name.
    faux_dict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation
          x_ifu, y_ify, wavelength -> x_detector, y_detector
    verbose : bool
        If True, display/plot additional information.

    Returns
    -------
    """

    if verbose:
        for item in faux_dict:
            print(f'- Required file for item {item}:\n  {faux_dict[item]}')
        # display SKYCALC predictions for sky radiance and transmission
        display_skycalc(grating=grating, faux_skycalc=faux_dict['skycalc'])

    wv_lincal = LinearWaveCalFRIDA(grating=grating)
    if verbose:
        print(wv_lincal)
        print(wv_lincal.__repr__())
