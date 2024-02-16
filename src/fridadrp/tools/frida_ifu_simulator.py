#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import argparse
from astropy.io import fits
import matplotlib.pyplot as plt
import pooch
import sys

from fridadrp._version import version

# Parameters
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NAXIS1_IFU
from fridadrp.core import FRIDA_NAXIS2_IFU
from fridadrp.core import FRIDA_NSLICES


def download_auxiliary_images():
    """"Download auxiliary files when necessary

    # Note: compute md5 hash from terminal using:
    # linux $ md5sum <filename>
    # macOS $ md5 <filename>

    Returns
    -------
    outdict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2hawaii_medium-K: 2D polynomial transformation

    """

    faux_dict = {}

    # SKYCALC Sky Model Calculator prediction table
    faux_skycalc = pooch.retrieve(
        'http://nartex.fis.ucm.es/~ncl/fridadrp_simulator_data/skycalc_R300000_table.fits',
        known_hash='md5:49df0de4fc935de130eceacf5771350c',
        progressbar=True
    )
    faux_dict['skycalc'] = faux_skycalc

    # pixel-to-pixel flat field
    faux_flatpix2pix = pooch.retrieve(
        'http://nartex.fis.ucm.es/~ncl/fridadrp_simulator_data/simulated_flat_pix2pix.fits',
        known_hash='md5:327b983843897df229cee42513912631',
        progressbar=True
    )
    faux_dict['flatpix2pix'] = faux_flatpix2pix

    # 2D polynomial transformation from IFU (x_ifu, y_ifu, wavelength) to
    # Hawaii coordinates (x_hawaii, y_hawaii)
    faux_model_ifu2hawaii_medium_K = pooch.retrieve(
        'http://nartex.fis.ucm.es/~ncl/fridadrp_simulator_data/model_IFU2HAWAII_medium-K.json',
        known_hash='md5:a708f8cf3f94e12f53a9833b853c2a3c',
        progressbar=True
    )
    faux_dict['model_ifu2hawaii_medium-K'] = faux_model_ifu2hawaii_medium_K

    return faux_dict


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


def main(args=None):
    # parse command-line options
    parser = argparse.ArgumentParser(
        description=f"description: simulator of FRIDA IFU images ({version})"
    )

    parser.add_argument("--grating", help="Grating name", type=str,
                        choices=["low-zJ", "low-JH", "medium-z", "medium-J",
                                 "medium-H", "medium-K", "high-H", "high-K"],
                        default="medium-K")
    parser.add_argument("--scale", help="Scale", type=str, choices=["fine", "medium", "coarse"], default="fine")
    parser.add_argument("--transmission", help="Apply atmosphere transmission", action="store_true")
    parser.add_argument("--rnoise", help="Readout noise (ADU)", type=float, default=0)
    parser.add_argument("--flatpix2pix", help="Pixel-to-pixel flat field", type=str, default="default",
                        choices=["default", "none"])
    parser.add_argument("--seed", help="Seed for random number generator", type=int, default=1234)
    parser.add_argument("-v", "--verbose", help="increase program verbosity", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")

    args = parser.parse_args(args=args)
    print(f"Welcome to fridadrp-ifu_simulator\nversion {version}\n")

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    # Download auxiliary files when necessary
    faux_dict = download_auxiliary_images()

    # check grating calibration is available
    grating = args.grating
    if f'model_ifu2hawaii_{grating}' in faux_dict:
        print(f'Grating: {grating}')
    else:
        raise SystemExit(f'ERROR: calibration for grating "{grating}" not available yet!')

    rnoise = args.rnoise
    if rnoise < 0:
        raise ValueError(f'Invalid readout noise value: {rnoise}')

    verbose = args.verbose

    if verbose:
        for item in faux_dict:
            print(f'- Required file: {faux_dict[item]}')

    # display SKYCALC predictions for sky radiance and transmission
    if verbose:
        display_skycalc(faux_dict['skycalc'])


if __name__ == "__main__":

    main()
