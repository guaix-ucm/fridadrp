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
        description="description: overplot traces"
    )

    parser.add_argument(
        "--grating",
        help="Grating name (low-zJ/JH, medium-z/J/H/K, high-H/K)",
        type=str,
        choices=["low-zJ", "low-JH", "medium-z", "medium-J", "medium-H", "medium-K", "high-H", "high-K"],
        default="medium-K"
    )
    parser.add_argument(
        "--scale",
        help="Scale (fine, medium, coarse)",
        type=str,
        choices=["fine", "medium", "coarse"],
        default="fine"
    )
    parser.add_argument(
        "--transmission",
        help="Apply atmosphere transmission",
        action="store_true"
    )
    parser.add_argument("--rnoise", help="Readout noise (ADU)", type=float, default=0)
    parser.add_argument("--flatpix2pix", help="Pixel-to-pixel flat field", type=str, default="default",
                        choices=["default", "none"])
    parser.add_argument("-v", "--verbose", help="increase program verbosity", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")

    args = parser.parse_args(args=args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    rnoise = args.rnoise
    if rnoise < 0:
        raise ValueError(f'Invalid readout noise value: {rnoise}')

    verbose = args.verbose

    # ---
    # Download auxiliary files when necessary
    # Note: compute md5 hash from terminal using:
    # linux $ md5sum <filename>
    # macOS $ md5 <filename>

    # SKYCALC Sky Model Calculator prediction table
    faux_skycalc = pooch.retrieve(
        'http://nartex.fis.ucm.es/~ncl/fridadrp_simulator_data/skycalc_R300000_table.fits',
        known_hash='md5:49df0de4fc935de130eceacf5771350c',
        progressbar=True
    )

    # pixel-to-pixel flat field
    faux_flatpix2pix = pooch.retrieve(
        'http://nartex.fis.ucm.es/~ncl/fridadrp_simulator_data/simulated_flat_pix2pix.fits',
        known_hash='md5:327b983843897df229cee42513912631',
        progressbar=True
    )

    if args.verbose:
        print(f'- Required file: {faux_skycalc}')
        print(f'- Required file: {faux_flatpix2pix}')

    # display SKYCALC predictions for sky radiance and transmission
    if verbose:
        display_skycalc(faux_skycalc)


if __name__ == "__main__":

    main()
