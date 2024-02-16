#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import argparse
import pooch
import sys

from .ifu_simulator import ifu_simulator

from fridadrp._version import version

# Parameters
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NAXIS1_IFU
from fridadrp.core import FRIDA_NAXIS2_IFU
from fridadrp.core import FRIDA_NSLICES


def download_auxiliary_images(grating):
    """"Download auxiliary files when necessary

    # Note: compute md5 hash from terminal using:
    # linux $ md5sum <filename>
    # macOS $ md5 <filename>

    Parameters
    ----------
    grating : str
        Grating name.

    Returns
    -------
    outdict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation
          x_ifu, y_ify, wavelength -> x_detector, y_detector

    """

    registry = {
        'skycalc_R300000_table.fits': 'md5:49df0de4fc935de130eceacf5771350c',
        'simulated_flat_pix2pix.fits': 'md5:327b983843897df229cee42513912631',
        'model_IFU2HAWAII_medium-K.json': 'md5:a708f8cf3f94e12f53a9833b853c2a3c',
    }

    if f'model_IFU2HAWAII_{grating}.json' not in registry:
        raise SystemExit(f'Error: grating {grating} has not yet been defined!')

    pooch_inst = pooch.create(
        # use the default cache folder for the operating system
        path=pooch.os_cache(project="fridadrp"),
        # base URL for the remote data source
        base_url='http://nartex.fis.ucm.es/~ncl/fridadrp_simulator_data/',
        # specify the files that can be fetched
        registry=registry
    )

    # initialize output dictionary
    faux_dict = {}

    # SKYCALC Sky Model Calculator prediction table
    try:
        faux_skycalc = pooch_inst.fetch('skycalc_R300000_table.fits', progressbar=True)
        faux_dict['skycalc'] = faux_skycalc
    except BaseException as e:
        raise SystemExit(e)

    # pixel-to-pixel flat field
    try:
        faux_flatpix2pix = pooch_inst.fetch('simulated_flat_pix2pix.fits', progressbar=True)
        faux_dict['flatpix2pix'] = faux_flatpix2pix
    except BaseException as e:
        raise SystemExit(e)

    # 2D polynomial transformation from IFU (x_ifu, y_ifu, wavelength) to
    # Hawaii coordinates (x_hawaii, y_hawaii)
    try:
        faux_model_ifu2detector = pooch_inst.fetch(f'model_IFU2HAWAII_{grating}.json', progressbar=True)
        faux_dict['model_ifu2detector'] = faux_model_ifu2detector
    except BaseException as e:
        raise SystemExit(e)

    return faux_dict


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

    grating = args.grating

    # Download auxiliary files when necessary
    faux_dict = download_auxiliary_images(grating)

    rnoise = args.rnoise
    if rnoise < 0:
        raise ValueError(f'Invalid readout noise value: {rnoise}')

    verbose = args.verbose
    ifu_simulator(
        faux_dict=faux_dict,
        verbose=verbose
    )


if __name__ == "__main__":

    main()
