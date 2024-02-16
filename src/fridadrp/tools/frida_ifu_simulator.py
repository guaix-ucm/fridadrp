#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import argparse
import json
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


def define_auxiliary_files(grating, verbose):
    """"Define auxiliary files for requested configuration

    Parameters
    ----------
    grating : str
        Grating name.
    verbose : bool
        If True, display/plot additional information.

    Returns
    -------
    outdict : Python dictionary
        File names of auxiliary files:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation
          x_ifu, y_ify, wavelength -> x_detector, y_detector

    """

    # retrieve configuration file
    base_url = 'http://nartex.fis.ucm.es/~ncl/fridadrp_simulator_data'
    # note: compute md5 hash from terminal using:
    # linux $ md5sum <filename>
    # macOS $ md5 <filename>
    fconf = pooch.retrieve(
        f'{base_url}/configuration_FRIDA_IFU_simulator.json',
        known_hash='md5:2599ac77b395e699b90a0711c8c15988',
        path=pooch.os_cache(project="fridadrp"),
        progressbar=True
    )
    dconf = json.loads(open(fconf, mode='rt').read())
    if verbose:
        print(f"Configuration file uuid: {dconf['uuid']}")

    # generate registry for all the auxiliary files to be used by Pooch
    d = dconf['auxfiles']
    registry_md5 = {}
    registry_label = {}
    # SKYCALC Sky Model Calculator prediction table
    label = 'skycalc'
    registry_label[d[label]['filename']] = label
    registry_md5[d[label]['filename']] = f"md5:{d[label]['md5']}"
    # pixel-to-pixel flat field
    label = 'flatpix2pix'
    filename = d[label][grating]['filename']
    md5 = d[label][grating]['md5']
    if (filename is not None) and (md5 is not None):
        registry_label[filename] = label
        registry_md5[filename] = f'md5:{md5}'
    else:
        raise SystemExit(f'Error: grating {grating} has not yet been defined!')
    # 2D polynomial transformation from IFU (x_ifu, y_ifu, wavelength) to
    # Hawaii coordinates (x_hawaii, y_hawaii)
    label = 'model_ifu2detector'
    filename = d[label][grating]['filename']
    md5 = d[label][grating]['md5']
    if (filename is not None) and (md5 is not None):
        registry_label[filename] = label
        registry_md5[filename] = f'md5:{md5}'
    else:
        raise SystemExit(f'Error: grating {grating} has not yet been defined!')

    # create a Pooch instance with the previous registry
    pooch_inst = pooch.create(
        # use the default cache folder for the operating system
        path=pooch.os_cache(project="fridadrp"),
        # base URL for the remote data source
        base_url=base_url,
        # specify the files that can be fetched
        registry=registry_md5
    )

    # initialize output dictionary
    faux_dict = {}
    for item in registry_md5:
        try:
            faux = pooch_inst.fetch(item, progressbar=True)
            label = registry_label[item]
            faux_dict[label] = faux
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
    verbose = args.verbose

    # define auxiliary files
    faux_dict = define_auxiliary_files(grating, verbose=verbose)

    rnoise = args.rnoise
    if rnoise < 0:
        raise ValueError(f'Invalid readout noise value: {rnoise}')

    ifu_simulator(
        faux_dict=faux_dict,
        verbose=verbose
    )


if __name__ == "__main__":

    main()
