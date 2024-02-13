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
    parser.add_argument("-v", "--verbose", help="increase program verbosity", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")

    args = parser.parse_args(args=args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    # Download SKYCALC Sky Model Calculator prediction table
    # compute md5 hash from terminal using:
    # linux $ md5sum <filename>
    # macOS $ md5 <filename>
    faux_skycalc = pooch.retrieve(
        'http://nartex.fis.ucm.es/~ncl/fridadrp_simulator_data/skycalc_R300000_table.fits',
        known_hash='md5:49df0de4fc935de130eceacf5771350c'
    )

    if args.verbose:
        print(f'- Required file: {faux_skycalc}')


if __name__ == "__main__":

    main()
