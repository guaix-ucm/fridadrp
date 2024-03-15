#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import argparse
from astropy.coordinates import SkyCoord
import astropy.units as u
import json
import numpy as np
import pooch
import sys

from .ifu_simulator import ifu_simulator

from fridadrp._version import version
from fridadrp.processing.define_3d_wcs import define_3d_wcs
from fridadrp.processing.linear_wavelength_calibration_frida import LinearWaveCalFRIDA

# Parameters
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NAXIS1_IFU
from fridadrp.core import FRIDA_NAXIS2_IFU
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_VALID_GRATINGS
from fridadrp.core import FRIDA_VALID_SPATIAL_SCALES
from fridadrp.core import FRIDA_SPATIAL_SCALE


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
    outdict : dictionary
        Dictionary with the file name of the auxiliary files.
        The dictionary keys are the following:
        - skycalc: table with SKYCALC Sky Model Calculator predictions
        - flatpix2pix: pixel-to-pixel flat field
        - model_ifu2detector: 2D polynomial transformation from
          (x_ifu, y_ify, wavelength) to (x_detector, y_detector)

    """

    # retrieve configuration file
    base_url = 'http://nartex.fis.ucm.es/~ncl/fridadrp_simulator_data'
    # note: compute md5 hash from terminal using:
    # linux $ md5sum <filename>
    # macOS $ md5 <filename>
    fconf = pooch.retrieve(
        f'{base_url}/configuration_FRIDA_IFU_simulator.json',
        known_hash='md5:414bc015dc3d68c1a19b8a5a298afbf2',
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
    filename = d[label]['filename']
    registry_label[filename] = label
    registry_md5[filename] = f"md5:{d[label]['md5']}"
    # EMIR arc lines
    label = 'EMIR-arc-delta-lines'
    filename = d[label]['filename']
    registry_label[filename] = label
    registry_md5[filename] = f"md5:{d[label]['md5']}"
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
    parser.add_argument("scene", help="YAML scene file name", type=str)
    parser.add_argument("--grating", help="Grating name", type=str, choices=FRIDA_VALID_GRATINGS, default="medium-K")
    parser.add_argument("--scale", help="Scale", type=str, choices=FRIDA_VALID_SPATIAL_SCALES, default="fine")
    parser.add_argument("--ra_teles_deg", help="Telescope central RA (deg)", type=float, default=0.0)
    parser.add_argument("--dec_teles_deg", help="Telescope central DEC (deg)", type=float, default=0.0)
    parser.add_argument("--delta_ra_teles_arcsec", help="Offset in RA (arcsec)", type=float, default=0.0)
    parser.add_argument("--delta_dec_teles_arcsec", help="Offset in DEC (arcsec)", type=float, default=0.0)
    parser.add_argument("--seeing_fwhm_arcsec", help="Seeing FWHM (arcsec)", type=float, default=0.0)
    parser.add_argument("--seeing_psf", help="Seeing PSF", type=str, default="gaussian",
                        choices=["gaussian"])
    parser.add_argument("--noversampling_whitelight", help="Oversampling white light image", type=int, default=10)
    parser.add_argument("--transmission", help="Atmosphere transmission", type=str, default="default",
                        choices=["default", "none"])
    parser.add_argument("--rnoise", help="Readout noise standard deviation (ADU)", type=float, default=0)
    parser.add_argument("--flatpix2pix", help="Pixel-to-pixel flat field", type=str, default="default",
                        choices=["default", "none"])
    parser.add_argument("--seed", help="Seed for random number generator", type=int, default=1234)
    parser.add_argument("--prefix_intermediate_FITS", help="Prefix for intermediate FITS files", type=str, default="")
    parser.add_argument("-v", "--verbose", help="increase program verbosity", action="store_true")
    parser.add_argument("--plots", help="plot intermediate results", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")

    args = parser.parse_args(args=args)

    print(f"Welcome to fridadrp-ifu_simulator\nversion {version}\n")

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    # simplify argument names
    scene = args.scene
    grating = args.grating
    scale = args.scale
    seeing_fwhm_arcsec = args.seeing_fwhm_arcsec * u.arcsec
    if seeing_fwhm_arcsec.value < 0:
        raise ValueError(f'Unexpected {seeing_fwhm_arcsec=}. This number must be >= 0.')
    seeing_psf = args.seeing_psf
    noversampling_whitelight = args.noversampling_whitelight
    if noversampling_whitelight < 1:
        raise ValueError(f'Unexpected {noversampling_whitelight=} (must be > 1)')
    atmosphere_transmission = args.transmission
    rnoise = args.rnoise
    if rnoise < 0:
        raise ValueError(f'Invalid readout noise value: {rnoise}')
    rnoise *= u.adu
    flatpix2pix = args.flatpix2pix
    prefix_intermediate_fits = args.prefix_intermediate_FITS
    seed = args.seed
    verbose = args.verbose
    plots = args.plots

    # define auxiliary files
    faux_dict = define_auxiliary_files(grating, verbose=verbose)

    # World Coordinate System of the data cube
    ra_teles_deg = args.ra_teles_deg
    dec_teles_deg = args.dec_teles_deg
    delta_ra_teles_arcsec = args.delta_ra_teles_arcsec
    delta_dec_teles_arcsec = args.delta_dec_teles_arcsec
    skycoord_center = SkyCoord(
        ra=ra_teles_deg * u.deg + (delta_ra_teles_arcsec * u.arcsec).to(u.deg),
        dec=dec_teles_deg * u.deg + (delta_dec_teles_arcsec * u.arcsec).to(u.deg),
        frame='icrs'
    )

    # linear wavelength calibration
    wv_lincal = LinearWaveCalFRIDA(grating=grating)
    if verbose:
        print(f'\n{wv_lincal}')

    # define WCS object to store the spatial 2D WCS
    # and the linear wavelength calibration
    wcs3d = define_3d_wcs(
        naxis1_ifu=FRIDA_NAXIS1_IFU,
        naxis2_ifu=FRIDA_NAXIS2_IFU,
        skycoord_center=skycoord_center,
        spatial_scale=FRIDA_SPATIAL_SCALE[scale],
        wv_lincal=wv_lincal,
        verbose=verbose
    )

    # initialize random number generator with provided seed
    rng = np.random.default_rng(seed)

    ifu_simulator(
        wcs3d=wcs3d,
        naxis1_detector=FRIDA_NAXIS1_HAWAII,
        naxis2_detector=FRIDA_NAXIS2_HAWAII,
        nslices=FRIDA_NSLICES,
        noversampling_whitelight=noversampling_whitelight,
        scene_fname=scene,
        seeing_fwhm_arcsec=seeing_fwhm_arcsec,
        seeing_psf=seeing_psf,
        flatpix2pix=flatpix2pix,
        atmosphere_transmission=atmosphere_transmission,
        rnoise=rnoise,
        faux_dict=faux_dict,
        rng=rng,
        prefix_intermediate_fits=prefix_intermediate_fits,
        verbose=verbose,
        instname='FRIDA',
        subtitle=f'scale: {scale}, grating: {grating}',
        plots=plots
    )


if __name__ == "__main__":

    main()
