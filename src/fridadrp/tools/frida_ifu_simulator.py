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
from astropy.io import fits
import astropy.units as u
from datetime import datetime
import numpy as np
import platform
import sys

from numina.instrument.simulation.ifu.ifu_simulator import ifu_simulator
from numina.instrument.simulation.ifu.define_3d_wcs import define_3d_wcs

from fridadrp._version import version
from fridadrp.instrument.define_auxiliary_files import define_auxiliary_files
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


def main(args=None):

    # parse command-line options
    parser = argparse.ArgumentParser(
        description=f"description: simulator of FRIDA IFU images ({version})"
    )
    parser.add_argument("--scene", help="YAML scene file name", type=str)
    parser.add_argument("--grating", help="Grating name", type=str)
    parser.add_argument("--scale", help="Scale", type=str)
    parser.add_argument("--ra_teles_deg", help="Telescope central RA (deg)", type=float, default=0.0)
    parser.add_argument("--dec_teles_deg", help="Telescope central DEC (deg)", type=float, default=0.0)
    parser.add_argument("--delta_ra_teles_arcsec", help="Offset in RA (arcsec)", type=float, default=0.0)
    parser.add_argument("--delta_dec_teles_arcsec", help="Offset in DEC (arcsec)", type=float, default=0.0)
    parser.add_argument("--seeing_fwhm_arcsec", help="Seeing FWHM (arcsec)", type=float, default=0.0)
    parser.add_argument("--seeing_psf", help="Seeing PSF", type=str, default="gaussian",
                        choices=["gaussian"])
    parser.add_argument("--instrument_pa_deg", help="Instrument Position Angle (deg)", type=float, default=0.0)
    parser.add_argument("--airmass", help="Airmass", type=float, default=1.0)
    parser.add_argument("--parallactic_angle_deg", help="Parallactic angle (deg)", type=float, default=0.0)
    parser.add_argument("--noversampling_whitelight", help="Oversampling white light image", type=int, default=10)
    parser.add_argument("--atmosphere_transmission", help="Atmosphere transmission", type=str, default="default",
                        choices=["default", "none"])
    parser.add_argument("--bias", help="Bias level (ADU)", type=int, default=0)
    parser.add_argument("--rnoise", help="Readout noise standard deviation (ADU)", type=float, default=0)
    parser.add_argument("--flatpix2pix", help="Pixel-to-pixel flat field", type=str, default="default",
                        choices=["default", "none"])
    parser.add_argument("--spectral_blurring_pixel",
                        help="Spectral blurring when converting the original 3D data cube to the original 2D RSS " + \
                             "(in pixel units)",
                        type=float, default=1.0)
    parser.add_argument("--seed", help="Seed for random number generator", type=int, default=None)
    parser.add_argument("--noparallel", help="Do not use parallel processing", action="store_true")
    parser.add_argument("--prefix_intermediate_FITS", help="Prefix for intermediate FITS files", type=str,
                        default="test")
    parser.add_argument("--stop_after_ifu_3D_method0", help="Stop after computing ifu_3D_method0 image",
                        action="store_true")
    parser.add_argument("-v", "--verbose", help="increase program verbosity", action="store_true")
    parser.add_argument("--plots", help="Plot intermediate results", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")
    parser.add_argument("--version", help="Display version", action="store_true")
    args = parser.parse_args(args=args)

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    if args.version:
        print(version)
        raise SystemExit()

    if args.verbose:
        for arg, value in vars(args).items():
            print(f'--{arg} {value}')

    if args.echo:
        print('\033[1m\033[31m% ' + ' '.join(sys.argv) + '\033[0m\n')

    print(f"Welcome to fridadrp-ifu_simulator\nversion {version}\n")

    if args.scale is None:
        raise ValueError(f'You must specify --scale from\n{FRIDA_VALID_SPATIAL_SCALES}')
    elif args.scale.upper() not in FRIDA_VALID_SPATIAL_SCALES:
        raise ValueError(f'Invalid scale: {args.scale}. It must be one of {FRIDA_VALID_SPATIAL_SCALES}')
    scale = args.scale.upper()

    if args.grating is None:
        raise ValueError(f'You must specify --grating from\n{FRIDA_VALID_GRATINGS}')
    elif args.grating.upper() not in FRIDA_VALID_GRATINGS:
        raise ValueError(f'Invalid grating: {args.grating}. It must be one of {FRIDA_VALID_GRATINGS}')
    grating = args.grating.upper()

    # keywords that should be included in the FITS header
    header_keys = fits.Header()
    header_keys['OBSERVAT'] = ('ORM', 'Name of the observatory (IRAF style)')
    header_keys['TELESCOP'] = ('GTC','Telescope name')
    header_keys['ORIGIN'] = ('fridadrp-ifu_simulator', 'FITS file originator')
    header_keys['LATITUDE'] = ('+28:45:43.2', 'Telescope latitude (degrees), +28:45:43.2')
    header_keys['LONGITUD'] = ('+17:52:39.5', 'Telescope longitude (degrees), +17:52:39.5')
    header_keys['HEIGHT'] = (2348, 'Telescope height above sea level (m)')
    header_keys['AIRMASS'] = (args.airmass, 'Airmass')
    header_keys['IPA'] =  (args.instrument_pa_deg, 'Instrument position angle (degrees)')
    header_keys['PARANGLE'] = (args.parallactic_angle_deg, 'Parallactic angle (degrees)')
    header_keys['INSTRUME'] = ('FRIDA', 'Instrument name')
    header_keys['OBSMODE'] = ('IFS', 'Observation mode' )
    header_keys['SCALE'] = (f'{scale}', 'Camera scale')
    header_keys['GRATING'] = (f'{grating}', 'Grating')
    header_keys['HISTORY'] = '-' * 25
    header_keys['HISTORY'] = f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    header_keys['HISTORY'] = '-' * 25
    header_keys['HISTORY'] = f'Node: {platform.uname().node}'
    header_keys['HISTORY'] = f'Python: {sys.executable}'
    header_keys['HISTORY'] = f'$ fridadrp-ifu_simulator'
    header_keys['HISTORY'] = f'(version: {version})'
    for arg, value in vars(args).items():
        header_keys['HISTORY'] = f'--{arg} {value}'

    # simplify additional argument names
    scene = args.scene
    if scene is None:
        raise ValueError(f'Scene file name has not been specified!')
        
    seeing_fwhm_arcsec = args.seeing_fwhm_arcsec * u.arcsec
    if seeing_fwhm_arcsec.value < 0:
        raise ValueError(f'Unexpected {seeing_fwhm_arcsec=}. This number must be >= 0.')
    
    seeing_psf = args.seeing_psf
    
    airmass = args.airmass
    if airmass < 1.0:
        raise ValueError(f'Unexpected {airmass=}. This number must be greater than or equal to 1.0')
    
    parallactic_angle = args.parallactic_angle_deg * u.deg
    if abs(parallactic_angle.value) > 90:
        raise ValueError(f'Unexpected {parallactic_angle.value}. This number must be within the range [-90, +90]')
    if (parallactic_angle.value != 0) and (airmass == 1):
        raise ValueError(f'{parallactic_angle=} has no meaning when {airmass=}')
    
    noversampling_whitelight = args.noversampling_whitelight
    if noversampling_whitelight < 1:
        raise ValueError(f'Unexpected {noversampling_whitelight=} (must be > 1)')
    
    atmosphere_transmission = args.atmosphere_transmission
    
    bias = args.bias
    if bias < 0:
        raise ValueError(f'Invalid bias value: {bias}. It must be >= 0')
    bias *= u.adu
    
    rnoise = args.rnoise
    if rnoise < 0:
        raise ValueError(f'Invalid readout noise value: {rnoise}')
    rnoise *= u.adu
    
    # avoid negative values in the output
    if bias.value < 10*rnoise.value:
        raise ValueError(f'Invalid {bias=} value. It must be >= 10*{rnoise=} to avoid negative values')
    
    flatpix2pix = args.flatpix2pix
    
    spectral_blurring_pixel = args.spectral_blurring_pixel * u.pix
    if spectral_blurring_pixel.value < 0:
        raise ValueError(f'Invalid {spectral_blurring_pixel=}')
    
    prefix_intermediate_fits = args.prefix_intermediate_FITS
    
    seed = args.seed
    
    noparallel_computation = args.noparallel
    
    stop_after_ifu_3D_method0 = args.stop_after_ifu_3D_method0
    
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
    header_keys['RA'] = (
        skycoord_center.ra.to_string(unit=u.hour, sep=':', precision=3, pad=True),
        'Telescope right ascension (HH:MM:SS)'
    )
    header_keys['DEC'] = (
        skycoord_center.dec.to_string(unit=u.deg, sep=':', precision=3, pad=True),
        'Telescope declination (DD:MM:SS)'
    )
    header_keys['RADEG'] = (
        skycoord_center.ra.to(u.deg).value,
        'Telescope right ascension (degrees)'
    )
    header_keys['DECDEG'] = (
        skycoord_center.dec.to(u.deg).value,
        'Telescope declination (degrees)'
    )

    # linear wavelength calibration
    wv_lincal = LinearWaveCalFRIDA(grating=grating)
    if verbose:
        print(f'\n{wv_lincal}')

    # instrument Position Angle
    instrument_pa = args.instrument_pa_deg * u.deg

    # define WCS object to store the spatial 2D WCS
    # and the linear wavelength calibration
    wcs3d = define_3d_wcs(
        naxis1_ifu=FRIDA_NAXIS1_IFU,
        naxis2_ifu=FRIDA_NAXIS2_IFU,
        skycoord_center=skycoord_center,
        spatial_scale=FRIDA_SPATIAL_SCALE[scale],
        wv_lincal=wv_lincal,
        instrument_pa=instrument_pa,
        verbose=verbose
    )

    # initialize random number generator with provided seed
    rng = np.random.default_rng(seed)

    ifu_simulator(
        wcs3d=wcs3d,
        header_keys=header_keys,
        naxis1_detector=FRIDA_NAXIS1_HAWAII,
        naxis2_detector=FRIDA_NAXIS2_HAWAII,
        nslices=FRIDA_NSLICES,
        noversampling_whitelight=noversampling_whitelight,
        scene_fname=scene,
        seeing_fwhm_arcsec=seeing_fwhm_arcsec,
        seeing_psf=seeing_psf,
        instrument_pa=instrument_pa,
        airmass=airmass,
        parallactic_angle=parallactic_angle,
        flatpix2pix=flatpix2pix,
        atmosphere_transmission=atmosphere_transmission,
        bias=bias,
        rnoise=rnoise,
        spectral_blurring_pixel=spectral_blurring_pixel,
        faux_dict=faux_dict,
        rng=rng,
        noparallel_computation=noparallel_computation,
        prefix_intermediate_fits=prefix_intermediate_fits,
        stop_after_ifu_3D_method0=stop_after_ifu_3D_method0,
        verbose=verbose,
        instname='FRIDA',
        subtitle=f'scale: {scale}, grating: {grating}',
        plots=plots
    )

    print('Program stopped')


if __name__ == "__main__":

    main()
