#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Define a 3D WCS"""
# ToDo: this module should be moved to numina

from astropy.coordinates import SkyCoord
from astropy.io import fits
import astropy.units as u
from astropy.wcs import WCS


def define_3d_wcs(naxis1_ifu, naxis2_ifu, skycoord_center, spatial_scale, wv_lincal, verbose):
    """Define a 3D WCS.

    Parameters
    ----------
    naxis1_ifu : `~astropy.units.Quantity`
        NAXIS1 value of the WCS object (along slice).
    naxis2_ifu : `~astropy.units.Quantity`
        NAXIS2 value of the WCS object (perpendicular to the slice).
    skycoord_center : `~astropy.coordinates.sky_coordinate.SkyCoord`
        Coordinates at the center of the detector.
    spatial_scale : `~astropy.units.Quantity`
        Spatial scale per pixel.
    wv_lincal : `~fridadrp.processing.linear_wavelength_calibration_frida.LinearWaveCalFRIDA`
        Linear wavelength calibration object.
    verbose : bool
        If True, display additional information.

    Returns
    -------
    wcs : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.

    """

    # initial checks
    if not naxis1_ifu.unit.is_equivalent(u.pix):
        raise ValueError(f'Unexpected naxis1 unit: {naxis1_ifu.unit}')
    if not naxis2_ifu.unit.is_equivalent(u.pix):
        raise ValueError(f'Unexpected naxis2 unit: {naxis2_ifu.unit}')
    if not isinstance(skycoord_center, SkyCoord):
        raise ValueError(f'Expected SkyCoord instance not found: {skycoord_center} of type {type(skycoord_center)}')
    if not spatial_scale.unit.is_equivalent(u.deg / u.pix):
        raise ValueError(f'Unexpected spatial_scale unit: {spatial_scale.unit}')

    # define FITS header
    header = fits.Header()
    header['NAXIS'] = 3
    header['NAXIS1'] = int(naxis1_ifu.value)
    header['NAXIS2'] = int(naxis2_ifu.value)
    header['NAXIS3'] = int(wv_lincal.naxis1_wavecal.value)
    header['CTYPE1'] = 'RA---TAN'
    header['CTYPE2'] = 'DEC--TAN'
    header['CTYPE3'] = 'WAVE'
    header['CRVAL1'] = skycoord_center.ra.deg
    header['CRVAL2'] = skycoord_center.dec.deg
    header['CRVAL3'] = wv_lincal.crval1_wavecal.to('m').value
    header['CRPIX1'] = (naxis1_ifu.value + 1) / 2
    header['CRPIX2'] = (naxis2_ifu.value + 1) / 2
    header['CRPIX3'] = wv_lincal.crpix1_wavecal.value
    spatial_scale_deg_pix = spatial_scale.to(u.deg / u.pix).value
    header['CD1_1'] = -spatial_scale_deg_pix
    header['CD2_2'] = spatial_scale_deg_pix
    header['CD3_3'] = 1.0
    header['CUNIT1'] = 'deg'
    header['CUNIT2'] = 'deg'
    header['CUNIT3'] = 'm'

    # define wcs object
    wcs = WCS(header)
    if verbose:
        print(f'\n{wcs}')

    return wcs
