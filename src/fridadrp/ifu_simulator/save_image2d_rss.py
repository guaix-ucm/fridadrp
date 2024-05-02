#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy import wcs
from astropy.io import fits
import astropy.units as u
import numpy as np

from fridadrp.processing.define_3d_wcs import get_wvparam_from_wcs3d


def save_image2d_rss(
        wcs3d,
        image2d_rss,
        method,
        prefix_intermediate_fits
):
    """Save the RSS image.

    Parameters
    ----------
    wcs3d : `~astropy.wcs.wcs.WCS`
        WCS of the data cube.
    image2d_rss : `~numpy.ndarray`
        2D array containing the RSS image.
    method : int
        Integer indicating the method: 0 or 1.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    """

    if len(prefix_intermediate_fits) > 0:
        # ------------------------------------------------
        # 1) spectroscopic 2D image with contiguous slices
        # ------------------------------------------------
        # ToDo: compute properly the parameters corresponding to the spatial axis
        # Note that using: wcs2d = wcs3d.sub(axes=[0, 1])
        # selecting the 1D spectral and one of the 1D spatial info of the 3D WCS
        # does not work:
        # "astropy.wcs._wcs.InconsistentAxisTypesError: ERROR 4 in wcs_types()
        #  Unmatched celestial axes."
        # For that reason we try a different approach:
        wv_cunit1, wv_crpix1, wv_crval1, wv_cdelt1 = get_wvparam_from_wcs3d(wcs3d)
        wcs2d = wcs.WCS(naxis=2)
        wcs2d.wcs.crpix = [wv_crpix1.value, 1]  # reference pixel coordinate
        wcs2d.wcs.crval = [wv_crval1.value, 0]  # world coordinate at reference pixel
        wcs2d.wcs.cdelt = [wv_cdelt1.value, 1]
        wcs2d.wcs.ctype = ["WAVE", ""]   # ToDo: fix this
        wcs2d.wcs.cunit = [wv_cunit1, u.pix]
        hdu = fits.PrimaryHDU(image2d_rss.astype(np.float32))
        hdu.header.extend(wcs2d.to_header(), update=True)
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_rss_2D_method{method}.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(outfile, overwrite='yes')
