#
# Copyright 2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.io import fits
import numpy as np


def save_image2d_detector_method0(
        image2d_detector_method0,
        prefix_intermediate_fits,
        instname=None,
):
    """Save the two 2D images: RSS and detector.

    Parameters
    ----------
    image2d_detector_method0 : `~numpy.ndarray`
        2D array containing the detector image.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    instname : str
        Instrument name
    """

    if len(prefix_intermediate_fits) > 0:
        # --------------------------------------
        # spectroscopic 2D image in the detector
        # --------------------------------------
        hdu = fits.PrimaryHDU(image2d_detector_method0.astype(np.float32))
        if instname is not None:
            hdu.header['INSTRUME'] = instname
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_detector_2D_method0.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(outfile, overwrite='yes')
