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
        header_keys,
        image2d_detector_method0,
        prefix_intermediate_fits
):
    """Save the two 2D images: RSS and detector.

    Parameters
    ----------
    header_keys : `~astropy.io.fits.header.Header`
        FITS header with additional keywords to be merged together with
        the default keywords.
    image2d_detector_method0 : `~numpy.ndarray`
        2D array containing the detector image.
    prefix_intermediate_fits : str
        Prefix for output intermediate FITS files. If the length of
        this string is 0, no output is generated.
    """

    if len(prefix_intermediate_fits) > 0:
        # --------------------------------------
        # spectroscopic 2D image in the detector
        # --------------------------------------
        hdu = fits.PrimaryHDU(image2d_detector_method0.astype(np.float32))
        pos0 = len(hdu.header) - 1
        hdu.header.update(header_keys)
        hdu.header.insert(
            pos0, ('COMMENT', "FITS (Flexible Image Transport System) format is defined in 'Astronomy"))
        hdu.header.insert(
            pos0 + 1, ('COMMENT', "and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"))
        hdul = fits.HDUList([hdu])
        outfile = f'{prefix_intermediate_fits}_detector_2D_method0.fits'
        print(f'Saving file: {outfile}')
        hdul.writeto(outfile, overwrite='yes')
