#
# Copyright 2023-2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import astropy.io.fits as fits
from numina.core import Result
from numina.core.requirements import ObservationResultRequirement
from numina.core.recipes import BaseRecipe
from numina.util.context import manage_fits

from fridadrp.products import FridaFrame


class Test1Recipe(BaseRecipe):
    """ This is a test recipe"""

    obresult = ObservationResultRequirement()

    reduced_image = Result(FridaFrame)

    def run(self, rinput):
        frames = rinput.obresult.frames
        with manage_fits(frames) as list_of:
            image1 = list_of[0]
            image2 = list_of[1]
            data = image1[0].data - image2[0].data
            header = image1[0].header
            hdu = fits.PrimaryHDU(data, header)
            reduced_image = hdu

        return self.create_result(reduced_image=reduced_image)
