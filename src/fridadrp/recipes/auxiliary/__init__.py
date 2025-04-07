#
# Copyright 2023-2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import astropy.io.fits as fits
import numpy as np

from numina.core import Result
from numina.core import Parameter
from numina.core.requirements import ObservationResultRequirement
from numina.core.recipes import BaseRecipe
from numina.util.context import manage_fits

from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII
from fridadrp.products import FridaFrame


class Test1Recipe(BaseRecipe):
    """Subtract two frames"""

    obresult = ObservationResultRequirement()

    reduced_image = Result(FridaFrame)

    def run(self, rinput):
        frames = rinput.obresult.frames
        nimages = len(frames)
        if nimages != 2:
            raise ValueError(f'Expected 2 images, got {nimages}')
        with manage_fits(frames) as list_of:
            image1 = list_of[0]
            image2 = list_of[1]
            data = image1[0].data - image2[0].data
            header = image1[0].header
            hdu = fits.PrimaryHDU(data, header)
            reduced_image = hdu

        return self.create_result(reduced_image=reduced_image)


class Test2Recipe(BaseRecipe):
    """Subtract sky from target frames"""

    obresult = ObservationResultRequirement()

    # Basic pattern: 'T': target, 'S': sky
    basic_pattern = Parameter(
        value='TS',
        description='Observation pattern of Target and Sky',
        choices=['TS', 'ST'],
        optional=False,
    )

    nexposures_before_moving = Parameter(
        value=1,
        description='Number of exposures at each T and S position before moving',
        optional=False,
    )

    # Combination method: do not allow 'sum' as combination method in order
    # to avoid confusion with number of counts in resulting image
    method = Parameter(
        'mean',
        description='Combination method',
        choices=['mean', 'median', 'sigmaclip'],
        optional=False,
    )
    method_kwargs = Parameter(
        value=dict(),
        description='Arguments for combination method',
        optional=True,
    )

    reduced_image = Result(FridaFrame)

    def run(self, rinput):
        frames = rinput.obresult.frames
        basic_pattern = rinput.basic_pattern
        nexposures_before_moving = rinput.nexposures_before_moving
        method = rinput.method
        method_kwargs = rinput.method_kwargs

        # Create the pattern by repeating each character in basic_pattern
        pattern = ''
        basic_pattern_length = len(basic_pattern)
        for i in range(basic_pattern_length):
            pattern += basic_pattern[i] * nexposures_before_moving
        print(f'Pattern: {pattern}')

        # Check pattern length
        pattern_length = len(pattern)
        if pattern_length != nexposures_before_moving * basic_pattern_length:
            raise ValueError(f'Unexpected mismatch: {pattern_length=} != {nexposures_before_moving * basic_pattern_length=}')

        # Check combination method
        if method != 'sigmaclip':
            if method_kwargs != {}:
                raise ValueError(f'Unexpected {method_kwargs=} for {method=}')

        # Check pattern sequence matches number of frames
        nimages = len(frames)
        nsequences = nimages // pattern_length
        if nsequences * pattern_length != nimages:
            raise ValueError(f'Expected {nsequences * pattern_length=} images, got {nimages=}')
        print(f'Number of sequences: {nsequences}')
        full_pattern = pattern * nsequences
        print(f'Full pattern: {full_pattern}')

        # Perform combination of Target and Sky frames, and compute subtraction
        # between them
        ntarget = full_pattern.count('T')
        nsky = full_pattern.count('S')
        print(f'Number of target frames: {ntarget}')
        print(f'Number of sky frames...: {nsky}')
        data3d_target = np.zeros(shape=(ntarget, FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), 
                                 dtype=np.float32)
        data3d_sky = np.zeros(shape=(nsky, FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), 
                              dtype=np.float32)
        with manage_fits(frames) as list_of:
            itarget = -1
            isky = -1
            for i in range(nimages):
                if full_pattern[i] == 'T':
                    itarget +=1
                    data3d_target[itarget, :, :] = list_of[i][0].data
                elif full_pattern[i] == 'S':
                    isky +=1
                    data3d_sky[isky, :, :] = list_of[i][0].data
        if itarget + 1 != ntarget:
            raise ValueError(f'Unexpected {itarget+1=} frames. It should be {ntarget=}')
        if isky + 1 != nsky:
            raise ValueError(f'Unexpected {isky+1=} frames. It should be {nsky=}')
        if method == 'mean':
            result = np.mean(data3d_target, axis=0) - np.mean(data3d_sky, axis=0)
        elif method == 'median':
            result = np.median(data3d_target, axis=0) - np.median(data3d_sky, axis=0)
        else:
            raise ValueError(f'Unexpected {method=}')
        
        # TODO: generate 2D RSS and 3D IFU frames
        
        # prepare output result
        header = list_of[0][0].header
        hdu = fits.PrimaryHDU(result, header)
        reduced_image = hdu

        return self.create_result(reduced_image=reduced_image)
