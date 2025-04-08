#
# Copyright 2023-2024 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
import astropy.units as u
import json
import numpy as np

from numina.core import Result
from numina.core import Parameter
from numina.core.requirements import ObservationResultRequirement
from numina.core.recipes import BaseRecipe
from numina.instrument.simulation.ifu.compute_image2d_rss_from_detector_method1 import compute_image2d_rss_from_detector_method1
from numina.instrument.simulation.ifu.compute_image3d_ifu_from_rss_method1 import compute_image3d_ifu_from_rss_method1
from numina.instrument.simulation.ifu.define_3d_wcs import define_3d_wcs, get_wvparam_from_wcs3d
from numina.util.context import manage_fits

from fridadrp.instrument.define_auxiliary_files import define_auxiliary_files
from fridadrp.processing.linear_wavelength_calibration_frida import LinearWaveCalFRIDA
from fridadrp.products import FridaFrame, FridaRSSFrame, Frida3DFrame

from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NAXIS1_IFU
from fridadrp.core import FRIDA_NAXIS2_IFU
from fridadrp.core import FRIDA_SPATIAL_SCALE
from fridadrp.core import FRIDA_NSLICES


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

    # Recipe results
    reduced_image = Result(FridaFrame)
    reduced_rss = Result(FridaRSSFrame)
    reduced_3d = Result(Frida3DFrame)

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

        # ------------
        # Target - Sky
        # ------------
        # Perform combination of Target and Sky frames, and compute subtraction between them
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
        
        # Prepare output result
        header = list_of[0][0].header
        hdu_subtraction = fits.PrimaryHDU(result, header)
        reduced_image = hdu_subtraction

        # --------------
        # Compute 2D RSS
        # --------------
        # Get information from header and define 2D WCS and linear wavelength calibration
        grating = header['GRATING'].upper()
        scale = header['SCALE']
        instrument_pa = float(header['IPA']) * u.deg
        radeg = float(header['RADEG']) * u.deg
        decdeg = float(header['DECDEG']) * u.deg
        skycoord_center = SkyCoord(
            ra=radeg,
            dec=decdeg,
            frame='icrs'
        )
        faux_dict = define_auxiliary_files(grating=grating, verbose=True)
        dict_ifu2detector = json.loads(open(faux_dict['model_ifu2detector'], mode='rt').read())
        wv_lincal = LinearWaveCalFRIDA(grating=grating)
        print(f'wv_lincal: {wv_lincal}')
        wcs3d = define_3d_wcs(
            naxis1_ifu=FRIDA_NAXIS1_IFU,
            naxis2_ifu=FRIDA_NAXIS2_IFU,
            skycoord_center=skycoord_center,
            spatial_scale=FRIDA_SPATIAL_SCALE[scale],
            wv_lincal=wv_lincal,
            instrument_pa=instrument_pa,
            verbose=False
        )
        
        # Generate 2D RSS
        image2d_rss_method1 = compute_image2d_rss_from_detector_method1(
            image2d_detector_method0=result,
            naxis1_detector=FRIDA_NAXIS1_HAWAII,
            naxis1_ifu=FRIDA_NAXIS1_IFU,
            nslices=FRIDA_NSLICES,
            dict_ifu2detector=dict_ifu2detector,
            wcs3d=wcs3d,
            noparallel_computation=True,
            verbose=False
        )

        # Prepare output result
        header = list_of[0][0].header
        hdu_rss = fits.PrimaryHDU(image2d_rss_method1, header)
        reduced_rss = hdu_rss

        # --------------------
        # Compute 3D IFU image
        # --------------------
        image3d_ifu_method1 = compute_image3d_ifu_from_rss_method1(
            image2d_rss_method1=image2d_rss_method1,
            naxis1_detector=FRIDA_NAXIS1_HAWAII,
            naxis2_ifu=FRIDA_NAXIS2_IFU,
            naxis1_ifu=FRIDA_NAXIS1_IFU,
            nslices=FRIDA_NSLICES,
            verbose=False
        )
        header = list_of[0][0].header
        hdu_rss = fits.PrimaryHDU(image3d_ifu_method1, header)
        reduced_3d = hdu_rss

        return self.create_result(
            reduced_image=reduced_image,
            reduced_rss=reduced_rss,
            reduced_3d=reduced_3d,
        )
