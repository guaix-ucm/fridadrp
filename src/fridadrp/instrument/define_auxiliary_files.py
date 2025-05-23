#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import json
import pooch


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
    base_url = 'https://guaix.fis.ucm.es/~ncl/fridadrp_simulator_data'
    # note: compute md5 hash from terminal using:
    # linux $ md5sum <filename>
    # macOS $ md5 <filename>
    fconf = pooch.retrieve(
        f'{base_url}/configuration_FRIDA_IFU_simulator.json',
        known_hash='md5:981684f1920fac9835a6067782b1b531',
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
