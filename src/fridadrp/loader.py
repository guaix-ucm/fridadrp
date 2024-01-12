#
# Copyright 2023 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import numina.core


def drp_load():
    """Entry point to load FRIDA DRP."""
    return numina.core.drp_load('fridadrp', 'drp.yaml')
