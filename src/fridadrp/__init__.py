#
# Copyright 2025 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""The FRIDA Data Reduction Pipeline"""

import logging

from fridadrp._version import __version__  # noqa: F401


# Top level NullHandler
logging.getLogger("fridadrp").addHandler(logging.NullHandler())
