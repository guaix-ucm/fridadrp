#
# Copyright 2024-2025 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from numina.core import DataFrameType


class FridaFrame(DataFrameType):
    pass


class FridaRSSFrame(FridaFrame):
    """Frida RSS frame"""
    pass


class Frida3DFrame(FridaFrame):
    """Frida 3D frame"""
    pass
