#
# Copyright 2023 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from numina.core.recipes import BaseRecipe


class WaveTestRecipe(BaseRecipe):
    """ This is a test recipe"""
    def run(self, rinput):
        return self.create_result()
