#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from fridadrp.core import slicenum_from_index
from fridadrp.core import sliceindex_from_num


def test_slicenum_from_index():
    # Test valid slice indices
    for i in range(30):
        assert (
            slicenum_from_index(i)
            == [
                30,
                1,
                29,
                2,
                28,
                3,
                27,
                4,
                26,
                5,
                25,
                6,
                24,
                7,
                23,
                8,
                22,
                9,
                21,
                10,
                20,
                11,
                19,
                12,
                18,
                13,
                17,
                14,
                16,
                15,
            ][i]
        )

    # Test invalid slice indices
    try:
        slicenum_from_index(-1)
        assert False
    except ValueError:
        pass

    try:
        slicenum_from_index(30)
        assert False
    except ValueError:
        pass


def test_sliceindex_from_slicenum():
    # Test valid slice numbers
    for i in range(1, 31):
        assert sliceindex_from_num(i) == [
            30,
            1,
            29,
            2,
            28,
            3,
            27,
            4,
            26,
            5,
            25,
            6,
            24,
            7,
            23,
            8,
            22,
            9,
            21,
            10,
            20,
            11,
            19,
            12,
            18,
            13,
            17,
            14,
            16,
            15,
        ].index(i)

    # Test invalid slice numbers
    try:
        sliceindex_from_num(0)
        assert False
    except ValueError:
        pass

    try:
        sliceindex_from_num(31)
        assert False
    except ValueError:
        pass
