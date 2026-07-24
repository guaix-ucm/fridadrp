#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

from fridadrp.core import sliceid_from_sliceindex
from fridadrp.core import sliceindex_from_sliceid


def test_slicenum_from_index():
    # Test valid slice indices
    for i in range(30):
        assert (
            sliceid_from_sliceindex(i)
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
        sliceid_from_sliceindex(-1)
        assert False
    except ValueError:
        pass

    try:
        sliceid_from_sliceindex(30)
        assert False
    except ValueError:
        pass


def test_sliceindex_from_slicenum():
    # Test valid slice numbers
    for i in range(1, 31):
        assert sliceindex_from_sliceid(i) == [
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
        sliceindex_from_sliceid(0)
        assert False
    except ValueError:
        pass

    try:
        sliceindex_from_sliceid(31)
        assert False
    except ValueError:
        pass
