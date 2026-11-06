#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

import numpy as np
import pytest

from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import DEF_SLICEID_FROM_SLICEINDEX
from fridadrp.core import DEF_SLICEINDEX_FROM_SLICEID
from fridadrp.core import sliceid_from_sliceindex
from fridadrp.core import sliceindex_from_sliceid
from fridadrp.core import slicenum_list_from_slice_lists

# Expected slice ID (1-30) for each slice index (0-29)
EXPECTED_SLICEID_FROM_SLICEINDEX = np.array(
    [1, 30, 2, 29, 3, 28, 4, 27, 5, 26, 6, 25, 7, 24, 8, 23, 9, 22, 10, 21, 11, 20, 12, 19, 13, 18, 14, 17, 15, 16]
)

# Expected slice index (0-29) for each slice ID (1-30); element 0 is a dummy
# value (-1) for the non-existent slice ID 0
EXPECTED_SLICEINDEX_FROM_SLICEID = np.array(
    [-1, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1]
)


def test_slicenum_from_index():
    # Test valid slice indices
    for i in range(30):
        assert sliceid_from_sliceindex(i) == EXPECTED_SLICEID_FROM_SLICEINDEX[i]

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
        assert sliceindex_from_sliceid(i) == EXPECTED_SLICEINDEX_FROM_SLICEID[i]

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


def test_sliceid_from_sliceindex_values():
    assert DEF_SLICEID_FROM_SLICEINDEX.shape == (FRIDA_NSLICES,)
    assert np.issubdtype(DEF_SLICEID_FROM_SLICEINDEX.dtype, np.integer)
    np.testing.assert_array_equal(DEF_SLICEID_FROM_SLICEINDEX, EXPECTED_SLICEID_FROM_SLICEINDEX)


def test_sliceindex_from_sliceid_values():
    assert DEF_SLICEINDEX_FROM_SLICEID.shape == (FRIDA_NSLICES + 1,)
    assert np.issubdtype(DEF_SLICEINDEX_FROM_SLICEID.dtype, np.integer)
    # element 0 is the dummy value -1
    np.testing.assert_array_equal(DEF_SLICEINDEX_FROM_SLICEID, EXPECTED_SLICEINDEX_FROM_SLICEID)


def test_slice_arrays_are_mutual_inverses():
    # DEF_SLICEID_FROM_SLICEINDEX must be a permutation of 1..NSLICES
    np.testing.assert_array_equal(np.sort(DEF_SLICEID_FROM_SLICEINDEX), np.arange(1, FRIDA_NSLICES + 1))
    # index -> id -> index
    np.testing.assert_array_equal(DEF_SLICEINDEX_FROM_SLICEID[DEF_SLICEID_FROM_SLICEINDEX], np.arange(FRIDA_NSLICES))
    # id -> index -> id
    np.testing.assert_array_equal(
        DEF_SLICEID_FROM_SLICEINDEX[DEF_SLICEINDEX_FROM_SLICEID[1:]], np.arange(1, FRIDA_NSLICES + 1)
    )


@pytest.mark.parametrize(
    "slicenum_list, sliceid_list, expected",
    [
        ([0], [0], list(range(1, FRIDA_NSLICES + 1))),
        ([3, 7], [0], [3, 7]),
        ([7, 3], [0], [7, 3]),
        ([0], [1, 30, 16], [1, 2, 30]),
    ],
)
def test_slicenum_list_from_slice_lists_valid(slicenum_list, sliceid_list, expected):
    assert slicenum_list_from_slice_lists(slicenum_list, sliceid_list) == expected


def test_slicenum_list_from_slice_lists_all_sliceids():
    # slice ID -> slice number must reproduce slice index + 1 for every slice
    sliceid_list = [int(sliceid) for sliceid in DEF_SLICEID_FROM_SLICEINDEX]
    result = slicenum_list_from_slice_lists([0], sliceid_list)
    assert result == list(range(1, FRIDA_NSLICES + 1))


def test_slicenum_list_from_slice_lists_return_type():
    # numpy integers in the input are returned as Python int
    result = slicenum_list_from_slice_lists([np.int64(3), np.int32(5)], [0])
    assert result == [3, 5]
    assert all(type(slicenum) is int for slicenum in result)
    result = slicenum_list_from_slice_lists([0], [np.int64(30)])
    assert result == [2]
    assert all(type(slicenum) is int for slicenum in result)


def test_slicenum_list_from_slice_lists_returns_copy():
    slicenum_list = [4, 6]
    result = slicenum_list_from_slice_lists(slicenum_list, [0])
    assert result is not slicenum_list
    result.append(8)
    assert slicenum_list == [4, 6]


@pytest.mark.parametrize(
    "slicenum_list, sliceid_list",
    [
        ((0,), [0]),
        ([0], (0,)),
        (np.array([0]), [0]),
        ([0], np.array([0])),
        ([3.0], [0]),
        ([0], [3.0]),
        (["3"], [0]),
        ([0], ["3"]),
        ([True], [0]),
        ([0], [False]),
    ],
)
def test_slicenum_list_from_slice_lists_type_error(slicenum_list, sliceid_list):
    with pytest.raises(TypeError):
        slicenum_list_from_slice_lists(slicenum_list, sliceid_list)


@pytest.mark.parametrize(
    "slicenum_list, sliceid_list",
    [
        ([], [0]),  # empty list
        ([0], []),
        ([-1], [0]),  # out of range
        ([FRIDA_NSLICES + 1], [0]),
        ([0], [-1]),
        ([0], [FRIDA_NSLICES + 1]),
        ([5, 5], [0]),  # duplicates
        ([0], [5, 5]),
        ([0, 3], [0]),  # 0 together with other values
        ([0], [0, 3]),
        ([3], [4]),  # both lists provided
    ],
)
def test_slicenum_list_from_slice_lists_value_error(slicenum_list, sliceid_list):
    with pytest.raises(ValueError):
        slicenum_list_from_slice_lists(slicenum_list, sliceid_list)
