#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Tests for read_arc_lines"""

from astropy.io import fits
from astropy.table import Table
import numpy as np
import pytest

from fridadrp.core import FRIDA_NSLICES
from fridadrp.tools.read_arc_lines import read_arc_lines


def _write_arc_lines_file(filename, keycode="ARC_LINES", slices=None, drop_column=None):
    """Write a synthetic arc lines file, with the structure of the find_arc_lines output.

    slices is a dictionary {slice number (1-based): number of detections}.
    """
    if slices is None:
        slices = {3: 4, 7: 0}
    primary_hdu = fits.PrimaryHDU()
    if keycode is not None:
        primary_hdu.header["KEYCODE"] = keycode
    primary_hdu.header["IMAGE"] = "arc.fits"
    primary_hdu.header["UUID"] = "00000000-0000-0000-0000-000000000000"
    hdu_list = fits.HDUList([primary_hdu])
    for slicenum, nrows in slices.items():
        table = Table(
            {
                "feature_id": np.ones(nrows, dtype="i2"),
                "row_index": np.arange(nrows, dtype="i2") + 10 * slicenum,
                "x_peak": np.linspace(100.0, 200.0, nrows).astype("f4"),
            }
        )
        if drop_column is not None:
            table.remove_column(drop_column)
        hdu = fits.table_to_hdu(table)
        hdu.name = f"SLCNUM{slicenum:02d}"
        hdu_list.append(hdu)
    hdu_list.writeto(filename)


def test_read_arc_lines(tmp_path):
    filename = tmp_path / "arc_lines.fits"
    _write_arc_lines_file(filename)
    list_tables = read_arc_lines(str(filename))
    assert len(list_tables) == FRIDA_NSLICES
    # Slices not present in the file are None (index = slice number - 1)
    assert [islice for islice, table in enumerate(list_tables) if table is not None] == [2, 6]
    # Processed slice with detections
    assert len(list_tables[2]) == 4
    assert np.array_equal(list_tables[2]["row_index"], [30, 31, 32, 33])
    assert np.allclose(list_tables[2]["x_peak"], np.linspace(100.0, 200.0, 4))
    # Processed slice without detections: empty table
    assert len(list_tables[6]) == 0


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"keycode": None}, "KEYCODE"),
        ({"keycode": "SLICE_TRACES_POLYNOMIALS"}, "Invalid KEYCODE"),
        ({"slices": {}}, "SLCNUMnn"),
        ({"drop_column": "x_peak"}, "x_peak"),
        ({"drop_column": "row_index"}, "row_index"),
    ],
)
def test_read_arc_lines_invalid_file(tmp_path, kwargs, match):
    filename = tmp_path / "arc_lines.fits"
    _write_arc_lines_file(filename, **kwargs)
    with pytest.raises(ValueError, match=match):
        read_arc_lines(str(filename))
