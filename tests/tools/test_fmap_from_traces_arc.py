#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Tests for fmap_from_traces_arc"""

from astropy.table import Table
from matplotlib.figure import Figure
import numpy as np
from numpy.polynomial import Polynomial
import pytest

from fridadrp.core import FRIDA_NSLICES
from fridadrp.tools.fmap_from_traces_arc import compute_fmap_from_traces_arc
from fridadrp.tools.fmap_from_traces_arc import fit_arc_line_polynomials
from fridadrp.tools.fmap_from_traces_arc import plot_arc_line_polynomials
from fridadrp.tools.fmap_from_traces_arc import plot_arc_lines


def test_plot_arc_lines():
    list_tables = [None] * FRIDA_NSLICES
    list_tables[0] = Table({"row_index": [10, 11], "x_peak": [100.5, 101.5]})
    list_tables[1] = Table({"row_index": np.array([], dtype=int), "x_peak": np.array([], dtype=float)})
    list_tables[5] = Table({"row_index": [80], "x_peak": [300.25]})
    ax = Figure().subplots()
    line = plot_arc_lines(ax, list_tables, color="red")
    # A single line with the points of all the slices (0-based array indices)
    assert list(ax.lines) == [line]
    assert np.allclose(line.get_xdata(), [100.5, 101.5, 300.25])
    assert np.allclose(line.get_ydata(), [10, 11, 80])
    assert line.get_linestyle() == "None"
    assert line.get_color() == "red"


def test_plot_arc_lines_without_detections():
    list_tables = [None] * FRIDA_NSLICES
    list_tables[3] = Table({"row_index": np.array([], dtype=int), "x_peak": np.array([], dtype=float)})
    ax = Figure().subplots()
    assert plot_arc_lines(ax, list_tables) is None
    assert len(ax.lines) == 0


# Synthetic arc lines (x_peak as a function of row_index, 0-based array indices)
POLY_LINE1 = Polynomial([300.0, 0.05, 1.0e-4])
POLY_LINE2 = Polynomial([700.0, -0.02, 2.0e-4])


def _synthetic_arc_line_table():
    """Table with the structure of read_arc_lines: two lines, a short one and a NaN value."""
    rows1 = np.arange(100, 160)
    rows2 = np.arange(102, 158)
    x2 = POLY_LINE2(rows2.astype(float))
    x2[10] = np.nan  # failed fit in one row (it must be ignored)
    feature_id = np.concatenate([np.full(rows1.size, 1), np.full(rows2.size, 2), [3, 3]])
    row_index = np.concatenate([rows1, rows2, [120, 121]])
    x_peak = np.concatenate([POLY_LINE1(rows1.astype(float)), x2, [900.0, 900.5]])
    return Table({"feature_id": feature_id, "row_index": row_index, "x_peak": x_peak})


def test_fit_arc_line_polynomials():
    list_fits = fit_arc_line_polynomials(_synthetic_arc_line_table(), deg=2)
    # Line 3 has only 2 points (< deg + 1) and is skipped
    assert [fit["feature_id"] for fit in list_fits] == [1, 2]
    fit1, fit2 = list_fits
    assert np.allclose(fit1["poly"].coef, POLY_LINE1.coef)
    assert np.allclose(fit2["poly"].coef, POLY_LINE2.coef)
    assert (fit1["row_min"], fit1["row_max"], fit1["npoints"]) == (100.0, 159.0, 60)
    # The row with NaN x_peak is not fitted
    assert (fit2["row_min"], fit2["row_max"], fit2["npoints"]) == (102.0, 157.0, 55)
    assert fit1["rms"] < 1e-8 and fit2["rms"] < 1e-8


def test_fit_arc_line_polynomials_degree_zero():
    # With deg=0 the two-point line is also fitted (constant = mean of x_peak)
    list_fits = fit_arc_line_polynomials(_synthetic_arc_line_table(), deg=0)
    assert [fit["feature_id"] for fit in list_fits] == [1, 2, 3]
    assert np.allclose(list_fits[2]["poly"].coef, [900.25])


def test_plot_arc_line_polynomials():
    list_fits = fit_arc_line_polynomials(_synthetic_arc_line_table(), deg=2)
    ax = Figure().subplots()
    list_lines = plot_arc_line_polynomials(ax, list_fits, color="yellow")
    assert list(ax.lines) == list_lines
    assert len(list_lines) == 2
    for line, fit in zip(list_lines, list_fits):
        ydata = line.get_ydata()
        # The polynomial gives the NAXIS1 position (x) as a function of the NAXIS2 position (y)
        assert np.allclose(line.get_xdata(), fit["poly"](ydata))
        assert (ydata[0], ydata[-1]) == (fit["row_min"], fit["row_max"])
        assert line.get_color() == "yellow"


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"deg": -1}, "deg"),
        ({"slicenum_list": [1], "sliceid_list": [1]}, "Cannot specify both"),
    ],
)
def test_compute_fmap_from_traces_arc_invalid_parameters(kwargs, match):
    # The parameters are checked before reading the input files
    with pytest.raises(ValueError, match=match):
        compute_fmap_from_traces_arc("missing_arc.fits", "missing_lines.fits", "missing_traces.fits", **kwargs)
