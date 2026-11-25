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

from fridadrp.core import FRIDA_NSLICES
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
