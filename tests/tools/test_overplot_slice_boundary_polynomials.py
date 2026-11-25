"""The PDF mosaic of the slices is saved without pyplot"""

import logging

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import pytest

from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import sliceid_from_sliceindex
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_fitted_boundary_polynomials
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_traces_within_slice_boundary_polynomials_mosaic
from fridadrp.tools.overplot_slice_boundary_polynomials import polynomial_from_voffset
import fridadrp.tools.overplot_slice_boundary_polynomials as overplot_module


def test_mosaic_pdf(tmp_path):
    rng = np.random.default_rng(1)
    image = rng.normal(100.0, 10.0, size=(100, 100))
    list_poly_left = [Polynomial([islice * 68.0 + 5]) for islice in range(FRIDA_NSLICES)]
    list_poly_right = [Polynomial([islice * 68.0 + 65]) for islice in range(FRIDA_NSLICES)]
    traces = [[Polynomial([islice * 68.0 + 35])] for islice in range(FRIDA_NSLICES)]
    pdf_output = tmp_path / "mosaic.pdf"
    before = plt.get_fignums()
    plot_traces_within_slice_boundary_polynomials_mosaic(
        pdf_output,
        montage=None,
        image_data=image,
        title="mosaic",
        list_poly_left=list_poly_left,
        list_poly_right=list_poly_right,
        list_poly_traces_all_slices=traces,
        show_traceid=True,
    )
    assert pdf_output.stat().st_size > 0
    assert plt.get_fignums() == before


@pytest.mark.parametrize("plot_left_boundary", [True, False])
@pytest.mark.parametrize("plot_right_boundary", [True, False])
def test_plot_fitted_boundary_polynomials_selected_boundaries(plot_left_boundary, plot_right_boundary):
    # Constant boundaries (0-based array indices along NAXIS2)
    list_poly_left = [Polynomial([islice * 68.0 + 5]) for islice in range(FRIDA_NSLICES)]
    list_poly_right = [Polynomial([islice * 68.0 + 65]) for islice in range(FRIDA_NSLICES)]
    islice = 14
    fig = Figure()
    ax = fig.add_subplot()
    ax.set_xlim(-0.5, 2047.5)
    ax.set_ylim(-0.5, 2047.5)
    plot_fitted_boundary_polynomials(
        ax,
        list_poly_left,
        list_poly_right,
        show_sliceid=True,
        isliceplot=islice,
        plot_left_boundary=plot_left_boundary,
        plot_right_boundary=plot_right_boundary,
    )
    # Each boundary is drawn as two curves (white outline + colored line)
    yplotted = sorted({line.get_ydata()[0] for line in ax.lines})
    yexpected = []
    if plot_left_boundary:
        yexpected.append(list_poly_left[islice].coef[0])
    if plot_right_boundary:
        yexpected.append(list_poly_right[islice].coef[0])
    assert len(ax.lines) == 2 * len(yexpected)
    assert yplotted == yexpected
    # The slice ID is displayed even when no boundary is plotted
    assert [text.get_text() for text in ax.texts] == [f"id#{sliceid_from_sliceindex(islice)}"]


@pytest.mark.parametrize("voffset", [[0.0], [0.0, 0.0], [0.0, 0.0, 0.0]])
def test_polynomial_from_voffset_all_zero(voffset, monkeypatch):
    # No fit must be performed when all the offsets are zero
    def fail(*args, **kwargs):
        raise AssertionError("polfit_residuals should not be called")

    monkeypatch.setattr(overplot_module, "polfit_residuals", fail)
    poly = polynomial_from_voffset(voffset)
    assert isinstance(poly, Polynomial)
    assert np.all(poly.coef == 0)
    assert np.all(poly(np.arange(2048)) == 0)


@pytest.mark.parametrize("voffset", [[2.5], [1.0, -3.0], [1.0, -2.0, 3.0], [0.0, 1.5, 0.0, -1.0]])
def test_polynomial_from_voffset_reproduces_offsets(voffset, caplog):
    # Avoid the interactive debug plot (enabled when the logger is at DEBUG level)
    caplog.set_level(logging.INFO, logger=overplot_module.__name__)
    poly = polynomial_from_voffset(voffset)
    assert isinstance(poly, Polynomial)
    assert poly.degree() <= len(voffset) - 1
    # The offsets are evenly spaced along the useful NAXIS1 region (0-based array indices),
    # and the polynomial of degree len(voffset)-1 interpolates them exactly
    xnodes = np.linspace(
        FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL.value - 1,
        FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL.value - 1,
        len(voffset),
    )
    assert np.allclose(poly(xnodes), voffset, rtol=0, atol=1e-8)
    assert "Fitting a polynomial of degree" in caplog.text


def test_polynomial_from_voffset_empty():
    with pytest.raises(ValueError, match="at least one value"):
        polynomial_from_voffset([])


@pytest.mark.parametrize(
    "debugplot, loglevel, expected",
    [(None, logging.INFO, 0), (None, logging.DEBUG, 2), (-2, logging.INFO, -2), (0, logging.DEBUG, 0)],
)
def test_polynomial_from_voffset_debugplot(debugplot, loglevel, expected, monkeypatch, caplog):
    # Capture the debugplot value passed to polfit_residuals (no plot is generated)
    captured = {}

    def fake_polfit_residuals(*args, **kwargs):
        captured["debugplot"] = kwargs["debugplot"]
        return Polynomial([1.0, 0.0]), None

    monkeypatch.setattr(overplot_module, "polfit_residuals", fake_polfit_residuals)
    caplog.set_level(loglevel, logger=overplot_module.__name__)
    polynomial_from_voffset([1.0, 1.0], debugplot=debugplot)
    assert captured["debugplot"] == expected
