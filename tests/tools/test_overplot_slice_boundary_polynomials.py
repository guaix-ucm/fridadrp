"""The PDF mosaic of the slices is saved without pyplot"""

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import pytest

from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import sliceid_from_sliceindex
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_fitted_boundary_polynomials
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_traces_within_slice_boundary_polynomials_mosaic


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
