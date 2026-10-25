"""The PDF mosaic of the slices is saved without pyplot"""

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial

from fridadrp.core import FRIDA_NSLICES
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
        traceid=True,
    )
    assert pdf_output.stat().st_size > 0
    assert plt.get_fignums() == before
