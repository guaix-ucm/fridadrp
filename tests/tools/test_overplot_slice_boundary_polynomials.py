"""The PDF mosaic of the slices is saved without pyplot"""

import logging
import types

from astropy.visualization import ZScaleInterval
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
import pytest

from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import sliceid_from_sliceindex
from fridadrp.tools.overplot_slice_boundary_polynomials import connect_image_display_keys
from fridadrp.tools.overplot_slice_boundary_polynomials import interactive_image_display
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


@pytest.fixture
def agg_pyplot():
    """Use the non-interactive Agg backend with pyplot, restoring the previous backend afterwards"""
    previous_backend = mpl.get_backend()
    plt.switch_backend("Agg")
    yield
    plt.close("all")
    try:
        plt.switch_backend(previous_backend)
    except ImportError:
        pass


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
def test_connect_image_display_keys(agg_pyplot, caplog):
    caplog.set_level(logging.INFO, logger=overplot_module.__name__)
    # Image with value 100 * row + column (0-based array indices)
    naxis2, naxis1 = 100, 100
    image = np.add.outer(100.0 * np.arange(naxis2), np.arange(naxis1))
    vmin, vmax = 0.0, image.max()
    with mpl.rc_context():  # the function modifies the global keymaps
        fig, ax = plt.subplots()
        ax.imshow(image, origin="lower", vmin=vmin, vmax=vmax)
        init_xlim, init_ylim = (-0.5, naxis1 - 0.5), (-0.5, naxis2 - 0.5)
        ax.set_xlim(init_xlim)
        ax.set_ylim(init_ylim)
        on_key = connect_image_display_keys(fig, ax, image, vmin, vmax)

        # Help message on startup, and conflicting default keymaps disabled
        assert "'q': quit" in caplog.text
        assert mpl.rcParams["keymap.home"] == []
        assert mpl.rcParams["keymap.back"] == []
        assert mpl.rcParams["keymap.forward"] == []

        # ',': min and max of the zoomed region (columns 10-20, rows 30-40)
        ax.set_xlim(10.5, 20.5)
        ax.set_ylim(30.5, 40.5)
        on_key(types.SimpleNamespace(key=","))
        assert ax.images[0].get_clim() == (100.0 * 30 + 10, 100.0 * 40 + 20)

        # '/': zscale of the zoomed region
        on_key(types.SimpleNamespace(key="/"))
        zscale_limits = ZScaleInterval().get_limits(image[30:41, 10:21])
        assert np.allclose(ax.images[0].get_clim(), zscale_limits)

        # 'h': initial axis limits, keeping the last display limits
        on_key(types.SimpleNamespace(key="h"))
        assert ax.get_xlim() == init_xlim
        assert ax.get_ylim() == init_ylim
        assert np.allclose(ax.images[0].get_clim(), zscale_limits)

        # 'a': toggle aspect (imshow sets aspect='equal' by default)
        on_key(types.SimpleNamespace(key="a"))
        assert ax.get_aspect() == "auto"
        on_key(types.SimpleNamespace(key="a"))
        assert ax.get_aspect() in ["equal", 1.0]

        # 'q': close the figure
        assert plt.fignum_exists(fig.number)
        on_key(types.SimpleNamespace(key="q"))
        assert not plt.fignum_exists(fig.number)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
def test_connect_image_display_keys_several_axes_ds9mode(agg_pyplot):
    # Two images with values 100 * row + column and twice that (0-based array indices),
    # displayed with 1-based axis coordinates (as teareduce.imshow with ds9mode=True)
    naxis2, naxis1 = 100, 100
    image1 = np.add.outer(100.0 * np.arange(naxis2), np.arange(naxis1))
    image2 = 2 * image1
    extent = [0.5, naxis1 + 0.5, 0.5, naxis2 + 0.5]
    with mpl.rc_context():  # the function modifies the global keymaps
        fig, axarr = plt.subplots(ncols=2, sharex=True, sharey=True)
        clim1, clim2 = (0.0, image1.max()), (0.0, image2.max())
        axarr[0].imshow(image1, origin="lower", extent=extent, vmin=clim1[0], vmax=clim1[1])
        axarr[1].imshow(image2, origin="lower", extent=extent, vmin=clim2[0], vmax=clim2[1])
        init_xlim, init_ylim = axarr[0].get_xlim(), axarr[0].get_ylim()
        on_key = connect_image_display_keys(
            fig, axarr, [image1, image2], [clim1[0], clim2[0]], [clim1[1], clim2[1]], ds9mode=True
        )

        # Zoom on FITS pixels 11-21 along NAXIS1 and 31-41 along NAXIS2,
        # i.e., array indices 10-20 (columns) and 30-40 (rows)
        axarr[0].set_xlim(11.5, 21.5)
        axarr[0].set_ylim(31.5, 41.5)

        # The key acts only on the image under the cursor
        on_key(types.SimpleNamespace(key=",", inaxes=axarr[1]))
        assert axarr[1].images[0].get_clim() == (2 * (100.0 * 30 + 10), 2 * (100.0 * 40 + 20))
        assert axarr[0].images[0].get_clim() == clim1
        on_key(types.SimpleNamespace(key=",", inaxes=axarr[0]))
        assert axarr[0].images[0].get_clim() == (100.0 * 30 + 10, 100.0 * 40 + 20)

        # Ignored when the cursor is not over any of the images
        on_key(types.SimpleNamespace(key="/", inaxes=None))
        assert axarr[0].images[0].get_clim() == (100.0 * 30 + 10, 100.0 * 40 + 20)

        # 'h': initial limits for all the axes, each image keeping its last display limits
        on_key(types.SimpleNamespace(key="h"))
        for ax in axarr:
            assert ax.get_xlim() == init_xlim
            assert ax.get_ylim() == init_ylim
        assert axarr[0].images[0].get_clim() == (100.0 * 30 + 10, 100.0 * 40 + 20)
        assert axarr[1].images[0].get_clim() == (2 * (100.0 * 30 + 10), 2 * (100.0 * 40 + 20))

        # 'a': all the axes change their aspect
        on_key(types.SimpleNamespace(key="a"))
        assert [ax.get_aspect() for ax in axarr] == ["auto", "auto"]


def test_connect_image_display_keys_inconsistent_lengths(agg_pyplot):
    image = np.zeros((10, 10))
    with mpl.rc_context():
        fig, axarr = plt.subplots(ncols=2)
        for ax in axarr:
            ax.imshow(image)
        with pytest.raises(ValueError, match="same length"):
            connect_image_display_keys(fig, axarr, [image], [0.0, 0.0], [1.0, 1.0])
        with pytest.raises(ValueError, match="must be sequences"):
            connect_image_display_keys(fig, axarr, [image, image], 0.0, 1.0)


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
def test_connect_image_display_keys_no_nested_event_loop(agg_pyplot, monkeypatch):
    # The handler runs inside the GUI event loop started by the waiting loop of
    # interactive_image_display; calling plt.pause() from the handler raises
    # this error with the Qt backend (Agg allows nesting, so it is simulated here)
    def fail_pause(interval):
        raise RuntimeError("Event loop already running")

    image = np.add.outer(100.0 * np.arange(100), np.arange(100))
    with mpl.rc_context():  # the function modifies the global keymaps
        fig, ax = plt.subplots()
        ax.imshow(image, origin="lower", vmin=0.0, vmax=image.max())
        on_key = connect_image_display_keys(fig, ax, image, 0.0, image.max())
        monkeypatch.setattr(overplot_module.plt, "pause", fail_pause)
        ax.set_xlim(10.5, 20.5)
        ax.set_ylim(30.5, 40.5)
        for key in [",", "/", "h", "a"]:
            on_key(types.SimpleNamespace(key=key))
        assert np.allclose(ax.images[0].get_clim(), ZScaleInterval().get_limits(image[30:41, 10:21]))


@pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")
def test_interactive_image_display_shows_window_once(agg_pyplot, monkeypatch):
    # plt.pause() calls show() at every call, which raises the window and steals the
    # focus from the terminal: the waiting loop must show the window only once
    def fail_pause(interval):
        raise AssertionError("plt.pause() should not be called in the waiting loop")

    monkeypatch.setattr(overplot_module.plt, "pause", fail_pause)
    image = np.zeros((10, 10))
    with mpl.rc_context():  # the function modifies the global keymaps
        fig, ax = plt.subplots()
        ax.imshow(image, vmin=0.0, vmax=1.0)
        calls = {"show": 0, "event_loop": 0}

        def fake_show():
            calls["show"] += 1

        def fake_start_event_loop(timeout):
            calls["event_loop"] += 1
            if calls["event_loop"] == 3:
                plt.close(fig)  # the user closes the window

        monkeypatch.setattr(fig, "show", fake_show)
        monkeypatch.setattr(fig.canvas, "start_event_loop", fake_start_event_loop)
        interactive_image_display(fig, ax, image, 0.0, 1.0)
    assert calls == {"show": 1, "event_loop": 3}
    assert not plt.fignum_exists(fig.number)
