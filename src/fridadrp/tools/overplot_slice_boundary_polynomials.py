#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Overplot the slice boundary polynomials, borders and/or traces on image.

Since the FITS file defining the slice traces already contains
the slice boundary polynomials, --poly and --traces are mutually exclusive options.
If both are provided, and error will be raised.
"""

import argparse
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import logging
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from pathlib import Path
from rich_argparse import RichHelpFormatter
import shutil
import subprocess
import teareduce as tea
import types

from numina.array.display.polfit_residuals import polfit_residuals
from numina.tools.initialize_script_with_args import NuminaScriptDefinition
from numina.tools.input_number import input_number

from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import sliceid_from_sliceindex
from fridadrp.tools.read_slice_boundary_borders import read_slice_boundary_borders
from fridadrp.tools.read_slice_boundary_polynomials import read_slice_boundary_polynomials
from fridadrp.tools.read_slice_trace_polynomials import read_slice_trace_polynomials


def find_imagemagick_montage_executable():
    """Check if the ImageMagick 'montage' executable is available in the system PATH."""
    path = shutil.which("montage")
    if path is None:
        return None

    try:
        result = subprocess.run(
            [path, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None

    output = (result.stdout + result.stderr).lower()
    if "imagemagick" in output:
        return path
    return None


def plot_traces_within_slice_boundary_polynomials_mosaic(
    pdf_output,
    montage,
    image_data,
    title,
    list_poly_left=None,
    list_poly_right=None,
    array_left_border=None,
    array_right_border=None,
    ibad=None,
    list_poly_traces_all_slices=None,
    show_traceid=False,
    vlim=None,
):
    """Plot the traces within the slice boundary polynomials for every slice in a PDF mosaic

    Parameters
    ----------
    pdf_output : str or Path
        The path to the output PDF file where the mosaic will be saved.
    montage : str or None
        If provided, display a montage of the final plots with traces for every slice.
        Possible values are "png" or "pdf". Requires ImageMagick's montage.
        If None, no montage will be created.
    image_data : numpy.ndarray
        The image data to be displayed.
    title : str
        The title for the plot.
    list_poly_left : list of numpy.polynomial.Polynomial, optional
        The list of left slice boundary polynomials.
    list_poly_right : list of numpy.polynomial.Polynomial, optional
        The list of right slice boundary polynomials.
    array_left_border : numpy.ndarray, optional
        The array of left slice boundary borders (0-based indices).
    array_right_border : numpy.ndarray, optional
        The array of right slice boundary borders (0-based indices).
    ibad : list of int, optional
        List of indices of bad columns (to be ignored).
    list_poly_traces_all_slices : list of list of numpy.polynomial.Polynomial, optional
        The list of polynomials for each slice.
    show_traceid : bool, optional
        If True, overplot the trace ID at the center of each trace.
    vlim : tuple of float, optional
        Background and foreground limits for the plot (vmin, vmax).
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Saving final plots of traces for every slice in PDF file:\n[green]{Path(pdf_output).name}[/green]")
    pdf_out = PdfPages(pdf_output)

    if vlim is not None:
        vmin, vmax = vlim
    else:
        vmin, vmax = ZScaleInterval().get_limits(image_data)

    for islice in range(
        FRIDA_NSLICES - 1, -1, -1
    ):  # (0-based index) loop in reverse order to have the first slice on top of the PDF
        sliceid = sliceid_from_sliceindex(islice)
        # the figure is only saved in the PDF file
        fig = Figure(figsize=(10, 6))
        ax = fig.subplots()
        tea.imshow(
            fig,
            ax,
            image_data,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            ds9mode=False,  # note that the polynomials are fitted using array indices (0-based)
            title=title + f"\nSlice num {islice + 1} (ID {sliceid})",
        )
        # Plot the boundary polynomials
        if list_poly_left is not None and list_poly_right is not None:
            plot_fitted_boundary_polynomials(
                ax,
                list_poly_left,
                list_poly_right,
                show_sliceid=False,
                isliceplot=islice,
            )
        # Overplot the boundary borders
        if array_left_border is not None and array_right_border is not None and ibad is not None:
            plot_borders(ax, array_left_border, array_right_border, ibad, show_sliceid=False, isliceplot=islice)
        ixdum = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
        ymin = np.min(list_poly_left[islice](ixdum)) - 10
        ymax = np.max(list_poly_right[islice](ixdum)) + 10
        ax.set_ylim(ymin, ymax)
        # Overplot the slice traces
        if list_poly_traces_all_slices is not None:
            plot_traces(ax, list_poly_traces_all_slices, show_traceid=False, isliceplot=islice)
            if show_traceid:
                ntraces = len(list_poly_traces_all_slices[islice])
                xcenter = (FRIDA_NAXIS1_HAWAII.value - 1) / 2
                for itrace in range(ntraces):
                    poly_trace = list_poly_traces_all_slices[islice][itrace]
                    ytrace = poly_trace(ixdum)
                    ax.plot(ixdum, ytrace, color="cyan", lw=1.5, label=f"Trace {itrace+1}")
                    ycenter = poly_trace(xcenter)
                    if ymin <= ycenter <= ymax:
                        ax.text(
                            xcenter,
                            ycenter,
                            f"Trace {itrace+1}",
                            color="white",
                            fontsize=8,
                            ha="center",
                            va="center",
                            fontweight="bold",
                            alpha=1.0,
                            bbox=dict(facecolor="black", alpha=0.3, edgecolor="black", boxstyle="round,pad=0.5"),
                        )
        #
        fig.tight_layout()  # Fails if show_sliceid=True in plot_fitted_boundary_polynomials
        pdf_out.savefig(fig, bbox_inches="tight")
    pdf_out.close()

    # Check montage
    if montage is not None:
        if montage not in ["png", "pdf"]:
            raise ValueError("Invalid value for montage. Must be 'png' or 'pdf'.")
        montage_executable = find_imagemagick_montage_executable()
        if montage_executable is None:
            raise ValueError(
                "The --montage option is defined, but the ImageMagick 'montage' executable is not found "
                "in the system PATH. Please install ImageMagick or provide a valid path to the 'montage' executable."
            )
        else:
            logger.info(f"ImageMagick 'montage' executable found at: [green]{montage_executable}[/green]")
        # Build output filename for montage from pdf_output, inserting "_montage" before the extension
        pdf_output_path = Path(pdf_output)
        montage_output = pdf_output_path.with_name(pdf_output_path.stem + "_montage." + montage)
        logger.info(f"Creating montage of the final plots in file:\n[green]{montage_output}[/green]")
        montage_command = [
            montage_executable,
            "-density",
            "300",
            str(pdf_output_path),
            "-tile",
            "6x5",
            "-geometry",
            "600x375+5+5",
            "-background",
            "white",
            str(montage_output),
        ]
        logger.debug(f"Running montage command: {' '.join(montage_command)}")
        subprocess.run(montage_command, check=True)


def plot_fitted_boundary_polynomials(
    ax,
    list_poly_left,
    list_poly_right,
    show_sliceid=False,
    isliceplot=None,
    plot_left_boundary=True,
    plot_right_boundary=True,
):
    """Plot the fitted slice boundary polynomials on the given axes

    The polynomials are assumed to be fitted using as independent variable
    the array index along the NAXIS1 axis, which ranges from 0 to FRIDA_NAXIS1_HAWAII-1,
    and as dependent variable the array index along the NAXIS2 axis,
    which ranges from 0 to FRIDA_NAXIS2_HAWAII-1.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the polynomials.
    list_poly_left : list of numpy.polynomial.Polynomial
        The list of left slice boundary polynomials.
    list_poly_right : list of numpy.polynomial.Polynomial
        The list of right slice boundary polynomials.
    show_sliceid : bool, optional
        If True, overplot the slice ID at the center of each slice.
    isliceplot : int, optional
        If provided, only plot the polynomials for the specified slice index (0-based).
    plot_left_boundary : bool, optional
        If True, plot the left boundary polynomial.
    plot_right_boundary : bool, optional
        If True, plot the right boundary polynomial.
    """
    xmin, xmax = ax.get_xlim()
    xdum = np.linspace(xmin, xmax, 1000)
    ymin, ymax = ax.get_ylim()
    if isliceplot is None:
        islice_range = range(FRIDA_NSLICES)
    else:
        islice_range = [isliceplot]
    for islice in islice_range:
        if plot_left_boundary and list_poly_left[islice] is not None:
            ax.plot(xdum, list_poly_left[islice](xdum), color="white", lw=5.0, alpha=0.7)
            ax.plot(xdum, list_poly_left[islice](xdum), color="C0", lw=2.0, alpha=0.7)
        if plot_right_boundary and list_poly_right[islice] is not None:
            ax.plot(xdum, list_poly_right[islice](xdum), color="white", lw=5.0, alpha=0.7)
            ax.plot(xdum, list_poly_right[islice](xdum), color="C1", lw=2.0, alpha=0.7)
        if show_sliceid:
            if list_poly_left[islice] is not None and list_poly_right[islice] is not None:
                xcenter = (xmin + xmax) / 2
                ycenter = (list_poly_left[islice](xcenter) + list_poly_right[islice](xcenter)) / 2
                if ymin <= ycenter <= ymax:
                    ax.text(
                        xcenter,
                        ycenter,
                        f"id#{sliceid_from_sliceindex(islice)}",
                        color="white",
                        fontsize=10,
                        ha="center",
                        va="center",
                        fontweight="bold",
                        alpha=1.0,
                        bbox=dict(facecolor="black", alpha=0.3, edgecolor="black", boxstyle="round,pad=0.5"),
                    )


def plot_borders(
    ax,
    array_left_border,
    array_right_border,
    ibad,
    show_sliceid=False,
    isliceplot=None,
    color="white",
    marker=".",
    markersize=0.5,
    alpha=1.0,
):
    """Plot the slice boundary borders on the given axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the borders.
    array_left_border : numpy.ndarray
        The array of left slice boundary borders (0-based indices).
    array_right_border : numpy.ndarray
        The array of right slice boundary borders (0-based indices).
    ibad : list of int
        List of indices of bad columns (to be ignored).
    show_sliceid : bool, optional
        If True, overplot the slice ID at the center of each slice.
    isliceplot : int, optional
        If provided, only plot the borders for the specified slice index (0-based).
    color : str, optional
        Color of the markers for the borders.
    marker : str, optional
        Marker style for the borders.
    markersize : float, optional
        Size of the markers for the borders.
    alpha : float, optional
        Transparency level of the markers for the borders (0.0 to 1.0).
    """
    x = np.arange(FRIDA_NAXIS1_HAWAII.value)
    xplot = x[~ibad]

    if isliceplot is None:
        islice_range = range(FRIDA_NSLICES)
    else:
        islice_range = [isliceplot]

    for islice in islice_range:
        y_left = array_left_border[islice, ~ibad]
        y_right = array_right_border[islice, ~ibad]
        if not np.all(np.isnan(y_left)):
            ax.plot(xplot, y_left, color=color, marker=marker, markersize=markersize, linestyle="None", alpha=alpha)
        if not np.all(np.isnan(y_right)):
            ax.plot(xplot, y_right, color=color, marker=marker, markersize=markersize, linestyle="None", alpha=alpha)
        if not np.all(np.isnan(y_left)) and not np.all(np.isnan(y_right)) and show_sliceid:
            xcenter = (FRIDA_NAXIS1_HAWAII.value - 1) / 2
            ycenter = (array_left_border[islice, int(xcenter)] + array_right_border[islice, int(xcenter)]) / 2
            ax.text(
                xcenter,
                ycenter,
                f"id#{sliceid_from_sliceindex(islice)}",
                color=color,
                fontsize=10,
                ha="center",
                va="center",
                fontweight="bold",
                alpha=1.0,
                bbox=dict(facecolor="black", alpha=0.3, edgecolor="black", boxstyle="round,pad=0.5"),
            )


def plot_traces(ax, list_poly_traces_all_slices, show_traceid=False, color="cyan", alpha=1.0, isliceplot=None):
    """Plot the slice trace polynomials on the given axes

    The polynomials are assumed to be fitted using as independent variable
    the array index along the NAXIS1 axis, which ranges from 0 to FRIDA_NAXIS1_HAWAII-1,
    and as dependent variable the array index along the NAXIS2 axis,
    which ranges from 0 to FRIDA_NAXIS2_HAWAII-1.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the traces.
    list_poly_traces_all_slices : list of list of numpy.polynomial.Polynomial
        The list of polynomials for each slice.
    show_traceid : bool, optional
        If True, overplot the trace ID at the center of each trace.
    color : str, optional
        Color of the lines for the traces.
    alpha : float, optional
        Transparency level of the lines for the traces (0.0 to 1.0).
    isliceplot : int, optional
        If provided, only plot the traces for the specified slice index (0-based).
    """
    xdum = np.arange(FRIDA_NAXIS1_HAWAII.value)
    if isliceplot is None:
        islice_range = range(FRIDA_NSLICES)
    else:
        islice_range = [isliceplot]

    for islice in islice_range:
        for itrace, poly in enumerate(list_poly_traces_all_slices[islice]):
            ax.plot(
                xdum,
                poly(xdum),
                color=color,
                lw=1.0,
                alpha=alpha,
            )
            if show_traceid:
                xcenter = (FRIDA_NAXIS1_HAWAII.value - 1) / 2
                ycenter = poly(xcenter)
                ax.text(
                    xcenter,
                    ycenter,
                    f"id#{sliceid_from_sliceindex(islice)}, trace {itrace+1}",
                    color="white",
                    fontsize=10,
                    ha="center",
                    va="center",
                    fontweight="bold",
                    alpha=1.0,
                    bbox=dict(facecolor="black", alpha=0.3, edgecolor="black", boxstyle="round,pad=0.5"),
                )


def polynomial_from_voffset(voffset, debugplot=None):
    """Compute the polynomial describing a vertical offset along NAXIS1

    The vertical offsets are assumed to be evenly spaced along the useful
    region of the detector in the NAXIS1 direction, from
    FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL to FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL
    (both included). A polynomial of degree len(voffset)-1 is fitted to these
    values, using as independent variable the array index (0-based) along NAXIS1.

    Parameters
    ----------
    voffset : list of float
        Vertical offsets (pixels) along NAXIS2. A positive value corresponds
        to a shift towards higher array indices along NAXIS2.
    debugplot : int or None, optional
        Debugplot code passed to polfit_residuals (see
        numina.array.display.pause_debugplot for valid values). If None,
        2 is used when the logger of this module is enabled for DEBUG, and
        0 otherwise. It has no effect when all the offsets are zero.

    Returns
    -------
    poly_voffset : numpy.polynomial.Polynomial
        Polynomial that returns the vertical offset as a function of the
        array index (0-based) along NAXIS1. If all the offsets are zero,
        no fit is performed and the null polynomial is returned.

    Raises
    ------
    ValueError
        If voffset is empty.
    """
    logger = logging.getLogger(__name__)

    if len(voffset) < 1:
        raise ValueError("The list of vertical offsets (voffset) must contain at least one value.")
    if np.any(np.array(voffset) != 0):
        logger.info(f"Fitting a polynomial of degree {len(voffset)-1} to the vertical offsets")
        nvoffsets = len(voffset)
        xdum_voffsets = np.linspace(
            FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL.value - 1,
            FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL.value - 1,
            nvoffsets,
        )  # using array indices (0 based)
        if debugplot is None:
            if logger.isEnabledFor(logging.DEBUG):
                debugplot = 2
            else:
                debugplot = 0
        poly_voffset, _ = polfit_residuals(
            x=xdum_voffsets,
            y=np.array(voffset),
            deg=len(voffset) - 1,
            xlabel="array index along NAXIS1",
            ylabel="array index along NAXIS2",
            title="Vertical offset to be applied",
            debugplot=debugplot,
        )
    else:
        # all offsets are zero: skip the fit (and its debug plot)
        poly_voffset = Polynomial([0.0])

    return poly_voffset


def connect_image_display_keys(fig, ax, image_data, vmin, vmax, ds9mode=False):
    """Connect the interactive keys to a figure displaying one or several images

    The following keys are available ('?' displays this list, which is
    also shown when this function is called):

    - 'a': toggle imshow aspect='equal' / aspect='auto' (all the axes)
    - 'h': reset zoom to the axis limits at the time of this call, for all
      the axes (and the display limits of each image to the last vmin,
      vmax selected for it)
    - 'v': set vmin and vmax manually (read from standard input)
    - ',': set vmin and vmax to the min and max of the zoomed region
    - '/': set vmin and vmax using zscale of the zoomed region
    - 'q': close the figure

    When a single axes is given, the keys 'v', ',' and '/' act on its image
    wherever the cursor is. When several axes are given, they act on the
    image of the axes under the cursor, and are ignored when the cursor is
    not over any of them.

    The default matplotlib keymaps for 'h', 'r', 'c' and 'v' are disabled
    to avoid conflicts. The figure is set in zoom mode (equivalent to
    pressing the zoom button of the toolbar or the key 'o') when the figure
    window has a NavigationToolbar2.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the image(s).
    ax : matplotlib.axes.Axes or sequence of matplotlib.axes.Axes
        Axes where each image has been displayed (the image must be
        ax.images[0] in each axes).
    image_data : numpy.ndarray or sequence of numpy.ndarray
        2D array(s) with the displayed image(s), employed to compute the
        display limits of the zoomed region. If ax is a sequence, it must
        be a sequence of the same length.
    vmin : float or sequence of float
        Initial background display limit(s). If ax is a sequence, it must
        be a sequence of the same length.
    vmax : float or sequence of float
        Initial foreground display limit(s). If ax is a sequence, it must
        be a sequence of the same length.
    ds9mode : bool, optional
        If False, the axis coordinates are array indices (0-based), i.e.,
        the image extent is [-0.5, NAXIS1-0.5, -0.5, NAXIS2-0.5]. If True,
        the axis coordinates are FITS pixels (1-based), i.e., the image
        extent is [0.5, NAXIS1+0.5, 0.5, NAXIS2+0.5], as with
        teareduce.imshow(..., ds9mode=True).

    Returns
    -------
    on_key : callable
        Function connected to the 'key_press_event' of the figure. It
        receives an object with a 'key' attribute (and, when several axes
        are given, an 'inaxes' attribute).

    Raises
    ------
    ValueError
        If ax is a sequence and image_data, vmin and vmax are not
        sequences of the same length.
    """
    logger = logging.getLogger(__name__)

    # Normalise the input to lists with one element per displayed image
    if isinstance(ax, Axes):
        list_ax = [ax]
        list_data = [image_data]
        list_vmin = [vmin]
        list_vmax = [vmax]
    else:
        list_ax = list(ax)
        try:
            list_data = list(image_data)
            list_vmin = list(vmin)
            list_vmax = list(vmax)
        except TypeError:
            raise ValueError("When ax is a sequence, image_data, vmin and vmax must be sequences too.")
        if not (len(list_data) == len(list_vmin) == len(list_vmax) == len(list_ax)):
            raise ValueError("ax, image_data, vmin and vmax must have the same length.")
    nax = len(list_ax)
    # offset to convert axis coordinates into 0-based array indices
    ioffset = 1 if ds9mode else 0

    mpl.rcParams["keymap.home"] = []  # disable 'h' and 'r'
    mpl.rcParams["keymap.back"] = []  # disable 'c'
    mpl.rcParams["keymap.forward"] = []  # disable 'v' (conflict with 'v' for setting vmin/vmax)
    init_xylimits = [(ax_.get_xlim(), ax_.get_ylim()) for ax_ in list_ax]

    def on_key(event):
        if event.key == "?":
            logger.info("-" * 79)
            logger.info("'?': show this help message")
            logger.info("'a': toggle imshow aspect='equal' / aspect='auto'")
            logger.info("'h': reset zoom to initial limits")
            logger.info("'v': set vmin and vmax manually")
            logger.info("',': set vmin and vmax to min and max of the zoomed region")
            logger.info("'/': set vmin and vmax using zscale of the zoomed region")
            if nax > 1:
                logger.info("('v', ',' and '/' act on the image under the cursor)")
            logger.info("'q': quit")
            logger.info("-" * 79)
        elif event.key == "h":
            for iax, ax_ in enumerate(list_ax):
                xlim, ylim = init_xylimits[iax]
                ax_.set_xlim(xlim)
                ax_.set_ylim(ylim)
                ax_.images[0].set_clim(list_vmin[iax], list_vmax[iax])
            fig.set_layout_engine("none")  # deactivate accumulated tight_layout adjustments
            fig.tight_layout()  # apply new tight_layout adjustments
            # the redraw is performed by the event loop running this handler
            # (do not call plt.pause() here: it would start a nested event loop)
            fig.canvas.draw_idle()
        elif event.key == "a":
            # the new aspect is decided from the first axes and applied to all of them
            if list_ax[0].get_aspect() in ["equal", 1.0]:
                logger.info("Setting aspect to 'auto'.")
                new_aspect = "auto"
            else:
                logger.info("Setting aspect to 'equal'.")
                new_aspect = "equal"
            for ax_ in list_ax:
                ax_.set_aspect(new_aspect)
            fig.set_layout_engine("none")  # deactivate accumulated tight_layout adjustments
            fig.tight_layout()  # apply new tight_layout adjustments
            fig.canvas.draw()
        elif event.key in ["v", ",", "/"]:
            # select the image on which the key acts
            if nax == 1:
                iax = 0
            else:
                inaxes = getattr(event, "inaxes", None)
                if inaxes not in list_ax:
                    return
                iax = list_ax.index(inaxes)
            ax_ = list_ax[iax]
            if event.key == "v":
                current_vmin, current_vmax = ax_.images[0].get_clim()
                vmin_ = input_number(expected_type="float", prompt="Enter vmin: ", default=current_vmin)
                vmax_ = input_number(expected_type="float", prompt="Enter vmax: ", default=current_vmax)
            else:
                xlim = ax_.get_xlim()
                ylim = ax_.get_ylim()
                # axis coordinates -> 0-based array indices
                x1, x2 = int(xlim[0] - ioffset), int(xlim[1] - ioffset)
                y1, y2 = int(ylim[0] - ioffset), int(ylim[1] - ioffset)
                x1 = max(0, min(x1, FRIDA_NAXIS1_HAWAII.value - 1))
                x2 = max(0, min(x2, FRIDA_NAXIS1_HAWAII.value - 1))
                y1 = max(0, min(y1, FRIDA_NAXIS2_HAWAII.value - 1))
                y2 = max(0, min(y2, FRIDA_NAXIS2_HAWAII.value - 1))
                data = list_data[iax]
                if event.key == ",":
                    vmin_ = np.nanmin(data[y1 : y2 + 1, x1 : x2 + 1])
                    vmax_ = np.nanmax(data[y1 : y2 + 1, x1 : x2 + 1])
                else:
                    vmin_, vmax_ = ZScaleInterval().get_limits(data[y1 : y2 + 1, x1 : x2 + 1])
                logger.info(f"Setting vmin={vmin_}, vmax={vmax_} for the zoomed region.")
            list_vmin[iax], list_vmax[iax] = vmin_, vmax_
            ax_.images[0].set_clim(vmin_, vmax_)
            # the redraw is performed by the event loop running this handler
            # (do not call plt.pause() here: it would start a nested event loop)
            fig.canvas.draw_idle()
        elif event.key == "q":
            plt.close(fig)

    on_key(event=types.SimpleNamespace(key="?"))  # Show help message on startup
    fig.canvas.mpl_connect("key_press_event", on_key)
    # Start in zoom mode (equivalent to pressing the zoom button of the toolbar or the key 'o').
    # Nothing is done when there is no NavigationToolbar2 (non-interactive backend, or
    # rcParams['toolbar'] set to 'toolmanager').
    toolbar = getattr(fig.canvas.manager, "toolbar", None)
    if isinstance(toolbar, NavigationToolbar2):
        toolbar.zoom()

    return on_key


def interactive_image_display(fig, ax, image_data, vmin, vmax, ds9mode=False):
    """Display one or several images interactively until the figure is closed

    The interactive keys defined in connect_image_display_keys are
    connected to the figure, and the execution is blocked until the
    figure is closed by the user (window closed or key 'q').

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure containing the image(s) (created with pyplot).
    ax : matplotlib.axes.Axes or sequence of matplotlib.axes.Axes
        Axes where each image has been displayed (the image must be
        ax.images[0] in each axes).
    image_data : numpy.ndarray or sequence of numpy.ndarray
        2D array(s) with the displayed image(s).
    vmin : float or sequence of float
        Initial background display limit(s).
    vmax : float or sequence of float
        Initial foreground display limit(s).
    ds9mode : bool, optional
        If True, the axis coordinates are FITS pixels (1-based) instead of
        array indices (0-based). See connect_image_display_keys.
    """
    connect_image_display_keys(fig, ax, image_data, vmin, vmax, ds9mode=ds9mode)
    # fig.set_layout_engine("none")  # deactivate accumulated tight_layout adjustments
    fig.tight_layout()  # apply new tight_layout adjustments
    # instead of plt.show(), use a loop to keep the figure open until closed by the user
    # (otherwise, after using input_number() in the on_key function, the matplotlib event
    # loop is not properly restored and the execution of the code continues as if the
    # figure was closed, which is not the case)
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)


def overplot_slice_boundary_polynomials(
    input_poly,
    input_borders,
    input_traces,
    image,
    voffset=[0.0],
    xlim=None,
    ylim=None,
    vlim=None,
    show_sliceid=False,
    show_traceid=False,
    pdf_mosaic=False,
    montage=None,
    output_dir=".",
):
    """Overplot the slice boundary borders and/or polynomials on an image

    The slice boundary borders are given as 2D arrays of shape
    (FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII), where each row corresponds to a slice
    and each column corresponds to a pixel along the NAXIS1 axis. The coordinates
    of the slice boundaries are given as 0-based indices along the NAXIS2 axis.

    The slice boundary polynomials are given as a list of polynomial objects,
    one for each slice, which can be evaluated at any pixel along the NAXIS1 axis
    to obtain the corresponding boundary position along the NAXIS2 axis.

    The polynomials are assumed to be fitted using as independent variable
    the array index along the NAXIS1 axis, which ranges from 0 to FRIDA_NAXIS1_HAWAII-1,
    and as dependent variable the array index along the NAXIS2 axis,
    which ranges from 0 to FRIDA_NAXIS2_HAWAII-1.

    Parameters
    ----------
    input_poly : str
        Path to the FITS file containing the slice boundary polynomials.
    input_borders : str
        Path to the file containing the slice boundary borders.
        This is optional and can be used to overplot the borders as well.
    input_traces : str
        Path to the file containing the slice trace polynomials.
        This is optional and can be used to overplot the traces as well.
    image : str
        Path to the FITS file containing the image on which to overplot
        the slice boundaries.
    xlim : tuple of float, optional
        X-axis limits for the plot (xmin, xmax); array indices (0-based).
    ylim : tuple of float, optional
        Y-axis limits for the plot (ymin, ymax); array indices (0-based).
    vlim : tuple of float, optional
        Background and foreground limits for the plot (vmin, vmax).
    voffset : list of float, optional
        Vertical offsets to apply to the slice boundaries (pixels).
        If more than one value is provided, a polynomial of degree len(voffset)-1
        will be fitted to the offsets and applied to the displayed
        slice boundaries and polynomials. A positive value shifts the
        displayed objects upwards (i.e., towards higher array indices
        along NAXIS2).
    show_sliceid : bool, optional
        If True, overplot the slice ID at the center of each slice.
    show_traceid : bool, optional
        If True, overplot the trace ID at the center of each trace.
    pdf_mosaic : str, optional
        Path to the output PDF file where the mosaic will be saved.
    montage : str, optional
        If provided, display a montage of the final plots with traces
        for every slice. Possible values are "png" or "pdf".
        Requires ImageMagick's montage.
    output_dir : str, optional
        Path to the output directory where the PDF mosaic will be saved.
    """
    logger = logging.getLogger(__name__)

    # Read the image data from the input FITS file
    with fits.open(image) as hdul:
        image_data = hdul[0].data.astype(float)  # Convert to float for processing

    # Compute polynomial to take into account voffset when plotting:
    # - boundary data
    # - boundary polynomials
    # - trace polynomials
    # Note that the voffset is applied to the slice boundaries and polynomials,
    # but not to the image data.
    poly_voffset = polynomial_from_voffset(voffset)

    # Read the boundary polynomial coefficients from the input FITS file
    if input_poly is not None or input_traces is not None:
        if input_poly is not None:
            list_poly_left, list_poly_right, _ = read_slice_boundary_polynomials(input_poly)
        else:
            list_poly_left, list_poly_right, _ = read_slice_boundary_polynomials(input_traces)
        # apply requested voffset
        for islice in range(FRIDA_NSLICES):
            if list_poly_left[islice] is not None:
                list_poly_left[islice] += poly_voffset
            if list_poly_right[islice] is not None:
                list_poly_right[islice] += poly_voffset
    else:
        list_poly_left, list_poly_right = None, None

    # Read the boundary borders from the input file and overplot them on the image
    if input_borders is not None:
        array_left_border, array_right_border, ibad, _, _ = read_slice_boundary_borders(input_borders)
        logger.info(
            f"Read {len(array_left_border)} left borders and {len(array_right_border)} right borders "
            f"from\n[blue]{input_borders}[/blue]."
        )
        # apply requested voffset
        xdum = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
        array_left_border += poly_voffset(xdum)
        array_right_border += poly_voffset(xdum)
    else:
        array_left_border, array_right_border, ibad = None, None, None

    # Read the slice trace polynomials from the input file and overplot them on the image
    if input_traces is not None:
        list_poly_traces_all_slices = read_slice_trace_polynomials(input_traces)
        logger.info(
            f"Read {len(list_poly_traces_all_slices)} slices with {len(list_poly_traces_all_slices[0])} traces "
            f"per slice from\n[blue]{input_traces}[/blue]."
        )
        # apply requested voffset
        for islice in range(FRIDA_NSLICES):
            for itrace in range(len(list_poly_traces_all_slices[islice])):
                if list_poly_traces_all_slices[islice][itrace] is not None:
                    list_poly_traces_all_slices[islice][itrace] += poly_voffset
    else:
        list_poly_traces_all_slices = [None] * FRIDA_NSLICES

    # Define the title for the plot based on the input files
    title = f"Image: {Path(image).name}"
    if input_poly is not None:
        title += f"\nSlice boundary polynomials: {Path(input_poly).name}"
    elif input_traces is not None:
        title += f"\nSlice trace polynomials: {Path(input_traces).name}"

    # If the user requested to save the plots in a PDF mosaic,
    # create a PdfPages object and save each slice's plot in reverse order
    if pdf_mosaic is not None:
        if output_dir is not None:
            pdf_output = Path(output_dir) / pdf_mosaic
        else:
            pdf_output = pdf_mosaic
        plot_traces_within_slice_boundary_polynomials_mosaic(
            pdf_output=pdf_output,
            montage=montage,
            image_data=image_data,
            title=title,
            list_poly_left=list_poly_left,
            list_poly_right=list_poly_right,
            array_left_border=array_left_border,
            array_right_border=array_right_border,
            ibad=ibad,
            list_poly_traces_all_slices=list_poly_traces_all_slices,
            show_traceid=show_traceid,
            vlim=vlim,
        )
    else:
        if vlim is not None:
            vmin, vmax = vlim
        else:
            vmin, vmax = ZScaleInterval().get_limits(image_data)
        fig, ax = plt.subplots(figsize=(10, 8))
        tea.imshow(
            fig,
            ax,
            image_data,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            title=title,
        )
        if xlim is None:
            xlim = [-0.5, FRIDA_NAXIS1_HAWAII.value - 0.5]
        else:
            xlim[0] -= 1  # (0-based)
            xlim[1] -= 1  # (0-based)
        if ylim is None:
            ylim = [-0.5, FRIDA_NAXIS2_HAWAII.value - 0.5]
        else:
            ylim[0] -= 1  # (0-based)
            ylim[1] -= 1  # (0-based)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        # Plot the boundary polynomials
        if input_poly is not None or input_traces is not None:
            plot_fitted_boundary_polynomials(
                ax,
                list_poly_left,
                list_poly_right,
                show_sliceid,
            )
        # Overplot the boundary borders
        if input_borders is not None:
            show_sliceid_ = show_sliceid
            if input_poly is not None:
                show_sliceid_ = False  # Avoid overplotting slice IDs twice
            plot_borders(ax, array_left_border, array_right_border, ibad, show_sliceid=show_sliceid_)
        # Overplot the slice traces
        if input_traces is not None:
            plot_traces(ax, list_poly_traces_all_slices, show_traceid=show_traceid)

        interactive_image_display(fig, ax, image_data, vmin, vmax)


def main(args=None):

    parser = argparse.ArgumentParser(
        description="Overplot the slice boundaries (borders and/or polynomials) and/or traces on image",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument(
        "--poly", help="Path to the file with the boundary polynomials (optional)", type=str, required=False
    )
    parser.add_argument(
        "--borders", help="Path to the file with the boundary borders (optional)", type=str, required=False
    )
    parser.add_argument(
        "--traces", help="Path to the file with the slice trace polynomials (optional)", type=str, required=False
    )
    parser.add_argument("--image", help="Image to display boundaries on", type=str, required=False)
    parser.add_argument(
        "--voffset",
        help="Vertical offset to apply to the slice boundaries (default: 0.0 pixels; + upward, - downward). "
        "If more than one value is provided, a polynomial of degree len(voffset)-1 will be fitted "
        "to the offsets and applied to the slice boundaries.",
        type=float,
        nargs="+",
        default=[0.0],
    )
    parser.add_argument("--show-sliceid", help="Overplot slice ID", action="store_true")
    parser.add_argument("--show-traceid", help="Overplot trace ID", action="store_true")
    parser.add_argument(
        "--xlim",
        help="X-axis limits for the plot (xmin, xmax); array indices (1-based along NAXIS1 axis)",
        type=float,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "--ylim",
        help="Y-axis limits for the plot (ymin, ymax); array indices (1-based along NAXIS2 axis)",
        type=float,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "--vlim",
        help="Background and foreground limits (vmin, vmax)",
        type=float,
        nargs=2,
        default=None,
    )
    parser.add_argument(
        "--pdf-mosaic", help="Output PDF file to save zoomed images of all the slices", type=str, required=False
    )
    parser.add_argument(
        "--montage",
        help="Display a montage of the final plots with traces for every slice (requires ImageMagick's montage)",
        type=str,
        choices=["png", "pdf"],
        default=None,
    )

    # Include default arguments for common actions, and initialize console and logging
    myscript = NuminaScriptDefinition(parser, argv=args)
    args = myscript.args
    logger = myscript.logger

    # Check input files are defined
    if args.poly is None and args.borders is None and args.traces is None:
        raise ValueError(
            "At least one of the input files (--poly or --borders or --traces) must be defined "
            "to overplot the slice boundaries."
        )
    if args.poly is not None and args.traces is not None:
        raise ValueError(
            "Both --poly and --traces are defined. Please define only one of them to overplot the slice boundaries."
        )

    # Check the input image file is defined if the user wants to overplot the boundaries
    if args.image is None:
        logger.warning("No input image file defined. The slice boundaries will not be overplotted on an image.")

    # Check that both show_sliceid and show_traceid are not defined at the same time
    if args.show_sliceid and args.show_traceid:
        raise ValueError(
            "Both --show-sliceid and --show-traceid are defined. Please define only one of them to overplot the IDs."
        )

    # Check that if show_traceid is defined, traces file must be provided
    if args.show_traceid and args.traces is None:
        raise ValueError(
            "The --show-traceid option is defined, but no slice trace polynomials file is provided. "
            "Please provide a valid --traces file."
        )

    #
    if args.pdf_mosaic is not None and (args.xlim is not None or args.ylim is not None):
        logger.warning(
            "Both --pdf-mosaic and --xlim/--ylim are defined. The --xlim/--ylim options will be ignored "
            "when generating the PDF mosaic."
        )

    # Overplot the slice boundary polynomials
    overplot_slice_boundary_polynomials(
        input_poly=args.poly,
        input_borders=args.borders,
        input_traces=args.traces,
        image=args.image,
        xlim=args.xlim,
        ylim=args.ylim,
        vlim=args.vlim,
        voffset=args.voffset,
        show_sliceid=args.show_sliceid,
        show_traceid=args.show_traceid,
        pdf_mosaic=args.pdf_mosaic,
        montage=args.montage,
        output_dir=args.output_dir,
    )

    # Display goodbye message and save console if recording is enabled
    myscript.goodbye_message_and_save_console()


if __name__ == "__main__":
    main()
