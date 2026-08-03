#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Overplot the slice boundary polynomials, borders and/or traces on image"""

import argparse
from astropy import logger
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from datetime import datetime
import logging
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rich_argparse import RichHelpFormatter
import sys
import teareduce as tea
import types

from numina.tools.input_number import input_number

from fridadrp.core import FRIDA_NAXIS1_HAWAII, FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import sliceid_from_sliceindex
from fridadrp.tools.initialize_script_with_args import initialize_script_with_args
from fridadrp.tools.read_slice_boundary_borders import read_slice_boundary_borders
from fridadrp.tools.read_slice_boundary_polynomials import read_slice_boundary_polynomials
from fridadrp.tools.read_slice_trace_polynomials import read_slice_trace_polynomials


def plot_fitted_boundary_polynomials(ax, list_poly_left, list_poly_right, voffset=0.0, sliceid=False, isliceplot=None):
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
    voffset : float, optional
        Vertical constant offset (pixels) to apply. A positive value
        shifts the displayed objects upwards, while a negative value
        shifts them downwards.
    sliceid : bool, optional
        If True, overplot the slice ID at the center of each slice.
    isliceplot : int, optional
        If provided, only plot the polynomials for the specified slice index (0-based).
    """
    xmin, xmax = ax.get_xlim()
    xdum = np.linspace(xmin, xmax, 1000)
    if isliceplot is None:
        islice_range = range(FRIDA_NSLICES)
    else:
        islice_range = [isliceplot]
    for islice in islice_range:
        if list_poly_left[islice] is not None:
            ax.plot(xdum, list_poly_left[islice](xdum) + voffset, color="white", lw=5.0, alpha=0.7)
            ax.plot(xdum, list_poly_left[islice](xdum) + voffset, color="C0", lw=2.0, alpha=0.7)
        if list_poly_right[islice] is not None:
            ax.plot(xdum, list_poly_right[islice](xdum) + voffset, color="white", lw=5.0, alpha=0.7)
            ax.plot(xdum, list_poly_right[islice](xdum) + voffset, color="C1", lw=2.0, alpha=0.7)
        if sliceid:
            if list_poly_left[islice] is not None and list_poly_right[islice] is not None:
                xcenter = (FRIDA_NAXIS1_HAWAII.value - 1) / 2
                ycenter = (list_poly_left[islice](xcenter) + list_poly_right[islice](xcenter)) / 2 + voffset
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
    voffset=0.0,
    sliceid=False,
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
    voffset : float, optional
        Vertical constant offset (pixels) to apply. A positive value
        shifts the displayed objects upwards, while a negative value
        shifts them downwards.
    sliceid : bool, optional
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
        y_left = array_left_border[islice, ~ibad] + voffset
        y_right = array_right_border[islice, ~ibad] + voffset
        if not np.all(np.isnan(y_left)):
            ax.plot(xplot, y_left, color=color, marker=marker, markersize=markersize, linestyle="None", alpha=alpha)
        if not np.all(np.isnan(y_right)):
            ax.plot(xplot, y_right, color=color, marker=marker, markersize=markersize, linestyle="None", alpha=alpha)
        if not np.all(np.isnan(y_left)) and not np.all(np.isnan(y_right)) and sliceid:
            xcenter = (FRIDA_NAXIS1_HAWAII.value - 1) / 2
            ycenter = (array_left_border[islice, int(xcenter)] + array_right_border[islice, int(xcenter)]) / 2 + voffset
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


def plot_traces(ax, list_poly_traces_all_slices, voffset=0.0, traceid=False, color="cyan", alpha=1.0, isliceplot=None):
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
    voffset : float, optional
        Vertical constant offset (pixels) to apply. A positive value
        shifts the displayed objects upwards, while a negative value
        shifts them downwards.
    traceid : bool, optional
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
                poly(xdum) + voffset,
                color=color,
                lw=1.0,
                alpha=alpha,
            )
            if traceid:
                xcenter = (FRIDA_NAXIS1_HAWAII.value - 1) / 2
                ycenter = poly(xcenter) + voffset
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


def overplot_slice_boundary_polynomials(
    input_poly,
    input_borders,
    input_traces,
    image,
    voffset=0.0,
    sliceid=False,
    traceid=False,
    pdf_mosaic=False,
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
    voffset : float, optional
        Vertical constant offset (pixels) to apply to the polynomials.
        A positive value shifts the polynomials upwards, while a negative
        value shifts them downwards.
    sliceid : bool, optional
        If True, overplot the slice ID at the center of each slice.
    traceid : bool, optional
        If True, overplot the trace ID at the center of each trace.
    pdf_mosaic : str, optional
        Path to the output PDF file where the mosaic will be saved.
    output_dir : str, optional
        Path to the output directory where the PDF mosaic will be saved.
    """
    logger = logging.getLogger(__name__)

    # Read the image data from the input FITS file
    with fits.open(image) as hdul:
        image_data = hdul[0].data
    vmin, vmax = ZScaleInterval().get_limits(image_data)

    # Read the boundary polynomial coefficients from the input FITS file
    if input_poly is not None or input_traces is not None:
        if input_poly is not None:
            list_poly_left, list_poly_right, poldeg = read_slice_boundary_polynomials(input_poly)
        else:
            list_poly_left, list_poly_right, poldeg = read_slice_boundary_polynomials(input_traces)
    else:
        list_poly_left, list_poly_right, poldeg = None, None, None

    # Read the boundary borders from the input file and overplot them on the image
    if input_borders is not None:
        array_left_border, array_right_border, ibad, uuid_borders, islice_ok = read_slice_boundary_borders(
            input_borders
        )
        logger.info(
            f"Read {len(array_left_border)} left borders and {len(array_right_border)} right borders from {input_borders}."
        )
    else:
        array_left_border, array_right_border, ibad, uuid_borders, islice_ok = None, None, None, None, None

    # Read the slice trace polynomials from the input file and overplot them on the image
    if input_traces is not None:
        list_poly_traces_all_slices = read_slice_trace_polynomials(input_traces)
        logger.info(
            f"Read {len(list_poly_traces_all_slices)} slices with {len(list_poly_traces_all_slices[0])} traces per slice from {input_traces}."
        )
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
        logger.info(f"Saving final plots of traces for every slice in PDF file: {pdf_mosaic}")
        if output_dir is not None:
            pdf_output = PdfPages(Path(output_dir) / pdf_mosaic)
        else:
            pdf_output = PdfPages(pdf_mosaic)
        for islice in range(
            FRIDA_NSLICES - 1, -1, -1
        ):  # (0-based index) loop in reverse order to have the first slice on top of the PDF
            sliceid = sliceid_from_sliceindex(islice)
            fig, ax = plt.subplots(figsize=(10, 6))
            tea.imshow(
                fig,
                ax,
                image_data,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                ds9mode=False,  # note that the polynomials are fitted using array indices (0-based)
                title=title + f"\nSlice {islice + 1} (ID {sliceid})",
            )
            # Plot the boundary polynomials
            if input_poly is not None or input_traces is not None:
                plot_fitted_boundary_polynomials(
                    ax, list_poly_left, list_poly_right, voffset, sliceid=False, isliceplot=islice
                )
            # Overplot the boundary borders
            if input_borders is not None:
                plot_borders(
                    ax, array_left_border, array_right_border, ibad, voffset=voffset, sliceid=False, isliceplot=islice
                )
            # Overplot the slice traces
            if input_traces is not None:
                plot_traces(ax, list_poly_traces_all_slices, voffset=voffset, traceid=False, isliceplot=islice)
                if traceid:
                    ntraces = len(list_poly_traces_all_slices[islice])
                    xcenter = (FRIDA_NAXIS1_HAWAII.value - 1) / 2
                    ixdum = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
                    for itrace in range(ntraces):
                        poly_trace = list_poly_traces_all_slices[islice][itrace]
                        ytrace = poly_trace(ixdum)
                        ax.plot(ixdum, ytrace, color="cyan", lw=1.5, label=f"Trace {itrace+1}")
                        ycenter = poly_trace(xcenter)
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
            ixdum = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
            ymin = np.min(list_poly_left[islice](ixdum)) - 10
            ymax = np.max(list_poly_right[islice](ixdum)) + 10
            ax.set_ylim(ymin, ymax)
            plt.tight_layout()  # Fails if sliceid=True in plot_fitted_boundary_polynomials
            pdf_output.savefig(fig, bbox_inches="tight")
            plt.close(fig)
        pdf_output.close()
    else:
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
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        # Plot the boundary polynomials
        if input_poly is not None or input_traces is not None:
            plot_fitted_boundary_polynomials(ax, list_poly_left, list_poly_right, voffset, sliceid)
        # Overplot the boundary borders
        if input_borders is not None:
            sliceid_ = sliceid
            if input_poly is not None:
                sliceid_ = False  # Avoid overplotting slice IDs twice
            plot_borders(ax, array_left_border, array_right_border, ibad, voffset=voffset, sliceid=sliceid_)
        # Overplot the slice traces
        if input_traces is not None:
            plot_traces(ax, list_poly_traces_all_slices, voffset=voffset, traceid=traceid)

        # reset the x and y limits to the original values after plotting the boundaries (and borders if provided)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        mpl.rcParams["keymap.home"] = []  # disable 'h' and 'r'
        mpl.rcParams["keymap.back"] = []  # disable 'c'
        mpl.rcParams["keymap.forward"] = []  # disable 'v' (conflict with 'v' for setting vmin/vmax)
        init_xylimits = ax.get_xlim(), ax.get_ylim()

        def on_key(event):
            nonlocal vmin, vmax
            nonlocal init_xylimits
            if event.key == "?":
                logger.info("-" * 79)
                logger.info("'?': show this help message")
                logger.info("'a': toggle imshow aspect='equal' / aspect='auto'")
                logger.info("'h': reset zoom to initial limits")
                logger.info("'v': set vmin and vmax manually")
                logger.info("',': set vmin and vmax to min and max of the zoomed region")
                logger.info("'/': set vmin and vmax using zscale of the zoomed region")
                logger.info("'q': quit")
                logger.info("-" * 79)
            elif event.key == "h":
                xlim, ylim = init_xylimits
                ax.set_xlim(xlim)
                ax.set_ylim(ylim)
                ax.images[0].set_clim(vmin, vmax)
                fig.set_tight_layout(False)  # deactivate accumulated tight_layout adjustments
                fig.tight_layout()  # apply new tight_layout adjustments
                ax.figure.canvas.draw_idle()
                plt.pause(0.1)  # allow time for the figure to update
            elif event.key == "a":
                if ax.get_aspect() in ["equal", 1.0]:
                    logger.info("Setting aspect to 'auto'.")
                    ax.set_aspect("auto")
                else:
                    logger.info("Setting aspect to 'equal'.")
                    ax.set_aspect("equal")
                fig.set_tight_layout(False)  # deactivate accumulated tight_layout adjustments
                fig.tight_layout()  # apply new tight_layout adjustments
                fig.canvas.draw()
            elif event.key in ["v", ",", "/"]:
                if event.key == "v":
                    current_vmin, current_vmax = ax.images[0].get_clim()
                    vmin = input_number(expected_type="float", prompt="Enter vmin: ", default=current_vmin)
                    vmax = input_number(expected_type="float", prompt="Enter vmax: ", default=current_vmax)
                else:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    x1, x2 = int(xlim[0]), int(xlim[1])
                    y1, y2 = int(ylim[0]), int(ylim[1])
                    x1 = max(0, min(x1, FRIDA_NAXIS1_HAWAII.value - 1))
                    x2 = max(0, min(x2, FRIDA_NAXIS1_HAWAII.value - 1))
                    y1 = max(0, min(y1, FRIDA_NAXIS2_HAWAII.value - 1))
                    y2 = max(0, min(y2, FRIDA_NAXIS2_HAWAII.value - 1))
                    if event.key == ",":
                        vmin = np.nanmin(image_data[y1 : y2 + 1, x1 : x2 + 1])
                        vmax = np.nanmax(image_data[y1 : y2 + 1, x1 : x2 + 1])
                    else:
                        vmin, vmax = ZScaleInterval().get_limits(image_data[y1 : y2 + 1, x1 : x2 + 1])
                    logger.info(f"Setting vmin={vmin}, vmax={vmax} for the zoomed region.")
                ax.images[0].set_clim(vmin, vmax)
                ax.figure.canvas.draw_idle()
                plt.pause(0.1)  # allow time for the figure to update
            elif event.key == "q":
                plt.close(fig)

        on_key(event=types.SimpleNamespace(key="?"))  # Show help message on startup
        fig.canvas.mpl_connect("key_press_event", on_key)
        # fig.set_tight_layout(False)  # deactivate accumulated tight_layout adjustments
        fig.tight_layout()  # apply new tight_layout adjustments
        # instead of plt.show(), use a loop to keep the figure open until closed by the user
        # (otherwise, after using input_number() in the on_key function, the matplotlib event
        # loop is not properly restored and the execution of the code continues as if the
        # figure was closed, which is not the case)
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)


def main(args=None):

    datetime_ini = datetime.now()

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
    parser.add_argument("--image", help="Image to display boundaries on", type=str, required=True)
    parser.add_argument("--voffset", help="Vertical constant offset (pixels) to apply", type=float, default=0.0)
    parser.add_argument("--sliceid", help="Overplot slice ID", action="store_true")
    parser.add_argument("--traceid", help="Overplot trace ID", action="store_true")
    parser.add_argument(
        "--pdf-mosaic", help="Output PDF file to save zoomed images of all the slices", type=str, required=False
    )
    parser.add_argument("--output-dir", help="Output directory (default: .)", type=str, default=".")
    parser.add_argument("--record", help="Record terminal output", action="store_true")
    parser.add_argument("--echo", help="Display full command line", action="store_true")
    parser.add_argument("--version", help="Display version", action="store_true")
    parser.add_argument(
        "--log-level",
        help="Set the logging level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    args = parser.parse_args(args)

    # Initialize the script with the provided arguments
    console, logger = initialize_script_with_args(sys.argv, parser, args, __name__)

    # Check input files are defined
    if args.poly is None and args.borders is None and args.traces is None:
        raise ValueError(
            "At least one of the input files (--poly or --borders or --traces) must be defined to overplot the slice boundaries."
        )
    if args.poly is not None and args.traces is not None:
        raise ValueError(
            "Both --poly and --traces are defined. Please define only one of them to overplot the slice boundaries."
        )

    # Check the input image file is defined if the user wants to overplot the boundaries
    if args.image is None:
        logger.warning("No input image file defined. The slice boundaries will not be overplotted on an image.")

    # Check that both sliceid and traceid are not defined at the same time
    if args.sliceid and args.traceid:
        raise ValueError(
            "Both --sliceid and --traceid are defined. Please define only one of them to overplot the IDs."
        )

    # Check that if traceid is defined, traces file must be provided
    if args.traceid and args.traces is None:
        raise ValueError(
            "The --traceid option is defined, but no slice trace polynomials file is provided. Please provide a valid --traces file."
        )

    # Overplot the slice boundary polynomials
    overplot_slice_boundary_polynomials(
        input_poly=args.poly,
        input_borders=args.borders,
        input_traces=args.traces,
        image=args.image,
        voffset=args.voffset,
        sliceid=args.sliceid,
        traceid=args.traceid,
        pdf_mosaic=args.pdf_mosaic,
        output_dir=args.output_dir,
    )

    # Execution time
    datetime_end = datetime.now()
    time_elapsed = datetime_end - datetime_ini
    logger.info("Total time elapsed: %s", str(time_elapsed))

    # Goodbye message
    console.rule("[bold magenta] Goodbye! [/bold magenta]")

    # Save console log if recording is enabled
    if args.record:
        log_filename = "terminal_output.txt"
        output_dir_path = Path(args.output_dir)
        if not output_dir_path.exists():
            output_dir_path.mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / log_filename, "wt") as f:
            f.write(console.export_text(styles=True))
        logger.info(f"terminal output recorded in [green]{log_filename}[/green]")


if __name__ == "__main__":
    main()
