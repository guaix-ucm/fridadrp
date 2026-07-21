#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Overplot the slice boundary polynomials on image"""

import argparse
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from datetime import datetime
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import Polynomial
from pathlib import Path
from rich.logging import RichHandler
from rich_argparse import RichHelpFormatter
import sys
import teareduce as tea
import types

from numina.tools.input_number import input_number
from numina.user.console import NuminaConsole

from fridadrp._version import version
from fridadrp.core import FRIDA_NAXIS1_HAWAII, FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import slicenum_from_index


def read_slice_boundary_polynomials(input):
    """Read the slice boundary polynomials from a FITS file

    Parameters
    ----------
    input : str
        Path to the FITS file containing the slice boundary polynomials.

    Returns
    -------
    list_poly_left : list of Polynomial
        List of Polynomial objects for the left slice boundaries.
    list_poly_right : list of Polynomial
        List of Polynomial objects for the right slice boundaries.
    """
    logger = logging.getLogger(__name__)

    with fits.open(input) as hdul:
        if "KEYCODE" not in hdul[0].header or hdul[0].header["KEYCODE"] != "SLICE_BOUNDARY_POLYNOMIALS":
            raise ValueError(f"Input file {input} is not a valid FITS file with slice boundary polynomials (missing or incorrect KEYCODE).")
        array_coefs_left = hdul["L-BORDER"].data
        naxis2_left, deg_left = array_coefs_left.shape
        if naxis2_left != FRIDA_NSLICES:
            raise ValueError(f"Input file {input} has {naxis2_left} slices, but FRIDA_NSLICES is {FRIDA_NSLICES}.")
        logger.info(f"Reading {naxis2_left} slices with polynomial degree {deg_left-1}.")
        list_poly_left = [Polynomial(array_coefs_left[islice]) for islice in range(FRIDA_NSLICES)]
        array_coefs_right = hdul["R-BORDER"].data
        naxis2_right, deg_right = array_coefs_right.shape
        if naxis2_right != FRIDA_NSLICES:
            raise ValueError(f"Input file {input} has {naxis2_right} slices, but FRIDA_NSLICES is {FRIDA_NSLICES}.")
        if deg_right != deg_left:
            raise ValueError(f"Input file {input} has different polynomial degrees for left and right borders: {deg_left-1} and {deg_right-1}.")
        logger.info(f"Reading {naxis2_right} slices with polynomial degree {deg_right-1}.")
        list_poly_right = [Polynomial(array_coefs_right[islice]) for islice in range(FRIDA_NSLICES)]

    return list_poly_left, list_poly_right


def plot_fitted_boundaries(ax, list_poly_left, list_poly_right, voffset=0.0, sliceid=False):
    """Plot the fitted slice boundary polynomials on the given axes

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the polynomials.
    list_poly_left : list of numpy.polynomial.Polynomial
        The list of left slice boundary polynomials.
    list_poly_right : list of numpy.polynomial.Polynomial
        The list of right slice boundary polynomials.
    voffset : float, optional
        Vertical (constant) offset to apply to the polynomials.
    sliceid : bool, optional
        If True, overplot the slice ID at the center of each slice.
    """
    xmin, xmax = ax.get_xlim()
    xdum = np.linspace(xmin, xmax, 1000)
    for islice in range(FRIDA_NSLICES):
        ax.plot(xdum, list_poly_left[islice](xdum) + voffset, color="white", lw=5.0, alpha=0.7)
        ax.plot(xdum, list_poly_left[islice](xdum) + voffset, color="C0", lw=2.0, alpha=1.0)
        ax.plot(xdum, list_poly_right[islice](xdum) + voffset, color="white", lw=5.0, alpha=0.7)
        ax.plot(xdum, list_poly_right[islice](xdum) + voffset, color="C1", lw=2.0, alpha=1.0)
        if sliceid:
            xcenter = (FRIDA_NAXIS1_HAWAII.value - 1) / 2
            ycenter = (list_poly_left[islice](xcenter) + list_poly_right[islice](xcenter)) / 2
            ax.text(xcenter, ycenter, f"#{slicenum_from_index(islice)}", 
                    color="white", fontsize=12,
                    ha="center", va="center", fontweight="bold", alpha=1.0)


def overplot_slice_boundary_polynomials(input, image, voffset=0.0, sliceid=False):
    """Overplot the slice boundary polynomials on an image

    The polynomials are assumed to be fitted using as independent variable
    the array index along the NAXIS1 axis, which ranges from 0 to FRIDA_NAXIS1_HAWAII-1, 
    and as dependent variable the array index along the NAXIS2 axis, 
    which ranges from 0 to FRIDA_NAXIS2_HAWAII-1.

    Parameters
    ----------
    input : str
        Path to the FITS file containing the slice boundary polynomials.
    image : str
        Path to the FITS file containing the image on which to overplot
        the slice boundaries.
    voffset : float, optional
        Vertical (constant) offset to apply to the polynomials.
    sliceid : bool, optional
        If True, overplot the slice ID at the center of each slice.
    """
    logger = logging.getLogger(__name__)

    # Read the polynomial coefficients from the input FITS file
    list_poly_left, list_poly_right = read_slice_boundary_polynomials(input)

    # Read the image data from the input FITS file
    with fits.open(image) as hdul:
        image_data = hdul[0].data

    fig, ax = plt.subplots(figsize=(10, 8))
    vmin, vmax = ZScaleInterval().get_limits(image_data)
    tea.imshow(fig, ax, image_data, vmin=vmin, vmax=vmax, aspect="auto",
               title=f"Image: {Path(image).name}\nPolynomials: {Path(input).name}")
    plot_fitted_boundaries(ax, list_poly_left, list_poly_right, voffset, sliceid)
    mpl.rcParams["keymap.home"] = []    # disable 'h' and 'r'
    mpl.rcParams["keymap.back"] = []    # disable 'c'
    mpl.rcParams["keymap.forward"] = [] # disable 'v' (conflict with 'v' for setting vmin/vmax)
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
                vmin = input_number(expected_type='float', prompt="Enter vmin: ", default=current_vmin)
                vmax = input_number(expected_type='float', prompt="Enter vmax: ", default=current_vmax)
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
                    vmin = np.nanmin(image_data[y1:y2 + 1, x1:x2 + 1])
                    vmax = np.nanmax(image_data[y1:y2 + 1, x1:x2 + 1])
                else:
                    vmin, vmax = ZScaleInterval().get_limits(image_data[y1:y2 + 1, x1:x2 + 1])
                logger.info(f"Setting vmin={vmin}, vmax={vmax} for the zoomed region.")
            ax.images[0].set_clim(vmin, vmax)
            ax.figure.canvas.draw_idle()
            plt.pause(0.1)  # allow time for the figure to update
        elif event.key == "q":
            plt.close(fig)

    on_key(event=types.SimpleNamespace(key="?"))  # Show help message on startup
    fig.canvas.mpl_connect("key_press_event", on_key)
    #fig.set_tight_layout(False)  # deactivate accumulated tight_layout adjustments
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
        description="Overplot the slice boundaries polynomial on image", formatter_class=RichHelpFormatter
    )
    parser.add_argument("--input", help="Path to the file with the boundary polynomials", type=str, required=True)
    parser.add_argument("--image", help="Image to display boundaries on", type=str, required=True)
    parser.add_argument("--voffset", help="Vertical (constant) offset to apply to the polynomials", type=float, default=0.0)
    parser.add_argument("--sliceid", help="Overplot slice ID", action="store_true")
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

    if len(sys.argv) == 1:
        parser.print_usage()
        raise SystemExit()

    # Configure rich console
    console = NuminaConsole(record=args.record)

    if args.version:
        console.print(version)
        raise SystemExit()

    if args.echo:
        console.print(f"[bright_red]Executing:\n{' '.join(sys.argv)}[/bright_red]\n", end="")

    # Configure logging
    if args.log_level in ["DEBUG", "WARNING", "ERROR", "CRITICAL"]:
        format_log = "%(name)s %(levelname)s %(message)s"
        handlers = [RichHandler(console=console, show_time=False, markup=True)]
    else:
        format_log = "%(message)s"
        handlers = [RichHandler(console=console, show_time=False, markup=True, show_path=False, show_level=False)]
    logging.basicConfig(level=args.log_level, format=format_log, handlers=handlers)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)  # Suppress matplotlib debug logs

    # Welcome message
    console.rule(f"[bold magenta]Welcome to fridadrp-overplot_slice_boundaries_polynomials[/bold magenta]")

    # Display version info
    logger = logging.getLogger(__name__)
    logger.info(f"Using {__name__} version {version}")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Command line arguments: {args}")

    # Check input polynomials file is defined
    if args.input is None:
        raise ValueError("Input file is not defined. Use --input to specify the input file.")

    # Check the input image file is defined if the user wants to overplot the boundaries
    if args.image is None:
        logger.warning("No input image file defined. The slice boundaries will not be overplotted on an image.")

    # Overplot the slice boundary polynomials
    overplot_slice_boundary_polynomials(
        input=args.input, 
        image=args.image,
        voffset=args.voffset,
        sliceid=args.sliceid
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
        with open(Path(args.output_dir) / log_filename, "wt") as f:
            f.write(console.export_text(styles=True))
        logger.info(f"terminal output recorded in [green]{log_filename}[/green]")


if __name__ == "__main__":
    main()
