#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Compute the mapping from slice traces and an arc image."""

import argparse
from astropy.io import fits
from astropy.visualization import ZScaleInterval
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rich_argparse import RichHelpFormatter
import teareduce as tea

from numina.tools.initialize_script_with_args import NuminaScriptDefinition

from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NSLICES
from fridadrp.tools.check_plot_limits import check_plot_limits
from fridadrp.tools.overplot_slice_boundary_polynomials import interactive_image_display
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_fitted_boundary_polynomials
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_traces
from fridadrp.tools.overplot_slice_boundary_polynomials import polynomial_from_voffset
from fridadrp.tools.read_arc_lines import read_arc_lines
from fridadrp.tools.read_slice_boundary_polynomials import read_slice_boundary_polynomials
from fridadrp.tools.read_slice_trace_polynomials import read_slice_trace_polynomials


def plot_arc_lines(ax, list_arc_line_tables, color="red", markersize=2):
    """Overplot the positions of the detected arc lines on the given axes

    The positions are taken from the columns 'x_peak' (0-based array index
    along NAXIS1) and 'row_index' (0-based array index along NAXIS2) of the
    tables, so the axes must use 0-based array indices.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the arc lines.
    list_arc_line_tables : list of astropy.table.Table or None
        Tables with the arc line detections of each slice, as returned by
        read_arc_lines. None elements and empty tables are ignored.
    color : str, optional
        Color of the markers.
    markersize : float, optional
        Size of the markers.

    Returns
    -------
    line : matplotlib.lines.Line2D or None
        The line with all the plotted points, or None if there are no
        detections to plot.
    """
    list_tables = [table for table in list_arc_line_tables if table is not None and len(table) > 0]
    if len(list_tables) == 0:
        return None
    # A single Line2D with the points of all the slices
    x_peak = np.concatenate([np.asarray(table["x_peak"], dtype=float) for table in list_tables])
    row_index = np.concatenate([np.asarray(table["row_index"], dtype=float) for table in list_tables])
    (line,) = ax.plot(x_peak, row_index, color=color, marker=".", markersize=markersize, linestyle="none")
    return line


def compute_fmap_from_traces_arc(
    arc_image,
    arc_lines,
    traces_file,
    voffset=[0.0],
    show_sliceid=False,
    xlim=None,
    ylim=None,
    vlim=None,
):
    """Compute the mapping from slice traces and an arc image.

    Parameters
    ----------
    arc_image : str
        Path to the arc image file.
    arc_lines : str
        Path to the file with arc lines found by fridadrp-find_arc_lines.
        Their positions are overplotted on the arc image (without applying
        voffset, since they are measured positions in the image).
    traces_file : str
        Path to the file with the slice trace polynomials.
    voffset : list[float], optional
        Vertical offset to apply to the slice boundaries (default: 0.0 pixels; + upward, - downward).
        If more than one value is provided, a polynomial of degree len(voffset)-1 will be fitted
        to the offsets and applied to the slice boundaries.
    show_sliceid : bool, optional
        Whether to overplot slice IDs (default: False).
    xlim : list[float], optional
        X-axis limits for the plot (xmin, xmax); array indices (1-based along NAXIS1 axis).
    ylim : list[float], optional
        Y-axis limits for the plot (ymin, ymax); array indices (1-based along NAXIS2 axis).
    vlim : list[float], optional
        Background and foreground limits (vmin, vmax).

    Raises
    ------
    ValueError
        If xlim, ylim or vlim are not None and do not contain two finite
        and different numbers, or if vlim[0] >= vlim[1].
    """
    logger = logging.getLogger(__name__)

    # Check the plot limits before reading any file
    check_plot_limits(xlim, "xlim")
    check_plot_limits(ylim, "ylim")
    check_plot_limits(vlim, "vlim", require_increasing=True)

    # Read the arc image
    with fits.open(arc_image) as hdulist:
        arc_data = hdulist[0].data

    # Determine the voffset polynomial if more than one value is provided
    poly_voffset = polynomial_from_voffset(voffset)

    # Read the slice trace polynomials
    list_poly_traces_all_slices = read_slice_trace_polynomials(traces_file)
    logger.info(
        f"Read {len(list_poly_traces_all_slices)} slices with {len(list_poly_traces_all_slices[0])} traces "
        f"per slice from \n[blue]{traces_file}[/blue]"
    )
    # Apply requested voffset to the slice trace polynomials
    for islice in range(FRIDA_NSLICES):
        for itrace in range(len(list_poly_traces_all_slices[islice])):
            if list_poly_traces_all_slices[islice][itrace] is not None:
                list_poly_traces_all_slices[islice][itrace] += poly_voffset

    # Read the slice boundary polynomials (also stored in the traces file)
    list_poly_left, list_poly_right, _ = read_slice_boundary_polynomials(traces_file)
    # Apply requested voffset to the slice boundary polynomials
    for islice in range(FRIDA_NSLICES):
        if list_poly_left[islice] is not None:
            list_poly_left[islice] += poly_voffset
        if list_poly_right[islice] is not None:
            list_poly_right[islice] += poly_voffset

    # Read the arc line detections (0-based array indices). The voffset is not applied
    # to them: they are positions measured in the arc image, while voffset corrects
    # the boundary and trace polynomials.
    list_arc_line_tables = read_arc_lines(arc_lines)
    nslices_arc_lines = sum(table is not None for table in list_arc_line_tables)
    ndetections = sum(len(table) for table in list_arc_line_tables if table is not None)
    logger.info(f"Read {ndetections} arc line detections in {nslices_arc_lines} slices from\n[blue]{arc_lines}[/blue]")
    image_in_arc_lines = fits.getheader(arc_lines)["IMAGE"]
    if image_in_arc_lines != Path(arc_image).name:
        logger.warning(
            f"The arc lines were found in {image_in_arc_lines}, which is not the arc image {Path(arc_image).name}"
        )

    # Compute the mapping from slice traces and arc image
    # (Implementation of the mapping computation goes here)

    # Plot the arc image and overplot the slice boundaries
    title = f"Image: {Path(arc_image).name}"
    title += f"\nSlice trace polynomials: {Path(traces_file).name}"
    if vlim is not None:
        vmin, vmax = vlim
    else:
        vmin, vmax = ZScaleInterval().get_limits(arc_data)
    fig, ax = plt.subplots(figsize=(10, 8))
    tea.imshow(
        fig,
        ax,
        arc_data,
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
    # The axis limits must be set before plotting the boundary polynomials
    # (they are evaluated within the current x-axis limits)
    plot_fitted_boundary_polynomials(ax, list_poly_left, list_poly_right, show_sliceid)
    plot_traces(ax, list_poly_traces_all_slices, show_traceid=False)
    plot_arc_lines(ax, list_arc_line_tables)

    interactive_image_display(fig, ax, arc_data, vmin, vmax)


def main(args=None):

    parser = argparse.ArgumentParser(
        description="Compute mapping from slice traces and arc image",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("--arc-image", help="Arc image file", type=str, required=True)
    parser.add_argument(
        "--arc-lines", help="File with arc lines found by fridadrp-find_arc_lines", type=str, required=True
    )
    parser.add_argument(
        "--traces", help="Path to the file with the slice trace polynomials (optional)", type=str, required=True
    )
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

    # Include default arguments for common actions, and initialize console and logging
    myscript = NuminaScriptDefinition(parser, argv=args)
    args = myscript.args
    logger = myscript.logger

    # Check the input files are defined
    if args.arc_image is None:
        raise ValueError("No input arc image file defined. Please provide a valid --arc-image file.")
    if args.arc_lines is None:
        raise ValueError("No input arc lines file defined. Please provide a valid --arc-lines file.")
    if args.traces is None:
        raise ValueError("No input slice trace polynomials file defined. Please provide a valid --traces file.")

    # Compute the mapping from slice traces and an arc image
    logger.info("Computing mapping from slice traces and arc image...")
    compute_fmap_from_traces_arc(
        arc_image=args.arc_image,
        arc_lines=args.arc_lines,
        traces_file=args.traces,
        voffset=args.voffset,
        show_sliceid=args.show_sliceid,
        xlim=args.xlim,
        ylim=args.ylim,
        vlim=args.vlim,
    )

    # Display goodbye message and save console if recording is enabled
    myscript.goodbye_message_and_save_console()


if __name__ == "__main__":
    main()
