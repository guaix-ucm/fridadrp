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
from astropy.table import Table
import astropy.units as u
from astropy.visualization import ZScaleInterval
import logging
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from rich_argparse import RichHelpFormatter
import teareduce as tea

from numina.array.display.polfit_residuals import polfit_residuals
from numina.tools.initialize_script_with_args import NuminaScriptDefinition

from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import sliceid_from_sliceindex
from fridadrp.core import slicenum_list_from_slice_lists
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


def fit_arc_line_polynomials(arc_line_table, deg):
    """Fit a polynomial to each arc line of a slice

    Since the arc lines run along NAXIS2 within each slice, the fitted
    polynomial gives the position of the line along NAXIS1 ('x_peak',
    0-based array index) as a function of the position along NAXIS2
    ('row_index', 0-based array index). Note that this is the opposite
    of the slice boundary and trace polynomials. The polynomial is
    returned in the unscaled numpy.polynomial.Polynomial basis.

    Parameters
    ----------
    arc_line_table : astropy.table.Table
        Table with the arc line detections of a single slice, as returned
        by read_arc_lines, with the columns 'feature_id', 'row_index' and
        'x_peak'. Each arc line is identified by its 'feature_id'.
    deg : int
        Degree of the polynomial fitted to each arc line.

    Returns
    -------
    list_arc_line_fits : list of dict
        One dictionary for each fitted arc line, with the keys:
        'feature_id' (int), 'poly' (numpy.polynomial.Polynomial),
        'row_min' and 'row_max' (float, range of row_index of the fitted
        points), 'npoints' (int, number of fitted points) and 'rms'
        (float, rms of the residuals, in pixels along NAXIS1). Arc lines
        with fewer than deg + 1 points with finite x_peak are skipped.
    """
    logger = logging.getLogger(__name__)

    feature_id = np.asarray(arc_line_table["feature_id"])
    row_index = np.asarray(arc_line_table["row_index"], dtype=float)
    x_peak = np.asarray(arc_line_table["x_peak"], dtype=float)

    list_arc_line_fits = []
    for fid in np.unique(feature_id):
        mask = (feature_id == fid) & np.isfinite(x_peak)
        npoints = int(np.sum(mask))
        if npoints < deg + 1:
            logger.debug(f"Arc line {fid} has {npoints} valid points (< deg + 1 = {deg + 1}). Skipping it.")
            continue
        poly, residuals = polfit_residuals(
            x=row_index[mask],
            y=x_peak[mask],
            deg=deg,
            xlabel="row_index (0-based array index along NAXIS2)",
            ylabel="x_peak (0-based array index along NAXIS1)",
            title=f"Arc line {fid}",
            debugplot=0,
        )
        list_arc_line_fits.append(
            dict(
                feature_id=int(fid),
                poly=poly,
                row_min=float(np.min(row_index[mask])),
                row_max=float(np.max(row_index[mask])),
                npoints=npoints,
                rms=float(np.sqrt(np.mean(residuals**2))),
            )
        )

    return list_arc_line_fits


def plot_arc_line_polynomials(ax, list_arc_line_fits, color="yellow", linewidth=1):
    """Overplot the polynomials fitted to the arc lines on the given axes

    Each polynomial is evaluated within the range of row_index of its
    fitted points. The axes must use 0-based array indices.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the polynomials.
    list_arc_line_fits : list of dict
        Fitted arc lines, as returned by fit_arc_line_polynomials.
    color : str, optional
        Color of the lines.
    linewidth : float, optional
        Width of the lines.

    Returns
    -------
    list_lines : list of matplotlib.lines.Line2D
        One line for each fitted arc line.
    """
    list_lines = []
    for arc_line_fit in list_arc_line_fits:
        # the polynomial gives the NAXIS1 position as a function of the NAXIS2 position
        ydum = np.linspace(arc_line_fit["row_min"], arc_line_fit["row_max"], 100)
        (line,) = ax.plot(arc_line_fit["poly"](ydum), ydum, color=color, linewidth=linewidth)
        list_lines.append(line)
    return list_lines


def compute_reference_points(list_arc_line_fits, poly_left, poly_right, list_poly_traces, maxiter=50, tol=1e-10):
    """Compute the intersections of the arc lines with the boundaries and traces of a slice

    Each arc line is described by x = P(y), and each slice boundary or
    trace by y = Q(x), where x and y are 0-based array indices along
    NAXIS1 and NAXIS2, respectively. Their intersection is the solution
    of y = Q(P(y)), computed with the fixed-point iteration
    y_{k+1} = Q(P(y_k)), starting at y_0 = Q(P(y_mid)), where y_mid is
    the centre of the row range of the arc line.

    The iteration converges because both curves are nearly perpendicular:
    the convergence factor |dQ/dx * dP/dy| is the product of two small
    slopes, so a few iterations are enough to reach the tolerance.

    Parameters
    ----------
    list_arc_line_fits : list of dict
        Fitted arc lines of the slice, as returned by fit_arc_line_polynomials.
    poly_left : numpy.polynomial.Polynomial or None
        Left (lower) boundary polynomial of the slice. Skipped if None.
    poly_right : numpy.polynomial.Polynomial or None
        Right (upper) boundary polynomial of the slice. Skipped if None.
    list_poly_traces : list of numpy.polynomial.Polynomial
        Trace polynomials of the slice, in the order of the traces file.
    maxiter : int, optional
        Maximum number of iterations.
    tol : float, optional
        Convergence tolerance (pixels): the iteration stops when the
        change in y is smaller than this value.

    Returns
    -------
    reference_points : astropy.table.Table
        One row for each pair (arc line, curve), with the columns:
        'feature_id' (arc line), 'curve_index' (0 = left boundary,
        1 to N = traces, N + 1 = right boundary, where N is the number
        of traces), 'curve_type' ('left', 'trace' or 'right'), 'x' and
        'y' (intersection, 0-based array indices along NAXIS1 and NAXIS2;
        NaN if the iteration did not converge), 'niter' (number of
        iterations), 'converged' (bool) and 'extrapolation' (distance, in
        pixels along NAXIS2, of the intersection outside the row range
        employed to fit the arc line; 0 if it is inside).
    """
    # Curves ordered along NAXIS2: (curve_index, curve_type, polynomial)
    list_curves = []
    if poly_left is not None:
        list_curves.append((0, "left", poly_left))
    for itrace, poly_trace in enumerate(list_poly_traces):
        if poly_trace is not None:
            list_curves.append((itrace + 1, "trace", poly_trace))
    if poly_right is not None:
        list_curves.append((len(list_poly_traces) + 1, "right", poly_right))
    ncurves = len(list_curves)
    curve_index = np.array([curve[0] for curve in list_curves], dtype=int)
    curve_type = np.array([curve[1] for curve in list_curves], dtype="U5")

    def evaluate_curves(x):
        # y = Q_j(x_j) for every curve j
        return np.array([poly(xj) for (_, _, poly), xj in zip(list_curves, x)], dtype=float)

    columns = {
        "feature_id": [],
        "curve_index": [],
        "curve_type": [],
        "x": [],
        "y": [],
        "niter": [],
        "converged": [],
        "extrapolation": [],
    }
    for arc_line_fit in list_arc_line_fits:
        poly_arc = arc_line_fit["poly"]
        row_min, row_max = arc_line_fit["row_min"], arc_line_fit["row_max"]
        # Iterate simultaneously for all the curves of the slice
        y_mid = 0.5 * (row_min + row_max)
        y = evaluate_curves(np.full(ncurves, poly_arc(y_mid)))
        niter = np.zeros(ncurves, dtype=int)
        converged = np.zeros(ncurves, dtype=bool)
        with np.errstate(over="ignore", invalid="ignore"):
            for k in range(1, maxiter + 1):
                active = ~converged
                if not np.any(active):
                    break
                y_new = evaluate_curves(poly_arc(y))
                delta = np.abs(y_new - y)
                y = np.where(active, y_new, y)  # converged points are kept fixed
                niter[active] = k
                converged |= active & (delta < tol)
            x = poly_arc(y)
        x[~converged] = np.nan
        y[~converged] = np.nan
        # Distance outside the row range of the fitted arc line (NaN for not converged points)
        extrapolation = np.maximum(np.maximum(row_min - y, y - row_max), 0.0)
        extrapolation[~converged] = np.nan
        columns["feature_id"].append(np.full(ncurves, arc_line_fit["feature_id"], dtype=int))
        columns["curve_index"].append(curve_index)
        columns["curve_type"].append(curve_type)
        columns["x"].append(x)
        columns["y"].append(y)
        columns["niter"].append(niter)
        columns["converged"].append(converged)
        columns["extrapolation"].append(extrapolation)

    dtypes = dict(
        feature_id=int,
        curve_index=int,
        curve_type="U5",
        x=float,
        y=float,
        niter=int,
        converged=bool,
        extrapolation=float,
    )
    reference_points = Table(
        {
            name: np.concatenate(values) if len(values) > 0 else np.array([], dtype=dtypes[name])
            for name, values in columns.items()
        }
    )
    for colname in ["x", "y", "extrapolation"]:
        reference_points[colname].unit = u.pix
    return reference_points


def plot_reference_points(ax, list_reference_tables, color="C1", marker="o", markersize=6):
    """Overplot the reference points (converged intersections) on the given axes

    The axes must use 0-based array indices.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to plot the reference points.
    list_reference_tables : list of astropy.table.Table or None
        Reference points of each slice, as returned by
        compute_reference_points. None elements are ignored, and only
        the converged points are plotted.
    color : str, optional
        Color of the markers.
    marker : str, optional
        Marker style.
    markersize : float, optional
        Size of the markers.

    Returns
    -------
    line : matplotlib.lines.Line2D or None
        The line with all the plotted points, or None if there are no
        converged points to plot.
    """
    list_tables = [table[table["converged"]] for table in list_reference_tables if table is not None]
    list_tables = [table for table in list_tables if len(table) > 0]
    if len(list_tables) == 0:
        return None
    # A single Line2D with the points of all the slices
    x = np.concatenate([np.asarray(table["x"], dtype=float) for table in list_tables])
    y = np.concatenate([np.asarray(table["y"], dtype=float) for table in list_tables])
    (line,) = ax.plot(x, y, color=color, marker=marker, markersize=markersize, linestyle="none")
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
    deg=2,
    slicenum_list=[0],
    sliceid_list=[0],
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
    deg : int, optional
        Degree of the polynomial fitted to each arc line (default: 2).
        The polynomial gives the position along NAXIS1 as a function of
        the position along NAXIS2 (0-based array indices).
    slicenum_list : list of int, optional
        List of slice numbers to process (1-based). If [0], all slices
        will be processed. This parameter is mutually exclusive with `sliceid_list`.
    sliceid_list : list of int, optional
        List of slice IDs to process (1-based). If [0], all slices
        will be processed. This parameter is mutually exclusive with `slicenum_list`.

    Raises
    ------
    ValueError
        If xlim, ylim or vlim are not None and do not contain two finite
        and different numbers, if vlim[0] >= vlim[1], if deg is negative,
        or if slicenum_list and sliceid_list are not valid.
    """
    logger = logging.getLogger(__name__)

    # Check the input parameters before reading any file
    check_plot_limits(xlim, "xlim")
    check_plot_limits(ylim, "ylim")
    check_plot_limits(vlim, "vlim", require_increasing=True)
    if deg < 0:
        raise ValueError(f"{deg=} must be non-negative")
    slicenum_to_process = slicenum_list_from_slice_lists(slicenum_list, sliceid_list)

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

    # Fit a polynomial to each arc line of the selected slices (x_peak as a function
    # of row_index, both 0-based array indices), and compute the reference points
    # (intersections of the arc lines with the slice boundaries and traces)
    list_arc_line_fits_all_slices = [None] * FRIDA_NSLICES
    list_reference_points_all_slices = [None] * FRIDA_NSLICES
    for islice in range(FRIDA_NSLICES):
        slicenum = islice + 1
        sliceid = sliceid_from_sliceindex(islice)
        if slicenum not in slicenum_to_process:
            continue
        arc_line_table = list_arc_line_tables[islice]
        if arc_line_table is None or len(arc_line_table) == 0:
            logger.warning(f"Slice #{slicenum:2d} (Slice ID: {sliceid:2d}) has no arc line detections. Skipping it.")
            continue
        list_arc_line_fits = fit_arc_line_polynomials(arc_line_table, deg)
        nlines = len(np.unique(arc_line_table["feature_id"]))
        logger.info(
            f"Slice #{slicenum:2d} (Slice ID: {sliceid:2d}): {len(list_arc_line_fits)} arc lines fitted "
            f"({nlines - len(list_arc_line_fits)} skipped)"
        )
        list_arc_line_fits_all_slices[islice] = list_arc_line_fits
        reference_points = compute_reference_points(
            list_arc_line_fits,
            list_poly_left[islice],
            list_poly_right[islice],
            list_poly_traces_all_slices[islice],
        )
        list_reference_points_all_slices[islice] = reference_points
        converged = np.asarray(reference_points["converged"])
        if np.any(converged):
            logger.info(
                f"Slice #{slicenum:2d} (Slice ID: {sliceid:2d}): {np.sum(converged)} reference points "
                f"(max. iterations: {np.max(reference_points['niter'][converged])}, "
                f"max. extrapolation: {np.max(reference_points['extrapolation'][converged]):.2f} pix)"
            )
        if np.any(~converged):
            logger.warning(
                f"Slice #{slicenum:2d} (Slice ID: {sliceid:2d}): {np.sum(~converged)} reference points "
                "did not converge"
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
    for list_arc_line_fits in list_arc_line_fits_all_slices:
        if list_arc_line_fits is not None:
            plot_arc_line_polynomials(ax, list_arc_line_fits)
    plot_reference_points(ax, list_reference_points_all_slices)

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
    parser.add_argument(
        "--deg",
        help="Degree of the polynomial fitted to each arc line (default: 2)",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--slicenum-list",
        help=f"Slice number (from 1 to {FRIDA_NSLICES}; 0=ALL, default)",
        type=int,
        nargs="+",
        default=[0],
        choices=list(range(0, FRIDA_NSLICES + 1)),
    )
    parser.add_argument(
        "--sliceid-list",
        help=f"Slice ID (from 1 to {FRIDA_NSLICES}; 0=ALL, default)",
        type=int,
        nargs="+",
        default=[0],
        choices=list(range(0, FRIDA_NSLICES + 1)),
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
        deg=args.deg,
        slicenum_list=args.slicenum_list,
        sliceid_list=args.sliceid_list,
    )

    # Display goodbye message and save console if recording is enabled
    myscript.goodbye_message_and_save_console()


if __name__ == "__main__":
    main()
