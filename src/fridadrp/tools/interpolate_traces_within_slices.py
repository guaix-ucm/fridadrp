#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Interpolate/extrapolate traces within slices."""

import argparse
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter, generic_filter
import sys
from pathlib import Path
from rich_argparse import RichHelpFormatter
import shutil
import teareduce as tea
import uuid

from numina.array.display.polfit_residuals import polfit_residuals_with_sigma_rejection
from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history

from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import DEF_SLICEID_FROM_SLICEINDEX
from fridadrp.core import sliceindex_from_sliceid
from fridadrp.tools.check_output_file_overwrite import check_output_file_overwrite
from fridadrp.tools.columns_to_analyze_from_colranges import columns_to_analyze_from_colranges
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_fitted_boundary_polynomials
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_traces
from fridadrp.tools.read_slice_boundary_polynomials import read_slice_boundary_polynomials
from fridadrp.tools.read_slice_trace_polynomials import read_slice_trace_polynomials
from fridadrp.tools.initialize_script_with_args import initialize_script_with_args


def interpolate_traces_within_slices(image_path, input_traces, sliceids, skip_sliceids, columns_to_analyze, xmedian=21, degslice=2, plots=False):
    """Interpolate/extrapolate traces within slices.

    Each polynomial is fitted using as independent variable the array index
    along NAXIS1 (0-based) and as dependent variable the array index
    along NAXIS2 (0-based). The traces are fitted using a polynomial of the
    same degree as the original trace polynomials. The fitted polynomials are then
    evaluated at the specified columns to analyze, and the resulting values are used
    to update the trace polynomials for the specified slices.

    The traces are fitted using only the columns specified in `columns_to_analyze`.
    If this parameter is None, all columns within the useful image region are employed.
    The columns are specified as a list of tuples (1-based indices along NAXIS1),
    where each tuple contains the minimum and maximum column (1-based index)
    to analyze. For example, [(1, 100), (200, 300)] means that columns 1 to 100 
    and 200 to 300 will be used for fitting the traces.

    Parameters
    ----------
    image_path : str
        Path to the input image file (FITS format).
    input_traces : str
        Path to the file containing the slice trace polynomials.
    sliceids : list of int
        List of slice IDs to process.
    skip_sliceids : list of int
        List of slice IDs to skip.
    columns_to_analyze : list of int
        List of column indices to analyze.
    xmedian : int, optional
        Size of the median filter along NAXIS1 axis (odd; default: 21).
    degslice : int, optional
        Degree of the polynomial to fit traces across slices (default: 2).
    plots : bool, optional
        Whether to display plots of the interpolated/extrapolated traces (default: False).

    Returns
    -------
    list_poly_traces_all_slices : list of list of numpy.poly1d
        List of lists containing the trace polynomials for all slices.
    """
    logger = logging.getLogger(__name__)

    # Read the input image
    logger.debug(f"Reading input image from {image_path}")
    with fits.open(image_path) as hdul:
        image_data = hdul[0].data

    # Median filter the data to remove bad pixels. If there are NaN values,
    # use generic_filter with np.nanmedian to ignore NaN values.
    # Otherwise, use median_filter directly, which is faster.
    if xmedian % 2 == 0:
        xmedian += 1  # Ensure the median filter size is odd
        logger.warning(f"Median filter size adjusted to {xmedian} to ensure it is odd.")
    if xmedian >= 3:
        if np.isnan(image_data).any():
            logger.debug("NaN values found in image data. Using generic_filter with np.nanmedian to ignore NaN values.")
            image_data_filtered = generic_filter(image_data, np.nanmedian, size=(1, xmedian), mode="nearest")
        else:
            logger.debug("No NaN values found in image data. Using median_filter directly.")
            image_data_filtered = median_filter(image_data, size=(1, xmedian), mode="nearest")
    else:
        logger.warning(f"Median filter size {xmedian} is less than 3. Skipping median filtering.")
        image_data_filtered = image_data.copy()

    # Check that the slice IDs are valid
    for sliceid in sliceids:
        if sliceid < 0 or sliceid >= FRIDA_NSLICES:
            logger.error(f"Invalid slice ID {sliceid}. Must be between 0 and {FRIDA_NSLICES - 1}.")
            sys.exit(1)

    # Read first the boundary polynomials
    list_poly_left, list_poly_right, poldeg = read_slice_boundary_polynomials(input_traces)

    # Read also the trace polynomials
    list_poly_traces_all_slices = read_slice_trace_polynomials(input_traces)
    logger.info(
        f"Read {len(list_poly_traces_all_slices)} slices with {len(list_poly_traces_all_slices[0])} traces per slice from\n[blue]{input_traces}[/blue]."
    )
    deg = len(list_poly_traces_all_slices[0][0].coef) - 1  # Degree of the trace polynomials
    logger.info(f"Degree of the trace polynomials: {deg}")

    # Define columns to analyze based on the specified column ranges
    icolumns_to_analyze = np.array(columns_to_analyze) - 1  # Convert to 0-based index
    ncolumns_to_analyze = len(icolumns_to_analyze)
    if ncolumns_to_analyze == 0:
        raise ValueError("No columns to analyze. Please specify valid column ranges with --colrange.")
    elif ncolumns_to_analyze == 1:
        icolumns_to_plot = icolumns_to_analyze[0]
    elif ncolumns_to_analyze == 2:
        icolumns_to_plot = [icolumns_to_analyze[0], icolumns_to_analyze[1]]
    else:
        icolumns_to_plot = [
            icolumns_to_analyze[0],
            icolumns_to_analyze[len(icolumns_to_analyze) // 2],
            icolumns_to_analyze[-1],
        ]

    # Main loop to interpolate/extrapolate traces within slices
    sliceids_remaining = sliceids.copy()  # Keep track of remaining slice IDs to process
    for sliceid in sliceids:
        islice = sliceindex_from_sliceid(sliceid)
        ntraces = len(list_poly_traces_all_slices[islice])
        logger.info(f"Processing slice {islice + 1} (ID {sliceid}) with {ntraces} traces...")
        # Determine slice group
        slicesid_group1 = DEF_SLICEID_FROM_SLICEINDEX[0::2].tolist()
        slicesid_group2 = DEF_SLICEID_FROM_SLICEINDEX[1::2].tolist()
        if sliceid in slicesid_group1:
            slicesid_group = slicesid_group1
        else:
            slicesid_group = slicesid_group2
        logger.info(f"Slice {sliceid} belongs to group with slice IDs:\n{slicesid_group}")
        # Remove from slicesid_group the slices remaining
        for sid in sliceids_remaining:
            if sid in slicesid_group:
                slicesid_group.remove(sid)
        # Remove from slicesid_group the slices to skip
        for sid in skip_sliceids:
            if sid in slicesid_group:
                slicesid_group.remove(sid)
        logger.info(f"Useful slice IDs to perform interpolation/extrapolation:\n{slicesid_group}")
        # Fit polynomials to each trace across the slices in the group, using only the columns to analyze
        xfit = np.array([sliceindex_from_sliceid(sid) for sid in slicesid_group])  # (0-based)
        xpredicted = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
        list_poly_traces_slice = []
        for itrace in range(ntraces):
            ypredicted = np.full(FRIDA_NAXIS1_HAWAII.value, fill_value=np.nan, dtype=float)  # Initialize with NaN
            logger.info(f"Fitting trace {itrace + 1}/{ntraces} of slice ID {sliceid} along NAXIS1...")
            for icolumn in icolumns_to_analyze:
                debugplot = 0
                if icolumn in icolumns_to_plot and plots:
                    debugplot = 2
                yfit = np.array([
                    list_poly_traces_all_slices[sliceindex_from_sliceid(sid)][itrace](icolumn)
                    for sid in slicesid_group
                ])
                poly_trace_across_slices, _, _ = polfit_residuals_with_sigma_rejection(
                    x=xfit,
                    y=yfit,
                    deg=degslice,
                    times_sigma_reject=3.0,
                    ylimres_with_rejected=True,
                    xlabel="slice index",
                    ylabel="array index along NAXIS2",
                    title=f"Slice ID {sliceid}, Trace {itrace+1} / {ntraces}, Column {icolumn+1}",
                    debugplot=debugplot,
                )
                ypredicted[icolumn] = poly_trace_across_slices(sliceindex_from_sliceid(sliceid))
            # fit a polynomial of degree `deg` to determine the trace
            iok = ~np.isnan(ypredicted)
            if plots:
                debugplot = 2
            else:
                debugplot = 0
            poly_trace, _, _ = polfit_residuals_with_sigma_rejection(
                x=xpredicted[iok],
                y=ypredicted[iok],
                deg=deg,
                times_sigma_reject=3.0,
                ylimres_with_rejected=True,
                xlabel="array index along NAXIS1",
                ylabel="array index along NAXIS2",
                title=f"Slice ID {sliceid}, Predicted trace {itrace+1} / {ntraces}",
                debugplot=debugplot,
            )
            list_poly_traces_slice.append(poly_trace)
        if len(list_poly_traces_slice) != ntraces:
            raise ValueError(
                f"Slice {islice+1} (ID {sliceid}) has {len(list_poly_traces_slice)} traces, "
                f"but {ntraces} traces were expected."
            )
        # plot results for this slice
        if plots:
            vmin, vmax = ZScaleInterval().get_limits(image_data_filtered)
            logger.debug(f"ZScale limits: vmin={vmin}, vmax={vmax}")
            fig, ax = plt.subplots(figsize=(10, 6))
            tea.imshow(
                fig,
                ax,
                image_data_filtered,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                ds9mode=False,  # note that the polynomials are fitted using array indices (0-based)
                title=f"Slice {islice + 1} (ID {sliceid}) - Interpolating/Extrapolating Traces",
            )
            plot_fitted_boundary_polynomials(ax, list_poly_left, list_poly_right, voffset=0.0,sliceid=False, isliceplot=islice)
            plot_traces(ax, list_poly_traces_all_slices, voffset=0.0, traceid=False, isliceplot=islice)
            ntraces = len(list_poly_traces_all_slices[islice])
            xcenter1 = np.percentile(icolumns_to_analyze, 20)  # Center of the columns to analyze
            xcenter2 = np.percentile(icolumns_to_analyze, 80)  # Center of the columns to analyze
            for itrace in range(ntraces):
                label = None
                for k in range(2):
                    if k == 0:
                        poly_trace = list_poly_traces_all_slices[islice][itrace]
                        xcenter = xcenter1
                        color = "C1"
                        if itrace == 0:
                            label = f"Inicial traces"
                    else:
                        poly_trace = list_poly_traces_slice[itrace]
                        xcenter = xcenter2
                        color = "white"
                        if itrace == 0:
                            label = f"Interpolated traces"
                    ytrace = poly_trace(icolumns_to_analyze)
                    ax.plot(icolumns_to_analyze, ytrace, color=color, lw=1.5, label=label)
                    ycenter = poly_trace(xcenter)
                    ax.text(
                        xcenter,
                        ycenter,
                        f"Trace {itrace+1}",
                        color=color,
                        fontsize=8,
                        ha="center",
                        va="center",
                        fontweight="bold",
                        alpha=1.0,
                        bbox=dict(facecolor="black", alpha=0.3, edgecolor="black", boxstyle="round,pad=0.5"),
                    )
            ixdum = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
            ymin = np.min(list_poly_left[islice](ixdum)) - 10
            ymax = np.max(list_poly_right[islice](ixdum)) + 10
            ax.set_ylim(ymin, ymax)
            ax.legend()
            plt.tight_layout()
            plt.show()
            plt.close(fig)
        # Update the list of trace polynomials for this slice with the interpolated/extrapolated traces
        list_poly_traces_all_slices[islice] = list_poly_traces_slice
        # Update sliceids_remaining
        sliceids_remaining.remove(sliceid)

    return list_poly_traces_all_slices


def main(args=None):

    datetime_ini = datetime.now()

    parser = argparse.ArgumentParser(
        description="Interpolate/extrapolate traces within slices", formatter_class=RichHelpFormatter
    )
    parser.add_argument("--image", help="Path to the input image file (FITS format)", type=str, required=True)
    parser.add_argument("--traces", help="Path to the file with the slice trace polynomials", type=str, required=True)
    parser.add_argument(
        "--sliceid",
        help="Slice ID to process. This option can be specified multiple times",
        type=int,
        action="append",
        required=True
    )
    parser.add_argument(
        "--skip-sliceid",
        help="Slice ID to ignore in the interpolation/extrapolation. This option can be specified multiple times",
        type=int,
        action="append",
        default=[]
    )
    parser.add_argument(
        "--colrange",
        help="Column range to analyze (1-based index) along NAXIS1. This option can be specified multiple times",
        nargs=2,
        type=int,
        action="append",
        metavar=("MIN", "MAX"),
        default=None,
    )
    parser.add_argument(
        "--xmedian", help="Size of the median filter along NAXIS1 axis (odd; default: 21)", type=int, default=21
    )
    parser.add_argument(
        "--degslice", help="Degree of the polynomial to fit traces across slices (default: 2)", type=int, default=2
    )
    parser.add_argument("--plots", help="Display plots of the interpolated/extrapolated traces", action="store_true")
    parser.add_argument("--output", help="Output file name for the predicted polynomials", type=str, required=True)
    parser.add_argument("--overwrite", help="Overwrite existing output file", action="store_true")
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

    # Check input image file is defined
    if args.image is None:
        raise ValueError("Input image file is not defined. Use --image to specify the input image file.")
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Input image file {args.image} does not exist.")

    # Check median filter size
    if args.xmedian < 0:
        raise ValueError("Median filter size must be a non-negative integer.")
    
    # Define the columns to analyze based on the specified column ranges
    columns_to_analyze = columns_to_analyze_from_colranges(args.colrange)

    # Check output file
    output_fname = check_output_file_overwrite(args.output, args.output_dir, args.overwrite)

    # Interpolate/extrapolate traces within slices
    list_poly_traces_all_slices = interpolate_traces_within_slices(
        image_path=args.image,
        input_traces=args.traces,
        sliceids=args.sliceid,
        skip_sliceids=args.skip_sliceid,
        columns_to_analyze=columns_to_analyze,
        xmedian=args.xmedian,
        degslice=args.degslice,
        plots=args.plots,
    )

    # Save the predicted polynomials to the output file
    # Copy the input polynomials file to the output file
    shutil.copyfile(args.traces, output_fname)  # This always overwrites the output file if it exists, but we have already checked for overwrite permission
    logger.info(f"Copied input traces file [blue]{args.traces}[/blue] to output file [green]{output_fname}[/green].")
    logger.info("Updating the output file with the interpolated/extrapolated traces for the specified slices.")
    with fits.open(output_fname, mode="update") as hdul:
        primary_hdu = hdul[0]
        primary_hdu.header["UUID"] = str(uuid.uuid4())
        primary_hdu.header["OUTFILE"] = Path(output_fname).name
        add_script_info_to_fits_history(
            hdul[0].header,
            args,
            title="Interpolate/extrapolate traces within slices",
        )
        # Update the HDUs with the interpolated/extrapolated traces for the specified slices
        for sliceid in args.sliceid:
            islice = sliceindex_from_sliceid(sliceid)
            ntraces = len(list_poly_traces_all_slices[islice])
            deg = len(list_poly_traces_all_slices[islice][0].coef) - 1  # Degree of the trace polynomials
            logger.info(f"Slice {islice + 1} (ID {sliceid}) has {ntraces} traces after interpolation/extrapolation.")
            extname = f"SLCNUM{islice + 1:02d}"
            hdu = hdul[extname]
            array2d_coeffs = np.full((ntraces, deg + 1), fill_value=np.nan, dtype=float)
            for itrace in range(ntraces):
                poly_trace = list_poly_traces_all_slices[islice][itrace]
                array2d_coeffs[itrace, :] = poly_trace.convert().coef
            hdu.data = array2d_coeffs
            logger.info(f"Updated HDU {extname} with the interpolated/extrapolated traces for slice ID {sliceid}.")
        hdul.flush()

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
