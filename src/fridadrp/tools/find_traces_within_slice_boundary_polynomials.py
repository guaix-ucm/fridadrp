#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Find and fit traces within slice boundary polynomials."""

import argparse
from astropy import logger
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from rich_argparse import RichHelpFormatter
from scipy.ndimage import median_filter, generic_filter
import teareduce as tea
import uuid

from numina.array.display.polfit_residuals import polfit_residuals, polfit_residuals_with_sigma_rejection
from numina.array.wavecalib.peaks_spectrum import find_highest_peaks_spectrum, refine_peaks_spectrum
from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history
from numina.tools.progressbarlines import ProgressBarLines

from fridadrp.core import DEF_SLICEID_FROM_SLICEINDEX
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import sliceid_from_sliceindex, sliceindex_from_sliceid
from fridadrp.tools.columns_to_analyze_from_colranges import columns_to_analyze_from_colranges
from fridadrp.tools.initialize_script_with_args import initialize_script_with_args
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_fitted_boundary_polynomials
from fridadrp.tools.read_slice_boundary_polynomials import read_slice_boundary_polynomials


def find_traces_within_slice_boundary_polynomials(
    image_path, poly_path, ntraces, deg, xmedian=21, columns_to_analyze=None, yborder=1, degslice=2, plotsliceid=None
):
    """
    Find and fit traces within slice boundary polynomials.

    Each trace if fitted with a polynomial of degree `deg`.
    The traces are found by searching for the highest peaks in
    each column along NAXIS1, restricted to the y-range defined
    by the slice boundary polynomials. Each polynomial is fitted
    using as independent variable the array index along NAXIS1 (0-based)
    and as dependent variable the array index along NAXIS2 (0-based).

    The traces are fitted using only the columns specified in `columns_to_analyze`.
    If this parameter is None, all columns are employed. The columns
    are specified as a list of tuples, where each tuple contains the minimum and
    maximum column (1-based index) to analyze. For example,
    `columns_to_analyze=[(1, 100), (200, 300)]` will analyze
    columns 1 to 100 and 200 to 300.

    When any of the two boundaries of a particular slice are out
    of the useful pixel range, the traces for that slice are
    extrapolated from the traces of the same group of slices.
    This is implemented only for the first and last slices,
    which are allowed to be skipped. If any other slice is skipped,
    an error is raised. The extrapolation is done by fitting a
    polynomial of degree `degslice`.

    Parameters
    ----------
    image_path : str
        Path to the input image file (FITS format).
    poly_path : str
        Path to the input file with the boundary polynomials.
    ntraces : int
        Number of traces per slice to find.
    deg : int
        Degree of the polynomial to fit.
    xmedian : int, optional
        Size of the median filter to apply to the flat data along NAXIS1
        to remove bad pixels.
    columns_to_analyze : list of tuple, optional
        List of column ranges to analyze. Each tuple contains (min_col, max_col).
    yborder : int, optional
        Additional distance to the border to avoid edge effects. Default is 1.
    degslice : int, optional
        Degree of the polynomial to fit traces across slices. Default is 2.
    plotsliceid : list of int, optional
        Slice ids for which to display plots. Default is None.

    Returns
    -------
    list_poly_traces_all_slices : list
        List of polynomial coefficients for traces in all slices.
        Each element in this list is another list containing ntraces polynomials
        for the corresponding slice. If a slice is skipped due to out-of-bounds
        boundaries, the corresponding element will be None.
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

    # Read the slice boundary polynomials
    list_poly_left, list_poly_right, poldeg = read_slice_boundary_polynomials(poly_path)

    # Plot the filtered image with the slice polynomial boundaries overplotted
    if plotsliceid is not None:
        fig, ax = plt.subplots(figsize=(10, 6))
        vmin, vmax = ZScaleInterval().get_limits(image_data_filtered)
        tea.imshow(
            fig,
            ax,
            image_data_filtered,
            vmin=vmin,
            vmax=vmax,
            aspect="auto",
            ds9mode=False,  # note that the polynomials are fitted using array indices (0-based)
            title="Filtered Image with Slice Boundaries",
        )
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        plot_fitted_boundary_polynomials(
            ax=ax, list_poly_left=list_poly_left, list_poly_right=list_poly_right, voffset=1.0, sliceid=True
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.tight_layout()
        plt.show()

    # Main loop to find and fit traces within slice boundary polynomials
    icolumns_to_analyze = np.array(columns_to_analyze) - 1  # array indices (0-based)
    islice_skipped = (
        []
    )  # (0-based index) list to store the indices of slices that are skipped due to out-of-bounds boundaries
    list_poly_traces_all_slices = []  # list to store the polynomial coefficients for traces in all slices
    logger.info("Finding and fitting traces within slice boundary polynomials...")
    pbar = ProgressBarLines(total=FRIDA_NSLICES, logger=logger)
    for islice in range(FRIDA_NSLICES):  # (0-based index)
        sliceid = sliceid_from_sliceindex(islice)

        # Get the left and right polynomial coefficients for the current slice
        poly_left = list_poly_left[islice]
        poly_right = list_poly_right[islice]

        # Compute the minimum and maximum y-values of the left and right polynomials over the specified column range
        ymin_left = int(np.min(poly_left(icolumns_to_analyze) + 0.5) + 1)  # (1-based index)
        ymax_right = int(np.max(poly_right(icolumns_to_analyze) + 0.5) + 1)  # (1-based index)
        # Check if the computed y-values are within the useful pixel range
        if ymin_left < FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value + yborder:
            logger.warning(f"Slice #{islice+1} (ID {sliceid}) left boundary is out of the useful pixel range.")
            islice_skipped.append(islice)
        if ymax_right > FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value - yborder:
            logger.warning(f"Slice #{islice+1} (ID {sliceid}) right boundary is out of the useful pixel range.")
            if islice not in islice_skipped:
                islice_skipped.append(islice)
        # Handle the case where the slice is skipped due to out-of-bounds boundaries
        if islice in islice_skipped:  # (0-based index)
            list_poly_traces_all_slices.append(None)
        else:
            # If the slice is not skipped, proceed to find and fit traces within the slice boundaries
            sliceid = sliceid_from_sliceindex(islice)
            # Initialize an array to store the peak positions for each trace in the current slice
            array2d_peaks_slice = np.full((ntraces, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)
            for icolumn in icolumns_to_analyze:  # (0-based index)
                if len(icolumns_to_analyze) == 1 and sliceid in plotsliceid:
                    debugplot = 2
                else:
                    debugplot = 0
                # Find the highest peaks in the spectrum (y-data)
                iypeaks = find_highest_peaks_spectrum(
                    sx=image_data_filtered[:, icolumn][ymin_left - 1 : ymax_right],
                    nmaxpeaks=ntraces,
                    nclean_around_peak=4,
                    nwinwidth=5,
                    title=f"Slice {islice+1} (ID {sliceid}), Column {icolumn+1}",
                    debugplot=debugplot,
                )
                # Refine the peak positions to sub-pixel accuracy
                fypeaks, _ = refine_peaks_spectrum(
                    sx=image_data_filtered[:, icolumn][ymin_left - 1 : ymax_right],
                    ixpeaks=iypeaks,
                    nwinwidth=3,
                    method="poly2",
                    title=f"Slice {islice+1} (ID {sliceid}), Column {icolumn+1}",
                    debugplot=debugplot,
                )
                fypeaks_slice = fypeaks + ymin_left - 1  # Adjust the peak positions to the original y-data range
                array2d_peaks_slice[:, icolumn] = (
                    fypeaks_slice  # Store the refined peak positions in the array for the current slice
                )
            # Fit polynomials to the traces found in the current slice
            collapsed_array2d_peaks_slice = np.sum(array2d_peaks_slice, axis=0)
            ibad_array2d_peaks_slice = np.isnan(collapsed_array2d_peaks_slice)
            list_poly_traces_slice = []
            ixdum = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
            if len(ixdum[~ibad_array2d_peaks_slice]) < deg + 1:
                raise ValueError(
                    f"Slice {islice+1} (ID {sliceid}) has too few valid data points ({len(ixdum[~ibad_array2d_peaks_slice])}) to fit a polynomial of degree {deg}. At least {deg + 1} valid data points are required."
                )
            if plotsliceid is not None and sliceid in plotsliceid:
                debugplot = 2
            else:
                debugplot = 0
            for itrace in range(ntraces):
                xfit = ixdum[~ibad_array2d_peaks_slice]
                yfit = array2d_peaks_slice[itrace][~ibad_array2d_peaks_slice]
                poly_trace, _, _ = polfit_residuals_with_sigma_rejection(
                    x=xfit,
                    y=yfit,
                    deg=deg,
                    times_sigma_reject=3.0,
                    xlabel="array index along NAXIS1",
                    ylabel="array index along NAXIS2",
                    title=f"Slice {islice+1} (ID {sliceid}), Trace {itrace+1} / {ntraces}",
                    debugplot=debugplot,
                )
                list_poly_traces_slice.append(poly_trace)
            if plotsliceid is not None and sliceid in plotsliceid:
                fig, ax = plt.subplots(figsize=(10, 6))
                vmin, vmax = ZScaleInterval().get_limits(image_data_filtered)
                tea.imshow(
                    fig,
                    ax,
                    image_data_filtered,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="auto",
                    ds9mode=False,  # note that the polynomials are fitted using array indices (0-based)
                    title=f"Slice {islice+1} (ID {sliceid}), Traces Found",
                )
                # set sliceid=False in plot_fitted_boundary_polynomials to avoid a problem
                # in plt.tight_layout() when sliceid=True (it fails to compute the layout properly)
                plot_fitted_boundary_polynomials(
                    ax=ax, list_poly_left=list_poly_left, list_poly_right=list_poly_right, voffset=0.0, sliceid=False
                )
                ymin = np.min(list_poly_left[islice](ixdum)) - 10
                ymax = np.max(list_poly_right[islice](ixdum)) + 10
                for itrace in range(ntraces):
                    poly_trace = list_poly_traces_slice[itrace]
                    ytrace = poly_trace(ixdum)
                    ax.plot(ixdum, ytrace, color="cyan", lw=1.5, label=f"Trace {itrace+1}")
                ax.set_ylim(ymin, ymax)
                plt.tight_layout()  # Fails if sliceid=True in plot_fitted_boundary_polynomials
                plt.show()
            list_poly_traces_all_slices.append(list_poly_traces_slice)
        pbar.update()

    # Check if any of the skipped slices is not in the allowed list of skipped slices
    allowed_islice_skipped = [
        0,
        FRIDA_NSLICES - 1,
    ]  # (0-based): only the first and last slices are allowed to be skipped
    if len(islice_skipped) > 0:
        for islice in islice_skipped:
            if islice not in allowed_islice_skipped:
                sliceid = sliceid_from_sliceindex(islice)
                logger.error(
                    f"Slice {islice+1} (ID {sliceid}) was skipped, but it is not in the allowed list of skipped slices."
                )
                raise ValueError(
                    f"Slice {islice+1} (ID {sliceid}) was skipped, but it is not in the allowed list of skipped slices."
                )

    # Extrapolate the traces for the skipped slices using the traces from the same group of slices
    if len(islice_skipped) > 0:
        slicesid_group1 = DEF_SLICEID_FROM_SLICEINDEX[0::2].tolist()
        slicesid_group2 = DEF_SLICEID_FROM_SLICEINDEX[1::2].tolist()
        for islice in islice_skipped:
            sliceid = sliceid_from_sliceindex(islice)
            plots = plotsliceid is not None and sliceid in plotsliceid
            logger.warning(f"Skipped slice: #{islice+1} (ID {sliceid})")
            if sliceid in slicesid_group1:
                slicesid_group = slicesid_group1
            else:
                slicesid_group = slicesid_group2
            if islice == 0:  # first slice
                logger.info(f"Extrapolating traces for the first slice (ID {sliceid})")
                logger.info(f"Slice ID {sliceid} belongs to the group of slices with ID:\n{slicesid_group}.")
            elif islice == FRIDA_NSLICES - 1:  # last slice
                logger.info(f"Extrapolating traces for the last slice (ID {sliceid})")
                logger.info(f"Slice ID {sliceid} belongs to the group of slices with ID:\n{slicesid_group}.")
            else:
                raise ValueError(
                    f"Slice {islice+1} (ID {sliceid}) was skipped, "
                    f"but it is not the first or last slice. This should not happen."
                )
            list_slicesid_to_fit = []
            for sid in slicesid_group:
                if sid != sliceid:
                    list_slicesid_to_fit.append(sid)
            logger.info(f"Fitting traces for the following slices (ID):\n{list_slicesid_to_fit}")
            # Fit polynomials to each trace across the slices in the same group
            xfit = np.array([sliceindex_from_sliceid(sid) for sid in list_slicesid_to_fit])  # (0-based)
            xpredicted = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
            list_poly_traces_slice = []
            for itrace in range(ntraces):
                ypredicted = np.zeros(FRIDA_NAXIS1_HAWAII.value, dtype=float)
                logger.info(f"Fitting trace {itrace+1}/{ntraces} of slice ID {sliceid} along NAXIS1...")
                for icolumn in range(FRIDA_NAXIS1_HAWAII.value):
                    debugplot = 0
                    if plots:
                        if icolumn in [0, FRIDA_NAXIS1_HAWAII.value // 2, FRIDA_NAXIS1_HAWAII.value - 1]:
                            debugplot = 2
                    yfit = np.array(
                        [
                            list_poly_traces_all_slices[sliceindex_from_sliceid(sid)][itrace](icolumn)
                            for sid in list_slicesid_to_fit
                        ]
                    )
                    poly_trace_across_slices, _ = polfit_residuals(
                        x=xfit,
                        y=yfit,
                        deg=degslice,
                        xlabel="slice index",
                        ylabel="array index along NAXIS2",
                        title=f"Slice ID {sliceid}, Trace {itrace+1} / {ntraces}, Column {icolumn+1}",
                        debugplot=debugplot,
                    )
                    ypredicted[icolumn] = poly_trace_across_slices(sliceindex_from_sliceid(sliceid))
                # fit a polynomial of degree `deg` to determine the trace
                poly_trace, _ = polfit_residuals(
                    x=xpredicted,
                    y=ypredicted,
                    deg=deg,
                    xlabel="array index along NAXIS1",
                    ylabel="array index along NAXIS2",
                    title=f"Slice ID {sliceid}, Trace {itrace+1} / {ntraces}",
                    debugplot=debugplot,
                )

            # TODO: plot the extrapolated traces
            if plotsliceid is not None and sliceid in plotsliceid:
                pass

    return list_poly_traces_all_slices


def main(args=None):

    datetime_ini = datetime.now()

    parser = argparse.ArgumentParser(
        description="Find traces within slice boundary polynomials", formatter_class=RichHelpFormatter
    )
    parser.add_argument("--image", help="Path to the input image file (FITS format)", type=str, required=True)
    parser.add_argument("--poly", help="Path to the input file with the boundary polynomials", type=str, required=True)
    parser.add_argument("--ntraces", help="Number of traces per slice to find", type=int, required=True)
    parser.add_argument("--deg", help="Degree of the polynomial to fit each trace", type=int, required=True)
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
    parser.add_argument("--degslice", help="Degree of the polynomial to fit traces across slices", type=int, default=2)
    parser.add_argument("--output", help="Output file name for the predicted polynomials", type=str, required=True)
    parser.add_argument("--overwrite", help="Overwrite existing output file", action="store_true")
    parser.add_argument(
        "--plotsliceid",
        help="Display plots for slice id (this option can be specified multiple times)",
        type=int,
        action="append",
        default=None,
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

    # Check input polynomials file is defined
    if args.poly is None:
        raise ValueError("Input file is not defined. Use --poly to specify the input file with polynomials.")

    # Check input image file is defined
    if args.image is None:
        raise ValueError("Input image file is not defined. Use --image to specify the input image file.")
    if not Path(args.image).exists():
        raise FileNotFoundError(f"Input image file {args.image} does not exist.")

    # Check number of traces per slice
    if args.ntraces is None:
        raise ValueError("Number of traces per slice is not defined. Use --ntraces to specify it.")
    if args.ntraces <= 0:
        raise ValueError("Number of traces per slice must be a positive integer.")

    # Check polynomial degree
    if args.deg is None:
        raise ValueError("Polynomial degree is not defined. Use --deg to specify it.")

    # Check median filter size
    if args.xmedian < 0:
        raise ValueError("Median filter size must be a non-negative integer.")

    # If output directory does not exist, create it
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory {args.output_dir} created.")
    # If output file is not an absolute path, prepend the output directory
    if not Path(args.output).is_absolute():
        output_fname = str(Path(args.output_dir) / args.output)
    else:
        output_fname = args.output
    # Check output file
    if Path(output_fname).exists():
        if Path(output_fname).is_dir():
            raise IsADirectoryError(
                f"Output file {output_fname} is a directory. Please specify a valid output file name."
            )
        if not args.overwrite:
            raise FileExistsError(f"Output file {output_fname} already exists. Use --overwrite to overwrite it.")

    # Check plotsliceid
    if args.plotsliceid is not None:
        for sliceid in args.plotsliceid:
            if sliceid < 1 or sliceid > FRIDA_NSLICES:
                raise ValueError(f"Invalid slice id {sliceid}. Must be between 1 and {FRIDA_NSLICES}.")

    # Define the columns to analyze based on the specified column ranges
    columns_to_analyze = columns_to_analyze_from_colranges(args.colrange)

    # Call the function to find traces within slice boundary polynomials
    list_poly_traces_all_slices = find_traces_within_slice_boundary_polynomials(
        image_path=args.image,
        poly_path=args.poly,
        ntraces=args.ntraces,
        deg=args.deg,
        xmedian=args.xmedian,
        columns_to_analyze=columns_to_analyze,
        degslice=args.degslice,
        plotsliceid=args.plotsliceid,
    )

    # TODO: save output_fname with the results

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
