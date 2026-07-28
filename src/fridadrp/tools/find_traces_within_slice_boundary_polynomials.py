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

from numina.array.display.polfit_residuals import polfit_residuals
from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history
from numina.tools.progressbarlines import ProgressBarLines

from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import DEF_SLICEID_FROM_SLICEINDEX
from fridadrp.core import sliceid_from_sliceindex
from fridadrp.core import sliceindex_from_sliceid
from fridadrp.tools.columns_to_analyze_from_colranges import columns_to_analyze_from_colranges
from fridadrp.tools.initialize_script_with_args import initialize_script_with_args
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_fitted_boundary_polynomials
from fridadrp.tools.read_slice_boundary_polynomials import read_slice_boundary_polynomials


def find_traces_within_slice_boundary_polynomials(image_path, poly_path, ntraces, deg, xmedian=21, columns_to_analyze=None, plots=False):
    """
    Find and fit traces within slice boundary polynomials.

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
    plots : bool, optional
        If True, display plots of the column analysis. Default is False.

    Returns
    -------
    None
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
    if plots:
        fig, ax = plt.subplots(figsize=(10, 6))
        vmin, vmax = ZScaleInterval().get_limits(image_data_filtered)
        tea.imshow(fig, ax, image_data_filtered, vmin=vmin, vmax=vmax, aspect="auto", ds9mode=True, title="Filtered Image with Slice Boundaries")
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        plot_fitted_boundary_polynomials(
            ax=ax,
            list_poly_left=list_poly_left,
            list_poly_right=list_poly_right,
            voffset=0.0,
            sliceid=True
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        plt.tight_layout()
        plt.show()

    # Main loop to find and fit traces within slice boundary polynomials
    xdum = np.array(columns_to_analyze) - 1  # array indices
    logger.info("Finding and fitting traces within slice boundary polynomials...")
    pbar = ProgressBarLines(total=FRIDA_NSLICES, logger=logger)
    islice_skipped = []
    for islice in range(FRIDA_NSLICES):
        sliceid = sliceid_from_sliceindex(islice)

        # Get the left and right polynomial coefficients for the current slice
        poly_left = list_poly_left[islice]
        poly_right = list_poly_right[islice]

        # Compute the minimum and maximum y-values of the left and right polynomials over the specified column range
        ymin_left = np.min(poly_left(xdum)) + 1  # (1-based index)
        ymax_right = np.max(poly_right(xdum)) +1  # (1-based index)
        # Check if the computed y-values are within the useful pixel range
        yborder = 1  # additional distance to the border to avoid edge effects
        if ymin_left < FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value + yborder:
            logger.warning(f"Slice {islice+1} (ID {sliceid}) left boundary is out of the useful pixel range. Skipping this slice.")
            islice_skipped.append(islice)
        if ymax_right > FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value - yborder:
            logger.warning(f"Slice {islice+1} (ID {sliceid}) right boundary is out of the useful pixel range. Skipping this slice.")
            if islice not in islice_skipped:
                islice_skipped.append(islice)
        pbar.update()

    logger.info(f"Skipped slices: {', '.join([f'#{i+1} (ID {sliceid_from_sliceindex(i)})' for i in islice_skipped])}")

def main(args=None):

    datetime_ini = datetime.now()

    parser = argparse.ArgumentParser(
        description="Find traces within slice boundary polynomials", formatter_class=RichHelpFormatter
    )
    parser.add_argument("--image", help="Path to the input image file (FITS format)", type=str, required=True)
    parser.add_argument("--poly", help="Path to the input file with the boundary polynomials", type=str, required=True)
    parser.add_argument("--ntraces", help="Number of traces per slice to find", type=int, required=True)
    parser.add_argument("--deg", help="Degree of the polynomial to fit", type=int, required=True)
    parser.add_argument(
        "--colrange",
        help="Column range to analyze (1-based index) along NAXIS1. This option can be specified multiple times",
        nargs=2,
        type=int,
        action="append",
        metavar=("MIN", "MAX"),
        default=None,
    )
    parser.add_argument("--xmedian", help="Size of the median filter along NAXIS1 axis (odd; default: 21)", type=int, default=21)
    parser.add_argument("--output", help="Output file name for the predicted polynomials", type=str, required=True)
    parser.add_argument("--overwrite", help="Overwrite existing output file", action="store_true")
    parser.add_argument("--plots", help="Display plots of the polynomial fitting", action="store_true")
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

    # Define the columns to analyze based on the specified column ranges
    columns_to_analyze = columns_to_analyze_from_colranges(args.colrange)

    # TODO: Implement the logic to find and fit traces within slice boundary polynomials
    find_traces_within_slice_boundary_polynomials(
        image_path=args.image,
        poly_path=args.poly,
        ntraces=args.ntraces,
        deg=args.deg,
        xmedian=args.xmedian,
        columns_to_analyze=columns_to_analyze,
        plots=args.plots,
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
