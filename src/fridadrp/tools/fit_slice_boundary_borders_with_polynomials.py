#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Fit the slice boundaries determined from the flats"""

import argparse
from astropy.io import fits
from datetime import datetime
import logging
import numpy as np
from pathlib import Path
from rich.logging import RichHandler
from rich_argparse import RichHelpFormatter
import sys
from tqdm import tqdm
import uuid

from numina.array.display.polfit_residuals import polfit_residuals_with_sigma_rejection
from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history
from numina.user.console import NuminaConsole

from fridadrp._version import version
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import slicenum_from_index
from fridadrp.tools.columns_to_analyze_from_colranges import columns_to_analyze_from_colranges
from fridadrp.tools.read_slice_boundary_borders import read_slice_boundary_borders


def fit_slice_boundary_borders_with_polynomials(
    array_left_border, array_right_border, ibad, islice_ok, deg=None, columns_to_analyze=None, plots=False
):
    """Fit the slice boundaries determined from the flats

    The polynomials are fitted using as independent variable
    the array index along the NAXIS1 axis, which ranges from 0 to FRIDA_NAXIS1_HAWAII-1,
    and as dependent variable the array index along the NAXIS2 axis,
    which ranges from 0 to FRIDA_NAXIS2_HAWAII-1.

        Parameters
    ----------
    array_left_border : numpy.ndarray
        2D array containing the left slice boundary borders.
    array_right_border : numpy.ndarray
        2D array containing the right slice boundary borders.
    ibad : numpy.ndarray
        2D boolean array indicating the bad columns with NaN values (0-based).
    islice_ok : numpy.ndarray
        1D array containing the indices of slices to be analyzed (0-based).
    deg : int
        Degree of the polynomial to fit. If None, an error will be raised.
    columns_to_analyze : list of int
        List of columns to analyze (1-based index along NAXIS1).
    plots : bool, optional
        If True, display plots of the fitted polynomials. Default is False.

    Returns
    -------
    list_poly_left : list
        List of polynomials fitted to the left slice boundaries.
    list_poly_right : list
        List of polynomials fitted to the right slice boundaries.
    """
    logger = logging.getLogger(__name__)

    # Check polynomial degree is defined
    if deg is None:
        raise ValueError("Polynomial degree is not defined.")

    # Check enough valid columns are available for fitting
    logger.info(f"Number of initial valid columns to fit each boundary: {np.sum(~ibad)}")
    # Take into account columns_to_analyze
    iskip = np.ones(FRIDA_NAXIS1_HAWAII.value, dtype=bool)
    for col in columns_to_analyze:
        iskip[col - 1] = False
    ibad = ibad | iskip
    logger.info(f"Number of valid columns to fit each boundary after applying columns_to_analyze: {np.sum(~ibad)}")
    logger.info(f"Polynomial degree: {deg}")
    if np.sum(~ibad) < deg + 1:
        raise ValueError(
            f"Not enough valid columns for fitting. Number of valid columns: {np.sum(~ibad)}, required: {deg + 1}."
        )

    # Fit the slice boundaries with a polynomial
    list_poly_left = []
    list_poly_right = []
    x = np.arange(FRIDA_NAXIS1_HAWAII.value)
    xfit = x[~ibad]
    for islice in tqdm(range(FRIDA_NSLICES), desc="Fitting slice boundaries"):
        if islice not in islice_ok:
            list_poly_left.append(None)
            list_poly_right.append(None)
            continue
        y_left = array_left_border[islice, :]
        y_right = array_right_border[islice, :]
        # Fit a polynomial of degree 3 to the left and right boundaries
        yfit = y_left[~ibad]
        poly_left, _, _ = polfit_residuals_with_sigma_rejection(
            x=xfit,
            y=yfit,
            deg=deg,
            times_sigma_reject=3.0,
            xlabel="array index along NAXIS1 axis",
            ylabel="array index along NAXIS2 axis",
            title=f"Slice {slicenum_from_index(islice)} - Left boundary fit",
            debugplot=0 if not plots else 2,
        )
        list_poly_left.append(poly_left)
        # Fit a polynomial of degree 3 to the right boundary
        xfit = x[~ibad]
        yfit = y_right[~ibad]
        poly_right, _, _ = polfit_residuals_with_sigma_rejection(
            x=xfit,
            y=yfit,
            deg=deg,
            times_sigma_reject=3.0,
            xlabel="array index along NAXIS1 axis",
            ylabel="array index along NAXIS2 axis",
            title=f"Slice {slicenum_from_index(islice)} - Right boundary fit",
            debugplot=0 if not plots else 2,
        )
        list_poly_right.append(poly_right)

    return list_poly_left, list_poly_right


def main(args=None):

    datetime_ini = datetime.now()

    parser = argparse.ArgumentParser(
        description="Fit the slice boundaries determined from flat image", formatter_class=RichHelpFormatter
    )
    parser.add_argument("--input", help="Path to the FITS file with border data", type=str, required=True)
    parser.add_argument(
        "--colrange",
        help="Column range to fit (1-based index). This option can be specified multiple times",
        nargs=2,
        type=int,
        action="append",
        metavar=("MIN", "MAX"),
        default=None,
    )
    parser.add_argument("--deg", help="Degree of the polynomial to fit", type=int, required=True)
    parser.add_argument("--output", help="Output FITS file name", type=str, default=None)
    parser.add_argument("--overwrite", help="Overwrite output file if it exists", action="store_true")
    parser.add_argument("--plots", help="Display plots", action="store_true")
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
    console.rule(f"[bold magenta]Welcome to fridadrp-fit_slice_boundaries_from_flat[/bold magenta]")

    # Display version info
    logger = logging.getLogger(__name__)
    logger.info(f"Using {__name__} version {version}")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Command line arguments: {args}")

    # Check input file is defined
    if args.input is None:
        raise ValueError("Input file is not defined. Use --input to specify the input file.")

    # Check polynomial degree is defined
    if args.deg is None:
        raise ValueError("Polynomial degree is not defined. Use --deg to specify the polynomial degree.")

    # Define columns to be employed for fitting
    columns_to_analyze = columns_to_analyze_from_colranges(args.colrange)

    # Read the slice boundary borders from the input FITS file
    array_left_border, array_right_border, ibad, keywords_dict = read_slice_boundary_borders(args.input)
    slice_ini = keywords_dict["SLICEINI"]
    slice_end = keywords_dict["SLICEEND"]
    islice_ok = np.arange(slice_ini - 1, slice_end)  # indices of slices to be analyzed (0-based index)

    # Set output file name if not defined
    if args.output is None:
        args.output = f"slice_boundary_polynomials_{slice_ini}-{slice_end}.fits"
    # check if the output file already exists and handle overwrite option
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file {args.output} already exists. Use --overwrite to overwrite it.")

    # Fit the slice boundaries from the flat file
    list_poly_left, list_poly_right = fit_slice_boundary_borders_with_polynomials(
        array_left_border=array_left_border,
        array_right_border=array_right_border,
        ibad=ibad,
        islice_ok=islice_ok,
        deg=args.deg,
        columns_to_analyze=columns_to_analyze,
        plots=args.plots,
    )

    # Compute slice widths as a function of the array index along the NAXIS1 axis
    array_widths = np.full((FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)
    xdum = np.arange(FRIDA_NAXIS1_HAWAII.value)
    for islice in range(FRIDA_NSLICES):
        if islice not in islice_ok:
            continue
        y_left = list_poly_left[islice](xdum)
        y_right = list_poly_right[islice](xdum)
        array_widths[islice, :] = y_right - y_left

    # Save the fitted polynomials to a FITS file
    array_coefs_left = np.full((FRIDA_NSLICES, args.deg + 1), np.nan, dtype=float)
    for islice in range(FRIDA_NSLICES):
        if islice not in islice_ok:
            continue
        array_coefs_left[islice, :] = list_poly_left[islice].convert().coef
    logger.info(f"coefs_left shape....: {array_coefs_left.shape}")
    array_coefs_right = np.full((FRIDA_NSLICES, args.deg + 1), np.nan, dtype=float)
    for islice in range(FRIDA_NSLICES):
        if islice not in islice_ok:
            continue
        array_coefs_right[islice, :] = list_poly_right[islice].convert().coef
    logger.info(f"coefs_right shape...: {array_coefs_right.shape}")
    header1 = fits.Header()
    header1["EXTNAME"] = "L-BORDER"
    hdu1 = fits.ImageHDU(data=array_coefs_left, header=header1)
    header2 = fits.Header()
    header2["EXTNAME"] = "R-BORDER"
    hdu2 = fits.ImageHDU(data=array_coefs_right, header=header2)
    header3 = fits.Header()
    header3["EXTNAME"] = "SLIWIDTH"
    hdu3 = fits.ImageHDU(data=array_widths, header=header3)
    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["INPFILE"] = Path(args.input).name
    for key, value in keywords_dict.items():
        primary_hdu.header[key] = value
    primary_hdu.header["POLDEG"] = args.deg
    primary_hdu.header["KEYCODE"] = "SLICE_BOUNDARY_POLYNOMIALS"
    primary_hdu.header["UUID"] = str(uuid.uuid4())
    add_script_info_to_fits_history(primary_hdu.header, args)
    hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
    hdul.writeto(args.output, overwrite=args.overwrite)
    logger.info(f"Slice boundary polynomials saved to: [green]{args.output}[/green]")

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
