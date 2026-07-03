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

from numina.array.display.polfit_residuals import polfit_residuals_with_sigma_rejection
from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history
from numina.user.console import NuminaConsole

from fridadrp._version import version
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import slicenum_from_index


def fit_slice_boundary_borders_with_polynomials(input_file, deg=None, plots=False):
    """Fit the slice boundaries determined from the flats

    The polynomials are fitted using as independent variable
    the array index along the NAXIS1 axis, which ranges from 0 to FRIDA_NAXIS1_HAWAII-1, 
    and as dependent variable the array index along the NAXIS2 axis, 
    which ranges from 0 to FRIDA_NAXIS2_HAWAII-1.

        Parameters
    ----------
    input_file : str
        Path to the FITS file containing the slice boundary borders.
    deg : int
        Degree of the polynomial to fit. If None, an error will be raised.
    plots : bool, optional
        If True, display plots of the fitted polynomials. Default is False.

    Returns
    -------
    list_poly_left : list
        List of polynomials fitted to the left slice boundaries.
    list_poly_right : list
        List of polynomials fitted to the right slice boundaries.
    """
    # Check polynomial degree is defined
    if deg is None:
        raise ValueError("Polynomial degree is not defined.")
    # Check input file corresponds to the expected FITS file
    with fits.open(input_file) as hdul:
        if "KEYCODE" not in hdul[0].header:
            raise ValueError(f"Input file {input_file} does not contain a KEYCODE header keyword.")
        keycode = hdul[0].header["KEYCODE"]
        if keycode != "SLICE_BOUNDARY_BORDERS_FROM_FLAT":
            raise ValueError(f"Invalid KEYCODE={keycode}.\nExpected value is 'SLICE_BOUNDARY_BORDERS_FROM_FLAT'.")
        for extname in ["L-BORDER", "R-BORDER"]:
            if extname not in hdul:
                raise ValueError(f"Input file {input_file} does not contain the expected extension '{extname}'.")
        array_left_border = hdul["L-BORDER"].data
        if array_left_border.shape != (FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value):
            raise ValueError(
                f"Input file {input_file} has unexpected shape for L-BORDER extension: {array_left_border.shape}. "
                f"Expected shape is ({FRIDA_NSLICES}, {FRIDA_NAXIS1_HAWAII.value})."
            )
        array_right_border = hdul["R-BORDER"].data
        if array_right_border.shape != (FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value):
            raise ValueError(
                f"Input file {input_file} has unexpected shape for R-BORDER extension: {array_right_border.shape}. "
                f"Expected shape is ({FRIDA_NSLICES}, {FRIDA_NAXIS1_HAWAII.value})."
            )

    # Fit the slice boundaries with a polynomial
    list_poly_left = []
    list_poly_right = []
    for islice in tqdm(range(FRIDA_NSLICES), desc="Fitting slice boundaries"):
        x = np.arange(FRIDA_NAXIS1_HAWAII.value)
        y_left = array_left_border[islice, :]
        y_right = array_right_border[islice, :]
        # Fit a polynomial of degree 3 to the left and right boundaries
        ibad = np.isnan(y_left)
        xfit = x[~ibad]
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
        ibad = np.isnan(y_right)
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
    parser.add_argument("--input", help="Path to the flat file", type=str, required=True)
    parser.add_argument("--deg", help="Degree of the polynomial to fit", type=int, required=True)
    parser.add_argument("--output", help="Output FITS file name", type=str, default="slice_boundary_polynomials.fits")
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

    # Fit the slice boundaries from the flat file
    list_poly_left, list_poly_right = fit_slice_boundary_borders_with_polynomials(
        input_file=args.input,
        deg=args.deg,
        plots=args.plots,
    )

    # Compute slice widths as a function of the array index along the NAXIS1 axis
    array_widths = np.zeros((FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value))
    xdum = np.arange(FRIDA_NAXIS1_HAWAII.value)
    for islice in range(FRIDA_NSLICES):
        y_left = list_poly_left[islice](xdum)
        y_right = list_poly_right[islice](xdum)
        array_widths[islice, :] = y_right - y_left

    # Save the fitted polynomials to a FITS file
    array_coefs_left = np.array([p.convert().coef for p in list_poly_left])
    logger.info(f"coefs_left shape....: {array_coefs_left.shape}")
    array_coefs_right = np.array([p.convert().coef for p in list_poly_right])
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
    primary_hdu.header["POLDEG"] = args.deg
    primary_hdu.header["KEYCODE"] = "SLICE_BOUNDARY_POLYNOMIALS"
    add_script_info_to_fits_history(primary_hdu.header, args)
    hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
    hdul.writeto(args.output, overwrite=True)
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
