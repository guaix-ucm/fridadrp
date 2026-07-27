#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Predict the polynomial borders of a slice"""

import argparse
from astropy.io import fits
from datetime import datetime
import logging
import numpy as np
import shutil
import sys
from pathlib import Path
from rich.logging import RichHandler
from rich_argparse import RichHelpFormatter
from tqdm import tqdm
import uuid

from numina.array.display.polfit_residuals import polfit_residuals
from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history
from numina.user.console import NuminaConsole

from fridadrp._version import version
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import DEF_SLICEID_FROM_SLICEINDEX
from fridadrp.core import sliceid_from_sliceindex
from fridadrp.core import sliceindex_from_sliceid
from fridadrp.tools.read_slice_boundary_polynomials import read_slice_boundary_polynomials


def predict_polynomial_slice_borders(input_polynomial, slicenum, degslice=2, force=False, plots=False):
    """
    Predict the polynomial borders of a slice by
    interpolating/extrapolating the polynomials of slices belonging
    to the same group (i.e., slices id 1-15 and slices id 16-30).

    The predicted polynomials are fitted to the left and right borders
    of the slices in the group, and then used to predict the left and
    right borders of the specified slice. The predicted borders are then fitted
    to a new polynomial degree degslice. We have found that degslice=2 is the
    best compromise between accuracy and stability. A 3rd-degree polynomial
    tends to overfit the data and produce shifts in the predicted borders, while
    a 1st-degree polynomial does not capture the curvature of the borders well enough.

    Parameters
    ----------
    input_polynomial : str
        Path to the input FITS file containing the polynomial coefficients.
    slicenum : int
        Number of the slice to be predicted. This is a number between 1 and FRIDA_NSLICES,
        where slice 1 corresponds to the one appearing at the bottom of the H2RG
        image and slice FRIDA_NSLICES corresponds to the one appearing at the top.
    degslice : int, optional
        Degree of the polynomial to be fitted.
    force : bool, optional
        If True, overwrite existing polynomial borders for the specified slice.
        If False, raise an error if the polynomial borders for the specified
        slice are already defined.
    plots : bool, optional
        If True, display plots of the polynomial fitting.
    """

    logger = logging.getLogger(__name__)

    # Read the polynomial coefficients from the input FITS file
    list_poly_left, list_poly_right, poldeg = read_slice_boundary_polynomials(input_polynomial)

    # Slice ID corresponding to the slicenum (1-based index)
    sliceindex = slicenum - 1  # Convert slice number to zero-based index
    sliceid = sliceid_from_sliceindex(sliceindex)
    logger.info(f"Predicting polynomial borders for slice number {slicenum} (slice ID {sliceid})")

    # Check if the polynomial for the specified slice is already defined
    already_defined = False
    if list_poly_left[sliceindex] is not None:
        logger.warning(f"Left polynomial border for slice number {slicenum} (slice ID {sliceid}) is already defined.")
        already_defined = True
    if list_poly_right[sliceindex] is not None:
        logger.warning(f"Right polynomial border for slice number {slicenum} (slice ID {sliceid}) is already defined.")
        already_defined = True
    if already_defined:
        if not force:
            raise ValueError(
                f"Polynomial borders for slice number {slicenum} (slice ID {sliceid}) are already defined. Use --force to overwrite."
            )
        else:
            logger.warning(f"Overwriting polynomial borders for slice number {slicenum} (slice ID {sliceid}).")

    # Determine the group of slices to which the specified slice belongs
    slicesid_group1 = DEF_SLICEID_FROM_SLICEINDEX[0::2].tolist()
    slicesid_group2 = DEF_SLICEID_FROM_SLICEINDEX[1::2].tolist()
    if sliceid in slicesid_group1:
        slicesid_group = slicesid_group1
    else:
        slicesid_group = slicesid_group2
    logger.info(f"Slice ID {sliceid} belongs to the group of slices with ID:\n{slicesid_group}.")

    # List of slice IDs to be used for fitting: excluding the specified slice and
    # those for which the polynomial borders are undefined (i.e., None)
    list_slicesid_to_fit = []
    for sid in slicesid_group:
        if sid != sliceid:
            idx = sliceindex_from_sliceid(sid)
            if list_poly_left[idx] is not None and list_poly_right[idx] is not None:
                list_slicesid_to_fit.append(sid)
            else:
                logger.warning(f"Polynomial borders for slice ID {sid} are undefined.")
    logger.info(f"Using the polynomials of the following slice IDs to predict the new borders:\n{list_slicesid_to_fit}")

    # Fit a polynomial of degree degslice for each column along NAXIS1,
    # to the left and right borders of the slices in the group.
    xfit = np.array([sliceindex_from_sliceid(sid) for sid in list_slicesid_to_fit])
    ypredicted_left = np.zeros(FRIDA_NAXIS1_HAWAII.value)
    ypredicted_right = np.zeros(FRIDA_NAXIS1_HAWAII.value)
    for icol in tqdm(range(FRIDA_NAXIS1_HAWAII.value), desc=f"Fitting along NAXIS1"):
        debugplot = 0
        if plots:
            if icol in [0, FRIDA_NAXIS1_HAWAII.value // 2, FRIDA_NAXIS1_HAWAII.value - 1]:
                debugplot = 2
        yfit_left = np.array([list_poly_left[sliceindex_from_sliceid(sid)](icol) for sid in list_slicesid_to_fit])
        poly, _ = polfit_residuals(
            xfit,
            yfit_left,
            deg=degslice,
            debugplot=debugplot,
            xlabel="slice index",
            ylabel="array index along NAXIS2",
            title=f"Fitted right border polynomial for pixel {icol+1} along NAXIS1",
        )
        ypredicted_left[icol] = poly(
            sliceindex_from_sliceid(sliceid)
        )  # predict the left border for the specified slice
        yfit_right = np.array([list_poly_right[sliceindex_from_sliceid(sid)](icol) for sid in list_slicesid_to_fit])
        poly, _ = polfit_residuals(
            xfit,
            yfit_right,
            deg=degslice,
            debugplot=debugplot,
            xlabel="slice index",
            ylabel="array index along NAXIS2",
            title=f"Fitted right border polynomial for pixel {icol+1} along NAXIS1",
        )
        ypredicted_right[icol] = poly(
            sliceindex_from_sliceid(sliceid)
        )  # predict the right border for the specified slice

    # Fit a polynomial of the same degree as the one used for the slices in the group,
    # to the predicted left and right borders
    if plots:
        debugplot = 2
    else:
        debugplot = 0
    predicted_poly_left, _ = polfit_residuals(
        np.arange(FRIDA_NAXIS1_HAWAII.value),
        ypredicted_left,
        deg=poldeg,
        debugplot=debugplot,
        xlabel="array index along NAXIS1",
        ylabel="array index along NAXIS2",
        title=f"Predicted left border polynomial for slice number {slicenum} (slice ID {sliceid})",
    )
    predicted_poly_right, _ = polfit_residuals(
        np.arange(FRIDA_NAXIS1_HAWAII.value),
        ypredicted_right,
        deg=poldeg,
        debugplot=debugplot,
        xlabel="array index along NAXIS1",
        ylabel="array index along NAXIS2",
        title=f"Predicted right border polynomial for slice number {slicenum} (slice ID {sliceid})",
    )

    return predicted_poly_left, predicted_poly_right


def main(args=None):

    datetime_ini = datetime.now()

    parser = argparse.ArgumentParser(
        description="Predict slice boundary polynomials", formatter_class=RichHelpFormatter
    )
    parser.add_argument("--poly", help="Path to the input file with the boundary polynomials", type=str, required=True)
    parser.add_argument(
        "--slicenum",
        help="Slice number to be predicted (1 to 30, where 1 is the bottom slice and 30 is the top slice)",
        type=int,
        required=True,
        metavar="SLICE_INDEX",
        choices=range(1, FRIDA_NSLICES + 1),
    )
    parser.add_argument(
        "--degslice", help="Degree of the polynomial to be fitted (slice border vs. slice index)", type=int, default=2
    )
    parser.add_argument("--output", help="Output file name for the predicted polynomials", type=str, required=True)
    parser.add_argument("--overwrite", help="Overwrite existing output file", action="store_true")
    parser.add_argument(
        "--force", help="Force recomputation of existing polynomials in chosen slice", action="store_true"
    )
    parser.add_argument("--plots", help="Display plots of the polynomial fitting", action="store_true")
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
    console.rule(f"[bold magenta]Welcome to fridadrp-predict_polynomial_slice_borders[/bold magenta]")

    # Display version info
    logger = logging.getLogger(__name__)
    logger.info(f"Using {__name__} version {version}")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Command line arguments: {args}")

    # Check input polynomials file is defined
    if args.poly is None:
        raise ValueError("Input file is not defined. Use --poly to specify the input file with polynomials.")

    # Check output file
    if Path(args.output).exists():
        if Path(args.output).is_dir():
            raise IsADirectoryError(
                f"Output file {args.output} is a directory. Please specify a valid output file name."
            )
        if not args.overwrite:
            raise FileExistsError(f"Output file {args.output} already exists. Use --overwrite to overwrite it.")

    # Predict the polynomial slice borders for the specified slice
    predicted_poly_left, predicted_poly_right = predict_polynomial_slice_borders(
        input_polynomial=args.poly, slicenum=args.slicenum, degslice=args.degslice, force=args.force, plots=args.plots
    )

    # Save the predicted polynomials to the output FITS file
    # Copy the input polynomials file to the output file
    shutil.copyfile(args.poly, args.output)  # This always overwrites the output file if it exists
    logger.info(f"Copied input polynomials file {args.poly} to output file {args.output}.")
    logger.info(f"Updating the output FITS file with the predicted polynomials for slice number {args.slicenum} (slice ID {sliceid_from_sliceindex(args.slicenum - 1):02d}).")
    with fits.open(args.output, mode="update") as hdul:
        uuid_pol = hdul[0].header["UUID"]
        hdul[0].header["UUID"] = str(uuid.uuid4())  # Generate a new UUID for the output file
        idx = hdul[0].header.index("UUID-BOR")
        hdul[0].header.insert(idx + 1, ("UUID-POL", uuid_pol))  # Store the UUID of the input polynomials file
        hdul[0].header[f"SLCNUM{args.slicenum:02d}"] = (  # Set the SLCNUM keyword for the specified slice to True
            True,
            f"Slice number {args.slicenum:02d} (ID: {sliceid_from_sliceindex(args.slicenum - 1):02d}) is included",
        )
        add_script_info_to_fits_history(
            hdul[0].header,
            args,
            title=f"Predicted polynomial borders for slice number {args.slicenum} (slice ID {sliceid_from_sliceindex(args.slicenum - 1):02d})",
        )
        hdul["L-BORDER"].data[args.slicenum - 1, :] = predicted_poly_left.convert().coef
        hdul["R-BORDER"].data[args.slicenum - 1, :] = predicted_poly_right.convert().coef
        xdum = np.arange(FRIDA_NAXIS1_HAWAII.value)
        hdul["SLIWIDTH"].data[args.slicenum - 1, :] = predicted_poly_right(xdum) - predicted_poly_left(xdum)
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
