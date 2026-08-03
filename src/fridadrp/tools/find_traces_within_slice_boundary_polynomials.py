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
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from rich_argparse import RichHelpFormatter
from scipy.ndimage import median_filter, generic_filter
import teareduce as tea
import uuid

from numina.array.display.polfit_residuals import polfit_residuals, polfit_residuals_with_sigma_rejection
from numina.array.robustfit import fit_theil_sen
from numina.array.stats import robust_std
from numina.array.wavecalib.peaks_spectrum import find_highest_peaks_spectrum, refine_peaks_spectrum
from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history
from numina.tools.progressbarlines import ProgressBarLines

from fridadrp.core import DEF_SLICEID_FROM_SLICEINDEX
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_NAXIS1_HAWAII
from fridadrp.core import FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import sliceid_from_sliceindex, sliceindex_from_sliceid
from fridadrp.tools.columns_to_analyze_from_colranges import columns_to_analyze_from_colranges
from fridadrp.tools.initialize_script_with_args import initialize_script_with_args
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_fitted_boundary_polynomials
from fridadrp.tools.read_slice_boundary_polynomials import read_slice_boundary_polynomials


def find_traces_within_slice_boundary_polynomials(
    image_path,
    poly_path,
    voffset=[0.0],
    ntraces=None,
    nextend=2,
    nclean_around_peaks=-1,
    deg=None,
    xmedian=21,
    theilsen=False,
    columns_to_analyze=None,
    yborder=1,
    degslice=2,
    degrefine=2,
    refine=True,
    plotsliceid=None,
    pdf_out=None,
    output_dir=None,
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
    If this parameter is None, all columns within the useful image region are employed.
    The columns are specified as a list of tuples (1-based indices along NAXIS1),
    where each tuple contains the minimum and maximum column (1-based index)
    to analyze. For example, `columns_to_analyze=[(1, 100), (200, 300)]`
    will analyze columns 1 to 100 and 200 to 300.

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
    voffset : list of float, optional
        Vertical offsets to apply to the slice boundaries. Default is [0.0].
        If more than one value is provided, a polynomial of degree len(voffset)-1
        will be fitted to the offsets and applied to the slice boundaries.
    ntraces : int or None
        Number of traces per slice to find.
    nextend : int, optional
        Number of pixels to extend the search for traces beyond the slice boundaries.
    nclean_around_peaks : int, optional
        Number of pixels to clean around each peak when searching for
        the highest peaks. If -1, set to zero pixels moving outward from the
        peak while the signal keeps decreasing.
    deg : int or None
        Degree of the polynomial to fit.
    xmedian : int, optional
        Size of the median filter to apply to the flat data along NAXIS1
        to remove bad pixels.
    theilsen : bool, optional
        If True, use initial Theil-Sen regression to reject outliers in the
        polynomial fitting.
    columns_to_analyze : list of tuple, optional
        List of column ranges to analyze. Each tuple contains (min_col, max_col).
        This numbers are 1-based indices along NAXIS1.
    yborder : int, optional
        Additional distance to the border to avoid edge effects. Default is 1.
    degslice : int, optional
        Degree of the polynomial to fit traces across slices. Default is 2.
    degrefine : int, optional
        Degree of the polynomial to refine the trace positions. Default is 2.
    refine : bool, optional
        If True, refine extrapolated traces using the smoothed image data.
        Default is True.
    plotsliceid : list of int, optional
        Slice ids for which to display plots. Default is None.
    pdf_out : str, optional
        Output PDF file for final plots with traces for every slice.
        Default is None.
    output_dir : str, optional
        Directory where output files will be saved. Default is None.

    Returns
    -------
    list_poly_left : list of Polynomial
        List of Polynomial objects for the left slice boundaries.
        Important: this polynomials are updated after applying voffset.
    list_poly_right : list of Polynomial
        List of Polynomial objects for the right slice boundaries.
        Important: this polynomials are updated after applying voffset.
    list_poly_traces_all_slices : list
        List of polynomial coefficients for traces in all slices.
        Each element in this list is another list containing ntraces polynomials
        for the corresponding slice. If a slice is skipped due to out-of-bounds
        boundaries, the corresponding element will be None.
    """
    logger = logging.getLogger(__name__)

    if ntraces is None:
        raise ValueError("The number of traces per slice (ntraces) must be specified.")
    if deg is None:
        raise ValueError("The degree of the polynomial to fit each trace (deg) must be specified.")

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

    # Apply offset to the slice boundary polynomials
    if len(voffset) < 1:
        raise ValueError("The list of vertical offsets (voffset) must contain at least one value.")
    elif len(voffset) == 1:
        logger.info(f"Applying constant vertical offset of {voffset[0]} pixels to slice boundary polynomials.")
        for islice in range(FRIDA_NSLICES):
            list_poly_left[islice] += voffset[0]
            list_poly_right[islice] += voffset[0]
    else:
        logger.info(
            f"Fitting a polynomial of degree {len(voffset)-1} to the vertical offsets and applying it to slice boundary polynomials."
        )
        nvoffets = len(voffset)
        x_offsets = np.linspace(
            FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL.value, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL.value, nvoffets
        )
        if plotsliceid is not None:
            debugplot = 2
        else:
            debugplot = 0
        poly_voffset, _ = polfit_residuals(
            x=x_offsets,
            y=np.array(voffset),
            deg=len(voffset) - 1,
            xlabel="array index along NAXIS1",
            ylabel="array index along NAXIS2",
            title=f"Vertical offset to be applied to all the slice boundary polynomials",
            debugplot=debugplot,
        )
        for islice in range(FRIDA_NSLICES):
            list_poly_left[islice] += poly_voffset
            list_poly_right[islice] += poly_voffset

    # Plot the filtered image with the slice polynomial boundaries overplotted
    if plotsliceid is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
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
            ax=ax, list_poly_left=list_poly_left, list_poly_right=list_poly_right, sliceid=True
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        logger.info("Press 'q' to close the plot or 'x' to close the plot and exit the program.")

        def on_key(event):
            if event.key == "q":
                plt.close(fig)
            elif event.key == "x":
                logger.info("Exiting the program as requested by the user.")
                plt.close(fig)
                sys.exit(0)

        fig.canvas.mpl_connect("key_press_event", on_key)
        plt.tight_layout()
        plt.show()

    # Main loop to find and fit traces within slice boundary polynomials
    icolumns_to_analyze = np.array(columns_to_analyze) - 1  # array indices (0-based)
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
        ymin_left = int(np.min(poly_left(icolumns_to_analyze) + 0.5))  # (0-based index)
        ymin_left -= nextend  # extend the search for traces beyond the slice boundaries
        ymin_left += 1  # (1-based index)
        ymax_right = int(np.max(poly_right(icolumns_to_analyze) + 0.5))  # (0-based index)
        ymax_right += nextend  # extend the search for traces beyond the slice boundaries
        ymax_right += 1  # (1-based index)
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
                # Recompute ymin_left and ymax_right for the current column to avoid making use
                # of data from neighboring slices when the slice boundaries are not parallel to the NAXIS1 axis
                ymin_left = int(poly_left(icolumn) + 0.5) - nextend + 1  # (1-based index)
                ymax_right = int(poly_right(icolumn) + 0.5) + nextend + 1  # (1-based index)
                # Find the highest peaks in the spectrum (y-data)
                iypeaks = find_highest_peaks_spectrum(
                    sx=image_data_filtered[:, icolumn][ymin_left - 1 : ymax_right],
                    nmaxpeaks=ntraces,
                    nclean_around_peak=nclean_around_peaks,
                    nwinwidth=5,
                    title=f"Slice {islice+1} (ID {sliceid}), Column {icolumn+1}",
                    debugplot=debugplot,
                )
                # Ignore all the peaks when the first peak is at the first pixel of the y-data range, as it may be an artifact
                if iypeaks[0] != 0:
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
                    if debugplot == 2:
                        logger.info(
                            f"Slice {islice+1} (ID {sliceid}), Column {icolumn+1}: Found peaks at\n{fypeaks_slice}"
                        )
            # Fit polynomials to the traces found in the current slice
            collapsed_array2d_peaks_slice = np.sum(array2d_peaks_slice, axis=0)
            ibad_array2d_peaks_slice = np.isnan(collapsed_array2d_peaks_slice)
            list_poly_traces_slice = []
            ixdum = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
            if len(ixdum[~ibad_array2d_peaks_slice]) >= deg + 1:
                if plotsliceid is not None and sliceid in plotsliceid:
                    debugplot = 2
                else:
                    debugplot = 0
                for itrace in range(ntraces):
                    xfit = ixdum[~ibad_array2d_peaks_slice]
                    yfit = array2d_peaks_slice[itrace][~ibad_array2d_peaks_slice]
                    if theilsen:
                        coef_theilsen = fit_theil_sen(xfit, yfit)
                        residuals_theilsen = yfit - (coef_theilsen[0] + coef_theilsen[1] * xfit)
                        rms_residuals_theilsen = robust_std(residuals_theilsen)
                        outliers_theilsen = np.abs(residuals_theilsen) > 3 * rms_residuals_theilsen
                        xfit_filtered = xfit[~outliers_theilsen]
                        yfit_filtered = yfit[~outliers_theilsen]
                        xplot_theilsen = np.array([np.min(xfit_filtered), np.max(xfit_filtered)])
                        if debugplot == 2:
                            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
                            ax1.scatter(
                                xfit_filtered,
                                yfit_filtered,
                                marker="o",
                                color="b",
                                edgecolor="k",
                                s=75,
                                label=f"fitted data ({len(xfit_filtered)})",
                            )
                            ax1.plot(
                                xplot_theilsen,
                                coef_theilsen[0] + coef_theilsen[1] * xplot_theilsen,
                                "c-",
                                label="Theil-Sen fit",
                            )
                            ax1.scatter(
                                xfit[outliers_theilsen],
                                yfit[outliers_theilsen],
                                marker="x",
                                color="r",
                                s=75,
                                label=f"rejected ({len(xfit[outliers_theilsen])})",
                            )
                            ax1.set_ylabel("array index along NAXIS2")
                            ax1.legend()
                            ax1.set_title(f"Slice {islice+1} (ID {sliceid}), Trace {itrace+1} / {ntraces}")
                            ax2.scatter(xfit, residuals_theilsen, marker="o", color="b", edgecolor="k", s=75)
                            ax2.axhline(0, color="k", linestyle="--")
                            ax2.set_xlabel("array index along NAXIS1")
                            ax2.set_ylabel("residuals")
                            plt.tight_layout()
                            plt.show()
                    else:
                        xfit_filtered = xfit
                        yfit_filtered = yfit
                    poly_trace, _, _ = polfit_residuals_with_sigma_rejection(
                        x=xfit_filtered,
                        y=yfit_filtered,
                        deg=deg,
                        times_sigma_reject=3.0,
                        ylimres_with_rejected=True,
                        xlabel="array index along NAXIS1",
                        ylabel="array index along NAXIS2",
                        title=f"Slice {islice+1} (ID {sliceid}), Trace {itrace+1} / {ntraces}",
                        debugplot=debugplot,
                    )
                    list_poly_traces_slice.append(poly_trace)
                if plotsliceid is not None and sliceid in plotsliceid:
                    fig, ax = plt.subplots(figsize=(10, 8))
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
                        ax=ax,
                        list_poly_left=list_poly_left,
                        list_poly_right=list_poly_right,
                        voffset=0.0,
                        sliceid=False,
                    )
                    ymin = np.min(list_poly_left[islice](ixdum)) - 10
                    ymax = np.max(list_poly_right[islice](ixdum)) + 10
                    for itrace in range(ntraces):
                        poly_trace = list_poly_traces_slice[itrace]
                        ytrace = poly_trace(ixdum)
                        ax.plot(ixdum, ytrace, color="cyan", lw=1.5, label=f"Trace {itrace+1}")
                        xfit = ixdum[~ibad_array2d_peaks_slice]
                        yfit = array2d_peaks_slice[itrace][~ibad_array2d_peaks_slice]
                        ax.plot(xfit, yfit, f"C{itrace}.", markersize=0.5)
                    ax.set_ylim(ymin, ymax)
                    logger.info("Press 'q' to close the plot or 'x' to close the plot and exit the program.")

                    def on_key(event):
                        if event.key == "q":
                            plt.close(fig)
                        elif event.key == "x":
                            logger.info("Exiting the program as requested by the user.")
                            plt.close(fig)
                            sys.exit(0)

                    fig.canvas.mpl_connect("key_press_event", on_key)
                    plt.tight_layout()
                    plt.show()

                list_poly_traces_all_slices.append(list_poly_traces_slice)
            else:
                logger.warning(
                    f"Slice {islice+1} (ID {sliceid}) has too few valid data points ({len(ixdum[~ibad_array2d_peaks_slice])}) to fit a polynomial of degree {deg}. At least {deg + 1} valid data points are required."
                )
        pbar.update()
    if len(icolumns_to_analyze) == 1:
        logger.info("Only one column was analyzed. No extrapolation is needed. Exiting the program.")
        sys.exit(0)  # Exit the program if only one column was analyzed, as no extrapolation is needed
    if len(list_poly_traces_all_slices) != FRIDA_NSLICES:
        logger.warning(
            f"Expected {FRIDA_NSLICES} slices, but found {len(list_poly_traces_all_slices)} slices in the list of polynomial traces."
        )
        logger.info("Exiting the program due to the mismatch in the number of slices.")
        sys.exit(0)

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
                ypredicted = np.full(FRIDA_NAXIS1_HAWAII.value, fill_value=np.nan, dtype=float)
                logger.info(f"Fitting trace {itrace+1}/{ntraces} of slice ID {sliceid} along NAXIS1...")
                for icolumn in icolumns_to_analyze:
                    debugplot = 0
                    if plots:
                        if icolumn in icolumns_to_plot:
                            debugplot = 2
                    yfit = np.array(
                        [
                            list_poly_traces_all_slices[sliceindex_from_sliceid(sid)][itrace](icolumn)
                            for sid in list_slicesid_to_fit
                        ]
                    )
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
            # refine the polynomial extrapolated for each trace using the smoothed image data
            list_poly_traces_slice_refined = list_poly_traces_slice.copy()
            if refine:
                logger.info(f"Refining the fitted traces for slice ID {sliceid} using the smoothed image data...")
                deltay_refined = np.full((ntraces, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)
                for itrace in range(ntraces):
                    poly_trace = list_poly_traces_slice[itrace]
                    # check if the predicted trace is within the useful image region
                    ypredicted = (poly_trace(icolumns_to_analyze) + 0.5).astype(int)  # (0-based) rounded integer
                    trace_is_within_bounds = True
                    if np.min(ypredicted) + 1 < FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value + 5 * yborder:
                        trace_is_within_bounds = False
                    if np.max(ypredicted) + 1 > FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value - 5 * yborder:
                        trace_is_within_bounds = False
                    if trace_is_within_bounds:
                        logger.info(f"Refining trace {itrace+1}/{ntraces} of slice ID {sliceid}...")
                        for icol in icolumns_to_analyze:
                            debugplot = 0
                            if plots:
                                if icol in icolumns_to_plot:
                                    debugplot = 0  # change to 2 for debugging
                            # predict the peak position using the polynomial trace
                            ipeak = (poly_trace(icol) + 0.5).astype(int)  # (0-based) rounded integer
                            # find the peak position in the smoothed image data around the predicted position
                            naround_peak = 7  # number of pixels around the peak to use for refinement
                            ycut_refinement = image_data_filtered[:, icol][
                                ipeak - naround_peak // 2 : ipeak + naround_peak // 2 + 1
                            ]
                            ipeak_local = np.nanargmax(ycut_refinement)
                            # refine the peak position
                            ipeak = ipeak - naround_peak // 2 + ipeak_local  # (0-based)
                            # refine the peak position to sub-pixel accuracy using a polynomial fit
                            naround_peak = 5  # number of pixels around the peak to use for refinement
                            ycut_refinement = image_data_filtered[:, icol][
                                ipeak - naround_peak // 2 : ipeak + naround_peak // 2 + 1
                            ]
                            fxpeaks, _ = refine_peaks_spectrum(
                                sx=ycut_refinement,
                                ixpeaks=np.array([naround_peak // 2]),  # peak is at the center of the window
                                nwinwidth=3,
                                method="poly2",
                                title=f"Slice ID {sliceid}, Trace {itrace+1} / {ntraces}, Column {icol+1} (Refinement)",
                                debugplot=debugplot,
                            )
                            deltay_refined[itrace, icol] = (fxpeaks[0] + ipeak - naround_peak // 2) - poly_trace(icol)
                        # fit a polynomial of degree degrefine to the refinement
                        if plots:
                            debugplot = 2
                        else:
                            debugplot = 0
                        poly_refinement, _, _ = polfit_residuals_with_sigma_rejection(
                            x=icolumns_to_analyze,
                            y=deltay_refined[itrace][icolumns_to_analyze],
                            deg=degrefine,
                            times_sigma_reject=3.0,
                            ylimres_with_rejected=True,
                            xlabel="array index along NAXIS1",
                            ylabel="delta array index along NAXIS2",
                            title=f"Slice ID {sliceid}, Refinement of trace {itrace+1} / {ntraces}",
                            debugplot=debugplot,
                        )
                        # update the polynomial trace
                        # (note that instances of the class Polynomial can be added together to produce a new
                        # instance of the class Polynomial that represents the sum of the two polynomials)
                        list_poly_traces_slice_refined[itrace] += poly_refinement
                    else:
                        logger.warning(
                            f"Trace {itrace+1}/{ntraces} of slice ID {sliceid} is out of bounds. Skipping refinement."
                        )
                if np.all(np.isnan(deltay_refined)):
                    logger.warning(f"All traces of slice ID {sliceid} are out of bounds. Skipping average refinement.")
                else:
                    # compute the average refinement across all traces (work only with valid
                    # columns, i.e., those that have at least one valid refinement value,
                    # to avoid a RuntimeWarning when computing the mean of an empty slice)
                    ivalid_cols = ~np.all(np.isnan(deltay_refined), axis=0)
                    deltay_refined_mean = np.full(deltay_refined.shape[1], np.nan, dtype=float)
                    deltay_refined_mean[ivalid_cols] = np.nanmean(deltay_refined[:, ivalid_cols], axis=0)
                    # fit a polynomial of degree degrefine to the average refinement
                    if plots:
                        debugplot = 2
                    else:
                        debugplot = 0
                    poly_refinement_mean, _, _ = polfit_residuals_with_sigma_rejection(
                        x=icolumns_to_analyze,
                        y=deltay_refined_mean[icolumns_to_analyze],
                        deg=degrefine,
                        times_sigma_reject=3.0,
                        ylimres_with_rejected=True,
                        xlabel="array index along NAXIS1",
                        ylabel="delta array index along NAXIS2",
                        title=f"Slice ID {sliceid}, Average Refinement of all traces",
                        debugplot=debugplot,
                    )
                    # update the polynomial trace for each trace out of bounds in the previous refinement step
                    for itrace in range(ntraces):
                        if np.all(np.isnan(deltay_refined[itrace])):
                            logger.info(
                                f"Applying average refinement to Trace {itrace+1}/{ntraces} of slice ID {sliceid}."
                            )
                            list_poly_traces_slice_refined[itrace] += poly_refinement_mean

            # show extrapolated (and when required refined) traces over the smoothed image data
            if plots:
                fig, ax = plt.subplots(figsize=(10, 8))
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
                ixdum = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
                ymin = np.min(list_poly_left[islice](ixdum)) - 10
                ymax = np.max(list_poly_right[islice](ixdum)) + 10
                xcenter = np.mean(icolumns_to_analyze)
                for itrace in range(ntraces):
                    if itrace == 0:
                        label1 = f"extrapolated trace"
                        label2 = f"extrapolated and refined trace"
                    else:
                        label1 = None
                        label2 = None
                    poly_trace = list_poly_traces_slice[itrace]
                    ytrace = poly_trace(ixdum)
                    ax.plot(ixdum, ytrace, color="magenta", lw=0.5, ls="--", label=label1)
                    if refine:
                        poly_trace_refined = list_poly_traces_slice_refined[itrace]
                        ytrace_refined = poly_trace_refined(ixdum)
                        ax.plot(ixdum, ytrace_refined, color="cyan", lw=1.5, ls="-", label=label2)
                        ycenter = poly_trace_refined(xcenter)
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
                ax.set_ylim(ymin, ymax)
                ax.legend()
                plt.tight_layout()  # Fails if sliceid=True in plot_fitted_boundary_polynomials
                plt.show()

            # update the list of polynomial traces for all slices with the extrapolated
            # (and refined when requested) traces for the skipped slice
            list_poly_traces_all_slices[islice] = list_poly_traces_slice_refined

    # final double-check of the list of polynomial traces for all slices
    for islice in range(FRIDA_NSLICES):
        if list_poly_traces_all_slices[islice] is None:
            sliceid = sliceid_from_sliceindex(islice)
            raise ValueError(
                f"Slice {islice+1} (ID {sliceid}) was skipped and could not be extrapolated from other slices."
            )
        if len(list_poly_traces_all_slices[islice]) != ntraces:
            sliceid = sliceid_from_sliceindex(islice)
            raise ValueError(
                f"Slice {islice+1} (ID {sliceid}) has {len(list_poly_traces_all_slices[islice])} traces, "
                f"but {ntraces} traces were expected."
            )
        for itrace in range(ntraces):
            poly_trace = list_poly_traces_all_slices[islice][itrace]
            if poly_trace is None:
                sliceid = sliceid_from_sliceindex(islice)
                raise ValueError(f"Trace {itrace+1} of slice {islice+1} (ID {sliceid}) could not be fitted.")

    # save PDF file with the final plots of traces for every slice
    if pdf_out is not None:
        logger.info(f"Saving final plots of traces for every slice in PDF file: {pdf_out}")
        if output_dir is not None:
            pdf_output = PdfPages(Path(output_dir) / pdf_out)
        else:
            pdf_output = PdfPages(pdf_out)
        vmin, vmax = ZScaleInterval().get_limits(image_data_filtered)
        for islice in range(
            FRIDA_NSLICES - 1, -1, -1
        ):  # (0-based index) loop in reverse order to have the first slice on top of the PDF
            sliceid = sliceid_from_sliceindex(islice)
            fig, ax = plt.subplots(figsize=(10, 6))
            tea.imshow(
                fig,
                ax,
                image_data_filtered,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                ds9mode=False,  # note that the polynomials are fitted using array indices (0-based)
                title=f"{Path(image_path).name} - xmedian={xmedian}\n"
                f"Final Traces for Slice {islice+1} (ID {sliceid})",
            )
            # set sliceid=False in plot_fitted_boundary_polynomials to avoid a problem
            # in plt.tight_layout() when sliceid=True (it fails to compute the layout properly)
            plot_fitted_boundary_polynomials(
                ax=ax, list_poly_left=list_poly_left, list_poly_right=list_poly_right, voffset=0.0, sliceid=False
            )
            ixdum = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
            ymin = np.min(list_poly_left[islice](ixdum)) - 10
            ymax = np.max(list_poly_right[islice](ixdum)) + 10
            xcenter = np.mean(icolumns_to_analyze)
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
            ax.set_ylim(ymin, ymax)
            plt.tight_layout()  # Fails if sliceid=True in plot_fitted_boundary_polynomials
            pdf_output.savefig(fig, bbox_inches="tight")
        pdf_output.close()

    # return the list of polynomial traces for all slices
    return list_poly_left, list_poly_right, list_poly_traces_all_slices


def main(args=None):

    datetime_ini = datetime.now()

    parser = argparse.ArgumentParser(
        description="Find traces within slice boundary polynomials", formatter_class=RichHelpFormatter
    )
    parser.add_argument("--image", help="Path to the input image file (FITS format)", type=str, required=True)
    parser.add_argument("--poly", help="Path to the input file with the boundary polynomials", type=str, required=True)
    parser.add_argument(
        "--voffset",
        help="Vertical offset to apply to the slice boundaries (default: 0.0 pixels; + upward, - downward). "
        "If more than one value is provided, a polynomial of degree len(voffset)-1 will be fitted to the offsets and applied to the slice boundaries.",
        type=float,
        nargs="+",
        default=[0.0],
    )
    parser.add_argument("--ntraces", help="Number of traces per slice to find", type=int, required=True)
    parser.add_argument(
        "--nextend",
        help="Number of pixels to extend the slice boundaries for trace finding (default: 2)",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--nclean-around-peaks",
        help="Number of pixels to clean at both sides of each peak (default: -1; set to zero pixels moving outward from the peak while the signal keeps decreasing)",
        type=int,
        default=-1,
    )
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
    parser.add_argument(
        "--theilsen",
        help="Use initial Theil-Sen regression to reject outliers in the polynomial fitting",
        action="store_true",
    )
    parser.add_argument(
        "--degslice", help="Degree of the polynomial to fit traces across slices (default: 2)", type=int, default=2
    )
    parser.add_argument(
        "--degrefine", help="Degree of the polynomial to refine the trace positions (default: 2)", type=int, default=2
    )
    parser.add_argument("--norefine", help="Do not refine the extrapolated traces", action="store_true")
    parser.add_argument("--output", help="Output file name for the predicted polynomials", type=str, required=True)
    parser.add_argument("--overwrite", help="Overwrite existing output file", action="store_true")
    parser.add_argument(
        "--plotsliceid",
        help="Display plots for slice id (this option can be specified multiple times)",
        type=int,
        action="append",
        default=None,
    )
    parser.add_argument(
        "--pdf-out",
        help="Output PDF file for final plots with traces for every slice (default: None, no output)",
        type=str,
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
    list_poly_left, list_poly_right, list_poly_traces_all_slices = find_traces_within_slice_boundary_polynomials(
        image_path=args.image,
        poly_path=args.poly,
        voffset=args.voffset,
        ntraces=args.ntraces,
        nextend=args.nextend,
        nclean_around_peaks=args.nclean_around_peaks,
        deg=args.deg,
        xmedian=args.xmedian,
        theilsen=args.theilsen,
        columns_to_analyze=columns_to_analyze,
        degslice=args.degslice,
        degrefine=args.degrefine,
        refine=not args.norefine,
        plotsliceid=args.plotsliceid,
        pdf_out=args.pdf_out,
        output_dir=args.output_dir,
    )

    # Save output_fname with the results, including:
    # - polynomial coefficients for the two boundaries of each slice (from the input file)
    # - slice widths for each slice (computed from the two previous boundaries)
    # - polynomial coefficients for the traces of each slice (from the output of this function)
    poldeg = len(list_poly_left[0].convert().coef) - 1
    array_coefs_left = np.full((FRIDA_NSLICES, poldeg + 1), np.nan, dtype=float)
    for islice in range(FRIDA_NSLICES):
        array_coefs_left[islice, :] = list_poly_left[islice].convert().coef
    array_coefs_right = np.full((FRIDA_NSLICES, poldeg + 1), np.nan, dtype=float)
    for islice in range(FRIDA_NSLICES):
        array_coefs_right[islice, :] = list_poly_right[islice].convert().coef
    array_widths = np.full((FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)
    xdum = np.arange(FRIDA_NAXIS1_HAWAII.value)  # (0-based)
    for islice in range(FRIDA_NSLICES):
        y_left = list_poly_left[islice](xdum)
        y_right = list_poly_right[islice](xdum)
        array_widths[islice, :] = y_right - y_left
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
    primary_hdu.header["KEYCODE"] = "SLICE_TRACES_POLYNOMIALS"
    primary_hdu.header["UUID"] = str(uuid.uuid4())
    primary_hdu.header["UUID-POL"] = fits.getheader(args.poly, extension=0)["UUID"]
    primary_hdu.header["POLYFILE"] = Path(args.poly).name
    primary_hdu.header["IMAGFILE"] = Path(args.image).name
    primary_hdu.header["OUTFILE"] = Path(output_fname).name
    primary_hdu.header["POLDEG"] = (poldeg, "Degree of the slice boundary polynomials")
    primary_hdu.header["TRACDEG"] = (args.deg, "Degree of the trace polynomials")
    primary_hdu.header["TRACESLC"] = (args.ntraces, "Number of traces per slice")
    primary_hdu.header["SLCNUMT"] = (FRIDA_NSLICES, "Number of slices with traces")
    for i in range(1, FRIDA_NSLICES + 1):
        primary_hdu.header[f"SLCNUM{i:02d}"] = (
            True,
            f"Slice number {i:02d} (ID: {sliceid_from_sliceindex(i-1):02d}) is included",
        )
    add_script_info_to_fits_history(primary_hdu.header, args, title="Traces within slice boundary polynomials")
    hdul = [primary_hdu, hdu1, hdu2, hdu3]
    # Generate an extension SLCNUMXX for each slice with the polynomial coefficients of the traces
    for islice in range(FRIDA_NSLICES):
        sliceid = sliceid_from_sliceindex(islice)
        if list_poly_traces_all_slices[islice] is None:
            raise ValueError(f"Slice {islice+1} (ID {sliceid}) has no traces. This should not happen.")
        if len(list_poly_traces_all_slices[islice]) != args.ntraces:
            raise ValueError(
                f"Slice {islice+1} (ID {sliceid}) has {len(list_poly_traces_all_slices[islice])} traces, "
                f"but {args.ntraces} traces were expected."
            )
        array2d_coeffs = np.full((args.ntraces, args.deg + 1), np.nan, dtype=float)
        for itrace in range(args.ntraces):
            poly_trace = list_poly_traces_all_slices[islice][itrace]
            # convert to standard polynomial representation and get coefficients
            array2d_coeffs[itrace] = poly_trace.convert().coef
        hdu = fits.ImageHDU(data=array2d_coeffs)
        hdu.header.comments["NAXIS1"] = "Degree of the polynomial + 1"
        hdu.header.comments["NAXIS2"] = "Number of traces per slice"
        hdu.header["EXTNAME"] = f"SLCNUM{islice+1:02d}"
        hdu.header["SLICEID"] = (sliceid, "Slice ID")
        hdu.header["TRACEDEG"] = (len(array2d_coeffs[itrace]) - 1, "Degree of the polynomial")
        hdu.header["COMMENT"] = f"Polynomial coefficients for traces in SLCTNUM{islice+1:02d}"
        hdul.append(hdu)
    hdul = fits.HDUList(hdul)
    hdul.writeto(output_fname, overwrite=args.overwrite)
    logger.info(f"Traces within slice boundary polynomials saved to [green]{output_fname}[/green]")

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
