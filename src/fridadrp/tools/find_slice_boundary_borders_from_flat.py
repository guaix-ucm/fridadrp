#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Find slice boundary borders from flats"""

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
from scipy.ndimage import median_filter, generic_filter
from scipy.signal import savgol_filter
import sys
import teareduce as tea
from tqdm import tqdm
import types

from numina.array.wavecalib.peaks_spectrum import find_highest_peaks_spectrum, find_peaks_spectrum
from numina.tools.add_script_info_to_fits_history import add_script_info_to_fits_history
from numina.tools.input_number import input_number
from numina.user.console import NuminaConsole

from fridadrp._version import version
from fridadrp.core import FRIDA_NAXIS1_HAWAII, FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import slicenum_from_index
from fridadrp.tools.columns_to_analyze_from_colranges import columns_to_analyze_from_colranges


def find_slice_boundary_borders_from_flat(
    flatfile,
    columns_to_analyze,
    slice_ini,
    slice_end,
    row_ini,
    row_end,
    median_filter_xsize=21,
    savgol_window_length=5,
    savgol_polyorder=2,
    plots=False,
):
    """Find slice boundary borders from flat image

    Important: the slice boundaries are determined using array indices
    along NAXIS2, which ranges from 0 to FRIDA_NAXIS2_HAWAII-1, and not using
    the physical coordinates along NAXIS2, which range from 1 to FRIDA_NAXIS2_HAWAII.

    This function starts by smoothing the flat data using a median filter
    along NAXIS1 to remove bad pixels. Then, it applies a Savitzky-Golay filter
    along NAXIS2 to smooth the data and compute the first and second derivatives.
    The code detects NFRIDA_NSLICES peaks in the first derivative (up and down)
    and checks that they are in the expected order. Then it finds the closest peaks
    in the second derivative and checks that they are also in the expected order.
    Finally, it computes the slice boundaries by fitting a line using for each
    boundary the points enclosed between the first and second derivative peaks.
    This fit is used to compute the slice boundary at a level of 1% of the difference
    between the median signal within the slice and the minimum signal in the
    corresponding gap between slices. The function returns two arrays containing
    the left and right slice boundaries for each slice and each column.

    If plots is True, it displays plots of the data, the smoothed data,
    the first and second derivatives. If columns_to_analyze only specify a single
    column, it analyzes only that column and displays the slice boundaries for it.

    Parameters
    ----------
    flatfile : str
        Path to the flat file.
    columns_to_analyze : list of int
        List of columns to analyze (1-based index along NAXIS1).
    slice_ini : int
        Initial slice number (1-based index).
    slice_end : int
        Final slice number (1-based index).
    row_ini : int
        Initial row number (1-based index along NAXIS2).
    row_end : int
        Final row number (1-based index along NAXIS2).
    median_filter_xsize : int, optional
        Size of the median filter to apply to the flat data to remove
        bad pixels.
    savgol_window_length : int, optional
        Window length for the Savitzky-Golay filter to smooth the data.
    savgol_polyorder : int, optional
        Polynomial order for the Savitzky-Golay filter to smooth the data
        and compute the first and second derivatives.
    plots : bool, optional
        If True, display plots of the data and the slice boundaries.

    Returns
    -------
    array_left_border : np.ndarray
        Array of shape (FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII) containing the
        left border of each slice for each column.
    array_right_border : np.ndarray
        Array of shape (FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII) containing the
        right border of each slice for each column.

    """
    logger = logging.getLogger(__name__)

    with fits.open(flatfile) as hdul:
        flat_data = hdul[0].data

    naxis2, naxis1 = flat_data.shape
    if naxis1 != FRIDA_NAXIS1_HAWAII.value or naxis2 != FRIDA_NAXIS2_HAWAII.value:
        raise ValueError(
            f"Flat file dimensions ({naxis1}, {naxis2}) do not match expected dimensions ({FRIDA_NAXIS1_HAWAII.value}, {FRIDA_NAXIS2_HAWAII.value})"
        )

    # Set to zero the pixels outside the specified row range (1-based index along NAXIS2)
    flat_data[: (row_ini - 1), :] = 0.0
    flat_data[row_end:, :] = 0.0

    # Median filter the flat data to remove bad pixels. If there are NaN values,
    # use generic_filter with np.nanmedian to ignore NaN values. Otherwise, use median_filter directly,ç
    # which is faster.
    if median_filter_xsize % 2 == 0:
        median_filter_xsize += 1  # Ensure the median filter size is odd
        logger.debug(f"Median filter size adjusted to {median_filter_xsize} to ensure it is odd.")
    if median_filter_xsize >= 3:
        if np.isnan(flat_data).any():
            logger.debug("NaN values found in flat data. Using generic_filter with np.nanmedian to ignore NaN values.")
            flat_data_filtered = generic_filter(flat_data, np.nanmedian, size=(1, median_filter_xsize), mode="nearest")
        else:
            logger.debug("No NaN values found in flat data. Using median_filter directly.")
            flat_data_filtered = median_filter(flat_data, size=(1, median_filter_xsize), mode="nearest")
    else:
        logger.warning(f"Median filter size {median_filter_xsize} is less than 3. Skipping median filtering.")
        flat_data_filtered = flat_data.copy()

    # Apply Savitzky-Golay filter along the slice direction (axis=0) to smooth the data
    # and compute first and second derivatives to find the slice boundaries
    flat_data_savgol_deriv1 = savgol_filter(
        x=flat_data_filtered,
        window_length=savgol_window_length,
        polyorder=savgol_polyorder,
        axis=0,
        deriv=1,
        mode="nearest",
    )
    flat_data_savgol_deriv2 = savgol_filter(
        x=flat_data_filtered,
        window_length=savgol_window_length,
        polyorder=savgol_polyorder,
        axis=0,
        deriv=2,
        mode="nearest",
    )
    if plots:
        fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(12, 10), sharex=True, sharey=True)
        axarr = axarr.flatten()
        vmin, vmax = ZScaleInterval().get_limits(flat_data_filtered)
        tea.imshow(
            fig,
            axarr[0],
            data=flat_data,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            aspect="auto",
            ds9mode=True,
            title=f"{Path(flatfile).name}\nOriginal Flat Data",
        )
        tea.imshow(
            fig,
            axarr[1],
            data=flat_data_filtered,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            aspect="auto",
            ds9mode=True,
            title=f"{Path(flatfile).name}\nMedian Filtered Flat Data",
        )
        vmin, vmax = ZScaleInterval().get_limits(flat_data_savgol_deriv1)
        tea.imshow(
            fig,
            axarr[2],
            data=flat_data_savgol_deriv1,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            aspect="auto",
            ds9mode=True,
            title=f"{Path(flatfile).name}\nSavitzky-Golay First Derivative Filtered Flat Data",
        )
        vmin, vmax = ZScaleInterval().get_limits(flat_data_savgol_deriv2)
        tea.imshow(
            fig,
            axarr[3],
            data=flat_data_savgol_deriv2,
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
            aspect="auto",
            ds9mode=True,
            title=f"{Path(flatfile).name}\nSavitzky-Golay Second Derivative Filtered Flat Data",
        )
        mpl.rcParams["keymap.forward"] = []  # disable 'v' (conflict with 'v' for setting vmin/vmax)

        def on_key(event):
            if event.key == "?":
                logger.info("-" * 79)
                logger.info("'?': show this help message")
                logger.info("'a': toggle imshow aspect='equal' / aspect='auto'")
                logger.info("'v': set vmin and vmax manually")
                logger.info("',': set vmin and vmax to min and max of the zoomed region")
                logger.info("'/': set vmin and vmax using zscale of the zoomed region")
                logger.info("'q': quit")
                logger.info("-" * 79)
            elif event.key == "a":
                for iax, ax in enumerate(axarr):
                    if ax.get_aspect() in ["equal", 1.0]:
                        if iax == 0:
                            logger.info("Setting aspect to 'auto' for all axes.")
                        ax.set_aspect("auto")
                    else:
                        if iax == 0:
                            logger.info("Setting aspect to 'equal' for all axes.")
                        ax.set_aspect("equal")
                fig.set_tight_layout(False)  # deactivate accumulated tight_layout adjustments
                fig.tight_layout()  # apply new tight_layout adjustments
                fig.canvas.draw()
            elif event.key in ["v", ",", "/"]:
                if event.inaxes in axarr:
                    if event.inaxes == axarr[0]:
                        data = flat_data
                    elif event.inaxes == axarr[1]:
                        data = flat_data_filtered
                    elif event.inaxes == axarr[2]:
                        data = flat_data_savgol_deriv1
                    else:
                        data = flat_data_savgol_deriv2
                    if event.key == "v":
                        current_vmin, current_vmax = event.inaxes.images[0].get_clim()
                        vmin = input_number(expected_type="float", prompt="Enter vmin: ", default=current_vmin)
                        vmax = input_number(expected_type="float", prompt="Enter vmax: ", default=current_vmax)
                    else:
                        xlim = event.inaxes.get_xlim()
                        ylim = event.inaxes.get_ylim()
                        x1, x2 = int(xlim[0]), int(xlim[1])
                        y1, y2 = int(ylim[0]), int(ylim[1])
                        x1 = max(0, min(x1, FRIDA_NAXIS1_HAWAII.value - 1))
                        x2 = max(0, min(x2, FRIDA_NAXIS1_HAWAII.value - 1))
                        y1 = max(0, min(y1, FRIDA_NAXIS2_HAWAII.value - 1))
                        y2 = max(0, min(y2, FRIDA_NAXIS2_HAWAII.value - 1))
                        if event.key == ",":
                            vmin = np.nanmin(data[y1 : (y2 + 1), x1 : (x2 + 1)])
                            vmax = np.nanmax(data[y1 : (y2 + 1), x1 : (x2 + 1)])
                        else:
                            vmin, vmax = ZScaleInterval().get_limits(data[y1 : (y2 + 1), x1 : (x2 + 1)])
                    event.inaxes.images[0].set_clim(vmin, vmax)
                    fig.set_tight_layout(False)  # deactivate accumulated tight_layout adjustments
                    fig.tight_layout()  # apply new tight_layout adjustments
                    fig.canvas.draw()
            elif event.key == "q":
                plt.close(fig)

        on_key(event=types.SimpleNamespace(key="?"))  # Show help message on startup
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.set_tight_layout(False)  # deactivate accumulated tight_layout adjustments
        fig.tight_layout()  # apply new tight_layout adjustments
        # instead of plt.show(), use a loop to keep the figure open until closed by the user
        # (otherwise, after using input_number() in the on_key function, the matplotlib event
        # loop is not properly restored and the execution of the code continues as if the
        # figure was closed, which is not the case)
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)
        ## plt.show()

    # Define arrays to store slice boundaries for all the FRIDA_NSLICES slices
    array_left_border = np.full((FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)
    array_right_border = np.full((FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)

    # Number of slices to be found
    nslices_eff = slice_end - slice_ini + 1

    # Main loop
    num_useful_columns = 0
    for col in tqdm(
        columns_to_analyze,
        desc="Working on columns",
    ):
        if len(columns_to_analyze) == 1:
            plots_extra = plots  # Only plot for the specified column
        else:
            plots_extra = False
        ydata = flat_data[:, col - 1]
        ydata_smoothed = flat_data_filtered[:, col - 1]
        yderiv1 = flat_data_savgol_deriv1[:, col - 1]
        yderiv2 = flat_data_savgol_deriv2[:, col - 1]
        # Find the highest FRIDA_NSLICES peaks in the 1st derivative (up and down refer to the positive and negative peaks)
        list_ipeaks_deriv1_up = find_highest_peaks_spectrum(
            sx=yderiv1,
            nmaxpeaks=nslices_eff,
            nclean_around_peak=40,
            nwinwidth=5,
            nborder_to_ignore=5,
            debugplot=0,
        )
        list_ipeaks_deriv1_down = find_highest_peaks_spectrum(
            sx=-yderiv1,
            nmaxpeaks=nslices_eff,
            nclean_around_peak=40,
            nwinwidth=5,
            nborder_to_ignore=5,
            debugplot=0,
        )
        # Check that the up and down peaks are in the expected order (up, down, up, down, ...)
        flag_deriv1_ok = True
        for islice in range(nslices_eff):
            if list_ipeaks_deriv1_up[islice] > list_ipeaks_deriv1_down[islice]:
                flag_deriv1_ok = False
                logger.debug(
                    f"Column {col}: Up peak {list_ipeaks_deriv1_up[islice]} is after down peak {list_ipeaks_deriv1_down[islice]}."
                )
        for igap in range(nslices_eff - 1):
            if list_ipeaks_deriv1_down[igap] > list_ipeaks_deriv1_up[igap + 1]:
                flag_deriv1_ok = False
                logger.debug(
                    f"Column {col}: Down peak {list_ipeaks_deriv1_down[igap]} is after next up peak {list_ipeaks_deriv1_up[igap + 1]}."
                )
        if flag_deriv1_ok:
            # Median signal within each slice
            list_perc90 = np.zeros(nslices_eff, dtype=float)
            for islice in range(nslices_eff):
                list_perc90[islice] = np.percentile(
                    ydata_smoothed[list_ipeaks_deriv1_up[islice] : list_ipeaks_deriv1_down[islice]], 90
                )
            # Location of minimum between slices
            imin_between_slices = np.zeros(nslices_eff - 1, dtype=int)
            for igap in range(nslices_eff - 1):
                imin_between_slices[igap] = (
                    np.argmin(ydata_smoothed[list_ipeaks_deriv1_down[igap] : (list_ipeaks_deriv1_up[igap + 1] + 1)])
                    + list_ipeaks_deriv1_down[igap]
                )
            # Find all the positive peaks in the 2nd derivative
            list_ipeaks_deriv2 = find_peaks_spectrum(
                sx=yderiv2,
                nwinwidth=3,
                debugplot=0,
            )
            # Closest peak in the 2nd derivative to each peak in the 1st derivative
            list_ipeaks_deriv2_up = np.zeros(nslices_eff, dtype=int)
            for islice in range(nslices_eff):
                ipeak_deriv1_up = list_ipeaks_deriv1_up[islice]
                idx = np.searchsorted(list_ipeaks_deriv2, ipeak_deriv1_up, side="left") - 1
                if idx < 0:
                    raise ValueError(f"No peak found in 2nd derivative for 1st derivative peak at {ipeak_deriv1_up}")
                list_ipeaks_deriv2_up[islice] = list_ipeaks_deriv2[idx]
            list_ipeaks_deriv2_down = np.zeros(nslices_eff, dtype=int)
            for islice in range(nslices_eff):
                ipeak_deriv1_down = list_ipeaks_deriv1_down[islice]
                idx = np.searchsorted(list_ipeaks_deriv2, ipeak_deriv1_down, side="right")
                if idx < 0:
                    raise ValueError(f"No peak found in 2nd derivative for 1st derivative peak at {ipeak_deriv1_down}")
                list_ipeaks_deriv2_down[islice] = list_ipeaks_deriv2[idx]
            # Check that the up and down peaks in the 2nd derivative are also in the expected order
            flag_deriv2_ok = True
            for islice in range(nslices_eff):
                if list_ipeaks_deriv2_up[islice] > list_ipeaks_deriv2_down[islice]:
                    flag_deriv2_ok = False
                    logger.debug(
                        f"Column {col}: 2nd derivative up peak {list_ipeaks_deriv2_up[islice]} is after down peak {list_ipeaks_deriv2_down[islice]}."
                    )
            for igap in range(nslices_eff - 1):
                if list_ipeaks_deriv2_down[igap] > list_ipeaks_deriv2_up[igap + 1]:
                    flag_deriv2_ok = False
                    logger.debug(
                        f"Column {col}: 2nd derivative down peak {list_ipeaks_deriv2_down[igap]} is after next up peak {list_ipeaks_deriv2_up[igap + 1]}."
                    )
            for igap in range(nslices_eff - 1):
                x1 = list_ipeaks_deriv1_down[igap]
                x2 = list_ipeaks_deriv1_up[igap + 1]
                xx1 = list_ipeaks_deriv2_down[igap]
                xx2 = list_ipeaks_deriv2_up[igap + 1]
                if not (x1 <= xx1 <= xx2 <= x2):
                    flag_deriv2_ok = False
                    logger.debug(
                        f"Column {col}: 2nd derivative peaks ({xx1}, {xx2}) are not within the 1st derivative peaks ({x1}, {x2})."
                    )
            if flag_deriv2_ok:
                # Display result for column if plots is enabled
                if plots_extra:
                    xdum = np.arange(FRIDA_NAXIS2_HAWAII.value)
                    fig, ax = plt.subplots(figsize=(15, 8))
                    ax.plot(xdum, ydata, "-", color="gray", label="data")
                    ax.plot(xdum, ydata_smoothed, "C0-", label="smoothed data along NAXIS1 axis")
                    ax.plot(
                        list_ipeaks_deriv1_up,
                        ydata_smoothed[list_ipeaks_deriv1_up],
                        "C3o",
                        label="1st derivative peaks (up)",
                    )
                    ax.plot(
                        list_ipeaks_deriv1_down,
                        ydata_smoothed[list_ipeaks_deriv1_down],
                        "C3.",
                        label="1st derivative peaks (down)",
                    )
                    ax.plot(
                        list_ipeaks_deriv2_up,
                        ydata_smoothed[list_ipeaks_deriv2_up],
                        "C2o",
                        label="2nd derivative peaks (up)",
                    )
                    ax.plot(
                        list_ipeaks_deriv2_down,
                        ydata_smoothed[list_ipeaks_deriv2_down],
                        "C2.",
                        label="2nd derivative peaks (down)",
                    )
                    ymin, ymax = ax.get_ylim()
                    dy = ymax - ymin
                    for islice in range(nslices_eff):
                        x1 = list_ipeaks_deriv1_up[islice]
                        x2 = list_ipeaks_deriv1_down[islice]
                        y1 = ydata_smoothed[x1]
                        y2 = ydata_smoothed[x2]
                        ymid = (y1 + y2) / 2
                        ax.plot([x1, x2], [ymid] * 2, "k-", lw=2)
                        xmid = (x1 + x2) / 2
                        ax.text(
                            xmid,
                            ymid + dy / 100,
                            f"#{slicenum_from_index(islice + slice_ini - 1)}",
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            color="k",
                        )
                        label = None
                        if islice == 0:
                            label = "percentile 90 (smoothed data)"
                        ax.plot(xmid, list_perc90[islice], "ko", label=label)
                    ax.plot(
                        imin_between_slices,
                        ydata_smoothed[imin_between_slices],
                        "kx",
                        label="min. in gap between slices (smoothed data)",
                    )
                    ax.set_xlabel("array index along NAXIS2 axis (spatial direction)", fontsize=12)
                    ax.set_ylabel("data value", fontsize=12)
                    ymax += dy * 0.05
                    ax.set_ylim(ymin, ymax)
                    ax.set_title(
                        f"{Path(flatfile).name}\nColumn {col} (from 1 to NAXIS1) - Slice boundaries from flat",
                        fontsize=14,
                    )
                    ax.legend(ncols=4, loc="upper center", fontsize=10)
                    plt.tight_layout()
                    plt.show()
                # Determine the slice boundaries
                list_left_border = np.full(nslices_eff, np.nan, dtype=float)
                list_ymin_left_border = np.full(nslices_eff, np.nan, dtype=float)
                list_right_border = np.full(nslices_eff, np.nan, dtype=float)
                list_ymin_right_border = np.full(nslices_eff, np.nan, dtype=float)
                if plots_extra:
                    fig, axrr = plt.subplots(nrows=6, ncols=5, figsize=(15, 10))
                    axarr = axrr.flatten()
                    for ax in axarr:
                        ax.axis("off")  # Turn off all axes initially
                for igap in range(nslices_eff - 1):
                    x1 = list_ipeaks_deriv1_down[igap]
                    x2 = list_ipeaks_deriv1_up[igap + 1]
                    xx1 = list_ipeaks_deriv2_down[igap]
                    xx2 = list_ipeaks_deriv2_up[igap + 1]
                    xdum = np.arange(x1, x2 + 1)
                    ydum = ydata_smoothed[x1 : (x2 + 1)]
                    ymin_in_gap = ydata_smoothed[imin_between_slices[igap]]
                    # Right border
                    if (xx1 - x1 + 1) >= 2:
                        nfit = xx1 - x1 + 1
                    else:
                        nfit = 2
                    xfit_right = xdum[:nfit]
                    yfit_right = ydum[:nfit]
                    fit_right = Polynomial.fit(xfit_right, yfit_right, deg=1)
                    deltay_right = list_perc90[igap] - ymin_in_gap
                    ymin_border_right = ymin_in_gap + deltay_right * 0.01
                    c0, c1 = fit_right.convert().coef
                    xborder_right = (ymin_border_right - c0) / c1
                    list_right_border[igap] = xborder_right
                    list_ymin_right_border[igap] = ymin_border_right
                    # Left border
                    if (x2 - xx2 + 1) >= 2:
                        nfit = x2 - xx2 + 1
                    else:
                        nfit = 2
                    xfit_left = xdum[-nfit:]
                    yfit_left = ydum[-nfit:]
                    fit_left = Polynomial.fit(xfit_left, yfit_left, deg=1)
                    deltay_left = list_perc90[igap + 1] - ymin_in_gap
                    ymin_border_left = ymin_in_gap + deltay_left * 0.01
                    c0, c1 = fit_left.convert().coef
                    xborder_left = (ymin_border_left - c0) / c1
                    list_left_border[igap + 1] = xborder_left
                    list_ymin_left_border[igap + 1] = ymin_border_left
                    # Display result for each slice if plots is enabled
                    if plots_extra:
                        ax = axarr[igap]
                        ax.axis("on")  # Turn on the axis for this subplot
                        ax.plot(xdum, ydum, ".")
                        igap_eff = igap + slice_ini - 1
                        ax.set_title(f"Gap #{slicenum_from_index(igap_eff)}-{slicenum_from_index(igap_eff + 1)}")
                        ax.set_xlabel("array index along NAXIS2 axis")
                        ax.set_ylabel("data")
                        ax.plot(x1, ydata_smoothed[x1], "C3o")
                        ax.plot(x2, ydata_smoothed[x2], "C3.")
                        ax.plot(xx1, ydata_smoothed[xx1], "C2o")
                        ax.plot(xx2, ydata_smoothed[xx2], "C2.")
                        ax.plot(imin_between_slices[igap], ymin_in_gap, "kx")
                        ax.axhline(ymin_border_right, linestyle="--")
                        ax.axhline(ymin_border_left, linestyle=":")
                        xxdum = np.linspace(xfit_right[0], xfit_right[-1], 100)
                        ax.plot(xxdum, fit_right(xxdum), "-", color="gray")
                        ax.plot(xborder_right, ymin_border_right, "ko")
                        if yfit_right[-1] > ymin_border_right:
                            xxdum = np.array([xfit_right[-1], xborder_right])
                            ax.plot(xxdum, fit_right(xxdum), ":", color="gray")
                        xxdum = np.linspace(xfit_left[0], xfit_left[-1], 100)
                        ax.plot(xxdum, fit_left(xxdum), "-", color="gray")
                        ax.plot(xborder_left, ymin_border_left, "ko")
                        if yfit_left[0] > ymin_border_left:
                            xxdum = np.array([xfit_left[0], xborder_left])
                            ax.plot(xxdum, fit_left(xxdum), ":", color="gray")
                if plots_extra:
                    plt.suptitle(
                        f"{Path(flatfile).name}\nColumn {col} (from 1 to NAXIS1) - Slice boundaries from flat",
                        fontsize=14,
                    )
                    plt.tight_layout()
                    plt.show()
                # Compute left border of the first slice
                x1 = 0
                x2 = list_ipeaks_deriv1_up[0]
                xx2 = list_ipeaks_deriv2_up[0]
                xdum = np.arange(x1, x2 + 1)
                ydum = ydata_smoothed[x1 : (x2 + 1)]
                if (x2 - xx2 + 1) >= 2:
                    nfit = x2 - xx2 + 1
                else:
                    nfit = 2
                xfit_left = xdum[-nfit:]
                yfit_left = ydum[-nfit:]
                fit_left = Polynomial.fit(xfit_left, yfit_left, deg=1)
                ymin_border_left = list_ymin_right_border[0]  # Use the same ymin as the right border of the first slice
                c0, c1 = fit_left.convert().coef
                xborder_left = (ymin_border_left - c0) / c1
                list_left_border[0] = xborder_left
                list_ymin_left_border[0] = ymin_border_left
                if plots_extra:
                    fig, ax = plt.subplots()
                    ax.plot(xdum, ydum, ".")
                    ax.set_title(f"Left border of slice #{slicenum_from_index(slice_ini - 1)}")
                    ax.set_xlabel("array index along NAXIS2 axis")
                    ax.set_ylabel("data")
                    ax.plot(x2, ydata_smoothed[x2], "C3.")
                    ax.plot(xx2, ydata_smoothed[xx2], "C2.")
                    ax.axhline(ymin_border_left, linestyle=":")
                    xxdum = np.linspace(xfit_left[0], xfit_left[-1], 100)
                    ax.plot(xxdum, fit_left(xxdum), "-", color="gray")
                    ax.plot(xborder_left, ymin_border_left, "ko")
                    if yfit_left[0] > ymin_border_left:
                        xxdum = np.array([xfit_left[0], xborder_left])
                        ax.plot(xxdum, fit_left(xxdum), ":", color="gray")
                    plt.tight_layout()
                    plt.show()
                # Compute right border of the last slice
                x1 = list_ipeaks_deriv1_down[-1]
                xx1 = list_ipeaks_deriv2_down[-1]
                x2 = FRIDA_NAXIS2_HAWAII.value - 1
                xdum = np.arange(x1, x2 + 1)
                ydum = ydata_smoothed[x1 : (x2 + 1)]
                if (xx1 - x1 + 1) >= 2:
                    nfit = xx1 - x1 + 1
                else:
                    nfit = 2
                xfit_right = xdum[:nfit]
                yfit_right = ydum[:nfit]
                fit_right = Polynomial.fit(xfit_right, yfit_right, deg=1)
                ymin_border_right = list_ymin_left_border[-1]  # Use the same ymin as the left border of the last slice
                c0, c1 = fit_right.convert().coef
                xborder_right = (ymin_border_right - c0) / c1
                list_right_border[-1] = xborder_right
                list_ymin_right_border[-1] = ymin_border_right
                if plots_extra:
                    fig, ax = plt.subplots()
                    ax.plot(xdum, ydum, ".")
                    ax.set_title(f"Right border of slice #{slicenum_from_index(slice_end - 1)}")
                    ax.set_xlabel("array index along NAXIS2 axis")
                    ax.set_ylabel("data")
                    ax.plot(x1, ydata_smoothed[x1], "C3.")
                    ax.plot(xx1, ydata_smoothed[xx1], "C2o")
                    ax.axhline(ymin_border_right, linestyle="--")
                    xxdum = np.linspace(xfit_right[0], xfit_right[-1], 100)
                    ax.plot(xxdum, fit_right(xxdum), "-", color="gray")
                    ax.plot(xborder_right, ymin_border_right, "ko")
                    if yfit_right[-1] > ymin_border_right:
                        xxdum = np.array([xfit_right[-1], xborder_right])
                        ax.plot(xxdum, fit_right(xxdum), ":", color="gray")
                    plt.tight_layout()
                    plt.show()
                # Display the final slice boundaries for this column
                if plots_extra:
                    fig, ax = plt.subplots(nrows=6, ncols=5, figsize=(15, 10))
                    axarr = ax.flatten()
                    for ax in axarr:
                        ax.axis("off")  # Turn off all axes initially
                    xdum = np.arange(FRIDA_NAXIS2_HAWAII.value)
                    for islice in range(nslices_eff):
                        ax = axarr[islice]
                        ax.axis("on")  # Turn on the axis for this subplot
                        ax.plot(xdum, ydata, "-", color="gray")
                        ax.plot(xdum, ydata_smoothed, "C0-")
                        ax.axhline(0, linestyle=":", color="gray")
                        ax.plot(xdum, yderiv1 * 1.5, "C3-")
                        ax.plot(xdum, yderiv2 * 4.0, "C2-")
                        ax.plot(list_ipeaks_deriv1_up, ydata_smoothed[list_ipeaks_deriv1_up], "C3o")
                        ax.plot(list_ipeaks_deriv1_down, ydata_smoothed[list_ipeaks_deriv1_down], "C3.")
                        ax.plot(list_ipeaks_deriv2_up, ydata_smoothed[list_ipeaks_deriv2_up], "C2o")
                        ax.plot(list_ipeaks_deriv2_down, ydata_smoothed[list_ipeaks_deriv2_down], "C2.")
                        x1 = list_ipeaks_deriv1_up[islice]
                        x2 = list_ipeaks_deriv1_down[islice]
                        y1 = ydata_smoothed[x1]
                        y2 = ydata_smoothed[x2]
                        ymid = (y1 + y2) / 2
                        ax.plot([x1, x2], [ymid] * 2, "k-", lw=2)
                        xmid = (x1 + x2) / 2
                        ax.set_xlim(xmid - 50, xmid + 50)
                        ax.axvline(list_left_border[islice], linestyle="--", color="k")
                        ax.axvline(list_right_border[islice], linestyle="--", color="k")
                        ax.set_xlabel("array index along NAXIS2 axis")
                        ax.set_ylabel("data")
                        ax.set_title(f"Slice #{slicenum_from_index(islice + slice_ini - 1)}")
                    plt.suptitle(
                        f"{Path(flatfile).name}\nColumn {col} (from 1 to NAXIS1) - Slice boundaries from flat",
                        fontsize=14,
                    )
                    plt.tight_layout()
                    plt.show()
                # Store the slice boundaries for this column
                array_left_border[slice_ini - 1 : slice_end, col - 1] = list_left_border
                array_right_border[slice_ini - 1 : slice_end :, col - 1] = list_right_border
                num_useful_columns += 1
            else:
                logger.debug(f"Column {col}: 2nd derivative peaks are not in the expected order. Skipping this column.")
        else:
            logger.debug(f"Column {col}: 1st derivative peaks are not in the expected order. Skipping this column.")

    logger.info(f"Number of useful columns processed: {num_useful_columns} out of {len(columns_to_analyze)}")

    return array_left_border, array_right_border


def main(args=None):

    datetime_ini = datetime.now()

    parser = argparse.ArgumentParser(
        description="Find the slice boundaries from flat image", formatter_class=RichHelpFormatter
    )
    parser.add_argument("--flatfile", type=str, help="Path to the flat file", required=True)
    parser.add_argument(
        "--output", help="Output FITS file name", type=str, default=None
    )
    parser.add_argument("--slice-ini", help="Initial slice number (1-based index)", type=int, default=1)
    parser.add_argument("--slice-end", help="Final slice number (1-based index)", type=int, default=FRIDA_NSLICES)
    parser.add_argument(
        "--row-ini",
        help="Initial row number (1-based index) along NAXIS2",
        type=int,
        default=FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value,
    )
    parser.add_argument(
        "--row-end",
        help="Final row number (1-based index) along NAXIS2",
        type=int,
        default=FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value,
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
    console.rule(f"[bold magenta]Welcome to fridadrp-find_slice_boundaries_from_flat[/bold magenta]")

    # Display version info
    logger = logging.getLogger(__name__)
    logger.info(f"Using {__name__} version {version}")

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Command line arguments: {args}")

    # Check flat file is defined
    if args.flatfile is None:
        raise ValueError("Flat file is not defined. Use --flatfile to specify the flat file.")

    # Check input parameters
    if args.slice_ini < 1 or args.slice_ini > FRIDA_NSLICES:
        raise ValueError(f"Initial slice number must be between 1 and {FRIDA_NSLICES}.")
    if args.slice_end < 1 or args.slice_end > FRIDA_NSLICES:
        raise ValueError(f"Final slice number must be between 1 and {FRIDA_NSLICES}.")
    if args.slice_ini > args.slice_end:
        raise ValueError("Initial slice number cannot be greater than final slice number.")
    if (
        args.row_ini < FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value
        or args.row_ini > FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value
    ):
        raise ValueError(
            f"Initial row number must be between {FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value} and {FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value}."
        )
    if (
        args.row_end < FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value
        or args.row_end > FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value
    ):
        raise ValueError(
            f"Final row number must be between {FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value} and {FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value}."
        )
    if args.row_ini > args.row_end:
        raise ValueError("Initial row number cannot be greater than final row number.")

    # Define the columns to analyze based on the specified column ranges
    columns_to_analyze = columns_to_analyze_from_colranges(args.colrange)

    # Compute the slice boundaries from the flat file
    array_left_border, array_right_border = find_slice_boundary_borders_from_flat(
        flatfile=args.flatfile,
        columns_to_analyze=columns_to_analyze,
        slice_ini=args.slice_ini,
        slice_end=args.slice_end,
        row_ini=args.row_ini,
        row_end=args.row_end,
        plots=args.plots,
    )

    # Compute slice widths as a function of the array index along the NAXIS1 axis
    array_widths = np.zeros((FRIDA_NSLICES, FRIDA_NAXIS1_HAWAII.value))
    for islice in range(FRIDA_NSLICES):
        array_widths[islice, :] = array_right_border[islice, :] - array_left_border[islice, :]

    # If more than a single column is specified, the slice boundaries are saved to a FITS file
    if len(columns_to_analyze) > 1:
        header1 = fits.Header()
        header1["EXTNAME"] = "L-BORDER"
        hdu1 = fits.ImageHDU(data=array_left_border, header=header1)
        header2 = fits.Header()
        header2["EXTNAME"] = "R-BORDER"
        hdu2 = fits.ImageHDU(data=array_right_border, header=header2)
        header3 = fits.Header()
        header3["EXTNAME"] = "SLIWIDTH"
        hdu3 = fits.ImageHDU(data=array_widths, header=header3)
        primary_hdu = fits.PrimaryHDU()
        primary_hdu.header["FLATFILE"] = Path(args.flatfile).name
        primary_hdu.header["KEYCODE"] = "SLICE_BOUNDARY_BORDERS_FROM_FLAT"
        primary_hdu.header["SLICEINI"] = (args.slice_ini, "Initial slice number (1-based index)")
        primary_hdu.header["SLICEEND"] = (args.slice_end, "Final slice number (1-based index)")
        primary_hdu.header["ROWINI"] = (args.row_ini, "Initial row number (1-based index)")
        primary_hdu.header["ROWEND"] = (args.row_end, "Final row number (1-based index)")
        add_script_info_to_fits_history(primary_hdu.header, args)
        hdul = fits.HDUList([primary_hdu, hdu1, hdu2, hdu3])
        if args.output is None:
            args.output = f"slice_boundary_borders_from_flat_{args.slice_ini}-{args.slice_end}.fits"
        hdul.writeto(args.output, overwrite=True)
        logger.info(f"Slice boundary borders saved to: [green]{args.output}[/green]")
    else:
        logger.info(
            "Slice boundary borders computed for column %d. Not saved to FITS file since a single column is specified.",
            columns_to_analyze[0],
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
