#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Find and characterize arc lines in FRIDA IFU images"""

import argparse
from astropy.io import fits
from astropy.table import Table
import astropy.units as u
from astropy.visualization import ZScaleInterval, MinMaxInterval
import logging
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from rich_argparse import RichHelpFormatter
from scipy import ndimage
from skimage import restoration
import sys
import teareduce as tea
import uuid
import warnings

from numina.array.display.disable_keys_in_figure import disable_keys_in_figure
from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.wavecalib.peaks_spectrum import find_peaks_spectrum, refine_peaks_spectrum
from numina.array.rescale_array_z1z2 import rescale_array_to_z1z2, rescale_array_from_z1z2
from numina.tools.initialize_script_with_args import NuminaScriptDefinition
from numina.tools.input_number import input_number

from fridadrp.core import FRIDA_NAXIS1_HAWAII, FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import sliceid_from_sliceindex, sliceindex_from_sliceid
from fridadrp.core import set_hawaii_border_pixels_to_nan
from fridadrp.tools.overplot_slice_boundary_polynomials import plot_fitted_boundary_polynomials
from fridadrp.tools.read_slice_boundary_polynomials import read_slice_boundary_polynomials


def retrieve_initial_data(image, input_poly, xlim=None, voffset=[0.0]):
    """Retrieve initial data from the input image and slice boundary polynomials

    This functions avoids code duplication in the find_arc_lines and
    characterize_arc_lines functions.

    Parameters
    ----------
    image : str
        Path to the input FITS image file.
    input_poly : str
        Path to the input FITS file containing the slice boundary polynomials.
    xlim : list, optional
        The x-axis limits for the plot.
    voffset : list of float, optional
        Vertical offsets to apply to the polynomial boundaries (pixels).
        If more than one value is provided, a polynomial of degree len(voffset)-1
        will be fitted to the offsets and applied to the displayed
        slice boundary polynomials. A positive value shifts the
        displayed polynomials upwards (i.e., towards higher array indices
        along NAXIS2).

    Returns
    -------
    image_data : numpy.ndarray
        The image data read from the input FITS file.
    list_poly_left : list of Polynomial
        List of polynomial objects representing the left slice boundaries.
    list_poly_right : list of Polynomial
        List of polynomial objects representing the right slice boundaries.
    xlim : list
        The x-axis limits for the plot, adjusted to be 0-based along NAXIS1 axis.
    """
    logger = logging.getLogger(__name__)

    # Read the image data from the input FITS file
    with fits.open(image) as hdul:
        image_data = hdul[0].data.astype(float)  # Convert to float for processing

    # Read the slice boundary polynomials from the input FITS file
    list_poly_left, list_poly_right, _ = read_slice_boundary_polynomials(input_poly)

    # Compute polynomial to take into account voffset. Note that voffset is applied
    # to the slice boundaries and polynomials, but not to the image data.
    if len(voffset) < 1:
        raise ValueError("The list of vertical offsets (voffset) must contain at least one value.")
    else:
        if voffset != [0.0]:
            logger.info(
                f"Fitting a polynomial of degree {len(voffset)-1} " "to the vertical offsets (boundary polynomials)"
            )
        nvoffsets = len(voffset)
        xdum_voffsets = np.linspace(
            FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL.value - 1,
            FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL.value - 1,
            nvoffsets,
        )  # using array indices (0 based)
        if logger.isEnabledFor(logging.DEBUG):
            debugplot = 2
        else:
            debugplot = 0
        poly_voffset, _ = polfit_residuals(
            x=xdum_voffsets,
            y=np.array(voffset),
            deg=len(voffset) - 1,
            xlabel="array index along NAXIS1",
            ylabel="array index along NAXIS2",
            title="Vertical offset to be applied",
            debugplot=debugplot,
        )
        logger.debug("Polynomial coefficients for voffset: %s", poly_voffset)
        # Apply voffset to the slice boundary polynomials
        # apply requested voffset
        logger.debug("Applying voffset to the slice boundary polynomials")
        for islice in range(FRIDA_NSLICES):
            if list_poly_left[islice] is not None:
                list_poly_left[islice] += poly_voffset
            if list_poly_right[islice] is not None:
                list_poly_right[islice] += poly_voffset

    # Set the x-axis limits for the plot (1-based along NAXIS1 axis)
    # These limits should be computed outside the main loop to avoid
    # a systematic shift of the x-axis limits for each slice when
    # xlim is not None.
    if xlim is None:
        xlim = [-0.5, FRIDA_NAXIS1_HAWAII.value - 0.5]
    else:
        xlim[0] -= 1  # (0-based)
        xlim[1] -= 1  # (0-based)

    return image_data, list_poly_left, list_poly_right, xlim


def imshow_slice(
    ax,
    image_slice,
    arc_line_table=None,
    current_arc_line=None,
    xlim=None,
    ylim=None,
    vlim=None,
    vmethod="zscale",
    **kwargs,
):
    """
    Display a slice of the image with optional limits and title.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes on which to display the image.
    image_slice : ndarray
        2D array representing the slice of the image.
    arc_line_table : astropy.table.Table, optional
        Table containing the detected arc lines. If provided, rectangles
        will be drawn around the valid arc lines.
    current_arc_line : int, optional
        The index of the current arc line being processed. If provided,
        an arrow will be drawn pointing to the current arc line.
    xlim : list, tuple or "auto", optional
        The x-axis limits for the plot. If "auto", the limits are
        determined from the finite values of `image_slice`.
    ylim : list, tuple or "auto", optional
        The y-axis limits for the plot. If "auto", the limits are
        determined from the finite values of `image_slice`.
    vlim : list, optional
        The vmin and vmax values for the display. If None, they are
        computed from the finite values of `image_slice` using `vmethod`.
    vmethod : {'zscale', 'minmax'}, optional
        Method used to compute the display limits if `vlim` is None.
    kwargs : dict
        Additional keyword arguments passed to `tea.imshow`.

    Returns
    -------
    result : dict
        Dictionary containing the image, color axis, color bar, and limits.
    """
    if vlim is not None:
        vmin, vmax = vlim
    else:
        if vmethod == "zscale":
            vmin, vmax = ZScaleInterval().get_limits(image_slice)
        elif vmethod == "minmax":
            vmin, vmax = MinMaxInterval().get_limits(image_slice)
        else:
            raise ValueError(f"Unknown method: {vmethod}")

    img, cax, cbar = tea.imshow(
        fig=ax.figure,
        ax=ax,
        data=image_slice,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        **kwargs,
    )

    if xlim == "auto":
        # get the first column and last column with finite values
        finite_cols = np.where(np.isfinite(image_slice).any(axis=0))[0]
        if len(finite_cols) > 0:
            ax.set_xlim(finite_cols[0], finite_cols[-1])
    elif isinstance(xlim, (list, tuple)):
        if len(xlim) == 2:
            ax.set_xlim(xlim)
        else:
            raise ValueError("xlim must be a list or tuple of length 2.")
    else:
        ax.set_xlim(-0.5, image_slice.shape[1] - 0.5)

    if ylim == "auto":
        # get the first row and last row with finite values
        finite_rows = np.where(np.isfinite(image_slice).any(axis=1))[0]
        if len(finite_rows) > 0:
            ax.set_ylim(finite_rows[0], finite_rows[-1])
    elif isinstance(ylim, (list, tuple)):
        if len(ylim) == 2:
            ax.set_ylim(ylim)
        else:
            raise ValueError("ylim must be a list or tuple of length 2.")
    else:
        ax.set_ylim(-0.5, image_slice.shape[0] - 0.5)

    # Add rectangles around valid arc lines if arc_line_table is provided
    if arc_line_table is not None:
        num_features = len(arc_line_table)
        for k in range(num_features):
            if arc_line_table["valid"][k]:
                feature_id = arc_line_table["feature"][k]
                x_start = arc_line_table["x_start"][k]
                x_stop = arc_line_table["x_stop"][k]
                y_start = arc_line_table["y_start"][k]
                y_stop = arc_line_table["y_stop"][k]
                rect = Rectangle(
                    (x_start - 0.5, y_start - 0.5),
                    x_stop - x_start,
                    y_stop - y_start,
                    edgecolor="black",
                    facecolor="none",
                    linewidth=0.5,
                )  # plot a rectangle around the valid arc line
                ax.add_patch(rect)
                if current_arc_line is not None and feature_id == current_arc_line:
                    # Draw an arrow pointing to the current arc line
                    x_mean = (x_start + x_stop) / 2
                    ymin, ymax = ax.get_ylim()
                    ax.annotate(
                        "",
                        xy=(x_mean, ymax - 4),
                        xytext=(x_mean, ymax),
                        arrowprops=dict(arrowstyle="->", color="red", lw=1),
                        annotation_clip=False,
                    )

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    result = dict(img=img, cax=cax, cbar=cbar, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, vmin=vmin, vmax=vmax)
    return result


def find_arc_lines(
    image,
    input_poly,
    voffset=[0.0],
    fit_method="gaussian+moments",
    xlim=None,
    ylim=None,
    vlim=None,
    slicenum_list=[0],
    sliceid_list=[0],
    denoise_patch_size=3,
    denoise_patch_distance=2,
    times_sigma_threshold=10.0,
    minheight_percentage=80.0,
    maxwidth_pixels=None,
    nwinwidth_search_peaks=5,
    nwinwidth_refine_peaks=3,
    fwhm_diffraction=2.0,
    show_sliceid=False,
    allplots=False,
):
    """Find and characterize arc lines in FRIDA IFU exposures

    The function processes each slice of the FRIDA IFU image, applies background subtraction,
    denoising, and detects arc lines based on a thresholding method. It also provides options
    for plotting the results and overplotting slice boundaries.

    Parameters
    ----------
    image : str
        Path to the input FITS image file.
    input_poly : str
        Path to the input FITS file containing the slice boundary polynomials.
    voffset : list of float, optional
        Vertical offsets to apply to the polynomial boundaries (pixels).
        If more than one value is provided, a polynomial of degree len(voffset)-1
        will be fitted to the offsets and applied to the displayed
        slice boundary polynomials. A positive value shifts the
        displayed polynomials upwards (i.e., towards higher array indices
        along NAXIS2).
    fit_method : str, optional
        The method employed to fit the arc line profile (default: gaussian+moments).
        Valid options are "gaussian+moments" and "skewnorm".
    xlim : list, optional
        The x-axis limits for the plot.
    ylim : list, optional
        The y-axis limits for the plot.
    vlim : list, optional
        The vertical axis limits for the plot.
    slicenum_list : list of int, optional
        List of slice numbers to process (1-based). If [0], all slices
        will be processed. This parameter is mutually exclusive with `sliceid_list`.
    sliceid_list : list of int, optional
        List of slice IDs to process (1-based). If [0], all slices
        will be processed. This parameter is mutually exclusive with `slicenum_list`.
    denoise_patch_size : int, optional
        Patch size for denoising (default: 3).
    denoise_patch_distance : int, optional
        Patch distance for denoising (default: 2).
    times_sigma_threshold : float, optional
        The number of standard deviations above which a pixel is considered an arc line.
    minheight_percentage : float, optional
        The minimum height of the detected arc lines as a percentage
        of the maximum height of the slice at the line location (default: 80.0).
    maxwidth_pixels : float, optional
        The maximum width of the detected arc lines in pixels.
    nwinwidth_search_peaks : int, optional
        The window width (in pixels) for searching peaks in the arc line characterization.
    nwinwidth_refine_peaks : int, optional
        The window width (in pixels) for refining peaks in the arc line characterization.
        It must be less than or equal to `nwinwidth_search_peaks`.
    fwhm_diffraction : float, optional
        The diffraction-limited FWHM in pixels (default: 2.0). Since the
        Strehl ratio scales linearly with fwhm_diffraction, a value computed
        with an assumed FWHM can be corrected afterwards, without refitting,
        by multiplying it by fwhm_diffraction_real / fwhm_diffraction.
    show_sliceid : bool, optional
        If True, overplot the slice ID at the center of each slice.
    allplots : bool, optional
        If True, all plots will be displayed.

    Returns
    -------
    list_detection_tables : list of astropy.table.Table
        List of tables containing the arc line detections for each slice.
        If a slice is not processed, the corresponding table will be None.
        If the boundary polynomials for a slice are not valid, the corresponding
        table will also be None.
        If a slice does not have a minimum number of rows within the useful
        detector region, the corresponding table will also be None.
        If a slice has no detections, the corresponding table will not be None,
        but will be empty (i.e., it will have zero rows).
    """
    logger = logging.getLogger(__name__)

    # Check some relevant input parameters
    if nwinwidth_search_peaks < 3:
        raise ValueError(f"{nwinwidth_search_peaks=} must be at least 3")
    if nwinwidth_search_peaks % 2 == 0:
        raise ValueError(f"{nwinwidth_search_peaks=} must be odd")
    if nwinwidth_refine_peaks < 3:
        raise ValueError(f"{nwinwidth_refine_peaks=} must be at least 3")
    if nwinwidth_refine_peaks % 2 == 0:
        raise ValueError(f"{nwinwidth_refine_peaks=} must be odd")
    if nwinwidth_refine_peaks > nwinwidth_search_peaks:
        raise ValueError(f"{nwinwidth_refine_peaks=} must be less than or equal to {nwinwidth_search_peaks=}")
    if fit_method not in ["gaussian+moments", "skewnorm"]:
        raise ValueError(f"Invalid {fit_method=}. Must be 'gaussian+moments' or 'skewnorm'.")
    if fwhm_diffraction <= 0:
        raise ValueError(f"{fwhm_diffraction=} must be positive")

    # Retrieve initial data
    image_data, list_poly_left, list_poly_right, xlim = retrieve_initial_data(image, input_poly, xlim, voffset)

    # Avoid incompatible choices of slicenum and sliceid
    if slicenum_list != [0] and sliceid_list != [0]:
        raise ValueError("Cannot specify both slicenum_list and sliceid_list. Please choose one.")
    if slicenum_list != [0] and 0 in slicenum_list:
        raise ValueError("Invalid value in slicenum_list: 0. Use 0 only to indicate all slices.")
    if sliceid_list != [0] and 0 in sliceid_list:
        raise ValueError("Invalid value in sliceid_list: 0. Use 0 only to indicate all slices.")
    if sliceid_list == [0]:
        slicenum_to_process = slicenum_list
    else:
        slicenum_to_process = [int(sliceindex_from_sliceid(i) + 1) for i in sliceid_list]
    if slicenum_to_process == [0]:
        slicenum_to_process = list(range(1, FRIDA_NSLICES + 1))  # All slices
    logger.debug(f"Slices to process: #{slicenum_to_process}")

    # Check that the provided slice numbers are valid
    for slicenum in slicenum_to_process:
        if slicenum < 1 or slicenum > FRIDA_NSLICES:
            raise ValueError(f"Invalid slice #{slicenum}. Must be between 1 and {FRIDA_NSLICES}.")

    # Initialize list to store the table with the arc line detections for each slice
    list_detection_tables = [None] * FRIDA_NSLICES

    # Main loop to detect arc lines in each slice
    for islice in range(FRIDA_NSLICES):
        slicenum = islice + 1
        sliceid = sliceid_from_sliceindex(islice)
        if slicenum not in slicenum_to_process:
            logger.debug("Skipping slice #%2d (Slice ID: %2d)", slicenum, sliceid)
            continue  # Skip this slice if it is not in the list of slices to process

        logger.info("Processing slice #%2d / %2d (Slice ID: %2d)", slicenum, FRIDA_NSLICES, sliceid)

        # Check if the polynomial boundaries for the current slice are valid
        if list_poly_left[islice] is None or list_poly_right[islice] is None:
            logger.warning(
                "Slice #%2d (Slice ID: %2d) has no valid polynomial boundaries. Skipping this slice.",
                slicenum,
                sliceid,
            )
            continue  # Skip this slice if the polynomial boundaries are not valid

        # Create a copy of the image data to work on the current slice
        image_slice = image_data.copy()

        # Set to NaN the pixels outside the useful detector region
        image_slice = set_hawaii_border_pixels_to_nan(image_slice, inplace=True)

        # Set to NaN the pixels outside the slice boundaries
        for j in range(FRIDA_NAXIS1_HAWAII.value):
            yleft = int(list_poly_left[islice](j) + 0.5)
            if yleft >= FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value - 1:
                image_slice[:yleft, j] = np.nan
            yright = int(list_poly_right[islice](j) + 0.5)
            if yright <= FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value - 1:
                image_slice[yright + 1 :, j] = np.nan

        # Check that a minimum number of rows are within the useful detector region
        minimum_number_of_rows_within_useful_region = 20
        num_rows_within_useful_region = np.sum(np.isfinite(image_slice).any(axis=1))
        if num_rows_within_useful_region < minimum_number_of_rows_within_useful_region:
            logger.warning(
                "Slice #%2d (Slice ID: %2d) has pixels outside the useful detector region.\n"
                "Only %d rows are within the useful detector region (minimum required: %d).",
                slicenum,
                sliceid,
                num_rows_within_useful_region,
                minimum_number_of_rows_within_useful_region,
            )
            continue  # Skip this slice if it has less than the required minimum number of valid rows

        # Initialize table to store the arc line detections for each slice
        table_names = ["feature_id", "row_index", "x_peak"]
        table_dtype = ["i2", "i2", "f4"]
        if fit_method == "gaussian+moments":
            table_names += ["amplitude", "mean", "stddev", "skewness", "fwhm_np", "strehl"]
            table_dtype += ["f4", "f4", "f4", "f4", "f4", "f4"]
        detection_table = Table(names=table_names, dtype=table_dtype)
        # Units, display formats and provenance of each slice table; these are
        # written to the FITS header (TUNITn, TDISPn and header keywords) by
        # fits.table_to_hdu. Indices are 0-based array indices.
        for colname in ["x_peak", "mean", "stddev", "fwhm_np"]:
            if colname in detection_table.colnames:
                detection_table[colname].unit = u.pix
                detection_table[colname].format = "9.3f"
        for colname in ["amplitude", "skewness", "strehl"]:
            if colname in detection_table.colnames:
                detection_table[colname].format = "10.4f"
        detection_table.meta["SLICENUM"] = (slicenum, "Slice number (1-based)")
        detection_table.meta["SLICEID"] = (sliceid, "Slice ID")

        # automatic limits for the y-axis (NAXIS2) based on the polynomial boundaries of the slices
        xdum = np.arange(FRIDA_NAXIS1_HAWAII.value)
        extra_yroom = 2  # pixels
        ymin = max(-0.5, np.min(list_poly_left[islice](xdum)) - extra_yroom)
        ymax = min(FRIDA_NAXIS2_HAWAII.value - 0.5, np.max(list_poly_right[islice](xdum)) + extra_yroom)
        ylim = [ymin, ymax]
        if ylim[0] < -0.5:
            ylim[0] = -0.5
        if ylim[1] > FRIDA_NAXIS2_HAWAII.value - 0.5:
            ylim[1] = FRIDA_NAXIS2_HAWAII.value - 0.5
        logger.debug("Automatic y-axis limits for slice #%2d: [%.2f, %.2f]", slicenum, ylim[0], ylim[1])

        # Background subtraction using median filtering along the NAXIS1 axis
        image_slice_bkgsub = image_slice.copy()
        for i in range(int(ylim[0] + 0.5), int(ylim[1] + 0.5) + 1):
            half_kernel_size = 100  # pixels
            padded_slice = np.pad(image_slice[i, :], half_kernel_size, mode="constant", constant_values=np.nan)
            if not np.all(np.isnan(padded_slice)):
                # windows containing only NaN values return NaN (expected)
                warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                spbackground = np.nanmedian(sliding_window_view(padded_slice, 2 * half_kernel_size + 1), axis=1)
                image_slice_bkgsub[i, :] -= spbackground
        image_slice_bkg = image_slice - image_slice_bkgsub

        # Rescale the background-subtracted slice
        image_slice_rescaled, coef_rs = rescale_array_to_z1z2(image_slice_bkgsub, z1z2=(-1, 1))
        mask_nan_slice = np.isnan(image_slice_rescaled)
        image_slice_rescaled[mask_nan_slice] = 0.0  # Set NaN values to zero for denoising

        # Denoise the rescaled background-subtracted slice
        image_slice_denoised = restoration.denoise_nl_means(
            image_slice_rescaled,
            patch_size=denoise_patch_size,
            patch_distance=denoise_patch_distance,
            channel_axis=None,
        )
        image_slice_denoised = rescale_array_from_z1z2(image_slice_denoised, coef_rs)
        image_slice_denoised[mask_nan_slice] = np.nan  # Restore NaN values in the denoised slice

        # Detect arc lines in the denoised background-subtracted slice using a simple thresholding method
        q25, q50, q75 = np.percentile(
            image_slice_denoised.flatten()[~np.isnan(image_slice_denoised.flatten())], q=[25.0, 50.0, 75.0]
        )
        sigmag = 0.7413 * (q75 - q25)  # robust standard deviation
        threshold = q50 + times_sigma_threshold * sigmag  # threshold for arc line detection
        image_slice_denoised[np.isnan(image_slice_denoised)] = 0.0  # Set NaN values to zero for labeling
        # Binary mask and initial labelling (labels follow row-major scan order)
        mask_to_find_arc_lines = image_slice_denoised > threshold
        labels_arc_lines, num_features = ndimage.label(mask_to_find_arc_lines)
        # X coordinate (column index) of the centroid of each feature.
        # Using the boolean mask gives unweighted (geometric) centroids;
        # pass image_slice_denoised instead to get flux-weighted centroids.
        centroids = ndimage.center_of_mass(
            mask_to_find_arc_lines, labels_arc_lines, index=np.arange(1, num_features + 1)
        )
        xcentroids = np.array([c[1] for c in centroids])
        # order[k] is the (0-based) old label of the k-th feature from the left
        order = np.argsort(xcentroids, kind="stable")
        # Lookup table: old label -> new label (0 = background is preserved)
        lut = np.zeros(num_features + 1, dtype=labels_arc_lines.dtype)
        lut[order + 1] = np.arange(1, num_features + 1)
        # Relabelled image: label 1 is now the leftmost feature
        labels_arc_lines = lut[labels_arc_lines]

        if num_features > 0:
            # Select arc lines by imposing a condition based on the dimensions
            # of the detected features
            candidate_arc_line_slices = ndimage.find_objects(labels_arc_lines)
            arc_line_table = Table(
                names=[
                    "feature",
                    "x_start",
                    "x_stop",
                    "x_pos",
                    "y_start",
                    "y_stop",
                    "height",
                    "minheight",
                    "width",
                    "valid",
                ],
                dtype=["i4", "i4", "i4", "f8", "i4", "i4", "f8", "f8", "f8", "bool"],
            )
            for colname in ["x_pos", "height", "width", "minheight"]:
                arc_line_table[colname].format = ".1f"
            for i in range(num_features):
                if candidate_arc_line_slices[i] is not None:
                    average_x = (candidate_arc_line_slices[i][1].start + candidate_arc_line_slices[i][1].stop) / 2
                    maxheight = list_poly_right[islice](average_x) - list_poly_left[islice](average_x)
                    slice_y, slice_x = candidate_arc_line_slices[i]
                    height = slice_y.stop - slice_y.start
                    width = slice_x.stop - slice_x.start
                    minheight = minheight_percentage / 100.0 * maxheight
                    if maxwidth_pixels is None:
                        valid = height >= minheight
                    else:
                        valid = height >= minheight and width <= maxwidth_pixels
                    arc_line_table.add_row(
                        [
                            i + 1,
                            slice_x.start,
                            slice_x.stop,
                            average_x,
                            slice_y.start,
                            slice_y.stop,
                            height,
                            minheight,
                            width,
                            valid,
                        ]
                    )
                else:
                    arc_line_table.add_row([i + 1, -1, -1, np.nan, -1, -1, np.nan, np.nan, np.nan, False])
            num_valid_arc_lines = np.sum(arc_line_table["valid"])
        else:
            logger.warning(
                "Slice #%2d (Slice ID: %2d) has no detected arc lines. Skipping this slice.",
                slicenum,
                sliceid,
            )
            arc_line_table = None
            num_valid_arc_lines = 0

        # Show the plots if requested
        if allplots:
            # Show relevant information in the console
            logger.info("Threshold for arc line detection: %.3f", threshold)
            if arc_line_table is not None:
                logger.info("Arc line table:\n%s", "\n".join(arc_line_table.pformat(max_lines=-1)))
            logger.info("Number of arc lines detected.......: %d", num_features)
            logger.info("Number of arc lines after selection: %d", num_valid_arc_lines)

            fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), sharex=True, sharey=True)
            ax1, ax2, ax3, ax4, ax5, ax6 = axarr.flatten()

            # Display the original slice
            title = f"Image: {Path(image).name}\n"
            subtitle = f"[Slice #{slicenum} / {FRIDA_NSLICES}, Slice ID: {sliceid}]"
            title += subtitle
            imshow_slice(ax1, image_slice, xlim=xlim, ylim=ylim, vlim=vlim, title=title)
            plot_fitted_boundary_polynomials(ax1, list_poly_left, list_poly_right, show_sliceid, isliceplot=islice)

            # Display the background
            title = "Background slice\n"
            title += subtitle
            imshow_slice(ax2, image_slice_bkg, xlim=xlim, ylim=ylim, vlim=vlim, title=title)
            plot_fitted_boundary_polynomials(ax2, list_poly_left, list_poly_right, show_sliceid, isliceplot=islice)

            # Display the background-subtracted slice
            title = "Background-subtracted slice\n"
            title += subtitle
            imshow3 = imshow_slice(ax3, image_slice_bkgsub, xlim=xlim, ylim=ylim, vlim=vlim, title=title)
            plot_fitted_boundary_polynomials(ax3, list_poly_left, list_poly_right, show_sliceid, isliceplot=islice)

            # Display the denoised background-subtracted slice
            # (use the same vmin and vmax as for the background-subtracted slice)
            title = "Denoised background-subtracted slice\n"
            title += subtitle
            vmin, vmax = (imshow3["vmin"], imshow3["vmax"])
            imshow_slice(ax4, image_slice_denoised, xlim=xlim, ylim=ylim, vlim=(vmin, vmax), title=title)
            plot_fitted_boundary_polynomials(ax4, list_poly_left, list_poly_right, show_sliceid, isliceplot=islice)

            # Display the detected arc lines
            if num_features > 0:
                title = f"Detected arc lines ({num_features})\n"
                title += subtitle
                cmap = plt.cm.rainbow.copy()
                cmap.set_under("white")
                imshow_slice(
                    ax5,
                    labels_arc_lines,
                    xlim=xlim,
                    ylim=ylim,
                    vlim=(1, num_features),
                    title=title,
                    cmap=cmap,
                    cblabel="Object label",
                )
                plot_fitted_boundary_polynomials(
                    ax5, list_poly_left, list_poly_right, show_sliceid=False, isliceplot=islice
                )

                # Display the valid arc lines after selection
                title = f"Valid arc lines after selection ({num_valid_arc_lines}/{num_features})\n"
                title += subtitle
                imshow_slice(
                    ax6,
                    labels_arc_lines,
                    arc_line_table=arc_line_table,
                    xlim=xlim,
                    ylim=ylim,
                    vlim=(1, num_features),
                    title=title,
                    cmap=cmap,
                    cblabel="Object label",
                )
                plot_fitted_boundary_polynomials(
                    ax6, list_poly_left, list_poly_right, show_sliceid=False, isliceplot=islice
                )
            else:
                # Do not display the last two plots if there are no detected arc lines
                ax5.axis("off")
                ax6.axis("off")

            # Adjust layout and display the plots
            fig.tight_layout()
            logger.info("Press 'q' to close this figure and continue...")
            plt.show()

        # Characterize the valid arc lines by fitting a peak to each row of the valid arc line
        if num_valid_arc_lines > 0:
            if allplots:
                next_feature_id_to_be_displayed = 1
            for k in range(num_features):
                if arc_line_table["valid"][k]:
                    slice_y, slice_x = candidate_arc_line_slices[k]
                    logger.debug(
                        "Slice #%2d (Slice ID: %2d) - Arc line %2d: x_pos=%.1f, height=%.1f, width=%.1f",
                        slicenum,
                        sliceid,
                        k,
                        arc_line_table["x_pos"][k],
                        arc_line_table["height"][k],
                        arc_line_table["width"][k],
                    )
                    logger.debug(f"slice_y: {slice_y}, slice_x: {slice_x}")
                    logger.debug(f"arc_line_table[{k}]:\n{arc_line_table[k]}")
                    # Select the pixels corresponding to the valid arc line
                    feature_id = arc_line_table["feature"][k]
                    logger.debug(f"feature_id: {feature_id}")
                    # Find the indices of the pixels that belong to the valid arc line
                    iok = np.argwhere(labels_arc_lines == feature_id)
                    if allplots and feature_id >= next_feature_id_to_be_displayed:
                        allplots_effective = True
                        last_figure_is_closed = False
                        logger.info(
                            f"Displaying plots for feature {feature_id} in slice #{slicenum} (Slice ID: {sliceid})"
                        )
                        plt.ion()
                        fig = plt.figure(figsize=(12, 8))
                        # disable some keys employed to control the loop over rows, to avoid confusion
                        # with the keys used in the interactive plots
                        disable_keys_in_figure(fig, keys=["p"])
                        gs = fig.add_gridspec(nrows=2, ncols=2, height_ratios=[1, 1])
                        ax1 = fig.add_subplot(gs[0, 0])  # row 0, column 0
                        ax2 = fig.add_subplot(gs[0, 1], sharex=ax1, sharey=ax1)  # row 0, column 1
                        ax3 = fig.add_subplot(gs[1, 0])  # row 1, column 0
                        ax4 = fig.add_subplot(gs[1, 1])  # row 1, column 1
                        title = "Background-subtracted slice\n"
                        title += subtitle
                        imshow_slice(ax1, image_slice_bkgsub, xlim=xlim, ylim=ylim, vlim=vlim, title=title)
                        plot_fitted_boundary_polynomials(
                            ax1, list_poly_left, list_poly_right, show_sliceid, isliceplot=islice
                        )
                        title = "Valid arc lines after selection\n"
                        title += subtitle
                        cmap = plt.cm.rainbow.copy()
                        cmap.set_under("white")
                        imshow_slice(
                            ax2,
                            labels_arc_lines,
                            arc_line_table=arc_line_table,
                            current_arc_line=feature_id,
                            xlim=xlim,
                            ylim=ylim,
                            vlim=(1, num_features),
                            title=title,
                            cmap=cmap,
                            cblabel="Object label",
                        )
                        plot_fitted_boundary_polynomials(
                            ax2, list_poly_left, list_poly_right, show_sliceid=False, isliceplot=islice
                        )
                        xmin, xmax = slice_x.start - 0.5, slice_x.stop - 0.5  # .stop is exclusive, so we subtract 0.5
                        ymin, ymax = slice_y.start - 0.5, slice_y.stop - 0.5  # .stop is exclusive, so we subtract 0.5
                        vmin, vmax = ZScaleInterval().get_limits(image_slice_bkgsub[slice_y, slice_x])
                        imshow_slice(
                            ax3,
                            image_slice_bkgsub,
                            xlim=(xmin, xmax),
                            ylim=(ymin, ymax),
                            vlim=(vmin, vmax),
                            title=f"feature {feature_id}",
                        )

                        pause_between_rows = True
                        last_event_key = None

                        def on_key(event):
                            nonlocal allplots_effective
                            nonlocal next_feature_id_to_be_displayed
                            nonlocal pause_between_rows
                            nonlocal last_event_key
                            if event.key == "x":
                                logger.info("Exiting the program by user request")
                                sys.exit()
                            elif event.key == "c":
                                pause_between_rows = False
                            elif event.key == "p":
                                pause_between_rows = True
                            elif event.key == "n":
                                pass
                            elif event.key == "q":
                                pause_between_rows = False
                                allplots_effective = False
                            elif event.key == "j":
                                if feature_id < num_features:
                                    next_feature_id_to_be_displayed = input_number(
                                        expected_type="int",
                                        prompt="Next feature ID to be displayed "
                                        f"(from {feature_id + 1} to {num_features}; {num_features + 1}: skip all)?",
                                        default=feature_id + 1,
                                        min_val=feature_id + 1,
                                        max_val=num_features + 1,
                                    )
                                else:
                                    next_feature_id_to_be_displayed = num_features + 1
                                pause_between_rows = False
                                allplots_effective = False
                            if event.key in ["q", "j"]:
                                logger.info("Working (please wait)...")
                            if event.key in ["c", "n", "q", "j"]:
                                # resume the loop over rows
                                event.canvas.stop_event_loop()
                            if event.key in ["c", "n", "q"]:
                                next_feature_id_to_be_displayed = feature_id + 1
                            last_event_key = event.key

                        def on_close(event):
                            nonlocal allplots_effective
                            nonlocal last_figure_is_closed
                            nonlocal pause_between_rows
                            allplots_effective = False
                            last_figure_is_closed = True
                            pause_between_rows = False
                            # avoid blocking the loop forever if the figure is closed
                            event.canvas.stop_event_loop()

                        fig.canvas.mpl_connect("key_press_event", on_key)
                        fig.canvas.mpl_connect("close_event", on_close)
                    else:
                        ax1 = ax2 = ax3 = ax4 = None
                        allplots_effective = False
                        last_figure_is_closed = True
                        pause_between_rows = False
                        last_event_key = None
                    for j in range(slice_y.start, slice_y.stop):
                        if j == slice_y.start:
                            if allplots_effective:
                                # Display the instructions only once, at the beginning of the loop over rows
                                logger.info("Press any of the following keys to continue:")
                                logger.info(f"'n': fit arc line in next row of feature {feature_id}")
                                logger.info(
                                    f"'c': continue fitting the remaining rows of feature {feature_id}"
                                    " without pausing"
                                )
                                logger.info("     ('p' to pause again)")
                                logger.info(
                                    f"'q': continue fitting the remaining rows of feature {feature_id}"
                                    " without plotting"
                                )
                                logger.info("'j': jump to feature <num>")
                                logger.info("'x': exit the program")
                        # Diplay result of full feature
                        if j == (slice_y.stop - 1) and last_event_key == "c":
                            pause_between_rows = True
                        ixfit = iok[iok[:, 0] == j, 1]  # x indices of valid arc line pixels in row j
                        mean_ixfit = np.mean(ixfit) if len(ixfit) > 0 else np.nan
                        if allplots_effective:
                            ax3.plot(ixfit, [j] * len(ixfit), "C0o")
                        logger.debug(f"Row {j}: x indices of valid arc line pixels: {ixfit}")
                        if len(ixfit) == 0:
                            logger.warning(
                                "Row %d of feature %d has no valid arc line pixels. Skipping this row.",
                                j,
                                feature_id,
                            )
                            continue
                        # Use as reference image the background-subtracted slice, which is the one
                        # used for peak fitting (other options could be the denoised image or the original image)
                        image_slice_reference = image_slice_bkgsub
                        # Use the contiguous x range spanned by the arc line pixels in row j
                        # (ixfit may have gaps), so that indices in spectrum map to x = x0 + index
                        x0 = ixfit.min()
                        spectrum = image_slice_reference[j, x0 : ixfit.max() + 1]
                        logger.debug(f"{spectrum=}")
                        ixpeaks = []
                        min_len_spectrum = 5 if fit_method == "skewnorm" else nwinwidth_refine_peaks
                        if len(spectrum) >= min_len_spectrum:
                            loop_search_ixpeaks = True
                            nwinwidth = nwinwidth_search_peaks
                            while loop_search_ixpeaks:
                                if len(spectrum) >= nwinwidth:  # require at least nwinwidth pixels to fit a peak
                                    ixpeaks = find_peaks_spectrum(spectrum, nwinwidth=nwinwidth)
                                    if len(ixpeaks) == 0:  # if no peaks were found, try with a smaller window
                                        nwinwidth -= 2
                                    else:
                                        loop_search_ixpeaks = False
                                else:
                                    nwinwidth -= 2
                                if (
                                    nwinwidth < nwinwidth_refine_peaks
                                ):  # if the window width is too small, stop searching
                                    loop_search_ixpeaks = False
                        else:
                            logger.debug(
                                f"Number of points for fit: {len(spectrum)} < {min_len_spectrum}. "
                                f"Skipping row {j} of feature {feature_id}."
                            )
                        if len(ixpeaks) <= 0:
                            msg = f"Row {j} of feature {feature_id} has no valid peak. Skipping this row."
                            if allplots_effective:
                                logger.info(msg)
                                if pause_between_rows:
                                    # It is necessary to insert a pause to allow the event loop to start before
                                    # the execution of the second iteration of the row loop
                                    ax4.cla()  # clear the axes for the refined peak plot
                                    ax4.axis("off")
                                    fig.tight_layout()
                                    plt.pause(0.2)
                                    fig.canvas.start_event_loop(timeout=0)
                            else:
                                logger.debug(msg)
                        else:
                            # Determine the highest peak
                            imax = np.argmax(spectrum[ixpeaks])
                            ixpeaks = ixpeaks[imax : imax + 1]  # keep only the highest peak
                            ax_external = None
                            if allplots_effective:
                                if ax4 is not None:
                                    ax4.cla()  # clear the axes for the refined peak plot
                                    ax_external = ax4
                            if fit_method == "gaussian+moments":
                                fwhm_diffraction_effective = fwhm_diffraction
                            else:
                                fwhm_diffraction_effective = None
                            xpeak, _, dict_fit_result = refine_peaks_spectrum(
                                spectrum,
                                ixpeaks,
                                nwinwidth=nwinwidth_refine_peaks,
                                method=fit_method,
                                title=f"mean_x_data {mean_ixfit:.1f}, feature id {feature_id}, row index {j}",
                                ax_external=ax_external,
                                return_dict_result=True,
                                preserve_flux=True,
                                fwhm_diffraction=fwhm_diffraction_effective,
                            )
                            if ax_external is not None:
                                ax_external.legend(loc="upper left", fontsize=8, title=fit_method)
                                fig.canvas.draw_idle()
                                fig.tight_layout()
                                plt.pause(0.1)  # pause to allow the plot to update
                            # In debug mode, show the fit result in the console
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug("\n".join([f"{k}={v!r}" for k, v in dict_fit_result.items()]))
                            # Store the results in the detection table
                            xpeak = xpeak[0]  # convert to scalar
                            xpeak += x0  # convert to x indices in the original image
                            if fit_method == "gaussian+moments":
                                if dict_fit_result["fit_ok"]:
                                    table_row = [
                                        feature_id,
                                        j,
                                        xpeak,
                                        dict_fit_result["gaussian_amp"],  # amplitude
                                        dict_fit_result["gaussian_mean"] + x0,  # mean
                                        dict_fit_result["gaussian_sigma"],  # stddev
                                        dict_fit_result["moment_skewness"],  # skewness
                                        dict_fit_result["fwhm_np"],  # FWHM from non-parametric method
                                        dict_fit_result["strehl"],  # Strehl ratio
                                    ]
                                else:
                                    table_row = [feature_id, j, xpeak] + [np.nan] * 6
                            else:
                                table_row = [feature_id, j, xpeak]
                            detection_table.add_row(table_row)
                            if allplots_effective:
                                ixpeak = ixpeaks + x0  # convert to x index in the original image
                                ax3.plot(ixpeak, j, "C1o")
                                ax3.plot(xpeak, j, "C3o")
                                # Wait until the user takes a decision
                                if pause_between_rows:
                                    # It is necessary to insert a pause to allow the event loop to start before
                                    # the execution of the second iteration of the row loop
                                    plt.pause(0.2)
                                    fig.canvas.start_event_loop(timeout=0)
                    if not last_figure_is_closed:
                        plt.ioff()
                        plt.close(fig)
                        last_figure_is_closed = True

        list_detection_tables[islice] = detection_table

    return list_detection_tables


def main(args=None):

    parser = argparse.ArgumentParser(
        description="Find and characterize arc lines in FRIDA IFU images",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("--image", help="Image to display boundaries on", type=str, required=True)
    parser.add_argument(
        "--poly", help="Path to the file with the boundary polynomials (mandatory)", type=str, required=True
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
        "--fit-method",
        help="Method employed to fit the arc line profile (default: 'gaussian+moments')",
        type=str,
        choices=["gaussian+moments", "skewnorm"],
        default="gaussian+moments",
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
    parser.add_argument(
        "--denoise-patch-size",
        help="Patch size for denoising (default: 3)",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--denoise-patch-distance",
        help="Patch distance for denoising (default: 2)",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--times-sigma-threshold",
        help="Threshold for arc line detection in units of the robust standard deviation (default: 5.0)",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--minheight-percentage",
        help="Minimum height of the detected arc lines as a percentage of the maximum height of the slice"
        " at the line location (default: 80.0)",
        type=float,
        default=80.0,
    )
    parser.add_argument(
        "--maxwidth-pixels",
        help="Maximum width of the detected arc lines in pixels (default: 20.0)",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--nwinwidth-search-peaks",
        help="Window width for searching peaks in the spectrum (default: 5; minimum: 3; odd number)",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--nwinwidth-refine-peaks",
        help="Window width for refining peaks in the spectrum (default: 3; minimum: 3; odd number). "
        "It must be smaller or equal to nwinwidth-search-peaks.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--fwhm-diffraction",
        help="Diffraction-limited FWHM in pixels (default: 2.0).",
        type=float,
        default=2.0,
    )
    parser.add_argument("--show-sliceid", help="Overplot slice ID", action="store_true")
    parser.add_argument("--allplots", help="Display all plots", action="store_true")
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
    parser.add_argument(
        "--output",
        help="Output FITS file name to save the arc line detection tables (default: None)",
        type=str,
        default=None,
    )

    # Include default arguments for common actions, and initialize console and logging
    myscript = NuminaScriptDefinition(parser, argv=args)
    args = myscript.args
    logger = myscript.logger

    # Find and characterize arc lines in the image using the provided parameters
    logger.debug("Calling find_arc_lines script function")
    list_detection_tables = find_arc_lines(
        image=args.image,
        input_poly=args.poly,
        voffset=args.voffset,
        fit_method=args.fit_method,
        xlim=args.xlim,
        ylim=args.ylim,
        vlim=args.vlim,
        slicenum_list=args.slicenum_list,
        sliceid_list=args.sliceid_list,
        denoise_patch_size=args.denoise_patch_size,
        denoise_patch_distance=args.denoise_patch_distance,
        times_sigma_threshold=args.times_sigma_threshold,
        minheight_percentage=args.minheight_percentage,
        maxwidth_pixels=args.maxwidth_pixels,
        nwinwidth_search_peaks=args.nwinwidth_search_peaks,
        nwinwidth_refine_peaks=args.nwinwidth_refine_peaks,
        fwhm_diffraction=args.fwhm_diffraction,
        show_sliceid=args.show_sliceid,
        allplots=args.allplots,
    )

    # Save the arc line detection tables to a FITS file if an output filename is provided
    if args.output is not None:
        logger.info("Saving arc line detection tables to FITS file: %s", args.output)
        # Check that at least one detection table is not None and has at least one row
        if not any(
            detection_table is not None and len(detection_table) > 0 for detection_table in list_detection_tables
        ):
            logger.warning("No valid arc line detection tables to save. The output FITS file will not be created.")
        else:
            # Primary HDU with the provenance of the tables
            primary_hdu = fits.PrimaryHDU()
            hdr = primary_hdu.header
            hdr["IMAGE"] = (Path(args.image).name, "Input arc image")
            hdr["POLYFILE"] = (Path(args.poly).name, "Slice bound. pol.")
            hdr["FITMETH"] = (args.fit_method, "Line profile fit method")
            hdr["FWHMDIFF"] = (args.fwhm_diffraction, "[pix] Assumed diffraction-limited FWHM")
            hdr["NWSEARCH"] = (args.nwinwidth_search_peaks, "Window width to search peaks")
            hdr["NWREFINE"] = (args.nwinwidth_refine_peaks, "Window width to refine peaks")
            hdr["TSIGMA"] = (args.times_sigma_threshold, "Detection threshold (robust sigma)")
            hdr["MINHPCT"] = (args.minheight_percentage, "Minimum line height (percent)")
            hdr["MAXWPIX"] = (args.maxwidth_pixels, "[pix] Maximum line width")
            hdr["UUID"] = (str(uuid.uuid4()), "Unique identifier for this run")
            hdr["COMMENT"] = "row_index, x_peak and mean are 0-based array indices"
            hdu_list = fits.HDUList([primary_hdu])
            # Descriptions of the columns, stored as comments of the TTYPEn keywords
            # (the column description attribute is not written by fits.table_to_hdu)
            column_comments = {
                "feature_id": "Arc line label (1 = leftmost)",
                "row_index": "Row (0-based index along NAXIS2)",
                "x_peak": "Refined peak (0-based index along NAXIS1)",
                "mean": "Gaussian mean (0-based index along NAXIS1)",
                "stddev": "Gaussian sigma",
                "skewness": "Moment skewness",
                "fwhm_np": "Non-parametric FWHM",
                "strehl": "Strehl ratio",
            }
            # One extension per processed slice, named "SLCNUMXX" where XX is the slice
            # number (1-based); processed slices without detections are also written
            # (NAXIS2 = 0), so they can be distinguished from slices not processed
            for islice, detection_table in enumerate(list_detection_tables):
                if detection_table is not None:
                    hdu = fits.table_to_hdu(detection_table)
                    hdu.name = f"SLCNUM{islice + 1:02d}"
                    for i, colname in enumerate(detection_table.colnames, start=1):
                        if colname in column_comments:
                            hdu.header.comments[f"TTYPE{i}"] = column_comments[colname]
                    hdu_list.append(hdu)
            hdu_list.writeto(args.output, overwrite=True)
            logger.info("Arc line detection tables saved to FITS file: %s", args.output)

    # Display goodbye message and save console if recording is enabled
    myscript.goodbye_message_and_save_console()


if __name__ == "__main__":
    main()
