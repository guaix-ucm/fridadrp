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
from astropy.visualization import ZScaleInterval
import logging
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from rich_argparse import RichHelpFormatter
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage import restoration
import teareduce as tea
import warnings

from numina.array.display.polfit_residuals import polfit_residuals
from numina.array.wavecalib.peaks_spectrum import find_peaks_spectrum, refine_peaks_spectrum
from numina.array.rescale_array_z1z2 import rescale_array_to_z1z2, rescale_array_from_z1z2
from numina.tools.initialize_script_with_args import NuminaScriptDefinition

from fridadrp.core import FRIDA_NAXIS1_HAWAII, FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL
from fridadrp.core import sliceid_from_sliceindex
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
            logger.info(f"Fitting a polynomial of degree {len(voffset)-1} to the vertical offsets")
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


def fill_voronoi_kdtree(array, pixel_scale=(1.0, 1.0), max_distance=np.inf):
    """
    Fill NaN pixels with the value of the nearest valid pixel using a
    k-d tree. Pixels farther than `max_distance` from any valid pixel
    are left as NaN.

    Parameters
    ----------
    array : ndarray, shape (ny, nx)
        2D array with NaN at unknown pixels.
    pixel_scale : tuple of float, optional
        Pixel size along (y, x), used to scale pixel indices to
        physical coordinates.
    max_distance : float, optional
        Maximum distance (in the units of `pixel_scale`) to a valid
        pixel for a NaN pixel to be filled.

    Returns
    -------
    filled : ndarray, shape (ny, nx)
        Copy of `array` with NaN replaced where a valid pixel lies
        within `max_distance`.
    """
    array = np.asarray(array, dtype=float)
    valid = ~np.isnan(array)

    if not valid.any():
        raise ValueError("The array contains no valid (non-NaN) pixels.")

    # Coordinates (in physical units) of all pixels of the grid
    yy, xx = np.indices(array.shape)
    coords = np.column_stack([yy.ravel() * pixel_scale[0], xx.ravel() * pixel_scale[1]])

    # Tree built only on the valid pixels (the Voronoi seeds)
    seeds = coords[valid.ravel()]
    tree = cKDTree(seeds)

    # Nearest seed for every pixel; dist = inf if beyond max_distance
    dist, idx = tree.query(coords, distance_upper_bound=max_distance)

    values = array[valid]  # seed values, same order as seeds
    filled = np.full(array.size, np.nan)
    found = np.isfinite(dist)  # idx == len(seeds) where not found
    filled[found] = values[idx[found]]

    return filled.reshape(array.shape)


def find_arc_lines(
    image,
    input_poly,
    xlim=None,
    ylim=None,
    vlim=None,
    voffset=[0.0],
    denoise_patch_size=3,
    denoise_patch_distance=2,
    times_sigma_threshold=10.0,
    minheight_percentage=80.0,
    maxwidth_pixels=10.0,
    sliceid=False,
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
    xlim : list, optional
        The x-axis limits for the plot.
    ylim : list, optional
        The y-axis limits for the plot.
    vlim : list, optional
        The vertical axis limits for the plot.
    voffset : list of float, optional
        Vertical offsets to apply to the polynomial boundaries (pixels).
        If more than one value is provided, a polynomial of degree len(voffset)-1
        will be fitted to the offsets and applied to the displayed
        slice boundary polynomials. A positive value shifts the
        displayed polynomials upwards (i.e., towards higher array indices
        along NAXIS2).
    denoise_patch_size : int, optional
        Patch size for denoising (default: 3).
    denoise_patch_distance : int, optional
        Patch distance for denoising (default: 2).
    times_sigma_threshold : float, optional
        The number of standard deviations above which a pixel is considered an arc line.
    minheight_percentage : float, optional
        The minimum height of the detected arc lines as a percentage
        of the maximum height of the slice at the line location (default: 80.0).
    sliceid : bool, optional
        If True, overplot the slice ID at the center of each slice.
    allplots : bool, optional
        If True, all plots will be displayed.

    """
    logger = logging.getLogger(__name__)

    # Retrieve initial data
    image_data, list_poly_left, list_poly_right, xlim = retrieve_initial_data(image, input_poly, xlim, voffset)

    image_data_denoised = np.full((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)

    # Initialize an array to store the FWHM, skewness and kurtosis excess values for each pixel in the image
    image_fwhm_np = np.full((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)
    image_skewness = np.full((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)
    image_kurtosis_excess = np.full((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)

    # Initialize dictionary to store the list of arc line detections
    dict_arc_lines_detections = {}

    # Main loop to detect arc lines in each slice
    for islice in range(FRIDA_NSLICES):
        slicenum = islice + 1
        sliceid = sliceid_from_sliceindex(islice)
        logger.info("Processing slice num %2d / %2d (Slice ID: %2d)", slicenum, FRIDA_NSLICES, sliceid)

        # Check if the polynomial boundaries for the current slice are valid
        if list_poly_left[islice] is None or list_poly_right[islice] is None:
            logger.warning(
                "Slice num %2d (Slice ID: %2d) has no valid polynomial boundaries. Skipping this slice.",
                slicenum,
                sliceid,
            )
            continue
        # Create a copy of the image data to work on the current slice
        image_slice = image_data.copy()

        # Set to NaN the pixels outside the slice boundaries
        slice_within_useful_detector_region = True
        for j in range(FRIDA_NAXIS1_HAWAII.value):
            yleft = int(list_poly_left[islice](j) + 0.5)
            if yleft >= FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value - 1:
                image_slice[:yleft, j] = np.nan
            else:
                slice_within_useful_detector_region = False
            yright = int(list_poly_right[islice](j) + 0.5)
            if yright <= FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value - 1:
                image_slice[yright + 1 :, j] = np.nan
            else:
                slice_within_useful_detector_region = False
        if not slice_within_useful_detector_region:
            logger.warning(
                "Slice num %2d (Slice ID: %2d) has pixels outside the useful detector region. "
                "Some pixels may be set to NaN.",
                slicenum,
                sliceid,
            )
            continue  # Skip this slice if it is outside the useful detector region

        # Define a mask for the pixels that are within the slice boundaries
        mask_slice = np.zeros_like(image_slice, dtype=bool)
        for j in range(FRIDA_NAXIS1_HAWAII.value):
            yleft = int(list_poly_left[islice](j) + 0.5)
            yright = int(list_poly_right[islice](j) + 0.5)
            if (
                yleft >= FRIDA_NAXIS2_HAWAII_FIRST_USEFUL_PIXEL.value - 1
                and yright <= FRIDA_NAXIS2_HAWAII_LAST_USEFUL_PIXEL.value - 1
            ):
                mask_slice[yleft : yright + 1, j] = True

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
        logger.debug("Automatic y-axis limits for slice num %2d: [%.2f, %.2f]", slicenum, ylim[0], ylim[1])

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

        image_data_denoised[mask_slice] = image_slice_denoised[mask_slice]  # Store the denoised slice in the full image

        # Detect arc lines in the denoised background-subtracted slice using a simple thresholding method
        q25, q50, q75 = np.percentile(
            image_slice_denoised.flatten()[~np.isnan(image_slice_denoised.flatten())], q=[25.0, 50.0, 75.0]
        )
        sigmag = 0.7413 * (q75 - q25)  # robust standard deviation
        threshold = q50 + times_sigma_threshold * sigmag  # threshold for arc line detection
        image_slice_denoised[np.isnan(image_slice_denoised)] = 0.0  # Set NaN values to zero for labeling
        labels_arc_lines, num_features = ndimage.label(image_slice_denoised > threshold)

        if num_features > 0:
            # Select arc lines by imposing a condition based on the dimensions
            # of the detected features
            candidate_arc_line_slices = ndimage.find_objects(labels_arc_lines)
            arc_line_table = Table(
                names=["feature", "x_pos", "height", "minheight", "width", "valid"],
                dtype=["i4", "f8", "f8", "f8", "f8", "bool"],
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
                    valid = height >= minheight and width <= maxwidth_pixels
                    arc_line_table.add_row([i, average_x, height, minheight, width, valid])
                else:
                    arc_line_table.add_row([i, np.nan, np.nan, np.nan, np.nan, False])
            num_valid_arc_lines = np.sum(arc_line_table["valid"])
            arc_line_table.sort("x_pos")  # sort the table by x_pos
            # sort the candidate_arc_line_slices according to the sorted table
            candidate_arc_line_slices = [candidate_arc_line_slices[i] for i in arc_line_table["feature"]]
        else:
            logger.warning(
                "Slice num %2d (Slice ID: %2d) has no detected arc lines. Skipping this slice.",
                slicenum,
                sliceid,
            )
            num_valid_arc_lines = 0

        # Show the plots if requested
        if allplots:
            # Show relevant information in the console
            logger.info("Threshold for arc line detection: %.3f", threshold)
            logger.info("Arc line table:\n%s", "\n".join(arc_line_table.pformat(max_lines=-1)))
            logger.info("Number of arc lines detected.......: %d", num_features)
            logger.info("Number of arc lines after selection: %d", num_valid_arc_lines)
            # Display slice
            if vlim is not None:
                vmin, vmax = vlim
            else:
                # Use ZScaleInterval to compute the display limits for the slice
                # ignoring the NaN values outside the slice boundaries
                vmin, vmax = ZScaleInterval().get_limits(image_slice.flatten()[~np.isnan(image_slice.flatten())])
            title = f"Image: {Path(image).name}\n"
            subtitle = f"[Slice number: {slicenum} / {FRIDA_NSLICES}, Slice ID: {sliceid}]"
            title += subtitle
            fig, axarr = plt.subplots(nrows=3, ncols=2, figsize=(15, 10), sharex=True, sharey=True)
            ax1, ax2, ax3, ax4, ax5, ax6 = axarr.flatten()
            tea.imshow(
                fig,
                ax1,
                image_slice,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                title=title,
            )
            ax1.set_xlim(xlim)
            ax1.set_ylim(ylim)
            plot_fitted_boundary_polynomials(ax1, list_poly_left, list_poly_right, sliceid, isliceplot=islice)

            # Display the background
            vmin, vmax = ZScaleInterval().get_limits(image_slice_bkg.flatten()[~np.isnan(image_slice_bkg.flatten())])
            title = "Background slice\n"
            title += subtitle
            tea.imshow(
                fig,
                ax2,
                image_slice_bkg,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                title=title,
            )
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
            plot_fitted_boundary_polynomials(ax2, list_poly_left, list_poly_right, sliceid, isliceplot=islice)

            # Display the background-subtracted slice
            if vlim is not None:
                vmin, vmax = vlim
            else:
                vmin, vmax = ZScaleInterval().get_limits(
                    image_slice_bkgsub.flatten()[~np.isnan(image_slice_bkgsub.flatten())]
                )
            title = "Background-subtracted slice\n"
            title += subtitle
            tea.imshow(
                fig,
                ax3,
                image_slice_bkgsub,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                title=title,
            )
            ax3.set_xlim(xlim)
            ax3.set_ylim(ylim)
            plot_fitted_boundary_polynomials(ax3, list_poly_left, list_poly_right, sliceid, isliceplot=islice)

            # Display the denoised background-subtracted slice
            # (use the same vmin and vmax as for the background-subtracted slice)
            title = "Denoised background-subtracted slice\n"
            title += subtitle
            tea.imshow(
                fig,
                ax4,
                image_slice_denoised,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
                title=title,
            )
            ax4.set_xlim(xlim)
            ax4.set_ylim(ylim)
            plot_fitted_boundary_polynomials(ax4, list_poly_left, list_poly_right, sliceid, isliceplot=islice)

            if num_features > 0:
                # Display the detected arc lines
                title = "Detected arc lines\n"
                title += subtitle
                tea.imshow(
                    fig,
                    ax5,
                    labels_arc_lines,
                    vmin=0,
                    vmax=num_features,
                    aspect="auto",
                    title=title,
                    cmap="nipy_spectral",
                    cblabel="Object label",
                )
                ax5.set_xlim(xlim)
                ax5.set_ylim(ylim)
                plot_fitted_boundary_polynomials(ax5, list_poly_left, list_poly_right, sliceid, isliceplot=islice)

                # Display the valid arc lines after selection
                title = "Valid arc lines after selection\n"
                title += subtitle
                tea.imshow(
                    fig,
                    ax6,
                    labels_arc_lines,
                    vmin=0,
                    vmax=num_features,
                    aspect="auto",
                    title=title,
                    cmap="nipy_spectral",
                    cblabel="Object label",
                )
                for k in range(num_features):
                    if arc_line_table["valid"][k]:
                        slice_y, slice_x = candidate_arc_line_slices[k]
                        rect = Rectangle(
                            (slice_x.start, slice_y.start),
                            slice_x.stop - slice_x.start,
                            slice_y.stop - slice_y.start,
                            edgecolor="white",
                            facecolor="none",
                            linewidth=1.5,
                        )  # plot a rectangle around the valid arc line
                        ax6.add_patch(rect)
                ax6.set_xlim(xlim)
                ax6.set_ylim(ylim)
                plot_fitted_boundary_polynomials(ax6, list_poly_left, list_poly_right, sliceid, isliceplot=islice)
            else:
                # Do not display the last two plots if there are no detected arc lines
                ax5.axis("off")
                ax6.axis("off")

            # Adjust layout and display the plots
            fig.tight_layout()
            plt.show()

        # initialize the list of arc line tables for the current slice
        dict_arc_lines_detections[islice] = []

        # Characterize the valid arc lines by fitting a peak to each row of the valid arc line
        slice_fwhm_np = np.full((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)
        slice_skewness = np.full((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)
        slice_kurtosis_excess = np.full((FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value), np.nan, dtype=float)
        if num_valid_arc_lines > 0:
            for k in range(num_features):
                if arc_line_table["valid"][k]:
                    slice_y, slice_x = candidate_arc_line_slices[k]
                    logger.debug(
                        "Slice num %2d (Slice ID: %2d) - Arc line %2d: x_pos=%.1f, height=%.1f, width=%.1f",
                        slicenum,
                        sliceid,
                        k,
                        arc_line_table["x_pos"][k],
                        arc_line_table["height"][k],
                        arc_line_table["width"][k],
                    )
                    logger.debug(f"slice_y: {slice_y}, slice_x: {slice_x}")
                    logger.debug(f"arc_line_table[k]: {arc_line_table[k]}")
                    # Select the pixels corresponding to the valid arc line
                    feature_id = arc_line_table["feature"][k]
                    logger.debug(f"feature_id: {feature_id}")
                    # Find the indices of the pixels that belong to the valid arc line
                    iok = np.argwhere(labels_arc_lines == feature_id + 1)  # +1 because labels start from 1
                    if allplots:
                        arc_line_box = image_slice_bkgsub[slice_y, slice_x]
                        vmin, vmax = ZScaleInterval().get_limits(
                            arc_line_box.flatten()[~np.isnan(arc_line_box.flatten())]
                        )
                        fig, ax = plt.subplots(figsize=(8, 6))
                        tea.imshow(
                            fig,
                            ax,
                            image_slice_bkgsub,
                            vmin=vmin,
                            vmax=vmax,
                            aspect="auto",
                            title=f"feature {feature_id}",
                        )
                        ax.set_xlim(slice_x.start - 0.5, slice_x.stop - 0.5)  # .stop is exclusive, so we subtract 0.5
                        ax.set_ylim(slice_y.start - 0.5, slice_y.stop - 0.5)  # .stop is exclusive, so we subtract 0.5
                    detection_table = Table(names=["row_index", "x_peak_index"], dtype=["i4", "f8"])
                    detection_table["row_index"].format = "d"
                    detection_table["x_peak_index"].format = ".2f"
                    for j in range(slice_y.start, slice_y.stop):
                        ixfit = iok[iok[:, 0] == j, 1]  # x indices of valid arc line pixels in row j
                        if allplots:
                            ax.plot(ixfit, [j] * len(ixfit), "C0o")
                        logger.debug(f"Row {j}: x indices of valid arc line pixels: {ixfit}")
                        # TODO: check if this is the correct reference image for fitting:
                        # it can be the denoised image, the background-subtracted image, or the original image
                        image_slice_reference = image_slice_denoised
                        logger.debug(f"{image_slice_reference[j, ixfit]=}")
                        if len(ixfit) >= 5:  # require at least 5 pixels to fit a peak
                            nwinwidth = 5
                            ixpeaks = find_peaks_spectrum(image_slice_reference[j, ixfit], nwinwidth=nwinwidth)
                            if len(ixpeaks) == 0:  # if no peaks were found, try with a smaller window
                                nwinwidth = 3
                                ixpeaks = find_peaks_spectrum(image_slice_reference[j, ixfit], nwinwidth=nwinwidth)
                        elif len(ixfit) >= 3:  # require at least 3 pixels to fit a peak
                            nwinwidth = 3
                            ixpeaks = find_peaks_spectrum(image_slice_reference[j, ixfit], nwinwidth=nwinwidth)
                        else:
                            ixpeaks = []
                        if len(ixpeaks) == 1:  # a single peak was detected
                            ixpeak = ixfit[ixpeaks]  # convert to x indices in the original image
                            xpeak, sxpeak, dict_moments = refine_peaks_spectrum(
                                image_slice_reference[j, ixfit], ixpeaks, nwinwidth=nwinwidth, method="gaussian+moments"
                            )
                            xpeak = xpeak[0]  # convert to scalar
                            xpeak += ixfit[0]  # convert to x indices in the original image
                            detection_table.add_row([j, xpeak])
                            if allplots:
                                ax.plot(ixpeak, j, "C1o")
                                ax.plot(xpeak, j, "C3o")
                            # Store the FWHM in the FWHM array
                            if dict_moments is not None:
                                fwhm_np = dict_moments["fwhm_np"]
                                if fwhm_np is not None and not np.isnan(fwhm_np):
                                    slice_fwhm_np[j, int(xpeak)] = fwhm_np
                                skewness = dict_moments["skewness"]
                                if skewness is not None and not np.isnan(skewness):
                                    slice_skewness[j, int(xpeak)] = skewness
                                kurtosis_excess = dict_moments["kurtosis_excess"]
                                if kurtosis_excess is not None and not np.isnan(kurtosis_excess):
                                    slice_kurtosis_excess[j, int(xpeak)] = kurtosis_excess
                    dict_arc_lines_detections[islice].append(detection_table)
                    if allplots:
                        logger.debug("Showing plot for valid arc line.")
                        plt.show()
        filled_slice_fwhm_np = fill_voronoi_kdtree(slice_fwhm_np)
        filled_slice_skewness = fill_voronoi_kdtree(slice_skewness)
        filled_slice_kurtosis_excess = fill_voronoi_kdtree(slice_kurtosis_excess)
        image_fwhm_np[mask_slice] = filled_slice_fwhm_np[mask_slice]
        image_skewness[mask_slice] = filled_slice_skewness[mask_slice]
        image_kurtosis_excess[mask_slice] = filled_slice_kurtosis_excess[mask_slice]

    fig, axarr = plt.subplots(nrows=2, ncols=2, figsize=(15, 12), sharex=True, sharey=True)
    ax1, ax2, ax3, ax4 = axarr.flatten()

    if vlim is not None:
        vmin, vmax = vlim
    else:
        vmin, vmax = ZScaleInterval().get_limits(
            image_data_denoised.flatten()[~np.isnan(image_data_denoised.flatten())]
        )
    img_ax1, _, _ = tea.imshow(
        fig,
        ax1,
        image_data_denoised,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        title=f"Input image: {Path(image).name} (denoised)",
    )
    for islice in range(FRIDA_NSLICES):
        if islice in dict_arc_lines_detections and len(dict_arc_lines_detections[islice]) > 0:
            for detection_table in dict_arc_lines_detections[islice]:
                ax1.plot(
                    detection_table["x_peak_index"],
                    detection_table["row_index"],
                    "C1+",
                    markersize=2,
                )

    vmin, vmax = ZScaleInterval().get_limits(
        image_fwhm_np[mask_slice].flatten()[~np.isnan(image_fwhm_np[mask_slice].flatten())]
    )
    img_ax2, _, _ = tea.imshow(
        fig,
        ax2,
        image_fwhm_np,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        title="FWHM of valid arc lines",
        cmap="viridis",
        cblabel="FWHM (pixels)",
    )

    vmin, vmax = ZScaleInterval().get_limits(
        image_skewness[mask_slice].flatten()[~np.isnan(image_skewness[mask_slice].flatten())]
    )
    img_ax3, _, _ = tea.imshow(
        fig,
        ax3,
        image_skewness,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        title="Skewness of valid arc lines",
        cmap="viridis",
        cblabel="Skewness",
    )

    vmin, vmax = ZScaleInterval().get_limits(
        image_kurtosis_excess[mask_slice].flatten()[~np.isnan(image_kurtosis_excess[mask_slice].flatten())]
    )
    img_ax4, _, _ = tea.imshow(
        fig,
        ax4,
        image_kurtosis_excess,
        vmin=vmin,
        vmax=vmax,
        aspect="auto",
        title="Kurtosis excess of valid arc lines",
        cmap="viridis",
        cblabel="Kurtosis excess",
    )

    mouse_info = {"ax": None, "x": None, "y": None}

    def on_mouse_move(event):
        if event.inaxes in [ax1, ax2, ax3, ax4]:
            mouse_info["ax"] = event.inaxes
            mouse_info["x"] = event.xdata
            mouse_info["y"] = event.ydata

    fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)

    def on_key(event):
        ax_mouse = mouse_info["ax"]
        if event.key in [",", "/"]:
            if ax_mouse in [ax1, ax2, ax3, ax4]:
                xmin, xmax = list(ax_mouse.get_xlim())
                ymin, ymax = list(ax_mouse.get_ylim())
                ixmin = int(round(xmin))
                ixmax = int(round(xmax))
                iymin = int(round(ymin))
                iymax = int(round(ymax))
                if ixmin > ixmax:
                    ixmin, ixmax = ixmax, ixmin
                if iymin > iymax:
                    iymin, iymax = iymax, iymin
                if ixmin < 1:
                    ixmin = 1
                if ixmax > FRIDA_NAXIS1_HAWAII.value:
                    ixmax = FRIDA_NAXIS1_HAWAII.value
                if iymin < 1:
                    iymin = 1
                if iymax > FRIDA_NAXIS2_HAWAII.value:
                    iymax = FRIDA_NAXIS2_HAWAII.value
                region2d = tea.SliceRegion2D(f"[{ixmin}:{ixmax},{iymin}:{iymax}]", mode="fits").python
                if ax_mouse == ax1:
                    if event.key == ",":
                        vmin, vmax = np.min(image_data[region2d]), np.max(image_data[region2d])
                    elif event.key == "/":
                        vmin, vmax = ZScaleInterval().get_limits(image_data[region2d].flatten())
                    img_ax1.set_clim(vmin, vmax)
                    ax1.figure.canvas.draw_idle()
                if ax_mouse == ax2:
                    if event.key == ",":
                        vmin, vmax = np.min(image_fwhm_np[region2d]), np.max(image_fwhm_np[region2d])
                    elif event.key == "/":
                        vmin, vmax = ZScaleInterval().get_limits(image_fwhm_np[region2d].flatten())
                    img_ax2.set_clim(vmin, vmax)
                    ax2.figure.canvas.draw_idle()
                if ax_mouse == ax3:
                    if event.key == ",":
                        vmin, vmax = np.min(image_skewness[region2d]), np.max(image_skewness[region2d])
                    elif event.key == "/":
                        vmin, vmax = ZScaleInterval().get_limits(image_skewness[region2d].flatten())
                    img_ax3.set_clim(vmin, vmax)
                    ax3.figure.canvas.draw_idle()
                if ax_mouse == ax4:
                    if event.key == ",":
                        vmin, vmax = np.min(image_kurtosis_excess[region2d]), np.max(image_kurtosis_excess[region2d])
                    elif event.key == "/":
                        vmin, vmax = ZScaleInterval().get_limits(image_kurtosis_excess[region2d].flatten())
                    img_ax4.set_clim(vmin, vmax)
                    ax4.figure.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    fig.tight_layout()
    plt.show()


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
    parser.add_argument("--sliceid", help="Overplot slice ID", action="store_true")
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

    # Include default arguments for common actions, and initialize console and logging
    myscript = NuminaScriptDefinition(parser, argv=args)
    args = myscript.args
    logger = myscript.logger

    # Find and characterize arc lines in the image using the provided parameters
    logger.debug("Calling find_arc_lines script function")
    find_arc_lines(
        image=args.image,
        input_poly=args.poly,
        xlim=args.xlim,
        ylim=args.ylim,
        vlim=args.vlim,
        voffset=args.voffset,
        denoise_patch_size=args.denoise_patch_size,
        denoise_patch_distance=args.denoise_patch_distance,
        times_sigma_threshold=args.times_sigma_threshold,
        minheight_percentage=args.minheight_percentage,
        maxwidth_pixels=args.maxwidth_pixels,
        sliceid=args.sliceid,
        allplots=args.allplots,
    )

    # Display goodbye message and save console if recording is enabled
    myscript.goodbye_message_and_save_console()


if __name__ == "__main__":
    main()
