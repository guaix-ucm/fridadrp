#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Tests for find_arc_lines"""

import astropy.units as u
from astropy.io import fits
import numpy as np
import pytest
from scipy.special import erf

from fridadrp.core import FRIDA_NAXIS1_HAWAII, FRIDA_NAXIS2_HAWAII
from fridadrp.core import FRIDA_NSLICES
from fridadrp.core import sliceid_from_sliceindex
from fridadrp.tools.find_arc_lines import find_arc_lines

# Synthetic slice geometry: constant boundaries (0-based array indices along NAXIS2),
# with slices of 60 pixels separated by 8 pixels
SLICE_PITCH = 68.0
SLICE_LEFT_OFFSET = 10.0
SLICE_HEIGHT = 60.0

# Slice to be processed (slice number is 1-based)
SLICENUM = 15

# True positions (0-based array indices along NAXIS1) of the synthetic arc lines.
# They are separated by much more than the 201-pixel median filter employed to
# estimate the background, and have non-integer values to check the subpixel accuracy.
X_LINES = np.array([300.3, 700.0, 1100.6, 1500.25])
LINE_SIGMA = 1.0  # pixels
LINE_FLUX = 500.0  # counts per row
BACKGROUND = 100.0  # counts
NOISE_SIGMA = 1.0  # counts


def _gaussian_pixel_integrated(x_edges, mean, sigma, flux):
    """Flux of a Gaussian profile integrated within each pixel."""
    cdf = 0.5 * (1 + erf((x_edges - mean) / (np.sqrt(2) * sigma)))
    return flux * np.diff(cdf)


@pytest.fixture(scope="module")
def synthetic_files(tmp_path_factory):
    """Synthetic arc image and slice boundary polynomial file."""
    tmp_path = tmp_path_factory.mktemp("find_arc_lines")
    naxis1 = FRIDA_NAXIS1_HAWAII.value
    naxis2 = FRIDA_NAXIS2_HAWAII.value

    # Arc spectrum (one row): pixel i covers [i - 0.5, i + 0.5] in array coordinates
    x_edges = np.arange(naxis1 + 1) - 0.5
    spectrum = np.zeros(naxis1)
    for xline in X_LINES:
        spectrum += _gaussian_pixel_integrated(x_edges, xline, LINE_SIGMA, LINE_FLUX)

    # Image: the same arc spectrum in every row (lines perpendicular to the slices),
    # on top of a constant background with Gaussian noise
    rng = np.random.default_rng(1234)
    image = BACKGROUND + spectrum[np.newaxis, :] + rng.normal(0.0, NOISE_SIGMA, size=(naxis2, naxis1))
    image_file = tmp_path / "arc.fits"
    fits.PrimaryHDU(image.astype(np.float32)).writeto(image_file)

    # Slice boundary polynomials (degree 0) for all the slices
    coefs_left = np.array([[islice * SLICE_PITCH + SLICE_LEFT_OFFSET] for islice in range(FRIDA_NSLICES)])
    coefs_right = coefs_left + SLICE_HEIGHT
    header = fits.Header()
    header["KEYCODE"] = "SLICE_BOUNDARY_POLYNOMIALS"
    header["UUID"] = "00000000-0000-0000-0000-000000000000"
    header["POLDEG"] = 0
    for i in range(1, FRIDA_NSLICES + 1):
        header[f"SLCNUM{i:02d}"] = True
    header["SLCNUMT"] = FRIDA_NSLICES
    hdul = fits.HDUList(
        [
            fits.PrimaryHDU(header=header),
            fits.ImageHDU(coefs_left, name="L-BORDER"),
            fits.ImageHDU(coefs_right, name="R-BORDER"),
        ]
    )
    poly_file = tmp_path / "poly.fits"
    hdul.writeto(poly_file)

    return image_file, poly_file


@pytest.fixture(scope="module")
def detection_tables(synthetic_files):
    """Result of find_arc_lines applied to a single slice of the synthetic image."""
    image_file, poly_file = synthetic_files
    return find_arc_lines(image=str(image_file), input_poly=str(poly_file), slicenum_list=[SLICENUM])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nwinwidth_search_peaks": 1},
        {"nwinwidth_search_peaks": 6},
        {"nwinwidth_refine_peaks": 1},
        {"nwinwidth_refine_peaks": 4},
        {"nwinwidth_search_peaks": 5, "nwinwidth_refine_peaks": 7},
        {"fit_method": "poly2"},
        {"fwhm_diffraction": 0.0},
        {"fwhm_diffraction": -1.0},
    ],
)
def test_invalid_parameters(kwargs):
    # The parameters are checked before reading the input files
    with pytest.raises(ValueError):
        find_arc_lines(image="nonexistent.fits", input_poly="nonexistent.fits", **kwargs)


def test_only_requested_slice_is_processed(detection_tables):
    assert len(detection_tables) == FRIDA_NSLICES
    for islice, table in enumerate(detection_tables):
        if islice == SLICENUM - 1:
            assert table is not None
        else:
            assert table is None


def test_table_metadata(detection_tables):
    table = detection_tables[SLICENUM - 1]
    assert table.meta["SLICENUM"][0] == SLICENUM
    assert table.meta["SLICEID"][0] == sliceid_from_sliceindex(SLICENUM - 1)
    assert table["x_peak"].unit == u.pix
    assert table.colnames == [
        "feature_id",
        "row_index",
        "x_peak",
        "amplitude",
        "mean",
        "stddev",
        "skewness",
        "fwhm_np",
        "strehl",
    ]


def test_detected_lines(detection_tables):
    table = detection_tables[SLICENUM - 1]
    # One feature per arc line, numbered from left to right
    feature_ids = np.unique(table["feature_id"])
    assert len(feature_ids) == len(X_LINES)
    median_x_peak = np.array([np.median(table["x_peak"][table["feature_id"] == fid]) for fid in feature_ids])
    assert np.all(np.diff(median_x_peak) > 0)
    # Subpixel accuracy of the line positions and widths
    median_mean = np.array([np.median(table["mean"][table["feature_id"] == fid]) for fid in feature_ids])
    median_stddev = np.array([np.median(table["stddev"][table["feature_id"] == fid]) for fid in feature_ids])
    assert np.allclose(median_x_peak, X_LINES, atol=0.05)
    assert np.allclose(median_mean, X_LINES, atol=0.05)
    assert np.allclose(median_stddev, LINE_SIGMA, atol=0.05)
    # Row indices are within the slice boundaries (0-based array indices)
    ylower = (SLICENUM - 1) * SLICE_PITCH + SLICE_LEFT_OFFSET
    yupper = ylower + SLICE_HEIGHT
    assert np.all(table["row_index"] >= int(ylower + 0.5))
    assert np.all(table["row_index"] <= int(yupper + 0.5))
