#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Define columns to analyze from column ranges for FRIDA DRP tools."""

import logging

from fridadrp.core import FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL


def columns_to_analyze_from_colranges(colranges):
    """Given a list of column ranges, return a list of columns to analyze.

    The column ranges are specified as a list of lists, where each
    inner list contains two integers [start, end] representing
    the start and end columns (inclusive) to analyze.
    The function checks that the specified ranges are within the valid
    bounds of the FRIDA HAWAII detector along NAXIS1, taking into account
    the first and last useful pixels, and returns a sorted list of
    unique columns to analyze.

    Parameters
    ----------
    colranges : list of lists or None
        List of column ranges, where each range is a list of two integers [start, end].
        If None, the default range from FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL
        to FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL is used.

    Returns
    -------
    columns_to_analyze : list
        List of columns to analyze.
    """
    logger = logging.getLogger(__name__)

    # Check column ranges
    valid_colranges = []
    if colranges is None:
        valid_colranges.append(
            [FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL.value, FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL.value]
        )
        logger.info(
            f"Column ranges not specified. Using default range: {valid_colranges[0][0]} to {valid_colranges[0][1]}."
        )
    else:
        for rng in colranges:
            if len(rng) != 2:
                raise ValueError(f"Invalid column range: {rng}. Must be a pair of integers.")
            if not isinstance(rng[0], int) or not isinstance(rng[1], int):
                raise ValueError(f"Invalid column range: {rng}. Start and end must be integers.")
            if (
                rng[0] < FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL.value
                or rng[1] > FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL.value
            ):
                raise ValueError(
                    f"Column range {rng} is out of bounds. Valid range is [{FRIDA_NAXIS1_HAWAII_FIRST_USEFUL_PIXEL.value}, {FRIDA_NAXIS1_HAWAII_LAST_USEFUL_PIXEL.value}]."
                )
            if rng[0] > rng[1]:
                raise ValueError(f"Invalid column range: {rng}. Start column must be less than or equal to end column.")
            valid_colranges.append(rng)

    # Define columns to analyze
    columns_to_analyze = []
    for colrange in valid_colranges:
        columns_to_analyze.extend(range(colrange[0], colrange[1] + 1))

    # Remove duplicates
    columns_to_analyze = list(set(columns_to_analyze))

    # Sort the columns
    columns_to_analyze.sort()

    if len(columns_to_analyze) == columns_to_analyze[-1] - columns_to_analyze[0] + 1:
        mode = "continuous"
    else:
        mode = "discontinuous"
    logger.info(
        f"Columns to analyze: from {columns_to_analyze[0]} to {columns_to_analyze[-1]} (total {len(columns_to_analyze)} columns) - Mode: {mode}"
    )

    return columns_to_analyze
