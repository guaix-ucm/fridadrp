#
# Copyright 2026 Universidad Complutense de Madrid
#
# This file is part of FRIDA DRP
#
# SPDX-License-Identifier: GPL-3.0-or-later
# License-Filename: LICENSE.txt
#

"""Check the limits (xlim, ylim, vlim) provided for a plot."""

from collections.abc import Sequence

import numpy as np


def check_plot_limits(
    limits: Sequence[float] | np.ndarray | None,
    name: str,
    require_increasing: bool = False,
) -> None:
    """Check that the limits provided for a plot are valid.

    The limits must be None (no limits provided) or a sequence of exactly
    two finite and different numbers. Decreasing limits are accepted by
    default, since matplotlib interprets them as an inverted axis (this is
    valid for xlim and ylim, but not for vlim).

    The input is not modified.

    Parameters
    ----------
    limits : sequence of float, numpy.ndarray or None
        Limits to be checked, e.g. (xmin, xmax), (ymin, ymax) or
        (vmin, vmax). If None, nothing is checked.
    name : str
        Name of the limits (e.g. "xlim"), employed in the error messages.
    require_increasing : bool, optional
        If True, the first value must be smaller than the second one
        (use it for vlim).

    Raises
    ------
    ValueError
        If the limits are not None and do not contain exactly two finite
        numbers, if both numbers are equal, or if require_increasing is
        True and the first number is not smaller than the second one.

    Examples
    --------
    >>> check_plot_limits(None, "xlim")
    >>> check_plot_limits([100.0, 1.0], "ylim")  # inverted axis
    >>> check_plot_limits([100.0, 1.0], "vlim", require_increasing=True)
    Traceback (most recent call last):
        ...
    ValueError: vlim=[100.0, 1.0] must satisfy vlim[0] < vlim[1].
    """
    if limits is None:
        return

    # strings are sequences, but np.asarray would not convert them into numbers
    if isinstance(limits, (str, bytes)):
        raise ValueError(f"{name}={limits!r} must contain two numbers.")
    try:
        array_limits = np.asarray(limits, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"{name}={limits!r} must contain two numbers.") from None
    if array_limits.shape != (2,):
        raise ValueError(f"{name}={limits!r} must contain exactly two numbers.")
    if not np.all(np.isfinite(array_limits)):
        raise ValueError(f"{name}={limits!r} must contain finite numbers.")
    if array_limits[0] == array_limits[1]:
        raise ValueError(f"{name}={limits!r} must contain two different numbers.")
    if require_increasing and array_limits[0] > array_limits[1]:
        raise ValueError(f"{name}={limits!r} must satisfy {name}[0] < {name}[1].")
