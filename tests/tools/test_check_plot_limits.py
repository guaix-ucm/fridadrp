"""Tests for the check of the plot limits (xlim, ylim, vlim)"""

import numpy as np
import pytest

from fridadrp.tools.check_plot_limits import check_plot_limits


@pytest.mark.parametrize(
    "limits",
    [None, [1.0, 2048.0], (1, 2048), np.array([10.5, 20.5]), [2048.0, 1.0], [-100.0, 0.0]],
)
def test_check_plot_limits_valid(limits):
    original = None if limits is None else np.array(limits, copy=True)
    assert check_plot_limits(limits, "xlim") is None
    # The input is not modified
    if limits is not None:
        assert np.array_equal(np.asarray(limits), original)


@pytest.mark.parametrize(
    "limits, match",
    [
        ([1.0], "exactly two numbers"),
        ([1.0, 2.0, 3.0], "exactly two numbers"),
        ([], "exactly two numbers"),
        ([[1.0, 2.0]], "exactly two numbers"),
        (5.0, "exactly two numbers"),
        ("12", "must contain two numbers"),
        (["a", "b"], "must contain two numbers"),
        ([1.0, None], "finite numbers"),
        ([np.nan, 2.0], "finite numbers"),
        ([1.0, np.inf], "finite numbers"),
        ([3.0, 3.0], "two different numbers"),
    ],
)
def test_check_plot_limits_invalid(limits, match):
    with pytest.raises(ValueError, match=match):
        check_plot_limits(limits, "xlim")


def test_check_plot_limits_name_in_message():
    with pytest.raises(ValueError, match=r"^ylim=\[1\.0\] must contain exactly two numbers\.$"):
        check_plot_limits([1.0], "ylim")


def test_check_plot_limits_require_increasing():
    check_plot_limits([0.0, 100.0], "vlim", require_increasing=True)
    with pytest.raises(ValueError, match=r"vlim\[0\] < vlim\[1\]"):
        check_plot_limits([100.0, 0.0], "vlim", require_increasing=True)
