"""The recipe of the mode test2, with the slow functions of the IFU replaced by fakes"""

import inspect

import astropy.io.fits as fits
import numpy as np
import pytest
from numina.core import DataFrame, ObservationResult

import fridadrp.recipes.auxiliary as auxiliary
from fridadrp.core import FRIDA_NAXIS1_HAWAII, FRIDA_NAXIS2_HAWAII


def fake_of(function, calls, value):
    """A fake of function that checks its arguments against the real signature"""

    def fake(*args, **kwargs):
        inspect.signature(function).bind(*args, **kwargs)
        calls[function.__name__] = kwargs
        return value

    return fake


@pytest.fixture
def calls(monkeypatch, tmp_path):
    """The arguments of the functions called by the recipe"""
    calls = {}
    model = tmp_path / "model_ifu2detector.json"
    model.write_text("{}")
    fakes = {
        "define_auxiliary_files": {"model_ifu2detector": str(model)},
        "define_3d_wcs": "wcs3d",
        "compute_image2d_rss_from_detector_method1": np.zeros((2, 2)),
        "compute_image3d_ifu_from_rss_method1": np.zeros((2, 2, 2)),
    }
    for name, value in fakes.items():
        monkeypatch.setattr(auxiliary, name, fake_of(getattr(auxiliary, name), calls, value))
    return calls


def frame(value):
    shape = (FRIDA_NAXIS2_HAWAII.value, FRIDA_NAXIS1_HAWAII.value)
    hdu = fits.PrimaryHDU(np.full(shape, value, dtype=np.float32))
    for key, val in {"GRATING": "medium-k", "SCALE": "FINE", "IPA": 0.0, "RADEG": 10.0, "DECDEG": 20.0}.items():
        hdu.header[key] = val
    return DataFrame(frame=fits.HDUList([hdu]))


def test_test2(calls):
    """Target minus sky, and the functions of the IFU receive the logger of the recipe"""
    ob = ObservationResult(instrument="FRIDA", mode="test2")
    ob.frames = [frame(5.0), frame(2.0)]
    recipe = auxiliary.Test2Recipe()
    result = recipe.run(recipe.create_input(obresult=ob))

    assert np.allclose(result.reduced_image.frame[0].data, 3.0)
    assert sorted(calls) == [
        "compute_image2d_rss_from_detector_method1",
        "compute_image3d_ifu_from_rss_method1",
        "define_3d_wcs",
        "define_auxiliary_files",
    ]
    for kwargs in calls.values():
        assert kwargs["logger"] is recipe.logger
