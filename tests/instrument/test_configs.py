import pytest
import astropy.io.fits as fits
import numina.instrument.generic
import numina.core
from fridadrp.loader import drp_load


def create_simple_frame():
    hdr = {'INSTRUME': 'FRIDA', 'INSCONF': 'c7f94f7d-1f57-4644-86d6-7004a2506680'}
    hdu = fits.PrimaryHDU(data=[[1]])
    for k, v in hdr.items():
        hdu.header[k] = v
    hdulist = fits.HDUList([hdu])
    frame = numina.core.DataFrame(frame=hdulist)
    return frame


@pytest.mark.parametrize("conf, uuix", [
    ['default', 'c7f94f7d-1f57-4644-86d6-7004a2506680'],
    ['c7f94f7d-1f57-4644-86d6-7004a2506680', 'c7f94f7d-1f57-4644-86d6-7004a2506680'],
])
def test_loader1(conf, uuix):
    import numina.core
    from numina.instrument.assembly import assembly_instrument

    obs = numina.core.ObservationResult(instrument='FRIDA')
    obs.frames.append(create_simple_frame())
    drpm = drp_load()
    obs.configuration = conf

    key, date_obs, keyname = drpm.select_profile(obs)
    ins = assembly_instrument(drpm.configurations, key, date_obs, by_key=keyname)
    assert isinstance(ins, numina.instrument.generic.InstrumentGeneric)
    assert str(ins.origin.uuid) == uuix
