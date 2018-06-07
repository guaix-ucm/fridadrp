import pytest
import numina.core.pipeline

import fridadrp.loader


@pytest.mark.parametrize('cfgid', [
        'c7f94f7d-1f57-4644-86d6-7004a2506680',
    ])
def test_conf1(cfgid):
    drp = fridadrp.loader.drp_load()

    assert cfgid in drp.configurations

    cfg = drp.configurations[cfgid]
    assert isinstance(cfg, numina.core.pipeline.InstrumentConfiguration)
    assert cfg.instrument == 'FRIDA'
    assert cfg.uuid == cfgid

