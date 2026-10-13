import pytest
from fridadrp.tools.frida_ifu_simulator import main

def test_main():
    with pytest.raises(SystemExit):
        main(['--version'])
