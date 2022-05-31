

import numina.core


def drp_load():
    """Entry point to load FRIDA DRP."""
    return numina.core.drp_load('fridadrp', 'drp.yaml')
