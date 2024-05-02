=========
FRIDA DRP
=========

|travis| |coveralls|

This is FRIDA DRP, the data reduction pipeline for FRIDA.

`FRIDA (inFrared Imager and Dissector for Adaptative optics)
<https://www.gtc.iac.es/instruments/frida/frida.php>`_ is a
second-generation instrument for the 10m `Gran Telescopio de Canarias
<https://www.gtc.iac.es/>`_ and the first proposed for working with
its Adaptive Optics system.

The software is under development. It is possible that changes may
be made that render the code backward incompatible. For the time
being, it is recommended to install the development version.
This code makes use of the `numina <https://github.com/guaix-ucm/numina>`_
package, which contains common functionality for different GTC instruments.

Installing and running the development version
==============================================

Please follow the instructions available in the `online documentation
<https://guaix.fis.ucm.es/~ncl/fridadrp-tutorials/>`_.

Licensing
=========

FRIDA DRP is distributed under GNU GPL, either version 3 of the License,
or (at your option) any later version. See the file LICENSE.txt for details.

Authors
=======

Maintainers: Nicolás Cardiel <cardiel@ucm.es>, Sergio Pascual: <sergiopr@fis.ucm.es>


.. |travis| image:: https://img.shields.io/travis/guaix-ucm/fridadrp/master?logo=travis%20ci&logoColor=white&label=Travis%20CI
    :target: https://travis-ci.org/guaix-ucm/fridadrp
    :alt: fridadrp's Travis CI Status

.. |coveralls| image:: https://coveralls.io/repos/guaix-ucm/fridadrp/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/guaix-ucm/fridadrp?branch=master
     :alt: fridadrp's Coverall Status
