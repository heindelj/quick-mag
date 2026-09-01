"""Offloading quick-mag calculations to a machine that has the compute.

The subpackage is deliberately split by what each module is allowed to import:

* :mod:`quick_mag.remote.protocol` and :mod:`quick_mag.remote.client` are staged
  into the Pyodide build, so they must stay numpy/stdlib-only. Neither may import
  ase, chgnet, torch, or :mod:`quick_mag.chgnet_runner`.
* :mod:`quick_mag.remote.executor` and :mod:`quick_mag.remote.server` only ever
  run on the compute host, and are excluded from the browser manifest.

Nothing here is imported at package-import time: pulling in ``quick_mag.remote``
must not drag the HTTP server into the UI process.
"""

from quick_mag.remote.protocol import PROTOCOL_VERSION

__all__ = ["PROTOCOL_VERSION"]
