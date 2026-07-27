"""quick_mag — perovskite/crystal structure builder, visualization, and magnetism tools.

The package exposes a collinear-magnetism pipeline (structure -> oxidation states ->
exchange couplings -> low-energy spin configurations), a command-line front-end
(``quick-mag``), and an interactive Dear ImGui UI (``quick_mag.quick_mag_ui``) that runs
both as a native desktop window and, compiled to WASM via Pyodide, in the browser.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
