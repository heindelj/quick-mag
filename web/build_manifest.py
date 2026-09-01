"""Generate ``web/manifest.json`` listing everything the browser build must stage.

The Pyodide bootstrap (``web/index.html``) reads this manifest, fetches each file,
and writes it into the Pyodide virtual filesystem under ``/app`` before launching
``quick_mag.quick_mag_ui.main()``. The ``quick_mag`` package is staged at
``/app/quick_mag`` and ``/app`` is placed on ``sys.path``, so the same absolute
imports (``from quick_mag.structure import ...``) used on the desktop work in-browser.

Regenerate after adding/removing package modules, the fitted parameter set, or sample
assets:

    python web/build_manifest.py
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PKG = REPO / "src" / "quick_mag"

# Runtime third-party packages available in the Pyodide distribution.
PACKAGES = ["numpy", "scipy"]

# Pyodide-specific imgui-bundle wheel shipped alongside this page.
WHEEL = "local_wheels/imgui_bundle-1.92.700-cp313-cp313-pyodide_2025_0_wasm32.whl"

# Package modules that must NOT be staged: they only ever run on a compute host
# and would pull ase/chgnet/torch into a build that has none of them. Everything
# else under src/quick_mag, including subpackages, is discovered automatically.
EXCLUDED_MODULES = {
    "remote/server.py",
    "remote/executor.py",
}

# Sample geometries staged so the app has something to load in the browser.
ASSETS = [
    "assets/goethite_ZnH81_121.vasp",
    "assets/A_type/LaMnO3_222.cif",
]


def _entry(url_rel: str, dest_rel: str) -> dict:
    """Manifest entry: fetch URL (relative to web/) + Pyodide FS destination."""
    return {"url": f"../{url_rel}", "dest": f"/app/{dest_rel}"}


def build_manifest() -> dict:
    files: list[dict] = []

    # First-party package modules -> /app/quick_mag/<relative path>.py
    #
    # rglob rather than glob: quick_mag.remote is a subpackage, and a non-recursive
    # sweep would leave the browser build importing a module it never staged --
    # which fails at startup with nothing but an ImportError to go on.
    for module in sorted(PKG.rglob("*.py")):
        relative = module.relative_to(PKG).as_posix()
        if relative in EXCLUDED_MODULES:
            continue
        files.append(_entry(f"src/quick_mag/{relative}", f"quick_mag/{relative}"))

    # Bundled fitted parameter set(s) read at runtime via __file__-relative paths.
    for params in sorted((PKG / "exchange_params").glob("*.json")):
        files.append(
            _entry(
                f"src/quick_mag/exchange_params/{params.name}",
                f"quick_mag/exchange_params/{params.name}",
            )
        )

    # Sample assets -> /app/assets/... (kept outside the package, next to /app/quick_mag).
    for asset in ASSETS:
        if not (REPO / asset).exists():
            raise FileNotFoundError(f"Sample asset missing: {asset}")
        files.append(_entry(asset, asset))

    return {"packages": PACKAGES, "wheel": WHEEL, "files": files}


def main() -> None:
    manifest = build_manifest()
    out = REPO / "web" / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {out.relative_to(REPO)} with {len(manifest['files'])} files.")


if __name__ == "__main__":
    main()
