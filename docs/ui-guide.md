# Web & Desktop UI

quick_mag ships an interactive application for building perovskites, loading
geometries, and visualizing predicted spin structures. The **same Python application**
runs two ways:

- **Desktop** — a native window launched from the command line.
- **Web** — the identical app compiled to WebAssembly with
  [Pyodide](https://pyodide.org) so it runs entirely in the browser.

## Run on the web

Open the hosted build:

**[heindelj.github.io/quick-mag/app/](https://heindelj.github.io/quick-mag/app/)**

The first load takes a couple seconds while the browser downloads Pyodide,
`imgui-bundle` (the UI dependency), `numpy`, `scipy`, and the application modules.

Click **Load sample asset** for a sample structure or upload your own:

- In the **Geometry loader** panel, click **Upload geometry file…** and pick a `.cif`
  (P1) or VASP/POSCAR file, **or**
- drag a geometry file straight onto the 3D view.

## Run on the desktop

```bash
pip install "quick_mag[ui]"
quick-mag ui
```

## The interface

The window is a docked workspace with panels for:

- **Controls** — perovskite builder parameters and geometry loading;
- **Structure View** — 3D rendering of atoms, octahedra, and spin arrows;
- **Active Structure** — details and per-site assignments for the focused structure;
- **Calculation Output** — oxidation-state, exchange, and spin-solve results;
- **Export** — write structures (CIF / VASP with magmom lines) and build scripts.

## Run the web build locally

From a checkout:

```bash
python web/build_manifest.py      # regenerate the staging manifest after src changes
python -m http.server 8000
# open http://localhost:8000/web/index.html
```

`web/index.html` is data-driven: it stages exactly the files listed in
`web/manifest.json`, which `web/build_manifest.py` regenerates by auto-discovering the
`quick_mag` package modules, the fitted `exchange_params/*.json`, and the sample assets.
