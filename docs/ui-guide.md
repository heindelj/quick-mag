# Web & Desktop UI

quick_mag ships an interactive Dear ImGui application for building perovskites, loading
geometries, and visualizing predicted spin structures. The **same Python application**
runs two ways:

- **Desktop** — a native window (SDL/OpenGL), launched from the command line.
- **Web** — the identical app compiled to WebAssembly with
  [Pyodide](https://pyodide.org) and served as static files, so it runs entirely in the
  browser with no server-side Python.

## Run on the web

Open the hosted build:

**[heindelj.github.io/quick-mag/app/](https://heindelj.github.io/quick-mag/app/)**

The first load takes a little while: the browser downloads Pyodide, a Pyodide-specific
`imgui-bundle` wheel, `numpy`, `scipy`, and the application modules. Two sample
geometries are staged into the in-browser filesystem so you have something to load:

- `/app/assets/goethite_ZnH81_121.vasp`
- `/app/assets/A_type/LaMnO3_222.cif`

Click **Load sample asset** for the bundled sample, or bring your own structure:

- In the **Geometry loader** panel, click **Upload geometry file…** and pick a `.cif`
  (P1) or VASP/POSCAR file, **or**
- drag a geometry file straight onto the 3D view.

The uploaded file is staged into the in-browser filesystem and loaded through the same
parsers as the desktop app; it appears as a new focused structure.

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
