# Pyodide Prototype

This folder contains a browser bootstrap for the `imgui-bundle` molecular UI prototype.
It runs the exact same `imgui_mol_prototype.main()` used on the desktop, in-browser via
Pyodide — no server-side Python.

## Run locally

From the repository root:

```bash
python web/build_manifest.py      # regenerate the staging manifest (see below)
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/web/index.html
```

## What the page does

1. Loads Pyodide from the jsDelivr CDN.
2. Binds Pyodide's SDL layer to the page canvas.
3. Loads the runtime packages listed in `manifest.json` (`numpy`, `scipy`) from the
   Pyodide distribution, plus the local Pyodide-specific `imgui_bundle` wheel.
4. Fetches every file in `manifest.json` and writes it into Pyodide's virtual
   filesystem under `/app` (the `quick_mag` package at `/app/quick_mag`, the fitted
   `exchange_params/*.json`, and the sample geometries under `/app/assets`).
5. Puts `/app` on `sys.path` and launches `quick_mag.quick_mag_ui.main()`.

## The manifest

`web/index.html` is data-driven: it stages exactly the files listed in
`web/manifest.json`. Regenerate that file whenever the `src/` modules, the fitted
parameter set, or the sample assets change:

```bash
python web/build_manifest.py
```

`build_manifest.py` auto-discovers all `src/quick_mag/*.py` modules and
`src/quick_mag/exchange_params/*.json`, and pins the sample assets + runtime packages.
If you add a new package module, just rerun it — no need to hand-edit `index.html`.

## Sample geometries

Two structures are staged into the browser filesystem so you have something to load:

- `/app/assets/goethite_ZnH81_121.vasp`
- `/app/assets/A_type/LaMnO3_222.cif`

Click **Load sample asset** in the **Geometry loader** panel to load the bundled sample.

## Uploading your own geometry

The browser build accepts local `.cif` (P1) and VASP/POSCAR files. In the **Geometry
loader** panel click **Upload geometry file…** (or drag a file onto the 3D view). The
hidden `<input type="file">` in `index.html` reads the file, writes it into the Pyodide
filesystem under `/app/uploads/`, and queues its path on `window.quickMagPendingUploads`;
`quick_mag_ui` drains that queue each frame and loads it through the same parsers as the
desktop app.
