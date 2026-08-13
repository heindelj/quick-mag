# Web & Desktop UI

quick_mag ships an interactive application for building perovskites, loading
geometries, and visualizing predicted spin structures. The **same Python application**
runs two ways:

- **Desktop** — a native window launched from the command line.
- **Web** — the identical app compiled to WebAssembly with
  [Pyodide](https://pyodide.org) so it runs entirely in the browser.

## Run on the web

Open the web app:

**[heindelj.github.io/quick-mag/app/](https://heindelj.github.io/quick-mag/app/)**

The first load takes a couple seconds while the browser downloads Pyodide,
`imgui-bundle` (the UI dependency), `numpy`, `scipy`, and the application modules.

Click **Load sample asset** for a sample structure or upload your own:

- In the **Geometry loader** panel, click **Upload geometry file…** and pick a `.cif`
  (P1) or VASP/POSCAR file, **or**
- drag a geometry file straight onto the 3D view.

## Run on the desktop

From a checkout, with your conda environment active (see [Install](index.md#install)):

```bash
pip install -e ".[ui]"
quick-mag ui
```

## The interface

The window is a docked workspace with panels for:

- **Controls** — perovskite builder parameters and geometry loading;
- **Structure View** — 3D rendering of atoms, octahedra, and spin arrows;
- **Active Structure** — the list of structures, plus the saved spin configurations of each;
- **Calculation Output** — oxidation-state, exchange, and spin-solve results;
- **Export** — write structures to disk (CIF / VASP with magmom lines).

The app opens with one structure already built from the default builder settings, and
there is always exactly one **active structure**. Builder edits regenerate the active
structure in place — there is no separate save step, so you can run calculations on it
right away. **New structure** (in either the Controls or Active Structure panel) resets
the builder to its defaults and adds another structure to the list. Right-click a
structure in the Active Structure panel to rename or delete it. Structures loaded from a
file have no builder provenance, so the builder is disabled while one is active.

## The spin energy landscape

The plot under the 3D view is seeded automatically, before you run anything. Any structure
whose B-site grid is at least two cells wide gets the eight canonical orderings —
`G`, `C(a)`, `C(b)`, `C(c)`, `F`, `A(a)`, `A(b)`, `A(c)` — evaluated as single points
against the current exchange matrix. `A` and `C` each single out one axis, so a cubic cell
shows the three `A` orientations (and the three `C`s) exactly degenerate, and any
distortion — a tilt, an anisotropic lattice constant — splits them apart.

Those points **persist as you edit the structure**. A builder edit rebuilds the exchange
matrix and re-evaluates every plotted configuration against it, so the plot tracks the
structure rather than resetting. New configurations come only from **Run Magnetic
Structure**, which fills in the rest of the landscape.

Colors mark **exact** matches: a point is labelled `A(c)` only if its spin arrangement *is*
`A(c)` (up to a global spin flip). Everything else is gray `Other`, which in a solved
landscape is most of the points. Reference points are ordinary configurations in every
other respect — click one to see its spins in the 3D view, or save it with **Save magnetic
configuration**.

**Max plotted configurations** (under Solver Settings, default 100) caps how many points
are kept; the reference orderings always keep their slots.

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
