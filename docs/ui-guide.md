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
- **Calculate** — the solver workflow and its settings;
- **Calculation Output** — oxidation-state, exchange, and spin-solve results;
- **Export** — write structures to disk (CIF / VASP with magmom lines).

The builder is on the left; **Calculate** and **Calculation Output** are tabs sharing the
wider dock on the right, so setting up a solve and reading its results happen in the same
place.

On the web there is no filesystem to write to, so **Export** has no folder field: the two
buttons hand the files to your browser as downloads instead. A structure with no saved
spin configurations arrives as a single `.cif`; anything that produces more than one file
arrives as `quick_mag_export.zip`. The contents are identical to what the desktop app
writes.

The app opens with one structure already built from the default builder settings, and
there is always exactly one **active structure**. Builder edits regenerate the active
structure in place — there is no separate save step, so you can run calculations on it
right away. **New structure** (in either the Controls or Active Structure panel) resets
the builder to its defaults and adds another structure to the list. Right-click a
structure in the Active Structure panel to rename or delete it. Structures loaded from a
file have no builder provenance, so the builder is disabled while one is active.


## Supercell size

The Lattice panel's *Supercell a / b / c* count primitive cells, so `1` is the
primitive cell itself (5 atoms for a plain ABX₃) and `3` is a 3×3×3 supercell. The
ordered formula modes (double, quadruple, doubly-ordered) already double the grid to
carry their alternating site patterns, so one primitive cell of those is two plain
perovskite cells. The app opens on a 3×3×3 grid (135 atoms); switching to an ordered
mode opens on supercell 2, which is a 4×4×4 octahedron grid. Both are comfortably above
the two cells per axis the `A`/`C`/`G` reference orderings need.

## Inspecting individual atoms

Under the spin-configuration list, **Per-site oxidation states and moments** is a
scrollable list with one row per atom — element, assigned oxidation state, and moment
vector. Click a row to draw a circle around that atom in the 3D view and click again to clear the circle.

## The spin energy landscape

The plot under the 3D view is seeded automatically, before you run anything. Any structure
whose magnetic sites sit on a lattice gets the canonical orderings evaluated as single
points against the current exchange matrix. `A` and `C` each single out one axis, so a
cubic cell shows the three `A` orientations (and the three `C`s) exactly degenerate, and
any distortion — a tilt, an anisotropic lattice constant — splits them apart.

### Orderings are plane patterns

An ordering is a **family of lattice planes plus a sign string repeated across
successive planes**. `G` is "alternate the sign on successive (111) planes"; `A(c)` is the
same on (001); `F` is a single plane taking a single sign. The eight classical orderings
are exactly the eight period-2 patterns — the set is complete, with nothing left over:

| ordering | plane | string | | ordering | plane | string |
|---|---|---|---|---|---|---|
| `F` | (000) | `+` | | `C(a)` | (011) | `+-` |
| `A(a)` | (100) | `+-` | | `C(b)` | (101) | `+-` |
| `A(b)` | (010) | `+-` | | `C(c)` | (110) | `+-` |
| `A(c)` | (001) | `+-` | | `G` | (111) | `+-` |

Longer strings extend the set. `++--` is the up-up-down-down modulation, so `E(a)` is
`(100) ++--`; the diagonal members of that family have no classical name and are shown in
plane notation, as `(110) ++--`. A pattern is only offered when the cell has enough planes
to tell it apart from a shorter one — `E` needs four planes along its axis, so it appears
once a supercell is four cells wide.

The Miller indices are relative to the **magnetic sublattice's own lattice**, not to the
cell. On a 3×3×3 supercell the `G` planes are the cell's (333); the app converts for you
when it draws them.

Because an ordering is defined site by site, a configuration no longer has to sit exactly
on one to be recognised. Each is reported as the ordering it is nearest to plus a **defect
concentration** — the fraction of magnetic sites whose spin disagrees with the ideal,
minimized over a global spin flip and over the phase of the sign string. `G  7.4% defects`
means two of 27 sites are wrong. A configuration further than 25% from every pattern is
still `Other`: past that, "nearest" stops meaning anything. Tick **Ring deviating sites**
at the top of Calculation Output to circle exactly those sites in the 3D view.

The configuration you have selected persists too. The landscape is re-sorted by energy
whenever it is re-evaluated, so an edit moves a given arrangement up or down the list —
the selection follows the **arrangement**, not its position, so the same spins stay on
screen with their new energy. A change that alters the number of magnetic sites has
nothing to hold on to, and falls back to the ground state; so does a fresh solve, which
deliberately presents the state it just found.

Those points **persist as you edit the structure**. Re-evaluating them means rebuilding
the oxidation assignments and the exchange matrix, which costs tens to hundreds of
milliseconds on a large cell — far too much to repeat on every frame of a slider drag —
so it does not happen automatically. A builder edit instead marks the energies **stale**,
and the plot holds its last values until you either tick **Update spin energies
interactively** (under **Solve selected oxidation state** in Calculation Output),
press **Refresh energies**, or run a solve. With the checkbox on, every edit rebuilds the
exchange matrix and re-evaluates every plotted configuration against it, so the plot
tracks the structure continuously. New configurations come only from **Run Magnetic
Structure**, which fills in the rest of the landscape.

**Color atoms by spin**, at the top of Calculation Output, draws magnetic atoms in the
spin-up (turquoise) and spin-down (yellow) colors instead of their element colors. It is
off by default, so the 3D view stays element-colored until you ask for the spins.

Point colors follow the ordering a configuration is nearest to, with gray `Other` for
anything past the cutoff. Reference points are ordinary configurations in every other
respect — click one to see its spins in the 3D view, or save it with **Save magnetic
configuration**, which records both the ordering and its defect concentration.

## Ordering planes

**Draw ordering planes**, beside **Color atoms by spin** at the top of Calculation Output,
overlays translucent sheets on the planes the selected configuration's ordering alternates
across, tinting each with the spin its sign string assigns. An `A`-type structure becomes
obvious as alternating turquoise and yellow layers, and `G` as the seven stacked (111)
sheets of a 3×3×3 cell. The sheets sit on the planes the magnetic sites actually occupy,
so every magnetic atom lies on one. `F` has no modulation and so draws nothing.

## Which sites are drawn

The **Sites drawn** column at the top of Calculation Output, beside the spin view options,
carries an **A / B / X** toggle per site role, and an **Octahedra** toggle under them. All
three site roles are on by default. Switching X off does not remove the octahedra, so the
framework stays visible without the oxygen spheres, which is a quick way to see the B
sublattice on its own. A structure loaded from a file has no site roles recorded, so all
of its atoms are drawn whatever these are set to, and it has no lattice to draw octahedra
from either.

**Max oxidation assignments** under Solver Settings (default 200) caps how many
energy-ranked oxidation-state assignments are kept. A cell with several mixed-valence
cations can enumerate tens of thousands of charge-balanced distributions — a
doubly-ordered perovskite with a Cu A' site reached 78,312 — and expanding them all into
per-site assignments is the slowest part of setting up a solve. They are ranked, so the
kept head is the part worth choosing between; set it to 0 to keep every one.

Two settings under Solver Settings control what is plotted. **Max plotted
configurations** (default 100) caps how many points are shown; the reference orderings
always keep their slots. **Plot degenerate configs** (off by default) decides how that
budget is spent: the exchange model produces many distinct configurations at identical
energies, so leaving it off shows one representative per distinct energy and the budget
reaches much further up the landscape. On a tilted 3×3×3 cell, 100 points cover 12
distinct energies with it on, and 64 with it off. Collapsed points are not discarded —
the count of configurations sharing an energy is shown as `×N` beside the entry, and
turning the option back on restores them.

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
