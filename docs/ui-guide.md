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

To bring in your own structure:

- At the top of the **Controls** panel, click **Load structure…** and pick a `.cif`
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

- **Controls** — **New structure** and **Load structure…** at the top, then the
  perovskite builder with the **Defects & impurities** menu at the bottom; it has the
  whole left side of the window to itself;
- **Structure View** — 3D rendering of atoms, octahedra, and spin arrows. Right-drag to
  rotate, double right-click to return to the starting view, scroll to zoom, and use the
  small `a`, `b`, `c` buttons above the plot to look straight down a lattice vector —
  pressing one twice looks at the opposite face — and `-5°` / `+5°` beside them to step
  about a screen-space axis picked with the `x` / `y` / `z` buttons below (right, up, and
  into the screen, one at a time); the triad in the bottom-left corner tracks the cell's
  orientation;
- **Active Structure** — the list of structures, plus the saved spin configurations of
  each, with the export controls (CIF / VASP with magmom lines) at the bottom. The name
  of the structure the builder is editing reads over the 3D view, under the rotate hint;
- **Calculate** — the solver workflow and its settings;
- **Calculation Output** — oxidation-state, exchange, and spin-solve results;
- **Rendering** — the view options (unit cell, periodic images, radii, legend).

The builder is on the left; **Calculate**, **Calculation Output**, and **Rendering** are
tabs sharing the wider dock on the right, so setting up a solve and reading its results
happen in the same place.

On the web there is no filesystem to write to, so the export controls have no folder
field: the two buttons hand the files to your browser as downloads instead. A structure
with no saved spin configurations arrives as a single `.cif`; anything that produces more
than one file arrives as `quick_mag_export.zip`. The contents are identical to what the
desktop app writes.

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

**Hover an atom in the 3D view** for its number, element, assigned oxidation state and
moment vector — `15. La  ox=+3  m=(+0.00, +0.00, +0.00)`, from `site_hover_tooltip`. Every
atom answers, not only the magnetic ones.

This replaces the **Per-site oxidation states and moments** list that used to sit under the
spin configurations. The list was one row per atom, needed a `ListClipper` to stay
responsive on a 1080-atom cell, and made you match a row against the structure by eye;
`oxidation_site_rows` survives as the batch form of the same formatter for the CLI and the
tests.

The assignment's **distribution** and **model energy** used to be printed under it too.
They are gone as well, and nothing was lost: `format_oxidation_assignment_label` already
puts both on every row of the assignment combo above
(`1. 27xFe+3 | 27xLa+3 | 81xO-2 [E=-0.387]`), so the block restated the selected row.

## The spin energy landscape

The pane under the 3D view holds two plots, chosen with the dropdown in its top-left
corner: this spin-energy landscape, and the [exchange couplings](#the-exchange-couplings)
behind it. Which configuration is drawn in the 3D view is chosen from the
spin-configuration list or by clicking a point here — the same selection either way.

The rule between the 3D view and this pane is a splitter: drag it to change the share each
gets. `split_pane_heights` holds both above `MIN_PLOT3D_HEIGHT` and `MIN_TWO_D_HEIGHT`, and
shares the space in proportion when the pane is too short for even those. The split is
stored as a fraction rather than a height (`AppState.two_d_pane_fraction`) so it survives the
pane being resized around it.

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

### Entering an ordering by hand

Since that is all an ordering is, you can type one: **Custom ordering**, at the foot of the
Calculation Output panel, takes a plane family and a sign string and scores the result into
the landscape on the same footing as the built-in ones. Periods up to
`MAX_CUSTOM_PATTERN_PERIOD` (8) are accepted — `PlanePattern` itself has no limit, but a
period the cell has too few planes to resolve would fold back onto a shorter ordering and be
scored as something it is not, so it is refused with the count.

Custom orderings are held on `AppState.custom_spin_patterns` as plane-notation labels, not as
moments, and `compute_reference_configs` appends them to the canonical set. That is what makes
them survive a builder edit: like the canonical orderings they are rescored against the
current `J` rather than carrying stale moments, and like them they are scored on **unit**
moments — handing over the formal ones instead makes every custom ordering |μ|² too large,
25× on an Fe(3+) cell, which sinks them below the real ground state.

Two cases the tool reports rather than doing silently:

- A canonical ordering re-entered is refused by name: `(111) +-` reports that it is `G`.
- An ordering the cell cannot distinguish from a listed one is added, but named: on a 3×3×3
  cubic cell `(123) +-` produces exactly `C(b)`'s moments and is folded into it by the
  deduplication, so the list looks unchanged and the message has to say why.

A custom ordering also joins the set the classifier chooses from, via
`AppState.match_pattern_candidates`. Without that it would be *listed* under its own name
and *reported* under the nearest canonical one — `(001) +++-` reading as "F, 25% defects" in
the same plot as the row that added it. Canonical patterns are offered first, so
`patterns_for_sites` keeps the classical name when a custom ordering turns out to be one the
cell cannot tell apart from a canonical one; that is the same rule
`select_custom_spin_pattern` reports under, and the two would otherwise disagree. In the
energy scatter each custom label becomes its own legend category, coloured from
`CUSTOM_SPIN_CLASS_COLORS` by its position in the user's list — teal, pink, gold, brown,
hues the canonical palette leaves free — so it is keyed on something that does not move when
another ordering is added above it.

Because an ordering is defined site by site, a configuration no longer has to sit exactly
on one to be recognised. Each is reported as the ordering it is nearest to plus a **defect
concentration** — the fraction of magnetic sites whose spin disagrees with the ideal,
minimized over a global spin flip and over the phase of the sign string. `G  7.4% defects`
means two of 27 sites are wrong. A configuration further than 25% from every pattern is
still `Other`: past that, "nearest" stops meaning anything. Tick **Ring deviating sites**
at the top of Calculation Output to circle exactly those sites in the 3D view.

### Reading M

The `M` on each row, and the **Magnetization** line under the list, is a physical moment
**per perovskite cell**, in μ<sub>B</sub> — `config_magnetization`, not the solver's own
`SpinConfig.magnetization`. Two corrections separate them.

The solver runs on unit ±1 moments, because the size of a spin is already inside `J` (see
`build_unit_moment_assignment`), so its `magnetization` is a count of *sites*, not of Bohr
magnetons. The formal high-spin magnitudes come back off the oxidation assignment —
`magnetization_basis`, the same source `displayed_site_moment_magnitudes` and export use.

It is then divided by the cell count, `unit_cell_count`, which is the product of the B-site
grid shape: one B site is one perovskite cell. Without that, the number scales with the
supercell and cannot be compared between structures — ferromagnetic LaFeO₃ reads 8, 27 or
64 on a 2×2×2, 3×3×3 or 4×4×4, where the meaningful answer is 5 μ<sub>B</sub>/cell, the
high-spin moment of Fe(3+), on all three.

The unit is printed rather than assumed, because it is not always available: with no
oxidation assignment the magnitudes are unknown and the number falls back to the solver's
site count with **no** unit; with an assignment but no B-site grid (a loaded structure whose
magnetic sites do not form one) it is the whole structure's moment in `μB`.

An odd cell cannot compensate: 3×3×3 `G` has 27 sites and leaves one 5 μ<sub>B</sub> spin
over, so it reads ±0.185 rather than 0.

The configuration you have selected persists too. The landscape is re-sorted by energy
whenever it is re-evaluated, so an edit moves a given arrangement up or down the list —
the selection follows the **arrangement**, not its position, so the same spins stay on
screen with their new energy. A change that alters the number of magnetic sites has
nothing to hold on to, and falls back to the ground state; so does a fresh solve, which
deliberately presents the state it just found.

Those points **persist as you edit the structure**, and are re-evaluated as you go:
**Live energies** (with the site toggles in Calculation Output) is on by default, so every
edit rebuilds the exchange matrix and re-scores every plotted configuration against it and
the plot tracks the structure continuously. New configurations come only from **Run
Magnetic Structure**, which fills in the rest of the landscape.

Re-evaluating costs tens to hundreds of milliseconds on a large cell, which on a big enough
structure is the whole frame. Rather than making that your problem, the app watches the
frame rate and **pauses live updates below `AUTO_SPIN_UPDATE_MIN_FPS` (20 fps)**, resuming
once the view is back above `AUTO_SPIN_UPDATE_RESUME_FPS` (30). While it is paused the
checkbox says so, and edits fall back to marking the energies **stale** — the plot holds its
last values until you press **Refresh energies** or run a solve.

The two thresholds are what keep it from flapping. With a single one, pausing would free
exactly the time that pushed the rate under it, the next frame would clear the bar, and the
landscape would rebuild every other frame — worse than either state. ImGui's frame rate is
smoothed over ~60 frames, so the gate responds over about a second rather than to one slow
frame. Unticking the checkbox switches live updates off regardless of the frame rate.

**Color atoms by spin**, at the top of Calculation Output, draws magnetic atoms in the
spin-up (turquoise) and spin-down (yellow) colors instead of their element colors. It is
off by default, so the 3D view stays element-colored until you ask for the spins.

It also sizes each sphere by the moment the site carries. The size is measured against a
fixed reference of 5 μB, so a high-spin Fe(3+) draws at its full element radius and a
spin-1 ion at a fifth of it, and a given moment is the same size in every structure. The
magnitudes come from the oxidation-state assignment — the same place export takes them
from — because the solver's own moments are unit-magnitude directions, with the size of
each spin already folded into the exchange couplings. Sites carrying no moment (O, La, and
any magnetic site this particular configuration leaves unpolarized) shrink to nothing and
drop out of the view, which is what leaves the magnetic sublattice on its own. With no
assignment solved yet there are no magnitudes to use, and the spheres keep their element
radii.

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

This is a separate overlay from the **defect planes** of the builder's
[Defects & impurities](tutorial/user-interface.md#defects) panel: ordering planes come from
the selected spin configuration and are tinted by spin, while defect planes come from the
builder. Both can be on at once, but they draw different things: an ordering needs the
whole family, because the alternation between its planes *is* the ordering, whereas a
defect overlay draws only the layer being worked in — as many sheets as that layer has
positions in the cell, and none anywhere else — drawing the layer's own atoms fully
opaque and fading everything off it to faint translucent context.

Defect planes appear whenever the builder's **Defects & impurities** menu is expanded, and
disappear when it is collapsed; they have no switch of their own. The exchange-coupling
view shares the 3D view with them by recency, and only ever one at a time: selecting an
atom's couplings (in the plot or the 3D view) takes the view over, clicking anywhere in
the defects menu takes it back, and clearing the coupling selection drops to the plain
structure until one of them is touched again. Dialling a plane — the
`h k l` steppers and the layer slider at the top of the menu — *is* selecting it: whatever
the dial points at is the plane drawn, faded around, and picked in. The sheet is
kind-neutral — a plane names a place, and the defects on it can mix kinds. The kinds show
as rings on the defected sites instead: green for a substitution, amber for a proton,
fuchsia for a vacancy (which is also how a substitution with no element yet is built), and
a white ring on top marks the entry selected in the panel.

While the menu is open the 3D view takes over the left mouse button, which is otherwise
unused. Clicking an undefected atom places a new defect there, stamped from the mode
widgets beside the dial, and records the dialled plane as the defect's own; clicking a
defected atom selects its entry — which also dials that plane back up — and clicking it
again while it is the selection removes it. The boundary layer **Render periodic images**
draws puts a corner site on screen up to eight times; every copy answers to a click, and
hovering any of them rings all of them, so they read as the one site they are.

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

## The exchange couplings

The pane's second plot, chosen from the dropdown in its top-left corner, is the exchange
matrix drawn out as a bar chart: one bar per coupled pair of magnetic sites, in meV,
sorted by magnitude so the couplings that decide the ordering are on the left. Like the
landscape it is seeded before you run anything, since the exchange matrix is built as
soon as a structure is focused.

Bars are grouped into one series per pairing of metal elements — `Fe - Fe`, `Fe - Mn` — so
the legend names the chemistry and the colours separate it. The colours are blended from
the same per-element colours the 3D view uses.

The sign is the **model's** convention, `J > 0` antiferromagnetic, matching
`build_Jeff_matrix` and `docs/theory/magnetism-model.md`. The solver runs on `-J` (see
`to_solver_couplings`), so the bars are the negative of what the spin solver minimises.
Hovering a bar names the pair and gives its coupling, the metal–metal distance, and how
many ligands bridge it at what mean M–L–M angle — a pair that shares an edge or a face has
more than one bridge, and their contributions are summed into the one bar.

Very large cells are capped at 200 bars, and the pair names drop off the axis past 30 bars
— the hover tooltip still names every one.

The y range always spans zero — which side of it a bar falls on is the whole point — and is
otherwise fitted to the data, so an all-AFM structure does not spend half the pane on an
empty FM half. Below the data it uses `EXCHANGE_BOTTOM_HEADROOM` rather than the pane's usual
`TWO_D_BOTTOM_HEADROOM`: every bar stands on `y = 0`, so the default margin puts the whole
row of feet on the axis line and a short bar cannot be told from no bar. Above the data the
margin stays `TWO_D_TOP_HEADROOM`, which is what keeps the corner pickers off the tallest bar.

Bar order is established when an atom is selected and held until the couplings are rebuilt,
so bars keep their positions rather than resorting each frame. Ties matter here: a cubic
cell's six neighbours are physically identical, but summing their bridges in different
orders leaves them differing in the last two bits of a float. The order therefore compares
magnitudes quantised to `EXCHANGE_TIE_DECIMALS` places of meV and breaks the resulting ties
on the atom indices, which is what keeps the axis labels in a readable sequence.

### Frustration is the configuration-dependent part

`J` does not depend on the spin configuration, so the bars themselves do not change when you
pick a different one. What changes is which couplings that configuration satisfies. A yellow
ring on a bar's tip marks a *frustrated* coupling — one where `J_ij (m_i · m_j) > 0`, so the
model would rather those two spins were the other way round.

Summed over every pair, that quantity is exactly the configuration's model energy, which is
the identity `test_pair_contributions_sum_to_the_configuration_energy` pins. The count rises
monotonically up the landscape: for cubic LaFeO3, 27 of 81 in the G-type ground state, 45 in
the C-type orderings, 63 in the A-type, and 81 in F.

### Selecting an atom

With this plot showing, clicking an atom in the 3D view narrows it to that atom's couplings.
Two ways back out: click the atom again, or press **`<- All couplings`**, which sits on the
header line above the plot, right-aligned. That is outside the plot rather than floating in a
corner of it, unlike the plot picker — it leaves the atom's couplings for all of them, which
is a step out of the view rather than a control over what the view draws.

Only magnetic sites are clickable, and only while this plot is up — the gate is
`exchange_selection_site`, since the decoration has nowhere to be read against the energy
landscape. Hovering names the ion rather than the atom index
(`Fe(3+) — 6 couplings`): the index says nothing you cannot already see, while the oxidation
state is what sets the d-shell and therefore the couplings.

While an atom is selected, the pick candidates narrow to that atom and the atoms it couples
to (`exchange_pick_candidates`, read off the ends of the drawn paths). Everything else is
faded and inert, so a click can only ever walk along a bond that is on screen rather than
jumping to an atom whose couplings share nothing with what is being read. The bridging
ligands are drawn prominently but are not targets — they carry no couplings, and selecting
one would empty the plot. Clicking a bar does the same walk from the other side.

### Exchange paths in the 3D view

Selecting an atom draws its superexchange network as it physically runs — metal, bridging
ligand, metal — one polyline per bridge, so an edge- or face-sharing pair shows each of its
pathways. Opacity is linear in `|J|` relative to the strongest path on screen (floored at
`EXCHANGE_PATH_MIN_ALPHA` so the weakest is still visible), which leaves strength in the one
channel the eye reads as "more".

Colour carries frustration, and nothing else: yellow where the selected configuration fights
the coupling, white where it satisfies it — the same `EXCHANGE_FRUSTRATED_COLOR` the bars are
ringed in. A-type ordering is the clearest read of this: the four in-plane bonds of a B site
come out yellow and the two along the stacking axis white, which is what FM-within-(001),
AFM-between means against an antiferromagnetic `J`. The two colours are submitted under
separate labels so the legend can name them.

The geometry comes off `BridgeGeometry` rather than being re-derived: `u_iL` and `u_jL` point
outward from the ligand, so stepping `r_iL` and `r_jL` back along them recovers the two
metals' *image* positions and a path across a cell boundary comes out contiguous. A path that
runs off the edge of the drawn cell is a coupling to a periodic image whose partner is not in
the render.

Atoms are faded unless a drawn path passes through them, and the octahedra fade with them.
Prominence is read off the paths, not off the coupled sites — taking the sites and lighting
up every periodic image of each lights up images on the far side of the cell that no path
reaches, which reads as a coupling that is not there.

Hovering a path gives the same tooltip as its bar. The atom pick takes priority: paths run
between atoms and pass under them, and a tooltip describing something other than what a click
would act on would be describing the wrong thing.

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
