# Defects and impurities

The builder models three kinds of point defect — **vacancies** at any site,
**substitutions** of one element for another, and **protons** added to compensate
charge. This page covers how they are addressed, why they never interfere with editing
the geometry, and where the proton actually goes.

## The idealized structure stays the source of truth

`build_perovskite` always emits a complete, perfectly ordered lattice, and the builder
regenerates the whole structure from its parameters on every edit. Defects are kept out
of that generator entirely and applied as a **post-pass** afterwards:

```
builder parameters ──► build_perovskite ──► ideal (A, B, X) lattice
                                                   │
                             formula mode ──► atomic labels
                                                   │
                            params.defects ──► apply_defects
                                                   │
                                            ChemicalStructure
```

This is why `PerovskiteGenerationParameters.build_kwargs()` does not mention defects:
replaying it reproduces the *ideal* build, and the defect list is layered on top. Change
a tilt angle and the octahedra rotate as ideal rigid units, then the same defects are
re-applied to the fresh geometry. Nothing is frozen and nothing accumulates.

The other half of the trick is that a defect names its target by **grid address**, not by
array index — indices shift when the supercell is resized, grid addresses do not. The UI
picks a site with a slider over the sites of that role, but the slider is only an
*ordering*: what gets stored is the address, so resizing renumbers the sliders without
moving any defect.

## Site keys

| Role | Key | Range |
|---|---|---|
| A | `("A", i, j, k)` | `(nx, ny, nz)` periodic, one more per axis for a finite cluster |
| B | `("B", i, j, k)` | `(nx, ny, nz)`, the octahedron grid |
| X | `("X", i, j, k, v)` | cell `(i,j,k)`, `v` a vertex row of `[+a, -a, +b, -b, +c, -c]` |

`canonical_site_keys(grid_shape, periodic)` produces these in exactly the order
`build_perovskite` stacks its sites, and the two `index_deduplicated_*` functions
generate their coordinates by walking that same list — so key *n* always addresses
`build.all_sites[n]` and the two cannot drift apart. `role_site_keys` filters that list
to one role, which is a convenient way to script a defect list against a whole
sublattice.

## Finding a site: lattice planes

A grid address is a good way to *store* a defect and a poor way to *pick* one — "the 37th
oxygen" is not a fact about the crystal. The UI therefore addresses a site by the plane it
lies in: choose a Miller family `(hkl)`, step along it, and check the sites the plane cuts.

The trick that makes this work is the **doubled cube coordinate**. Measured from the cell
origin in units of the cube edge, `build_perovskite` puts

| Role | Cube coordinate | Doubled |
|---|---|---|
| A | `(i, j, k)` | `(2i, 2j, 2k)` |
| B | `(i+½, j+½, k+½)` | `(2i+1, 2j+1, 2k+1)` |
| X | B ± ½ on axis `v // 2` | B ± 1 on axis `v // 2` |

Doubling makes every site an integer point, so a plane family is exactly
`h·u + k·v + l·w == m` over the integers — no tolerance and no binning, and two sites share
a plane precisely when `plane_index_of_key` returns the same number for both.

Consecutive `m` step **half** a cube edge, and that is the whole point. A whole-cell step
reaches one sublattice only; the half step alternates, so a family cuts the A sites, the B
sites and the oxygens in turn:

| Family | Layers it alternates between |
|---|---|
| `(001)` | `AO` (A sites + apical O) and `BO₂` (B sites + equatorial O) |
| `(111)` | `AO₃` and `B` |
| `(110)` | `ABO` and `O₂` |

`occupied_planes` lists the indices that hold at least one site, which bounds the plane
slider — it can only ever name a plane the lattice actually has. `sites_in_plane` then lists that plane's members, in build
order.

### Folding periodic aliases

Under periodic boundaries a plane and the plane one cell along the normal are the same
layer, and the canonical key set names both: with only the `+a`/`+b`/`+c` vertex rows kept,
the apical oxygen of the top cell lands a whole cell above the A plane it actually shares.
`plane_period` returns the gcd of the three supercell shifts `2·n·h`, and folding by it puts
each layer back together. A finite build has no such aliasing — its closing layer is a
genuinely separate set of atoms — so `plane_period` returns 0 and nothing is folded.

Only the layer being worked in is drawn, and the sheets are placed from the *pick targets
themselves* — the atoms and vacancy markers a click can actually land on, in the structure
being rendered. Deriving them from the ideal key set instead gets two things wrong: folding
an index does not merge the positions it covers, so a folded `(001)` layer holds the A sites
at `z = 0` and the apical oxygens at `z = 1` and needs a sheet at each; and the rendered
closing boundary layer carries copies of a site that can sit a whole cell along the normal
from any canonical one. Placing from the targets makes the rule symmetrical by construction:
every site you can pick has a sheet through it, and no sheet is drawn anywhere else.

Only the *keys* are folded, never the plane being asked for. Folding the request too would
quietly relocate a plane authored in a larger supercell onto whichever layer it happens to
be congruent to in the smaller one — the same silent rewrite [wrapping is deliberately
narrow](#wrapping-is-deliberately-narrow) refuses to do for site addresses. An index with
no sites simply has none, and the panel says so.

### Aliasing

Octahedra share their corners, so an oxygen has two names: the `-a` vertex of cell `i`
is the same atom as the `+a` vertex of cell `i-1`. `canonicalize_key` folds every alias
onto one canonical representative, which lets you name an oxygen from whichever
octahedron you are looking at. The exception is the low face of a *finite* build, where
`("X", 0, j, k, 1)` has no cell to its left and is a site in its own right.

### Wrapping is deliberately narrow

A periodic lattice has no boundary, so it is tempting to reduce every index modulo the
grid. The builder does not, because then shrinking the supercell would silently move a
defect onto a *different* site rather than dropping it. Only the two folds that name the
**same atom** are allowed: the corner-shared oxygen alias above, and the closing A-site
plane that a finite build adds at `i == nx` (the image of the `i == 0` plane, so
toggling periodicity preserves meaning).

Everything else outside the grid is skipped with a warning and left in the defect list
untouched — shrink the cell and the defect goes dormant; grow it back and the defect
returns exactly as it was.

### Periodic images

The 3D view renders a periodic structure by rebuilding it non-periodically, which adds
the closing boundary layer. A corner A site gains up to 8 copies there and a
face-boundary oxygen 2. A vacancy has to remove all of them or the hole visibly fills
back in at the cell edge, so `resolve_key_to_indices` expands boundary images whenever a
*stored periodic* structure is rebuilt finite. A genuine finite cluster expands nothing:
vacating its corner A site removes one atom, as it should.

Keys are canonicalized against the *authoring* periodicity before images are expanded.
Resolving the finite way first would leave `X(0,0,0,-a)` as a low-face site with no
far-face partner, and the vacancy would refill at the cell edge.

## Proton placement

A proton trapped in an oxide sits on an oxygen at about 0.98 Å, pointing **away from the
cations it bridges** — so the O–H vector lies in the plane perpendicular to the B–O–B
axis. The builder takes that plane from the host octahedron's own two other axes, giving
exactly four candidate sites, selected with the `orientation` index.

### Why not point at the nearest oxygen

"Perpendicular to B–O–B" and "toward a neighbouring oxygen" sound like they agree. They
do not. In an ideal cubic perovskite the eight nearest oxygens to a given oxygen all sit
at **45° to the B–O–B axis**, and the perpendicular plane contains the A cations. Placing
H 0.98 Å along a nearest-oxygen direction puts it **1.48 Å from a B cation** — closer
than the O–H bond itself.

Measured nearest-contact distance for a proton on the `+a` oxygen (LaFeO₃, a = 4.2 Å):

| geometry | orientation 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `a0a0a0` (cubic) | 2.32 | 2.32 | 2.32 | 2.32 |
| `a-a-a-` @ 8° | 2.08 | 1.88 | 2.12 | 2.12 |
| `a-b+a-` @ 10° | 2.12 | 2.06 | 2.04 | 1.77 |
| *nearest-oxygen rule* | *1.57* | | | |

Taking the directions from the octahedron rather than from a neighbour search has a
second benefit: they rotate **rigidly with the cage**, so `orientation=2` names the same
physical site before and after a tilt edit. A neighbour-ranked ordering would silently
renumber itself whenever the geometry changed, which is exactly the reproducibility the
builder is supposed to guarantee. For the same reason, candidates are *not* re-sorted by
clash score.

The residual tightness under strong tilt (1.77 Å to an A cation) is an artefact of the
idealization — the builder leaves A cations at undisplaced positions, whereas a real
tilted perovskite displaces them. It is reported rather than corrected.

## Rendering

A vacancy has no atom to draw, so its marker position comes back from the ideal build the
defects were subtracted from (`vacancy_render_sites`), and its radius is copied from a
surviving atom of the same element (`vacancy_render_radii`) rather than recomputed — that
way the hole matches its neighbours even when atoms are being drawn at ionic radii from a
solved oxidation state, which a vacancy has no way to look up. Markers are drawn last so
they are never hidden behind a neighbouring atom, and boundary images are expanded on the
finite render so a corner vacancy does not appear to fill back in at the cell edge.

Markers are drawn in vivid fuchsia rather than white: white is hydrogen's own CPK colour,
and a protonated defect cell would then show two indistinguishable white spheres. No
element sits near that hue, so a hole can never be mistaken for an atom.

Octahedral cages are dropped only when the **B centre** is vacated. A cage that has merely
lost a vertex is still drawn: a five-coordinate cage with a magenta ball in the sixth
corner reads as a defect more clearly than a missing cage would.

## Charge compensation

Asking whether a defected cell "balances at zero" is not useful: it almost always does,
by pushing a cation off its normal valence. The question that matters is how far the cell
drifts from the *stoichiometric* charge balance, which is what `compensation_hint`
reports — holding every element at the charge it carries in the defect-free cell.

| defect | nominal | what happens |
|---|---|---|
| O vacancy | **+2** | absorbed by reducing two cations (`Fe(2+) ×2`) |
| Fe³⁺ → Zn²⁺ | **−1** | needs a proton; otherwise one Fe is promoted to Fe⁴⁺ |
| Fe³⁺ → Zn²⁺ + H⁺ | **0** | every iron stays 3+ |
| La³⁺ → Sr²⁺ | **−1** | needs a proton |

A positive nominal charge is absorbed by reduction and wants no proton. A negative one is
a cation deficit and does — which is when the UI offers the *Add compensating protons*
button.

No change to the oxidation-state model is needed for any of this. H is enumerable, and it
is in neither `TRANSITION_METALS` nor `ANION_SPECIES`, so an added proton is neither a
magnetic site nor a superexchange bridging ligand.

## Effect on the magnetic model

Most of the exchange machinery is distance-cutoff based and simply sees less:

- **Bridges** through a vacated oxygen disappear, so that superexchange path contributes
  nothing. A B site isolated by vacancies gets no couplings at all and its moment is
  unconstrained by the solver.
- **`local_octahedral_frame`** falls back to the global frame when fewer than two trans
  ligand pairs survive — its documented under-coordinated path.
- **The crystal-field integral** (`crystal_field_eg_orbital`) returns `None` only for a
  degenerate, perfectly ideal octahedron. A neighbouring vacancy breaks that degeneracy,
  so orbital order switches on where it physically should.

The B-site grid is the one place that needs a representation for holes: `grid_to_site`
stays a full lattice of `prod(b_grid_shape)` entries and uses `-1` for a vacated cell,
with `grid_present` as the matching mask. The classifier treats those cells as inactive
neighbours, which it already knew how to do for zero-moment sites, so a G-type ordering
still classifies as G with a B vacancy present.
