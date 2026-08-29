# User Interface
This tutorial can be followed via the (quick-mag web app)[https://heindelj.github.io/quick-mag/app/] which can run in your browser.

## 0. Controls

## 3D Panel
In the center of the screen, you should see a default structure which is the LaFeO3 perovskite.
Controls:
- Right click and drag to rotate the structure. The structure follows the cursor from any
  orientation, so there is no view you can get stuck in.
- Double right click to go back to the starting view
- Scroll to zoom
- Click the small `a`, `b` and `c` buttons above the panel to look straight down that
  lattice vector. Looking down `a` or `b` puts `c` up the screen; looking down `c` gives
  the usual ab-plane view with `a` to the right. Click the same button again to turn the
  cell around and look at the opposite face, so both sides of a plane are one click apart.
- The `-5°` and `+5°` buttons turn the view in 5° steps about an axis fixed in screen
  space, chosen with the `x` / `y` / `z` buttons on the second row. `x` points right, `y`
  up and `z` into the screen, and exactly one is selected at a time. Each turns the
  right-handed way about the axis as you see it, so `+5°` about `z` spins the picture
  clockwise and `+5°` about `x` brings the top of the cell towards you.
- `z` is selected to start with. That turn is the one dragging cannot make at all, and it
  leaves the face you are looking at alone, so it composes with the alignment buttons
  rather than undoing them.
- Click on entries in the legend to make them invisible

In the bottom-left corner of the panel is a small `a`/`b`/`c` triad showing how the cell is
currently oriented. It draws the real lattice vectors, so on a non-cubic cell in Cartesian
coordinates it does not line up with the box's own axis labels. An axis pointing straight at
you shrinks to a dot -- which is how you can tell an alignment button has landed.

**Hover any atom** to see what it is: its number, element, oxidation state, and magnetic
moment (`15. La  ox=+3  m=(+0.00, +0.00, +0.00)`). This works for every atom, not only the
magnetic ones. It is the same per-site information that used to be a long scrolling list in
the results panel, read off the atom itself so there is no matching a row against the
picture by eye.

## 2D Panel
Just beneath the 3D panel is a plot with a dropdown floating in its top-left corner,
choosing which of the two plots you are looking at.

Which configuration is drawn in the 3D view is chosen from the `Spin configurations` list
in the Calculation Output panel on the right, or by clicking a point in the spin-energy
plot. Those two are the same selection, so they always agree.

**Drag the rule between the two panels** -- the one with the small grip in the middle -- to
give either plot more room. Neither can be squeezed below a readable minimum, and the split
is stored as a share of the pane, so it holds when the window is resized.

### Spin energies
The default plot. Each point is one spin configuration, ranked left to right by energy,
with the y axis showing how far above the ground state it sits in eV. Points are coloured
by which ordering the configuration actually is -- F, G, and the three orientations each of
A and C, plus any ordering you have entered by hand -- with anything that is not a reference
ordering in grey. Click a point to select that configuration; the white ring marks the
selected one and a yellow ring follows the cursor.

Hand-entered orderings are classified on the same footing as the canonical ones, so a
configuration is reported under the name it was added as rather than as the nearest classical
ordering. They take their own colours, after the canonical entries in the legend. Where an
ordering you enter turns out to be one the cell cannot distinguish from a canonical one, the
classical name wins -- the same name the `Custom ordering` box reports it under.

Before you run `Magnetic Structure`, the plot already shows the canonical reference
orderings at their current energies, so it responds to builder edits immediately.

Re-scoring on every edit is not free -- it rebuilds the oxidation assignments and the
exchange matrix, which costs tens to hundreds of milliseconds on a large cell. `Live
energies`, with the site toggles in the Calculation Output panel, is on by default and
**pauses itself below 20 fps**, resuming once the view is back above 30. While it is paused
the checkbox says so and edits mark the energies stale, and the plot holds its last values
until you press `Refresh energies` or run a solve. So on a cell small enough to keep up, the
landscape tracks a slider drag; on one that cannot, dragging stays responsive instead.

### Exchange couplings
The second plot is a bar chart of the exchange constants that produce those energies: one
bar per coupled pair of magnetic sites, in meV, sorted strongest first. Bars are coloured
and grouped in the legend by the pair of metals being coupled (`Fe - Fe`, `Fe - Mn`, and so
on), so on a mixed B site you can see at a glance which pairing dominates.

The sign convention is the model's own, `J > 0` antiferromagnetic: bars above the zero line
favour antiparallel neighbours, bars below it favour parallel ones. (This is the opposite
of the sign the solver works in internally -- see `docs/theory/magnetism-model.md`.) The
y axis starts a little below zero so that the feet of the bars are visible rather than
sitting on the axis line, which matters when a short bar has to be told from no bar at all.

Hover a bar for the pair it belongs to, its coupling in meV, the metal-metal distance, and
how many ligands bridge it at what average M-L-M angle. Pair names appear as axis labels
too, as long as there are few enough bars for them to be readable.

Like the spin energies, this needs no solve -- the couplings are built as soon as a
structure is focused.

Bar order is fixed when you select an atom and holds until the couplings are rebuilt, so
bars do not swap places under the cursor. Symmetry-equivalent couplings are exactly equal,
so ties are broken on the atom indices and the order stays readable rather than arbitrary.

### Which couplings the ordering fights
`J` itself does not depend on the spin configuration -- the same bars are there whichever
one you pick. What does depend on it is whether each coupling gets what it wants, and that
is what the **yellow rings** mark: a ring on a bar's tip means that coupling is *frustrated*
in the selected configuration, holding its two spins the way it would rather they were not.

Pick a different configuration in the results list and the rings move. Going up the energy
landscape, more of them appear -- for cubic LaFeO3, 27 of the 81 couplings are frustrated in
the G-type ground state, 45 in the C-type orderings, and all 81 in ferromagnetic F, which is
exactly why F sits at the top. Summed over every pair, the frustration *is* the
configuration's energy.

### Couplings for one atom
With the exchange plot open, **click an atom in the 3D view** to narrow the plot to just
that atom's couplings. Hovering a magnetic site rings it and names the ion and how many
couplings it has (`Fe(3+) - 6 couplings`); clicking selects it, and clicking it again clears
the selection. Only magnetic sites are clickable, and only while this plot is showing --
there is nowhere to read the answer otherwise.

Selecting an atom changes the 3D view:

- The selected atom is circled.
- **The exchange paths are drawn**, running the way superexchange physically does: metal,
  to bridging ligand, to metal. A pair bridged by more than one ligand gets one line per
  bridge. The stronger the coupling, the more opaque its line, so the couplings that matter
  stand out from the ones that barely do.
- **A path is yellow where the current configuration frustrates it** and white where it is
  satisfied -- the same yellow the frustrated bars are ringed in. On A-type ordering you can
  see this directly: the four in-plane bonds come out yellow and the two out-of-plane ones
  white, which is exactly what "ferromagnetic within (001) planes, antiferromagnetic
  between them" means for an antiferromagnetic `J`.
- Everything not on one of those paths is drawn faded, and the octahedra fade with it, so the
  coupling network stands clear of the rest of the cell.

Hover a path for the same information the bar gives -- the pair, `J` in meV, the metal-metal
distance, the bridging ligand and its M-L-M angle, and whether the current configuration
satisfies it. Hovering an atom takes priority over the paths running under it, so the atom
you would click is always the one being described.

A path that runs off the edge of the drawn cell is a coupling to a periodic image; the
partner is the same site, wrapped around.

**While an atom is selected, only the atoms it couples to can be picked** -- everything else
is faded and inert. Clicking one walks the selection along that bond, so you move through the
exchange network rather than jumping to an atom whose couplings share nothing with what is on
screen. Clicking a bar in the plot does the same walk, from the other side.

To get back to every coupling, press **`<- All couplings`** at the top right of the pane,
just above the plot, or click the selected atom itself again.

## Reading the spin configurations
The Calculation Output panel lists every configuration in the landscape, cheapest first.
Click a row to put its spins in the 3D view; the same click is available on a point in the
energy plot. Directly under the list is what the selected row *is* -- its energy, its net
moment, the ordering it matches and how far off it is, and how many configurations share
its energy -- and under that, `Save magnetic configuration`, which records the ordering and
its defect concentration onto the structure.

`M` is a net moment **per perovskite cell**, in μB. Two things make that not the same as
the number the solver hands back. The solver works in unit +-1 moments, because the size
of a spin is already inside the exchange couplings, so its own figure counts *sites*; the
real magnitudes come from the oxidation-state assignment. And it is divided by the number
of cells, so it does not grow with the supercell -- ferromagnetic LaFeO3 reads
`5.000 μB/cell` on a 2x2x2, a 3x3x3 and a 4x4x4 alike, which is the high-spin moment of
Fe(3+). An odd cell cannot fully compensate: 3x3x3 `G` has 27 sites and leaves one spin
over, so it reads `0.185`, not zero.

The unit is printed rather than assumed, because it is not always there to be had. With no
oxidation-state assignment there are no magnitudes and the figure falls back to the
solver's own with no unit at all; on a structure whose magnetic sites do not form a cell
grid there is nothing to divide by and it is the whole structure's moment, in `μB`.

## Defining your own ordering
At the foot of the Calculation Output panel, below the spin configurations, is a
`Custom ordering`
section. Every canonical ordering is a **plane family plus a string of signs repeated across
successive planes** -- `G` is "flip on each (111) plane", `A(c)` is the same on (001), `F` is
a single plane taking a single sign -- so that is what you enter:

- `Plane (hkl)`: the three Miller indices of the plane family.
- `Signs`: the pattern to repeat, as `+` and `-`. Two characters gives an alternating
  ordering; longer strings give longer periods, up to eight.

Press `Add ordering` and it is scored against the current exchange matrix, ranked into the
landscape, classified, and selected -- on exactly the same footing as the built-in orderings,
including in the energy plot, which reports and colours it under its own name.
It can be drawn, saved and exported like any other. Because it is stored as a *pattern*
rather than as a set of moments, it is rescored whenever the structure changes instead of
going stale.

Next to the button is how many planes of that family the cell actually spans. A pattern needs
one plane per character, so a `++--` on a family with only three planes is refused rather
than silently folded back onto a shorter ordering and scored as something it is not.

Two other things it will tell you rather than doing quietly:

- Re-entering a canonical ordering is refused by name -- `(111) +-` reports that it is `G`.
- An ordering the cell cannot distinguish from one already listed is added, but reported as
  such: on a 3x3x3 cubic cell `(123) +-` puts the spins in exactly the pattern `C(b)` does,
  so the list cannot show both and the message says which one it turned out to be.

The `x` button beside a listed ordering removes it.

## Builder Panel
In the top left is the builder panel which allows for creating various types of perovskites. In the formula dropdown menu, the available structure are ABX3, A2BB'X6, AA'3B4X12, AA'3BB'X12, and high-entropy perovskites.

### Atoms
Under the `Atoms` menu, you can change which atoms sit in each site of the perovskite structure. The available fields will change depending on which type of perovskite you have selected.

### Lattice
If you are not familiar with perovskite structures, it could be helpful to click on the `Lattice` dropdown menu and change the structure to a 1x1x1 cell. If you have selected single perovskite, then you will see a single cube.

This view should make it clear what an idealized perovskite structure is. We have a cube formed by the A site atoms, at the center of the cube is the B site, and the X sites sit at the center of each face of the cube. By default, we draw octahedra connecting the X sites which tends to make it a little easier to understand the structure. (You might notice that this renders more atoms than are actually in the ABX3 unit cell. That option, which only affects visualization, can be toggled via a checkbox under `Rendering` called `Render Periodic Images`.)

You can also change the length of the lattice constants under the `Lattice` menu. There are three options `Cubic`, `Tetragonal`, and `Orthorhombic`. These control which lattice constants are allowed to be changed independently.

The structure is treated as a periodic material by default or as a cluster if you uncheck `Treat structure as periodic`.

### Tilt System
One unique structural feature of perovskites is that the octahedra often distort their orientation from the idealized structure. There are many so-called tilt systems which can be described via Glazer notation. This is easiest to understand visually.

- Make sure you have at least a 2x2x2 cell
- Select `a0a0c+` from the dropdown menu
- Orient the camera so that you are looking down the `c`-axis
- Move the slider around for the tilt angle

Notice that the octahedra in each plane stacked along this axis stay aligned with one another. That is the meaning of `c+` in `a0a0c+`.

- Try changing the tilt system to `a0a0c-` and see what is different
- Make a structure with more than two planes along the c-axis. What does `c-` mean?

### Defects
Real perovskites are rarely perfect, and where a defect sits usually matters more than
which one it is. The `Defects & impurities` menu therefore does not ask you for a site
number — it asks you for a **plane**, and then you click the atoms you want inside it.

Opening the menu is what puts the 3D view into plane mode, and collapsing it is what gives
the plain structure back. There is no switch beyond that.

Click `+ Add defect plane`. A row appears with a defect kind (`Substitute` or
`Proton (H)`), three boxes for the Miller indices `h`, `k`, `l`, and a slider. Leave the
kind on `Substitute` and the indices on `0 0 1`.

- Orient the camera so you are looking down `a` or `b`, with `c` up the screen
- Drag the plane slider one notch at a time

One translucent sheet marks the layer you are working in, and it moves up the cell as you
drag. The rest of the family is not drawn — a sheet is there to say where the sites you can
click are. A layer can be marked in several places at once, though: the `AO` layer at the
bottom face of the cell is the same layer as the one at the top face, and the view draws
the closing boundary layer besides, so each place it appears gets a sheet. The caption on
the slider changes as you drag: `A + X`, then `B + X`, then `A + X` again. Those are the
two layers a perovskite is built from — an `AO` layer of A sites and apical oxygens, and a
`BO₂` layer of B sites and equatorial oxygens. The planes step by *half* a cube edge, which
is what makes both of them reachable; a whole-cell step would only ever land on one of them.

Everything off the active plane — atoms and octahedra alike — fades back to a haze. An
octahedron counts as being in the plane when the plane runs through its centre, so on an
`AO` layer, which holds no B sites at all, every cage fades and the layer stands out
against a ghost framework.

**Click an atom in the plane to make it a defect.** Click it again to undo. Hovering rings
the atom under the cursor and says whether a click will pick or unpick it. A site at the
edge of the cell is drawn several times over — its periodic images — and all of its copies
ring together, because a click on any of them does the same thing. Only atoms in the active
plane respond, and the view still orbits on the *right* mouse button, so turning the cell
around never picks anything by accident.

Picked sites collect under `Selected sites` on the row, which opens and closes like any
other menu. Each one has an `x` to remove it and a box for the element that replaces it.

**An empty box is a vacancy.** That is the only way to make one: there is no separate
vacancy kind, because a substitution with nothing substituted already is one. A picked site
starts empty and marked `(vacancy)`, and the atom disappears from the view — the clearest
possible sign the click landed. Type `Sr` and it becomes a substitution; clear the box again
and the site empties again. A vacancy leaves a marker behind, and that marker is clickable
too, so you can always take one back.

The element box takes anything. A symbol no element table knows is marked `(?)` and
otherwise left alone — a placeholder species is a legitimate thing to be building with, and
the builder has always accepted one. The box is per site, so you can put `Sr` on one and
`Ca` on another in the same plane.

- Moving the slider away does **not** discard what you picked. Those sites are defects now,
  not a highlight. The line under the list tells you how many are on other planes
- With several planes in the list, the one marked `<- shown` is the one on screen and the
  one clicks go to. Click anywhere on another entry to move to its plane

Try other directions. `1 1 1` alternates between `AO₃` layers and planes of bare B sites;
`1 1 0` alternates between `ABO` and `O₂`. The sheets are hexagons for `(111)`, and however
you slice it, they always pass *through* atoms rather than between them — every atom you can
click has a sheet on it, and no sheet is drawn anywhere you cannot click.

The slider only stops on layers the kind can actually use. A `Proton (H)` plane goes on an
oxygen, so on `1 1 1` it offers the `AO₃` layers and skips the bare B ones — and draws no
sheet on them either.

If a substitution unbalances the charge, a message appears below the table offering to add
compensating protons; they arrive as a new proton plane you can edit or delete like any
other.

What actually gets stored is the site's address in the lattice, not its position in any
list. Grow the supercell to 3×3×3 and shrink it back: the defects come back exactly where
you left them. A plane that does not exist in a smaller cell is skipped with a note rather
than being quietly moved onto a different layer.

## Structure summary
A small box starts in the bottom right corner of the 3D panel, titled with the name of the
active structure. Inside it are the formula and whether the cell is `periodic` or a
`cluster`, the cell constants, the composition of each site role, and the tilt system. It
reports what the structure actually *contains* rather than what the ideal lattice would
hold — defects are applied after the build, so where the two differ it shows the ideal
alongside, dimmed.

Drag it by its title bar to move it, and click the triangle to roll it up to just the
structure name when it is in the way. It holds a fixed width, so dragging a tilt slider
does not make it twitch as the numbers change. It starts back in the corner each time the
app opens.
