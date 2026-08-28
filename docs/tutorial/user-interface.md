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

## 2D Panel
Just beneath the 3D panel, you will see a scatter plot which shows the spin energies of various canonical spin configurations, according to the spin model described in ...

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
A small box floats over the top right of the 3D panel with the cell constants, the
composition of each site role, and the tilt system. It reports what the structure actually
*contains* rather than what the ideal lattice would hold — defects are applied after the
build, so where the two differ it shows the ideal alongside, dimmed. The box ignores the
mouse, so you can drag straight through it to rotate.
