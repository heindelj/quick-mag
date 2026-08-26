# User Interface
This tutorial can be followed via the (quick-mag web app)[https://heindelj.github.io/quick-mag/app/] which can run in your browser.

## 0. Controls

## 3D Panel
In the center of the screen, you should see a default structure which is the LaFeO3 perovskite.
Controls:
- Right click and drag to rotate the structure
- Click on entries in the legend to make them invisible

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
TODO: