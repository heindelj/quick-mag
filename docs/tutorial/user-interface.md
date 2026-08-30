# User Interface
This tutorial can be followed via the [quick-mag web app](https://heindelj.github.io/quick-mag/app/) which runs in your browser.

## 3D Panel
In the center of the screen, you will see the default structure which is the $\mathrm{LaFeO_{3}}$ perovskite.

- Right click and drag to rotate the structure
- Double right click to go back to the starting view
- Scroll to zoom
- Click the small `a`, `b` and `c` buttons above the panel to look straight down that
  lattice vector. Click the same button again to turn the
  cell around and look at the opposite face.
- The `-5°` and `+5°` buttons turn the view in `5°` steps about an axis fixed in screen
  space, chosen with the `x` / `y` / `z` buttons on the second row.
- Click on entries in the legend to make them invisible
- Hover any atom to see what is, its assigned oxidation state, and assigned magnetic moment

## Builder Panel
In the top left is the builder panel which allows for creating various types of perovskites. In the formula dropdown menu, the available structure are $\mathrm{ABX_3}$, $\mathrm{A_2BB'X_6}$, $\mathrm{AA'_3B_4X_{12}}$, $\mathrm{AA'_3BB'X_{12}}$, and high-entropy perovskites.

### Atoms
Under the `Atoms` menu, you can change which atoms sit in each site of the perovskite structure. The available fields will change depending on which type of perovskite you have selected.

![screenshot of atoms menu](images/atoms.png)

### Lattice
![screenshot of lattice menu](images/lattice.png)

If you are not familiar with perovskite structures, it could be helpful to click on the `Lattice` dropdown menu and change the structure to a 1x1x1 cell. If you have selected single perovskite, then you will see a single cube.

![1x1x1 Perovskite structure](images/perovskite_111.png)

This view should make it clear what an idealized perovskite structure is. We have a cube formed by the `A site` atoms, at the center of the cube is the `B site`, and the `X sites` are the faces of the cube. By default, octahedra are drawn connecting the X sites which tends to make it a little easier to understand the structure. (You might notice that this renders more atoms than are actually in the ABX3 unit cell. That option, which only affects visualization, can be toggled via a checkbox called `Render Periodic Images` in the `Rendering` tab on the right side of the window.)

You can also change the length of the lattice constants in the `Lattice` menu. There are three options `Cubic`, `Tetragonal`, and `Orthorhombic`. These control which lattice constants are allowed to change independently.

The structure is treated as a periodic material by default or as a cluster if you uncheck `Treat structure as periodic`.

### Tilt System
One unique structural feature of perovskites is that the octahedra often distort their orientation from the idealized structure. There are many so-called tilt systems which can be described via Glazer notation. This is easiest to understand visually.

- Make sure you have at least a 2x2x2 cell
- Select `a0a0c+` from the dropdown menu
- Orient the camera so that you are looking down the `c`-axis (pressing the `c` button at the top of the 3D panel will do this)
- Move the slider around for the tilt angle

Notice that the octahedra in each plane stacked along this axis stay aligned with one another. That is the meaning of `c+` in `a0a0c+`.

![Simple tilt system example](images/tilt_system.png)

- Try changing the tilt system to `a0a0c-` and see what is different. What does `c-` mean?

### Defects

No material is perfect. Vacancies, substitutions, and charge-compensating protons can be included in structures from the `Defects & impurities` menu. Planes and defects are specified separately: first dial in *where*, then click *what*.

- **Pick a plane.** The `h k l` steppers at the top of the menu select a Miller family (arrows or typing both work), and the slider walks through its layers. The plane index steps half a cube edge, so `(001)` alternates between $\mathrm{AX}$ and $\mathrm{BX_2}$ layers, and the slider's caption says which sublattices the current layer cuts. The chosen plane is drawn in the 3D view, with its atoms visible and everything else faded.
- **Click atoms to place defects.** Clicking an atom in the selected plane adds a defect of the chosen kind (`Substitute` or `Proton (H)`). A subsitution with an empty element box empty makes the site a *vacancy*, drawn as a fuchsia marker. A proton only attaches to an oxygen.
- **Defect list.** As you add defects, they populate the `All defects` list at the bottom of the menu, each row leading with the plane it was specified in. Selecting a row (or clicking a defected atom) brings that plane back up and allows for editing that defect. Clicking the atom again while selected removes it, as does the row's `x` button.

If the substitutions leave the cell charged, a line under the defect list says how far off the stoichiometric balance the cell is and how many protons would compensate.

See if you can create this structure by making two $\mathrm{Ni}$ substitutions in $\mathrm{LaFeO_{3}}$ and adding an oxygen vacancy.

![Two defects and vacancy in structure](images/two_defects_and_vacancy.png)

## Structure Panel
In the top right is the list of structures you have built or loaded.
![structure panel](images/structure_panel.png)

- **Select:** Left click on an entry to make it the active structure
- **Rename or Delete:** Right click on an entry to change its name or delete that structure
- **Loading:** Structures in `.cif` or `POSCAR` format that are uploaded with the `Load Structure...` button appear here. Note that loaded structures disable most builder-related functionality but still allow for using the exchange model.
- **Exporting:** Structures can be exported by pressing `Export active structure` or `Export all structures`. On the web, this will download a .zip containing `.cif` files (and any associated spin configurations). On desktop, you can choose the location of the downloaded files.

## 2D Panel
Just beneath the 3D panel is the 2D panel which shows information about the predicted
magnetic properties of the material in question. There are two plots:

![2d panel with plot of spin energies](images/spin_energies.png)

### Spin energies
 - **Visualization of Spins:** You can visualize spin orderings more easily using the rendering options on the right. The below settings tend to be the most clear when you are looking at spins.
 
 ![spin rendering settings](images/spin_rendering_settings.png)
 
 - **Scatter Plot:** Each point corresponds to a particular magnetic configuration
 - **Selection:** Configurations can be selected by clicking on the plot or from the scrollable box in the bottom right panel
 - **Saving a Magnetic Configuration:** Magnetic configurations can be associated with a particular structure by selecting it, and then pressing the `Save magnetic configuration` button. The active structure in the `Structure Panel` can now be expanded and all saved magnetic configurations can be selected. When a structure is exported, you will also get a text file with all saved magnetic configurations.


### Exchange couplings
 - A second plot, which shows the computed exchange couplings, can be selected from the dropdown menu in the 2D plotting window
 - This bar chart shows all predicted exchange couplings between magnetic atoms

![2d panel with plot of exchange couplings](images/exchange_couplings.png)

 - Clicking on a bar or on an atom in the 3D view shows the nonzero exchange couplings for that atom

 ![2d panel with plot of exchange couplings for specific atom](images/exchange_couplings_by_atom.png)

 - Pairs of spins which are frustrated are highlighted with a yellow circle and connected by a yellow line in the 3D window
 - When the exchange couplings involving a particular atom are selected, an `All couplings` button appears in the top-right
 of the 2D pane and restores the original bar chart. Alternatively, you may click the relevant atom in the 3D view to deselect it.
 - While an atom is selected, only the atoms it interacts with magnetically can be selected
 - Changing the plot type or pressing `All couplings` restores the default 3D view
