# Examples: Predicting Magnetic Configurations

`quick-mag solve` takes a structure file (`.cif` in P1, or `.vasp`/POSCAR) and runs the
full pipeline: oxidation states → magnetic moments → exchange couplings → low-energy
collinear spin configurations → ordering classification. The
[Magnetism Model](theory/magnetism-model.md) page describes what each stage does.

The sample structure used below is available in the `assets/` folder. This structure is chosen as it is a Jahn-Teller distorted structure which results in an A-type magnetic ordering.

## A first solve, end to end

```bash
quick-mag solve assets/A_type/LaMnO3_222.cif --max-configs 5
```

```text
Structure: LaMnO3_222  (40 atoms)
  elements: La, Mn, O
  Top 1 oxidation distribution(s) of 29 will be solved.

================================================================================
Oxidation-state assignment:  8xLa+3 | 8xMn+3 | 24xO-2
  oxidation-model energy: -0.0497

  Reference magnetic configurations (scored independently of the solve):
      type          energy            ΔE  magnetization
      A(c)       -0.175020      0.000000          0.000
      C(a)       -0.061576      0.113443          0.000
      C(b)       -0.061471      0.113549          0.000
         F       -0.051973      0.123047          8.000
         G        0.051973      0.226992          0.000
      A(b)        0.061471      0.236490          0.000
      A(a)        0.061576      0.236596          0.000
      C(c)        0.175020      0.350039          0.000

  Solving 8 magnetic sites with method='exact'.
  ...
  Solved low-energy spin configurations:
      #          energy            ΔE  magnetization  n_unpaired    type
      1       -0.175020      0.000000          0.000        8.00       A
      2       -0.087519      0.087501         -2.000        8.00       A
      3       -0.087519      0.087501          2.000        8.00       A
      ...

  Ground state (solved + reference): E = -0.175020 (A)
```

Reading it:

- **Oxidation-state assignment** — the charge-balanced distribution chosen, and its
  oxidation-model energy. Here Mn³⁺, correct for LaMnO₃.
- **Reference configurations** — the canonical G/C/F/A orderings, always scored on the
  same coupling matrix and reported independently of the search. A and C single out one
  axis of the B-site grid, so each is scored in all three orientations: `A(c)` stacks its
  ferromagnetic planes along **c**, `C(a)` runs its ferromagnetic chains along **a**, and
  so on. G and F treat every axis alike and appear once. On a cubic cell the three
  orientations are degenerate; on this Jahn-Teller distorted one they are not, and only
  `A(c)` is the ordering LaMnO₃ actually adopts — a table that reported one arbitrary
  orientation of A could easily have shown the +0.06 state instead.
- **Solved configurations** — the ranked results from state enumeration or optimization, each classified F/A/C/G/E or blank if unclassifiable. `ΔE` is relative to the best magnetic configuration.
- **Ground state** — the lowest energy state over the search and reference configurations.

A-type is the experimentally known ordering of LaMnO₃, and the model finds it.

The energy scale depends on the fitted parameter set. The oxidation energies do not correspond to a physical unit and are better thought of as a score. The parameters of the exchange model are fit from DFT+U relative energies of various spin configurations in eV. The model aims to produce reasonable rankings rather than quantitatively accurate energies, so the energies should be taken with a grain of salt.

## The magnetic cell has to be big enough

An ordering can only appear if the cell can represent it. A 1×1×1 perovskite has a single
B site, so there is nothing to order:

```bash
quick-mag solve out/LaFeO.cif     # 5-atom cell
```

```text
      #          energy            ΔE  magnetization  n_unpaired    type
      1        0.186679      0.000000          1.000        1.00
```

One site, no classification, no reference table. Use at least 2×2×2 (8 B sites) — see
[Building Structures](examples-builder.md).

## Checking the known G-type set

The `assets/G_type/` structures all have G-type ground states experimentally. Each is
predicted correctly:

```bash
for f in assets/G_type/*.cif; do quick-mag solve "$f" --max-configs 1; done
```

| structure | assignment found | predicted |
|---|---|---|
| `LaFeO3_222.cif` | 8×Fe⁺³ \| 8×La⁺³ \| 24×O⁻² | **G** |
| `LaCrO3_222.cif` | 8×Cr⁺³ \| 8×La⁺³ \| 24×O⁻² | **G** |
| `CaMnO3_222.cif` | 8×Ca⁺² \| 8×Mn⁺⁴ \| 24×O⁻² | **G** |
| `BiFeO3.cif` | 8×Bi⁺³ \| 8×Fe⁺³ \| 24×O⁻² | **G** |
| `KMnF3_222.cif` | 24×F⁻¹ \| 8×K⁺¹ \| 8×Mn⁺² | **G** |
| `KNiF3_222.cif` | 24×F⁻¹ \| 8×K⁺¹ \| 8×Ni⁺² | **G** |

Note that the model has been parameters for oxides, sulfides, and fluorides but results will be most reliable for oxides.

## Exploring more than one oxidation state

By default only the single lowest-energy charge-balanced distribution is solved.
`--top-k N` solves the N lowest, each with its own reference table and search:

```bash
quick-mag solve assets/G_type/CaMnO3_222.cif --top-k 3 --max-configs 1
```

```text
Oxidation-state assignment:  8xCa+2 | 8xMn+4 | 24xO-2      -> G
Oxidation-state assignment:  8xCa+2 | 4xMn+3 + 4xMn+5 | 24xO-2   -> G
Oxidation-state assignment:  8xCa+2 | 6xMn+3 + 2xMn+7 | 24xO-2   -> Other
```

Useful when the oxidation state is ambiguous or to confirm that the magnetic
prediction is robust against the assignment. Raise it when the top assignment looks
chemically wrong.

## Controlling mixed valence

`--max-mixing` caps how many distinct oxidation states one element may take. The default
is 2, which permits mixed valence. Force a single valence per element with `--max-mixing 1`:

```bash
quick-mag solve assets/G_type/CaMnO3_222.cif --max-mixing 1
```

```text
  Top 1 oxidation distribution(s) of 1 will be solved.
Oxidation-state assignment:  8xCa+2 | 8xMn+4 | 24xO-2
```

Only one distribution survives, versus 4 at the default. Mixed valence also gates the
double-exchange term in the model — it only switches on for element pairs that charge
balance forces into mixed valence.

## Charged cells

`--charge` sets the net cell charge which is most useful for clusters but can also be applied to periodic cells. Removing two
electrons from CaMnO₃ pushes part of the Mn sublattice down to Mn³⁺:

```bash
quick-mag solve assets/G_type/CaMnO3_222.cif --charge -2
```

```text
Oxidation-state assignment:  8xCa+2 | 2xMn+3 + 6xMn+4 | 24xO-2
```

## Large cells and the optimizer

With ≤ `--exact-max-sites` magnetic sites (default 16) the solver enumerates all Ising
configurations exactly. Above that it switches to a multi-restart optimizer. You can force
the optimizer to compare:

```bash
quick-mag solve assets/G_type/LaFeO3_222.cif \
  --exact-max-sites 4 --n-trials 10 --n-steps 100 --max-configs 3
```

```text
  Solving 8 magnetic sites with method='optimizer'.
  Found 5 unique base state(s)
      1       -1.285444      0.000000          0.000        8.00       G
```

Same ground state and same energy as the exact run on this cell. `--n-trials` (restarts)
and `--n-steps` (steps per restart) trade runtime for thoroughness.

## Non-perovskite structures

The pipeline can be applied to non-perovskites with reasonable results so long as superexchange is expected to be relevant. The classification step only works for perovskite structures. Solving
the bundled goethite cluster:

```bash
quick-mag solve assets/goethite_ZnH81_121.vasp --max-configs 4
```

```text
Structure: goethite_ZnH81_121  (194 atoms)
  elements: Fe, H, O, Zn
Oxidation-state assignment:  46xFe+3 | 50xH+1 | 96xO-2 | 2xZn+2

  Reference magnetic configurations (scored independently of the solve):
    unavailable — magnetic sublattice is not a perovskite B grid.

  Solving 48 magnetic sites with method='optimizer'.
      #          energy            ΔE  magnetization  n_unpaired    type
      1       -1.534643      0.000000         -2.000       46.00
```

The reference table is reported as unavailable and the `type` column is blank, because
G/C/F/A are only defined on a perovskite B-site grid. The oxidation states, couplings, and
the spin search all still run — you get ranked configurations, just without perovskite
labels. Note the solver also picked up that 2 of the 48 transition-metal sites (the Zn²⁺)
carry no moment.

## Multiple magnetic species

A double perovskite puts two different magnetic ions on the B sublattice, and the pipeline
handles the mixed couplings without extra flags. First build the structure:

```bash
quick-mag build --formula double \
  --a-site Sr --b-site Fe --b2-site Mo --x-site O --a 3.95 -o out/
```

then find predicted low-energy magnetic configurations.

```bash
quick-mag solve out/SrFeMoO.cif --max-configs 3
```

```text
Structure: SrFeMoO  (40 atoms)
  elements: Fe, Mo, O, Sr
Oxidation-state assignment:  4xFe+4 | 4xMo+4 | 24xO-2 | 8xSr+2
  Solving 8 magnetic sites with method='exact'.
  Ground state (solved + reference): E = -0.002958 (G)
```

## Closing the build → solve loop

`::` chains the two commands into one process — the structures never touch disk:

```bash
# Build a tilt scan, then solve every structure in it.
quick-mag build --a-site La --b-site Mn --x-site O \
  --a 3.95 --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 \
  --tilt-system a-a-c+ --tilt-x 0:12:5 --tilt-y 0:12:5 --zip \
  :: solve --max-configs 1
```

Each structure is solved in turn and reports its ground state and type. For this
sweep the prediction changes as the octahedra rotate:

```text
Structure: LaMnO_tx0_ty0  (40 atoms)
  Ground state (solved + reference): E = -0.158515 (C)
Structure: LaMnO_tx3_ty3  (40 atoms)
  Ground state (solved + reference): E = -0.254634 (G)
Structure: LaMnO_tx6_ty6  (40 atoms)
  Ground state (solved + reference): E = -0.239830 (G)
Structure: LaMnO_tx9_ty9  (40 atoms)
  Ground state (solved + reference): E = -0.218020 (G)
Structure: LaMnO_tx12_ty12  (40 atoms)
  Ground state (solved + reference): E = -0.192430 (G)
```

Add `-o scan/` to the `build` stage to keep the generated CIFs as well; by default
only the last stage of a chain writes anything.

## Relaxing with CHGNet before solving

The sweep above shows how much the answer depends on the geometry — and a builder
seed is an idealized geometry. Inserting `chgnet` into the chain relaxes it first,
so the solver sees the structure the potential actually prefers:

```bash
# Idealized tilted seed, straight to the solver.
quick-mag build --a-site La --b-site Mn --x-site O --a 3.95 \
  --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 \
  --tilt-system a-a-c+ --tilt-x 8 --tilt-y 8 \
  :: solve --max-configs 1
```

```text
  Reference magnetic configurations (scored independently of the solve):
      type          energy            ΔE  magnetization
         G       -0.225872      0.000000          0.000
      C(c)       -0.085678      0.140194          0.000
      C(a)       -0.070258      0.155614          0.000
      C(b)       -0.069936      0.155937          0.000
      A(b)        0.069936      0.295808          0.000
      A(a)        0.070258      0.296131          0.000
      A(c)        0.085678      0.311550          0.000
         F        0.225872      0.451744          8.000

  Ground state (solved + reference): E = -0.225872 (G)
```

The rigidly-tilted seed has equal Mn–O bond lengths in every octahedron. Its three
axes are nearly equivalent — note how close `C(a)`, `C(b)`, and `C(c)` are — and on
that geometry the model predicts **G-type**. Now relax it first:

```bash
quick-mag build --a-site La --b-site Mn --x-site O --a 3.95 \
  --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 \
  --tilt-system a-a-c+ --tilt-x 8 --tilt-y 8 \
  :: chgnet :: solve --max-configs 1
```

```text
CHGNet cell+atoms: LaMnO (40 atoms)
  energy:      -351.172058 eV  (-8.779301/atom)
  max |force|: 0.0037 eV/A
  |m| (mu_B):  mean 0.814  max 4.008
  optimizer:   181 steps, converged
  cell abc:    7.900 7.900 7.900  ->  7.907 8.055 7.929

  Reference magnetic configurations (scored independently of the solve):
      type          energy            ΔE  magnetization
      A(a)       -0.175334      0.000000          0.000
      C(b)       -0.067090      0.108243          0.000
      C(c)       -0.057298      0.118036          0.000
         F       -0.050946      0.124388          8.000
         G        0.050946      0.226279          0.000
      A(c)        0.057298      0.232631          0.000
      A(b)        0.067090      0.242424          0.000
      C(a)        0.175334      0.350667          0.000

  Solved low-energy spin configurations:
      #          energy            ΔE  magnetization  n_unpaired    type
      1       -0.175334      0.000000          0.000        8.00       A

  Ground state (solved + reference): E = -0.175334 (A)
```

CHGNet breaks the cubic degeneracy (the cell goes from 7.90³ to 7.91 × 8.06 ×
7.93) and lets the Mn³⁺ octahedra Jahn-Teller distort into long and short Mn–O
bonds. That changes the superexchange pathways enough to invert the prediction:
G falls from best to nearly worst, the three orientations of A and C fan out over
a 0.35 window instead of sitting on top of one another, and both the reference
table and the solver's search settle on an **A-type** ground state — the
experimentally known ordering of LaMnO₃, and the same answer the solver gives for
the JT-distorted `assets/A_type/LaMnO3_222.cif` at the top of this page.

Which axis the distortion picks varies from run to run, so the winning row may be
`A(a)`, `A(b)`, or `A(c)`; the ordering is A-type either way. That is precisely
why all three orientations are scored — a table reporting one fixed orientation of
A would show a high-energy state here and miss the ground state entirely.

!!! warning "`--fmax` matters here"

    A symmetric seed is a stationary point, so the optimizer has to run far enough
    to fall off it. The default `--fmax 0.005` takes a few hundred steps and
    reaches the distorted minimum; loosening it to `--fmax 0.1` stops after ~10
    steps with the cell essentially unchanged, and the answer stays G.
