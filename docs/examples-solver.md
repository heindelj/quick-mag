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
      A(c)       -0.129917      0.000000          0.000
         F       -0.101979      0.027938          8.000
      C(b)       -0.014069      0.115848          0.000
      C(a)       -0.013869      0.116048          0.000
      A(a)        0.013869      0.143787          0.000
      A(b)        0.014069      0.143986          0.000
         G        0.101979      0.231896          0.000
      C(c)        0.129917      0.259835          0.000

  Solving 8 magnetic sites with method='exact'.
  ...
  Solved low-energy spin configurations:
      #          energy            ΔE  magnetization  n_unpaired    type
      1       -0.129917      0.000000          0.000        8.00       A
      2       -0.101979      0.027938          8.000        8.00       F
      3       -0.064972      0.064945          2.000        8.00       A
      ...

  Ground state (solved + reference): E = -0.129917 (A)
```

Reading it:

- **Oxidation-state assignment** — the charge-balanced distribution chosen, and its
  oxidation-model energy. Here Mn³⁺, correct for LaMnO₃.
- **Reference configurations** — the canonical G/C/F/A orderings, always scored on the
  same coupling matrix and reported independently of the search. A and C single out one
  axis of the B-site grid, so each is scored in all three orientations: `A(c)` stacks its
  ferromagnetic planes along **c**, `C(a)` runs its ferromagnetic chains along **a**, and
  so on. G and F treat every axis alike and appear once. The A and C orientations are degenerate for a cubic cell. This degeneracy is broken by Jahn-Teller distorted configurations, like this one.
  `A(c)` is the ordering LaMnO₃ actually adopts.
- **Solved configurations** — the ranked results from state enumeration or optimization, each classified F/A/C/G/E or blank if unclassifiable. `ΔE` is relative to the best magnetic configuration.
- **Ground state** — the lowest energy state over the search and reference configurations.

The energy scale depends on the fitted parameter set. The oxidation energies do not correspond to a physical unit and are better thought of as a score. The parameters of the exchange model are fit from DFT+U relative energies of various spin configurations in eV. The model aims to produce reasonable rankings rather than quantitatively accurate energies, so the energies should be taken with a grain of salt.

## The magnetic cell has to be big enough

An ordering can only appear if the cell can represent it. A 1×1×1 perovskite has a single
B site, so there is nothing to order:

```bash
quick-mag solve out/LaFeO.cif     # 5-atom cell
```

```text
      #          energy            ΔE  magnetization  n_unpaired    type
      1        0.175539      0.000000          1.000        1.00
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

Note that the model has parameters for oxides, sulfides, and fluorides but results will be most reliable for oxides.

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

`--charge` sets the net cell charge which is most useful for clusters but can also be applied to periodic cells (e.g. for uncompensated defects). Removing two
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
  Found 6 unique base state(s)
      1       -1.257475      0.000000          0.000        8.00       G
```

Same ground state and same energy as the exact run on this cell. `--n-trials` (restarts)
and `--n-steps` (steps per restart) trade runtime for thoroughness — the restarts are
randomly seeded, so the number of distinct base states found varies slightly between
runs even though the ground state does not.

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
      1       -1.510582      0.000000         -2.000       46.00
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
  Ground state (solved + reference): E = -0.003259 (G)
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
sweep the prediction changes as soon as the octahedra rotate: the untilted cell has
perfectly regular MnO₆, so there is no orbital order for the model to read
([§3.3](theory/magnetism-model.md#33-crystal-field-orbital-order)) and it comes out
G-type, while every tilted cell distorts the octahedra enough to fix the orbital
order and turn on the ferromagnetic in-plane channel:

```text
Structure: LaMnO_tx0_ty0  (40 atoms)
  Ground state (solved + reference): E = -0.267732 (G)
Structure: LaMnO_tx3_ty3  (40 atoms)
  Ground state (solved + reference): E = -0.396603 (A)
Structure: LaMnO_tx6_ty6  (40 atoms)
  Ground state (solved + reference): E = -0.356073 (A)
Structure: LaMnO_tx9_ty9  (40 atoms)
  Ground state (solved + reference): E = -0.299079 (A)
Structure: LaMnO_tx12_ty12  (40 atoms)
  Ground state (solved + reference): E = -0.236719 (A)
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
      A(c)       -0.319211      0.000000          0.000
      C(b)       -0.280276      0.038935          0.000
      C(a)       -0.279843      0.039368          0.000
         G       -0.240908      0.078303          0.000
         F        0.240908      0.560119          8.000
      A(a)        0.279843      0.599054          0.000
      A(b)        0.280276      0.599487          0.000
      C(c)        0.319211      0.638422          0.000

  Ground state (solved + reference): E = -0.319211 (A)
```

The rigidly-tilted seed has nearly equal Mn–O bond lengths in every octahedron
(1.994 Å × 4, 2.013 Å × 2 here). Its three axes are close to equivalent — note how
little separates `A(c)`, `C(a)`, and `C(b)` — but the residual distortion is still
enough to fix the crystal-field orbital order ([§3.3](theory/magnetism-model.md#33-crystal-field-orbital-order)),
and the model already reads it as **A-type**. Now relax it first:

```bash
quick-mag build --a-site La --b-site Mn --x-site O --a 3.95 \
  --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 \
  --tilt-system a-a-c+ --tilt-x 8 --tilt-y 8 \
  :: chgnet :: solve --max-configs 1
```

```text
CHGNet cell+atoms: LaMnO (40 atoms)
  energy:      -351.172058 eV  (-8.779301/atom)
  max |force|: 0.0042 eV/A
  |m| (mu_B):  mean 0.814  max 4.012
  optimizer:   249 steps, converged
  cell abc:    7.900 7.900 7.900  ->  8.058 7.883 7.952

  Reference magnetic configurations (scored independently of the solve):
      type          energy            ΔE  magnetization
      A(b)       -0.101192      0.000000          0.000
         F       -0.098862      0.002331          8.000
      C(a)       -0.041637      0.059556          0.000
      A(c)       -0.039306      0.061886          0.000
      C(c)        0.039306      0.140498          0.000
      A(a)        0.041637      0.142829          0.000
         G        0.098862      0.200054          0.000
      C(b)        0.101192      0.202384          0.000

  Solved low-energy spin configurations:
      #          energy            ΔE  magnetization  n_unpaired    type
      1       -0.101192      0.000000          0.000        8.00       A

  Ground state (solved + reference): E = -0.101192 (A)
```

CHGNet breaks the cubic degeneracy and lets the Mn³⁺ octahedra Jahn-Teller distort
into genuinely long and short Mn–O bonds (1.94 / 2.07 / 2.13 Å here, versus
1.99 / 2.01 Å in the seed). See discussion of the ferromagnetic occupied → empty
channel ([§3.4](theory/magnetism-model.md#34-the-occupied-to-empty-fm-channel)) for an explanation of how the model describes A-type configurations. Note that which axis the distortion picks varies from run to run due to starting near a saddle point, so the winning A row may
be `A(a)`, `A(b)`, or `A(c)`.

A caveat worth reading off this table: `F` is only 0.002 behind. The stronger the
Jahn-Teller distortion, the stronger the FM channel, and on these relaxed builder
seeds it very nearly cancels the antiferromagnetic in-plane coupling — so a run that
lands in a slightly different CHGNet minimum can report F instead of A. The margin is
comfortable on the properly distorted `assets/A_type/LaMnO3_222.cif` (A ahead of F by
0.028), so read a near-degenerate A/F table as a sign that the geometry, not the spin
model, is what needs pinning down.

