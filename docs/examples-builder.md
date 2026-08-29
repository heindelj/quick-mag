# Examples: Building Structures

Every example below is a complete `quick-mag build` command. Each writes CIFs into the
directory given by `-o`, and every generated CIF can be fed straight to
[`quick-mag solve`](examples-solver.md).

Add `--dry-run` to any command to list what *would* be built, with atom counts, without
writing files. It is the fastest way to check that a batch expands the way you expect.


`--n-cells-*` defaults to **1** for `perovskite` and `high_entropy`, and to **2** for
the ordered `double`, `quadruple`, and `dq` modes — those place two species on an
alternating sublattice, which a 1×1×1 grid cannot express. You only need to set
`--n-cells-*` when you want something other than that natural cell. Passing an odd
value to an ordered mode still works but prints a warning, since the alternation
cannot close consistently across the periodic boundary.

## Simple perovskite — ABX₃

The default mode. One A species, one B species, one X species.

```bash
quick-mag build --formula perovskite \
  --a-site La --b-site Fe --x-site O \
  --a 3.93 \
  -o out/
```

Gives a 5-atom cell, `LaFeO.cif`: `La₁Fe₁O₃`.

`--a` is the edge of a *single* octahedron cell in Å; `--b` and `--c` follow `--a` unless
given. To get a magnetic supercell, which is needed before any antiferromagnetic ordering
can be represented, replicate the grid:

```bash
quick-mag build --a-site La --b-site Fe --x-site O \
  --a 3.93 --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 \
  --name LaFeO3 -o out/
```

That is 40 atoms (`La₈Fe₈O₂₄`) with 8 B sites — enough for G, C, A, and F orderings.

## Double perovskite — A₂B′B″X₆

Two B-site species in a rock-salt alternation, set by `--b-site` and `--b2-site`.

```bash
quick-mag build --formula double \
  --a-site Sr --b-site Fe --b2-site Mo --x-site O \
  --a 3.95 \
  -o out/
```

40 atoms: `Sr₈Fe₄Mo₄O₂₄`, i.e. 4 formula units of Sr₂FeMoO₆ — the 2×2×2 default. The B
sublattice alternates by the parity of $i+j+k$, so Fe and Mo each occupy half of the 8
octahedra. Pass `--n-cells-x 4 --n-cells-y 4 --n-cells-z 4` for a larger cell.

## Quadruple perovskite — AA′₃B₄X₁₂

Two A-site species in a 1:3 pattern, set by `--a-site` and `--a2-site`. The A site takes
`--a-site` only where all three grid indices share the same parity, giving the
characteristic 1:3 split.

```bash
quick-mag build --formula quadruple \
  --a-site Ca --a2-site Cu --b-site Ti --x-site O \
  --a 3.75 \
  -o out/
```

40 atoms: `Ca₂Cu₆Ti₈O₂₄` — 2 formula units of CaCu₃Ti₄O₁₂.

## Doubly-ordered perovskite — AA′₃BB′X₁₂

The `dq` mode combines both orderings: A/A′ in the 1:3 pattern and B/B′ rock-salt.

```bash
quick-mag build --formula dq \
  --a-site Ca --a2-site Cu --b-site Fe --b2-site Re --x-site O \
  --a 3.80 \
  -o out/
```

40 atoms: `Ca₂Cu₆Fe₄Re₄O₂₄` — 2 formula units of CaCu₃Fe₂Re₂O₁₂. Note this particular examples puts magnetic ions on both the A′ (Cu) and B/B′ (Fe, Re) sublattices.

## High-entropy perovskite

Instead of one element per site, give each site a weighted mix as an `El:weight` list.
Occupancies are then sampled from those weights. Sampling is deterministic in that a given `--seed` will produce the same sequence of randomly-generated structures up to `--num-samples N`.

```bash
quick-mag build --formula high_entropy \
  --a-sites La \
  --b-sites Cr:0.2,Mn:0.2,Fe:0.2,Co:0.2,Ni:0.2 \
  --x-sites O \
  --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 \
  --num-samples 3 --seed 0 \
  -o out/
```

Writes `HEA_s0.cif`, `HEA_s1.cif`, `HEA_s2.cif` — 40 atoms each, all with `La₈…O₂₄` but
different B-site draws:

| sample | B-site occupancies |
|---|---|
| `HEA_s0` | Co₄ Fe₂ Ni₂ |
| `HEA_s1` | Co₂ Cr₁ Fe₂ Mn₁ Ni₂ |
| `HEA_s2` | Co₄ Mn₂ Ni₂ |

Weights are probabilities, not exact stoichiometry, so a single small cell will not hit
the nominal composition. Draw many samples, or use a larger grid, if you need the average
to converge. A bare element with no `:weight` counts as weight 1. This means you can omit the weights if you all randomly chosen elements to have equal probability.

## Octahedral tilts

`--tilt-system` takes a six-character Glazer string (`a0a0a0`, `a-a-c+`, `a0a0c-`, …) and
`--tilt-x/-y/-z` set the tilt angle in degrees (can be negative).

```bash
quick-mag build --a-site La --b-site Mn --x-site O \
  --a 3.95 --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 \
  --tilt-system a-a-c+ --tilt-x 8 --tilt-y 8 --tilt-z 6 \
  -o out/
```

Tilts change the M–X–M bridge angles which the exchange model is
sensitive to. See [What each term favors](theory/magnetism-model.md#32-what-each-term-favors).
Scanning a tilt angle and solving each structure can therefore change the predicted magnetic ordering.

## Defects, impurities, and protonic compensation

Three defect flags address a site by its **grid index**, and each may be repeated. A and
B sites are named `ROLE:i,j,k`; oxygens also need the octahedron vertex they occupy,
one of `+a -a +b -b +c -c`.

```bash
# Oxygen-deficient LaFeO3: La8Fe8O23
quick-mag build --a-site La --b-site Fe --x-site O \
  --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 \
  --vacancy X:0,0,0:+a \
  -o out/
```

An oxygen vacancy leaves the cell with a net **+2**, which the oxidation-state
enumerator absorbs by reducing two cations — the resulting composition is assigned
`Fe(2+) x2, Fe(3+) x6`. Nothing else is needed to keep it neutral.

An **aliovalent substitution** is different. Replacing one Fe³⁺ with Zn²⁺ leaves the
cell one charge short, and the enumerator can only balance it by promoting another iron
to Fe⁴⁺. Adding a proton restores every iron to 3+:

```bash
# La8Fe7ZnO24H -- Zn2+ on a B site, charge-compensated by one proton
quick-mag build --a-site La --b-site Fe --x-site O \
  --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 \
  --substitute B:0,0,0=Zn \
  --proton X:0,0,0:+b \
  -o out/
```

The proton sits 0.98 Å from its host oxygen, perpendicular to the B–O–B axis. Append
`@0`–`@3` to a `--proton` spec to pick among the four equivalent sites on that oxygen
(`--proton X:0,0,0:+b@2`).

The same thing from Python — defects are a keyword on every `generate_*` function:

```python
from quick_mag.defects import SiteDefect
from quick_mag.generation import generate_single_perovskite

structure = generate_single_perovskite(
    "LaFeZnO3H",
    a_site="La", b_site="Fe", x_site="O", a=4.0,
    n_cells_x=2, n_cells_y=2, n_cells_z=2,
    defects=[
        SiteDefect("substitution", ("B", 0, 0, 0), element="Zn"),
        SiteDefect("proton", ("X", 0, 0, 0, 2)),
    ],
)
```

Defects are stored as part of the structure's provenance and are applied *after* the
ideal lattice is generated, so tilt angles, lattice constants, and supercell size stay
fully editable — see [Defects and impurities](tutorial/user-interface.md#defects).

## Batch: element combinations

Any single-element site flag accepts a comma-separated list, and the **Cartesian product
across sites** is built.

```bash
quick-mag build --a-site La,Sr --b-site Fe,Co --x-site O --a 3.95 -o out/
```

Builds 4 structures: `LaFeO`, `LaCoO`, `SrFeO`, `SrCoO`.

## Batch: structural scans

Any structural variable accepts an inclusive `start:stop:num_steps` specification instead of a
scalar. Scanned axes form a Cartesian grid by default.

```bash
# 3 lattice constants x 3 tilt angles = 9 structures
quick-mag build --a 3.8:4.2:3 --tilt-z 0:10:3 --tilt-system a0a0c- -o out/
```

Add `--zip` to advance the scanned axes in lockstep instead (1-D scane). All scanned axes must then have the same number of arguments.

```bash
# 3 structures: (3.8, 0deg), (4.0, 5deg), (4.2, 10deg)
quick-mag build --a 3.8:4.2:3 --tilt-z 0:10:3 --tilt-system a0a0c- --zip -o out/
```

Supercell axes scan too, and are rounded to integers:

```bash
# 1x1x1, 2x2x2, 3x3x3 -> 5, 40, and 135 atoms
quick-mag build --a-site La --b-site Mn \
  --n-cells-x 1:3:3 --n-cells-y 1:3:3 --n-cells-z 1:3:3 --zip -o out/
```

Combinations and scans multiply: the total built is
element-combinations × scan-points × high-entropy-samples. Output names encode whatever
varied (`LaFeO_a4.2_tz10`, `LaMnO_nx2_ny2_nz2`, `HEA_s2`), so a batch never overwrites
itself.

## Finite clusters

By default cells are periodic. `--no-periodic` builds a terminated cluster instead, which
adds the outer A and X shells:

```bash
quick-mag build --a-site La --b-site Fe --x-site O \
  --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 --no-periodic -o out/
```

The same 2×2×2 grid gives **71 atoms** as a cluster versus **40** as a periodic cell.
