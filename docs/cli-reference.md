# CLI Reference

Installing quick_mag registers a single console command, `quick-mag`, with three
subcommands.

```text
quick-mag [-h] {build,solve,ui} ...
```

## `quick-mag build`

Build perovskite structures and write each one to disk as a CIF. Beyond a single
structure, two batch features let you generate a whole family at once:

- **Scans** — any structural variable accepts an *inclusive* `start:stop:num_steps`
  scan spec instead of a scalar. `--a 3.8:4.2:5` builds at 3.8, 3.9, 4.0, 4.1, 4.2.
  The `--zip` argument takes each scanned variable one at a time. For instance,
  `--a 3.8:4.2:5` `--b 3.8:4.2:5` `--c 3.8:4.2:5` would build five structures with 
  an isotropic lattice if `--zip` is included or would build 125 structures via a 3D scan
  if `--zip` is not included.
- **Element combinations** — every single-element site flag accepts a
  comma-separated list, and the full **Cartesian product across sites** is built.
  `--a-site La,Sr --b-site Fe,Co` builds all four LaFe / LaCo / SrFe / SrCo cells.

```text
quick-mag build [options]
```

| Option | Default | Description |
|---|---|---|
| `--formula` | `perovskite` | One of `perovskite`, `double`, `quadruple`, `dq`, `high_entropy`. |
| `--a-site` / `--b-site` / `--x-site` | `La` / `Fe` / `O` | Single-element site(s); comma-separated lists expand as a Cartesian product. |
| `--a2-site` / `--b2-site` | `Sr` / `Co` | Second A'/B' site element(s) for `quadruple` / `double` / `dq`. |
| `--a-sites` / `--b-sites` / `--x-sites` | `La:0.5,Sr:0.5` / `Fe:0.5,Co:0.5` / `O` | High-entropy site mixes as `El:weight` lists (bare `El` = weight 1). |
| `--num-samples` | `1` | High-entropy occupancy realizations to sample (weighted, reproducible). |
| `--a` / `--b` / `--c` | `4.0` / follow a / follow a | Cell edges (Å); scalar or `start:stop:num_steps` scan. |
| `--n-cells-x/-y/-z` | `1` | Supercell replications; scalar or scan (integer). |
| `--tilt-system` | `a0a0a0` | Glazer tilt system. |
| `--tilt-x/-y/-z` | `0.0` | Tilt angles (degrees); scalar or scan. |
| `--periodic` / `--no-periodic` | periodic | Treat the cell as periodic or a finite cluster. |
| `--zip` | off | Advance scanned axes in lockstep instead of a Cartesian grid. |
| `--name` | *(derived)* | Base name for outputs (default derived from elements / formula). |
| `-o`, `--output-dir` | `built_structures` | Directory to write CIFs into. |
| `--dry-run` | off | List the structures that would be built without writing files. |

### Examples

```bash
# Lattice scan across an element product (2 A x 2 B x 3 a = 12 CIFs).
quick-mag build --a-site La,Sr --b-site Fe,Co --x-site O --a 3.8:4.2:3 -o out/

# Four weighted high-entropy occupancy samples on a 2x2x2 grid.
quick-mag build --formula high_entropy \
  --b-sites Fe:0.5,Co:0.3,Ni:0.2 --num-samples 4 \
  --n-cells-x 2 --n-cells-y 2 --n-cells-z 2 -o hea/

# Lockstep scan: lattice and tilt advance together (3 CIFs, not 9).
quick-mag build --a 3.8:4.2:3 --tilt-z 0:10:3 --zip -o path/
```

Output file names encode what varied (e.g. `LaFeO_a4.2_tz10`, `HEA_s03`). Feed any
generated CIF straight into `quick-mag solve` to close the build → predict loop.

## `quick-mag solve`

Predict oxidation states, exchange couplings, and low-energy magnetic configurations for
a structure. The canonical G/C/F/A reference orderings are always scored independently of
the solver, so their energies are reported even when the solver does not land on them.

```text
quick-mag solve STRUCTURE [options]
```

| Argument / option | Default | Description |
|---|---|---|
| `STRUCTURE` | *(required)* | Structure file: `.cif` (P1) or `.vasp`/POSCAR. |
| `--charge` | `0` | Net cell charge. |
| `--max-mixing` | `2` | Max distinct oxidation states allowed per element. |
| `--top-k` | `1` | Number of lowest-energy oxidation distributions to solve. |
| `--exact-max-sites` | `16` | Use exact Ising enumeration when magnetic sites ≤ this, else the optimizer. |
| `--n-trials` | `30` | Optimizer restarts. |
| `--n-steps` | `250` | Optimizer steps per restart. |
| `--max-configs` | `20` | Cap on solved configs printed (`0` = no cap). |

### Example

```bash
quick-mag solve LaMnO3_222.cif --top-k 2 --max-configs 10
```

The output reports, for each oxidation-state assignment: the assignment and its
oxidation-model energy, the reference G/C/F/A ordering energies, and a ranked table of
solved spin configurations with their energy, magnetization, unpaired-moment count, and
classified type. Finally it prints the ground state across the solved search ∪ the
reference orderings.

## `quick-mag ui`

Launch the interactive desktop application.

```bash
quick-mag ui
```

This requires the optional UI dependency. If it is missing, the command prints an
install hint:

```bash
pip install "quick_mag[ui]"
```

See the [Web & Desktop UI guide](ui-guide.md) for what the interface offers.
