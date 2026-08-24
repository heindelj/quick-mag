# CLI Reference

Installing quick_mag registers a single console command, `quick-mag`, with four
subcommands.

```text
quick-mag [-h] {build,chgnet,solve,ui} ...
```

Commands also [chain with `::`](#chaining-commands-with-), passing structures from
one stage to the next in memory instead of through files.

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
| `--seed` | `0` | Base seed for high-entropy sampling. A different seed draws a different, still reproducible, family of realizations; non-zero seeds are tagged in the output name. |
| `--a` / `--b` / `--c` | `4.0` / follow a / follow a | Cell edges (Å); scalar or `start:stop:num_steps` scan. |
| `--n-cells-x/-y/-z` | `1`, or `2` for `double`/`quadruple`/`dq` | Supercell size in primitive cells (1 = the primitive cell); scalar or scan (integer). The ordered formulas default to an even grid because their alternating site patterns need one. |
| `--tilt-system` | `a0a0a0` | Glazer tilt system. |
| `--tilt-x/-y/-z` | `0.0` | Tilt angles (degrees); scalar or scan. |
| `--vacancy` | *(none)* | Remove a site; repeatable. `ROLE:i,j,k`, or `X:i,j,k:VERTEX` for oxygens (VERTEX is one of `+a -a +b -b +c -c`). |
| `--substitute` | *(none)* | Swap a site's element; repeatable. `B:0,1,0=Zn`. |
| `--proton` | *(none)* | Add a charge-compensating H on an oxygen; repeatable. `X:0,1,0:+a`, optionally `@0`-`@3` to pick among the four equivalent sites. |
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
generated CIF straight into `quick-mag solve`, or chain the two with `::`, to close
the build → predict loop.

See [Examples: Building Structures](examples-builder.md) for a worked command per
perovskite type.

## `quick-mag chgnet`

Run [CHGNet](https://github.com/CederGroupHub/chgnet) single-point energies or
geometry optimizations. This is how a builder seed becomes a physically relaxed
structure — including the Jahn-Teller distortions that decide the magnetic
ordering — before it reaches the solver.

```text
quick-mag chgnet [STRUCTURE ...] [options]
```

| Argument / option | Default | Description |
|---|---|---|
| `STRUCTURE ...` | *(required unless chained)* | Structure file(s): `.cif` (P1) or `.vasp`/POSCAR. Omit when the structures come from an earlier `::` stage. |
| `--opt` | on | Optimize the geometry. Relaxes **both** the cell and the atomic positions. |
| `--sp` | off | Single-point energy only; the geometry is unchanged. |
| `--fix-cell` | off | With `--opt`: relax atomic positions only, holding the lattice fixed. |
| `--fix-atoms` | off | With `--opt`: relax the lattice only, holding positions fixed. |
| `--optimizer` | `LBFGS` | ASE optimizer: `LBFGS`, `FIRE`, or `BFGS`. |
| `--fmax` | `0.005` | Force convergence threshold in eV/Å. |
| `--steps` | `500` | Maximum optimizer steps. |
| `--verbose` | off | Show the ASE optimizer's per-step output. |
| `-o`, `--output-dir` | `chgnet_structures` | Directory to write relaxed CIFs into. |

This requires the optional CHGNet dependencies. If they are missing, the command
prints an install hint — run it from the repository root:

```bash
pip install -e ".[chgnet]"
```

!!! note "`--fmax` decides whether symmetry breaks"

    The default `0.005` eV/Å is tight on purpose. A symmetric builder seed sits at
    a stationary point of the potential, so a loose threshold stops before the
    structure falls off it: at `--fmax 0.1` LaMnO₃ converges in ~10 steps with the
    cell barely changed, while the default takes a few hundred steps and finds the
    Jahn-Teller distorted minimum ~0.4 eV lower. Loosen it only when you want a
    quick, approximate geometry.

CHGNet also predicts per-atom magnetic moment *magnitudes*, reported as `|m|` in
the diagnostics. They are unsigned, so they are never written into the structure's
magnetic moments — signed spin configurations come from `quick-mag solve`.

### Examples

```bash
# Single-point energy, forces, and magnitudes.
quick-mag chgnet assets/A_type/LaMnO3_222.cif --sp

# Full relaxation, written to relaxed/.
quick-mag chgnet assets/A_type/LaMnO3_222.cif -o relaxed/

# Positions only, cell held at the experimental lattice.
quick-mag chgnet POSCAR --fix-cell --optimizer FIRE
```

## `quick-mag solve`

Predict oxidation states, exchange couplings, and low-energy magnetic configurations for
a structure. The canonical G/C/F/A reference orderings are always scored independently of
the solver, so their energies are reported even when the solver does not land on them.

```text
quick-mag solve [STRUCTURE ...] [options]
```

| Argument / option | Default | Description |
|---|---|---|
| `STRUCTURE ...` | *(required unless chained)* | Structure file(s): `.cif` (P1) or `.vasp`/POSCAR. Several files are solved in turn. Omit when the structures come from an earlier `::` stage. |
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
classified type. Finally it prints the ground state across the solved search + the
reference orderings.

See [Examples: Predicting Magnetic Configurations](examples-solver.md) for annotated
output and each option in context.

## `quick-mag ui`

Launch the interactive desktop application.

```bash
quick-mag ui
```

This requires the optional UI dependency. If it is missing, the command prints an
install hint — run it from the repository root:

```bash
pip install -e ".[ui]"
```

See the [Web & Desktop UI guide](ui-guide.md) for what the interface offers.

## Chaining commands with `::`

`::` joins commands into a pipeline that runs in a single process, handing the
structures from one stage to the next in memory:

```bash
quick-mag build --a-site La --b-site Mn --x-site O :: chgnet :: solve
```

`::` is an ordinary shell word, so — unlike `|` — it needs no quoting or escaping.
Each stage keeps all of its own options; they simply sit between the `::` tokens.

**Which stages may follow which**

| Stage | May be followed by | Why |
|---|---|---|
| `build` | `chgnet`, `solve` | Generates structures, so it is always first. |
| `chgnet` | `chgnet`, `solve` | Consumes and returns structures. |
| `solve` | *(nothing)* | Produces spin configurations, which no command consumes. |
| `ui` | *(nothing)* | Interactive; it never chains. |

A stage that receives structures from the previous one must not also be given
structure files — `quick-mag build … :: solve some.cif` is an error, not a merge.

**What gets written**

Only the **last** stage writes to disk. Anything earlier stays in memory unless
you explicitly give it `-o/--output-dir`, which is how you keep the intermediates:

```bash
# Nothing on disk; only the solver report is printed.
quick-mag build --a-site La --b-site Mn :: chgnet :: solve

# Keep the unrelaxed seeds and the relaxed geometries as well.
quick-mag build --a-site La --b-site Mn -o seeds/ :: chgnet -o relaxed/ :: solve
```

Run on its own, every command keeps its usual default — `quick-mag build` still
writes to `built_structures/`.

Chains are batched: a `build` that expands into a scan of twenty structures sends
all twenty through `chgnet` and then through `solve`, and CHGNet's model is loaded
once for the whole batch.
