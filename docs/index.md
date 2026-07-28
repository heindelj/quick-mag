# quick-mag

**quick-mag** is a toolkit for building perovskite (and related) crystal structures and predicting
their collinear magnetic ground states. There are three ways to interact with the same core model:

- a **command-line tool** (`quick-mag build …`, `quick-mag chgnet …`, `quick-mag solve …`,
  chainable with `::`) for scripted runs,
- an **interactive desktop UI** (`quick-mag ui`) built with Dear ImGui, and
- the **same UI in the browser**: the Python app compiled to WebAssembly via
  [Pyodide](https://pyodide.org).

## What it does

Given a structure (`.cif` in P1, or `.vasp`/POSCAR), quick_mag:

1. predicts charge-balanced **oxidation states** ranked by a simple
   energy model
2. assigns **magnetic moments** to transition-metal sites based on their expected spin configuration
3. builds an **exchange-coupling matrix** from an orbital-aware superexchange model
4. searches for **low-energy collinear spin configurations** (exact Ising enumeration
   for small magnetic sublattices, an optimizer otherwise) and
5. **classifies** the resulting orderings against the canonical perovskite patterns when possible (F / A / C / G / E).

## Install

quick-mag is not published to PyPI, so install it from a checkout. A conda environment is
recommended to keep its dependencies isolated from your system Python. If you don't
already have conda, follow the
[conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
— Miniconda is the minimal option and is all you need here.

```bash
conda create -n quick-mag python=3.12
conda activate quick-mag

git clone https://github.com/heindelj/quick-mag.git
cd quick-mag

pip install -e .              # core + CLI (numpy, scipy)
pip install -e ".[ui]"        # also install the desktop/interactive UI (imgui-bundle)
pip install -e ".[chgnet]"    # also install CHGNet relaxations (chgnet, ase)
```

Python 3.10 or newer is required. The `-e` (editable) install means edits to `src/` take
effect immediately, without reinstalling.

For development — the test suite and docs tooling as well:

```bash
pip install -e ".[dev,ui]"
```

## Quickstart

```bash
# Predict oxidation states, exchange, and spin orderings for a structure:
quick-mag solve LaMnO3.cif --top-k 2

# Chain commands with `::`: build, relax with CHGNet, then solve, all in memory.
quick-mag build --a-site La --b-site Mn --x-site O :: chgnet :: solve

# Launch the interactive builder/visualization window:
quick-mag ui
```

Worked examples: [Building Structures](examples-builder.md) covers every perovskite type
and the batch scan/product features; [Predicting Magnetic Configurations](examples-solver.md)
covers the solver end to end.

See the [CLI Reference](cli-reference.md) for every flag, the
[Web & Desktop UI guide](ui-guide.md) to run the interface, and
[Model Theory](theory/magnetism-model.md) for the underlying physics.
