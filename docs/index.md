# quick_mag

**quick_mag** is a toolkit for building perovskite (and related) crystal structures and predicting
their collinear magnetic ground states. It ships three ways to use the same core model:

- a **command-line tool** (`quick-mag solve …`) for scripted, headless runs,
- an **interactive desktop UI** (`quick-mag ui`) built with Dear ImGui, and
- the **same UI in your browser** — the Python app compiled to WebAssembly via
  [Pyodide](https://pyodide.org), served statically with no backend.

## What it does

Given a structure (`.cif` in P1, or `.vasp`/POSCAR), quick_mag:

1. predicts charge-balanced **oxidation states** ranked by a simple
   energy model;
2. assigns **magnetic moments** to transition-metal sites based on their expected spin configuration;
3. builds an **exchange-coupling matrix** from an orbital-aware superexchange model;
4. searches for **low-energy collinear spin configurations** (exact Ising enumeration
   for small magnetic sublattices, an optimizer otherwise); and
5. **classifies** the resulting orderings against the canonical perovskite patterns when possible
   (F / A / C / G / E).

## Install

```bash
pip install quick_mag          # core + CLI (numpy, scipy)
pip install "quick_mag[ui]"    # also install the desktop/interactive UI (imgui-bundle)
```

From a checkout, for development:

```bash
pip install -e ".[dev,ui]"
```

## Quickstart

```bash
# Predict oxidation states, exchange, and spin orderings for a structure:
quick-mag solve LaMnO3.cif --top-k 2

# Launch the interactive builder/visualization window:
quick-mag ui
```

See the [CLI Reference](cli-reference.md) for every flag, the
[Web & Desktop UI guide](ui-guide.md) to run the interface, and
[Model Theory](theory/magnetism-model.md) for the underlying physics.
