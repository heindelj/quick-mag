# quick_mag

Perovskite structure builder, visualization, and collinear-magnetism tools.

`quick_mag` attempts to rapidly generate low-energy magnetic configurations using an orbital-specific
implementation of the Goodenough-Kanamori-Anderson rules for superexchange. Predictions of oxidation states,
exchange couplings, and low-energy collinear spin configurations can be accessed in three ways:

- a **command-line tool** (`quick-mag build …`, `quick-mag chgnet …`, `quick-mag solve …`),
- an **interactive desktop app** (`quick-mag ui`), and
- the **same UI in the browser**, compiled to WebAssembly via [Pyodide](https://pyodide.org).

**Docs:** <https://heindelj.github.io/quick-mag/>

**Live web app:** <https://heindelj.github.io/quick-mag/app/>

## Install

`quick_mag` is not on PyPI — install it from a checkout. Working inside a conda
environment is recommended so its dependencies stay isolated. If you don't have conda
yet, see the [conda installation guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
(Miniconda is the lightweight option).

```bash
conda create -n quick-mag python=3.12
conda activate quick-mag

git clone https://github.com/heindelj/quick-mag.git
cd quick-mag

pip install -e .              # core + CLI (numpy, scipy)
pip install -e ".[ui]"        # also install the desktop/interactive UI (imgui-bundle)
pip install -e ".[chgnet]"    # also install CHGNet relaxations (chgnet, ase)
```

Python 3.10 or newer is required. Add the `dev` extra (`pip install -e ".[dev,ui]"`) for
the test suite and docs tooling.

## Usage

```bash
# Build perovskites from the command line (scans + element combinations):
quick-mag build --a-site La,Sr --b-site Fe,Co --x-site O --a 3.8:4.2:3 -o out/

# Predict oxidation states, exchange, and spin orderings for a structure:
quick-mag solve assets/A_type/LaMnO3_222.cif --top-k 2

# Relax a structure with the CHGNet machine-learning potential:
quick-mag chgnet assets/A_type/LaMnO3_222.cif -o relaxed/

# Chain commands with `::` — structures pass in memory, never through files:
quick-mag build --a-site La --b-site Mn --x-site O :: chgnet :: solve

# Launch the interactive builder/visualization window:
quick-mag ui
```

See the [CLI reference](https://heindelj.github.io/quick-mag/cli-reference/) for all flags.

## Repository layout

```text
src/quick_mag/     # the installable Python package (core model, CLI, imgui UI)
tests/             # pytest suite
web/               # static Pyodide bootstrap for the in-browser build
docs/              # MkDocs documentation source
assets/            # sample .cif / .vasp geometries
scripts/           # one-off data-generation utilities
```

## Model
Details of the implemented model are available in the [documentation](https://heindelj.github.io/quick-mag).