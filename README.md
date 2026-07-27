# quick_mag

Perovskite structure builder, visualization, and collinear-magnetism tools.

`quick_mag` attempts to rapidly generate low-energy magnetic configurations using an orbital-specific
implementation of the Goodenough-Kanamori-Anderson rules for superexchange. Predictions of oxidation states,
exchange couplings, and low-energy collinear spin configurations can be accessed in three ways:

- a **command-line tool** (`quick-mag build …`, `quick-mag solve …`),
- an **interactive desktop app** (`quick-mag ui`), and
- the **same UI in the browser**, compiled to WebAssembly via [Pyodide](https://pyodide.org).

📖 **Docs:** <https://heindelj.github.io/quick-mag/> &nbsp;·&nbsp;
🌐 **Live web app:** <https://heindelj.github.io/quick-mag/app/>

## Install

```bash
pip install quick_mag          # core + CLI (numpy, scipy)
pip install "quick_mag[ui]"    # also install the desktop/interactive UI (imgui-bundle)
```

## Usage

```bash
# Build perovskites from the command line (scans + element combinations):
quick-mag build --a-site La,Sr --b-site Fe,Co --x-site O --a 3.8:4.2:3 -o out/

# Predict oxidation states, exchange, and spin orderings for a structure:
quick-mag solve assets/A_type/LaMnO3_222.cif --top-k 2

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