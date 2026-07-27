# quick_mag

Perovskite/crystal structure builder, visualization, and collinear-magnetism tools.

`quick_mag` turns a bare structure into ranked magnetic ground states — predicting
oxidation states, exchange couplings, and low-energy collinear spin configurations — and
ships the same core model three ways:

- a **command-line tool** (`quick-mag build …`, `quick-mag solve …`),
- an **interactive Dear ImGui desktop app** (`quick-mag ui`), and
- the **same UI in the browser**, compiled to WebAssembly via
  [Pyodide](https://pyodide.org) and served as static files (no backend).

📖 **Docs:** <https://heindelj.github.io/quick-mag/> &nbsp;·&nbsp;
🌐 **Live web app:** <https://heindelj.github.io/quick-mag/app/>

## Install

```bash
pip install quick_mag          # core + CLI (numpy, scipy)
pip install "quick_mag[ui]"    # also install the desktop/interactive UI (imgui-bundle)
```

Development install from a checkout:

```bash
pip install -e ".[dev,ui]"
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

## Development

```bash
pip install -e ".[dev,ui]"
pytest                        # run the test suite
mkdocs serve                  # preview the docs at http://127.0.0.1:8000
python web/build_manifest.py && python -m http.server 8000
                              # then open http://localhost:8000/web/index.html
```
