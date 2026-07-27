#!/usr/bin/env python3
"""Command-line perovskite builder: ``quick-mag build ...``.

Exposes the high-level builders in :mod:`quick_mag.generation` on the command
line and writes each generated structure to disk as a CIF. Two batch features
sit on top of the single-structure builders:

* **Scans** — any structural variable (lattice ``--a/--b/--c``, tilt angles, or
  supercell ``--n-cells-*``) accepts an *inclusive* ``start:stop:num_steps``
  scan spec instead of a scalar. Scanned axes form a Cartesian grid by default,
  or advance in lockstep with ``--zip`` (all scanned axes must then share
  ``num_steps``).
* **Element combinations** — every single-element site flag (``--a-site`` etc.)
  accepts a comma-separated list; the full Cartesian product across sites is
  built. In ``--formula high_entropy`` the per-site mixes are given as weighted
  ``El:weight`` lists and ``--num-samples N`` draws N independent, reproducible
  occupancy realizations that respect those weights.

The generated set is ``element-combinations x scan-points x high-entropy-samples``.
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from quick_mag.element_data import is_valid_symbol
from quick_mag.export_utils import export_structure, sanitize_filename
from quick_mag.generation import (
    generate_dq_perovskite,
    generate_double_perovskite,
    generate_high_entropy_perovskite,
    generate_quadruple_perovskite,
    generate_single_perovskite,
    normalize_element_symbol,
)
from quick_mag.perovskite_builder import parse_glazer_tilt_system

# formula_mode -> the single-element site kwargs that mode consumes (in the
# order the corresponding generate_* function expects them). ``high_entropy`` is
# handled separately because it takes weighted distributions, not bare symbols.
_MODE_SINGLE_SITES: Dict[str, tuple] = {
    "perovskite": ("a_site", "b_site", "x_site"),
    "double": ("a_site", "b_site", "b2_site", "x_site"),
    "quadruple": ("a_site", "a2_site", "b_site", "x_site"),
    "dq": ("a_site", "a2_site", "b_site", "b2_site", "x_site"),
}
_MODE_BUILDERS = {
    "perovskite": generate_single_perovskite,
    "double": generate_double_perovskite,
    "quadruple": generate_quadruple_perovskite,
    "dq": generate_dq_perovskite,
    "high_entropy": generate_high_entropy_perovskite,
}

# Structural scan axes, in a stable order, and the short tag used in file names.
_STRUCTURAL_AXES = (
    "a", "b", "c", "tilt_x", "tilt_y", "tilt_z", "n_cells_x", "n_cells_y", "n_cells_z"
)
_AXIS_TAG = {
    "a": "a", "b": "b", "c": "c",
    "tilt_x": "tx", "tilt_y": "ty", "tilt_z": "tz",
    "n_cells_x": "nx", "n_cells_y": "ny", "n_cells_z": "nz",
}
_INTEGER_AXES = {"n_cells_x", "n_cells_y", "n_cells_z"}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def parse_scan_spec(value, *, name: str, integer: bool = False) -> List[float]:
    """Expand a scalar or ``start:stop:num_steps`` scan spec into a value list.

    The scan is *inclusive* of both endpoints (``np.linspace``). Integer axes are
    rounded and de-duplicated (so a scan can never emit a fractional supercell).
    A plain number returns a single-element list.
    """
    text = str(value).strip()
    if ":" in text:
        parts = text.split(":")
        if len(parts) != 3:
            raise ValueError(
                f"{name} scan must be 'start:stop:num_steps', got '{value}'."
            )
        try:
            start, stop, steps = float(parts[0]), float(parts[1]), int(parts[2])
        except ValueError:
            raise ValueError(
                f"{name} scan 'start:stop:num_steps' must be numbers, got '{value}'."
            )
        if steps < 1:
            raise ValueError(f"{name} scan num_steps must be >= 1, got {steps}.")
        points = np.linspace(start, stop, steps)
        if integer:
            return sorted({int(round(p)) for p in points})
        return [float(p) for p in points]

    try:
        return [int(round(float(text)))] if integer else [float(text)]
    except ValueError:
        raise ValueError(
            f"{name} must be a number or a 'start:stop:num_steps' scan, got '{value}'."
        )


def parse_element_list(value, *, name: str) -> List[str]:
    """Parse a comma-separated list of element symbols (validated)."""
    symbols: List[str] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        symbol = normalize_element_symbol(token)
        if not is_valid_symbol(symbol):
            raise ValueError(f"{name} element '{token}' is not a valid element symbol.")
        symbols.append(symbol)
    if not symbols:
        raise ValueError(f"{name} needs at least one element symbol.")
    return symbols


def parse_high_entropy_distribution(value, *, name: str) -> List[tuple]:
    """Parse ``El:weight`` comma lists (bare ``El`` means weight 1)."""
    entries: List[tuple] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" in token:
            symbol_text, _, weight_text = token.rpartition(":")
            try:
                weight = float(weight_text)
            except ValueError:
                raise ValueError(f"{name} weight in '{token}' is not a number.")
        else:
            symbol_text, weight = token, 1.0
        symbol = normalize_element_symbol(symbol_text.strip())
        if not is_valid_symbol(symbol):
            raise ValueError(
                f"{name} element '{symbol_text}' is not a valid element symbol."
            )
        entries.append((symbol, weight))
    if not entries:
        raise ValueError(f"{name} needs at least one 'El[:weight]' entry.")
    return entries


# ---------------------------------------------------------------------------
# Combination expansion
# ---------------------------------------------------------------------------


def _structural_points(
    axes: Dict[str, list], zip_scans: bool
) -> List[Dict[str, object]]:
    """Expand the scan axes into concrete points (Cartesian grid, or zipped)."""
    if zip_scans:
        scanned = [name for name in _STRUCTURAL_AXES if len(axes[name]) > 1]
        lengths = {len(axes[name]) for name in scanned}
        if len(lengths) > 1:
            detail = ", ".join(f"{name}={len(axes[name])}" for name in scanned)
            raise ValueError(
                f"--zip requires all scanned axes to share num_steps; got {detail}."
            )
        length = lengths.pop() if scanned else 1
        points = []
        for i in range(length):
            points.append(
                {
                    name: (axes[name][i] if len(axes[name]) > 1 else axes[name][0])
                    for name in _STRUCTURAL_AXES
                }
            )
        return points

    value_lists = [axes[name] for name in _STRUCTURAL_AXES]
    return [
        dict(zip(_STRUCTURAL_AXES, combo))
        for combo in itertools.product(*value_lists)
    ]


def _element_combinations(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Cartesian product across the single-element site lists for this mode."""
    site_kwargs = _MODE_SINGLE_SITES[args.formula]
    per_site = [
        parse_element_list(getattr(args, kwarg), name=f"--{kwarg.replace('_', '-')}")
        for kwarg in site_kwargs
    ]
    return [
        dict(zip(site_kwargs, combo)) for combo in itertools.product(*per_site)
    ]


def _structural_axes(args: argparse.Namespace) -> Dict[str, list]:
    """Build the per-axis value lists, honouring the "b/c default to a" rule."""
    axes: Dict[str, list] = {
        "a": parse_scan_spec(args.a, name="--a"),
        # ``None`` sentinel means "follow a" inside the generate_* functions.
        "b": parse_scan_spec(args.b, name="--b") if args.b is not None else [None],
        "c": parse_scan_spec(args.c, name="--c") if args.c is not None else [None],
        "tilt_x": parse_scan_spec(args.tilt_x, name="--tilt-x"),
        "tilt_y": parse_scan_spec(args.tilt_y, name="--tilt-y"),
        "tilt_z": parse_scan_spec(args.tilt_z, name="--tilt-z"),
        "n_cells_x": parse_scan_spec(args.n_cells_x, name="--n-cells-x", integer=True),
        "n_cells_y": parse_scan_spec(args.n_cells_y, name="--n-cells-y", integer=True),
        "n_cells_z": parse_scan_spec(args.n_cells_z, name="--n-cells-z", integer=True),
    }
    return axes


def _default_root(formula: str, elem_combo: Dict[str, str]) -> str:
    if formula == "high_entropy":
        return "HEA"
    return "".join(elem_combo.values())


def _structure_name(
    *,
    root: str,
    elem_combo: Dict[str, str],
    point: Dict[str, object],
    varying_axes: List[str],
    sample: int,
    num_samples: int,
    formula: str,
    append_elements: bool,
) -> str:
    """Compose a unique, human-readable name encoding everything that varied."""
    parts = [root]
    if append_elements and formula != "high_entropy":
        parts.append("".join(elem_combo.values()))
    for axis in varying_axes:
        value = point[axis]
        if value is None:
            continue
        tag = _AXIS_TAG[axis]
        if axis in _INTEGER_AXES:
            parts.append(f"{tag}{int(value)}")
        else:
            parts.append(f"{tag}{float(value):.3g}")
    if num_samples > 1:
        width = len(str(num_samples - 1))
        parts.append(f"s{sample:0{width}d}")
    return "_".join(part for part in parts if part)


# ---------------------------------------------------------------------------
# Structure construction
# ---------------------------------------------------------------------------


def _build_one(
    formula: str,
    name: str,
    elem_combo: Dict[str, str],
    point: Dict[str, object],
    high_entropy_sites: Optional[Dict[str, list]],
    sample: int,
    periodic: bool,
    tilt_system: str,
):
    common = dict(
        a=point["a"],
        b=point["b"],
        c=point["c"],
        n_cells_x=int(point["n_cells_x"]),
        n_cells_y=int(point["n_cells_y"]),
        n_cells_z=int(point["n_cells_z"]),
        tilt_system=tilt_system,
        tilt_angles_deg=(
            float(point["tilt_x"]),
            float(point["tilt_y"]),
            float(point["tilt_z"]),
        ),
        periodic=periodic,
    )
    builder = _MODE_BUILDERS[formula]
    if formula == "high_entropy":
        assert high_entropy_sites is not None
        return builder(
            name,
            a_sites=high_entropy_sites["a_sites"],
            b_sites=high_entropy_sites["b_sites"],
            x_sites=high_entropy_sites["x_sites"],
            sample_index=sample,
            **common,
        )
    return builder(name, **elem_combo, **common)


def run_build(args: argparse.Namespace) -> int:
    """Execute a (possibly batched) build from a parsed ``build`` namespace."""
    try:
        parse_glazer_tilt_system(args.tilt_system)
    except ValueError as exc:
        print(f"Invalid --tilt-system '{args.tilt_system}': {exc}")
        return 1

    try:
        axes = _structural_axes(args)
        points = _structural_points(axes, args.zip_scans)

        if args.formula == "high_entropy":
            element_combos: List[Dict[str, str]] = [{}]
            high_entropy_sites = {
                "a_sites": parse_high_entropy_distribution(args.a_sites, name="--a-sites"),
                "b_sites": parse_high_entropy_distribution(args.b_sites, name="--b-sites"),
                "x_sites": parse_high_entropy_distribution(args.x_sites, name="--x-sites"),
            }
            num_samples = max(1, int(args.num_samples))
        else:
            element_combos = _element_combinations(args)
            high_entropy_sites = None
            if args.num_samples != 1:
                print(
                    "Note: --num-samples only applies to --formula high_entropy; ignoring."
                )
            num_samples = 1
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    varying_axes = [name for name in _STRUCTURAL_AXES if len(axes[name]) > 1]
    append_elements = args.name is not None and len(element_combos) > 1

    out_dir = Path(args.output_dir).expanduser()
    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set()
    written = 0
    for elem_combo in element_combos:
        root = args.name if args.name is not None else _default_root(
            args.formula, elem_combo
        )
        for point in points:
            for sample in range(num_samples):
                name = _structure_name(
                    root=root,
                    elem_combo=elem_combo,
                    point=point,
                    varying_axes=varying_axes,
                    sample=sample,
                    num_samples=num_samples,
                    formula=args.formula,
                    append_elements=append_elements,
                )
                # Guard against accidental name collisions (e.g. two scan points
                # that round to the same tag) so no structure silently overwrites.
                unique = name
                suffix = 2
                while sanitize_filename(unique) in used_names:
                    unique = f"{name}_{suffix}"
                    suffix += 1
                used_names.add(sanitize_filename(unique))

                structure = _build_one(
                    args.formula,
                    unique,
                    elem_combo,
                    point,
                    high_entropy_sites,
                    sample,
                    args.periodic,
                    args.tilt_system,
                )
                if args.dry_run:
                    print(f"  would build {unique} ({structure.atom_count} atoms)")
                else:
                    export_structure(structure, out_dir)
                    print(f"  wrote {unique} ({structure.atom_count} atoms)")
                written += 1

    verb = "would write" if args.dry_run else "wrote"
    print(f"Done. {verb} {written} structure(s) under {out_dir}.")
    return 0


BUILD_DESCRIPTION = (
    "Build perovskite structures from the command line and write them as CIFs. "
    "Structural variables accept inclusive 'start:stop:num_steps' scans, and "
    "single-element site flags accept comma-separated element lists expanded as a "
    "Cartesian product across sites."
)


def configure_build_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the ``build`` command's arguments to ``parser`` (reused by the CLI)."""
    parser.add_argument(
        "--formula",
        choices=["perovskite", "double", "quadruple", "dq", "high_entropy"],
        default="perovskite",
        help="Perovskite formula mode (default perovskite).",
    )

    sites = parser.add_argument_group("elements (comma-separated lists -> Cartesian product)")
    sites.add_argument("--a-site", default="La", help="A-site element(s) (default La).")
    sites.add_argument("--b-site", default="Fe", help="B-site element(s) (default Fe).")
    sites.add_argument("--x-site", default="O", help="X-site element(s) (default O).")
    sites.add_argument("--a2-site", default="Sr", help="A'-site element(s), quadruple/dq (default Sr).")
    sites.add_argument("--b2-site", default="Co", help="B'/B''-site element(s), double/dq (default Co).")

    he = parser.add_argument_group("high-entropy sites ('El:weight' comma lists)")
    he.add_argument("--a-sites", default="La:0.5,Sr:0.5", help="A-site mix (default La:0.5,Sr:0.5).")
    he.add_argument("--b-sites", default="Fe:0.5,Co:0.5", help="B-site mix (default Fe:0.5,Co:0.5).")
    he.add_argument("--x-sites", default="O", help="X-site mix (default O).")
    he.add_argument(
        "--num-samples", type=int, default=1,
        help="High-entropy occupancy realizations to sample (default 1).",
    )

    struct = parser.add_argument_group(
        "structural variables (scalar or 'start:stop:num_steps' scan)"
    )
    struct.add_argument("--a", default="4.0", help="Cell edge a in Angstrom (default 4.0).")
    struct.add_argument("--b", default=None, help="Cell edge b (default: follow a).")
    struct.add_argument("--c", default=None, help="Cell edge c (default: follow a).")
    struct.add_argument("--n-cells-x", default="1", help="Supercell replications along x (default 1).")
    struct.add_argument("--n-cells-y", default="1", help="Supercell replications along y (default 1).")
    struct.add_argument("--n-cells-z", default="1", help="Supercell replications along z (default 1).")
    struct.add_argument("--tilt-system", default="a0a0a0", help="Glazer tilt system (default a0a0a0).")
    struct.add_argument("--tilt-x", default="0.0", help="Tilt angle about x in degrees (default 0).")
    struct.add_argument("--tilt-y", default="0.0", help="Tilt angle about y in degrees (default 0).")
    struct.add_argument("--tilt-z", default="0.0", help="Tilt angle about z in degrees (default 0).")

    periodic = parser.add_mutually_exclusive_group()
    periodic.add_argument(
        "--periodic", dest="periodic", action="store_true", default=True,
        help="Treat the structure as periodic (default).",
    )
    periodic.add_argument(
        "--no-periodic", dest="periodic", action="store_false",
        help="Treat the structure as a finite cluster.",
    )

    parser.add_argument(
        "--zip", dest="zip_scans", action="store_true",
        help="Advance scanned axes in lockstep instead of a Cartesian grid.",
    )
    parser.add_argument(
        "--name", default=None,
        help="Base name for outputs (default derived from elements / formula).",
    )
    parser.add_argument(
        "-o", "--output-dir", default="built_structures",
        help="Directory to write CIFs into (default built_structures).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the structures that would be built without writing files.",
    )
    return parser


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=BUILD_DESCRIPTION)
    configure_build_parser(parser)
    args = parser.parse_args(argv)
    return run_build(args)


if __name__ == "__main__":
    raise SystemExit(main())
