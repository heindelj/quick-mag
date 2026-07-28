#!/usr/bin/env python3
"""Command-line CHGNet calculations: ``quick-mag chgnet ...``.

Runs single-point energies (``--sp``) or geometry optimizations (``--opt``, the
default) with the CHGNet machine-learning potential, and hands the relaxed
structures on to the next stage of a ``::`` chain.

Requires the optional dependencies: ``pip install -e '.[chgnet]'``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np

from quick_mag.export_utils import export_structure
from quick_mag.structure import ChemicalStructure
from quick_mag.structure_utils import read_structure

DEFAULT_OUTPUT_DIR = "chgnet_structures"

MISSING_DEPENDENCY_MESSAGE = (
    "The 'chgnet' command requires the CHGNet and ASE dependencies.\n"
    "Install them from the repository root with:  pip install -e '.[chgnet]'"
)


def resolve_calculation(args: argparse.Namespace) -> str:
    """Map the ``--sp`` / ``--opt`` / ``--fix-*`` flags onto a calculation mode."""
    if args.single_point:
        if args.fix_cell or args.fix_atoms:
            raise ValueError("--fix-cell/--fix-atoms only apply to --opt, not --sp.")
        return "single-point"
    if args.fix_cell and args.fix_atoms:
        raise ValueError(
            "--fix-cell and --fix-atoms together leave nothing to optimize; "
            "use --sp for a single-point energy."
        )
    if args.fix_cell:
        return "atoms"
    if args.fix_atoms:
        return "cell"
    return "cell+atoms"


def _format_cell(structure: ChemicalStructure) -> str:
    lengths = np.linalg.norm(np.asarray(structure.lattice, dtype=float), axis=1)
    return " ".join(f"{value:.3f}" for value in lengths)


def _report(result) -> None:
    """Print one structure's energy, forces, and CHGNet magnitude diagnostics."""
    magmoms = np.abs(np.asarray(result.magnetic_moments, dtype=float))
    print(f"  energy:      {result.energy:.6f} eV  ({result.energy_per_atom:.6f}/atom)")
    print(f"  max |force|: {result.max_force:.4f} eV/A")
    if magmoms.size:
        # CHGNet magmoms are unsigned magnitudes: a diagnostic, not a spin config.
        print(f"  |m| (mu_B):  mean {magmoms.mean():.3f}  max {magmoms.max():.3f}")
    if result.calculation != "single-point":
        state = "converged" if result.converged else "NOT converged"
        print(f"  optimizer:   {result.steps} steps, {state}")
        print(
            f"  cell abc:    {_format_cell(result.initial_structure)}"
            f"  ->  {_format_cell(result.final_structure)}"
        )


def run_chgnet(
    args: argparse.Namespace,
    structures: Optional[List[ChemicalStructure]] = None,
    *,
    write: bool = True,
) -> List[ChemicalStructure]:
    """Run CHGNet over ``structures`` (or the paths in ``args``) and return them.

    ``write`` is set by the chain executor: a stage writes CIFs only when it is
    the last stage, or when ``-o/--output-dir`` was given explicitly.
    """
    try:
        from quick_mag.chgnet_runner import run_chgnet_calculation
    except ImportError as exc:
        raise ImportError(f"{MISSING_DEPENDENCY_MESSAGE}\n(import error: {exc})") from exc

    calculation = resolve_calculation(args)

    if structures is None:
        structures = [read_structure(path) for path in args.structures]
    if not structures:
        raise ValueError("No structures to run; give at least one structure file.")

    out_dir: Optional[Path] = None
    if write or args.output_dir is not None:
        out_dir = Path(args.output_dir or DEFAULT_OUTPUT_DIR).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)

    relaxed: List[ChemicalStructure] = []
    for structure in structures:
        print(
            f"\nCHGNet {calculation}: {structure.name} "
            f"({structure.atom_count} atoms)"
        )
        result = run_chgnet_calculation(
            structure,
            calculation,
            optimizer=args.optimizer,
            fmax=args.fmax,
            steps=args.steps,
            verbose=args.verbose,
        )
        _report(result)
        relaxed.append(result.final_structure)
        if out_dir is not None:
            export_structure(result.final_structure, out_dir)
            print(f"  wrote {result.final_structure.name}.cif -> {out_dir}")
    return relaxed


CHGNET_DESCRIPTION = (
    "Run CHGNet single-point energies or geometry optimizations on structures. "
    "Optimizations relax the cell and the atomic positions by default; --fix-cell "
    "and --fix-atoms narrow that to one or the other."
)


def configure_chgnet_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the ``chgnet`` command's arguments to ``parser`` (reused by the CLI)."""
    parser.add_argument(
        "structures", type=Path, nargs="*",
        help="Structure file(s): .cif (P1) or .vasp/POSCAR. Omit when the "
        "structures come from an earlier stage of a '::' chain.",
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--opt", dest="optimize", action="store_true", default=True,
        help="Optimize the geometry (default).",
    )
    mode.add_argument(
        "--sp", dest="single_point", action="store_true",
        help="Single-point energy only; the geometry is unchanged.",
    )

    parser.add_argument(
        "--fix-cell", action="store_true",
        help="Optimize atomic positions only, holding the lattice fixed.",
    )
    parser.add_argument(
        "--fix-atoms", action="store_true",
        help="Optimize the lattice only, holding atomic positions fixed.",
    )
    parser.add_argument(
        "--optimizer", choices=["LBFGS", "FIRE", "BFGS"], default="LBFGS",
        help="ASE optimizer to drive the relaxation (default LBFGS).",
    )
    parser.add_argument(
        "--fmax", type=float, default=0.005,
        help="Force convergence threshold in eV/A (default 0.005). Tight by "
        "design: a looser threshold stops on the symmetric starting geometry "
        "instead of finding the distorted minimum.",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Maximum optimizer steps (default 500).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show the ASE optimizer's per-step output.",
    )
    # Left as None so the chain executor can tell an explicit -o from a default:
    # a mid-chain stage writes nothing unless the user asked for it.
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help=f"Directory to write relaxed CIFs into (default {DEFAULT_OUTPUT_DIR}).",
    )
    return parser


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=CHGNET_DESCRIPTION)
    configure_chgnet_parser(parser)
    args = parser.parse_args(argv)
    try:
        run_chgnet(args)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1
    except ImportError as exc:
        print(str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
