#!/usr/bin/env python3
"""Command-line pipeline: structure -> oxidation states -> exchange -> spin configs.

Given a structure file, this predicts oxidation states, builds the exchange-coupling
matrix from the polarization model, and searches for low-energy collinear spin
configurations. The canonical perovskite orderings (G, C, F, A) are always scored
as *reference* configurations, independently of the solver's search, so their
energies are reported even when the solver does not land on them.

Usage:
    python magnetic_cli.py STRUCTURE.cif [--charge 0] [--top-k 1] ...
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from quick_mag.structure import ChemicalStructure, build_from_generation_parameters
from quick_mag.structure_utils import read_structure
from quick_mag.oxidation_state_energy import enumerate_oxidation_states_by_energy
from quick_mag.magnetic_moments import (
    expand_distribution_to_site_assignments,
    format_oxidation_distribution,
)
from quick_mag.ion_descriptors import structure_ion_descriptors
from quick_mag.polarization_model import (
    build_Jeff_matrix,
    build_bridges,
    default_params,
    to_solver_couplings,
)
from quick_mag.spin_solver import (
    NO_NONZERO_MAGNETIC_MOMENTS_MESSAGE,
    solve_for_assignment,
)
from quick_mag.classify_spin_structure import (
    classify_structure,
    site_indexing_from_generation_parameters,
    site_indexing_from_magnetic_sublattice,
)
from quick_mag.reference_configs import CANONICAL_SOLVER_SPIN_PATTERNS, reference_spin_configs


def _resolve_site_indexing(structure: ChemicalStructure, magnetic_site_indices):
    """Best-available perovskite B-site indexing, or ``None`` for non-perovskites.

    Prefers builder provenance (``generation_parameters``); otherwise fits the
    magnetic sublattice onto a B grid via ``site_indexing_from_magnetic_sublattice``.
    """
    params = getattr(structure, "generation_parameters", None)
    if params is not None:
        try:
            build = build_from_generation_parameters(params)
            indexing = site_indexing_from_generation_parameters(params, build)
            if indexing.b_site_indices.size > 0:
                return indexing
        except Exception:
            pass
    return site_indexing_from_magnetic_sublattice(structure, magnetic_site_indices)


def _label_config(structure, config, magnetic_site_indices, site_indexing) -> str:
    """Classify a compact spin config as F/A/C/G/E/Other, or "" if not classifiable."""
    if site_indexing is None:
        return ""
    try:
        tmp = ChemicalStructure.with_zero_magnetic_moments(
            name=structure.name,
            lattice=np.array(structure.lattice, dtype=np.float64, copy=True),
            cartesian_coords=np.array(structure.cartesian_coords, dtype=np.float64, copy=True),
            atomic_labels=list(structure.atomic_labels),
            is_periodic=structure.is_periodic,
        )
        compact = np.asarray(config.all_moments, dtype=float).reshape(-1)
        for i, site in enumerate(magnetic_site_indices):
            if 0 <= int(site) < structure.atom_count and i < compact.size:
                tmp.magnetic_moments[int(site), 2] = compact[i]
        return classify_structure(tmp, site_indexing=site_indexing).label
    except Exception:
        return ""


def _report_reference_configs(configs, patterns) -> None:
    """Print the reference-ordering energy table (lowest reference = ΔE 0)."""
    print("\n  Reference magnetic configurations (scored independently of the solve):")
    if not configs:
        print("    unavailable — magnetic sublattice is not a perovskite B grid.")
        return
    e_min = min(c.energy for c in configs)
    ordered = sorted(zip(patterns, configs), key=lambda pc: pc[1].energy)
    print(f"    {'type':>4}  {'energy':>14}  {'ΔE':>12}  {'magnetization':>13}")
    for name, config in ordered:
        print(
            f"    {name:>4}  {config.energy:>14.6f}  {config.energy - e_min:>12.6f}"
            f"  {config.magnetization:>13.3f}"
        )


def _report_solved_configs(
    structure, all_states, magnetic_site_indices, site_indexing, max_configs
) -> None:
    """Print ranked solved configurations with their F/A/C/G/E/Other labels."""
    print("\n  Solved low-energy spin configurations:")
    if not all_states:
        print("    (none found)")
        return
    e_min = all_states[0].energy
    shown = all_states if max_configs <= 0 else all_states[:max_configs]
    print(
        f"    {'#':>3}  {'energy':>14}  {'ΔE':>12}  {'magnetization':>13}"
        f"  {'n_unpaired':>10}  {'type':>6}"
    )
    for rank, config in enumerate(shown, start=1):
        label = _label_config(structure, config, magnetic_site_indices, site_indexing)
        print(
            f"    {rank:>3}  {config.energy:>14.6f}  {config.energy - e_min:>12.6f}"
            f"  {config.magnetization:>13.3f}  {config.n_unpaired:>10.2f}  {label:>6}"
        )
    if max_configs > 0 and len(all_states) > max_configs:
        print(f"    ... {len(all_states) - max_configs} more (raise --max-configs to see them)")


def _run_assignment(structure, assignment, args) -> None:
    print(f"\n{'=' * 80}")
    print(
        "Oxidation-state assignment:  "
        f"{format_oxidation_distribution(assignment.distributions)}"
    )
    print(f"  oxidation-model energy: {assignment.total_energy:.4f}")

    descriptors = structure_ion_descriptors(structure, assignment)
    if not descriptors:
        print("  No transition-metal magnetic sites in this assignment; skipping.")
        return
    magnetic_site_indices = sorted(descriptors)

    bridges = build_bridges(structure, descriptors)
    if not bridges:
        print("  No exchange couplings (no M-L-M bridges found); skipping solve.")
        return

    params = default_params()
    site_index = {site: i for i, site in enumerate(magnetic_site_indices)}
    j_matrix = to_solver_couplings(build_Jeff_matrix(bridges, site_index, params))

    site_indexing = _resolve_site_indexing(structure, magnetic_site_indices)

    # The polarization J already encodes spin magnitude, so everything runs on
    # unit (+-1) spins: this copy marks which sites carry a moment without
    # double-counting magnitude. Reference and solved configs must share it to be
    # energy-comparable.
    unit_assignment = replace(
        assignment,
        magnetic_moments=(np.abs(assignment.magnetic_moments) > 1e-8).astype(float),
    )

    # Reference G/C/F/A configurations — always reported, independent of the solve.
    reference = reference_spin_configs(
        structure, unit_assignment, j_matrix, magnetic_site_indices, site_indexing
    )
    _report_reference_configs(reference, CANONICAL_SOLVER_SPIN_PATTERNS)

    n_mag = len(magnetic_site_indices)
    method = "exact" if n_mag <= args.exact_max_sites else "optimizer"
    print(f"\n  Solving {n_mag} magnetic sites with method='{method}'.")
    try:
        _base_states, all_states = solve_for_assignment(
            unit_assignment,
            j_matrix,
            magnetic_site_indices=magnetic_site_indices,
            method=method,
            collinear=True,
            n_trials=args.n_trials,
            n_steps=args.n_steps,
        )
    except ValueError as exc:
        if str(exc) == NO_NONZERO_MAGNETIC_MOMENTS_MESSAGE:
            print("  This assignment has no unpaired electrons on any site; nothing to solve.")
            return
        raise

    _report_solved_configs(
        structure, all_states, magnetic_site_indices, site_indexing, args.max_configs
    )

    # Ground state across both the solved search and the reference orderings.
    candidates = list(all_states) + list(reference)
    if candidates:
        ground = min(candidates, key=lambda c: c.energy)
        label = _label_config(structure, ground, magnetic_site_indices, site_indexing)
        tag = f" ({label})" if label else ""
        print(f"\n  Ground state (solved ∪ reference): E = {ground.energy:.6f}{tag}")


SOLVE_DESCRIPTION = (
    "Predict oxidation states, exchange couplings, and low-energy magnetic "
    "configurations for a structure. Always reports the canonical G/C/F/A "
    "reference orderings."
)


def configure_solve_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Attach the ``solve`` pipeline's arguments to ``parser`` (reused by the CLI)."""
    parser.add_argument("structure", type=Path, help="Structure file (.cif P1, or .vasp/POSCAR).")
    parser.add_argument("--charge", type=int, default=0, help="Net cell charge (default 0).")
    parser.add_argument(
        "--max-mixing", type=int, default=2,
        help="Max distinct oxidation states per element (default 2).",
    )
    parser.add_argument(
        "--top-k", type=int, default=1,
        help="Number of lowest-energy oxidation distributions to solve (default 1).",
    )
    parser.add_argument(
        "--exact-max-sites", type=int, default=16,
        help="Use exact Ising enumeration when magnetic sites <= this, else the optimizer (default 16).",
    )
    parser.add_argument("--n-trials", type=int, default=30, help="Optimizer restarts (default 30).")
    parser.add_argument("--n-steps", type=int, default=250, help="Optimizer steps per restart (default 250).")
    parser.add_argument(
        "--max-configs", type=int, default=20,
        help="Cap on solved configs printed (0 = no cap; default 20).",
    )
    return parser


def run_solve(args) -> int:
    """Execute the magnetism pipeline for a parsed ``solve`` argument namespace."""
    structure = read_structure(args.structure)
    labels = structure.element_symbols()
    print(f"Structure: {structure.name}  ({structure.atom_count} atoms)")
    print(f"  elements: {', '.join(sorted(set(labels)))}")

    ranked = enumerate_oxidation_states_by_energy(
        labels, charge=args.charge, max_mixing=args.max_mixing
    )
    if not ranked:
        print("No charge-balanced oxidation-state assignments were found.")
        return 1

    top_distributions = [dist for dist, _energy in ranked[: max(1, args.top_k)]]
    print(f"  Top {len(top_distributions)} oxidation distribution(s) of {len(ranked)} will be solved.")

    for distribution in top_distributions:
        assignments = expand_distribution_to_site_assignments([distribution], structure)
        if not assignments:
            continue
        # Representative site ordering for this distribution.
        _run_assignment(structure, assignments[0], args)

    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=SOLVE_DESCRIPTION)
    configure_solve_parser(parser)
    args = parser.parse_args(argv)
    return run_solve(args)


if __name__ == "__main__":
    raise SystemExit(main())
