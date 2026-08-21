#!/usr/bin/env python3
"""Build LaFeO3 defect examples with the quick_mag Python API.

This example:

* builds an ideal LaFeO3 lattice
* removes one randomly chosen X-site oxygen
* replaces one Fe with Zn
* adds one proton to compensate the Fe3+ -> Zn2+ substitution
* reports quick_mag oxidation states and CHGNet magnetic-moment diagnostics

CHGNet is optional. Install it with ``pip install -e ".[chgnet]"`` from the
repository root, or pass ``--skip-chgnet`` to only exercise quick_mag's builder
and oxidation-state model.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

from quick_mag.cif_io import write_cif
from quick_mag.defects import SiteDefect, compensation_hint, site_key_display
from quick_mag.generation import generate_single_perovskite
from quick_mag.oxidation_state_energy import enumerate_oxidation_states_by_energy
from quick_mag.perovskite_builder import SiteKey, canonical_site_keys
from quick_mag.structure import ChemicalStructure


DEFAULT_CELL_EDGE_A = 3.93
DEFAULT_N_CELLS = 2


def build_lafeo3(
    name: str,
    *,
    n_cells: int,
    defects: Sequence[SiteDefect] = (),
) -> ChemicalStructure:
    return generate_single_perovskite(
        name,
        a_site="La",
        b_site="Fe",
        x_site="O",
        a=DEFAULT_CELL_EDGE_A,
        n_cells_x=n_cells,
        n_cells_y=n_cells,
        n_cells_z=n_cells,
        periodic=True,
        defects=defects,
    )


def random_x_site(n_cells: int, seed: int) -> SiteKey:
    rng = np.random.default_rng(seed)
    x_sites = [
        key
        for key in canonical_site_keys((n_cells, n_cells, n_cells), periodic=True)
        if key.role == "X"
    ]
    return x_sites[int(rng.integers(0, len(x_sites)))]


def format_composition(structure: ChemicalStructure) -> str:
    counts = Counter(structure.element_symbols())
    return " ".join(f"{element}{counts[element]}" for element in sorted(counts))


def format_oxidation_distribution(distribution: dict[str, list[tuple[int, int]]]) -> str:
    pieces: list[str] = []
    for element in sorted(distribution):
        tokens = [
            f"{count}x{element}{oxidation_state:+d}"
            for oxidation_state, count in sorted(distribution[element])
            if count > 0
        ]
        if tokens:
            pieces.append(" + ".join(tokens))
    return " | ".join(pieces) if pieces else "(no oxidation states)"


def print_oxidation_state_summary(structure: ChemicalStructure) -> None:
    ranked = enumerate_oxidation_states_by_energy(
        structure.element_symbols(),
        top_k=1,
    )
    if not ranked:
        print("  quick_mag oxidation states: none found")
        return

    distribution, energy = ranked[0]
    print(
        "  quick_mag oxidation states: "
        f"{format_oxidation_distribution(distribution)}"
    )
    print(f"  oxidation-model energy: {energy:.3f} eV")

def report_structure(
    label: str,
    structure: ChemicalStructure,
    *,
    reference: ChemicalStructure,
    output_dir: Path,
    run_chgnet: bool,
) -> None:
    path = output_dir / f"{structure.name}.cif"
    write_cif(structure, path)

    print(f"\n{label}: {structure.name}")
    print(f"  composition: {format_composition(structure)}")
    print(f"  wrote: {path}")
    print_oxidation_state_summary(structure)
    nominal_charge, message = compensation_hint(
        reference.atomic_labels,
        structure.atomic_labels,
    )
    print(f"  charge compensation hint: {message}")

    if not run_chgnet:
        return
    result = run_chgnet_single_point(structure)
    print(
        f"  CHGNet single-point energy: {result.energy:.6f} eV "
        f"({result.energy_per_atom:.6f} eV/atom)"
    )
    print(
        "  CHGNet magnetic moments: "
        f"{summarize_chgnet_moments(structure, result.magnetic_moments)}"
    )
    if nominal_charge:
        print(f"  nominal charge vs. pristine: {nominal_charge:+d}")

def summarize_chgnet_moments(
    structure: ChemicalStructure,
    magnetic_moments: np.ndarray,
) -> str:
    moments = np.abs(np.asarray(magnetic_moments, dtype=float))
    symbols = np.asarray(structure.element_symbols(), dtype=object)
    lines = []
    for element in ("Fe", "Zn", "H"):
        selected = moments[symbols == element]
        if selected.size == 0:
            continue
        lines.append(
            f"{element}: mean {selected.mean():.3f}, "
            f"min {selected.min():.3f}, max {selected.max():.3f} mu_B"
        )
    return "; ".join(lines) if lines else "no selected sites"


def run_chgnet_single_point(structure: ChemicalStructure):
    from quick_mag.chgnet_runner import run_chgnet_calculation

    return run_chgnet_calculation(structure, calculation="single-point")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for the X vacancy.",
    )
    parser.add_argument(
        "--n-cells",
        type=int,
        default=DEFAULT_N_CELLS,
        help="Number of perovskite cells along each lattice vector.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "generated",
        help="Directory for generated CIF files.",
    )
    parser.add_argument(
        "--skip-chgnet",
        action="store_true",
        help="Skip CHGNet single-point calculations.",
    )
    args = parser.parse_args()
    if args.n_cells < 1:
        parser.error("--n-cells must be at least 1.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Build LaFeO3 with a simple wrapper around the single perovskite builder
    pristine = build_lafeo3("LaFeO3_pristine", n_cells=args.n_cells)

    # Choose a random ligand site to remove the oxygen from
    vacancy_site = random_x_site(args.n_cells, args.seed)
    vacancy = SiteDefect("vacancy", vacancy_site)
    with_vacancy = build_lafeo3(
        "LaFeO3_random_X_vacancy",
        n_cells=args.n_cells,
        defects=[vacancy],
    )

    # Choose a B site (the body-centered sites) to substitue a Zn
    # then build the structure
    substitution_site = SiteKey("B", 0, 0, 0)
    substitution = SiteDefect("substitution", substitution_site, element="Zn")
    with_substitution = build_lafeo3(
        "LaFeO3_Zn_on_Fe",
        n_cells=args.n_cells,
        defects=[substitution],
    )

    # Choose an X site (the ligand sites) to place a proton
    # as charge compensation for replacing Fe3+ with Zn2+.
    proton_site = SiteKey("X", 0, 0, 0, 2)
    proton = SiteDefect("proton", proton_site, orientation=0)
    proton_compensated = build_lafeo3(
        "LaFeO3_Zn_on_Fe_plus_H",
        n_cells=args.n_cells,
        defects=[substitution, proton],
    )

    print(f"Random X vacancy site: {site_key_display(vacancy_site)}")
    print(f"Substitution site: {site_key_display(substitution_site)} -> Zn")
    print(f"Proton host site: {site_key_display(proton_site)}")

    run_chgnet = not args.skip_chgnet
    if run_chgnet:
        try:
            import ase
            import chgnet
            import quick_mag.chgnet_runner
        except ImportError as exc:
            run_chgnet = False
            print(
                "\nCHGNet is not installed, so this run will only print builder and "
                "oxidation-state results. Install it with: pip install -e '.[chgnet]'"
                f"\n(import error: {exc})"
            )

    report_structure(
        "Pristine reference",
        pristine,
        reference=pristine,
        output_dir=args.output_dir,
        run_chgnet=run_chgnet,
    )
    report_structure(
        "Random X vacancy",
        with_vacancy,
        reference=pristine,
        output_dir=args.output_dir,
        run_chgnet=run_chgnet,
    )
    report_structure(
        "B-site substitution",
        with_substitution,
        reference=pristine,
        output_dir=args.output_dir,
        run_chgnet=run_chgnet,
    )
    report_structure(
        "Proton-compensated substitution",
        proton_compensated,
        reference=pristine,
        output_dir=args.output_dir,
        run_chgnet=run_chgnet,
    )


if __name__ == "__main__":
    main()
