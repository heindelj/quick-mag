#!/usr/bin/env python3
"""Minimal examples for the quick_mag perovskite builder functions."""

from __future__ import annotations

from pathlib import Path

from quick_mag.cif_io import write_cif
from quick_mag.generation import (
    generate_double_perovskite,
    generate_dq_perovskite,
    generate_high_entropy_perovskite,
    generate_quadruple_perovskite,
    generate_single_perovskite,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)

    structures = [
        generate_single_perovskite(
            "01_single_LaFeO3",
            a_site="La",
            b_site="Fe",
            x_site="O",
            a=3.93,
            n_cells_x=2,
            n_cells_y=2,
            n_cells_z=2,
        ),
        generate_double_perovskite(
            "02_double_La2FeCoO6",
            a_site="La",
            b_site="Fe",
            b2_site="Co",
            x_site="O",
            a=3.93,
            n_cells_x=2,
            n_cells_y=2,
            n_cells_z=2,
        ),
        generate_quadruple_perovskite(
            "03_quadruple_LaSr3Fe4O12",
            a_site="La",
            a2_site="Sr",
            b_site="Fe",
            x_site="O",
            a=3.93,
            n_cells_x=2,
            n_cells_y=2,
            n_cells_z=2,
        ),
        generate_dq_perovskite(
            "04_dq_LaSr3FeCoO12",
            a_site="La",
            a2_site="Sr",
            b_site="Fe",
            b2_site="Co",
            x_site="O",
            a=3.93,
            n_cells_x=2,
            n_cells_y=2,
            n_cells_z=2,
        ),
        generate_high_entropy_perovskite(
            "05_high_entropy_A_B_O3",
            a_sites=[("La", 0.5), ("Sr", 0.5)],
            b_sites=[("Fe", 0.4), ("Co", 0.3), ("Mn", 0.3)],
            x_sites=[("O", 1.0)],
            a=3.93,
            n_cells_x=3,
            n_cells_y=3,
            n_cells_z=3,
            sample_index=0,
            seed=11,
        ),
    ]

    for structure in structures:
        path = output_dir / f"{structure.name}.cif"
        write_cif(structure, path)
        composition = " ".join(
            f"{element}{structure.element_symbols().count(element)}"
            for element in sorted(set(structure.element_symbols()))
        )
        print(f"{structure.name}: {structure.atom_count} atoms, {composition}")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
