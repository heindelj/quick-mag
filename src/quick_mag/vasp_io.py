from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

from quick_mag.structure import ChemicalStructure


@dataclass
class PoscarData:
    """Primitives parsed from a POSCAR/VASP file, in file atom order."""

    title: str
    lattice: np.ndarray
    species: List[str]
    counts: List[int]
    fractional_coords: np.ndarray
    cartesian_coords: np.ndarray
    species_labels: List[str]
    coordinate_mode: str  # "direct" or "cartesian"


def parse_poscar(text: str, *, title_fallback: str = "structure") -> PoscarData:
    """Parse POSCAR/VASP ``text`` into :class:`PoscarData` (float64)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError("Input does not look like a valid VASP/POSCAR file.")

    title = lines[0] or title_fallback
    scale = float(lines[1].split()[0])
    lattice = (
        np.array(
            [[float(value) for value in lines[row].split()[:3]] for row in range(2, 5)],
            dtype=np.float64,
        )
        * scale
    )

    species = lines[5].split()
    counts = [int(value) for value in lines[6].split()]
    atom_count = sum(counts)

    cursor = 7
    if lines[cursor].lower().startswith("s"):  # optional Selective dynamics line
        cursor += 1

    coordinate_mode = lines[cursor].lower()
    cursor += 1
    coord_lines = lines[cursor : cursor + atom_count]
    if len(coord_lines) != atom_count:
        raise ValueError(
            f"Expected {atom_count} atomic positions, found {len(coord_lines)}."
        )

    coords = np.array(
        [[float(value) for value in line.split()[:3]] for line in coord_lines],
        dtype=np.float64,
    )

    if coordinate_mode.startswith("d"):
        fractional_coords = coords
        cartesian_coords = coords @ lattice
    else:
        cartesian_coords = coords * scale
        fractional_coords = np.linalg.solve(lattice.T, cartesian_coords.T).T

    species_labels: List[str] = []
    for element, count in zip(species, counts):
        species_labels.extend([element] * count)

    return PoscarData(
        title=title,
        lattice=lattice,
        species=species,
        counts=counts,
        fractional_coords=fractional_coords,
        cartesian_coords=cartesian_coords,
        species_labels=species_labels,
        coordinate_mode="direct" if coordinate_mode.startswith("d") else "cartesian",
    )


def read_poscar(path: Union[str, Path], *, is_periodic: bool = True) -> ChemicalStructure:
    """Read a POSCAR/VASP file into a :class:`ChemicalStructure`."""
    path = Path(path)
    data = parse_poscar(path.read_text(), title_fallback=path.stem)
    return ChemicalStructure.with_zero_magnetic_moments(
        name=path.stem,
        lattice=data.lattice,
        cartesian_coords=data.cartesian_coords,
        atomic_labels=data.species_labels,
        is_periodic=is_periodic,
    )
