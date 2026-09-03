"""Read and write VASP structure files (POSCAR/CONTCAR, ``.vasp``).

Reading turns a POSCAR into a :class:`~quick_mag.structure.ChemicalStructure`;
writing goes the other way. The writer never reorders atoms -- a POSCAR always
comes out in the structure's own atom order, so it lines up line-for-line with a
CIF of the same structure and with the magmom lines
:mod:`quick_mag.export_utils` writes beside it. Atom order is the one thing a
MAGMOM has to agree with, and silently sorting it is how a spin ends up on the
wrong site.

VASP itself only needs the *species blocks* on lines 6 and 7 to describe the atom
order, and it accepts a symbol appearing in more than one block, so an ungrouped
structure (a rocksalt double perovskite, whose B block alternates B and B') still
writes a valid POSCAR. Its POTCAR would have to repeat entries to match, though,
so callers who want one block per element reorder first with
:func:`grouped_by_species` and write everything from that.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

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


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def species_blocks(symbols: Sequence[str]) -> List[Tuple[str, int]]:
    """``["La","La","Fe","Mn","Fe"]`` -> ``[("La",2),("Fe",1),("Mn",1),("Fe",1)]``.

    One block per *contiguous run*, which is exactly what POSCAR lines 6 and 7
    encode. An element that appears in more than one block means the structure is
    not grouped by species; see :func:`grouped_by_species`.
    """
    blocks: List[Tuple[str, int]] = []
    for symbol in symbols:
        if blocks and blocks[-1][0] == symbol:
            blocks[-1] = (symbol, blocks[-1][1] + 1)
        else:
            blocks.append((str(symbol), 1))
    return blocks


def is_grouped_by_species(structure: ChemicalStructure) -> bool:
    """Whether every element occupies one contiguous run of atoms."""
    blocks = species_blocks(structure.element_symbols())
    return len(blocks) == len({symbol for symbol, _count in blocks})


def species_grouping_permutation(structure: ChemicalStructure) -> np.ndarray:
    """Atom indices that group ``structure`` by element, first appearance first.

    The permutation is returned rather than applied so a caller holding anything
    else in atom order -- magmoms above all -- can reorder it the same way.
    :func:`grouped_by_species` applies it to the structure itself.
    """
    symbols = list(structure.element_symbols())
    elements: List[str] = []
    for symbol in symbols:
        if symbol not in elements:
            elements.append(symbol)
    order: List[int] = []
    for element in elements:
        order.extend(index for index, symbol in enumerate(symbols) if symbol == element)
    return np.asarray(order, dtype=int)


def grouped_by_species(structure: ChemicalStructure) -> ChemicalStructure:
    """A copy of ``structure`` with its atoms grouped into one block per element.

    Magnetic moments and saved spin configurations are permuted along with the
    atoms, so the copy stays self-consistent. ``generation_parameters`` is dropped:
    it describes the builder's atom order, and site indexing read through it after
    a reorder would land on the wrong sites.
    """
    from dataclasses import replace  # local: keeps the module import-light

    order = species_grouping_permutation(structure)
    reordered = ChemicalStructure(
        name=structure.name,
        lattice=np.array(structure.lattice, dtype=np.float64, copy=True),
        cartesian_coords=np.asarray(structure.cartesian_coords, dtype=np.float64)[order],
        atomic_labels=[structure.atomic_labels[index] for index in order],
        magnetic_moments=np.asarray(structure.magnetic_moments, dtype=np.float64)[order],
        is_periodic=structure.is_periodic,
        periodic_axes=getattr(structure, "periodic_axes", None),
    )
    for config in getattr(structure, "spin_configurations", []) or []:
        magnitudes = getattr(config, "site_moment_magnitudes", None)
        reordered.spin_configurations.append(
            replace(
                config,
                magnetic_moments=np.asarray(config.magnetic_moments)[order],
                site_moment_magnitudes=(
                    None if magnitudes is None else np.asarray(magnitudes)[order]
                ),
            )
        )
    return reordered


def poscar_text(
    structure: ChemicalStructure,
    *,
    comment: Optional[str] = None,
    direct: bool = True,
) -> str:
    """Render ``structure`` as POSCAR text, in its own atom order.

    ``direct`` writes fractional coordinates wrapped into the cell -- the same
    ``% 1.0`` :func:`quick_mag.cif_io.write_cif` applies, so a POSCAR and a CIF of
    one structure place their atoms identically. ``direct=False`` writes Cartesian
    coordinates instead, which is what a non-periodic structure wants.
    """
    lattice = np.asarray(structure.lattice, dtype=np.float64)
    blocks = species_blocks(structure.element_symbols())

    lines = [comment if comment is not None else (structure.name or "structure"), "1.0"]
    lines += ["  " + "  ".join(f"{value:>18.12f}" for value in row) for row in lattice]
    lines.append("  " + "  ".join(f"{symbol:>4}" for symbol, _count in blocks))
    lines.append("  " + "  ".join(f"{count:>4d}" for _symbol, count in blocks))

    if direct:
        lines.append("Direct")
        coords = np.asarray(structure.fractional_coords, dtype=np.float64) % 1.0
    else:
        lines.append("Cartesian")
        coords = np.asarray(structure.cartesian_coords, dtype=np.float64)
    lines += ["  " + "  ".join(f"{value:>18.12f}" for value in row) for row in coords]
    return "\n".join(lines) + "\n"


def write_poscar(
    structure: ChemicalStructure,
    path: Union[str, Path],
    *,
    comment: Optional[str] = None,
    direct: bool = True,
) -> None:
    """Write ``structure`` to ``path`` as a POSCAR. See :func:`poscar_text`."""
    Path(path).write_text(poscar_text(structure, comment=comment, direct=direct))
