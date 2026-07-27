from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

from quick_mag.cif_io import read_cif
from quick_mag.constants import METALS
from quick_mag.structure import ChemicalStructure
from quick_mag.vasp_io import read_poscar


def read_structure(path: Union[str, Path]) -> ChemicalStructure:
    """Read a structure file into a :class:`ChemicalStructure`.

    Supports VASP/POSCAR (``.vasp``, ``POSCAR``, ``CONTCAR``) and P1 ``.cif``.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".cif":
        return read_cif(path)
    if suffix == ".vasp" or path.name.upper() in {"POSCAR", "CONTCAR"}:
        return read_poscar(path)
    raise ValueError(
        f"Unsupported structure file '{path.name}'. Supported formats: "
        f"VASP/POSCAR (.vasp, POSCAR, CONTCAR) and P1 .cif."
    )


def find_magnetic_elements(structure: ChemicalStructure) -> list[str]:
    """Return unique magnetic element symbols in sorted order."""
    return sorted({symbol for symbol in structure.element_symbols() if symbol in METALS})


def find_magnetic_indices(
    structure: ChemicalStructure,
    magnetic_elements: Iterable[str] | None = None,
) -> list[int]:
    """Return site indices for magnetic atoms."""
    magnetic_set = set(magnetic_elements or find_magnetic_elements(structure))
    return [
        index
        for index, symbol in enumerate(structure.element_symbols())
        if symbol in magnetic_set
    ]
