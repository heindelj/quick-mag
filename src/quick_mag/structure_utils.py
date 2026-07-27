from __future__ import annotations

from pathlib import Path
from typing import Union

from quick_mag.cif_io import read_cif
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
