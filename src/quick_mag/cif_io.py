"""Minimal, dependency-free CIF reader/writer (P1 only).

Replaces the pymatgen CIF I/O the project used. Only the subset needed here is
supported: a single ``P 1`` block with cell parameters and an ``_atom_site_*``
loop of fractional coordinates. Atom order is preserved on write (so exported
magmom lines stay aligned) and on read.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Union

import numpy as np

from quick_mag.structure import ChemicalStructure

# Leading element symbol from a CIF type-symbol / label token ("Fe" <- "Fe2+").
_SYMBOL_RE = re.compile(r"([A-Z][a-z]?)")
# A bare number, dropping any CIF standard-uncertainty suffix like "0.5(3)".
_NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def lattice_parameters(lattice: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Return (a, b, c, alpha, beta, gamma) with angles in degrees."""
    vectors = np.asarray(lattice, dtype=np.float64)
    lengths = np.linalg.norm(vectors, axis=1)
    a, b, c = (float(x) for x in lengths)

    def angle(u: np.ndarray, v: np.ndarray) -> float:
        cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

    alpha = angle(vectors[1], vectors[2])
    beta = angle(vectors[0], vectors[2])
    gamma = angle(vectors[0], vectors[1])
    return a, b, c, alpha, beta, gamma


def lattice_from_parameters(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Standard-orientation lattice matrix (rows = vectors) from cell parameters."""
    alpha_r, beta_r, gamma_r = np.radians([alpha, beta, gamma])
    cos_a, cos_b, cos_g = np.cos([alpha_r, beta_r, gamma_r])
    sin_g = np.sin(gamma_r)

    cx = c * cos_b
    cy = c * (cos_a - cos_b * cos_g) / sin_g
    cz = np.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array(
        [
            [a, 0.0, 0.0],
            [b * cos_g, b * sin_g, 0.0],
            [cx, cy, cz],
        ],
        dtype=np.float64,
    )


def write_cif(structure: ChemicalStructure, path: Union[str, Path]) -> None:
    """Write ``structure`` as a P1 CIF, preserving atom order."""
    path = Path(path)
    a, b, c, alpha, beta, gamma = lattice_parameters(structure.lattice)
    fractional = np.asarray(structure.fractional_coords, dtype=np.float64) % 1.0
    symbols = structure.element_symbols()

    lines: List[str] = [
        f"data_{structure.name.replace(' ', '_') or 'structure'}",
        "_symmetry_space_group_name_H-M   'P 1'",
        f"_cell_length_a   {a:.8f}",
        f"_cell_length_b   {b:.8f}",
        f"_cell_length_c   {c:.8f}",
        f"_cell_angle_alpha   {alpha:.8f}",
        f"_cell_angle_beta   {beta:.8f}",
        f"_cell_angle_gamma   {gamma:.8f}",
        "_symmetry_Int_Tables_number   1",
        "loop_",
        " _symmetry_equiv_pos_site_id",
        " _symmetry_equiv_pos_as_xyz",
        "  1  'x, y, z'",
        "loop_",
        " _atom_site_type_symbol",
        " _atom_site_label",
        " _atom_site_symmetry_multiplicity",
        " _atom_site_fract_x",
        " _atom_site_fract_y",
        " _atom_site_fract_z",
        " _atom_site_occupancy",
    ]

    counts: dict[str, int] = {}
    for symbol, (fx, fy, fz) in zip(symbols, fractional):
        label = f"{symbol}{counts.get(symbol, 0)}"
        counts[symbol] = counts.get(symbol, 0) + 1
        lines.append(
            f"  {symbol}  {label}  1  {fx:.8f}  {fy:.8f}  {fz:.8f}  1"
        )

    path.write_text("\n".join(lines) + "\n")


def read_cif(path: Union[str, Path], *, is_periodic: bool = True) -> ChemicalStructure:
    """Read a P1 CIF into a ``ChemicalStructure`` (fractional coords -> cartesian)."""
    path = Path(path)
    raw_lines = path.read_text().splitlines()
    # Drop full-line comments and blank-strip; keep structure for loop parsing.
    lines = [line for line in raw_lines if not line.lstrip().startswith("#")]

    cell: dict[str, float] = {}
    symbols: List[str] = []
    fractional: List[List[float]] = []

    i = 0
    n = len(lines)
    while i < n:
        stripped = lines[i].strip()

        if stripped.startswith("_cell_length_") or stripped.startswith("_cell_angle_"):
            tokens = stripped.split()
            if len(tokens) >= 2:
                match = _NUMBER_RE.search(tokens[1])
                if match:
                    cell[tokens[0]] = float(match.group())
            i += 1
            continue

        if stripped == "loop_":
            headers: List[str] = []
            i += 1
            while i < n and lines[i].strip().startswith("_"):
                headers.append(lines[i].strip())
                i += 1
            if any(h.startswith("_atom_site_") for h in headers):
                col = {name: idx for idx, name in enumerate(headers)}
                sym_col = col.get("_atom_site_type_symbol", col.get("_atom_site_label"))
                fx_col = col.get("_atom_site_fract_x")
                fy_col = col.get("_atom_site_fract_y")
                fz_col = col.get("_atom_site_fract_z")
                while i < n:
                    row = lines[i].strip()
                    if not row or row == "loop_" or row.startswith("_"):
                        break
                    tokens = row.split()
                    if len(tokens) >= len(headers):
                        symbol_match = _SYMBOL_RE.match(tokens[sym_col])
                        symbols.append(
                            symbol_match.group(1) if symbol_match else tokens[sym_col]
                        )
                        fractional.append(
                            [
                                float(_NUMBER_RE.search(tokens[fx_col]).group()),
                                float(_NUMBER_RE.search(tokens[fy_col]).group()),
                                float(_NUMBER_RE.search(tokens[fz_col]).group()),
                            ]
                        )
                    i += 1
            # else: unrelated loop, already advanced past its headers.
            continue

        i += 1

    lattice = lattice_from_parameters(
        cell["_cell_length_a"],
        cell["_cell_length_b"],
        cell["_cell_length_c"],
        cell["_cell_angle_alpha"],
        cell["_cell_angle_beta"],
        cell["_cell_angle_gamma"],
    )
    frac_array = np.asarray(fractional, dtype=np.float64).reshape(-1, 3)
    cartesian = frac_array @ lattice
    return ChemicalStructure.with_zero_magnetic_moments(
        name=path.stem,
        lattice=lattice,
        cartesian_coords=cartesian,
        atomic_labels=symbols,
        is_periodic=is_periodic,
    )
