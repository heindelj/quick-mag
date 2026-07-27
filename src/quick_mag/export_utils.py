"""Export structure groups to disk: a CIF per structure plus a VASP-format
``<name>_spins.txt`` holding one magmom line per saved magnetic configuration.

The CIF is written in P1 with the original atom order, so the magmom lines (in
structure atom order) line up with the CIF atoms.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np

from quick_mag.cif_io import write_cif
from quick_mag.structure import ChemicalStructure, StructureGroup


def sanitize_filename(name: str) -> str:
    """Make ``name`` safe to use as a file/folder stem."""
    cleaned = re.sub(r"\s+", "_", name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    return cleaned or "unnamed"


def format_magmom_line(moments: np.ndarray, collinear: bool, eps: float = 1e-8) -> str:
    """Format one configuration's magmoms as a single VASP MAGMOM line.

    ``collinear`` controls the width (not the moment geometry): collinear emits one
    signed scalar per atom (the projection onto the dominant spin axis, which reduces
    to ±m_z for z-aligned spins); non-collinear emits ``mx my mz`` per atom.
    """
    vectors = np.asarray(moments, dtype=np.float64).reshape(-1, 3)
    if not collinear:
        return " ".join(f"{value:.6f}" for value in vectors.reshape(-1))

    norms = np.linalg.norm(vectors, axis=1)
    if norms.size == 0 or float(norms.max()) <= eps:
        return " ".join("0.000000" for _ in range(len(vectors)))
    axis = vectors[int(np.argmax(norms))]
    axis = axis / np.linalg.norm(axis)
    projections = vectors @ axis
    return " ".join(f"{value:.6f}" for value in projections)


def export_structure(structure: ChemicalStructure, out_dir: Path) -> Dict[str, int]:
    """Write ``<stem>.cif`` and (when present) ``<stem>_spins.txt`` for one structure."""
    out_dir = Path(out_dir)
    stem = sanitize_filename(structure.name)
    cif_path = out_dir / f"{stem}.cif"
    write_cif(structure, cif_path)

    configs = list(getattr(structure, "spin_configurations", []) or [])
    if configs:
        lines = [
            format_magmom_line(config.magnetic_moments, config.collinear)
            for config in configs
        ]
        (out_dir / f"{stem}_spins.txt").write_text("\n".join(lines) + "\n")
    return {"cif": 1, "spin_configs": len(configs)}


def export_group(group: StructureGroup, out_dir: Path) -> Dict[str, int]:
    """Export every structure in ``group`` into ``out_dir/<group name>/``."""
    group_dir = Path(out_dir) / sanitize_filename(group.name)
    group_dir.mkdir(parents=True, exist_ok=True)
    summary = {"structures": 0, "cif": 0, "spin_configs": 0}
    for structure in group.structures:
        result = export_structure(structure, group_dir)
        summary["structures"] += 1
        summary["cif"] += result["cif"]
        summary["spin_configs"] += result["spin_configs"]
    return summary


def export_groups(groups: List[StructureGroup], out_dir: Path) -> Dict[str, int]:
    """Export several groups, returning aggregate counts."""
    summary = {"groups": 0, "structures": 0, "cif": 0, "spin_configs": 0}
    for group in groups:
        result = export_group(group, out_dir)
        summary["groups"] += 1
        summary["structures"] += result["structures"]
        summary["cif"] += result["cif"]
        summary["spin_configs"] += result["spin_configs"]
    return summary
