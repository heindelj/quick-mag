"""Export structures to disk: a CIF per structure plus a VASP-format
``<name>_spins.txt`` holding one magmom line per saved magnetic configuration.

The CIF is written in P1 with the original atom order, so the magmom lines (in
structure atom order) line up with the CIF atoms.

``export_bundle_bytes`` packages the same output in memory, for the web build --
which has no filesystem to write to and hands the result to the browser instead.
"""

from __future__ import annotations

import io
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from quick_mag.cif_io import write_cif
from quick_mag.structure import ChemicalStructure


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


def export_structures(
    structures: List[ChemicalStructure], out_dir: Path
) -> Dict[str, int]:
    """Export every structure flat into ``out_dir``, returning aggregate counts."""
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary = {"structures": 0, "cif": 0, "spin_configs": 0}
    for structure in structures:
        result = export_structure(structure, target)
        summary["structures"] += 1
        summary["cif"] += result["cif"]
        summary["spin_configs"] += result["spin_configs"]
    return summary


# Content types for the two things an export can hand back. The CIF type is the
# registered one; browsers treat anything they do not recognise as a download either
# way, but naming it correctly keeps the saved file associated properly.
CIF_MIME_TYPE = "chemical/x-cif"
ZIP_MIME_TYPE = "application/zip"
EXPORT_ARCHIVE_NAME = "quick_mag_export.zip"


def export_bundle_bytes(
    structures: Sequence[ChemicalStructure],
) -> Tuple[str, bytes, str]:
    """``(filename, payload, mime_type)`` for an export that cannot go to disk.

    Runs the ordinary :func:`export_structures` against a temporary directory, so the
    bytes handed back are exactly what the desktop app writes. A single file is
    returned as itself; two or more are zipped, because a browser will not accept
    several downloads at once without prompting for each.

    Raises ``ValueError`` when ``structures`` produces no files at all.
    """
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch)
        export_structures(list(structures), target)
        files = sorted(path for path in target.iterdir() if path.is_file())
        if not files:
            raise ValueError("Nothing to export.")
        if len(files) == 1:
            return files[0].name, files[0].read_bytes(), CIF_MIME_TYPE

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                # arcname is the bare filename: export_structures writes flat, and a
                # zip entry must never carry a path component.
                archive.write(path, arcname=path.name)
        return EXPORT_ARCHIVE_NAME, buffer.getvalue(), ZIP_MIME_TYPE
