"""Export structures to disk: a geometry per structure plus a VASP-format
``<name>_spins.txt`` holding one magmom line per saved magnetic configuration.

The geometry is a CIF by default and a POSCAR (``<name>.vasp``) on request; see
``formats`` on :func:`export_structure`. Either is written with the original atom
order, so the magmom lines (in structure atom order) line up with its atoms.

Magmoms are written as formal moments in mu_B: the solver works in unit spins
(magnitude lives in the exchange couplings), so a saved configuration carrying
per-site magnitudes is rescaled on the way out -- an Fe(3+) site writes +-5.0
rather than +-1.0.

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
from quick_mag.vasp_io import grouped_by_species, write_poscar


#: Structure formats ``export_structure`` can write. ``"cif"`` writes
#: ``<stem>.cif``, ``"vasp"`` writes ``<stem>.vasp`` (a POSCAR).
STRUCTURE_FORMATS = ("cif", "vasp")
DEFAULT_STRUCTURE_FORMATS = ("cif",)

#: The CLI's ``--format`` values, each mapping to ``export_structure`` keywords.
#: Writing a POSCAR groups the atoms into one contiguous block per element -- the
#: arrangement its species blocks and a matching POTCAR need -- so every file the
#: export writes is reordered together and stays line-for-line comparable.
FORMAT_CHOICES = {
    "cif": {"formats": ("cif",), "group_species": False},
    "vasp": {"formats": ("vasp",), "group_species": True},
    "both": {"formats": ("cif", "vasp"), "group_species": True},
}

FORMAT_HELP = (
    "Geometry file(s) to write: 'cif' (default), 'vasp' for a POSCAR, or 'both'. "
    "Any 'vasp' output groups the atoms into one block per element, which reorders "
    "them; the CIF and the magmom file written alongside follow the same order."
)


def export_options(args) -> Dict[str, object]:
    """``formats``/``group_species`` keywords for a namespace's ``--format``."""
    return dict(FORMAT_CHOICES[getattr(args, "format", "cif")])


def sanitize_filename(name: str) -> str:
    """Make ``name`` safe to use as a file/folder stem."""
    cleaned = re.sub(r"\s+", "_", name.strip())
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", cleaned)
    return cleaned or "unnamed"


def scale_to_site_magnitudes(
    moments: np.ndarray,
    magnitudes: np.ndarray | None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Rescale each moment vector to its site's formal magnitude, keeping direction.

    ``magnitudes`` is one unsigned |μ| per atom (μ_B); ``None`` -- or a length that
    does not match the moments -- leaves the vectors untouched. Rescaling rather
    than multiplying makes this idempotent, so a configuration that already carries
    physical moments is not squared. Sites with no direction (a zero vector) and
    sites with zero magnitude both stay at zero.
    """
    vectors = np.asarray(moments, dtype=np.float64).reshape(-1, 3)
    if magnitudes is None:
        return vectors
    site_magnitudes = np.asarray(magnitudes, dtype=np.float64).reshape(-1)
    if len(site_magnitudes) != len(vectors):
        return vectors

    norms = np.linalg.norm(vectors, axis=1)
    scales = np.zeros_like(norms)
    nonzero = norms > eps
    scales[nonzero] = np.abs(site_magnitudes[nonzero]) / norms[nonzero]
    return vectors * scales[:, None]


def format_magmom_line(
    moments: np.ndarray,
    collinear: bool,
    magnitudes: np.ndarray | None = None,
    eps: float = 1e-8,
) -> str:
    """Format one configuration's magmoms as a single VASP MAGMOM line.

    ``collinear`` controls the width (not the moment geometry): collinear emits one
    signed scalar per atom (the projection onto the dominant spin axis, which reduces
    to ±m_z for z-aligned spins); non-collinear emits ``mx my mz`` per atom.

    ``magnitudes`` optionally carries the formal per-site moment each direction is
    scaled to; see :func:`scale_to_site_magnitudes`.
    """
    vectors = scale_to_site_magnitudes(moments, magnitudes, eps=eps)
    if not collinear:
        return " ".join(f"{value:.6f}" for value in vectors.reshape(-1))

    norms = np.linalg.norm(vectors, axis=1)
    if norms.size == 0 or float(norms.max()) <= eps:
        return " ".join("0.000000" for _ in range(len(vectors)))
    axis = vectors[int(np.argmax(norms))]
    axis = axis / np.linalg.norm(axis)
    projections = vectors @ axis
    return " ".join(f"{value:.6f}" for value in projections)


def export_structure(
    structure: ChemicalStructure,
    out_dir: Path,
    *,
    formats: Sequence[str] = DEFAULT_STRUCTURE_FORMATS,
    group_species: bool = False,
) -> Dict[str, int]:
    """Write one structure's geometry and (when present) ``<stem>_spins.txt``.

    ``formats`` selects the geometry files: ``"cif"`` for ``<stem>.cif``, ``"vasp"``
    for ``<stem>.vasp`` (a POSCAR), or both. Whatever is written comes out in one
    shared atom order, which is what lets line *i* of the magmom file mean atom *i*
    of every geometry beside it.

    ``group_species`` reorders the atoms into one contiguous block per element
    first -- the arrangement a POSCAR's species blocks and a matching POTCAR want.
    It is off by default because reordering invalidates builder provenance; turn it
    on when the export's purpose is a VASP run.
    """
    out_dir = Path(out_dir)
    unknown = [name for name in formats if name not in STRUCTURE_FORMATS]
    if unknown:
        raise ValueError(
            f"Unknown structure format(s) {unknown}; choose from {list(STRUCTURE_FORMATS)}."
        )
    if group_species:
        structure = grouped_by_species(structure)

    stem = sanitize_filename(structure.name)
    written = {"cif": 0, "vasp": 0}
    if "cif" in formats:
        write_cif(structure, out_dir / f"{stem}.cif")
        written["cif"] = 1
    if "vasp" in formats:
        write_poscar(structure, out_dir / f"{stem}.vasp", comment=structure.name)
        written["vasp"] = 1

    configs = list(getattr(structure, "spin_configurations", []) or [])
    if configs:
        lines = [
            format_magmom_line(
                config.magnetic_moments,
                config.collinear,
                getattr(config, "site_moment_magnitudes", None),
            )
            for config in configs
        ]
        (out_dir / f"{stem}_spins.txt").write_text("\n".join(lines) + "\n")
    return {**written, "spin_configs": len(configs)}


def export_structures(
    structures: List[ChemicalStructure],
    out_dir: Path,
    *,
    formats: Sequence[str] = DEFAULT_STRUCTURE_FORMATS,
    group_species: bool = False,
) -> Dict[str, int]:
    """Export every structure flat into ``out_dir``, returning aggregate counts.

    ``formats`` and ``group_species`` are passed through to
    :func:`export_structure`.
    """
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary = {"structures": 0, "cif": 0, "vasp": 0, "spin_configs": 0}
    for structure in structures:
        result = export_structure(
            structure, target, formats=formats, group_species=group_species
        )
        summary["structures"] += 1
        summary["cif"] += result["cif"]
        summary["vasp"] += result["vasp"]
        summary["spin_configs"] += result["spin_configs"]
    return summary


# Content types for the two things an export can hand back. The CIF type is the
# registered one; browsers treat anything they do not recognise as a download either
# way, but naming it correctly keeps the saved file associated properly.
CIF_MIME_TYPE = "chemical/x-cif"
VASP_MIME_TYPE = "text/plain"
ZIP_MIME_TYPE = "application/zip"
EXPORT_ARCHIVE_NAME = "quick_mag_export.zip"


def export_bundle_bytes(
    structures: Sequence[ChemicalStructure],
    *,
    formats: Sequence[str] = DEFAULT_STRUCTURE_FORMATS,
    group_species: bool = False,
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
        export_structures(
            list(structures), target, formats=formats, group_species=group_species
        )
        files = sorted(path for path in target.iterdir() if path.is_file())
        if not files:
            raise ValueError("Nothing to export.")
        if len(files) == 1:
            mime = CIF_MIME_TYPE if files[0].suffix == ".cif" else VASP_MIME_TYPE
            return files[0].name, files[0].read_bytes(), mime

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                # arcname is the bare filename: export_structures writes flat, and a
                # zip entry must never carry a path component.
                archive.write(path, arcname=path.name)
        return EXPORT_ARCHIVE_NAME, buffer.getvalue(), ZIP_MIME_TYPE
