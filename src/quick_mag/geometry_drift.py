"""How far a structure has moved from the geometry it was generated as.

A relaxation keeps a structure's provenance -- the atom order and the B-site grid
its ``generation_parameters`` describe are untouched -- but not its geometry. The
builder can no longer rebuild it, and the honest thing to show instead of the
builder's now-stale numbers is *how far it drifted*: what the cell did, how far
the atoms moved, and whether the cell is still something the builder could
express at all.

Pure numpy and stdlib, so this is importable in the Pyodide build.

What it does not do yet: recover builder parameters from an arbitrary geometry.
Fitting a tilt system and a lattice constant back out of a relaxed cell is a real
inverse problem, and until it exists the report below is the thing standing in
for it -- it says how bad the mismatch is rather than trying to absorb it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from quick_mag.structure import ChemicalStructure

# Below this, a difference is numerical noise rather than a change. Matches the
# tolerance ``structures_match_geometry`` uses for the same question.
EXACT_TOLERANCE = 1e-6

# Off-diagonal lattice components above this mean the cell has sheared into
# something the builder -- which only ever emits diagonal cells -- cannot express.
ORTHOGONAL_TOLERANCE = 1e-4


def cell_lengths(lattice: np.ndarray) -> Tuple[float, float, float]:
    lengths = np.linalg.norm(np.asarray(lattice, dtype=np.float64), axis=1)
    return (float(lengths[0]), float(lengths[1]), float(lengths[2]))


def is_orthogonal(lattice: np.ndarray) -> bool:
    """True when the cell is diagonal to within tolerance."""
    matrix = np.asarray(lattice, dtype=np.float64)
    off_diagonal = matrix - np.diag(np.diag(matrix))
    return bool(np.abs(off_diagonal).max() <= ORTHOGONAL_TOLERANCE)


@dataclass(frozen=True)
class GeometryDrift:
    """The difference between a structure and the one it was made from."""

    atom_count: int
    #: Root-mean-square atomic displacement, in Angstrom.
    rmsd: float
    #: The largest single-atom displacement, in Angstrom.
    max_displacement: float
    #: Index of the atom that moved furthest, for pointing at it in the view.
    max_displacement_index: int
    lengths_before: Tuple[float, float, float]
    lengths_after: Tuple[float, float, float]
    volume_ratio: float
    cell_is_orthogonal: bool
    comparable: bool = True

    @property
    def cell_changed(self) -> bool:
        return any(
            abs(after - before) > EXACT_TOLERANCE
            for before, after in zip(self.lengths_before, self.lengths_after)
        )

    @property
    def atoms_moved(self) -> bool:
        return self.max_displacement > EXACT_TOLERANCE

    @property
    def length_deltas(self) -> Tuple[float, float, float]:
        return tuple(
            after - before
            for before, after in zip(self.lengths_before, self.lengths_after)
        )  # type: ignore[return-value]

    def cell_summary(self) -> str:
        """``a b c`` before and after, which is the line people actually read."""
        before = "  ".join(f"{value:.4f}" for value in self.lengths_before)
        after = "  ".join(f"{value:.4f}" for value in self.lengths_after)
        return f"{before}  ->  {after}"

    def atom_summary(self) -> str:
        return (
            f"RMSD {self.rmsd:.4f} A, max {self.max_displacement:.4f} A "
            f"(atom {self.max_displacement_index + 1})"
        )

    def headline(self) -> str:
        """One line for a panel that has room for exactly one line."""
        if not self.comparable:
            return "Geometry changed (atom count differs; not comparable)."
        parts = []
        if self.cell_changed:
            volume = (self.volume_ratio - 1.0) * 100.0
            parts.append(f"cell {volume:+.2f}% by volume")
        if self.atoms_moved:
            parts.append(f"atoms RMSD {self.rmsd:.4f} A")
        if not parts:
            return "Geometry unchanged."
        return "Drift: " + ", ".join(parts) + "."


def geometry_drift(
    reference: ChemicalStructure, current: ChemicalStructure
) -> Optional[GeometryDrift]:
    """Compare ``current`` against the structure it came from.

    Displacements are compared in cartesian space and atom-by-atom in the order
    both structures share -- every path that produces a relaxed structure
    preserves atom order, which is exactly what makes this well defined. Returns
    None when there is nothing to compare against.

    Minimum-image wrapping is deliberately *not* applied: an atom that crossed a
    cell face during a relaxation genuinely moved that far in the trajectory, and
    silently folding it back would report a large real displacement as a small
    one.
    """
    if reference is None or current is None:
        return None

    before = cell_lengths(reference.lattice)
    after = cell_lengths(current.lattice)
    reference_volume = abs(float(np.linalg.det(np.asarray(reference.lattice, float))))
    current_volume = abs(float(np.linalg.det(np.asarray(current.lattice, float))))
    volume_ratio = (
        current_volume / reference_volume if reference_volume > 0.0 else float("nan")
    )

    if reference.atom_count != current.atom_count:
        return GeometryDrift(
            atom_count=current.atom_count,
            rmsd=float("nan"),
            max_displacement=float("nan"),
            max_displacement_index=-1,
            lengths_before=before,
            lengths_after=after,
            volume_ratio=volume_ratio,
            cell_is_orthogonal=is_orthogonal(current.lattice),
            comparable=False,
        )

    deltas = np.asarray(current.cartesian_coords, dtype=np.float64) - np.asarray(
        reference.cartesian_coords, dtype=np.float64
    )
    distances = np.linalg.norm(deltas, axis=1) if deltas.size else np.zeros(0)
    return GeometryDrift(
        atom_count=current.atom_count,
        rmsd=float(np.sqrt((distances**2).mean())) if distances.size else 0.0,
        max_displacement=float(distances.max()) if distances.size else 0.0,
        max_displacement_index=int(distances.argmax()) if distances.size else -1,
        lengths_before=before,
        lengths_after=after,
        volume_ratio=volume_ratio,
        cell_is_orthogonal=is_orthogonal(current.lattice),
    )


def builder_can_express(structure: ChemicalStructure) -> bool:
    """Whether the builder could even emit this cell shape.

    ``build_perovskite`` produces ``diag(2dx, 2dy, 2dz)`` and nothing else, so a
    cell relaxation that shears the lattice puts the structure permanently
    outside what any choice of builder parameters can reproduce. Worth saying out
    loud rather than leaving as a silent mismatch.
    """
    return is_orthogonal(structure.lattice)


__all__ = [
    "EXACT_TOLERANCE",
    "GeometryDrift",
    "builder_can_express",
    "cell_lengths",
    "geometry_drift",
    "is_orthogonal",
]
