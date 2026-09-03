"""Selecting atoms with a slab, and editing a loaded structure atom by atom.

Two things live here, both pure functions of arrays so the UI can stay thin.

The **selection slab** is a layer of finite thickness cut through the cell,
oriented by a lattice direction rather than a Cartesian one: ``[u v w]`` names
the vector ``u*a + v*b + w*c`` in the structure's own lattice, so ``[1 0 0]``
points along *a*, ``[1 1 0]`` toward the edge between *a* and *b*, and
``[1 1 1]`` toward the far vertex -- and the same three integers keep meaning
that when the cell is triclinic, where "the x axis" would not. Sliding the slab
along its normal and asking which atoms fall inside it is how a whole layer is
picked at once.

The **loaded-structure edits** are the counterparts of the builder's defects
for a structure that was not built: one that came from a file or a relaxation
and therefore has no grid to address a site by. A built structure is edited by
rewriting its defect list and regenerating; a loaded one has nothing to
regenerate from, so its atoms are edited in place -- relabelled, removed, or
given a proton -- and the vacated positions are remembered so a vacancy can be
put back.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from quick_mag.defects import DEFAULT_OH_BOND_LENGTH, PROTON_SYMBOL

# Bond lengths this far apart or less count as "bonded" when a proton's O-H
# direction is chosen from an oxygen's surroundings.
PROTON_NEIGHBOR_CUTOFF = 2.6


# ---------------------------------------------------------------------------
# Selection slab
# ---------------------------------------------------------------------------


@dataclass
class SelectionSlab:
    """A layer through the cell, addressed in lattice terms.

    ``direction`` is the ``[u v w]`` lattice direction of the normal.
    ``offset`` is where the slab's mid-plane sits, as a fraction of the cell's
    extent along that normal -- 0 is the low face, 1 the high one -- so the
    same offset means the same relative position when the cell is resized.
    ``thickness`` is in Angstrom, which is what a layer spacing is measured in.
    """

    direction: Tuple[int, int, int] = (0, 0, 1)
    offset: float = 0.5
    thickness: float = 1.0

    def direction_tuple(self) -> Tuple[int, int, int]:
        values = [int(value) for value in list(self.direction)[:3]]
        while len(values) < 3:
            values.append(0)
        return (values[0], values[1], values[2])


def slab_normal(lattice: np.ndarray, direction: Sequence[int]) -> Optional[np.ndarray]:
    """Unit Cartesian normal for lattice direction ``[u v w]``, or None for [0 0 0]."""
    rows = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    uvw = np.asarray([int(value) for value in direction], dtype=np.float64)
    vector = uvw @ rows
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return None
    return vector / norm


def cell_extent_along(
    lattice: np.ndarray, normal: np.ndarray, origin: Sequence[float] = (0.0, 0.0, 0.0)
) -> Tuple[float, float]:
    """``(low, high)`` projections of the cell's eight corners onto ``normal``."""
    rows = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    base = np.asarray(origin, dtype=np.float64).reshape(3)
    corners = np.array(
        [
            base + i * rows[0] + j * rows[1] + k * rows[2]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ],
        dtype=np.float64,
    )
    projected = corners @ np.asarray(normal, dtype=np.float64)
    return float(projected.min()), float(projected.max())


def slab_center_distance(
    lattice: np.ndarray,
    slab: SelectionSlab,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> Optional[float]:
    """Projection of the slab's mid-plane onto its normal, or None for [0 0 0]."""
    normal = slab_normal(lattice, slab.direction_tuple())
    if normal is None:
        return None
    low, high = cell_extent_along(lattice, normal, origin)
    return low + float(slab.offset) * (high - low)


def atoms_in_slab(
    cartesian_coords: np.ndarray,
    lattice: np.ndarray,
    slab: SelectionSlab,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> List[int]:
    """Indices of the atoms whose centres lie within the slab, ascending.

    The test is a half-open band ``|d - centre| <= thickness / 2`` on the
    projection onto the normal, with a little tolerance so a layer that sits
    exactly on the slab's mid-plane is caught rather than lost to rounding.
    """
    coords = np.asarray(cartesian_coords, dtype=np.float64).reshape(-1, 3)
    if coords.shape[0] == 0:
        return []
    normal = slab_normal(lattice, slab.direction_tuple())
    if normal is None:
        return []
    center = slab_center_distance(lattice, slab, origin)
    if center is None:
        return []
    half = max(0.0, float(slab.thickness)) * 0.5 + 1e-6
    distance = coords @ normal
    return [int(index) for index in np.flatnonzero(np.abs(distance - center) <= half)]


def slab_face_corners(
    lattice: np.ndarray,
    slab: SelectionSlab,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """The two bounding faces of the slab, each as four Cartesian corners.

    The faces are drawn as rectangles perpendicular to the normal, sized to
    cover the cell's projection onto the plane: large enough that every atom
    the slab can select is visibly between them, whatever the cell's shape.
    Returns ``(low_face, high_face)`` or None when there is no slab.
    """
    normal = slab_normal(lattice, slab.direction_tuple())
    if normal is None:
        return None
    center = slab_center_distance(lattice, slab, origin)
    if center is None:
        return None
    rows = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    base = np.asarray(origin, dtype=np.float64).reshape(3)
    corners = np.array(
        [
            base + i * rows[0] + j * rows[1] + k * rows[2]
            for i in (0, 1)
            for j in (0, 1)
            for k in (0, 1)
        ],
        dtype=np.float64,
    )
    # Two in-plane axes: the lattice row least aligned with the normal, made
    # perpendicular to it, and the cross product.
    alignment = [abs(float(np.dot(row / (np.linalg.norm(row) or 1.0), normal))) for row in rows]
    seed = rows[int(np.argmin(alignment))]
    axis_u = seed - float(np.dot(seed, normal)) * normal
    norm_u = float(np.linalg.norm(axis_u))
    if norm_u < 1e-12:
        return None
    axis_u /= norm_u
    axis_v = np.cross(normal, axis_u)
    in_plane = corners - np.outer(corners @ normal, normal)
    u_values = in_plane @ axis_u
    v_values = in_plane @ axis_v
    u_low, u_high = float(u_values.min()), float(u_values.max())
    v_low, v_high = float(v_values.min()), float(v_values.max())
    half = max(0.0, float(slab.thickness)) * 0.5

    def face(distance: float) -> np.ndarray:
        anchor = distance * normal
        return np.array(
            [
                anchor + u_low * axis_u + v_low * axis_v,
                anchor + u_high * axis_u + v_low * axis_v,
                anchor + u_high * axis_u + v_high * axis_v,
                anchor + u_low * axis_u + v_high * axis_v,
            ],
            dtype=np.float64,
        )

    return face(center - half), face(center + half)


def slab_offset_from_distance(
    lattice: np.ndarray,
    slab: SelectionSlab,
    distance: float,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> float:
    """The offset that would put the mid-plane at ``distance`` along the normal.

    Clamped to the cell, so a drag cannot push the slab out of sight.
    """
    normal = slab_normal(lattice, slab.direction_tuple())
    if normal is None:
        return float(slab.offset)
    low, high = cell_extent_along(lattice, normal, origin)
    span = high - low
    if span <= 1e-12:
        return float(slab.offset)
    return float(min(1.0, max(0.0, (float(distance) - low) / span)))


# ---------------------------------------------------------------------------
# Loaded-structure edits
# ---------------------------------------------------------------------------


@dataclass
class VacatedAtom:
    """An atom removed from a loaded structure, kept so it can be put back."""

    label: str
    cartesian: np.ndarray
    magnetic_moment: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )


def substitute_atom(structure, index: int, element: str) -> None:
    """Relabel one atom of ``structure`` in place.

    The label is replaced outright rather than patched: an oxidation-state
    suffix on the old label described the old element.
    """
    element = str(element).strip()
    if not element:
        raise ValueError("A substitution needs an element symbol.")
    if not 0 <= int(index) < structure.atom_count:
        raise IndexError(f"No atom {index} in {structure.name}.")
    labels = list(structure.atomic_labels)
    labels[int(index)] = element
    structure.atomic_labels = labels


def remove_atom(structure, index: int) -> VacatedAtom:
    """Take one atom out of ``structure`` in place and return what was removed.

    Every array shrinks together, so the structure is always consistent; the
    caller is responsible for anything indexed by atom that it holds elsewhere.
    """
    index = int(index)
    if not 0 <= index < structure.atom_count:
        raise IndexError(f"No atom {index} in {structure.name}.")
    coords = np.asarray(structure.cartesian_coords, dtype=np.float64).reshape(-1, 3)
    moments = np.asarray(structure.magnetic_moments, dtype=np.float64).reshape(-1, 3)
    removed = VacatedAtom(
        label=str(structure.atomic_labels[index]),
        cartesian=coords[index].copy(),
        magnetic_moment=moments[index].copy(),
    )
    keep = np.ones(coords.shape[0], dtype=bool)
    keep[index] = False
    structure.cartesian_coords = coords[keep]
    structure.magnetic_moments = moments[keep]
    structure.atomic_labels = [
        label for position, label in enumerate(structure.atomic_labels) if keep[position]
    ]
    return removed


def append_atom(
    structure,
    label: str,
    cartesian: Sequence[float],
    magnetic_moment: Sequence[float] = (0.0, 0.0, 0.0),
) -> int:
    """Add one atom to the end of ``structure`` in place; returns its index."""
    coords = np.asarray(structure.cartesian_coords, dtype=np.float64).reshape(-1, 3)
    moments = np.asarray(structure.magnetic_moments, dtype=np.float64).reshape(-1, 3)
    structure.cartesian_coords = np.vstack(
        (coords, np.asarray(cartesian, dtype=np.float64).reshape(1, 3))
    )
    structure.magnetic_moments = np.vstack(
        (moments, np.asarray(magnetic_moment, dtype=np.float64).reshape(1, 3))
    )
    structure.atomic_labels = list(structure.atomic_labels) + [str(label)]
    return structure.atom_count - 1


def proton_direction_for_host(
    structure,
    host_index: int,
    *,
    cutoff: float = PROTON_NEIGHBOR_CUTOFF,
) -> np.ndarray:
    """A unit O-H direction for a proton on atom ``host_index``.

    Without a grid to read the octahedron from, the direction comes from the
    host's own surroundings: a proton sits away from the cations its oxygen
    is bonded to, so the direction is the negative of the summed unit bond
    vectors to every neighbour within ``cutoff``. When that sum vanishes -- a
    linear M-O-M bridge, or an isolated atom -- any direction perpendicular
    to the bridge is as good as another, and the first one found is taken.
    """
    coords = np.asarray(structure.cartesian_coords, dtype=np.float64).reshape(-1, 3)
    host = coords[int(host_index)]
    bonds: List[np.ndarray] = []
    for neighbor in structure.neighbors(int(host_index), float(cutoff)):
        vector = np.asarray(neighbor.coords, dtype=np.float64) - host
        norm = float(np.linalg.norm(vector))
        if norm > 1e-9:
            bonds.append(vector / norm)
    if bonds:
        summed = -np.sum(np.asarray(bonds), axis=0)
        norm = float(np.linalg.norm(summed))
        if norm > 1e-3:
            return summed / norm
        # Linear bridge: pick a direction perpendicular to it.
        axis = bonds[0]
        trial = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(trial, axis))) > 0.9:
            trial = np.array([0.0, 1.0, 0.0])
        perpendicular = trial - float(np.dot(trial, axis)) * axis
        return perpendicular / float(np.linalg.norm(perpendicular))
    return np.array([0.0, 0.0, 1.0])


def add_proton(
    structure,
    host_index: int,
    *,
    bond_length: float = DEFAULT_OH_BOND_LENGTH,
) -> int:
    """Attach a proton to atom ``host_index`` of ``structure``; returns its index."""
    host_index = int(host_index)
    if not 0 <= host_index < structure.atom_count:
        raise IndexError(f"No atom {host_index} in {structure.name}.")
    direction = proton_direction_for_host(structure, host_index)
    host = np.asarray(structure.cartesian_coords, dtype=np.float64)[host_index]
    return append_atom(structure, PROTON_SYMBOL, host + float(bond_length) * direction)
