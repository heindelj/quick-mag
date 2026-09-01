"""Cell edits applied directly to a structure's coordinates.

The builder is a pure function of :class:`PerovskiteGenerationParameters`: every
edit throws the geometry away and rebuilds it from scratch. That is exactly what
a structure loaded from a file cannot survive -- its coordinates *are* the
information, and a relaxed cell rebuilt as an ideal perovskite has lost the
relaxation.

So loaded structures get the other kind of edit: a transformation of the
coordinates they already have. Every function here holds fractional coordinates
fixed and moves the cell around them, which is the operation that needs no
knowledge of the structure at all -- no site roles, no octahedral network, no
perovskite grid. It works on a triclinic CIF and on a molecule in a box alike.

Two properties are load-bearing for the UI:

* **Idempotence.** ``strain_structure`` re-derives fractional coordinates from
  the structure it is given, so applying it every frame with an unchanged target
  is a no-op rather than a slow drift.
* **Orientation preservation.** Rebuilding a lattice from (a, b, c, alpha, beta,
  gamma) picks an arbitrary orientation -- conventionally **a** along x. Doing
  that to a loaded file would silently rotate it the first time an angle is
  nudged. :func:`lattice_from_parameters` instead rebuilds in the canonical
  frame and then re-applies the rotation the original lattice carried, so an
  edit that changes nothing returns the original matrix to floating-point
  precision.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

# Below this a cell edge is not a cell edge. Shared with the UI's input clamps.
MIN_CELL_LENGTH = 0.5
# Angles outside this are degenerate long before they are unphysical; the real
# constraint (a positive-definite metric) is checked separately.
MIN_CELL_ANGLE = 5.0
MAX_CELL_ANGLE = 175.0


def cell_lengths(lattice: np.ndarray) -> Tuple[float, float, float]:
    """Lengths of the three cell vectors, in the lattice's own units."""
    rows = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    return tuple(float(value) for value in np.linalg.norm(rows, axis=1))  # type: ignore[return-value]


def cell_angles(lattice: np.ndarray) -> Tuple[float, float, float]:
    """Cell angles ``(alpha, beta, gamma)`` in degrees.

    Crystallographic convention: ``alpha`` is between **b** and **c**, ``beta``
    between **c** and **a**, ``gamma`` between **a** and **b**.
    """
    rows = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    lengths = np.linalg.norm(rows, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("cell vectors must have non-zero length.")
    angles = []
    for first, second in ((1, 2), (2, 0), (0, 1)):
        cosine = float(rows[first] @ rows[second] / (lengths[first] * lengths[second]))
        angles.append(float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))))
    return angles[0], angles[1], angles[2]


def cell_parameters(lattice: np.ndarray) -> Tuple[float, float, float, float, float, float]:
    """``(a, b, c, alpha, beta, gamma)`` for ``lattice``."""
    return cell_lengths(lattice) + cell_angles(lattice)


def canonical_lattice(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Lower-triangular lattice for the six cell parameters.

    The standard convention: **a** along x, **b** in the xy plane, **c** taking
    up whatever is left. Raises ``ValueError`` when the parameters describe no
    real cell -- an angle triple can be individually plausible and jointly
    impossible, which shows up here as a negative ``c_z**2``.
    """
    if min(a, b, c) < MIN_CELL_LENGTH:
        raise ValueError(f"cell edges must be at least {MIN_CELL_LENGTH} A.")
    if not all(MIN_CELL_ANGLE <= angle <= MAX_CELL_ANGLE for angle in (alpha, beta, gamma)):
        raise ValueError(
            f"cell angles must lie between {MIN_CELL_ANGLE} and {MAX_CELL_ANGLE} degrees."
        )
    cos_alpha = np.cos(np.radians(alpha))
    cos_beta = np.cos(np.radians(beta))
    cos_gamma = np.cos(np.radians(gamma))
    sin_gamma = np.sin(np.radians(gamma))
    if abs(sin_gamma) < 1e-12:
        raise ValueError("gamma is degenerate.")

    c_x = cos_beta
    c_y = (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    c_z_squared = 1.0 - c_x * c_x - c_y * c_y
    if c_z_squared <= 1e-12:
        raise ValueError("those cell angles do not describe a real cell.")

    return np.array(
        [
            [a, 0.0, 0.0],
            [b * cos_gamma, b * sin_gamma, 0.0],
            [c * c_x, c * c_y, c * np.sqrt(c_z_squared)],
        ],
        dtype=np.float64,
    )


def orientation_of(lattice: np.ndarray) -> np.ndarray:
    """Orthogonal transform carrying the canonical form of ``lattice`` onto it.

    Two lattices with the same six parameters differ only by an orthogonal
    transform, so ``canonical_lattice(*cell_parameters(L)) @ R == L`` defines
    ``R`` uniquely. Recovering it is what lets an angle edit rebuild the cell
    without moving the structure in space.

    A rotation for the right-handed cells that CIFs and POSCARs normally carry,
    and a reflection (``det == -1``) for a left-handed one, which some tools do
    emit. That is the *wanted* behaviour rather than a case to reject: the
    reflection is exactly what carries the handedness of the file through an
    edit, and ``canonical_lattice`` is always right-handed, so refusing it would
    silently mirror the structure.
    """
    rows = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    canonical = canonical_lattice(*cell_parameters(rows))
    return np.linalg.solve(canonical, rows)


def lattice_from_parameters(
    reference: np.ndarray,
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> np.ndarray:
    """Lattice for six parameters, oriented like ``reference``.

    Passing ``reference``'s own parameters returns ``reference`` back to
    floating-point precision, which is what makes it safe to call every frame.
    """
    return canonical_lattice(a, b, c, alpha, beta, gamma) @ orientation_of(reference)


def strained_coords(
    lattice: np.ndarray,
    cartesian_coords: np.ndarray,
    new_lattice: np.ndarray,
) -> np.ndarray:
    """``cartesian_coords`` carried onto ``new_lattice`` at fixed fractional position."""
    old = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    coords = np.asarray(cartesian_coords, dtype=np.float64).reshape(-1, 3)
    fractional = np.linalg.solve(old.T, coords.T).T
    return fractional @ np.asarray(new_lattice, dtype=np.float64).reshape(3, 3)


def strain_structure(structure, new_lattice: np.ndarray) -> None:
    """Re-cell ``structure`` in place, holding every atom's fractional position.

    Atom count, ordering, labels and moments are all untouched, so anything
    keyed by site index -- saved spin configurations above all -- stays valid.
    Only quantities that depend on distance (exchange couplings, and therefore
    energies) go stale, which is the caller's business.
    """
    target = np.asarray(new_lattice, dtype=np.float64).reshape(3, 3)
    if abs(float(np.linalg.det(target))) < 1e-9:
        raise ValueError("the requested cell is degenerate.")
    structure.cartesian_coords = strained_coords(
        structure.lattice, structure.cartesian_coords, target
    )
    structure.lattice = target


def tiled_cell(
    lattice: np.ndarray,
    cartesian_coords: np.ndarray,
    counts: Sequence[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate a cell ``counts`` times along each axis.

    Returns ``(new_lattice, coordinates, source_indices)``. ``source_indices``
    says which original atom each copy came from, so the caller can carry labels
    and moments across without knowing the tiling order.
    """
    rows = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    coords = np.asarray(cartesian_coords, dtype=np.float64).reshape(-1, 3)
    n_x, n_y, n_z = (max(1, int(value)) for value in counts)
    shifts = np.array(
        [
            i * rows[0] + j * rows[1] + k * rows[2]
            for i in range(n_x)
            for j in range(n_y)
            for k in range(n_z)
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    # Image-major, so every copy of the original cell stays a contiguous block
    # and source_indices is just the atom range repeated.
    tiled = (coords[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    source = np.tile(np.arange(len(coords), dtype=np.int64), len(shifts))
    new_lattice = rows * np.array([[n_x], [n_y], [n_z]], dtype=np.float64)
    return new_lattice, tiled, source


def tile_structure(structure, counts: Sequence[int]) -> None:
    """Replicate ``structure`` in place into a supercell.

    Unlike :func:`strain_structure` this changes the atom count, so anything
    indexed by site -- saved spin configurations, solver results -- is invalid
    afterwards and the caller must drop it.
    """
    new_lattice, coords, source = tiled_cell(
        structure.lattice, structure.cartesian_coords, counts
    )
    structure.lattice = new_lattice
    structure.cartesian_coords = coords
    structure.atomic_labels = [structure.atomic_labels[index] for index in source]
    structure.magnetic_moments = np.asarray(
        structure.magnetic_moments, dtype=np.float64
    ).reshape(-1, 3)[source]
