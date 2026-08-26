"""Canonical magnetic orderings (F/G/A/C/E) scored independently of the solver.

The orderings themselves live in :mod:`quick_mag.spin_planes` as plane patterns --
a Miller plane family plus a sign string across successive planes. This module is
the perovskite-facing layer over them: it keeps the ``"A(c)"`` spelling the builder,
the CLI and the UI use, and scores each ordering against an exchange matrix.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from quick_mag.spin_planes import (
    CANONICAL_PLANE_PATTERNS,
    PATTERNS_BY_NAME,
    MagneticSublattice,
    PlanePattern,
    parse_plane_label,
    pattern_signs,
    patterns_for_sites,
    signs_from_ordinals,
)
from quick_mag.structure import ChemicalStructure
from quick_mag.spin_solver import SpinConfig, compute_config_energy


# Patterns the B-site pattern assigner understands. ``A`` and ``C`` also accept an
# explicit orientation, written ``A(a)`` / ``C(b)`` etc. (see ``parse_pattern``).
SUPPORTED_B_SITE_SPIN_PATTERNS = ("F", "G", "A", "C", "E")
# Orderings that single out one axis of the B grid, and so have three distinct
# orientations whenever the cell is anisotropic. G (every bond antiferromagnetic)
# and F (every bond ferromagnetic) treat all axes alike.
ORIENTED_SPIN_PATTERNS = ("A", "C", "E")
AXIS_NAMES = ("a", "b", "c")
# The axis each bare (unqualified) oriented pattern refers to: A stacks its
# ferromagnetic planes along c, C runs its ferromagnetic chains along a, and E
# runs its up-up-down-down modulation along a.
_DEFAULT_PATTERN_AXIS = {"A": 2, "C": 0, "E": 0}


def parse_pattern(pattern: str) -> tuple[str, int]:
    """Split ``"A(b)"`` into ``("A", 1)``; bare names take their default axis.

    Returns the axis index as ``-1`` for orderings that have no orientation.
    """
    name, _, axis_text = str(pattern).partition("(")
    name = name.strip()
    if not axis_text:
        return name, _DEFAULT_PATTERN_AXIS.get(name, -1)
    axis = axis_text.rstrip(")").strip()
    if name not in ORIENTED_SPIN_PATTERNS or axis not in AXIS_NAMES:
        raise ValueError(f"Unsupported reference ordering '{pattern}'.")
    return name, AXIS_NAMES.index(axis)


def plane_pattern_for(pattern: str) -> PlanePattern | None:
    """The :class:`PlanePattern` behind a name like ``"A(c)"`` or bare ``"A"``.

    One source of truth: the builder, the reference scoring and the classifier all
    resolve a name through here, so what "G" means is defined in exactly one place.
    """
    text = str(pattern).strip()
    if text.startswith("("):
        # A pattern with no classical name, spelled in plane notation.
        return parse_plane_label(text)
    try:
        name, axis = parse_pattern(text)
    except ValueError:
        return None
    if name in ORIENTED_SPIN_PATTERNS:
        name = f"{name}({AXIS_NAMES[axis]})" if axis >= 0 else name
    return PATTERNS_BY_NAME.get(name)


def grid_sublattice(b_grid_shape: Sequence[int]) -> MagneticSublattice:
    """The magnetic sublattice of a full ``b_grid_shape`` B-site grid.

    Lets shape-only callers reach the plane machinery without building a structure.
    """
    return MagneticSublattice.from_grid(b_grid_shape)


def canonical_reference_patterns(
    b_grid_shape: Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Every distinct canonical ordering to score against a solve.

    A, C and E each single out one axis of the B grid, so on a distorted cell their
    three orientations are genuinely different states with different energies —
    all of them are returned (``A(a)``, ``A(b)``, ``A(c)``, ...) rather than one
    arbitrary choice. G and F are orientation-independent and appear once.

    Patterns the grid cannot distinguish are dropped: alternating across a single
    plane is only F, and an up-up-down-down modulation needs four planes to be
    anything other than A or F. That is the general form of the old "axes with
    fewer than two B sites" rule.
    """
    if b_grid_shape is None:
        patterns = CANONICAL_PLANE_PATTERNS
    else:
        patterns = patterns_for_sites(grid_sublattice(b_grid_shape).lattice_coords)
    return tuple(pattern.label for pattern in patterns)


# Backwards-compatible name for the un-oriented set.
CANONICAL_SOLVER_SPIN_PATTERNS = ("G", "C", "F", "A")


def builder_spin_sign(pattern: str, i: int, j: int, k: int) -> float:
    """Sign of the spin at B-grid cell ``(i, j, k)`` under ``pattern``.

    ``A(x)`` puts ferromagnetic planes perpendicular to axis ``x`` and alternates
    between them (antiferromagnetic along ``x``, ferromagnetic across it);
    ``C(x)`` inverts every bond — ferromagnetic chains *along* ``x``,
    antiferromagnetic in the two perpendicular directions; ``E(x)`` runs an
    up-up-down-down modulation along ``x``.

    On a full grid the cell's plane ordinal is just the Miller dot product, so this
    is the plane machinery evaluated without the binning step. Returns 0.0 for a
    pattern this module does not know.
    """
    plane_pattern = plane_pattern_for(pattern)
    if plane_pattern is None:
        return 0.0
    ordinal = int(np.dot(plane_pattern.miller, (int(i), int(j), int(k))))
    return float(signs_from_ordinals(np.array([ordinal]), plane_pattern.signs)[0])


def magnetic_sublattice_for(site_indexing) -> MagneticSublattice | None:
    """The magnetic sublattice behind a ``PerovskiteSiteIndexing``, if it has one.

    The B-site grid *is* the magnetic sublattice for a perovskite, including the
    grid recovered from a loaded structure by
    ``classify_spin_structure.site_indexing_from_magnetic_sublattice``. Returns
    ``None`` when no grid could be established, which is the signal to skip the
    reference orderings rather than guess at them.
    """
    if site_indexing is None:
        return None
    grid_shape = getattr(site_indexing, "b_grid_shape", None)
    grid_to_site = getattr(site_indexing, "grid_to_site", None)
    if grid_shape is None or grid_to_site is None:
        return None
    try:
        return MagneticSublattice.from_grid(grid_shape, grid_to_site)
    except ValueError:
        return None


def assign_b_site_spin_pattern(
    structure: ChemicalStructure,
    build: Any,
    *,
    pattern: str,
    moment_magnitude: float | None = None,
    site_magnitudes: np.ndarray | None = None,
    site_indexing=None,
) -> None:
    """Write a canonical spin pattern onto the B sublattice of ``structure``.

    ``build`` is only consulted for the B-site grid when ``site_indexing`` is not
    provided; callers that already have a ``PerovskiteSiteIndexing`` may pass
    ``build=None``.
    """
    try:
        name, _axis = parse_pattern(pattern)
    except ValueError:
        return
    if name not in SUPPORTED_B_SITE_SPIN_PATTERNS:
        return

    magnitudes = None if site_magnitudes is None else np.asarray(site_magnitudes, dtype=np.float64)
    if moment_magnitude is None and magnitudes is None:
        return
    if moment_magnitude is not None and moment_magnitude <= 0.0 and magnitudes is None:
        return

    if site_indexing is None:
        b_grid = np.asarray(build.b_site_indices, dtype=int).reshape(build.octahedra.shape)
    else:
        if site_indexing.grid_to_site is None or site_indexing.b_grid_shape is None:
            return
        b_grid = np.asarray(site_indexing.grid_to_site, dtype=int).reshape(
            site_indexing.b_grid_shape
        )

    for i, j, k in np.ndindex(b_grid.shape):
        site_index = int(b_grid[i, j, k])
        # -1 marks a B cell removed by a vacancy; writing there would land on the
        # last atom of the structure instead of being a no-op.
        if site_index < 0:
            continue
        magnitude = (
            float(magnitudes[site_index])
            if magnitudes is not None and 0 <= site_index < len(magnitudes)
            else float(moment_magnitude or 0.0)
        )
        if magnitude <= 0.0:
            continue
        structure.magnetic_moments[site_index, 2] = (
            magnitude * builder_spin_sign(pattern, i, j, k)
        )


def named_reference_spin_configs(
    structure: ChemicalStructure,
    assignment,
    j_matrix: np.ndarray,
    magnetic_site_indices: Sequence[int],
    site_indexing,
    *,
    patterns: Sequence[str] | None = None,
) -> list[tuple[str, SpinConfig]]:
    """Score the canonical reference orderings for one oxidation-state assignment.

    Returns ``(pattern name, config)`` pairs — one per entry in ``patterns``, or
    per :func:`canonical_reference_patterns` when ``patterns`` is omitted, which
    covers **every** orientation of A and C rather than one arbitrary choice.
    Each config holds the compact per-magnetic-site moments and its Heisenberg
    energy under ``j_matrix``.

    Returns ``[]`` when the reference orderings are not meaningful: no couplings,
    no magnetic sites, or no way to place the magnetic sites on a lattice (see
    :func:`magnetic_sublattice_for`).
    """
    if j_matrix is None or np.asarray(j_matrix).size == 0:
        return []

    site_list = [int(s) for s in magnetic_site_indices]
    if not site_list:
        return []

    assigned_magnitudes = np.asarray(assignment.magnetic_moments, dtype=np.float64)
    if assigned_magnitudes.shape != (structure.atom_count,):
        return []

    sublattice = magnetic_sublattice_for(site_indexing)
    if sublattice is None or sublattice.size == 0:
        return []

    # Only sites the sublattice actually places can carry a patterned spin; anything
    # else in site_list keeps a zero moment, exactly as before.
    position_of = {int(site): row for row, site in enumerate(sublattice.site_indices)}
    active = [
        site
        for site in site_list
        if site in position_of and abs(float(assigned_magnitudes[site])) > 1e-8
    ]
    if not active:
        return []

    if patterns is None:
        patterns = tuple(
            pattern.label for pattern in patterns_for_sites(sublattice.lattice_coords)
        )

    configs: list[tuple[str, SpinConfig]] = []
    for pattern in patterns:
        plane_pattern = plane_pattern_for(pattern)
        if plane_pattern is None:
            continue
        signs = pattern_signs(sublattice.lattice_coords, plane_pattern)
        compact_moments = np.array(
            [
                (
                    float(assigned_magnitudes[site]) * float(signs[position_of[site]])
                    if site in position_of and 0 <= site < structure.atom_count
                    else 0.0
                )
                for site in site_list
            ],
            dtype=np.float64,
        )
        energy = compute_config_energy(j_matrix, compact_moments)
        configs.append(
            (
                str(pattern),
                SpinConfig(
                    energy=float(energy),
                    all_moments=compact_moments,
                    magnetization=float(np.sum(compact_moments)),
                    n_unpaired=float(np.sum(np.abs(compact_moments))),
                ),
            )
        )
    return configs


def reference_spin_configs(
    structure: ChemicalStructure,
    assignment,
    j_matrix: np.ndarray,
    magnetic_site_indices: Sequence[int],
    site_indexing,
    *,
    patterns: Sequence[str] | None = None,
) -> list[SpinConfig]:
    """:func:`named_reference_spin_configs` without the pattern names."""
    return [
        config
        for _name, config in named_reference_spin_configs(
            structure,
            assignment,
            j_matrix,
            magnetic_site_indices,
            site_indexing,
            patterns=patterns,
        )
    ]
