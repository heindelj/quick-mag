"""Canonical B-site magnetic orderings (F/G/A/C/E) scored independently of the solver.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from quick_mag.structure import ChemicalStructure
from quick_mag.spin_solver import SpinConfig, compute_config_energy


# Patterns the B-site pattern assigner understands. ``A`` and ``C`` also accept an
# explicit orientation, written ``A(a)`` / ``C(b)`` etc. (see ``parse_pattern``).
SUPPORTED_B_SITE_SPIN_PATTERNS = ("F", "G", "A", "C", "E")
# Orderings that single out one axis of the B grid, and so have three distinct
# orientations whenever the cell is anisotropic. G (every bond antiferromagnetic)
# and F (every bond ferromagnetic) treat all axes alike.
ORIENTED_SPIN_PATTERNS = ("A", "C")
AXIS_NAMES = ("a", "b", "c")
# The axis each bare (unqualified) oriented pattern refers to: A stacks its
# ferromagnetic planes along c, C runs its ferromagnetic chains along a.
_DEFAULT_PATTERN_AXIS = {"A": 2, "C": 0}


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


def canonical_reference_patterns(
    b_grid_shape: Sequence[int] | None = None,
) -> tuple[str, ...]:
    """Every distinct canonical ordering to score against a solve.

    A and C each single out one axis of the B grid, so on a distorted cell their
    three orientations are genuinely different states with different energies —
    all of them are returned (``A(a)``, ``A(b)``, ``A(c)``, ...) rather than one
    arbitrary choice. G and F are orientation-independent and appear once.

    Axes with fewer than two B sites are dropped: alternating along a length-1
    axis collapses A onto F and C onto G, which would only duplicate rows.
    """
    axes = range(3)
    if b_grid_shape is not None:
        shape = [int(size) for size in b_grid_shape]
        axes = [axis for axis in range(3) if axis < len(shape) and shape[axis] >= 2]

    patterns = ["G"]
    patterns += [f"C({AXIS_NAMES[axis]})" for axis in axes]
    patterns.append("F")
    patterns += [f"A({AXIS_NAMES[axis]})" for axis in axes]
    return tuple(patterns)


# Backwards-compatible name for the un-oriented set.
CANONICAL_SOLVER_SPIN_PATTERNS = ("G", "C", "F", "A")


def builder_spin_sign(pattern: str, i: int, j: int, k: int) -> float:
    """Sign of the spin at B-grid cell ``(i, j, k)`` under ``pattern``.

    ``A(x)`` puts ferromagnetic planes perpendicular to axis ``x`` and alternates
    between them (antiferromagnetic along ``x``, ferromagnetic across it);
    ``C(x)`` inverts every bond — ferromagnetic chains *along* ``x``,
    antiferromagnetic in the two perpendicular directions.
    """
    cell = (i, j, k)
    name, axis = parse_pattern(pattern)
    if name == "F":
        return 1.0
    if name == "G":
        return 1.0 if (i + j + k) % 2 == 0 else -1.0
    if name == "A":
        return 1.0 if cell[axis] % 2 == 0 else -1.0
    if name == "C":
        perpendicular = sum(value for index, value in enumerate(cell) if index != axis)
        return 1.0 if perpendicular % 2 == 0 else -1.0
    if name == "E":
        return 1.0 if j % 2 == 0 else -1.0
    return 0.0


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
    no valid B-site indexing, or magnetic sites that do not sit on the perovskite
    B grid (e.g. a non-perovskite lattice).
    """
    if j_matrix is None or np.asarray(j_matrix).size == 0:
        return []
    if site_indexing is None or site_indexing.grid_to_site is None or site_indexing.b_grid_shape is None:
        return []
    if patterns is None:
        patterns = canonical_reference_patterns(site_indexing.b_grid_shape)

    site_list = [int(s) for s in magnetic_site_indices]
    if not site_list:
        return []

    assigned_magnitudes = np.asarray(assignment.magnetic_moments, dtype=np.float64)
    if assigned_magnitudes.shape != (structure.atom_count,):
        return []

    b_site_set = {int(s) for s in np.asarray(site_indexing.b_site_indices, dtype=int)}
    active = {
        s
        for s in site_list
        if 0 <= s < structure.atom_count and abs(float(assigned_magnitudes[s])) > 1e-8
    }
    if not active or not active.issubset(b_site_set):
        return []

    configs: list[tuple[str, SpinConfig]] = []
    for pattern in patterns:
        patterned = ChemicalStructure.with_zero_magnetic_moments(
            name=structure.name,
            lattice=np.array(structure.lattice, dtype=np.float64, copy=True),
            cartesian_coords=np.array(structure.cartesian_coords, dtype=np.float64, copy=True),
            atomic_labels=list(structure.atomic_labels),
            is_periodic=structure.is_periodic,
        )
        patterned.generation_parameters = structure.generation_parameters
        assign_b_site_spin_pattern(
            patterned,
            None,
            pattern=pattern,
            site_magnitudes=assigned_magnitudes,
            site_indexing=site_indexing,
        )
        compact_moments = np.array(
            [
                patterned.magnetic_moments[s, 2] if 0 <= s < structure.atom_count else 0.0
                for s in site_list
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
