"""Canonical B-site magnetic orderings (F/G/A/C/E) scored independently of the solver.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from quick_mag.structure import ChemicalStructure
from quick_mag.spin_solver_np import SpinConfig, compute_config_energy


# Patterns the B-site pattern assigner understands.
SUPPORTED_B_SITE_SPIN_PATTERNS = ("F", "G", "A", "C", "E")
# Canonical reference orderings reported alongside every solve.
CANONICAL_SOLVER_SPIN_PATTERNS = ("G", "C", "F", "A")


def builder_spin_sign(pattern: str, i: int, j: int, k: int) -> float:
    if pattern == "F":
        return 1.0
    if pattern == "G":
        return 1.0 if (i + j + k) % 2 == 0 else -1.0
    if pattern == "A":
        return 1.0 if k % 2 == 0 else -1.0
    if pattern == "C":
        return 1.0 if (j + k) % 2 == 0 else -1.0
    if pattern == "E":
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
    if pattern not in SUPPORTED_B_SITE_SPIN_PATTERNS:
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


def reference_spin_configs(
    structure: ChemicalStructure,
    assignment,
    j_matrix: np.ndarray,
    magnetic_site_indices: Sequence[int],
    site_indexing,
    *,
    patterns: Sequence[str] = CANONICAL_SOLVER_SPIN_PATTERNS,
) -> list[SpinConfig]:
    """Score the canonical reference orderings for one oxidation-state assignment.

    Returns one :class:`SpinConfig` per entry in ``patterns`` (same order), each
    holding the compact per-magnetic-site moments and its Heisenberg energy under
    ``j_matrix``. Returns ``[]`` when the reference orderings are not meaningful:
    no couplings, no valid B-site indexing, or magnetic sites that do not sit on
    the perovskite B grid (e.g. a non-perovskite lattice).
    """
    if j_matrix is None or np.asarray(j_matrix).size == 0:
        return []
    if site_indexing is None or site_indexing.grid_to_site is None or site_indexing.b_grid_shape is None:
        return []

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

    configs: list[SpinConfig] = []
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
            SpinConfig(
                energy=float(energy),
                all_moments=compact_moments,
                magnetization=float(np.sum(compact_moments)),
                n_unpaired=float(np.sum(np.abs(compact_moments))),
            )
        )
    return configs
