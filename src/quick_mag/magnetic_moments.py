"""Per-site oxidation-state expansion using electron-configuration-based moments."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import product as cartesian_product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from quick_mag.electron_configurations import count_unpaired_electrons_for_ion
from quick_mag.oxidation_state_energy import GroupAssignment, SiteGroup
from quick_mag.oxidation_state_enumeration import (
    OxidationStateDistribution,
    enumerate_possible_oxidation_state_assignments,
    score_assignment,
)
from quick_mag.structure import ChemicalStructure


logger = logging.getLogger(__name__)


def format_oxidation_distribution(distributions: Dict[str, Dict[int, int]]) -> str:
    """Render ``{element: {ox: count}}`` as e.g. ``1xFe+2 + 2xFe+3 | 4xO-2``.

    Shared by the CLI, the spin solver, and the UI so every surface reports an
    assignment the same way.
    """
    parts: List[str] = []
    for element, state_counts in sorted(distributions.items()):
        tokens = [
            f"{site_count}x{element}{oxidation_state:+d}"
            for oxidation_state, site_count in sorted(state_counts.items())
            if site_count > 0
        ]
        if tokens:
            parts.append(" + ".join(tokens))
    return " | ".join(parts) if parts else "(no oxidation-state distribution)"


@dataclass
class OxidationStateAssignment:
    site_oxidation_states: np.ndarray    # shape (n_atoms,), dtype int
    magnetic_moments: np.ndarray         # shape (n_atoms,), unsigned μ_B
    total_energy: float                  # model energy (lower ↔ more favorable)
    distributions: Dict[str, Dict[int, int]]  # element → {ox_state: count}
    label: str = ""

    def __repr__(self) -> str:
        parts = []
        for el, dist in sorted(self.distributions.items()):
            tokens = [f"{ct}×{el}{ox:+d}" for ox, ct in sorted(dist.items()) if ct > 0]
            if tokens:
                parts.append(" + ".join(tokens))
        dist_str = " | ".join(parts)
        return f"OxStateAssignment(E={self.total_energy:.3f}, {dist_str})"


def get_magnetic_moment(element: str, oxidation_state: int) -> float:
    """High-spin magnetic moment (in μ_B) for the given element/oxidation state."""
    return float(count_unpaired_electrons_for_ion(element, oxidation_state))


def _get_symmetry_info(structure: ChemicalStructure) -> Optional[Dict[str, object]]:
    """Approximate symmetry orbits (``equivalent_atoms``) for the structure.

    Uses the pymatgen-free coordination fingerprint on ``ChemicalStructure``; on
    any failure we fall back to no symmetry (each site its own orbit).
    """
    try:
        return {"equivalent_atoms": structure.equivalent_atoms()}
    except Exception:
        logger.warning("Symmetry analysis failed; falling back to no symmetry.")
        return None


def _wyckoff_orderings(
    sites: List[int],
    ox_counts: Dict[int, int],
    sym_info: Optional[Dict[str, object]],
) -> List[Dict[int, List[int]]]:
    """Generate symmetry-distinct ways to assign oxidation states to specific sites."""
    ox_states = sorted(ox_counts.keys())

    if sym_info is not None:
        equiv = sym_info["equivalent_atoms"]
        orbit_map: Dict[int, List[int]] = defaultdict(list)
        for site_index in sites:
            orbit_map[equiv[site_index]].append(site_index)
        orbits = list(orbit_map.values())
    else:
        orbits = [[site_index] for site_index in sites]

    valid_orderings: List[Dict[int, List[int]]] = []

    def _backtrack(
        orbit_index: int,
        remaining: Dict[int, int],
        partial: Dict[int, List[int]],
    ) -> None:
        if len(valid_orderings) > 20:
            return
        if orbit_index == len(orbits):
            if all(count == 0 for count in remaining.values()):
                valid_orderings.append(
                    {oxidation_state: list(site_list) for oxidation_state, site_list in partial.items()}
                )
            return

        orbit = orbits[orbit_index]
        orbit_size = len(orbit)
        for oxidation_state in ox_states:
            if remaining[oxidation_state] >= orbit_size:
                remaining[oxidation_state] -= orbit_size
                partial.setdefault(oxidation_state, []).extend(orbit)
                _backtrack(orbit_index + 1, remaining, partial)
                remaining[oxidation_state] += orbit_size
                partial[oxidation_state] = partial[oxidation_state][:-orbit_size]
                if not partial[oxidation_state]:
                    del partial[oxidation_state]

    _backtrack(0, dict(ox_counts), {})

    if valid_orderings:
        return valid_orderings

    logger.debug("Could not find Wyckoff-respecting orderings; using sequential fallback.")
    ordering: Dict[int, List[int]] = {}
    pointer = 0
    for oxidation_state in ox_states:
        site_count = ox_counts[oxidation_state]
        ordering[oxidation_state] = sites[pointer : pointer + site_count]
        pointer += site_count
    return [ordering]


def _distribute_to_sites(
    dist: Dict[str, Dict[int, int]],
    element_sites: Dict[str, List[int]],
    structure: ChemicalStructure,
    sym_info: Optional[Dict],
) -> List[np.ndarray]:
    """Convert an element-level distribution to one or more site-level oxidation-state arrays."""
    n_atoms = structure.atom_count

    pure_elements: Dict[str, int] = {}
    mixed_elements: Dict[str, Dict[int, int]] = {}
    for el, ox_counts in dist.items():
        nonzero = {ox: c for ox, c in ox_counts.items() if c > 0}
        if len(nonzero) == 1:
            pure_elements[el] = next(iter(nonzero.keys()))
        else:
            mixed_elements[el] = nonzero

    base = np.zeros(n_atoms, dtype=int)
    for el, ox in pure_elements.items():
        for i in element_sites[el]:
            base[i] = ox

    if not mixed_elements:
        return [base]

    per_element_options: List[List[Dict[int, List[int]]]] = []
    for el, ox_counts in mixed_elements.items():
        sites = element_sites[el]
        orderings = _wyckoff_orderings(sites, ox_counts, sym_info)
        per_element_options.append(orderings)

    mixed_els = list(mixed_elements.keys())
    results: List[np.ndarray] = []
    for combo in cartesian_product(*per_element_options):
        arr = base.copy()
        for el, ordering in zip(mixed_els, combo):
            for ox, site_list in ordering.items():
                for i in site_list:
                    arr[i] = ox
        results.append(arr)

    return results


def expand_distribution_to_site_assignments(
    distributions: List[OxidationStateDistribution],
    structure: ChemicalStructure,
    *,
    use_symmetry: bool = True,
    max_assignments: Optional[int] = None,
) -> List[OxidationStateAssignment]:
    """Expand element-level distributions into per-site OxidationStateAssignment records.

    ``total_energy`` is ``-score_assignment(...)`` so downstream solvers that sort
    ascending on ``total_energy`` keep their "lower = better" semantics.

    ``max_assignments`` stops the expansion once that many records exist.
    ``distributions`` arrive ranked best-first, but each one multiplies into the
    cartesian product of its per-element Wyckoff orderings — in low-symmetry
    (tilted) cells with several mixed-valent elements that product turns a
    couple hundred distributions into tens of thousands of records.
    """

    def _normalize(distribution: OxidationStateDistribution) -> Dict[str, Dict[int, int]]:
        return {
            element: {oxi: n for oxi, n in pairs if n > 0}
            for element, pairs in distribution.items()
        }

    symbols = structure.element_symbols()
    element_sites: Dict[str, List[int]] = defaultdict(list)
    for i, symbol in enumerate(symbols):
        element_sites[symbol].append(i)
    composition = {el: len(sites) for el, sites in element_sites.items()}
    sym_info = _get_symmetry_info(structure) if use_symmetry else None

    assignments: List[OxidationStateAssignment] = []
    for distribution in distributions:
        if max_assignments is not None and len(assignments) >= max_assignments:
            break
        normalized = _normalize(distribution)
        energy = -score_assignment(distribution, composition)
        site_arrays = _distribute_to_sites(normalized, element_sites, structure, sym_info)
        for site_ox in site_arrays:
            if max_assignments is not None and len(assignments) >= max_assignments:
                break
            moments = np.array([
                get_magnetic_moment(symbols[i], int(site_ox[i]))
                for i in range(structure.atom_count)
            ])
            assignments.append(OxidationStateAssignment(
                site_oxidation_states=site_ox,
                magnetic_moments=moments,
                total_energy=energy,
                distributions=normalized,
            ))
    return assignments


def _group_symmetry_info(
    group: SiteGroup, sym_info: Optional[Dict[str, object]]
) -> Optional[Dict[str, object]]:
    """``sym_info`` as seen from inside one group.

    A constraint that covers part of an orbit has broken that orbit's symmetry:
    the constrained atoms are no longer equivalent to the rest, so the rest are
    no longer obliged to share a state either. Those survivors become singleton
    orbits; orbits the group holds whole keep their equivalence.
    """
    if sym_info is None:
        return None
    equiv = list(sym_info["equivalent_atoms"])
    members: Dict[int, List[int]] = defaultdict(list)
    for atom, orbit in enumerate(equiv):
        members[int(orbit)].append(atom)
    in_group = set(group.sites)
    for atom in group.sites:
        orbit = members[int(equiv[atom])]
        if any(other not in in_group for other in orbit):
            equiv[atom] = atom
    return {**sym_info, "equivalent_atoms": equiv}


def _distributions_of(symbols: Sequence[str], states: np.ndarray) -> Dict[str, Dict[int, int]]:
    counts: Dict[str, Dict[int, int]] = {}
    for symbol, state in zip(symbols, states):
        per_element = counts.setdefault(str(symbol), {})
        per_element[int(state)] = per_element.get(int(state), 0) + 1
    return counts


def expand_group_assignments_to_site_assignments(
    group_assignments: Sequence[Tuple[GroupAssignment, float]],
    structure: ChemicalStructure,
    *,
    use_symmetry: bool = True,
    max_assignments: Optional[int] = None,
    prefer_like: Optional[np.ndarray] = None,
) -> List[OxidationStateAssignment]:
    """Expand group-resolved assignments into per-site records.

    The group form (``enumerate_site_group_assignments``) already says which
    atoms each ``[(oxi, n), ...]`` split belongs to, so a constrained atom can
    only ever receive a state it was allowed. Within a mixed-valent group the
    states are placed by the same Wyckoff-orbit backtracking as the
    element-level expansion, with orbits a constraint cut through treated as
    broken (see ``_group_symmetry_info``).

    ``total_energy`` is the search energy of the assignment -- the physical
    model energy, including the constrained atoms' share -- so a constrained
    result is directly comparable with the free one for the same cell.

    ``prefer_like``, when given, is a previous site array: among the orderings
    of one assignment those that change the fewest sites relative to it come
    first. Re-solving after an edit can otherwise move states between atoms the
    edit never touched, which reads as the tool fighting the user.
    """
    symbols = structure.element_symbols()
    n_atoms = structure.atom_count
    sym_info = _get_symmetry_info(structure) if use_symmetry else None
    previous = (
        np.asarray(prefer_like, dtype=int)
        if prefer_like is not None and len(prefer_like) == n_atoms
        else None
    )

    assignments: List[OxidationStateAssignment] = []
    for assignment, energy in group_assignments:
        if max_assignments is not None and len(assignments) >= max_assignments:
            break
        base = np.zeros(n_atoms, dtype=int)
        per_group_orderings: List[List[Dict[int, List[int]]]] = []
        for group, pairs in assignment.items():
            counts = {int(oxi): int(n) for oxi, n in pairs if n > 0}
            if len(counts) == 1:
                base[list(group.sites)] = next(iter(counts))
                continue
            per_group_orderings.append(
                _wyckoff_orderings(
                    list(group.sites), counts, _group_symmetry_info(group, sym_info)
                )
            )

        site_arrays: List[np.ndarray] = []
        for combo in cartesian_product(*per_group_orderings):
            arr = base.copy()
            for ordering in combo:
                for oxi, site_list in ordering.items():
                    arr[site_list] = oxi
            site_arrays.append(arr)
        if previous is not None and len(site_arrays) > 1:
            site_arrays.sort(key=lambda arr: int(np.count_nonzero(arr != previous)))

        for site_ox in site_arrays:
            if max_assignments is not None and len(assignments) >= max_assignments:
                break
            moments = np.array([
                _moment_or_zero(symbols[i], int(site_ox[i])) for i in range(n_atoms)
            ])
            assignments.append(OxidationStateAssignment(
                site_oxidation_states=site_ox,
                magnetic_moments=moments,
                total_energy=float(energy),
                distributions=_distributions_of(symbols, site_ox),
            ))
    return assignments


def _moment_or_zero(element: str, oxidation_state: int) -> float:
    """``get_magnetic_moment`` for ions the tables can build; 0 for the rest.

    A constrained state is the user's to choose, and one the
    electron-configuration tables cannot build simply carries no formal moment.
    """
    try:
        return abs(get_magnetic_moment(element, oxidation_state))
    except Exception:
        return 0.0
