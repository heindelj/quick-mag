"""Energy-based oxidation-state prediction.

Instead of an ad-hoc weighted score, rank charge-balanced oxidation-state
assignments by a physical, geometry-free energy that mimics "filling orbitals by
their energies":

    E(assignment) =  sum over cation atoms of the ionization-energy ladder
                  -  sum over anion atoms of the electron-attachment gain
                  -  gamma * charge-transfer stabilization from the electronegativity gap

Lower energy = more stable. The goal is reasonable options quickly, not exact results.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from functools import lru_cache
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Tuple

from quick_mag.element_data import electron_affinity as _electron_affinity
from quick_mag.element_data import ionization_energies as _ionization_energies

from quick_mag.oxidation_state_enumeration import (
    ELECTRONEG,
    OxidationStateDistribution,
    _default_possible_oxidation_states,
    _enumerate_element_options,
    _expand_possible_oxidation_states_with_shannon,
)

# Default electronegativity-gap weight (eV per Pauling unit per transferred
# electron). Calibrated so the test perovskites land on their textbook charges.
DEFAULT_GAMMA: float = 3.0

# Standard closed-shell oxidation state for the common anions. A geometry-free
# energy cannot derive O^2- (gas-phase O^2- is unbound; only the Madelung field
# stabilizes it), so anions are anchored to their closed-shell charge and the
# IE + electronegativity energy ranks the cation charges. Can be disabled.
STANDARD_ANION_STATES: dict[str, int] = {
    "O": -2, "S": -2, "Se": -2, "Te": -2,
    "F": -1, "Cl": -1, "Br": -1, "I": -1,
}


@lru_cache(maxsize=None)
def ionization_cost(element: str, charge: int) -> float:
    """Energy (eV) to remove ``charge`` electrons: sum of the first IEs."""
    if charge <= 0:
        return 0.0
    energies = _ionization_energies(element)
    available = [e for e in energies[:charge] if e is not None]
    if len(available) < charge:
        # Not enough tabulated IEs: extrapolate the last known step (rare, only
        # for implausibly high charges that the candidate lists won't propose).
        if not available:
            return float("inf")
        step = available[-1] - available[-2] if len(available) > 1 else available[-1]
        missing = charge - len(available)
        return float(sum(available) + missing * available[-1] + step * missing)
    return float(sum(available))


@lru_cache(maxsize=None)
def attach_gain(element: str, n_electrons: int, deep_anion_ea: float = 0.0) -> float:
    """Energy (eV) released by adding ``n_electrons`` to form an anion.

    The first electron releases the electron affinity; deeper electrons are not
    bound in the gas phase, so they contribute ``deep_anion_ea`` (the lattice /
    electronegativity field supplies their binding elsewhere in the model).
    """
    if n_electrons <= 0:
        return 0.0
    ea = _electron_affinity(element)
    first = float(ea) if ea is not None else 0.0
    return first + deep_anion_ea * (n_electrons - 1)


def _element_option_energy(
    element: str,
    pairs: Sequence[Tuple[int, int]],
    *,
    gamma: float = DEFAULT_GAMMA,
    deep_anion_ea: float = 0.0,
) -> float:
    """Energy contribution (eV) of one element's ``[(oxi, n), ...]`` option.

    The total assignment energy is the sum of these per-element terms — that
    separability is what lets the top-k search rank assignments without
    materializing them all.
    """
    chi = ELECTRONEG.get(element, 2.0)
    energy = 0.0
    for oxi, n in pairs:
        if oxi > 0:
            energy += n * ionization_cost(element, oxi)
        elif oxi < 0:
            energy -= n * attach_gain(element, -oxi, deep_anion_ea)
        # Electronegativity charge-transfer stabilization.
        energy += gamma * n * oxi * chi
    return energy


def assignment_energy(
    distribution: OxidationStateDistribution,
    *,
    gamma: float = DEFAULT_GAMMA,
    deep_anion_ea: float = 0.0,
) -> float:
    """Physical energy (eV) of an oxidation-state assignment; lower = more stable.

    Charge transfer is stabilized symmetrically by an electronegativity potential:
    ``+gamma * sum_atoms n * oxi * chi``. Because the assignment is charge-balanced
    (``sum n*oxi = 0``) this equals ``gamma * sum n*oxi*(chi - chi_ref)``, i.e. it
    rewards anti-correlation between charge and electronegativity — electronegative
    elements lower the energy when negative (so O prefers -2 over -1) and high-chi
    elements are penalized for being cations.
    """
    return sum(
        _element_option_energy(element, pairs, gamma=gamma, deep_anion_ea=deep_anion_ea)
        for element, pairs in distribution.items()
    )


def _initial_possible_oxi_states(
    composition: Dict[str, int],
    anchor_anions: bool,
) -> Dict[str, List[int]]:
    return {
        element: (
            [STANDARD_ANION_STATES[element]]
            if anchor_anions and element in STANDARD_ANION_STATES
            else _default_possible_oxidation_states(element)
        )
        for element in composition
    }


def _search_assignments_by_energy(
    composition: Dict[str, int],
    possible_oxi_states: Dict[str, List[int]],
    max_mixing: int,
    target_charge: int,
    *,
    gamma: float,
    deep_anion_ea: float,
    single_valent: FrozenSet[str] = frozenset(),
) -> Iterator[Tuple[OxidationStateDistribution, float]]:
    """Yield charge-balanced assignments in exact nondecreasing energy order.

    The assignment energy is a sum of independent per-element option energies,
    so a backward DP over (element, remaining charge) gives the exact minimum
    completion energy of any partial assignment. A best-first search over that
    bound then pops full assignments in true energy order — the top-k cost is
    proportional to what is emitted, not to the (combinatorially huge) space.

    Elements in ``single_valent`` are restricted to one oxidation state
    (``max_mixing = 1``), which is how the double-exchange gate asks "can this
    element avoid mixed valence?" without enumerating anything.
    """
    per_element_options: Dict[str, List[Tuple[List[Tuple[int, int]], int, float]]] = {}
    for element, count in composition.items():
        states = sorted(set(possible_oxi_states.get(element, [])))
        if not states:
            return
        options = _enumerate_element_options(
            states=states,
            count=count,
            max_mixing=1 if element in single_valent else max_mixing,
        )
        if not options:
            return
        per_element_options[element] = [
            (
                pairs,
                option_charge,
                _element_option_energy(element, pairs, gamma=gamma, deep_anion_ea=deep_anion_ea),
            )
            for pairs, option_charge in options
        ]

    elements = sorted(
        composition,
        key=lambda element: (
            len(per_element_options[element]),
            composition[element],
            element,
        ),
    )
    n_elements = len(elements)

    # suffix_best[i][q] = minimum energy of elements i.. summing to charge q.
    suffix_best: List[Dict[int, float]] = [{} for _ in range(n_elements + 1)]
    suffix_best[n_elements][0] = 0.0
    for index in range(n_elements - 1, -1, -1):
        current = suffix_best[index]
        following = suffix_best[index + 1]
        for _pairs, option_charge, option_energy in per_element_options[elements[index]]:
            for suffix_charge, suffix_energy in following.items():
                total_charge = option_charge + suffix_charge
                total_energy = option_energy + suffix_energy
                if total_energy < current.get(total_charge, math.inf):
                    current[total_charge] = total_energy

    if target_charge not in suffix_best[0]:
        return

    # Node: (exact lower bound, element index, remaining charge, energy so far,
    # chosen option index per element). The bound is exact, so the first time a
    # complete assignment is popped it is the true next-best one.
    heap: List[Tuple[float, int, int, float, Tuple[int, ...]]] = [
        (suffix_best[0][target_charge], 0, target_charge, 0.0, ())
    ]
    while heap:
        _bound, index, remaining, energy, chosen = heapq.heappop(heap)
        if index == n_elements:
            yield (
                {
                    elements[position]: list(per_element_options[elements[position]][option][0])
                    for position, option in enumerate(chosen)
                },
                energy,
            )
            continue
        following = suffix_best[index + 1]
        for option, (_pairs, option_charge, option_energy) in enumerate(
            per_element_options[elements[index]]
        ):
            rest = remaining - option_charge
            suffix_energy = following.get(rest)
            if suffix_energy is None:
                continue
            next_energy = energy + option_energy
            heapq.heappush(
                heap,
                (next_energy + suffix_energy, index + 1, rest, next_energy, chosen + (option,)),
            )


def enumerate_oxidation_states_by_energy(
    labels: Sequence[str],
    charge: int = 0,
    max_mixing: int = 2,
    *,
    gamma: float = DEFAULT_GAMMA,
    deep_anion_ea: float = 0.0,
    anchor_anions: bool = True,
    top_k: Optional[int] = None,
) -> List[Tuple[OxidationStateDistribution, float]]:
    """Energy-ranked charge-balanced oxidation-state assignments (best first).

    Drop-in alternative to ``enumerate_possible_oxidation_state_assignments`` that
    ranks by the physical energy above instead of the heuristic score. With
    ``anchor_anions`` (default), common anions are pinned to their closed-shell
    charge (``STANDARD_ANION_STATES``) and the energy ranks the cation charges.

    With ``top_k`` the search stops after that many assignments, and its cost is
    proportional to ``top_k`` rather than to the full combinatorial space — pass
    it whenever a bounded list is enough (large supercells and high-entropy
    compositions have astronomically many charge-balanced assignments).
    """
    composition = dict(Counter(labels))
    possible_oxi_states = _initial_possible_oxi_states(composition, anchor_anions)
    if any(not states for states in possible_oxi_states.values()):
        return []

    while True:
        results: List[Tuple[OxidationStateDistribution, float]] = []
        for item in _search_assignments_by_energy(
            composition,
            possible_oxi_states,
            max_mixing,
            charge,
            gamma=gamma,
            deep_anion_ea=deep_anion_ea,
        ):
            results.append(item)
            if top_k is not None and len(results) >= top_k:
                break
        if results:
            return results
        expanded = _expand_possible_oxidation_states_with_shannon(possible_oxi_states)
        if expanded == possible_oxi_states:
            return []
        possible_oxi_states = expanded


def min_energies_with_single_valence(
    labels: Sequence[str],
    element_groups: Iterable[FrozenSet[str]],
    charge: int = 0,
    max_mixing: int = 2,
    *,
    gamma: float = DEFAULT_GAMMA,
    deep_anion_ea: float = 0.0,
    anchor_anions: bool = True,
) -> Optional[Tuple[float, Dict[FrozenSet[str], float]]]:
    """Global minimum energy plus, per group, the minimum with those elements single-valent.

    Returns ``(global_min, {group: constrained_min})`` where ``constrained_min``
    is ``math.inf`` when no charge-balanced assignment keeps every element of the
    group in a single oxidation state, or ``None`` when no charge-balanced
    assignment exists at all. All minima are evaluated on the same
    (Shannon-expanded, if needed) candidate state sets as the unconstrained
    search, mirroring ``enumerate_oxidation_states_by_energy``. Each query is one
    cheap DP — no assignment enumeration.
    """
    composition = dict(Counter(labels))
    possible_oxi_states = _initial_possible_oxi_states(composition, anchor_anions)
    if any(not states for states in possible_oxi_states.values()):
        return None

    while True:
        best = next(
            _search_assignments_by_energy(
                composition,
                possible_oxi_states,
                max_mixing,
                charge,
                gamma=gamma,
                deep_anion_ea=deep_anion_ea,
            ),
            None,
        )
        if best is not None:
            break
        expanded = _expand_possible_oxidation_states_with_shannon(possible_oxi_states)
        if expanded == possible_oxi_states:
            return None
        possible_oxi_states = expanded

    constrained: Dict[FrozenSet[str], float] = {}
    for group in element_groups:
        group = frozenset(group)
        result = next(
            _search_assignments_by_energy(
                composition,
                possible_oxi_states,
                max_mixing,
                charge,
                gamma=gamma,
                deep_anion_ea=deep_anion_ea,
                single_valent=group,
            ),
            None,
        )
        constrained[group] = math.inf if result is None else result[1]
    return best[1], constrained
