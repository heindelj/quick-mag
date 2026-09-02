"""Energy-based oxidation-state prediction.

Instead of an ad-hoc weighted score, rank charge-balanced oxidation-state
assignments by a physical, geometry-free energy that mimics "filling orbitals by
their energies":

    E(assignment) =  sum over cation atoms of the ionization-energy ladder
                  -  sum over anion atoms of the electron-attachment gain
                  -  gamma * charge-transfer stabilization from the electronegativity gap

Lower energy = more stable. The goal is reasonable options quickly, not exact results.

Constraints
-----------
The search runs over *site groups* rather than elements: a group is a set of
atoms of one element that share the same candidate states. Without constraints
every element is one group and the search is exactly the per-element one. A
constraint -- "atom 7 is +2", "atoms 3 and 4 are one of {+3, +4}", "every Mn is
+3 or +4" -- splits an element into groups with their own candidate sets, and
the same dynamic program solves the remainder against the same total charge.
Constrained groups keep the states they were given: no anion anchoring, no
Shannon expansion, no second-guessing.

A group pinned to one state is not a special case; it is a group with exactly
one option, so its charge and energy simply shift what the other groups have to
make up. That is what keeps an edited cell charge-balanced by construction.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import (
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

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

# A constraint value: one state, or the states an atom (or element) may take.
StateSet = Union[int, Iterable[int]]
# Keys are atom indices (per-atom constraints) or element symbols (applying to
# every atom of that element that has no per-atom constraint of its own).
SiteConstraints = Mapping[Union[int, str], StateSet]

__all__ = [
    "DEFAULT_GAMMA",
    "STANDARD_ANION_STATES",
    "GroupAssignment",
    "SiteConstraints",
    "SiteGroup",
    "assignment_energy",
    "attach_gain",
    "constrained_site_states",
    "enumerate_oxidation_states_by_energy",
    "enumerate_site_group_assignments",
    "ionization_cost",
    "merge_group_assignment",
    "min_energies_with_single_valence",
    "normalize_site_constraints",
    "site_groups",
]


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


# ---------------------------------------------------------------------------
# Site groups and constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SiteGroup:
    """Atoms of one element that draw their oxidation state from one candidate set.

    ``states`` is None for a group the model chooses candidates for (anchored
    anions, the scored default list, Shannon expansion when nothing balances) and
    a frozen set for a group the caller constrained.
    """

    element: str
    sites: Tuple[int, ...]
    states: Optional[FrozenSet[int]] = None

    @property
    def count(self) -> int:
        return len(self.sites)

    @property
    def constrained(self) -> bool:
        return self.states is not None

    @property
    def pinned_state(self) -> Optional[int]:
        """The one state this group is held at, or None if it has a choice."""
        if self.states is not None and len(self.states) == 1:
            return next(iter(self.states))
        return None


# One search result: the ``[(oxi, n), ...]`` split of every group.
GroupAssignment = Dict[SiteGroup, List[Tuple[int, int]]]


def _as_state_set(value: StateSet) -> FrozenSet[int]:
    if isinstance(value, int):
        return frozenset([int(value)])
    states = frozenset(int(state) for state in value)
    if not states:
        raise ValueError("a constraint must allow at least one oxidation state")
    return states


def normalize_site_constraints(
    constraints: Optional[SiteConstraints],
) -> Tuple[Dict[int, FrozenSet[int]], Dict[str, FrozenSet[int]]]:
    """Split ``constraints`` into per-atom and per-element sets of allowed states.

    Values may be a single int (pin) or any iterable of ints (allowed subset);
    keys are atom indices or element symbols.
    """
    per_atom: Dict[int, FrozenSet[int]] = {}
    per_element: Dict[str, FrozenSet[int]] = {}
    if not constraints:
        return per_atom, per_element
    for key, value in constraints.items():
        states = _as_state_set(value)
        if isinstance(key, str):
            per_element[key] = states
        else:
            per_atom[int(key)] = states
    return per_atom, per_element


def constrained_site_states(
    labels: Sequence[str],
    constraints: Optional[SiteConstraints],
) -> Dict[int, FrozenSet[int]]:
    """Atom index -> allowed states, with element-level constraints expanded onto atoms.

    A per-atom constraint wins over a per-element one for the same atom. Atoms
    with no constraint are absent from the result. Indices outside ``labels`` are
    ignored rather than raised on: an override recorded against a previous
    structure is a stale note, not an error.
    """
    per_atom, per_element = normalize_site_constraints(constraints)
    resolved: Dict[int, FrozenSet[int]] = {}
    if per_element:
        for atom, label in enumerate(labels):
            states = per_element.get(str(label))
            if states is not None:
                resolved[atom] = states
    for atom, states in per_atom.items():
        if 0 <= atom < len(labels):
            resolved[atom] = states
    return resolved


def site_groups(
    labels: Sequence[str],
    constraints: Optional[SiteConstraints] = None,
) -> List[SiteGroup]:
    """Partition the atoms of ``labels`` into :class:`SiteGroup` records.

    Atoms of one element with the same allowed set share a group; the
    unconstrained atoms of an element form its free group. Order is first
    appearance of each element, free group first, then constrained groups by
    their sorted states, so the partition is deterministic for equal inputs.
    """
    resolved = constrained_site_states(labels, constraints)
    buckets: Dict[Tuple[str, Optional[FrozenSet[int]]], List[int]] = {}
    element_order: List[str] = []
    for atom, label in enumerate(labels):
        element = str(label)
        if element not in element_order:
            element_order.append(element)
        buckets.setdefault((element, resolved.get(atom)), []).append(atom)

    def sort_key(key: Tuple[str, Optional[FrozenSet[int]]]):
        element, states = key
        return (
            element_order.index(element),
            0 if states is None else 1,
            tuple(sorted(states)) if states is not None else (),
        )

    return [
        SiteGroup(element=element, sites=tuple(buckets[(element, states)]), states=states)
        for element, states in sorted(buckets, key=sort_key)
    ]


def merge_group_assignment(assignment: GroupAssignment) -> OxidationStateDistribution:
    """Fold a group assignment back into ``{element: [(oxi, n), ...]}``."""
    merged: Dict[str, Dict[int, int]] = {}
    for group, pairs in assignment.items():
        counts = merged.setdefault(group.element, {})
        for oxi, n in pairs:
            counts[oxi] = counts.get(oxi, 0) + n
    return {
        element: sorted(counts.items())
        for element, counts in merged.items()
    }


def _initial_candidates(
    groups: Sequence[SiteGroup],
    anchor_anions: bool,
) -> Dict[SiteGroup, List[int]]:
    candidates: Dict[SiteGroup, List[int]] = {}
    for group in groups:
        if group.states is not None:
            candidates[group] = sorted(group.states)
        elif anchor_anions and group.element in STANDARD_ANION_STATES:
            candidates[group] = [STANDARD_ANION_STATES[group.element]]
        else:
            candidates[group] = _default_possible_oxidation_states(group.element)
    return candidates


def _expand_candidates(
    candidates: Dict[SiteGroup, List[int]],
) -> Dict[SiteGroup, List[int]]:
    """Shannon-expand the free groups only; constrained groups keep their states."""
    expanded: Dict[SiteGroup, List[int]] = {}
    for group, states in candidates.items():
        if group.constrained:
            expanded[group] = list(states)
        else:
            expanded[group] = _expand_possible_oxidation_states_with_shannon(
                {group.element: states}
            )[group.element]
    return expanded


# (pairs, option charge, option energy)
_Option = Tuple[List[Tuple[int, int]], int, float]
_Key = TypeVar("_Key", bound=Hashable)


def _group_option_table(
    groups: Sequence[SiteGroup],
    candidates: Mapping[SiteGroup, List[int]],
    max_mixing: int,
    *,
    gamma: float,
    deep_anion_ea: float,
    single_valent: FrozenSet[str] = frozenset(),
) -> Optional[List[Tuple[SiteGroup, List[_Option]]]]:
    """Every ``[(oxi, n), ...]`` option of every group, with charge and energy.

    None when some group has no option at all, which is the caller's cue to
    expand the free candidate sets or give up.

    The mixing budget (``max_mixing`` distinct states) is enforced per group,
    not per element. A pinned atom is a statement the user made, and the budget
    is a limit on what the model is allowed to invent: pinning one Fe to +2 in a
    cell of Fe3+ should let the model answer with one Fe4+ (three Fe valences in
    the cell, two of them the model's), not force it to re-valence every Fe.
    """
    table: List[Tuple[SiteGroup, List[_Option]]] = []
    for group in groups:
        states = sorted(set(candidates.get(group, [])))
        if not states:
            return None
        mixing = 1 if group.element in single_valent else max_mixing
        options: List[_Option] = []
        for pairs, option_charge in _enumerate_element_options(
            states=states, count=group.count, max_mixing=mixing
        ):
            options.append(
                (
                    pairs,
                    option_charge,
                    _element_option_energy(
                        group.element, pairs, gamma=gamma, deep_anion_ea=deep_anion_ea
                    ),
                )
            )
        if not options:
            return None
        table.append((group, options))
    return table


def _search_assignments_by_energy(
    table: Sequence[Tuple[_Key, List[_Option]]],
    target_charge: int,
) -> Iterator[Tuple[Dict[_Key, List[Tuple[int, int]]], float]]:
    """Yield charge-balanced assignments in exact nondecreasing energy order.

    The assignment energy is a sum of independent per-group option energies,
    so a backward DP over (group, remaining charge) gives the exact minimum
    completion energy of any partial assignment. A best-first search over that
    bound then pops full assignments in true energy order — the top-k cost is
    proportional to what is emitted, not to the (combinatorially huge) space.

    ``table`` is searched in the order given, after a stable sort that puts the
    groups with the fewest options first (which keeps the heap narrow).
    """
    ordered = sorted(table, key=lambda item: len(item[1]))
    if not ordered:
        if target_charge == 0:
            yield {}, 0.0
        return
    keys = [key for key, _options in ordered]
    per_group_options = [options for _key, options in ordered]
    n_groups = len(ordered)

    # suffix_best[i][q] = minimum energy of groups i.. summing to charge q.
    suffix_best: List[Dict[int, float]] = [{} for _ in range(n_groups + 1)]
    suffix_best[n_groups][0] = 0.0
    for index in range(n_groups - 1, -1, -1):
        current = suffix_best[index]
        following = suffix_best[index + 1]
        for _pairs, option_charge, option_energy in per_group_options[index]:
            for suffix_charge, suffix_energy in following.items():
                total_charge = option_charge + suffix_charge
                total_energy = option_energy + suffix_energy
                if total_energy < current.get(total_charge, math.inf):
                    current[total_charge] = total_energy

    if target_charge not in suffix_best[0]:
        return

    # Node: (exact lower bound, group index, remaining charge, energy so far,
    # chosen option index per group). The bound is exact, so the first time a
    # complete assignment is popped it is the true next-best one.
    heap: List[Tuple[float, int, int, float, Tuple[int, ...]]] = [
        (suffix_best[0][target_charge], 0, target_charge, 0.0, ())
    ]
    while heap:
        _bound, index, remaining, energy, chosen = heapq.heappop(heap)
        if index == n_groups:
            yield (
                {
                    keys[position]: list(per_group_options[position][option][0])
                    for position, option in enumerate(chosen)
                },
                energy,
            )
            continue
        following = suffix_best[index + 1]
        for option, (_pairs, option_charge, option_energy) in enumerate(
            per_group_options[index]
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


def _iterate_group_assignments(
    groups: Sequence[SiteGroup],
    charge: int,
    max_mixing: int,
    *,
    gamma: float,
    deep_anion_ea: float,
    anchor_anions: bool,
    single_valent: FrozenSet[str] = frozenset(),
    top_k: Optional[int] = None,
) -> List[Tuple[GroupAssignment, float]]:
    """The search with its candidate-expansion loop around it.

    The free groups start from the anchored / default candidate lists and are
    Shannon-expanded when nothing balances; constrained groups never change.
    Empty when no charge-balanced assignment exists on any candidate set.
    """
    candidates = _initial_candidates(groups, anchor_anions)
    while True:
        table = _group_option_table(
            groups,
            candidates,
            max_mixing,
            gamma=gamma,
            deep_anion_ea=deep_anion_ea,
            single_valent=single_valent,
        )
        results: List[Tuple[GroupAssignment, float]] = []
        if table is not None:
            for item in _search_assignments_by_energy(table, charge):
                results.append(item)
                if top_k is not None and len(results) >= top_k:
                    break
        if results:
            return results
        expanded = _expand_candidates(candidates)
        if expanded == candidates:
            return []
        candidates = expanded


def enumerate_site_group_assignments(
    labels: Sequence[str],
    charge: int = 0,
    max_mixing: int = 2,
    *,
    constraints: Optional[SiteConstraints] = None,
    gamma: float = DEFAULT_GAMMA,
    deep_anion_ea: float = 0.0,
    anchor_anions: bool = True,
    top_k: Optional[int] = None,
) -> List[Tuple[GroupAssignment, float]]:
    """Energy-ranked charge-balanced assignments, resolved to site groups.

    ``constraints`` maps atom indices and/or element symbols to the states they
    may take -- an int pins, an iterable restricts (see :func:`site_groups`).
    Each result maps every group to its ``[(oxi, n), ...]`` split, which is
    enough to place states on sites without ever assigning a constrained atom a
    state it was not allowed; the energy includes the pinned atoms' share, so it
    is comparable with the unconstrained result for the same cell.
    """
    groups = site_groups(labels, constraints)
    if not groups:
        return []
    return _iterate_group_assignments(
        groups,
        int(charge),
        max_mixing,
        gamma=gamma,
        deep_anion_ea=deep_anion_ea,
        anchor_anions=anchor_anions,
        top_k=top_k,
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
    constraints: Optional[SiteConstraints] = None,
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

    ``constraints`` restricts atoms or elements to given states (see
    :func:`enumerate_site_group_assignments`). The element-level distributions
    returned here fold the per-atom groups back together, so two distinct
    site-level solutions can appear as one repeated distribution; callers that
    need to place states on sites want the group form instead.
    """
    return [
        (merge_group_assignment(assignment), energy)
        for assignment, energy in enumerate_site_group_assignments(
            labels,
            charge,
            max_mixing,
            constraints=constraints,
            gamma=gamma,
            deep_anion_ea=deep_anion_ea,
            anchor_anions=anchor_anions,
            top_k=top_k,
        )
    ]


def min_energies_with_single_valence(
    labels: Sequence[str],
    element_groups: Iterable[FrozenSet[str]],
    charge: int = 0,
    max_mixing: int = 2,
    *,
    gamma: float = DEFAULT_GAMMA,
    deep_anion_ea: float = 0.0,
    anchor_anions: bool = True,
    constraints: Optional[SiteConstraints] = None,
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
    groups = site_groups(labels, constraints)
    if not groups:
        return None
    candidates = _initial_candidates(groups, anchor_anions)

    def first_on(candidate_sets, single_valent=frozenset()):
        table = _group_option_table(
            groups,
            candidate_sets,
            max_mixing,
            gamma=gamma,
            deep_anion_ea=deep_anion_ea,
            single_valent=single_valent,
        )
        if table is None:
            return None
        return next(_search_assignments_by_energy(table, int(charge)), None)

    while True:
        best = first_on(candidates)
        if best is not None:
            break
        expanded = _expand_candidates(candidates)
        if expanded == candidates:
            return None
        candidates = expanded

    constrained: Dict[FrozenSet[str], float] = {}
    for group in element_groups:
        group = frozenset(group)
        result = first_on(candidates, group)
        constrained[group] = math.inf if result is None else result[1]
    return best[1], constrained
