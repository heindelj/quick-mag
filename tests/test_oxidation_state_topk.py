"""The top-k oxidation-state search must match brute-force enumeration exactly.

``enumerate_oxidation_states_by_energy`` ranks assignments with a DP-bounded
best-first search instead of materializing every charge-balanced assignment;
these tests pin it (and the constrained-minimum queries the double-exchange gate
uses) against the brute-force reference on compositions small enough to
enumerate fully.
"""

from __future__ import annotations

import math
from collections import Counter

import pytest

from quick_mag.oxidation_state_energy import (
    STANDARD_ANION_STATES,
    assignment_energy,
    enumerate_oxidation_states_by_energy,
    min_energies_with_single_valence,
)
from quick_mag.oxidation_state_enumeration import (
    _default_possible_oxidation_states,
    _expand_possible_oxidation_states_with_shannon,
    _find_charge_balanced_assignments,
)


def brute_force_ranked(labels, charge=0, max_mixing=2):
    """The original materialize-everything reference implementation."""
    composition = dict(Counter(labels))
    possible = {
        element: (
            [STANDARD_ANION_STATES[element]]
            if element in STANDARD_ANION_STATES
            else _default_possible_oxidation_states(element)
        )
        for element in composition
    }
    if any(not states for states in possible.values()):
        return []
    while True:
        assignments = _find_charge_balanced_assignments(
            composition=composition,
            possible_oxi_states=possible,
            max_mixing_per_element=max_mixing,
            target_charge=charge,
        )
        if assignments:
            break
        expanded = _expand_possible_oxidation_states_with_shannon(possible)
        if expanded == possible:
            return []
        possible = expanded
    scored = [(assignment, assignment_energy(assignment)) for assignment in assignments]
    scored.sort(key=lambda item: item[1])
    return scored


def canonical(distribution):
    return tuple(
        (element, tuple(sorted(pairs))) for element, pairs in sorted(distribution.items())
    )


COMPOSITIONS = [
    ("LaMnO3", ["La"] * 4 + ["Mn"] * 4 + ["O"] * 12, 0),
    ("Fe3O4", ["Fe"] * 3 + ["O"] * 4, 0),
    ("LaSr3FeCoO12", ["La"] + ["Sr"] * 3 + ["Fe"] * 2 + ["Co"] * 2 + ["O"] * 12, 0),
    ("LaSr3Mn4O12", ["La"] + ["Sr"] * 3 + ["Mn"] * 4 + ["O"] * 12, 0),
    ("small high-entropy", ["La"] * 2 + ["Sr"] * 2 + ["Fe", "Co", "Mn", "Ni"] + ["O"] * 12, 0),
    ("charged fragment", ["Mn"] * 2 + ["O"] * 3, 2),
]


@pytest.mark.parametrize("name,labels,charge", COMPOSITIONS, ids=[c[0] for c in COMPOSITIONS])
def test_search_matches_brute_force(name, labels, charge):
    reference = brute_force_ranked(labels, charge=charge)
    found = enumerate_oxidation_states_by_energy(labels, charge=charge, top_k=None)

    assert len(found) == len(reference) > 0
    ref_energies = [energy for _, energy in reference]
    new_energies = [energy for _, energy in found]
    assert new_energies == pytest.approx(ref_energies)
    # Energy order is exact but tie order is not specified: compare the
    # distributions as sets within each run of equal energies.
    assert {canonical(d) for d, _ in found} == {canonical(d) for d, _ in reference}
    for (dist, energy) in found:
        assert assignment_energy(dist) == pytest.approx(energy)


@pytest.mark.parametrize("top_k", [1, 5, 37])
def test_top_k_is_a_prefix_of_the_full_ranking(top_k):
    labels = ["La"] + ["Sr"] * 3 + ["Fe"] * 2 + ["Co"] * 2 + ["O"] * 12
    full = enumerate_oxidation_states_by_energy(labels, top_k=None)
    head = enumerate_oxidation_states_by_energy(labels, top_k=top_k)
    assert len(head) == min(top_k, len(full))
    assert [e for _, e in head] == pytest.approx([e for _, e in full[: len(head)]])


def test_unknown_element_returns_empty():
    assert enumerate_oxidation_states_by_energy(["Xx", "O"]) == []


def test_min_energies_with_single_valence_matches_brute_force():
    labels = ["La"] + ["Sr"] * 3 + ["Fe"] * 2 + ["Co"] * 2 + ["O"] * 12
    reference = brute_force_ranked(labels)
    groups = [frozenset({"Fe"}), frozenset({"Co"}), frozenset({"Fe", "Co"})]
    result = min_energies_with_single_valence(labels, groups)
    assert result is not None
    global_min, constrained = result
    assert global_min == pytest.approx(reference[0][1])
    for group in groups:
        expected = min(
            (
                energy
                for dist, energy in reference
                if all(len(dist.get(element, [])) == 1 for element in group)
            ),
            default=math.inf,
        )
        if math.isinf(expected):
            assert math.isinf(constrained[group])
        else:
            assert constrained[group] == pytest.approx(expected)


def test_high_entropy_composition_is_fast():
    # 5 mixed B-site elements over 27 sites: brute force enumerates ~13.4 million
    # assignments; the search must stay output-sensitive.
    labels = (
        ["La"] * 18 + ["Sr"] * 9
        + ["Mn"] * 11 + ["Cr"] * 3 + ["Co"] * 2 + ["Ni"] * 7 + ["Fe"] * 4
        + ["O"] * 81
    )
    import time

    start = time.perf_counter()
    ranked = enumerate_oxidation_states_by_energy(labels, top_k=200)
    elapsed = time.perf_counter() - start
    assert len(ranked) == 200
    assert elapsed < 5.0
    energies = [energy for _, energy in ranked]
    assert energies == sorted(energies)
