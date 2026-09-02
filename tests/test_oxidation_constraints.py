"""Constrained oxidation-state solves: pins and allowed subsets, per atom and per element.

A constraint is part of the solve, not a patch on its answer. The pinned atoms
keep their states, the model re-balances everything else against the same net
charge, and the energy it reports includes the pinned atoms' share so it is
comparable with the unconstrained result.
"""

from __future__ import annotations

import numpy as np
import pytest

from quick_mag.generation import generate_single_perovskite
from quick_mag.magnetic_moments import (
    expand_group_assignments_to_site_assignments,
)
from quick_mag.oxidation_state_energy import (
    assignment_energy,
    constrained_site_states,
    enumerate_oxidation_states_by_energy,
    enumerate_site_group_assignments,
    merge_group_assignment,
    min_energies_with_single_valence,
    site_groups,
)

LAFEO3 = ["La"] * 8 + ["Fe"] * 8 + ["O"] * 24
IRONS = [index for index, label in enumerate(LAFEO3) if label == "Fe"]


def counts(distribution, element):
    return dict(distribution[element])


def net_charge(distribution):
    return sum(oxi * n for pairs in distribution.values() for oxi, n in pairs)


class TestGroups:
    def test_without_constraints_every_element_is_one_free_group(self):
        groups = site_groups(LAFEO3)
        assert [(g.element, g.count, g.states) for g in groups] == [
            ("La", 8, None), ("Fe", 8, None), ("O", 24, None),
        ]

    def test_a_pin_splits_its_element_and_leaves_the_others_alone(self):
        groups = site_groups(LAFEO3, {IRONS[2]: 2})
        by_element = {}
        for group in groups:
            by_element.setdefault(group.element, []).append(group)
        assert len(by_element["La"]) == 1 and len(by_element["O"]) == 1
        free, pinned = by_element["Fe"]
        assert free.states is None and free.count == 7
        assert pinned.sites == (IRONS[2],) and pinned.pinned_state == 2

    def test_atoms_sharing_an_allowed_set_share_a_group(self):
        groups = site_groups(LAFEO3, {IRONS[0]: [2, 3], IRONS[1]: (3, 2)})
        constrained = [g for g in groups if g.constrained]
        assert len(constrained) == 1
        assert constrained[0].sites == (IRONS[0], IRONS[1])
        assert constrained[0].states == frozenset({2, 3})

    def test_element_constraints_apply_to_every_atom_without_its_own(self):
        resolved = constrained_site_states(LAFEO3, {"Fe": [3, 4], IRONS[0]: 2})
        assert resolved[IRONS[0]] == frozenset({2})
        assert all(resolved[index] == frozenset({3, 4}) for index in IRONS[1:])
        assert not any(index in resolved for index, l in enumerate(LAFEO3) if l != "Fe")

    def test_stale_indices_and_empty_sets_are_handled(self):
        assert constrained_site_states(LAFEO3, {10_000: 2}) == {}
        with pytest.raises(ValueError):
            site_groups(LAFEO3, {IRONS[0]: []})


class TestSearch:
    def test_no_constraints_reproduces_the_unconstrained_search(self):
        free = enumerate_oxidation_states_by_energy(LAFEO3, top_k=1)
        assert counts(free[0][0], "Fe") == {3: 8}
        assert free[0][1] == pytest.approx(assignment_energy(free[0][0]))

    def test_a_pin_is_kept_and_the_cell_is_rebalanced(self):
        ranked = enumerate_oxidation_states_by_energy(
            LAFEO3, top_k=1, constraints={IRONS[0]: 2}
        )
        distribution, energy = ranked[0]
        assert net_charge(distribution) == 0
        # One Fe2+ costs the cell one electron; the model puts the hole on another Fe.
        assert counts(distribution, "Fe") == {2: 1, 3: 6, 4: 1}
        assert counts(distribution, "La") == {3: 8}
        assert counts(distribution, "O") == {-2: 24}
        # The energy includes the pinned atom, so it is comparable with the free one.
        assert energy == pytest.approx(assignment_energy(distribution))
        assert energy > enumerate_oxidation_states_by_energy(LAFEO3, top_k=1)[0][1]

    def test_the_mixing_budget_is_the_models_not_the_users(self):
        # max_mixing=2 limits what the model invents per group; the pinned atom
        # does not eat into it, otherwise the sensible answer above is impossible.
        ranked = enumerate_oxidation_states_by_energy(
            ["Fe"] * 4 + ["O"] * 6, top_k=1, constraints={0: 2}
        )
        assert counts(ranked[0][0], "Fe") == {2: 1, 3: 2, 4: 1}

    def test_a_pin_respects_the_target_charge(self):
        ranked = enumerate_oxidation_states_by_energy(
            LAFEO3, charge=-1, top_k=1, constraints={IRONS[0]: 2}
        )
        assert net_charge(ranked[0][0]) == -1
        assert counts(ranked[0][0], "Fe") == {2: 1, 3: 7}

    def test_an_element_subset_restricts_the_whole_element(self):
        ranked = enumerate_oxidation_states_by_energy(
            LAFEO3, charge=-2, top_k=1, constraints={"Fe": [2, 3]}
        )
        assert counts(ranked[0][0], "Fe") == {2: 2, 3: 6}
        ranked = enumerate_oxidation_states_by_energy(
            LAFEO3, top_k=3, constraints={"Fe": [3]}
        )
        assert all(counts(d, "Fe") == {3: 8} for d, _ in ranked)

    def test_a_per_atom_subset_is_honoured_on_that_atom(self):
        ranked = enumerate_site_group_assignments(
            LAFEO3, top_k=1, constraints={IRONS[0]: [2, 4]}
        )
        assignment, _energy = ranked[0]
        (group,) = [g for g in assignment if g.constrained]
        assert group.sites == (IRONS[0],)
        assert {oxi for oxi, _n in assignment[group]} <= {2, 4}
        assert net_charge(merge_group_assignment(assignment)) == 0

    def test_constrained_states_are_taken_as_given(self):
        # Anchoring and Shannon expansion apply to the free groups only: the
        # oxygen the user restricted to -1 stays there even though O is anchored.
        oxygens = [i for i, l in enumerate(LAFEO3) if l == "O"]
        ranked = enumerate_oxidation_states_by_energy(
            LAFEO3, top_k=1, constraints={oxygens[0]: -1}
        )
        assert counts(ranked[0][0], "O") == {-1: 1, -2: 23}
        assert net_charge(ranked[0][0]) == 0

    def test_an_impossible_pin_yields_nothing(self):
        # Every Fe held at 4+ and every La at 3+ cannot be balanced by anchored O.
        assert enumerate_oxidation_states_by_energy(
            ["La", "Fe", "O", "O", "O"], top_k=1, constraints={"Fe": 4, "La": 3, "O": -2}
        ) == []

    def test_single_valence_minima_accept_constraints(self):
        result = min_energies_with_single_valence(
            LAFEO3, [frozenset({"Fe"})], constraints={IRONS[0]: 2}
        )
        assert result is not None
        best, constrained = result
        # The free Fe group can be single-valent (all 2+ or all 4+ or ...) only
        # at a higher energy than the mixed completion.
        assert constrained[frozenset({"Fe"})] > best


@pytest.fixture(scope="module")
def structure():
    return generate_single_perovskite(
        "t", a_site="La", b_site="Fe", x_site="O", a=4.0,
        n_cells_x=2, n_cells_y=2, n_cells_z=2, defects=[],
    )


class TestSiteExpansion:

    def test_pinned_atoms_get_their_state_and_the_rest_balance(self, structure):
        labels = structure.element_symbols()
        irons = [i for i, l in enumerate(labels) if l == "Fe"]
        ranked = enumerate_site_group_assignments(labels, top_k=1, constraints={irons[3]: 2})
        (assignment,) = expand_group_assignments_to_site_assignments(
            ranked, structure, max_assignments=1
        )
        states = assignment.site_oxidation_states
        assert int(states[irons[3]]) == 2
        assert int(np.sum(states)) == 0
        assert sorted(int(states[i]) for i in irons) == [2] + [3] * 6 + [4]
        assert assignment.distributions["Fe"] == {2: 1, 3: 6, 4: 1}
        assert assignment.total_energy == pytest.approx(ranked[0][1])
        # d6 high spin
        assert float(assignment.magnetic_moments[irons[3]]) == 4.0

    def test_a_broken_orbit_lets_the_hole_land_on_any_survivor(self, structure):
        labels = structure.element_symbols()
        irons = [i for i, l in enumerate(labels) if l == "Fe"]
        ranked = enumerate_site_group_assignments(labels, top_k=1, constraints={irons[0]: 2})
        expanded = expand_group_assignments_to_site_assignments(ranked, structure)
        holes = {
            next(i for i in irons if int(a.site_oxidation_states[i]) == 4) for a in expanded
        }
        assert holes == set(irons[1:])

    def test_prefer_like_orders_the_least_change_first(self, structure):
        labels = structure.element_symbols()
        irons = [i for i, l in enumerate(labels) if l == "Fe"]
        ranked = enumerate_site_group_assignments(labels, top_k=1, constraints={irons[0]: 2})
        previous = np.zeros(structure.atom_count, dtype=int)
        previous[[i for i, l in enumerate(labels) if l != "Fe"]] = -99
        previous[irons] = 3
        previous[irons[5]] = 4
        first = expand_group_assignments_to_site_assignments(
            ranked, structure, max_assignments=1, prefer_like=previous
        )[0]
        assert int(first.site_oxidation_states[irons[5]]) == 4
