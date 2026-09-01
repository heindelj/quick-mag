"""Hand-set oxidation states: what they mean, and how far each one reaches.

The model now assigns one set of oxidation states -- its lowest-energy one -- and
disagreeing with it is an edit rather than a different row in a ranked list. These
tests pin the two things an edit has to get right: which atoms it reaches, and
what it invalidates downstream.

The reach is decided by the structure rather than by a mode. On a *unit cell*
there is no site that is not part of the repeating motif, so naming one names it
in every cell of any supercell built from it afterwards. On a *supercell*,
breaking the periodicity is the reason to be in a supercell, so the edit stops at
the atom.
"""

from __future__ import annotations

import numpy as np
import pytest

from quick_mag.oxidation_overrides import (
    OxidationOverrides,
    assignment_with_overrides,
    distributions_from_states,
    reference_site_of_atoms,
    supercell_matrix,
)


def unit_cell_state():
    """An ``AppState`` focused on a 1x1x1 perovskite, solved."""
    from quick_mag.quick_mag_ui import AppState

    state = AppState()
    state.sync_builder_binding()
    # The first pass records the builder signature as the baseline without
    # regenerating; only the second sees a change to apply.
    state.regenerate_focus_from_builder_if_changed()
    state.perovskite_supercell_x = 1
    state.perovskite_supercell_y = 1
    state.perovskite_supercell_z = 1
    state.regenerate_focus_from_builder_if_changed()
    state.run_magnetic_structure_calculation(structure=state.focus)
    return state


def grow_to(state, cells: int):
    """Resize the focused build to ``cells`` per axis and re-run the analysis."""
    state.perovskite_supercell_x = cells
    state.perovskite_supercell_y = cells
    state.perovskite_supercell_z = cells
    state.regenerate_focus_from_builder_if_changed()
    state.run_magnetic_structure_calculation(structure=state.focus)
    return state.magnetic_analysis_structure


def states_of(state, element: str):
    """Every oxidation state carried by ``element`` in the analysed structure."""
    structure = state.magnetic_analysis_structure
    assignment = state.selected_oxidation_assignment()
    return [
        int(assignment.site_oxidation_states[index])
        for index, symbol in enumerate(structure.element_symbols())
        if symbol == element
    ]


class TestSupercellMatrix:
    def test_a_repeat_is_recognised_as_one(self):
        reference = np.diag([4.0, 4.0, 4.0])
        assert np.array_equal(
            supercell_matrix(reference * 3.0, reference), 3 * np.eye(3, dtype=int)
        )

    def test_a_strained_cell_still_resolves_against_the_unstrained_one(self):
        # The matrix is integer, so the metric cancels: nudging the lattice constant
        # must not throw away edits made before the nudge.
        reference = np.diag([4.0, 4.0, 4.0])
        assert np.array_equal(
            supercell_matrix(np.diag([8.3, 8.2, 8.1]), reference),
            2 * np.eye(3, dtype=int),
        )

    def test_the_integrality_bound_scales_with_the_repeat(self):
        # The drift from a lattice-constant edit is N times the fractional change,
        # so an absolute bound would reject on a large supercell what it accepts on
        # a small one -- the same edit, resolving or not depending on the cell size.
        reference = np.diag([4.0, 4.0, 4.0])
        assert np.array_equal(
            supercell_matrix(np.diag([13.2, 13.2, 13.2]), reference),  # 3 x 4.4
            3 * np.eye(3, dtype=int),
        )

    def test_an_unrelated_cell_does_not(self):
        reference = np.diag([4.0, 4.0, 4.0])
        # Neither one cell nor two along c, by more than any strain accounts for.
        assert supercell_matrix(np.diag([4.0, 4.0, 5.4]), reference) is None
        # ...and a shear is not a repeat however close its edge lengths come.
        sheared = np.array([[4.0, 1.4, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
        assert supercell_matrix(sheared, reference) is None

    def test_a_singular_lattice_is_refused_rather_than_raising(self):
        assert supercell_matrix(np.diag([4.0, 4.0, 4.0]), np.zeros((3, 3))) is None


class TestAssignmentRewrite:
    def test_distributions_are_counted_off_the_sites(self):
        assert distributions_from_states(
            ["Fe", "Fe", "O"], [3, 4, -2]
        ) == {"Fe": {3: 1, 4: 1}, "O": {-2: 1}}

    def test_an_untouched_assignment_comes_back_as_the_same_object(self):
        # Identity, not equality: the per-frame memo and the exchange rebuild both
        # lean on "nothing changed" being cheap to see.
        state = unit_cell_state()
        assignment = state.predicted_oxidation_assignment()
        assert (
            assignment_with_overrides(
                assignment, state.magnetic_analysis_structure, OxidationOverrides()
            )
            is assignment
        )

    def test_the_moment_follows_the_state(self):
        state = unit_cell_state()
        structure = state.magnetic_analysis_structure
        iron = structure.element_symbols().index("Fe")

        before = state.selected_oxidation_assignment()
        state.set_site_oxidation_state(iron, 4)
        after = state.selected_oxidation_assignment()

        assert int(after.site_oxidation_states[iron]) == 4
        # Fe(3+) is d5 and Fe(4+) is d4: the formal high-spin moment has to move
        # with the charge, or the spin solve would run on the old d-shell.
        assert float(after.magnetic_moments[iron]) != float(before.magnetic_moments[iron])
        assert after.distributions["Fe"] == {4: 1}


class TestReach:
    def test_a_unit_cell_edit_reaches_every_cell_of_a_later_supercell(self):
        state = unit_cell_state()
        structure = state.magnetic_analysis_structure
        assert state.oxidation_edits_propagate()

        state.set_site_oxidation_state(structure.element_symbols().index("Fe"), 4)
        assert states_of(state, "Fe") == [4]

        grow_to(state, 2)
        assert not state.oxidation_edits_propagate()
        # Every octahedron of the 2x2x2, not just the one the edit was made on.
        assert states_of(state, "Fe") == [4] * 8
        # ...and nothing else moved with it.
        assert set(states_of(state, "La")) == {3}
        assert set(states_of(state, "O")) == {-2}

    def test_a_supercell_edit_reaches_one_atom(self):
        state = unit_cell_state()
        structure = grow_to(state, 2)
        irons = [
            index
            for index, symbol in enumerate(structure.element_symbols())
            if symbol == "Fe"
        ]

        state.set_site_oxidation_state(irons[1], 2)
        assert states_of(state, "Fe").count(2) == 1
        assert state.site_oxidation_is_edited(irons[1])
        assert not state.site_oxidation_is_edited(irons[0])

    def test_reverting_in_a_supercell_falls_back_to_the_cell_edit(self):
        # The one-atom edit is the only thing that goes: dropping the propagating
        # edit from inside a supercell would silently change every other copy.
        state = unit_cell_state()
        state.set_site_oxidation_state(
            state.magnetic_analysis_structure.element_symbols().index("Fe"), 4
        )
        structure = grow_to(state, 2)
        irons = [
            index
            for index, symbol in enumerate(structure.element_symbols())
            if symbol == "Fe"
        ]

        state.set_site_oxidation_state(irons[1], 2)
        assert sorted(states_of(state, "Fe")) == [2] + [4] * 7
        state.revert_site_oxidation_state(irons[1])
        assert states_of(state, "Fe") == [4] * 8

    def test_resizing_drops_the_by_index_edits_and_keeps_the_geometric_ones(self):
        state = unit_cell_state()
        state.set_site_oxidation_state(
            state.magnetic_analysis_structure.element_symbols().index("Fe"), 4
        )
        structure = grow_to(state, 2)
        iron = structure.element_symbols().index("Fe")
        state.set_site_oxidation_state(iron, 2)

        grow_to(state, 3)
        # An atom index stopped naming the atom it was recorded against, so the
        # one-atom edit is gone; the propagating one is keyed geometrically.
        assert states_of(state, "Fe") == [4] * 27
        assert not state.oxidation_overrides.atom_states

    def test_a_cell_edit_survives_strain_tilt_and_a_further_resize(self):
        # The reference cell is stored with the lattice constant it had, so every
        # later edit to the geometry has to resolve against a stale one. Each of
        # these used to be a way for a set of edits to silently stop applying
        # partway through a session.
        state = unit_cell_state()
        state.set_site_oxidation_state(
            state.magnetic_analysis_structure.element_symbols().index("Fe"), 4
        )

        grow_to(state, 2)
        assert states_of(state, "Fe") == [4] * 8

        state.lattice_a = state.lattice_b = state.lattice_c = 4.4
        state.regenerate_focus_from_builder_if_changed()
        state.run_magnetic_structure_calculation(structure=state.focus)
        assert states_of(state, "Fe") == [4] * 8

        # A tilt pattern moves the copies of a site off identical fractional
        # coordinates; nearest-site matching is what carries the edit across it.
        state.perovskite_tilt_system = 3
        state.tilt_angle_x = state.tilt_angle_y = state.tilt_angle_z = 8.0
        state.regenerate_focus_from_builder_if_changed()
        state.run_magnetic_structure_calculation(structure=state.focus)
        assert states_of(state, "Fe") == [4] * 8

        grow_to(state, 3)
        assert states_of(state, "Fe") == [4] * 27

    def test_clearing_hands_everything_back_to_the_model(self):
        state = unit_cell_state()
        state.set_site_oxidation_state(
            state.magnetic_analysis_structure.element_symbols().index("Fe"), 4
        )
        grow_to(state, 2)
        assert states_of(state, "Fe") == [4] * 8

        state.clear_oxidation_overrides()
        assert states_of(state, "Fe") == [3] * 8
        assert state.oxidation_overrides.is_empty()

    def test_a_state_outside_the_editable_range_is_clamped(self):
        state = unit_cell_state()
        iron = state.magnetic_analysis_structure.element_symbols().index("Fe")
        state.set_site_oxidation_state(iron, 500)
        from quick_mag.quick_mag_ui import MAX_EDITABLE_OXIDATION_STATE

        assert state.site_oxidation_state(iron) == MAX_EDITABLE_OXIDATION_STATE


class TestReferenceResolution:
    def test_atoms_map_to_the_nearest_site_of_their_own_element(self):
        state = unit_cell_state()
        overrides = OxidationOverrides()
        overrides.capture_reference(state.magnetic_analysis_structure)
        structure = grow_to(state, 2)

        sites = reference_site_of_atoms(structure, overrides)
        assert sites is not None
        reference_symbols = overrides.reference_labels
        for atom, symbol in enumerate(structure.element_symbols()):
            assert reference_symbols[sites[atom]] == symbol
        # Every reference site is claimed by exactly one atom per cell of the 2x2x2.
        counts = np.bincount(sites, minlength=len(reference_symbols))
        assert set(counts.tolist()) == {8}

    def test_an_edit_authored_on_a_different_cell_is_not_carried_over(self):
        overrides = OxidationOverrides()
        state = unit_cell_state()
        overrides.set(state.magnetic_analysis_structure, 1, 4, propagate=True)
        assert overrides.cell_states == {1: 4}

        # A different composition is a different cell; keeping the old entries would
        # apply another material's chemistry here.
        state.b_site_element = "Mn"
        state.regenerate_focus_from_builder_if_changed()
        state.run_magnetic_structure_calculation(structure=state.focus)
        overrides.set(state.magnetic_analysis_structure, 2, -1, propagate=True)
        assert overrides.cell_states == {2: -1}


class TestSiteList:
    """The scrolling per-atom list: what it offers, and what it costs to build."""

    def test_the_filter_narrows_to_one_element_in_first_appearance_order(self):
        from quick_mag.quick_mag_ui import (
            oxidation_list_elements,
            oxidation_listed_atoms,
        )

        state = unit_cell_state()
        structure = grow_to(state, 2)
        elements = oxidation_list_elements(state, structure)
        assert elements == ["La", "Fe", "O"]

        state.oxidation_list_filter = 0
        assert oxidation_listed_atoms(state, structure) == list(range(structure.atom_count))

        state.oxidation_list_filter = 1 + elements.index("Fe")
        listed = oxidation_listed_atoms(state, structure)
        symbols = structure.element_symbols()
        assert listed and all(symbols[atom] == "Fe" for atom in listed)
        assert len(listed) == symbols.count("Fe")

    def test_an_out_of_range_filter_falls_back_to_showing_everything(self):
        # The filter index outlives the structure it was chosen against -- a
        # rebuild can drop an element -- so it has to degrade rather than raise.
        from quick_mag.quick_mag_ui import oxidation_listed_atoms

        state = unit_cell_state()
        structure = state.magnetic_analysis_structure
        state.oxidation_list_filter = 99
        assert oxidation_listed_atoms(state, structure) == list(range(structure.atom_count))


class TestDownstream:
    def test_an_edit_rebuilds_the_exchange_matrix_while_updates_are_live(self):
        # An oxidation state sets a site's d-shell, which sets its couplings. With
        # interactive updates on, an edit has to move J rather than leave the panel
        # showing couplings for the previous chemistry.
        state = unit_cell_state()
        grow_to(state, 2)
        state.update_spin_energies_interactively = True
        state._interactive_updates_live = True
        before = np.array(state.magnetic_j_matrix, copy=True)

        iron = state.magnetic_analysis_structure.element_symbols().index("Fe")
        state.set_site_oxidation_state(iron, 2)

        assert not state.spin_energies_stale
        assert state.magnetic_j_matrix.shape == before.shape
        assert not np.allclose(state.magnetic_j_matrix, before)

    def test_an_edit_only_marks_stale_when_updates_are_paused(self):
        # Same gate the builder edits go through: below the frame-rate floor the
        # rebuild stops paying for itself, and the panel says the energies are stale
        # rather than spending the frame.
        state = unit_cell_state()
        grow_to(state, 2)
        state.update_spin_energies_interactively = False
        before = np.array(state.magnetic_j_matrix, copy=True)

        iron = state.magnetic_analysis_structure.element_symbols().index("Fe")
        state.set_site_oxidation_state(iron, 2)

        assert state.spin_energies_stale
        assert np.allclose(state.magnetic_j_matrix, before)
        # The states themselves still moved: the readouts are immediate, only the
        # energies wait for a refresh.
        assert state.site_oxidation_state(iron) == 2

    def test_a_refused_edit_is_reported_rather_than_raised(self):
        # The box is wired to a rebuild of the exchange matrix over tables that do
        # not have an entry for every charge anyone can type. A refusal has to
        # reach the panel as a message; raising would take the session with it.
        state = unit_cell_state()
        iron = state.magnetic_analysis_structure.element_symbols().index("Fe")

        def explode(*_args, **_kwargs):
            raise ValueError("no descriptor for that ion")

        state.oxidation_overrides.set = explode  # type: ignore[method-assign]
        state.set_site_oxidation_state(iron, 4)

        assert "no descriptor for that ion" in state.oxidation_edit_message
        assert state.site_oxidation_state(iron) == 3

    def test_a_successful_edit_clears_a_previous_complaint(self):
        state = unit_cell_state()
        state.oxidation_edit_message = "something went wrong earlier"
        state.set_site_oxidation_state(
            state.magnetic_analysis_structure.element_symbols().index("Fe"), 4
        )
        assert state.oxidation_edit_message == ""

    def test_render_radii_follow_an_edit(self):
        # The radius lookup is a Shannon table keyed by (element, charge), so an
        # edited site has to change size in the 3D view -- that is the visible
        # confirmation the edit landed.
        from quick_mag.quick_mag_ui import (
            structure_atom_render_radii,
            structure_site_oxidation_states,
        )

        state = unit_cell_state()
        structure = state.magnetic_analysis_structure
        iron = structure.element_symbols().index("Fe")

        def radius():
            return float(
                structure_atom_render_radii(
                    structure,
                    structure_site_oxidation_states(state, structure),
                    render_with_ionic_radius=True,
                )[iron]
            )

        before = radius()
        state.set_site_oxidation_state(iron, 4)
        assert radius() != pytest.approx(before)
