"""The builder panel's domain model: a structure is a stack of domains."""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

pytest = __import__("pytest")
pytest.importorskip("imgui_bundle")

from quick_mag.quick_mag_ui import AppState  # noqa: E402


def _frame(state: AppState) -> None:
    """What one frame of the Controls panel does to the model."""
    state.apply_perovskite_constraints()
    state.regenerate_focus_from_builder_if_changed()
    state.sync_active_structure()


def _fresh_state() -> AppState:
    state = AppState()
    _frame(state)
    _frame(state)  # establishes the regeneration baseline
    return state


class DomainBuilderTest(unittest.TestCase):
    def test_a_new_structure_has_one_domain_that_is_the_buffer(self) -> None:
        state = _fresh_state()
        self.assertEqual(state.domain_count(), 1)
        domain = state.reference_domain()
        self.assertEqual(domain.n_cells, (3, 3, 3))
        self.assertEqual(domain.a_site_element, state.a_site_element)
        self.assertFalse(state.focus.generation_parameters.is_multi_domain())

    def test_adding_a_domain_grows_the_focus_along_the_stacking_axis(self) -> None:
        state = _fresh_state()
        before = state.focus.atom_count
        self.assertTrue(state.add_domain(), state.domain_message)
        self.assertEqual(state.active_domain_index, 1)
        state.a_site_element = "Sr"
        state.b_site_element = "Ti"
        state.perovskite_supercell_z = 2
        _frame(state)
        focus = state.focus
        self.assertEqual(focus.generation_parameters.grid_shape(), (3, 3, 5))
        self.assertGreater(focus.atom_count, before)
        counts = Counter(focus.atomic_labels)
        self.assertEqual(counts["Ti"], 18)
        self.assertEqual(counts["Fe"], 27)
        self.assertTrue(focus.generation_parameters.is_multi_domain())
        # The new domain's own spacing along c is its own; in plane it is shared.
        state.perovskite_type = 1
        state.lattice_c = 3.9
        _frame(state)
        np.testing.assert_allclose(np.diag(state.focus.lattice), [12.0, 12.0, 3 * 4.0 + 2 * 3.9])

    def test_in_plane_edits_propagate_and_impossible_ones_are_refused(self) -> None:
        state = _fresh_state()
        state.add_domain()
        state.select_domain(0)
        state.perovskite_supercell_x = 4
        _frame(state)
        self.assertEqual([d.n_cells[0] for d in state.builder_domains], [4, 4])
        state.select_domain(1)
        # A double perovskite needs an even in-plane grid: 4x3 is not.
        self.assertFalse(state.set_active_domain_formula(1))
        self.assertIn("even", state.domain_message)
        state.select_domain(0)
        state.perovskite_supercell_y = 4
        _frame(state)
        state.select_domain(1)
        self.assertTrue(state.set_active_domain_formula(1), state.domain_message)
        _frame(state)
        self.assertEqual(state.builder_domains[1].n_cells[:2], (2, 2))
        self.assertEqual(state.focus.generation_parameters.grid_shape()[:2], (4, 4))
        # Shrinking domain 0 in plane would orphan the double perovskite.
        state.select_domain(0)
        state.perovskite_supercell_x = 3
        _frame(state)
        self.assertEqual(state.perovskite_supercell_x, 4)
        self.assertTrue(state.domain_message)

    def test_tilts_apply_to_the_whole_stack(self) -> None:
        state = _fresh_state()
        state.add_domain()
        state.perovskite_tilt_system = 22  # a-a-a-
        state.tilt_angle_x = 8.0
        _frame(state)
        params = state.focus.generation_parameters
        self.assertEqual(params.tilt_system, "a-a-a-")
        self.assertTrue(params.is_multi_domain())
        # The tilt moves oxygens in the second domain too.
        tilted = state.focus.cartesian_coords.copy()
        top_layer = tilted[:, 2] > 12.0
        state.tilt_angle_x = 0.0
        _frame(state)
        untilted = state.focus.cartesian_coords
        self.assertEqual(state.focus.generation_parameters.tilt_angle_x_deg, 0.0)
        self.assertFalse(np.allclose(tilted[top_layer], untilted[top_layer]))

    def test_periodicity_is_per_axis(self) -> None:
        state = _fresh_state()
        periodic_count = state.focus.atom_count
        state.periodic_axes_flags = (True, True, False)
        _frame(state)
        focus = state.focus
        self.assertEqual(focus.periodic_axes, (True, True, False))
        self.assertTrue(focus.is_periodic)
        self.assertEqual(focus.generation_parameters.periodic_axes, (True, True, False))
        # One closing A plane and one X face along c only.
        self.assertEqual(focus.atom_count, periodic_count + 9 + 9)
        self.assertEqual(state.treat_as_periodic, (True, True, False))

    def test_loading_a_stacked_structure_rebinds_its_domains(self) -> None:
        state = _fresh_state()
        state.add_domain()
        state.a_site_element = "Sr"
        _frame(state)
        stacked = state.focus
        state.create_new_structure()
        _frame(state)
        self.assertEqual(state.domain_count(), 1)
        state.set_focus(stacked)
        state._builder_bound_id = None
        _frame(state)
        self.assertEqual(state.domain_count(), 2)
        self.assertEqual(state.builder_domains[1].a_site_element, "Sr")
        self.assertEqual(state.active_domain_index, 0)

    def test_removing_the_last_added_domain_restores_a_single_block(self) -> None:
        state = _fresh_state()
        single = state.focus.atom_count
        state.add_domain()
        _frame(state)
        state.remove_domain(1)
        _frame(state)
        self.assertEqual(state.domain_count(), 1)
        self.assertEqual(state.focus.atom_count, single)
        self.assertFalse(state.focus.generation_parameters.is_multi_domain())


if __name__ == "__main__":
    unittest.main()


class StructurePanelDomainChoiceTest(unittest.TestCase):
    def test_choosing_a_domain_from_the_panel_focuses_and_selects(self) -> None:
        from quick_mag.quick_mag_ui import domain_cell_vertices

        state = _fresh_state()
        state.add_domain()
        _frame(state)
        stacked = state.focus
        state.create_new_structure()
        _frame(state)
        self.assertIsNot(state.focus, stacked)
        state.choose_structure_domain(stacked, 1)
        _frame(state)
        self.assertIs(state.focus, stacked)
        self.assertEqual(state.active_domain_index, 1)
        self.assertEqual(state.domain_index_for_structure(stacked), 1)
        # The yellow box is the top block: full in plane, its own slice along c.
        corners = domain_cell_vertices(stacked, 1, use_cartesian=True)
        self.assertIsNotNone(corners)
        np.testing.assert_allclose(corners[:, 2].min(), 12.0)
        np.testing.assert_allclose(corners[:, 2].max(), 16.0)
        np.testing.assert_allclose(corners[:, 0].max(), 12.0)
        # A single-domain structure has no domain box.
        self.assertIsNone(domain_cell_vertices(state.structures[-1], 0, use_cartesian=True))
        # Refocusing another structure and coming back remembers the choice.
        state.set_focus(state.structures[-1])
        _frame(state)
        state.set_focus(stacked)
        _frame(state)
        self.assertEqual(state.active_domain_index, 1)
