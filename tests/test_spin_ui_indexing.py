from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The UI module imports imgui-bundle (the optional ``ui`` extra). Skip this whole module
# when it is not installed so the core suite still runs with just ``pip install .[dev]``.
pytest.importorskip("imgui_bundle")

from quick_mag.classify_spin_structure import (  # noqa: E402
    classify_structure_by_cubes,
    site_indexing_from_generation_parameters,
)
from imgui_bundle import implot3d  # noqa: E402
from quick_mag.defects import compensation_hint  # noqa: E402
from quick_mag.quick_mag_ui import (  # noqa: E402
    DEFECT_KIND_KEYS,
    FORMULA_MODE_KEYS,
    MAX_LISTED_OXIDATION_ASSIGNMENTS,
    MAX_MATCH_DEFECT_CONCENTRATION,
    SPHERE_LATITUDE_SEGMENTS,
    SPHERE_LONGITUDE_SEGMENTS,
    DEFECT_ROLE_LABELS,
    GLAZER_TILT_SYSTEMS,
    STRUCTURE_ZOOM_RANGE,
    AppState,
    build_sphere_mesh,
    sphere_detail_for,
    highlighted_render_indices,
    oxidation_site_rows,
    structure_atom_render_radii,
    visible_role_indices,
    vacancy_render_radii,
    vacancy_render_sites,
    compute_plot_box_limits,
    spin_alignment_edge_segments,
    structure_plot_flags,
    structure_with_moments,
    zoom_after_wheel,
)
from quick_mag.magnetic_moments import OxidationStateAssignment  # noqa: E402
from quick_mag.spin_solver import (  # noqa: E402
    SpinConfig,
    canonical_moment_key,
    compute_config_energy,
)
from quick_mag.structure import SavedSpinConfiguration  # noqa: E402


def apply_builder_edits(state: AppState, **fields: object) -> None:
    """Edit the focused structure through the builder, as one UI frame would.

    ``gui_controls`` binds the builder at the top of the frame and applies edits at
    the bottom, so a baseline pass is needed before the edited fields take effect.

    Interactive spin-energy updates are switched on here: most of these tests assert
    on the landscape immediately after an edit, which is the live behaviour. The
    off-by-default path has its own tests in ``StaleSpinEnergyTests``.
    """
    state.update_spin_energies_interactively = True
    state.sync_builder_binding()
    state.regenerate_focus_from_builder_if_changed()  # establish the baseline
    for name, value in fields.items():
        setattr(state, name, value)
    state.regenerate_focus_from_builder_if_changed()  # apply the edits in place


class StructureListTests(unittest.TestCase):
    def test_app_starts_with_one_focused_solve_ready_structure(self) -> None:
        state = AppState()
        self.assertEqual(len(state.structures), 1)
        structure = state.structures[0]
        self.assertIs(state.focus, structure)
        # Solve-ready straight away: real geometry plus builder provenance.
        self.assertGreater(structure.atom_count, 0)
        self.assertIsNotNone(structure.generation_parameters)
        self.assertTrue(state.builder_enabled())

    def test_new_structure_resets_the_builder_to_defaults(self) -> None:
        state = AppState()
        first = state.structures[0]
        apply_builder_edits(state, b_site_element="Mn", perovskite_supercell_x=3)
        self.assertEqual(state.b_site_element, "Mn")

        state.create_new_structure()
        self.assertEqual(len(state.structures), 2)
        self.assertIs(state.focus, state.structures[1])
        self.assertEqual(state.b_site_element, "Fe")
        self.assertEqual(state.perovskite_supercell_x, 3)
        # The earlier structure keeps the edits it was given.
        self.assertIn("Mn", first.atomic_labels)

    def test_new_structure_names_do_not_collide(self) -> None:
        state = AppState()
        state.rename_structure(state.structures[0], "Structure 2")
        state.create_new_structure()
        names = [structure.name for structure in state.structures]
        self.assertEqual(len(set(names)), len(names))

    def test_rename_disambiguates_against_other_structures_only(self) -> None:
        state = AppState()
        state.create_new_structure()
        first, second = state.structures
        state.rename_structure(first, "LaFeO3")
        state.rename_structure(second, "LaFeO3")
        self.assertEqual(first.name, "LaFeO3")
        self.assertEqual(second.name, "LaFeO3 (2)")
        # Renaming to its own name is a no-op, not a collision.
        state.rename_structure(second, "LaFeO3 (2)")
        self.assertEqual(second.name, "LaFeO3 (2)")

    def test_deleting_the_last_structure_leaves_a_fresh_default(self) -> None:
        state = AppState()
        original = state.structures[0]
        state.remove_structure(original)
        self.assertEqual(len(state.structures), 1)
        self.assertIsNot(state.structures[0], original)
        self.assertIs(state.focus, state.structures[0])

    def test_deleting_the_focus_falls_back_to_a_neighbour(self) -> None:
        state = AppState()
        state.create_new_structure()
        first, second = state.structures
        state.set_focus(second)
        state.remove_structure(second)
        self.assertEqual(state.structures, [first])
        self.assertIs(state.focus, first)


# The eight classical orderings -- the period-2 plane patterns. The reference set
# also carries longer-period patterns (E-type and friends) wherever the grid has
# enough planes to tell them apart, so these are asserted as a subset.
CANONICAL_REFERENCE_NAMES = {"G", "C(a)", "C(b)", "C(c)", "F", "A(a)", "A(b)", "A(c)"}


def reference_names(state: AppState) -> set[str]:
    return {name for name, _ in state.reference_configs}


def random_configs(state: AppState, count: int, seed: int = 0) -> list[SpinConfig]:
    """Arbitrary non-canonical configurations of the right width for ``state``."""
    n_mag = len(state.magnetic_site_indices)
    rng = np.random.default_rng(seed)
    return [
        SpinConfig(
            energy=0.0,
            all_moments=rng.choice([-1.0, 1.0], size=n_mag),
            magnetization=0.0,
            n_unpaired=float(n_mag),
        )
        for _ in range(count)
    ]


# The builder fields for a tilt. ``a0a0c+`` only reads the c angle, but the inactive
# ones are set too so the edit does not depend on which axis the system happens to use.
TILT_EDIT = {
    "perovskite_tilt_system": 1,
    "tilt_angle_x": 12.0,
    "tilt_angle_y": 12.0,
    "tilt_angle_z": 12.0,
}


def tilt_the_cell(state: AppState, degrees: float = 12.0) -> None:
    """Apply an a-a-a- style tilt through the builder, as the UI would."""
    apply_builder_edits(
        state,
        perovskite_tilt_system=1,
        tilt_angle_x=degrees,
        tilt_angle_y=degrees,
        tilt_angle_z=degrees,
    )


class StructurePlotViewTests(unittest.TestCase):
    """The view box is recomputed every frame, so it owns pan and zoom itself."""

    def test_implot3d_pan_and_zoom_are_disabled(self) -> None:
        # Both move the axis limits, which the per-frame centring would overwrite.
        flags = structure_plot_flags(show_legend=True)
        self.assertTrue(flags & implot3d.Flags_.no_pan.value)
        self.assertTrue(flags & implot3d.Flags_.no_zoom.value)

    def test_rotation_stays_enabled(self) -> None:
        flags = structure_plot_flags(show_legend=True)
        self.assertFalse(flags & implot3d.Flags_.no_rotate.value)
        self.assertFalse(flags & implot3d.Flags_.no_inputs.value)

    def test_equal_axes_and_legend_toggle_are_preserved(self) -> None:
        with_legend = structure_plot_flags(show_legend=True)
        without_legend = structure_plot_flags(show_legend=False)
        self.assertTrue(with_legend & implot3d.Flags_.equal.value)
        self.assertFalse(with_legend & implot3d.Flags_.no_legend.value)
        self.assertTrue(without_legend & implot3d.Flags_.no_legend.value)

    def test_box_is_centred_on_the_structure(self) -> None:
        # Coordinates deliberately offset from the origin and unequal per axis.
        coords = np.array(
            [[10.0, 4.0, -2.0], [14.0, 5.0, 1.0], [12.0, 9.0, -0.5]], dtype=float
        )
        lo_a, hi_a, lo_b, hi_b, lo_c, hi_c = compute_plot_box_limits(coords)
        centre = np.array(
            [(lo_a + hi_a) / 2, (lo_b + hi_b) / 2, (lo_c + hi_c) / 2], dtype=float
        )
        expected = 0.5 * (coords.min(axis=0) + coords.max(axis=0))
        np.testing.assert_allclose(centre, expected)
        # Equal spans on every axis, so "equal" axis scaling is not distorted.
        spans = [hi_a - lo_a, hi_b - lo_b, hi_c - lo_c]
        np.testing.assert_allclose(spans, [spans[0]] * 3)

    def test_zoom_shrinks_the_box_without_moving_its_centre(self) -> None:
        coords = np.array([[10.0, 4.0, -2.0], [14.0, 9.0, 1.0]], dtype=float)

        def box(zoom: float):
            limits = compute_plot_box_limits(coords, padding_scale=1.8 / zoom)
            centre = np.array(
                [
                    (limits[0] + limits[1]) / 2,
                    (limits[2] + limits[3]) / 2,
                    (limits[4] + limits[5]) / 2,
                ]
            )
            return centre, limits[1] - limits[0]

        centre_1, span_1 = box(1.0)
        centre_2, span_2 = box(2.0)
        np.testing.assert_allclose(centre_1, centre_2)  # stays centred
        self.assertAlmostEqual(span_2, span_1 / 2.0)  # and zooms in

    def test_wheel_zoom_is_symmetric_and_clamped(self) -> None:
        self.assertGreater(zoom_after_wheel(1.0, 1.0), 1.0)
        self.assertLess(zoom_after_wheel(1.0, -1.0), 1.0)
        self.assertEqual(zoom_after_wheel(1.0, 0.0), 1.0)
        # Scrolling in then out returns to where it started.
        self.assertAlmostEqual(zoom_after_wheel(zoom_after_wheel(1.0, 3.0), -3.0), 1.0)
        # The structure can never be zoomed away to nothing or past the box.
        low, high = STRUCTURE_ZOOM_RANGE
        self.assertEqual(zoom_after_wheel(low, -50.0), low)
        self.assertEqual(zoom_after_wheel(high, 50.0), high)


class SpinLandscapeTests(unittest.TestCase):
    def test_new_structure_is_seeded_with_reference_configurations(self) -> None:
        state = AppState()
        self.assertLessEqual(CANONICAL_REFERENCE_NAMES, reference_names(state))
        # Plotted with no solve, and every point is a labelled reference.
        self.assertEqual(
            len(state.displayed_spin_configs()), len(state.reference_configs)
        )
        self.assertEqual(set(state.spin_classification_labels()), reference_names(state))

    def test_reference_energies_are_single_point_evaluations(self) -> None:
        state = AppState()
        for config in state.displayed_spin_configs():
            self.assertAlmostEqual(
                config.energy,
                compute_config_energy(state.magnetic_j_matrix, config.all_moments),
                places=12,
            )

    def test_a_near_miss_keeps_the_ordering_and_reports_how_far_off_it_is(self) -> None:
        state = AppState()
        reference = state.displayed_spin_configs()[0]
        label = state.label_for_config(reference)
        self.assertNotEqual(label, "Other")
        self.assertEqual(state.described_config(reference), label)

        # One flipped spin is still that ordering, with a defect concentration on it.
        n_sites = len(reference.all_moments)
        perturbed_moments = np.array(reference.all_moments, dtype=float, copy=True)
        perturbed_moments[0] *= -1.0
        perturbed = replace(reference, all_moments=perturbed_moments)

        self.assertEqual(state.label_for_config(perturbed), label)
        match = state.match_for_config(perturbed)
        self.assertEqual(match.defect_count, 1)
        self.assertAlmostEqual(match.concentration, 1 / n_sites)
        self.assertIn("defects", state.described_config(perturbed))

    def test_a_configuration_no_better_than_chance_is_still_other(self) -> None:
        state = AppState()
        reference = state.displayed_spin_configs()[0]
        moments = np.array(reference.all_moments, dtype=float, copy=True)
        # Flip enough spins to push past the cutoff in every direction.
        rng = np.random.default_rng(11)
        moments *= np.where(rng.random(moments.shape) < 0.5, -1.0, 1.0)
        scrambled = replace(reference, all_moments=moments)

        match = state.match_for_config(scrambled)
        self.assertGreater(match.concentration, MAX_MATCH_DEFECT_CONCENTRATION)
        self.assertEqual(state.label_for_config(scrambled), "Other")
        self.assertEqual(state.described_config(scrambled), "Other")

    def test_the_deviating_sites_are_the_ones_that_were_flipped(self) -> None:
        state = AppState()
        reference = dict(state.reference_configs)["G"]
        moments = np.array(reference.all_moments, dtype=float, copy=True)
        moments[[2, 9]] *= -1.0
        perturbed = replace(reference, all_moments=moments)

        match = state.match_for_config(perturbed)
        self.assertEqual(np.flatnonzero(match.mismatched).tolist(), [2, 9])
        self.assertEqual(
            state.spin_defect_site_indices(perturbed),
            [state.magnetic_site_indices[2], state.magnetic_site_indices[9]],
        )

    def test_global_spin_inversion_is_the_same_ordering(self) -> None:
        state = AppState()
        reference = state.displayed_spin_configs()[0]
        flipped = replace(reference, all_moments=-np.asarray(reference.all_moments))
        self.assertEqual(
            state.label_for_config(flipped), state.label_for_config(reference)
        )

    def test_a_orientations_are_degenerate_when_cubic_and_split_when_tilted(self) -> None:
        state = AppState()

        def energies() -> dict[str, float]:
            return {
                state.label_for_config(config): round(config.energy, 9)
                for config in state.displayed_spin_configs()
            }

        cubic = energies()
        self.assertEqual(len({cubic["A(a)"], cubic["A(b)"], cubic["A(c)"]}), 1)

        tilt_the_cell(state)
        tilted = energies()
        self.assertGreater(len({tilted["A(a)"], tilted["A(b)"], tilted["A(c)"]}), 1)

    def test_a_builder_edit_re_energizes_rather_than_resets(self) -> None:
        state = AppState()
        before = sorted(tuple(c.all_moments) for c in state.spin_landscape)
        cubic_energies = [c.energy for c in state.spin_landscape]

        tilt_the_cell(state)

        after = sorted(tuple(c.all_moments) for c in state.spin_landscape)
        self.assertEqual(before, after)  # same configurations...
        self.assertNotEqual(cubic_energies, [c.energy for c in state.spin_landscape])

    def test_changing_the_cell_size_drops_configurations_of_the_old_cell(self) -> None:
        state = AppState()
        n_mag_before = len(state.magnetic_site_indices)
        apply_builder_edits(state, perovskite_supercell_x=4)

        self.assertNotEqual(len(state.magnetic_site_indices), n_mag_before)
        # Freshly seeded for the new cell; nothing carried over at the old length.
        self.assertLessEqual(CANONICAL_REFERENCE_NAMES, reference_names(state))
        for config in state.spin_landscape:
            self.assertEqual(len(config.all_moments), len(state.magnetic_site_indices))

    def test_a_solved_landscape_does_not_leak_into_the_next_structure(self) -> None:
        state = AppState()
        n_mag = len(state.magnetic_site_indices)
        solved = [
            SpinConfig(
                energy=0.0,
                all_moments=np.array(moments, dtype=float),
                magnetization=0.0,
                n_unpaired=float(n_mag),
            )
            for moments in ([1.0] * (n_mag - 1) + [-1.0], [-1.0] + [1.0] * (n_mag - 1))
        ]
        n_references = len(state.reference_configs)
        state.merge_solver_states_into_landscape(solved)
        self.assertGreater(len(state.spin_landscape), n_references)

        # The new structure has the same magnetic-site count, so a length check alone
        # would happily re-energize and present the old structure's configurations.
        state.create_new_structure()
        state.sync_active_structure()
        self.assertEqual(len(state.magnetic_site_indices), n_mag)
        self.assertEqual(len(state.spin_landscape), len(state.reference_configs))
        self.assertEqual(state.magnetic_solution_cache, {})

    def test_an_unanalysable_structure_clears_the_previous_analysis(self) -> None:
        state = AppState()
        self.assertTrue(state.spin_landscape)

        # SrMnF3 does not charge-balance with standard oxidation states, so no
        # assignment is found. The previous structure's J matrix and assignments must
        # not survive and be shown as if they described this one.
        apply_builder_edits(
            state, a_site_element="Sr", b_site_element="Mn", x_site_element="F"
        )
        self.assertEqual(state.spin_landscape, [])
        self.assertEqual(state.reference_configs, [])
        self.assertEqual(state.magnetic_oxidation_assignments, [])
        self.assertEqual(state.magnetic_site_indices, [])
        self.assertEqual(state.magnetic_j_matrix.size, 0)
        self.assertTrue(state.baseline_status)

    def test_the_retention_cap_never_truncates_the_references(self) -> None:
        state = AppState()
        state.plot_degenerate_configs = True
        state.spin_landscape = list(state.spin_landscape) + random_configs(state, 20)

        state.spin_plot_max_configs = 12
        state.refresh_landscape_energies()
        labels = state.spin_classification_labels()
        self.assertLessEqual(len(state.displayed_spin_configs()), 12)
        self.assertLessEqual(CANONICAL_REFERENCE_NAMES, set(labels))

        # A cap below the reference count still keeps every reference.
        state.spin_plot_max_configs = 2
        state.refresh_landscape_energies()
        self.assertEqual(set(state.spin_classification_labels()), reference_names(state))


class StaleSpinEnergyTests(unittest.TestCase):
    """Builder edits leave the landscape alone unless updates are interactive.

    Re-energizing means rebuilding the oxidation assignments and the exchange matrix,
    which is far too expensive to do on every frame of a slider drag.
    """

    @staticmethod
    def _edit(state: AppState, **fields: object) -> None:
        """``apply_builder_edits`` without its interactive-updates override."""
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        for name, value in fields.items():
            setattr(state, name, value)
        state.regenerate_focus_from_builder_if_changed()

    def test_an_edit_marks_the_energies_stale_and_leaves_them_untouched(self) -> None:
        state = AppState()
        self.assertFalse(state.update_spin_energies_interactively)
        before = [config.energy for config in state.spin_landscape]
        self.assertTrue(before)

        self._edit(state, **TILT_EDIT)

        self.assertTrue(state.spin_energies_stale)
        self.assertEqual([config.energy for config in state.spin_landscape], before)

    def test_refreshing_picks_up_the_edit(self) -> None:
        state = AppState()
        before = [config.energy for config in state.spin_landscape]
        self._edit(state, **TILT_EDIT)

        state.refresh_spin_energies()

        self.assertFalse(state.spin_energies_stale)
        # A tilt splits the degenerate reference orderings apart.
        self.assertNotEqual([config.energy for config in state.spin_landscape], before)

    def test_refreshing_is_a_no_op_when_nothing_is_stale(self) -> None:
        state = AppState()
        state.refresh_spin_energies()
        energies = [config.energy for config in state.spin_landscape]
        state.refresh_spin_energies()
        self.assertEqual([config.energy for config in state.spin_landscape], energies)

    def test_interactive_updates_re_energize_on_every_edit(self) -> None:
        state = AppState()
        state.update_spin_energies_interactively = True
        before = [config.energy for config in state.spin_landscape]

        self._edit(state, **TILT_EDIT)

        self.assertFalse(state.spin_energies_stale)
        self.assertNotEqual([config.energy for config in state.spin_landscape], before)

    def test_solving_clears_staleness(self) -> None:
        state = AppState()
        self._edit(state, **TILT_EDIT)
        self.assertTrue(state.spin_energies_stale)

        state.run_magnetic_structure_calculation(structure=state.focus)

        self.assertFalse(state.spin_energies_stale)


class LargeStructurePerformanceTest(unittest.TestCase):
    """Guards on the things that made a large cell unusable."""

    def test_assignment_labels_are_capped_and_cached(self):
        # A double perovskite enumerates tens of thousands of assignments; formatting
        # every one of them for the combo cost ~220 ms per frame.
        state = AppState()
        state.magnetic_oxidation_assignments = [
            state.magnetic_oxidation_assignments[0]
        ] * (MAX_LISTED_OXIDATION_ASSIGNMENTS * 3)
        labels = state.oxidation_assignment_labels()
        self.assertEqual(len(labels), MAX_LISTED_OXIDATION_ASSIGNMENTS)
        self.assertIs(labels, state.oxidation_assignment_labels())

    def test_the_assignment_limit_bounds_what_is_enumerated(self):
        # Expanding every charge-balanced distribution into per-site assignments was
        # the slowest part of setting up a solve: 78,312 of them on the old DQ
        # default, at ~5 s. They are energy-ranked, so the head is what matters.
        state = AppState()
        state.max_oxidation_assignments = 7
        state.formula_mode = FORMULA_MODE_KEYS.index("dq")
        state.apply_defaults_for_formula()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.run_magnetic_structure_calculation(structure=state.focus)

        self.assertEqual(len(state.magnetic_oxidation_assignments), 7)
        energies = [a.total_energy for a in state.magnetic_oxidation_assignments]
        self.assertEqual(energies, sorted(energies), "the kept assignments are the ranked head")

    def test_zero_means_no_limit(self):
        state = AppState()
        self.assertIsNone(AppState(max_oxidation_assignments=0).oxidation_assignment_limit())
        self.assertIsNone(AppState(max_oxidation_assignments=-1).oxidation_assignment_limit())
        self.assertEqual(state.oxidation_assignment_limit(), state.max_oxidation_assignments)

    def test_pattern_matches_are_computed_once_for_the_whole_list(self):
        state = AppState()
        matches = state.displayed_pattern_matches()
        self.assertEqual(len(matches), len(state.displayed_spin_configs()))
        # Same list object back: one cache entry for the landscape, not one per row.
        self.assertIs(matches, state.displayed_pattern_matches())
        # ...and it is invalidated when the landscape is rebuilt.
        state.spin_landscape = list(state.spin_landscape) + random_configs(state, 5)
        state.refresh_landscape_energies()
        self.assertIsNot(matches, state.displayed_pattern_matches())

    def test_sphere_detail_steps_down_only_for_large_structures(self):
        full = (SPHERE_LATITUDE_SEGMENTS, SPHERE_LONGITUDE_SEGMENTS)
        self.assertEqual(sphere_detail_for(199), full)
        self.assertEqual(sphere_detail_for(400), full)
        coarse = sphere_detail_for(1315)
        self.assertLess(coarse[0] * coarse[1], full[0] * full[1])
        # Monotonic: more atoms never means more vertices per atom.
        counts = [sphere_detail_for(n) for n in (100, 500, 1500, 10_000)]
        products = [lat * lon for lat, lon in counts]
        self.assertEqual(products, sorted(products, reverse=True))

    def test_the_detail_level_reaches_the_mesh(self):
        coords = np.zeros((1, 3))
        radii = np.array([1.0])
        lattice = np.eye(3) * 10.0
        fine = build_sphere_mesh(coords, radii, lattice, use_cartesian=True, detail=(8, 16))
        coarse = build_sphere_mesh(coords, radii, lattice, use_cartesian=True, detail=(5, 10))
        self.assertGreater(len(fine.idx), len(coarse.idx))


class OrderingPlaneOverlayTest(unittest.TestCase):
    """The sheets have to land on the atoms, not between them."""

    def _state(self) -> AppState:
        state = AppState()
        state.update_spin_energies_interactively = True
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        return state

    def test_every_magnetic_site_lies_on_a_drawn_plane(self):
        # The magnetic sublattice does not share the cell's origin -- a perovskite's
        # B sites sit half a primitive cell off the corner -- so an offset taken to be
        # the plane index puts every sheet exactly between two layers of atoms.
        state = self._state()
        sublattice = state.magnetic_sublattice()
        fractional = state.focus.fractional_coords[sublattice.site_indices]

        for name in ("G", "A(c)", "C(c)"):
            with self.subTest(ordering=name):
                config = dict(state.reference_configs)[name]
                miller, offsets, _colors = state.miller_plane_overlay(config)
                projection = fractional @ miller
                for value in projection:
                    self.assertLess(
                        float(np.min(np.abs(offsets - value))),
                        1e-9,
                        f"a magnetic site is off every drawn {name} plane",
                    )

    def test_the_sheets_account_for_every_site_exactly_once(self):
        state = self._state()
        sublattice = state.magnetic_sublattice()
        fractional = state.focus.fractional_coords[sublattice.site_indices]
        config = dict(state.reference_configs)["G"]
        miller, offsets, colors = state.miller_plane_overlay(config)

        projection = fractional @ miller
        occupancy = [int(np.sum(np.abs(projection - o) < 1e-9)) for o in offsets]
        self.assertEqual(sum(occupancy), len(projection))
        self.assertEqual(len(colors), len(offsets))
        # Alternating sheets: G flips sign on every successive plane.
        self.assertTrue(
            all(a != b for a, b in zip(colors, colors[1:])),
            "G-type sheets should alternate between the two spin colours",
        )


class SelectedConfigurationPersistenceTest(unittest.TestCase):
    """A builder edit re-sorts the landscape; the selection must follow its state."""

    def _solved(self) -> AppState:
        state = AppState()
        state.update_spin_energies_interactively = True
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.run_magnetic_structure_calculation(structure=state.focus)
        return state

    def test_an_edit_keeps_the_same_arrangement_on_screen(self):
        state = self._solved()
        state.selected_spin_config_index = 4
        before = state.selected_spin_config()
        key = canonical_moment_key(before.all_moments)

        apply_builder_edits(state, **TILT_EDIT)

        after = state.selected_spin_config()
        self.assertEqual(canonical_moment_key(after.all_moments), key)
        # It moved in the list -- which is the whole point of not holding the index.
        self.assertNotEqual(state.selected_spin_config_index, 4)

    def test_the_energy_shown_is_the_edited_one(self):
        state = self._solved()
        state.selected_spin_config_index = 4
        before_energy = state.selected_spin_config().energy

        apply_builder_edits(state, **TILT_EDIT)

        self.assertNotAlmostEqual(state.selected_spin_config().energy, before_energy)

    def test_a_replication_change_falls_back_to_the_ground_state(self):
        # The old arrangement has the wrong length for the new cell, so there is
        # nothing to hold on to.
        state = self._solved()
        state.selected_spin_config_index = 4
        state.selected_spin_config()

        apply_builder_edits(state, perovskite_supercell_x=4)

        self.assertEqual(state.selected_spin_config_index, 0)
        self.assertIsNotNone(state.selected_spin_config())

    def test_a_fresh_solve_still_presents_its_ground_state(self):
        state = self._solved()
        state.selected_spin_config_index = 4
        state.selected_spin_config()

        state.run_selected_oxidation_assignment(force=True)

        self.assertEqual(state.selected_spin_config_index, 0)

    def test_toggling_the_plot_cap_keeps_the_selection(self):
        state = self._solved()
        state.selected_spin_config_index = 3
        key = canonical_moment_key(state.selected_spin_config().all_moments)

        state.plot_degenerate_configs = not state.plot_degenerate_configs
        state.refresh_landscape_energies()

        self.assertEqual(
            canonical_moment_key(state.selected_spin_config().all_moments), key
        )


class SiteRoleVisibilityTest(unittest.TestCase):
    def test_roles_can_be_switched_off_independently(self):
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        rendered = state.rendered_structure()
        roles = rendered.generation_parameters.site_roles

        everything = visible_role_indices(rendered, show_a=True, show_b=True, show_x=True)
        self.assertEqual(len(everything), rendered.atom_count)

        b_only = visible_role_indices(rendered, show_a=False, show_b=True, show_x=False)
        self.assertTrue(all(roles[index] == "B" for index in b_only))
        self.assertEqual(len(b_only), sum(1 for role in roles if role == "B"))

    def test_a_structure_without_provenance_draws_everything(self):
        # A loaded file has no site roles, so there is nothing to switch off.
        state = AppState()
        structure = state.focus
        structure.generation_parameters = None
        self.assertEqual(
            len(visible_role_indices(structure, show_a=False, show_b=False, show_x=False)),
            structure.atom_count,
        )

    def test_every_role_starts_visible(self):
        state = AppState()
        self.assertTrue(state.show_a_sites)
        self.assertTrue(state.show_b_sites)
        self.assertTrue(state.show_x_sites)
        rendered = state.rendered_structure()
        self.assertEqual(
            len(
                visible_role_indices(
                    rendered,
                    show_a=state.show_a_sites,
                    show_b=state.show_b_sites,
                    show_x=state.show_x_sites,
                )
            ),
            rendered.atom_count,
        )


class SphereMeshCacheTests(unittest.TestCase):
    """The 3D view rebuilds its draw calls every frame; the meshes must not."""

    @staticmethod
    def _mesh(state: AppState):
        structure = state.rendered_structure()
        radii = structure_atom_render_radii(
            structure, None, render_with_ionic_radius=False
        )
        return build_sphere_mesh(
            structure.cartesian_coords,
            radii,
            structure.lattice,
            use_cartesian=True,
        )

    def test_identical_inputs_reuse_the_same_mesh(self) -> None:
        state = AppState()
        self.assertIs(self._mesh(state), self._mesh(state))

    def test_moving_the_atoms_builds_a_new_mesh(self) -> None:
        state = AppState()
        first = self._mesh(state)
        tilt_the_cell(state)
        self.assertIsNot(self._mesh(state), first)

    def test_a_zero_radius_site_is_dropped(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
        lattice = np.eye(3) * 10.0
        both = build_sphere_mesh(
            coords, np.array([1.0, 1.0]), lattice, use_cartesian=True
        )
        one = build_sphere_mesh(
            coords, np.array([1.0, 0.0]), lattice, use_cartesian=True
        )
        self.assertEqual(len(one.idx) * 2, len(both.idx))


class DegenerateConfigTests(unittest.TestCase):
    def test_collapsing_keeps_one_config_per_energy_plus_every_reference(self) -> None:
        state = AppState()
        state.spin_landscape = list(state.spin_landscape) + random_configs(state, 40)

        state.plot_degenerate_configs = False
        state.refresh_landscape_energies()
        shown = state.displayed_spin_configs()

        # Reference points are identified by their moments, not by their label: a
        # random configuration can now legitimately be labelled as the ordering it is
        # nearest to, so the label no longer says whether a point is a reference.
        reference_keys = {
            canonical_moment_key(config.all_moments)
            for _name, config in state.reference_configs
        }
        shown_keys = {canonical_moment_key(c.all_moments) for c in shown}
        self.assertLessEqual(reference_keys, shown_keys, "a reference was collapsed away")

        # One point per distinct energy, except where references share one: on a cubic
        # cell C(a)/C(b)/C(c) are degenerate and must all still be shown.
        others = [
            c for c in shown if canonical_moment_key(c.all_moments) not in reference_keys
        ]
        self.assertEqual(
            len({round(c.energy, 6) for c in others}),
            len(others),
            "collapsed non-reference points should have distinct energies",
        )

    def test_toggling_degeneracy_back_on_restores_the_hidden_points(self) -> None:
        state = AppState()
        state.spin_landscape = list(state.spin_landscape) + random_configs(state, 40)

        state.plot_degenerate_configs = False
        state.refresh_landscape_energies()
        collapsed = len(state.displayed_spin_configs())

        state.plot_degenerate_configs = True
        state.refresh_landscape_energies()
        expanded = len(state.displayed_spin_configs())
        self.assertGreater(expanded, collapsed)

        # ...and back again, so the pool is not consumed by collapsing.
        state.plot_degenerate_configs = False
        state.refresh_landscape_energies()
        self.assertEqual(len(state.displayed_spin_configs()), collapsed)

    def test_degeneracy_counts_the_configurations_at_each_energy(self) -> None:
        state = AppState()
        state.plot_degenerate_configs = True
        state.refresh_landscape_energies()
        by_label = dict(zip(state.spin_classification_labels(), state.displayed_spin_configs()))
        # On a cubic cell the three C orientations are degenerate with each other, and
        # so are the three diagonal up-up-down-down patterns -- all six share an energy.
        def sharing_energy(label: str) -> set[str]:
            energy = round(by_label[label].energy, 6)
            return {
                name
                for name, config in by_label.items()
                if round(config.energy, 6) == energy
            }

        self.assertLessEqual({"C(a)", "C(b)", "C(c)"}, sharing_energy("C(a)"))
        # Whatever shares an energy, every member reports the size of its group.
        for label in by_label:
            with self.subTest(label=label):
                self.assertEqual(
                    by_label[label].degeneracy, len(sharing_energy(label))
                )

    def test_collapsing_reaches_further_up_the_landscape(self) -> None:
        state = AppState()
        state.spin_landscape = list(state.spin_landscape) + random_configs(state, 200)
        # Above the reference count, so the cap has room to spend on the landscape
        # rather than being consumed by references that are always kept.
        state.spin_plot_max_configs = len(state.reference_configs) + 8

        def distinct_energies() -> int:
            state.refresh_landscape_energies()
            return len({round(c.energy, 6) for c in state.displayed_spin_configs()})

        state.plot_degenerate_configs = True
        with_degenerate = distinct_energies()
        state.plot_degenerate_configs = False
        without_degenerate = distinct_energies()
        self.assertGreater(without_degenerate, with_degenerate)


class SpinUiIndexingTests(unittest.TestCase):
    def test_compact_solver_moments_expand_to_builder_b_sites(self) -> None:
        state = AppState()
        structure = state.generated_chemical_structure()
        build = state.generated_perovskite()
        b_indices = np.asarray(build.b_site_indices, dtype=int)
        state.magnetic_site_indices = b_indices.tolist()

        compact_moments = np.arange(1, len(b_indices) + 1, dtype=float)
        expanded = state.expand_spin_moments_to_structure(compact_moments, structure)
        nonzero_indices = np.flatnonzero(np.linalg.norm(expanded, axis=1) > 1e-8)

        np.testing.assert_array_equal(nonzero_indices, b_indices)
        self.assertEqual({structure.atomic_labels[index] for index in nonzero_indices}, {"Fe"})
        self.assertEqual(
            {structure.atomic_labels[index] for index in range(int(b_indices[0]))},
            {"La"},
        )

    def test_selected_solver_moments_remap_to_rendered_generated_b_sites(self) -> None:
        state = AppState()
        source_structure = state.structures[-1]
        state.rename_structure(source_structure, "Saved LaFeO3")
        state.set_focus(source_structure)
        self.assertIsNotNone(source_structure)
        assert source_structure is not None
        source_build = state.generated_build_for_structure(source_structure)
        self.assertIsNotNone(source_build)
        assert source_build is not None
        state.render_periodic_images = True
        rendered_structure = state.rendered_structure()
        self.assertIsNotNone(rendered_structure)
        assert rendered_structure is not None
        rendered_build = state.generated_build_for_structure(rendered_structure)
        self.assertIsNotNone(rendered_build)
        assert rendered_build is not None

        state.magnetic_result_structure = source_structure
        state.magnetic_site_indices = np.asarray(source_build.b_site_indices, dtype=int).tolist()
        compact_moments = np.ones(len(state.magnetic_site_indices), dtype=float)
        config = type(
            "FakeSpinConfig",
            (),
            {"all_moments": compact_moments},
        )()
        state.magnetic_solution_cache[0] = ([], [config])

        remapped = state.selected_spin_moments_for_structure(rendered_structure)
        self.assertIsNotNone(remapped)
        assert remapped is not None
        nonzero_indices = np.flatnonzero(np.linalg.norm(remapped, axis=1) > 1e-8)

        np.testing.assert_array_equal(
            nonzero_indices,
            np.asarray(rendered_build.b_site_indices, dtype=int),
        )
        self.assertEqual(
            {rendered_structure.atomic_labels[index] for index in nonzero_indices},
            {"Fe"},
        )

    def test_spin_alignment_edges_group_by_alignment(self) -> None:
        state = AppState()
        structure = state.generated_chemical_structure()
        build = state.generated_perovskite()
        b_indices = np.asarray(build.b_site_indices, dtype=int).reshape(build.octahedra.shape)

        ferromagnetic = np.zeros((structure.atom_count, 3), dtype=float)
        g_type = np.zeros((structure.atom_count, 3), dtype=float)
        for i, j, k in np.ndindex(b_indices.shape):
            site_index = int(b_indices[i, j, k])
            ferromagnetic[site_index, 2] = 1.0
            g_type[site_index, 2] = 1.0 if (i + j + k) % 2 == 0 else -1.0

        ferromagnetic_edges = spin_alignment_edge_segments(
            structure.cartesian_coords,
            b_indices,
            ferromagnetic,
        )
        g_type_edges = spin_alignment_edge_segments(
            structure.cartesian_coords,
            b_indices,
            g_type,
        )

        # One bond per axis between adjacent cells: 3 * (n - 1) * n * n on an n^3 grid.
        nx, ny, nz = b_indices.shape
        expected_edges = (
            (nx - 1) * ny * nz + (ny - 1) * nx * nz + (nz - 1) * nx * ny
        )
        self.assertEqual(len(ferromagnetic_edges["aligned"]), expected_edges)
        self.assertEqual(len(ferromagnetic_edges["anti-aligned"]), 0)
        self.assertEqual(len(g_type_edges["aligned"]), 0)
        self.assertEqual(len(g_type_edges["anti-aligned"]), expected_edges)

    def test_solver_results_are_merged_with_the_canonical_references(self) -> None:
        state = AppState()
        structure = state.focus
        assert structure is not None
        b_indices = np.asarray(state.magnetic_site_indices, dtype=int)

        ferromagnetic = np.ones(len(b_indices), dtype=float)
        solver_states = [
            SpinConfig(
                energy=float(-0.5 * np.sum(ferromagnetic**2)),
                all_moments=ferromagnetic,
                magnetization=float(np.sum(ferromagnetic)),
                n_unpaired=float(np.sum(np.abs(ferromagnetic))),
            )
        ]

        with patch(
            "quick_mag.quick_mag_ui.solve_for_assignment",
            return_value=(solver_states, solver_states),
        ):
            state.run_selected_oxidation_assignment(force=True)

        # A single solver state does not crowd out the references: every canonical
        # ordering is still in the landscape, each labelled by exact match.
        labels = state.spin_classification_labels()
        self.assertEqual(
            {label for label in labels if label != "Other"},
            {name for name, _ in state.reference_configs},
        )
        # The solver's FM state is the reference F, so it merges rather than duplicating.
        self.assertEqual(labels.count("F"), 1)

    def test_focused_structure_is_rendered_independently_of_builder_edits(self) -> None:
        state = AppState()
        structure_a = state.structures[-1]
        state.rename_structure(structure_a, "LaFeO3")
        self.assertIsNotNone(structure_a)
        assert structure_a is not None

        state.create_new_structure()
        structure_b = state.structures[-1]
        state.rename_structure(structure_b, "SrMnF3")
        self.assertIsNot(structure_a, structure_b)
        apply_builder_edits(
            state,
            a_site_element="Sr",
            b_site_element="Mn",
            x_site_element="F",
        )

        state.set_focus(structure_a)
        state.render_periodic_images = False
        state.sync_active_structure()

        self.assertIs(state.active_structure, structure_a)
        self.assertIs(state.rendered_structure(), structure_a)
        self.assertEqual(structure_a.atomic_labels[0], "La")
        self.assertEqual(structure_b.atomic_labels[0], "Sr")

    def test_saved_spin_config_on_generated_structure_uses_focus_site_indexing(self) -> None:
        state = AppState()
        state.render_periodic_images = True
        structure = state.structures[-1]
        state.rename_structure(structure, "Saved LaFeO3")
        state.set_focus(structure)
        self.assertIsNotNone(structure)
        assert structure is not None
        build = state.generated_build_for_structure(structure)
        self.assertIsNotNone(build)
        assert build is not None

        b_indices = np.asarray(build.b_site_indices, dtype=int)
        saved_moments = np.zeros((structure.atom_count, 3), dtype=float)
        saved_moments[b_indices, 2] = np.where(
            np.arange(len(b_indices)) % 2 == 0,
            4.0,
            -4.0,
        )
        structure.spin_configurations.append(
            SavedSpinConfiguration(
                magnetic_moments=saved_moments,
                energy=-1.0,
                magnetization=0.0,
                classification="test",
                collinear=True,
            )
        )

        state.active_saved_spin_index = 0
        state.sync_active_structure()
        rendered = state.rendered_structure()
        self.assertIsNot(rendered, structure)
        assert rendered is not None

        displayed = state.displayed_saved_spin_moments(rendered)
        self.assertIsNotNone(displayed)
        assert displayed is not None
        nonzero_indices = np.flatnonzero(np.linalg.norm(displayed, axis=1) > 1e-8)
        rendered_build = state.generated_build_for_structure(rendered)
        self.assertIsNotNone(rendered_build)
        assert rendered_build is not None
        np.testing.assert_array_equal(
            nonzero_indices,
            np.asarray(rendered_build.b_site_indices, dtype=int),
        )
        self.assertEqual(
            {rendered.atomic_labels[index] for index in nonzero_indices},
            {"Fe"},
        )

    def test_displayed_configs_track_the_focused_structure(self) -> None:
        state = AppState()
        structure_a = state.structures[-1]
        self.assertIsNotNone(structure_a)
        assert structure_a is not None

        state.create_new_structure()
        structure_b = state.structures[-1]
        self.assertIsNot(structure_a, structure_b)
        apply_builder_edits(state, b_site_element="Mn", lattice_a=4.4)

        # Each structure carries its own landscape; switching focus re-seeds it
        # rather than showing the other structure's points.
        self.assertTrue(state.displayed_spin_configs())
        b_energies = [config.energy for config in state.displayed_spin_configs()]

        state.set_focus(structure_a)
        state.sync_active_structure()
        a_energies = [config.energy for config in state.displayed_spin_configs()]
        self.assertTrue(a_energies)
        self.assertNotEqual(a_energies, b_energies)

    def test_double_perovskite_uses_doubled_formula_cell_and_extra_b_site(self) -> None:
        state = AppState()
        state.perovskite_supercell_x = 1
        state.perovskite_supercell_y = 1
        state.perovskite_supercell_z = 1
        state.formula_mode = 1
        state.b2_site_element = "Co"

        structure = state.generated_chemical_structure()
        build = state.generated_perovskite()
        params = structure.generation_parameters
        assert params is not None

        self.assertEqual(build.octahedra.shape, (2, 2, 2))
        np.testing.assert_allclose(np.diag(structure.lattice), [8.0, 8.0, 8.0])
        self.assertEqual(params.formula_mode, "double")
        self.assertEqual((params.n_oct_x, params.n_oct_y, params.n_oct_z), (1, 1, 1))

        b_labels = [structure.atomic_labels[index] for index in build.b_site_indices]
        self.assertEqual(b_labels.count("Fe"), 4)
        self.assertEqual(b_labels.count("Co"), 4)

    def test_quadruple_perovskite_uses_doubled_formula_cell_and_extra_a_site(self) -> None:
        state = AppState()
        state.perovskite_supercell_x = 1
        state.perovskite_supercell_y = 1
        state.perovskite_supercell_z = 1
        state.formula_mode = 2
        state.a2_site_element = "Sr"

        structure = state.generated_chemical_structure()
        build = state.generated_perovskite()
        params = structure.generation_parameters
        assert params is not None

        self.assertEqual(build.octahedra.shape, (2, 2, 2))
        np.testing.assert_allclose(np.diag(structure.lattice), [8.0, 8.0, 8.0])
        self.assertEqual(params.formula_mode, "quadruple")

        a_labels = [structure.atomic_labels[index] for index in build.a_site_indices]
        self.assertEqual(a_labels.count("La"), 2)
        self.assertEqual(a_labels.count("Sr"), 6)

    def test_dq_perovskite_combines_a_and_b_site_ordering_with_defaults(self) -> None:
        state = AppState()
        state.formula_mode = 3
        state.apply_defaults_for_formula()
        # One primitive cell of a DQ perovskite is already a 2x2x2 octahedron grid,
        # which is the smallest cell that shows both orderings; the defaults open on
        # two of them, so shrink back to one after applying them.
        state.perovskite_supercell_x = 1
        state.perovskite_supercell_y = 1
        state.perovskite_supercell_z = 1

        structure = state.generated_chemical_structure()
        build = state.generated_perovskite()
        params = structure.generation_parameters
        assert params is not None

        self.assertEqual(build.octahedra.shape, (2, 2, 2))
        np.testing.assert_allclose(np.diag(structure.lattice), [8.0, 8.0, 8.0])
        self.assertEqual(params.formula_mode, "dq")
        self.assertEqual(params.a_site_element, "Ca")
        self.assertEqual(params.a2_site_element, "Mg")
        self.assertEqual(params.b_site_element, "Fe")
        self.assertEqual(params.b2_site_element, "Re")
        self.assertEqual(params.x_site_element, "O")

        a_labels = [structure.atomic_labels[index] for index in build.a_site_indices]
        b_labels = [structure.atomic_labels[index] for index in build.b_site_indices]
        x_labels = [structure.atomic_labels[index] for index in build.x_site_indices]
        self.assertEqual(a_labels.count("Ca"), 2)
        self.assertEqual(a_labels.count("Mg"), 6)
        self.assertEqual(b_labels.count("Fe"), 4)
        self.assertEqual(b_labels.count("Re"), 4)
        self.assertEqual(set(x_labels), {"O"})

    def test_formula_defaults_keep_initial_cell_sizes_consistent(self) -> None:
        # Supercell counts primitive cells, and the ordered modes already double
        # the grid through their unit factor -- so one of those equals two plain
        # perovskite cells. The plain modes open on a 3x3x3 octahedron grid and the
        # ordered ones on 4x4x4.
        expected_replications = {
            0: (3, 3, 3),
            1: (2, 2, 2),
            2: (2, 2, 2),
            3: (2, 2, 2),
            4: (3, 3, 3),
        }
        expected_lattice = {0: 12.0, 1: 16.0, 2: 16.0, 3: 16.0, 4: 12.0}

        for formula_mode, replications in expected_replications.items():
            with self.subTest(formula_mode=formula_mode):
                state = AppState()
                state.formula_mode = formula_mode
                state.apply_defaults_for_formula()
                self.assertEqual(
                    (
                        state.perovskite_supercell_x,
                        state.perovskite_supercell_y,
                        state.perovskite_supercell_z,
                    ),
                    replications,
                )
                np.testing.assert_allclose(
                    np.diag(state.generated_chemical_structure().lattice),
                    [expected_lattice[formula_mode]] * 3,
                )

    def test_high_entropy_normalizes_site_distributions_independently(self) -> None:
        state = AppState()
        state.perovskite_supercell_x = 1
        state.perovskite_supercell_y = 1
        state.perovskite_supercell_z = 1
        state.formula_mode = 4
        state.high_entropy_a_site_elements = ["La", "Sr"]
        state.high_entropy_a_site_fractions = [1.0, 3.0]
        state.high_entropy_b_site_elements = ["Fe", "Co"]
        state.high_entropy_b_site_fractions = [2.0, 2.0]
        state.high_entropy_x_site_elements = ["O", "F"]
        state.high_entropy_x_site_fractions = [9.0, 1.0]

        structure = state.generated_chemical_structure()
        params = structure.generation_parameters
        assert params is not None

        np.testing.assert_allclose(np.diag(structure.lattice), [4.0, 4.0, 4.0])
        self.assertEqual(params.formula_mode, "high_entropy")
        self.assertAlmostEqual(sum(fraction for _, fraction in params.high_entropy_a_sites), 1.0)
        self.assertAlmostEqual(sum(fraction for _, fraction in params.high_entropy_b_sites), 1.0)
        self.assertAlmostEqual(sum(fraction for _, fraction in params.high_entropy_x_sites), 1.0)
        self.assertEqual(dict(params.high_entropy_a_sites), {"La": 0.25, "Sr": 0.75})
        self.assertEqual(dict(params.high_entropy_b_sites), {"Fe": 0.5, "Co": 0.5})
        self.assertEqual(dict(params.high_entropy_x_sites), {"O": 0.9, "F": 0.1})


if __name__ == "__main__":
    unittest.main()


class BuilderDefectTests(unittest.TestCase):
    """Defects edited through the builder panel's parallel row lists."""

    def _state_with_supercell(self) -> AppState:
        state = AppState()
        apply_builder_edits(
            state, perovskite_supercell_x=2, perovskite_supercell_y=2, perovskite_supercell_z=2
        )
        return state

    def test_vacancy_row_regenerates_the_focus_with_one_fewer_atom(self) -> None:
        state = self._state_with_supercell()
        before = state.focus.atom_count
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"),
            role=DEFECT_ROLE_LABELS.index("X"),
            cell=(0, 0, 0),
            vertex=0,
        )
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(state.focus.atom_count, before - 1)

    def test_defects_enter_the_builder_change_signature(self) -> None:
        state = self._state_with_supercell()
        before = state.builder_fields_signature()
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("substitution"),
            role=DEFECT_ROLE_LABELS.index("B"),
            cell=(0, 0, 0),
            element="Zn",
        )
        self.assertNotEqual(state.builder_fields_signature(), before)

    def test_defects_survive_a_tilt_edit(self) -> None:
        state = self._state_with_supercell()
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("substitution"),
            role=DEFECT_ROLE_LABELS.index("B"),
            cell=(0, 1, 0),
            element="Zn",
        )
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("proton"),
            role=DEFECT_ROLE_LABELS.index("X"),
            cell=(0, 1, 0),
            vertex=0,
        )
        state.regenerate_focus_from_builder_if_changed()
        labels = list(state.focus.atomic_labels)
        coords = np.array(state.focus.cartesian_coords, copy=True)
        self.assertIn("Zn", state.focus.element_symbols())
        self.assertIn("H", state.focus.element_symbols())

        apply_builder_edits(
            state,
            perovskite_tilt_system=GLAZER_TILT_SYSTEMS.index("a-a-a-"),
            tilt_angle_x=10.0,
        )
        # Same composition, different geometry: the defect list was re-applied to
        # a freshly generated ideal lattice rather than frozen or duplicated.
        self.assertEqual(state.focus.atomic_labels, labels)
        self.assertFalse(np.allclose(state.focus.cartesian_coords, coords))

    def test_defect_rows_round_trip_through_generation_parameters(self) -> None:
        state = self._state_with_supercell()
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"),
            role=DEFECT_ROLE_LABELS.index("X"),
            cell=(1, 0, 1),
            vertex=2,
        )
        state.regenerate_focus_from_builder_if_changed()
        stored = list(state.focus.generation_parameters.defects)
        self.assertEqual(len(stored), 1)

        # Moving focus away and back must restore the rows from provenance.
        state.set_defect_rows([])
        state.set_defect_rows(stored)
        self.assertEqual(state.defect_row_count(), 1)
        self.assertEqual([d.signature() for d in state.builder_defects()],
                         [d.signature() for d in stored])

    def test_vacated_b_site_draws_no_alignment_edges(self) -> None:
        state = self._state_with_supercell()
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"),
            role=DEFECT_ROLE_LABELS.index("B"),
            cell=(1, 0, 1),
        )
        state.regenerate_focus_from_builder_if_changed()
        structure = state.focus
        b_grid = state.b_grid_for_structure(structure)
        self.assertIsNotNone(b_grid)
        self.assertEqual(int(np.sum(b_grid < 0)), 1)

        moments = np.zeros((structure.atom_count, 3), dtype=float)
        for site_index in b_grid.reshape(-1):
            if site_index >= 0:
                moments[int(site_index), 2] = 1.0
        edges = spin_alignment_edge_segments(
            structure.cartesian_coords, b_grid, moments
        )
        # A full 2x2x2 grid has 12 nearest-neighbour bonds; removing one site
        # removes the three it participated in.
        self.assertEqual(len(edges["aligned"]), 9)
        self.assertEqual(len(edges["anti-aligned"]), 0)

    def test_compensating_proton_button_balances_a_substitution(self) -> None:
        state = self._state_with_supercell()
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("substitution"),
            role=DEFECT_ROLE_LABELS.index("B"),
            cell=(0, 1, 0),
            element="Zn",
        )
        state.regenerate_focus_from_builder_if_changed()
        reference = state.atomic_labels_for_build(
            state.generated_perovskite(), periodic=state.treat_as_periodic
        )
        deficit, _ = compensation_hint(reference, state.focus.atomic_labels)
        self.assertEqual(deficit, -1)

        state.add_compensating_protons(-deficit)
        state.regenerate_focus_from_builder_if_changed()
        self.assertIn("H", state.focus.element_symbols())
        self.assertEqual(
            compensation_hint(reference, state.focus.atomic_labels)[0], 0
        )

    def test_incomplete_row_does_not_shift_later_rows(self) -> None:
        state = self._state_with_supercell()
        # A half-typed substitution (no element yet) is skipped by
        # builder_defects(), so rows must be resolved by index, not by position
        # in the filtered list.
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("substitution"),
            role=DEFECT_ROLE_LABELS.index("B"),
            cell=(0, 0, 0),
            element="",
        )
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"),
            role=DEFECT_ROLE_LABELS.index("X"),
            cell=(1, 0, 1),
            vertex=2,
        )
        self.assertIsNone(state.defect_for_row(0))
        self.assertEqual(len(state.builder_defects()), 1)
        second = state.defect_for_row(1)
        self.assertIsNotNone(second)
        self.assertEqual(tuple(second.site), ("X", 1, 0, 1, 2))

    def test_vacancy_is_rendered_at_the_removed_atom_position(self) -> None:
        state = self._state_with_supercell()
        before = np.array(state.focus.cartesian_coords, copy=True)
        self.assertEqual(len(vacancy_render_sites(state.focus)[0]), 0)

        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"),
            role=DEFECT_ROLE_LABELS.index("X"),
            cell=(0, 0, 0),
            vertex=0,
        )
        state.regenerate_focus_from_builder_if_changed()
        after = state.focus.cartesian_coords
        removed = [
            index
            for index, position in enumerate(before)
            if np.linalg.norm(after - position, axis=1).min() > 1e-9
        ]
        self.assertEqual(len(removed), 1)

        coords, labels = vacancy_render_sites(state.focus)
        self.assertEqual(labels, ["O"])
        np.testing.assert_allclose(coords[0], before[removed[0]], atol=1e-12)

    def test_vacancy_marker_matches_the_missing_species_radius(self) -> None:
        state = self._state_with_supercell()
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"),
            role=DEFECT_ROLE_LABELS.index("X"),
            cell=(0, 0, 0),
            vertex=0,
        )
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"),
            role=DEFECT_ROLE_LABELS.index("B"),
            cell=(1, 0, 1),
        )
        state.regenerate_focus_from_builder_if_changed()
        _, labels = vacancy_render_sites(state.focus)
        self.assertEqual(sorted(labels), ["Fe", "O"])
        structure = state.focus
        for ionic in (False, True):
            for oxidation_states in (None, np.full(structure.atom_count, -2)):
                with self.subTest(ionic=ionic, oxidation=oxidation_states is not None):
                    atom_radii = structure_atom_render_radii(
                        structure,
                        oxidation_states,
                        render_with_ionic_radius=ionic,
                    )
                    marker_radii = vacancy_render_radii(
                        labels,
                        structure,
                        atom_radii,
                        render_with_ionic_radius=ionic,
                    )
                    # Each marker is exactly the size the surviving atoms of that
                    # element are being drawn at, however radii were computed.
                    for label, radius in zip(labels, marker_radii):
                        same_element = [
                            atom_radii[index]
                            for index, element in enumerate(structure.atomic_labels)
                            if element == label
                        ]
                        self.assertTrue(same_element)
                        self.assertAlmostEqual(float(radius), float(same_element[0]))

    def test_vacancy_marker_follows_the_ideal_lattice_through_a_tilt(self) -> None:
        state = self._state_with_supercell()
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"),
            role=DEFECT_ROLE_LABELS.index("X"),
            cell=(0, 0, 0),
            vertex=0,
        )
        apply_builder_edits(
            state,
            perovskite_tilt_system=GLAZER_TILT_SYSTEMS.index("a-a-a-"),
            tilt_angle_x=12.0,
        )
        coords, _ = vacancy_render_sites(state.focus)

        # The same cell with no defect: the marker must land on its tilted O site.
        reference = self._state_with_supercell()
        apply_builder_edits(
            reference,
            perovskite_tilt_system=GLAZER_TILT_SYSTEMS.index("a-a-a-"),
            tilt_angle_x=12.0,
        )
        distances = np.linalg.norm(
            reference.focus.cartesian_coords - coords[0], axis=1
        )
        self.assertLess(float(distances.min()), 1e-12)
        self.assertEqual(
            reference.focus.atomic_labels[int(np.argmin(distances))], "O"
        )

    def test_every_periodic_image_of_a_vacancy_is_marked_in_the_render(self) -> None:
        state = self._state_with_supercell()
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"),
            role=DEFECT_ROLE_LABELS.index("A"),
            cell=(0, 0, 0),
        )
        state.regenerate_focus_from_builder_if_changed()
        # One atom leaves the periodic cell, but the finite render draws the
        # closing boundary layer, where that corner site has 8 copies.
        self.assertEqual(len(vacancy_render_sites(state.focus)[0]), 1)
        self.assertEqual(len(vacancy_render_sites(state.rendered_structure())[0]), 8)


class SupercellSemanticsTests(unittest.TestCase):
    """Supercell counts primitive cells: 1 is the primitive cell itself."""

    def test_supercell_one_is_the_primitive_cell(self) -> None:
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=1,
            perovskite_supercell_y=1,
            perovskite_supercell_z=1,
        )
        self.assertEqual(state.effective_oct_counts(), (0, 0, 0))
        self.assertEqual(state.focus.atom_count, 5)  # one ABX3 formula unit

    def test_supercell_scales_as_the_cube_of_the_count(self) -> None:
        for supercell, atoms in ((1, 5), (2, 40), (3, 135)):
            with self.subTest(supercell=supercell):
                state = AppState()
                apply_builder_edits(
                    state,
                    perovskite_supercell_x=supercell,
                    perovskite_supercell_y=supercell,
                    perovskite_supercell_z=supercell,
                )
                self.assertEqual(state.focus.atom_count, atoms)

    def test_supercell_never_drops_below_one(self) -> None:
        state = AppState()
        state.perovskite_supercell_x = 0
        state.perovskite_supercell_y = -3
        state.apply_perovskite_constraints()
        self.assertEqual(state.perovskite_supercell_x, 1)
        self.assertEqual(state.perovskite_supercell_y, 1)

    def test_defaults_open_on_a_three_cell_grid(self) -> None:
        # Comfortably above the two cells per axis the reference orderings need.
        state = AppState()
        self.assertEqual(state.effective_oct_counts(), (2, 2, 2))
        self.assertEqual(state.focus.atom_count, 135)


class DefectSiteSliderTests(unittest.TestCase):
    """Defect sites are picked by an index bounded by the sites that exist."""

    def _state(self) -> AppState:
        # Pinned to 2x2x2 rather than left on the builder default: this class is
        # about how a slider index maps to a site key, and the small cell keeps the
        # indices below enumerable by hand.
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        return state

    def test_option_counts_match_the_lattice(self) -> None:
        state = self._state()  # 2x2x2 primitive cells, periodic
        self.assertEqual(len(state.defect_site_options("A")), 8)
        self.assertEqual(len(state.defect_site_options("B")), 8)
        self.assertEqual(len(state.defect_site_options("X")), 24)

    def test_option_count_follows_the_supercell(self) -> None:
        state = self._state()
        apply_builder_edits(
            state,
            perovskite_supercell_x=3,
            perovskite_supercell_y=3,
            perovskite_supercell_z=3,
        )
        self.assertEqual(len(state.defect_site_options("B")), 27)
        self.assertEqual(len(state.defect_site_options("X")), 81)

    def test_site_key_round_trips_through_the_slider_index(self) -> None:
        state = self._state()
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"),
            role=DEFECT_ROLE_LABELS.index("X"),
        )
        options = state.defect_site_options("X")
        for index in (0, 7, len(options) - 1):
            with self.subTest(index=index):
                state.set_defect_site_key(0, options[index])
                self.assertEqual(state.defect_site_key(0), options[index])
                self.assertEqual(options.index(state.defect_site_key(0)), index)

    def test_slider_index_addresses_the_intended_atom(self) -> None:
        state = self._state()
        options = state.defect_site_options("X")
        state.add_defect_row(kind=DEFECT_KIND_KEYS.index("vacancy"), role=DEFECT_ROLE_LABELS.index("X"))
        before = state.focus.atom_count
        for index in (0, 11, 23):
            with self.subTest(index=index):
                state.set_defect_site_key(0, options[index])
                state.regenerate_focus_from_builder_if_changed()
                # Exactly one oxygen leaves, whichever site the slider names.
                self.assertEqual(state.focus.atom_count, before - 1)
                self.assertEqual(state.focus.element_symbols().count("O"), 23)

    def test_a_proton_row_is_forced_onto_an_x_site(self) -> None:
        state = self._state()
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("proton"),
            role=DEFECT_ROLE_LABELS.index("B"),  # nonsensical; must be corrected
        )
        self.assertEqual(state.defect_role(0), "X")
        self.assertEqual(state.defect_site_key(0).role, "X")

    def test_shrinking_the_cell_leaves_an_out_of_range_row_untouched(self) -> None:
        state = self._state()
        apply_builder_edits(
            state,
            perovskite_supercell_x=3,
            perovskite_supercell_y=3,
            perovskite_supercell_z=3,
        )
        far_site = state.defect_site_options("B")[-1]  # B(2,2,2)
        state.add_defect_row(
            kind=DEFECT_KIND_KEYS.index("vacancy"), role=DEFECT_ROLE_LABELS.index("B")
        )
        state.set_defect_site_key(0, far_site)
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        # Not in the smaller cell, so it is skipped -- but the row is preserved.
        self.assertNotIn(far_site, state.defect_site_options("B"))
        self.assertEqual(state.defect_site_key(0), far_site)
        self.assertEqual(state.focus.atom_count, 40)


class SiteSelectionTests(unittest.TestCase):
    """The per-site oxidation/moment list and its 3D highlight."""

    def _solved(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.run_magnetic_structure_calculation(structure=state.focus)
        return state

    def test_rows_cover_every_atom(self) -> None:
        state = self._solved()
        assignment = state.magnetic_oxidation_assignments[0]
        structure = state.magnetic_analysis_structure
        rows = oxidation_site_rows(structure, assignment)
        self.assertEqual(len(rows), structure.atom_count)
        for element in ("La", "Fe", "O"):
            self.assertTrue(any(f" {element:<2}  " in row for row in rows))

    def test_nothing_is_selected_by_default(self) -> None:
        self.assertEqual(AppState().selected_site_index, -1)

    def test_highlight_matches_the_same_structure_by_index(self) -> None:
        state = self._solved()
        structure = state.focus
        self.assertEqual(highlighted_render_indices(structure, structure, 4), [4])

    def test_highlight_finds_every_periodic_image_in_the_render(self) -> None:
        state = self._solved()
        focus = state.focus
        rendered = state.rendered_structure()
        self.assertGreater(rendered.atom_count, focus.atom_count)
        # A corner A site is imaged onto all eight corners of the finite render.
        corner = highlighted_render_indices(rendered, focus, 0)
        self.assertEqual(len(corner), 8)
        for index in corner:
            self.assertEqual(rendered.atomic_labels[index], focus.atomic_labels[0])

    def test_highlight_ignores_an_out_of_range_selection(self) -> None:
        state = self._solved()
        self.assertEqual(
            highlighted_render_indices(state.rendered_structure(), state.focus, 10_000), []
        )
        self.assertEqual(highlighted_render_indices(state.focus, state.focus, -1), [])
