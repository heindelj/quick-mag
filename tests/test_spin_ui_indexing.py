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
from quick_mag.quick_mag_ui import (  # noqa: E402
    STRUCTURE_ZOOM_RANGE,
    AppState,
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
    """
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
        apply_builder_edits(state, b_site_element="Mn", perovskite_rep_x=2)
        self.assertEqual(state.b_site_element, "Mn")

        state.create_new_structure()
        self.assertEqual(len(state.structures), 2)
        self.assertIs(state.focus, state.structures[1])
        self.assertEqual(state.b_site_element, "Fe")
        self.assertEqual(state.perovskite_rep_x, 1)
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


CANONICAL_REFERENCE_NAMES = {"G", "C(a)", "C(b)", "C(c)", "F", "A(a)", "A(b)", "A(c)"}


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
        self.assertEqual(
            {name for name, _ in state.reference_configs}, CANONICAL_REFERENCE_NAMES
        )
        # Plotted with no solve, and every point is a labelled reference.
        self.assertEqual(len(state.displayed_spin_configs()), 8)
        self.assertEqual(set(state.spin_classification_labels()), CANONICAL_REFERENCE_NAMES)

    def test_reference_energies_are_single_point_evaluations(self) -> None:
        state = AppState()
        for config in state.displayed_spin_configs():
            self.assertAlmostEqual(
                config.energy,
                compute_config_energy(state.magnetic_j_matrix, config.all_moments),
                places=12,
            )

    def test_labels_are_exact_matches_not_nearest_neighbours(self) -> None:
        state = AppState()
        reference = state.displayed_spin_configs()[0]
        self.assertNotEqual(state.label_for_config(reference), "Other")

        # One flipped spin is no longer that ordering, however close it looks.
        perturbed_moments = np.array(reference.all_moments, dtype=float, copy=True)
        perturbed_moments[0] *= -1.0
        perturbed = replace(reference, all_moments=perturbed_moments)
        self.assertEqual(state.label_for_config(perturbed), "Other")

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
        apply_builder_edits(state, perovskite_rep_x=2)

        self.assertNotEqual(len(state.magnetic_site_indices), n_mag_before)
        # Freshly seeded for the new cell; nothing carried over at the old length.
        self.assertEqual(
            {name for name, _ in state.reference_configs}, CANONICAL_REFERENCE_NAMES
        )
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
        state.merge_solver_states_into_landscape(solved)
        self.assertGreater(len(state.spin_landscape), 8)

        # The new structure has the same magnetic-site count, so a length check alone
        # would happily re-energize and present the old structure's configurations.
        state.create_new_structure()
        state.sync_active_structure()
        self.assertEqual(len(state.magnetic_site_indices), n_mag)
        self.assertEqual(len(state.spin_landscape), 8)
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
        self.assertEqual(set(labels) - {"Other"}, CANONICAL_REFERENCE_NAMES)

        # A cap below the reference count still keeps every reference.
        state.spin_plot_max_configs = 2
        state.refresh_landscape_energies()
        self.assertEqual(
            set(state.spin_classification_labels()), CANONICAL_REFERENCE_NAMES
        )


class DegenerateConfigTests(unittest.TestCase):
    def test_collapsing_keeps_one_config_per_energy_plus_every_reference(self) -> None:
        state = AppState()
        state.spin_landscape = list(state.spin_landscape) + random_configs(state, 40)

        state.plot_degenerate_configs = False
        state.refresh_landscape_energies()
        shown = state.displayed_spin_configs()
        labels = state.spin_classification_labels()

        # One point per distinct energy, except where references share one: on a cubic
        # cell C(a)/C(b)/C(c) are degenerate and must all still be shown.
        references = [c for c, l in zip(shown, labels) if l != "Other"]
        others = [c for c, l in zip(shown, labels) if l == "Other"]
        self.assertEqual(len(references), 8)
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
        # On a cubic cell the three C orientations are degenerate with each other.
        for name in ("C(a)", "C(b)", "C(c)"):
            self.assertEqual(by_label[name].degeneracy, 3)
        self.assertEqual(by_label["G"].degeneracy, 1)

    def test_collapsing_reaches_further_up_the_landscape(self) -> None:
        state = AppState()
        state.spin_landscape = list(state.spin_landscape) + random_configs(state, 200)
        state.spin_plot_max_configs = 12

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
            build,
            ferromagnetic,
        )
        g_type_edges = spin_alignment_edge_segments(
            structure.cartesian_coords,
            build,
            g_type,
        )

        self.assertEqual(len(ferromagnetic_edges["aligned"]), 12)
        self.assertEqual(len(ferromagnetic_edges["anti-aligned"]), 0)
        self.assertEqual(len(g_type_edges["aligned"]), 0)
        self.assertEqual(len(g_type_edges["anti-aligned"]), 12)

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
            {"G", "C(a)", "C(b)", "C(c)", "F", "A(a)", "A(b)", "A(c)"},
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
        state.perovskite_rep_x = 0
        state.perovskite_rep_y = 0
        state.perovskite_rep_z = 0
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
        state.perovskite_rep_x = 0
        state.perovskite_rep_y = 0
        state.perovskite_rep_z = 0
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
        state.perovskite_rep_x = 0
        state.perovskite_rep_y = 0
        state.perovskite_rep_z = 0
        state.formula_mode = 3
        state.apply_defaults_for_formula()

        structure = state.generated_chemical_structure()
        build = state.generated_perovskite()
        params = structure.generation_parameters
        assert params is not None

        self.assertEqual(build.octahedra.shape, (2, 2, 2))
        np.testing.assert_allclose(np.diag(structure.lattice), [8.0, 8.0, 8.0])
        self.assertEqual(params.formula_mode, "dq")
        self.assertEqual(params.a_site_element, "Ca")
        self.assertEqual(params.a2_site_element, "Cu")
        self.assertEqual(params.b_site_element, "Fe")
        self.assertEqual(params.b2_site_element, "Re")
        self.assertEqual(params.x_site_element, "O")

        a_labels = [structure.atomic_labels[index] for index in build.a_site_indices]
        b_labels = [structure.atomic_labels[index] for index in build.b_site_indices]
        x_labels = [structure.atomic_labels[index] for index in build.x_site_indices]
        self.assertEqual(a_labels.count("Ca"), 2)
        self.assertEqual(a_labels.count("Cu"), 6)
        self.assertEqual(b_labels.count("Fe"), 4)
        self.assertEqual(b_labels.count("Re"), 4)
        self.assertEqual(set(x_labels), {"O"})

    def test_formula_defaults_keep_initial_cell_sizes_consistent(self) -> None:
        expected_replications = {
            0: (1, 1, 1),
            1: (0, 0, 0),
            2: (0, 0, 0),
            3: (0, 0, 0),
            4: (1, 1, 1),
        }

        for formula_mode, replications in expected_replications.items():
            with self.subTest(formula_mode=formula_mode):
                state = AppState()
                state.formula_mode = formula_mode
                state.apply_defaults_for_formula()
                self.assertEqual(
                    (
                        state.perovskite_rep_x,
                        state.perovskite_rep_y,
                        state.perovskite_rep_z,
                    ),
                    replications,
                )
                np.testing.assert_allclose(
                    np.diag(state.generated_chemical_structure().lattice),
                    [8.0, 8.0, 8.0],
                )

    def test_high_entropy_normalizes_site_distributions_independently(self) -> None:
        state = AppState()
        state.perovskite_rep_x = 0
        state.perovskite_rep_y = 0
        state.perovskite_rep_z = 0
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
