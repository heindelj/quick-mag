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
from imgui_bundle import hello_imgui, imgui, immapp, implot3d  # noqa: E402
from quick_mag.defects import (  # noqa: E402
    compensation_hint,
    resolve_key_to_indices,
)
from quick_mag.defect_planes import (  # noqa: E402
    occupied_planes,
    plane_index_of_key,
    plane_miller_in_cell,
    plane_period,
    plane_role_label,
    sites_in_plane,
)
from quick_mag.defects import SiteDefect  # noqa: E402
from quick_mag.perovskite_builder import SiteKey  # noqa: E402
from quick_mag.quick_mag_ui import (  # noqa: E402
    DEFECT_KIND_KEYS,
    FORMULA_MODE_KEYS,
    MAX_LISTED_OXIDATION_ASSIGNMENTS,
    MAX_MATCH_DEFECT_CONCENTRATION,
    SPHERE_LATITUDE_SEGMENTS,
    SPHERE_LONGITUDE_SEGMENTS,
    PlaneFocus,
    GLAZER_TILT_SYSTEMS,
    DefectPlaneGroup,
    PlaneDefectSite,
    STRUCTURE_ZOOM_RANGE,
    AppState,
    build_sphere_mesh,
    builder_summary_rows,
    candidate_pixels,
    element_box_note,
    selected_sites_tree_label,
    nearest_picked_atom,
    view_space_depth,
    DEFAULT_STRUCTURE_ROTATION,
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


def render_frames(gui, frames: int = 3) -> None:
    """Run ``gui`` inside real ImGui/ImPlot3D frames, with nothing on screen.

    Some things -- projecting a point to a pixel, above all -- only mean anything
    inside a live plot. hello_imgui's null backend gives a genuine frame without
    a window; it just never uploads the font atlas, hence the backend flag.
    """
    counter = {"i": 0}

    def frame() -> None:
        counter["i"] += 1
        imgui.begin("##probe")
        gui()
        imgui.end()
        if counter["i"] >= frames:
            hello_imgui.get_runner_params().app_shall_exit = True

    params = hello_imgui.RunnerParams()
    params.callbacks.show_gui = frame
    params.callbacks.post_init = lambda: setattr(
        imgui.get_io(),
        "backend_flags",
        imgui.get_io().backend_flags | imgui.BackendFlags_.renderer_has_textures,
    )
    params.platform_backend_type = hello_imgui.PlatformBackendType.null
    params.renderer_backend_type = hello_imgui.RendererBackendType.null
    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.no_default_window
    )
    immapp.run(params, immapp.AddOnsParams(with_implot=True, with_implot3d=True))


def add_defect(
    state: AppState,
    kind: str,
    key: SiteKey,
    *,
    element: str = "",
    orientation: int = 0,
) -> DefectPlaneGroup:
    """Add one defect through the plane panel, on a plane that contains it.

    The panel names a site by picking a plane and then checking the site in it,
    so a test that wants one specific site has to do the same. (001) always works
    -- every site lies on exactly one plane of every family.
    """
    # A vacancy has no kind of its own in the panel: it is a substitution with
    # nothing in the element box.
    if kind == "vacancy":
        kind, element = "substitution", ""
    grid_shape = state.defect_grid_shape()
    miller = (0, 0, 1)
    group = state.add_defect_group(
        kind=DEFECT_KIND_KEYS.index(kind),
        miller=miller,
        plane=plane_index_of_key(
            key,
            miller,
            period=plane_period(grid_shape, state.treat_as_periodic, miller),
        ),
    )
    # Only the defect, not the plane mode: an open Defects panel suppresses the
    # periodic images and dims everything off the plane, which these tests do
    # not want.
    state.defect_panel_open = False
    state.ensure_defect_groups()
    group.sites.append(
        PlaneDefectSite(site=key, element=element, orientation=orientation)
    )
    return group


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

    def test_hovered_coordinate_readout_is_off(self) -> None:
        # It projects the cursor onto the view box's own back faces, so it reports a
        # corner of empty space rather than anything the user is pointing at.
        flags = structure_plot_flags(show_legend=True)
        self.assertTrue(flags & implot3d.Flags_.no_mouse_text.value)

    def test_rotation_is_handled_by_the_view_rather_than_implot3d(self) -> None:
        # ImPlot3D's rotation is a turntable pinned to the plot's c axis, which loses a
        # degree of freedom in exactly the poses the a/b/c buttons aim at. The view runs
        # its own trackball instead (see ``rotation_after_drag``), so ImPlot3D's is off.
        flags = structure_plot_flags(show_legend=True)
        self.assertTrue(flags & implot3d.Flags_.no_rotate.value)
        # Inputs as a whole stay on: hover, the legend and the context menu still work.
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
        add_defect(state, "vacancy", SiteKey("X", 0, 0, 0, 0))
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(state.focus.atom_count, before - 1)

    def test_defects_enter_the_builder_change_signature(self) -> None:
        state = self._state_with_supercell()
        before = state.builder_fields_signature()
        add_defect(state, "substitution", SiteKey("B", 0, 0, 0), element="Zn")
        self.assertNotEqual(state.builder_fields_signature(), before)

    def test_defects_survive_a_tilt_edit(self) -> None:
        state = self._state_with_supercell()
        add_defect(state, "substitution", SiteKey("B", 0, 1, 0), element="Zn")
        add_defect(state, "proton", SiteKey("X", 0, 1, 0, 0))
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
        add_defect(state, "vacancy", SiteKey("X", 1, 0, 1, 2))
        state.regenerate_focus_from_builder_if_changed()
        stored = list(state.focus.generation_parameters.defects)
        self.assertEqual(len(stored), 1)

        # Moving focus away and back must restore the rows from provenance.
        state.set_defect_rows([])
        state.set_defect_rows(stored)
        self.assertEqual(state.defect_group_count(), 1)
        self.assertEqual([d.signature() for d in state.builder_defects()],
                         [d.signature() for d in stored])

    def test_vacated_b_site_draws_no_alignment_edges(self) -> None:
        state = self._state_with_supercell()
        add_defect(state, "vacancy", SiteKey("B", 1, 0, 1))
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
        add_defect(state, "substitution", SiteKey("B", 0, 1, 0), element="Zn")
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

    def test_a_substitution_without_an_element_is_built_as_a_vacancy(self) -> None:
        state = self._state_with_supercell()
        # Picking is a click, so naming a site before naming what goes on it is
        # the normal case. The site is held as a vacancy meanwhile -- an atom
        # that vanishes is the clearest sign the click landed -- and the panel
        # says in yellow that the element is still missing.
        first = add_defect(state, "substitution", SiteKey("B", 0, 0, 0), element="")
        second_group = add_defect(state, "vacancy", SiteKey("X", 1, 0, 1, 2))
        pending = state.defect_for_site(first, first.sites[0])
        self.assertIsNotNone(pending)
        self.assertEqual(pending.kind, "vacancy")
        self.assertEqual(tuple(pending.site), ("B", 0, 0, 0, 0))
        self.assertEqual(state.pending_substitutions(first), 1)
        self.assertEqual(len(state.builder_defects()), 2)

        second = state.defect_for_site(second_group, second_group.sites[0])
        self.assertEqual(tuple(second.site), ("X", 1, 0, 1, 2))

        # Naming the element turns it back into the substitution it always was.
        first.sites[0].element = "Zn"
        self.assertEqual(state.defect_for_site(first, first.sites[0]).kind, "substitution")
        self.assertEqual(state.pending_substitutions(first), 0)
        state.regenerate_focus_from_builder_if_changed()
        self.assertIn("Zn", state.focus.element_symbols())

    def test_vacancy_is_rendered_at_the_removed_atom_position(self) -> None:
        state = self._state_with_supercell()
        before = np.array(state.focus.cartesian_coords, copy=True)
        self.assertEqual(len(vacancy_render_sites(state.focus)[0]), 0)

        add_defect(state, "vacancy", SiteKey("X", 0, 0, 0, 0))
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
        add_defect(state, "vacancy", SiteKey("X", 0, 0, 0, 0))
        add_defect(state, "vacancy", SiteKey("B", 1, 0, 1))
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
        add_defect(state, "vacancy", SiteKey("X", 0, 0, 0, 0))
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
        add_defect(state, "vacancy", SiteKey("A", 0, 0, 0))
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


class DefectPlaneSelectionTests(unittest.TestCase):
    """Defect sites are picked out of a lattice plane, not off a flat ordinal."""

    def _state(self) -> AppState:
        # Pinned to 2x2x2 rather than left on the builder default: this class is
        # about which sites a plane holds, and the small cell keeps them
        # enumerable by hand.
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        state.defect_panel_open = True
        return state

    def _group(self, state: AppState, miller, plane: int) -> DefectPlaneGroup:
        return state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=miller, plane=plane
        )

    def test_001_planes_alternate_between_the_ao_and_bo2_layers(self) -> None:
        state = self._state()
        grid_shape, periodic = state.defect_grid_shape(), state.treat_as_periodic
        planes = occupied_planes(grid_shape, periodic, (0, 0, 1))
        self.assertEqual(planes, [0, 1, 2, 3])
        # The half-cube step is the whole point: a whole-cell step would only
        # ever reach one sublattice.
        self.assertEqual(
            [plane_role_label(grid_shape, periodic, (0, 0, 1), m) for m in planes],
            ["A + X", "B + X", "A + X", "B + X"],
        )

    def test_a_plane_family_partitions_every_site_exactly_once(self) -> None:
        state = self._state()
        grid_shape, periodic = state.defect_grid_shape(), state.treat_as_periodic
        for miller in ((0, 0, 1), (1, 1, 0), (1, 1, 1), (1, -1, 0)):
            with self.subTest(miller=miller):
                seen: list = []
                for plane in occupied_planes(grid_shape, periodic, miller):
                    seen.extend(sites_in_plane(grid_shape, periodic, miller, plane))
                self.assertEqual(len(seen), len(set(seen)))
                self.assertEqual(len(seen), 40)  # 8 A + 8 B + 24 X

    def test_the_site_list_is_bounded_by_the_lattice(self) -> None:
        state = self._state()
        group = self._group(state, (0, 0, 1), 1)
        # A BO2 layer of a 2x2x2 cell: 4 B sites and the 8 equatorial oxygens.
        # A BO2 layer of a 2x2x2 cell: 4 B sites and the 8 equatorial oxygens.
        # A vacancy or a substitution may go on any of them -- the plane is not
        # narrowed by role, so a click can land on either sublattice.
        self.assertEqual(len(state.plane_site_options(group)), 12)
        self.assertEqual(group.pick_role(), "")

    def test_the_plane_reaches_a_sites_and_x_sites_not_just_b_sites(self) -> None:
        state = self._state()
        grid_shape, periodic = state.defect_grid_shape(), state.treat_as_periodic
        roles = {
            key.role
            for plane in occupied_planes(grid_shape, periodic, (0, 0, 1))
            for key in sites_in_plane(grid_shape, periodic, (0, 0, 1), plane)
        }
        self.assertEqual(roles, {"A", "B", "X"})

    def test_checking_two_sites_in_one_plane_makes_two_defects(self) -> None:
        state = self._state()
        group = self._group(state, (0, 0, 1), 1)
        options = state.plane_site_options(group)
        before = state.focus.atom_count
        for key in options[:2]:
            state.toggle_plane_site(group, key)
        self.assertEqual(len(state.builder_defects()), 2)
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(state.focus.atom_count, before - 2)

    def test_unpicking_a_site_removes_only_that_defect(self) -> None:
        state = self._state()
        group = self._group(state, (0, 0, 1), 1)
        options = state.plane_site_options(group)
        for key in options[:3]:
            state.toggle_plane_site(group, key)
        self.assertEqual(len(state.builder_defects()), 3)
        state.toggle_plane_site(group, options[1])
        self.assertEqual(
            [tuple(defect.site) for defect in state.builder_defects()],
            [tuple(options[0]), tuple(options[2])],
        )

    def test_picks_made_on_another_plane_are_kept_and_reported(self) -> None:
        state = self._state()
        group = self._group(state, (0, 0, 1), 1)
        first_plane = state.plane_site_options(group)[:2]
        for key in first_plane:
            state.toggle_plane_site(group, key)

        group.plane = 3
        second_plane = state.plane_site_options(group)[:3]
        for key in second_plane:
            state.toggle_plane_site(group, key)
        self.assertEqual(len(group.sites), len(first_plane) + len(second_plane))
        # The earlier plane's picks are untouched, and reported as off-plane so
        # they can still be found and removed from the panel.
        self.assertEqual(
            [entry.site for entry in state.sites_off_plane(group)], list(first_plane)
        )

    def test_moving_the_plane_slider_keeps_the_sites_already_checked(self) -> None:
        state = self._state()
        group = self._group(state, (0, 0, 1), 1)
        key = state.plane_site_options(group)[0]
        state.toggle_plane_site(group, key)

        group.plane = 3
        # The checked site is a defect, not a highlight: sliding past it must not
        # silently drop it, only report it as living on another plane.
        self.assertEqual(len(state.builder_defects()), 1)
        self.assertEqual([entry.site for entry in state.sites_off_plane(group)], [key])

    def test_a_named_site_removes_the_atom_it_names(self) -> None:
        state = self._state()
        group = self._group(state, (0, 0, 1), 1)
        options = [
            key for key in state.plane_site_options(group) if key.role == "X"
        ]
        before = state.focus.atom_count
        for key in (options[0], options[len(options) // 2], options[-1]):
            with self.subTest(site=key):
                group.sites = [PlaneDefectSite(site=key)]
                state.regenerate_focus_from_builder_if_changed()
                # Exactly one oxygen leaves, whichever site was checked.
                self.assertEqual(state.focus.atom_count, before - 1)
                self.assertEqual(state.focus.element_symbols().count("O"), 23)

    def test_a_proton_group_is_forced_onto_the_x_sites(self) -> None:
        state = self._state()
        group = state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("proton"), miller=(0, 0, 1), plane=1
        )
        self.assertEqual(group.pick_role(), "X")
        self.assertTrue(
            all(key.role == "X" for key in state.plane_site_options(group))
        )

    def test_shrinking_the_cell_leaves_an_out_of_range_group_untouched(self) -> None:
        state = self._state()
        apply_builder_edits(
            state,
            perovskite_supercell_x=3,
            perovskite_supercell_y=3,
            perovskite_supercell_z=3,
        )
        group = self._group(state, (0, 0, 1), 5)
        far_site = SiteKey("B", 2, 2, 2)
        self.assertIn(far_site, state.plane_site_options(group))
        state.toggle_plane_site(group, far_site)

        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        # Neither the plane nor the site exists in the smaller cell, so the group
        # is skipped -- but nothing about it is rewritten.
        self.assertNotIn(5, state.plane_options(group))
        self.assertEqual(group.plane, 5)
        self.assertEqual([entry.site for entry in group.sites], [far_site])
        self.assertEqual(state.focus.atom_count, 40)

    def test_stored_defects_come_back_grouped_by_plane(self) -> None:
        state = self._state()
        group = self._group(state, (0, 0, 1), 1)
        for key in state.plane_site_options(group)[:3]:
            state.toggle_plane_site(group, key)
        state.regenerate_focus_from_builder_if_changed()
        stored = list(state.focus.generation_parameters.defects)
        self.assertEqual(len(stored), 3)

        state.set_defect_rows(stored)
        # One layer's worth of vacancies comes back as one group, not three.
        self.assertEqual(state.defect_group_count(), 1)
        self.assertEqual(
            [d.signature() for d in state.builder_defects()],
            [d.signature() for d in stored],
        )

    def test_moving_focus_between_planes_does_not_rebuild_the_structure(self) -> None:
        state = self._state()
        group = self._group(state, (0, 0, 1), 1)
        state.toggle_plane_site(group, state.plane_site_options(group)[0])
        self._group(state, (1, 1, 1), 0)
        state.regenerate_focus_from_builder_if_changed()
        signature = state.builder_fields_signature()
        # Which plane is on screen is view state, so moving focus between entries
        # must not look like an edit to the structure.
        state.set_active_defect_group(0)
        self.assertEqual(state.builder_fields_signature(), signature)
        state.set_active_defect_group(1)
        self.assertEqual(state.builder_fields_signature(), signature)


class PlaneFocusTests(unittest.TestCase):
    """Picking into one plane: which atoms are in it, and which one a click hits."""

    def _state(self) -> AppState:
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        # The panel being open is what puts the view into plane mode; the real
        # UI sets this from the collapsing header every frame.
        state.defect_panel_open = True
        return state

    def test_adding_a_group_arms_it_for_picking(self) -> None:
        state = self._state()
        self.assertEqual(state.active_defect_group, -1)
        state.add_defect_group(kind=DEFECT_KIND_KEYS.index("substitution"))
        self.assertEqual(state.active_defect_group, 0)
        self.assertIsNotNone(state.active_group())

    def test_only_one_plane_is_shown_at_a_time(self) -> None:
        state = self._state()
        first = state.add_defect_group(kind=DEFECT_KIND_KEYS.index("substitution"))
        second = state.add_defect_group(kind=DEFECT_KIND_KEYS.index("substitution"))
        # A new entry takes focus, so you can add one and start clicking.
        self.assertIs(state.active_group(), second)
        state.set_active_defect_group(0)
        self.assertIs(state.active_group(), first)
        self.assertEqual(
            sum(state.active_group() is group for group in state.defect_groups), 1
        )

    def test_collapsing_the_panel_gives_the_plain_structure_back(self) -> None:
        state = self._state()
        state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=1
        )
        self.assertIsNotNone(state.active_group())
        # The planes have no Draw switch, so closing the panel is the way out of
        # plane mode -- and it has to take the fading and the picking with it.
        # Draw once first: the overlays are memoized, and collapsing has to
        # invalidate that rather than leave the last sheets cached on screen.
        self.assertTrue(state.defect_plane_overlays())
        state.defect_panel_open = False
        self.assertIsNone(state.active_group())
        self.assertFalse(state.plane_render_sites(state.rendered_structure()))
        self.assertEqual(state.defect_plane_overlays(), [])
        self.assertTrue(state.effective_render_periodic_images())
        state.defect_panel_open = True
        self.assertTrue(state.defect_plane_overlays())

    def test_removing_a_group_does_not_leave_picking_dangling(self) -> None:
        state = self._state()
        state.add_defect_group(kind=DEFECT_KIND_KEYS.index("substitution"))
        keeper = state.add_defect_group(kind=DEFECT_KIND_KEYS.index("substitution"))
        state.set_active_defect_group(1)
        state.remove_defect_group(0)
        # Picking follows the group, rather than staying on the index that the
        # next group slid into.
        self.assertIs(state.active_group(), keeper)
        state.remove_defect_group(0)
        self.assertIsNone(state.active_group())

    def test_the_focused_atoms_are_exactly_the_plane_members(self) -> None:
        state = self._state()
        group = state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=1
        )
        structure = state.rendered_structure()
        focused = state.plane_render_sites(structure)
        self.assertTrue(focused)
        expected = set(state.plane_site_options(group))
        self.assertEqual(set(focused.pick_keys()), expected)
        # Every focused atom really carries the role its key claims. The roles
        # have to come from the structure the indices are into -- the view draws
        # a finite rebuild, which is a different structure from the focus.
        roles = list(structure.generation_parameters.site_roles)
        for index, key in focused.atoms.items():
            with self.subTest(site=key):
                self.assertEqual(roles[index], key.role)

    def test_the_periodic_images_are_drawn_while_picking(self) -> None:
        state = self._state()
        state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=0
        )
        # Picking no longer takes the boundary layer away. It is still true that
        # a corner site is drawn several times over; what makes that readable is
        # ringing every copy together, not hiding them.
        self.assertTrue(state.effective_render_periodic_images())
        rendered = state.rendered_structure()
        self.assertGreater(rendered.atom_count, state.focus.atom_count)

    def test_every_copy_of_a_site_is_a_pick_target(self) -> None:
        state = self._state()
        group = state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=0
        )
        focus = state.plane_render_sites(state.rendered_structure())
        corner = SiteKey("A", 0, 0, 0)
        self.assertIn(corner, state.plane_site_options(group))
        copies = [key for key in focus.pick_keys() if key == corner]
        # The closing boundary layer gives a corner A site eight copies, and all
        # of them have to answer to a click and ring together.
        self.assertEqual(len(copies), 8)

    def test_a_vacated_site_stays_pickable_as_a_ghost(self) -> None:
        state = self._state()
        group = state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=1
        )
        key = [k for k in state.plane_site_options(group) if k.role == "X"][0]
        before = state.plane_render_sites(state.rendered_structure())
        self.assertIn(key, before.pick_keys())
        self.assertEqual(len(before.ghost_keys), 0)

        state.toggle_plane_site(group, key)
        state.regenerate_focus_from_builder_if_changed()
        after = state.plane_render_sites(state.rendered_structure())
        # The atom is gone, but its marker still answers to a click -- otherwise
        # a vacancy could be picked and never unpicked.
        self.assertNotIn(key, set(after.atoms.values()))
        self.assertIn(key, after.ghost_keys)
        self.assertIn(key, after.pick_keys())
        self.assertEqual(len(after.pick_keys()), len(before.pick_keys()))

        state.toggle_plane_site(group, key)
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(state.builder_defects(), [])

    def test_nothing_is_focused_without_a_defect_plane(self) -> None:
        state = self._state()
        self.assertIsNone(state.active_group())
        self.assertFalse(state.plane_render_sites(state.rendered_structure()))
        self.assertTrue(state.effective_render_periodic_images())

    def test_a_plane_outside_the_supercell_focuses_nothing(self) -> None:
        state = self._state()
        state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=97
        )
        self.assertFalse(state.plane_render_sites(state.rendered_structure()))

    def test_the_selected_sites_node_keeps_its_id_as_its_text_changes(self) -> None:
        """Typing an element must not rename the box being typed into.

        A tree node pushes its id onto the stack for its whole subtree, and ImGui
        derives that id from the label unless a ``###`` marker says otherwise.
        When the label carried a count, picking a site -- or typing the first
        character of an element, which changed a "needs an element" tally --
        renamed every widget under it and the keyboard focus went with them.
        """
        outcome: dict = {}

        def gui() -> None:
            outcome["ids"] = {
                imgui.get_id(selected_sites_tree_label(count)) for count in (0, 1, 7)
            }
            outcome["labels"] = {
                selected_sites_tree_label(count) for count in (0, 1, 7)
            }

        render_frames(gui)
        # Three different visible labels, one stable id.
        self.assertEqual(len(outcome["labels"]), 3)
        self.assertEqual(len(outcome["ids"]), 1)

    def test_a_click_on_an_atom_picks_that_atom(self) -> None:
        pixels = np.array([[100.0, 100.0], [140.0, 100.0], [180.0, 100.0]])
        depths = np.array([0.0, 0.0, 0.0])
        candidates = [7, 9, 11]
        for position, atom in enumerate(candidates):
            with self.subTest(atom=atom):
                aim = (pixels[position][0], pixels[position][1])
                self.assertEqual(
                    nearest_picked_atom(pixels, candidates, depths, aim), atom
                )

    def test_a_click_on_empty_space_picks_nothing(self) -> None:
        pixels = np.array([[100.0, 100.0]])
        self.assertEqual(
            nearest_picked_atom(pixels, [3], np.array([0.0]), (400.0, 400.0)), -1
        )

    def test_the_nearer_atom_wins_when_two_overlap(self) -> None:
        # Viewed edge-on, a plane's own atoms line up almost exactly. Larger
        # depth is nearer the viewer -- verified against which sphere ImPlot3D
        # actually draws in front.
        pixels = np.array([[100.0, 100.0], [101.0, 100.0]])
        candidates = [4, 5]
        self.assertEqual(
            nearest_picked_atom(pixels, candidates, np.array([-0.5, 0.5]), (100.0, 100.0)),
            5,
        )
        self.assertEqual(
            nearest_picked_atom(pixels, candidates, np.array([0.5, -0.5]), (100.0, 100.0)),
            4,
        )

    def test_depth_does_not_override_a_clear_aim(self) -> None:
        # A far atom directly under the cursor beats a nearer one that is merely
        # close by, or the frontmost atom would swallow clicks aimed past it.
        pixels = np.array([[100.0, 100.0], [113.0, 100.0]])
        self.assertEqual(
            nearest_picked_atom(
                pixels, [4, 5], np.array([-1.0, 1.0]), (100.0, 100.0)
            ),
            4,
        )

    def test_aiming_at_a_focused_atom_in_a_live_plot_picks_it(self) -> None:
        """End to end through ImPlot3D's own projection, not a stand-in for it."""
        state = self._state()
        state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=1
        )
        structure = state.rendered_structure()
        focused = state.plane_render_sites(structure)
        self.assertTrue(focused)
        coords = focused.pick_coords(structure.cartesian_coords)
        candidates = list(range(len(focused.pick_keys())))
        limits = compute_plot_box_limits(structure.cartesian_coords)
        depths = view_space_depth(coords, limits, state.structure_rotation)
        outcome: dict = {}

        def gui() -> None:
            if implot3d.begin_plot("##pick_probe", imgui.ImVec2(600.0, 600.0)):
                implot3d.setup_axes_limits(*limits, implot3d.Cond_.always)
                implot3d.setup_box_rotation(
                    implot3d.Quat(*state.structure_rotation),
                    False,
                    implot3d.Cond_.always,
                )
                pixels = candidate_pixels(coords, candidates)
                outcome["hits"] = sum(
                    nearest_picked_atom(
                        pixels, candidates, depths, tuple(pixels[position])
                    )
                    == atom
                    for position, atom in enumerate(candidates)
                )
                outcome["miss"] = nearest_picked_atom(
                    pixels, candidates, depths, (-500.0, -500.0)
                )
                implot3d.end_plot()

        render_frames(gui)
        self.assertEqual(outcome.get("hits"), len(candidates))
        self.assertEqual(outcome.get("miss"), -1)

    def test_depth_orders_points_along_the_view_direction(self) -> None:
        limits = (-4.0, 4.0, -4.0, 4.0, -4.0, 4.0)
        rotation = DEFAULT_STRUCTURE_ROTATION
        quaternion = implot3d.Quat(*rotation)
        towards = np.array(
            [
                (quaternion * implot3d.Point(*[float(v) for v in axis])).z
                for axis in np.eye(3)
            ]
        )
        towards = towards / np.linalg.norm(towards)
        depths = view_space_depth(
            np.array([2.0 * towards, -2.0 * towards]), limits, rotation
        )
        self.assertGreater(depths[0], depths[1])


class ElementBoxTests(unittest.TestCase):
    """What the note beside a substitution's element box says."""

    def test_a_blank_box_is_a_vacancy_not_a_mistake(self) -> None:
        for text in ("", "   "):
            with self.subTest(text=text):
                note, _ = element_box_note(text)
                self.assertEqual(note, "(vacancy)")

    def test_a_known_element_says_nothing(self) -> None:
        for text in ("Sr", "sr", "O", " Ca "):
            with self.subTest(text=text):
                self.assertEqual(element_box_note(text)[0], "")

    def test_an_unknown_symbol_is_marked_and_left_alone(self) -> None:
        for text in ("Xx", "Fe2", "?"):
            with self.subTest(text=text):
                self.assertEqual(element_box_note(text)[0], "(?)")

    def test_an_unknown_symbol_still_builds(self) -> None:
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        state.defect_panel_open = True
        group = state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=0
        )
        key = [k for k in state.plane_site_options(group) if k.role == "A"][0]
        state.toggle_plane_site(group, key)
        group.sites[0].element = "Xx"
        state.regenerate_focus_from_builder_if_changed()
        # The marker is a note, not a veto: a placeholder species is a legitimate
        # thing to be building with.
        self.assertIn("Xx", state.focus.element_symbols())


class VacancyIsABlankBoxTests(unittest.TestCase):
    """There is no vacancy kind; emptying the element box is how a site empties."""

    def _state(self) -> AppState:
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        state.defect_panel_open = True
        return state

    def test_the_panel_offers_no_vacancy_kind(self) -> None:
        self.assertNotIn("vacancy", DEFECT_KIND_KEYS)
        self.assertEqual(DEFECT_KIND_KEYS, ("substitution", "proton"))

    def test_emptying_the_box_empties_the_site(self) -> None:
        state = self._state()
        group = state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=0
        )
        key = [k for k in state.plane_site_options(group) if k.role == "A"][0]
        state.toggle_plane_site(group, key)
        group.sites[0].element = "Sr"
        state.regenerate_focus_from_builder_if_changed()
        self.assertIn("Sr", state.focus.element_symbols())
        substituted = state.focus.atom_count

        group.sites[0].element = ""
        state.regenerate_focus_from_builder_if_changed()
        self.assertNotIn("Sr", state.focus.element_symbols())
        self.assertEqual(state.focus.atom_count, substituted - 1)
        self.assertEqual([d.kind for d in state.builder_defects()], ["vacancy"])

    def test_a_stored_vacancy_comes_back_as_a_blank_box(self) -> None:
        state = self._state()
        stored = [SiteDefect("vacancy", SiteKey("X", 1, 0, 1, 2))]
        state.set_defect_rows(stored)
        # The panel has no kind for it, so it is held the way it would have been
        # made: a substitution with nothing in the box.
        self.assertEqual(state.defect_group_count(), 1)
        group = state.defect_groups[0]
        self.assertEqual(group.kind_key(), "substitution")
        self.assertEqual(group.sites[0].element, "")
        self.assertEqual(
            [d.signature() for d in state.builder_defects()],
            [d.signature() for d in stored],
        )


class StructureSummaryTests(unittest.TestCase):
    """The summary floated over the 3D view."""

    def _state(self) -> AppState:
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        return state

    def test_it_reports_the_cell_and_the_composition(self) -> None:
        rows = builder_summary_rows(self._state())
        text = [row.text for row in rows]
        self.assertTrue(any(line.startswith("a = ") for line in text))
        self.assertIn("Active structure: periodic", text)
        self.assertIn("A sites (La: 8)", text)
        self.assertIn("B sites (Fe: 8)", text)
        self.assertIn("X sites (O: 24)", text)
        self.assertTrue(any(line.startswith("Tilt system:") for line in text))

    def test_it_reports_what_the_structure_has_not_what_it_would_have(self) -> None:
        state = self._state()
        state.defect_panel_open = True
        group = state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=0
        )
        key = [k for k in state.plane_site_options(group) if k.role == "A"][0]
        state.toggle_plane_site(group, key)
        group.sites[0].element = "Sr"
        state.regenerate_focus_from_builder_if_changed()

        rows = {row.text: row.note for row in builder_summary_rows(state)}
        # Defects are applied after the ideal build, so the tally is of the
        # actual structure, with what the lattice would have held alongside.
        self.assertIn("A sites (La: 7, Sr: 1)", rows)
        self.assertEqual(rows["A sites (La: 7, Sr: 1)"], "ideal: La: 8")
        self.assertEqual(rows["B sites (Fe: 8)"], "")

    def test_a_broken_element_is_reported_rather_than_raised(self) -> None:
        state = self._state()
        state.a_site_element = ""
        rows = builder_summary_rows(state)
        self.assertTrue(any(row.error for row in rows))


class DefectPlaneOverlayTests(unittest.TestCase):
    """The drawn sheets have to land on the atoms, not between them."""

    def _state(self, periodic: bool = True) -> AppState:
        state = AppState()
        state.treat_as_periodic = periodic
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
            lattice_a=4.0,
            lattice_b=4.3,
            lattice_c=4.7,
        )
        state.defect_panel_open = True
        return state

    def test_only_the_focused_entry_is_drawn(self) -> None:
        state = self._state()
        state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=1
        )
        state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(1, 1, 1), plane=0
        )
        # The newest entry has focus, and only its family is on screen.
        labels = {overlay[3] for overlay in state.defect_plane_overlays()}
        self.assertEqual(labels, {"(111) substitution"})
        state.set_active_defect_group(0)
        labels = {overlay[3] for overlay in state.defect_plane_overlays()}
        self.assertEqual(labels, {"(001) substitution"})
        state.remove_defect_group(0)
        state.remove_defect_group(0)
        self.assertEqual(state.defect_plane_overlays(), [])

    def test_every_pickable_site_lies_on_a_drawn_sheet(self) -> None:
        """And every drawn sheet has pickable sites on it -- both directions.

        One sheet per plane is not enough: a layer can sit in more than one place
        at once, since the (001) layer holding the A sites at ``z = 0`` also holds
        the apical oxygens at ``z = 1``. Nor may a sheet be drawn anywhere else --
        a sheet is there to say where the sites you can click are.
        """
        for miller, kind in (
            ((0, 0, 1), "substitution"),
            ((1, 1, 1), "substitution"),
            ((1, 1, 1), "proton"),
            ((1, 1, 0), "substitution"),
        ):
            for position in range(2):
                state = self._state()
                group = state.add_defect_group(
                    kind=DEFECT_KIND_KEYS.index(kind), miller=miller, plane=0
                )
                options = state.plane_options(group)
                group.plane = options[position % len(options)]
                with self.subTest(miller=miller, kind=kind, plane=group.plane):
                    self._assert_sheets_match_pickable_sites(state, group, miller)

    def _assert_sheets_match_pickable_sites(self, state, group, miller) -> None:
        grid_shape = state.defect_grid_shape()
        structure = state.rendered_structure()
        normal = plane_miller_in_cell(grid_shape, miller)
        drawn = sorted(
            offset for overlay in state.defect_plane_overlays() for offset in overlay[1]
        )
        self.assertTrue(drawn)

        # The real pick targets, in the structure really being drawn -- including
        # the boundary-layer copies, whose plane coordinate can sit a whole cell
        # along the normal from any canonical site.
        focus = state.plane_render_sites(structure)
        targets = focus.pick_coords(structure.cartesian_coords)
        self.assertTrue(len(targets))
        fractional = np.linalg.solve(
            np.asarray(structure.lattice, dtype=np.float64).T, targets.T
        ).T
        covered = set()
        for projection in fractional @ normal:
            nearest = min(
                range(len(drawn)),
                key=lambda slot: abs(float(projection) - drawn[slot]),
            )
            self.assertLess(abs(float(projection) - drawn[nearest]), 1e-9)
            covered.add(nearest)
        # ...and no sheet is drawn that nothing pickable sits on.
        self.assertEqual(covered, set(range(len(drawn))))

    def _sheets(self, state) -> set:
        return {
            round(offset, 9)
            for overlay in state.defect_plane_overlays()
            for offset in overlay[1]
        }

    def test_only_the_worked_plane_is_drawn(self) -> None:
        state = self._state()
        group = state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=1
        )
        by_plane = {}
        for plane in state.plane_options(group):
            group.plane = plane
            by_plane[plane] = self._sheets(state)
            self.assertTrue(by_plane[plane])
        # A sheet is there to say where the sites you can click are, so no two
        # layers may claim the same one and none may be drawn for a layer that
        # is not the one being worked in.
        for plane, sheets in by_plane.items():
            others = set().union(
                *(other for key, other in by_plane.items() if key != plane)
            )
            with self.subTest(plane=plane):
                self.assertFalse(sheets & others)

    def test_a_layer_that_sits_in_two_places_gets_two_sheets(self) -> None:
        state = self._state()
        state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=0
        )
        # The (001) layer holding the A sites at z = 0 also holds the apical
        # oxygens at z = 1 -- the same layer, one cell along -- and the boundary
        # layer puts more copies of it on screen still. Every one needs a sheet.
        self.assertGreater(len(self._sheets(state)), 1)

    def test_a_proton_family_skips_the_layers_it_cannot_use(self) -> None:
        state = self._state()
        group = state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("proton"), miller=(1, 1, 1), plane=0
        )
        grid_shape = state.defect_grid_shape()
        # (111) alternates AO3 layers with bare B ones; a proton goes on an
        # oxygen, so only the AO3 layers are offered and only they are drawn.
        self.assertEqual(
            occupied_planes(grid_shape, state.treat_as_periodic, (1, 1, 1)),
            [0, 1, 2, 3],
        )
        self.assertEqual(state.plane_options(group), [0, 2])
        for plane in state.plane_options(group):
            self.assertTrue(
                sites_in_plane(grid_shape, state.treat_as_periodic, (1, 1, 1), plane, role="X")
            )

    def test_a_plane_outside_the_supercell_draws_nothing(self) -> None:
        state = self._state()
        state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 1), plane=97
        )
        # The plane the entry names is not in this cell, so it has no sites to
        # pick and nothing to draw a sheet through.
        self.assertEqual(state.defect_plane_overlays(), [])
        self.assertFalse(state.plane_render_sites(state.rendered_structure()))

    def test_the_legend_label_stays_a_readable_miller_index(self) -> None:
        state = self._state()
        state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(1, -1, 0), plane=0
        )
        label = state.defect_plane_overlays()[0][3]
        # "(1-10)" is not a Miller index anyone can read.
        self.assertEqual(label, "(1, -1, 0) substitution")

    def test_a_degenerate_miller_triple_draws_nothing(self) -> None:
        state = self._state()
        state.add_defect_group(
            kind=DEFECT_KIND_KEYS.index("substitution"), miller=(0, 0, 0), plane=0
        )
        self.assertEqual(state.plane_options(state.defect_groups[0]), [])
        self.assertEqual(state.defect_plane_overlays(), [])


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
