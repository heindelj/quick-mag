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
from quick_mag.atom_edits import SelectionSlab  # noqa: E402
from quick_mag.defects import SiteDefect  # noqa: E402
from quick_mag.perovskite_builder import SiteKey  # noqa: E402
from quick_mag.quick_mag_ui import (  # noqa: E402
    DEFECT_KIND_KEYS,
    FORMULA_MODE_KEYS,
    MAX_MATCH_DEFECT_CONCENTRATION,
    AUTO_SPIN_UPDATE_MIN_FPS,
    AUTO_SPIN_UPDATE_RESUME_FPS,
    EXCHANGE_PLOT_MAX_BARS,
    EXCHANGE_TIE_DECIMALS,
    MAX_CUSTOM_PATTERN_PERIOD,
    MIN_PLOT3D_HEIGHT,
    MIN_TWO_D_HEIGHT,
    PANE_SPLITTER_THICKNESS,
    TWO_D_PLOT_NAMES,
    EXCHANGE_BOTTOM_HEADROOM,
    TWO_D_BOTTOM_HEADROOM,
    TWO_D_TOP_HEADROOM,
    SPIN_CLASS_COLORS,
    SPHERE_LATITUDE_SEGMENTS,
    SPHERE_LONGITUDE_SEGMENTS,
    GLAZER_TILT_SYSTEMS,
    DefectEntry,
    STRUCTURE_ZOOM_RANGE,
    AppState,
    build_sphere_mesh,
    builder_summary_rows,
    candidate_pixels,
    cartesian_to_display,
    current_framerate,
    draw_pane_splitter,
    exchange_ion_label,
    exchange_pair_color,
    exchange_pair_frustration,
    exchange_pair_label,
    exchange_pair_tooltip,
    exchange_pairs_for_site,
    exchange_pick_candidates,
    exchange_path_alpha,
    exchange_prominent_render_atoms,
    exchange_unreached_partner_atoms,
    exchange_render_paths,
    exchange_selection_site,
    exchange_site_label,
    exchange_site_moments,
    gui_calculation_output,
    gui_two_d_pane,
    magnetic_pick_candidates,
    nearest_exchange_path,
    padded_two_d_limits,
    spin_class_color,
    spin_plot_categories,
    spin_plot_category,
    view_projection_key,
    source_site_for_render_index,
    split_pane_heights,
    visible_pair_couplings,
    summary_overlay_width,
    element_box_note,
    nearest_picked_atom,
    slab_arrow_endpoints,
    view_space_depth,
    DEFAULT_STRUCTURE_ROTATION,
    sphere_detail_for,
    highlighted_render_indices,
    oxidation_site_rows,
    site_hover_tooltip,
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
from quick_mag.polarization_model import PairCoupling  # noqa: E402
from quick_mag.spin_solver import (  # noqa: E402
    SpinConfig,
    canonical_moment_key,
    compute_config_energy,
)
from quick_mag.structure import ChemicalStructure, SavedSpinConfiguration  # noqa: E402


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
) -> DefectEntry:
    """Add one defect the way the Atoms panel would."""
    # A vacancy has no kind of its own in the panel: it is a substitution with
    # nothing in the element box.
    if kind == "vacancy":
        kind, element = "substitution", ""
    state.ensure_defect_entries()
    entry = DefectEntry(
        site=key,
        kind=DEFECT_KIND_KEYS.index(kind),
        element=element,
        orientation=orientation,
    )
    state.defect_entries.append(entry)
    return entry


def vacate_first(state: AppState, element: str) -> None:
    """Empty the first site holding ``element``, through the Atoms panel's edit."""
    row = next(row for row in state.atom_table() if row.element == element)
    state.set_atom_element(row, "")


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
    """Builder edits leave the landscape alone when updates are not interactive.

    Re-energizing means rebuilding the oxidation assignments and the exchange matrix,
    which is too expensive to do on every frame of a slider drag when the view cannot
    afford it -- either because the user switched live updates off, or because the
    frame-rate gate paused them.
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
        state.update_spin_energies_interactively = False
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
        self.assertTrue(
            state.update_spin_energies_interactively, "live updates are the default"
        )
        before = [config.energy for config in state.spin_landscape]

        self._edit(state, **TILT_EDIT)

        self.assertFalse(state.spin_energies_stale)
        self.assertNotEqual([config.energy for config in state.spin_landscape], before)

    def test_solving_clears_staleness(self) -> None:
        state = AppState()
        state.update_spin_energies_interactively = False
        self._edit(state, **TILT_EDIT)
        self.assertTrue(state.spin_energies_stale)

        state.run_magnetic_structure_calculation(structure=state.focus)

        self.assertFalse(state.spin_energies_stale)


class LargeStructurePerformanceTest(unittest.TestCase):
    """Guards on the things that made a large cell unusable."""

    def test_only_the_lowest_energy_assignment_is_enumerated(self):
        # Expanding every charge-balanced distribution into per-site assignments was
        # the slowest part of setting up a solve: 78,312 of them on the DQ default,
        # at ~5 s. Nothing past the head of the ranking is offered any more -- the
        # tail was mixed-valence distributions the energy model cannot tell apart,
        # and disagreeing with the head is now an edit rather than a selection.
        state = AppState()
        state.formula_mode = FORMULA_MODE_KEYS.index("dq")
        state.apply_defaults_for_formula()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.run_magnetic_structure_calculation(structure=state.focus)

        self.assertEqual(len(state.magnetic_oxidation_assignments), 1)
        self.assertEqual(state.selected_oxidation_assignment_index, 0)
        self.assertIs(
            state.predicted_oxidation_assignment(),
            state.magnetic_oxidation_assignments[0],
        )

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
        self.assertEqual(state.defect_entry_count(), 1)
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

    def test_compensating_protons_balance_a_substitution(self) -> None:
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
        second_entry = add_defect(state, "vacancy", SiteKey("X", 1, 0, 1, 2))
        pending = state.defect_for_entry(first)
        self.assertIsNotNone(pending)
        self.assertEqual(pending.kind, "vacancy")
        self.assertEqual(tuple(pending.site), ("B", 0, 0, 0, 0))
        self.assertEqual(len(state.builder_defects()), 2)

        second = state.defect_for_entry(second_entry)
        self.assertEqual(tuple(second.site), ("X", 1, 0, 1, 2))

        # Naming the element turns it back into the substitution it always was.
        first.element = "Zn"
        self.assertEqual(state.defect_for_entry(first).kind, "substitution")
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


class FormulaChangeGuardTests(unittest.TestCase):
    """Changing formula rebuilds from defaults, so it asks first."""

    def _untouched(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        return state

    def test_an_untouched_structure_has_nothing_to_lose(self) -> None:
        self.assertFalse(self._untouched().builder_has_edits())

    def test_an_edit_is_noticed_and_undoing_it_is_too(self) -> None:
        state = self._untouched()
        state.a_site_element = "Sr"
        self.assertTrue(state.builder_has_edits())
        state.a_site_element = "La"
        self.assertFalse(state.builder_has_edits())

    def test_a_defect_counts_as_an_edit(self) -> None:
        state = self._untouched()
        vacate_first(state, "La")
        self.assertTrue(state.builder_has_edits())

    def test_proceeding_opens_the_new_formula_at_its_defaults(self) -> None:
        state = self._untouched()
        state.a_site_element = "Sr"
        vacate_first(state, "Fe")
        structures = len(state.structures)

        state.pending_formula_mode = FORMULA_MODE_KEYS.index("double")
        state.apply_formula_change(state.pending_formula_mode)
        # The edits really are gone -- a cell half built from the old formula
        # would be neither one thing nor the other -- and no structure was
        # added: this is the same structure, rebuilt.
        self.assertEqual(state.formula_key(), "double")
        self.assertEqual(state.defect_entries, [])
        self.assertFalse(state.builder_has_edits())
        self.assertEqual(len(state.structures), structures)
        self.assertEqual(state.pending_formula_mode, -1)

    def test_new_structure_leaves_the_edited_one_alone(self) -> None:
        state = self._untouched()
        state.a_site_element = "Sr"
        state.regenerate_focus_from_builder_if_changed()
        edited = state.focus
        structures = len(state.structures)

        state.create_structure_with_formula(FORMULA_MODE_KEYS.index("high_entropy"))
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(len(state.structures), structures + 1)
        self.assertIsNot(state.focus, edited)
        self.assertEqual(state.formula_key(), "high_entropy")
        # The structure that was edited keeps its edit.
        self.assertIn("Sr", edited.element_symbols())

    def test_cancelling_leaves_everything_where_it_was(self) -> None:
        state = self._untouched()
        state.a_site_element = "Sr"
        before = state.formula_mode
        state.pending_formula_mode = FORMULA_MODE_KEYS.index("double")
        state.pending_formula_mode = -1  # what the Cancel button does
        self.assertEqual(state.formula_mode, before)
        self.assertEqual(state.a_site_element, "Sr")


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


class AtomTableEditTests(unittest.TestCase):
    """Editing a built structure through the Atoms panel's rows."""

    def _state(self) -> AppState:
        # Pinned to 2x2x2 rather than left on the builder default: the small
        # cell keeps the sites enumerable by hand.
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        return state

    def test_every_site_of_the_build_has_a_row_keyed_by_its_grid_address(self) -> None:
        state = self._state()
        rows = state.atom_table()
        self.assertEqual(len(rows), state.focus.atom_count)
        self.assertEqual(
            sorted(row.index for row in rows), list(range(state.focus.atom_count))
        )
        roles = {row.role for row in rows}
        self.assertEqual(roles, {"A", "B", "X"})
        for row in rows:
            self.assertIsNotNone(row.key)
            self.assertEqual(row.ref, ("site",) + tuple(row.key))
            self.assertEqual(row.element, state.focus.element_symbols()[row.index])
            np.testing.assert_allclose(
                row.cartesian, state.focus.cartesian_coords[row.index], atol=1e-9
            )

    def test_typing_an_element_substitutes_the_site(self) -> None:
        state = self._state()
        row = next(row for row in state.atom_table() if row.role == "A")
        state.set_atom_element(row, "Sr")
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(state.focus.element_symbols().count("Sr"), 1)
        self.assertEqual([d.kind for d in state.builder_defects()], ["substitution"])
        # The row is found again by its address, now carrying the new element.
        again = state.atom_row_for_ref(row.ref)
        self.assertEqual(again.element, "Sr")
        self.assertTrue(again.edited)
        self.assertEqual(again.ideal_element, "La")

    def test_typing_the_ideal_element_back_removes_the_defect(self) -> None:
        state = self._state()
        row = next(row for row in state.atom_table() if row.role == "A")
        state.set_atom_element(row, "Sr")
        state.regenerate_focus_from_builder_if_changed()
        state.set_atom_element(state.atom_row_for_ref(row.ref), "La")
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(state.defect_entries, [])
        self.assertNotIn("Sr", state.focus.element_symbols())

    def test_emptying_a_site_vacates_it_and_leaves_a_row_to_restore_from(self) -> None:
        state = self._state()
        before = state.focus.atom_count
        row = next(row for row in state.atom_table() if row.element == "O")
        state.set_atom_element(row, "")
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(state.focus.atom_count, before - 1)
        self.assertEqual([d.kind for d in state.builder_defects()], ["vacancy"])
        vacant = state.atom_row_for_ref(row.ref)
        self.assertTrue(vacant.vacant)
        self.assertEqual(vacant.index, -1)
        self.assertEqual(vacant.ideal_element, "O")
        # The row count is the site count, not the atom count.
        self.assertEqual(len(state.atom_table()), before)
        state.set_atom_element(vacant, vacant.ideal_element)
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(state.focus.atom_count, before)
        self.assertEqual(state.defect_entries, [])

    def test_a_proton_hangs_on_an_oxygen_and_is_a_row_of_its_own(self) -> None:
        state = self._state()
        before = state.focus.atom_count
        oxygen = next(row for row in state.atom_table() if row.element == "O")
        state.add_proton_to_atom(oxygen)
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(state.focus.atom_count, before + 1)
        protons = [row for row in state.atom_table() if row.role == "H"]
        self.assertEqual(len(protons), 1)
        proton = protons[0]
        self.assertEqual(proton.element, "H")
        self.assertEqual(proton.host_key, oxygen.key)
        self.assertEqual(proton.ref, ("proton",) + tuple(oxygen.key))
        self.assertEqual(proton.index, before)
        # Its position is the emitted atom's, a bond length from the host.
        bond = proton.cartesian - state.focus.cartesian_coords[oxygen.index]
        self.assertAlmostEqual(float(np.linalg.norm(bond)), 0.98, places=6)
        # A second proton on the same oxygen is refused.
        state.add_proton_to_atom(state.atom_row_for_ref(oxygen.ref))
        self.assertTrue(state.atom_edit_message)
        self.assertEqual(len(state.defect_entries), 1)
        # Emptying the proton's row takes it away again.
        state.set_atom_element(proton, "")
        state.regenerate_focus_from_builder_if_changed()
        self.assertEqual(state.focus.atom_count, before)
        self.assertEqual(state.defect_entries, [])

    def test_a_proton_refuses_anything_but_an_oxygen(self) -> None:
        state = self._state()
        iron = next(row for row in state.atom_table() if row.element == "Fe")
        state.add_proton_to_atom(iron)
        self.assertEqual(state.defect_entries, [])
        self.assertIn("oxygen", state.atom_edit_message)

    def test_a_vacancy_drops_its_own_proton(self) -> None:
        state = self._state()
        oxygen = next(row for row in state.atom_table() if row.element == "O")
        state.add_proton_to_atom(oxygen)
        state.regenerate_focus_from_builder_if_changed()
        state.set_atom_element(state.atom_row_for_ref(oxygen.ref), "")
        state.regenerate_focus_from_builder_if_changed()
        self.assertNotIn("H", state.focus.element_symbols())
        self.assertFalse([row for row in state.atom_table() if row.role == "H"])

    def test_a_named_site_removes_the_atom_it_names(self) -> None:
        state = self._state()
        oxygens = [row for row in state.atom_table() if row.element == "O"]
        before = state.focus.atom_count
        for row in (oxygens[0], oxygens[len(oxygens) // 2], oxygens[-1]):
            with self.subTest(site=row.key):
                state.defect_entries = [DefectEntry(site=row.key)]
                state.regenerate_focus_from_builder_if_changed()
                # Exactly one oxygen leaves, whichever site was checked.
                self.assertEqual(state.focus.atom_count, before - 1)
                self.assertEqual(state.focus.element_symbols().count("O"), 23)

    def test_shrinking_the_cell_leaves_an_out_of_range_entry_untouched(self) -> None:
        state = self._state()
        apply_builder_edits(
            state,
            perovskite_supercell_x=3,
            perovskite_supercell_y=3,
            perovskite_supercell_z=3,
        )
        far_site = SiteKey("B", 2, 2, 2)
        row = state.atom_row_for_ref(("site",) + tuple(far_site))
        self.assertIsNotNone(row)
        state.set_atom_element(row, "")
        state.regenerate_focus_from_builder_if_changed()

        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        # The site does not exist in the smaller cell, so the entry is skipped
        # -- but nothing about it is rewritten, and it has no row to show.
        self.assertEqual([entry.site for entry in state.defect_entries], [far_site])
        self.assertEqual(state.focus.atom_count, 40)
        self.assertIsNone(state.atom_row_for_ref(row.ref))

    def test_stored_defects_come_back_as_entries(self) -> None:
        state = self._state()
        rows = state.atom_table()
        state.set_atom_element(rows[0], "")
        state.set_atom_element(rows[1], "Sr")
        state.add_proton_to_atom(next(row for row in rows if row.element == "O"))
        state.regenerate_focus_from_builder_if_changed()
        stored = list(state.focus.generation_parameters.defects)
        self.assertEqual(len(stored), 3)

        state.set_defect_rows(stored)
        self.assertEqual(state.defect_entry_count(), 3)
        self.assertEqual(
            [d.signature() for d in state.builder_defects()],
            [d.signature() for d in stored],
        )

    def test_the_selection_survives_the_renumbering_a_vacancy_causes(self) -> None:
        state = self._state()
        rows = state.atom_table()
        first_oxygen = next(row for row in rows if row.element == "O")
        last_row = rows[-1]
        state.toggle_atom_selection(last_row.ref)
        state.set_atom_element(first_oxygen, "")
        state.regenerate_focus_from_builder_if_changed()
        # The last atom's index moved down by one, but its ref did not.
        selected = state.active_rows()
        self.assertEqual([row.ref for row in selected], [last_row.ref])
        self.assertEqual(selected[0].index, last_row.index - 1)

    def test_a_vacancy_drops_the_exchange_site_that_would_have_been_renumbered(self) -> None:
        state = self._state()
        state.selected_site_index = state.focus.atom_count - 1
        row = next(row for row in state.atom_table() if row.element == "O")
        state.set_atom_element(row, "")
        state.regenerate_focus_from_builder_if_changed()
        # The index would now name a different atom; rather than ring a
        # stranger, the selection is dropped.
        self.assertEqual(state.selected_site_index, -1)

    def test_a_relabel_reaches_the_analysis(self) -> None:
        state = self._state()
        row = next(row for row in state.atom_table() if row.element == "Fe")
        state.set_atom_element(row, "Co")
        state.regenerate_focus_from_builder_if_changed()
        analysis = state.magnetic_analysis_structure
        self.assertIs(analysis, state.focus)
        self.assertIn("Co", analysis.element_symbols())


class LoadedAtomEditTests(unittest.TestCase):
    """The same edits on a structure with no builder behind it."""

    def _state(self) -> tuple[AppState, ChemicalStructure]:
        state = AppState()
        structure = ChemicalStructure(
            name="loaded",
            lattice=np.eye(3) * 4.0,
            cartesian_coords=np.array(
                [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [2.0, 2.0, 0.0], [2.0, 0.0, 2.0]]
            ),
            atomic_labels=["Sr", "Ti", "O", "O"],
            magnetic_moments=np.zeros((4, 3)),
        )
        state.structures.append(structure)
        state.set_focus(structure)
        state.sync_active_structure()
        return state, structure

    def test_rows_are_keyed_by_position(self) -> None:
        state, structure = self._state()
        self.assertEqual(state.atom_edit_mode(), "loaded")
        rows = state.atom_table()
        self.assertEqual([row.element for row in rows], ["Sr", "Ti", "O", "O"])
        self.assertEqual(rows[1].ref, ("pos", 2.0, 2.0, 2.0))
        self.assertIsNone(rows[1].key)

    def test_substitution_relabels_in_place(self) -> None:
        state, structure = self._state()
        state.set_atom_element(state.atom_table()[1], "Zr")
        self.assertEqual(structure.atomic_labels, ["Sr", "Zr", "O", "O"])
        row = state.atom_row_for_index(1)
        self.assertTrue(row.edited)
        self.assertEqual(row.ideal_element, "Ti")
        # Typing the original back is an un-edit, not a second substitution.
        state.set_atom_element(row, "Ti")
        self.assertFalse(state.atom_row_for_index(1).edited)

    def test_emptying_removes_the_atom_and_keeps_a_vacancy_row(self) -> None:
        state, structure = self._state()
        row = state.atom_table()[2]
        state.toggle_atom_selection(row.ref)
        state.set_atom_element(row, "")
        self.assertEqual(structure.atom_count, 3)
        self.assertEqual(structure.atomic_labels, ["Sr", "Ti", "O"])
        rows = state.atom_table()
        self.assertEqual(len(rows), 4)
        vacant = state.atom_row_for_ref(row.ref)
        self.assertTrue(vacant.vacant)
        self.assertEqual(vacant.ideal_element, "O")
        # Still selected, so it stays listed and can be restored from there.
        self.assertEqual([r.ref for r in state.active_rows()], [row.ref])
        coords, labels = state.vacancy_markers(structure)
        self.assertEqual(labels, ["O"])
        np.testing.assert_allclose(coords, [[2.0, 2.0, 0.0]])
        state.set_atom_element(vacant, "O")
        self.assertEqual(structure.atom_count, 4)
        self.assertEqual(state.vacancy_markers(structure)[1], [])
        np.testing.assert_allclose(structure.cartesian_coords[3], [2.0, 2.0, 0.0])

    def test_a_proton_is_added_in_place(self) -> None:
        state, structure = self._state()
        oxygen = state.atom_table()[2]
        state.add_proton_to_atom(oxygen)
        self.assertEqual(structure.atom_count, 5)
        self.assertEqual(structure.atomic_labels[-1], "H")
        bond = structure.cartesian_coords[-1] - structure.cartesian_coords[2]
        self.assertAlmostEqual(float(np.linalg.norm(bond)), 0.98, places=6)

    def test_a_count_change_drops_site_indexed_results(self) -> None:
        state, structure = self._state()
        structure.spin_configurations.append(
            SavedSpinConfiguration(magnetic_moments=np.zeros((4, 3)), energy=0.0)
        )
        state.set_atom_element(state.atom_table()[0], "")
        self.assertEqual(structure.spin_configurations, [])


class SlabSelectionTests(unittest.TestCase):
    """The slab: a layer through the cell, addressed in lattice terms."""

    def _state(self) -> AppState:
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
            lattice_a=4.0,
        )
        return state

    def test_off_the_slab_selects_nothing(self) -> None:
        state = self._state()
        self.assertFalse(state.slab_enabled)
        self.assertEqual(state.slab_rows(), [])
        self.assertFalse(state.selection_active())

    def test_a_thin_001_slab_on_a_layer_picks_exactly_that_layer(self) -> None:
        state = self._state()
        state.slab_enabled = True
        state.selection_slab.direction = (0, 0, 1)
        state.selection_slab.thickness = 0.5
        rows = state.atom_table()
        # Every distinct height along c is a layer; the slab at each height
        # should hold exactly the rows at that height and nothing else.
        heights = sorted({round(float(row.cartesian[2]), 6) for row in rows})
        normal = np.array([0.0, 0.0, 1.0])
        low, high = 0.0, float(state.focus.lattice[2, 2])
        for height in heights:
            with self.subTest(height=height):
                state.selection_slab.offset = (height - low) / (high - low)
                picked = {row.ref for row in state.slab_rows()}
                expected = {
                    row.ref
                    for row in rows
                    if abs(float(row.cartesian @ normal) - height) < 0.25
                }
                self.assertEqual(picked, expected)
                self.assertTrue(picked)
        # The mid-height layer is a BO2 sheet: B sites and their equatorial
        # oxygens, and nothing else.
        state.selection_slab.offset = 0.25
        roles = {row.role for row in state.slab_rows()}
        self.assertEqual(roles, {"B", "X"})

    def test_a_111_slab_is_the_body_diagonal_of_the_cell(self) -> None:
        state = self._state()
        state.slab_enabled = True
        state.selection_slab.direction = (1, 1, 1)
        state.selection_slab.thickness = 0.1
        state.selection_slab.offset = 0.0
        # The low face of the cell along [111] holds the origin corner alone.
        picked = state.slab_rows()
        self.assertEqual(len(picked), 1)
        np.testing.assert_allclose(picked[0].cartesian, [0.0, 0.0, 0.0], atol=1e-6)

    def test_the_slab_joins_the_hand_picks_and_resetting_keeps_them(self) -> None:
        state = self._state()
        rows = state.atom_table()
        state.toggle_atom_selection(rows[-1].ref)
        state.slab_enabled = True
        state.selection_slab.direction = (0, 0, 1)
        state.selection_slab.thickness = 0.5
        state.selection_slab.offset = 0.0
        selected = {row.ref for row in state.active_rows()}
        self.assertIn(rows[-1].ref, selected)
        self.assertTrue(selected > {rows[-1].ref})
        # Putting the slab away leaves the hand picks exactly where they were.
        state.reset_slab_selection()
        self.assertFalse(state.slab_enabled)
        self.assertEqual([row.ref for row in state.picked_rows()], [rows[-1].ref])
        self.assertEqual([row.ref for row in state.active_rows()], [rows[-1].ref])
        state.clear_picked_atoms()
        self.assertEqual(state.active_rows(), [])

    def test_clicking_a_slab_atom_pulls_it_out_into_the_picked_list(self) -> None:
        state = self._state()
        state.slab_enabled = True
        state.selection_slab.direction = (0, 0, 1)
        state.selection_slab.thickness = 0.5
        state.selection_slab.offset = 0.0
        inside = state.slab_rows()
        self.assertTrue(len(inside) > 1)
        self.assertEqual(state.picked_rows(), [])

        # Clicking an atom the slab already holds leaves the active set alone
        # and moves that one atom into the picked list.
        before = {row.ref for row in state.active_rows()}
        state.toggle_atom_selection(inside[0].ref)
        self.assertEqual([row.ref for row in state.picked_rows()], [inside[0].ref])
        self.assertEqual({row.ref for row in state.active_rows()}, before)

        # The pick outlives the slab moving off it.
        state.selection_slab.offset = 0.5
        self.assertNotIn(inside[0].ref, {row.ref for row in state.slab_rows()})
        self.assertEqual([row.ref for row in state.picked_rows()], [inside[0].ref])
        self.assertIn(inside[0].ref, {row.ref for row in state.active_rows()})

        # Clearing the picks leaves the slab and its contents in place.
        state.clear_picked_atoms()
        self.assertEqual(state.picked_rows(), [])
        self.assertTrue(state.slab_enabled)
        self.assertEqual(
            {row.ref for row in state.active_rows()},
            {row.ref for row in state.slab_rows()},
        )

    def test_the_picked_list_is_in_table_order_not_click_order(self) -> None:
        state = self._state()
        rows = state.atom_table()
        for row in (rows[3], rows[1]):
            state.toggle_atom_selection(row.ref)
        self.assertEqual(
            [row.ref for row in state.picked_rows()], [rows[1].ref, rows[3].ref]
        )

    def test_a_vacated_site_stays_in_the_slab_by_its_ideal_position(self) -> None:
        state = self._state()
        state.slab_enabled = True
        state.selection_slab.direction = (0, 0, 1)
        state.selection_slab.thickness = 0.5
        state.selection_slab.offset = 0.25
        before = {row.ref for row in state.slab_rows()}
        victim = next(row for row in state.slab_rows() if row.element == "O")
        state.set_atom_element(victim, "")
        state.regenerate_focus_from_builder_if_changed()
        after = {row.ref for row in state.slab_rows()}
        self.assertEqual(before, after)
        self.assertTrue(state.atom_row_for_ref(victim.ref).vacant)

    def test_the_arrow_points_along_the_lattice_direction(self) -> None:
        lattice = np.array([[4.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
        slab = SelectionSlab(direction=(0, 1, 0), offset=0.5, thickness=1.0)
        tail, tip = slab_arrow_endpoints(lattice, slab)
        direction = (tip - tail) / np.linalg.norm(tip - tail)
        np.testing.assert_allclose(direction, lattice[1] / np.linalg.norm(lattice[1]))
        self.assertIsNone(slab_arrow_endpoints(lattice, SelectionSlab(direction=(0, 0, 0))))


class ViewModeTests(unittest.TestCase):
    """Which decoration the 3D view runs."""

    def test_the_exchange_plot_holds_the_view_only_while_a_site_is_selected(self) -> None:
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        self.assertEqual(state.structure_view_mode(), "plain")
        state.two_d_plot_index = 1
        state.selected_site_index = state.magnetic_site_indices[0]
        state.structure_view_focus = "exchange"
        self.assertEqual(state.structure_view_mode(), "exchange")
        # Clearing the selection drops to the *plain* view.
        state.selected_site_index = -1
        self.assertEqual(state.structure_view_mode(), "plain")


class PickTests(unittest.TestCase):
    """Which atom a click hits."""

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
        # Viewed edge-on, a layer's atoms line up almost exactly. Larger depth
        # is nearer the viewer -- verified against which sphere ImPlot3D
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

    def test_aiming_at_an_atom_in_a_live_plot_picks_it(self) -> None:
        """End to end through ImPlot3D's own projection, not a stand-in for it."""
        state = AppState()
        apply_builder_edits(
            state,
            perovskite_supercell_x=2,
            perovskite_supercell_y=2,
            perovskite_supercell_z=2,
        )
        structure = state.rendered_structure()
        coords = structure.cartesian_coords
        candidates = list(range(len(coords)))
        limits = compute_plot_box_limits(coords)
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
                # Aiming exactly at an atom lands on *an* atom: with the
                # boundary layer drawn several coincide on screen, and depth
                # decides between them.
                outcome["hits"] = sum(
                    nearest_picked_atom(
                        pixels, candidates, depths, tuple(pixels[position])
                    )
                    >= 0
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
        row = next(row for row in state.atom_table() if row.role == "A")
        state.set_atom_element(row, "Xx")
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
        return state

    def test_the_panel_offers_no_vacancy_kind(self) -> None:
        self.assertNotIn("vacancy", DEFECT_KIND_KEYS)
        self.assertEqual(DEFECT_KIND_KEYS, ("substitution", "proton"))

    def test_emptying_the_box_empties_the_site(self) -> None:
        state = self._state()
        row = next(row for row in state.atom_table() if row.role == "A")
        state.set_atom_element(row, "Sr")
        state.regenerate_focus_from_builder_if_changed()
        self.assertIn("Sr", state.focus.element_symbols())
        substituted = state.focus.atom_count

        row = next(row for row in state.atom_table() if row.element == "Sr")
        state.set_atom_element(row, "")
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
        self.assertEqual(state.defect_entry_count(), 1)
        entry = state.defect_entries[0]
        self.assertEqual(entry.kind_key(), "substitution")
        self.assertEqual(entry.element, "")
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

    def test_it_reports_the_formula_cell_and_composition(self) -> None:
        rows = builder_summary_rows(self._state())
        text = [row.text for row in rows]
        self.assertEqual(text[0], "Formula: Perovskite (ABX3)")
        self.assertTrue(any(line.startswith("a = ") for line in text))
        self.assertIn("A sites (La: 8)", text)
        self.assertIn("B sites (Fe: 8)", text)
        self.assertIn("X sites (O: 24)", text)
        self.assertTrue(any(line.startswith("Tilt system:") for line in text))

    def test_the_lattice_constants_are_those_of_the_whole_cell(self) -> None:
        state = self._state()
        # 2x2x2 of a 4 A cube: an 8 A cell, and that is what the picture shows.
        self.assertEqual(state.builder_cell_repeats(), (2, 2, 2))
        self.assertEqual(state.builder_cell_lengths(), (8.0, 8.0, 8.0))
        np.testing.assert_allclose(
            np.linalg.norm(state.focus.lattice, axis=1), state.builder_cell_lengths()
        )
        rows = builder_summary_rows(state)
        line = next(row for row in rows if row.text.startswith("a = "))
        self.assertTrue(line.text.startswith("a = 8.000 A"))
        self.assertIn("cube edge 4.000", line.note)

    def test_the_formula_line_says_periodic_or_cluster(self) -> None:
        state = self._state()
        self.assertEqual(builder_summary_rows(state)[0].note, "periodic")
        apply_builder_edits(state, treat_as_periodic=False)
        self.assertEqual(builder_summary_rows(state)[0].note, "cluster")

    def test_the_formula_line_follows_the_builder_mode(self) -> None:
        state = self._state()
        apply_builder_edits(
            state, formula_mode=FORMULA_MODE_KEYS.index("double")
        )
        self.assertEqual(
            builder_summary_rows(state)[0].text,
            "Formula: Double Perovskite (A2B'B''X6)",
        )

    def test_it_reports_what_the_structure_has_not_what_it_would_have(self) -> None:
        state = self._state()
        row = next(row for row in state.atom_table() if row.role == "A")
        state.set_atom_element(row, "Sr")
        state.regenerate_focus_from_builder_if_changed()

        rows = {row.text: row.note for row in builder_summary_rows(state)}
        # Defects are applied after the ideal build, so the tally is of the
        # actual structure, with what the lattice would have held alongside.
        self.assertIn("A sites (La: 7, Sr: 1)", rows)
        self.assertEqual(rows["A sites (La: 7, Sr: 1)"], "ideal: La: 8")
        self.assertEqual(rows["B sites (Fe: 8)"], "")

    def test_the_name_is_not_a_row_it_is_the_title(self) -> None:
        state = self._state()
        state.rename_structure(state.focus, "LaFeO3 test")
        # Collapsing the box has to leave the name behind, so the name lives in
        # the title bar and must not be duplicated as a row inside it.
        self.assertNotIn(
            "LaFeO3 test", [row.text for row in builder_summary_rows(state)]
        )
        self.assertEqual(state.focus.name, "LaFeO3 test")

    def test_renaming_retitles_the_box_without_replacing_it(self) -> None:
        outcome: dict = {}

        def gui() -> None:
            # "###" fixes the id: without it a rename would make a brand new
            # window, back in the corner, having forgotten where it was dragged
            # to and whether it was rolled up.
            outcome["ids"] = {
                imgui.get_id(f"{name}###structure_summary")
                for name in ("Structure 1", "LaFeO3 test", "")
            }

        render_frames(gui)
        self.assertEqual(len(outcome["ids"]), 1)

    def test_the_box_does_not_resize_as_a_tilt_angle_changes(self) -> None:
        state = self._state()
        apply_builder_edits(
            state,
            perovskite_tilt_system=GLAZER_TILT_SYSTEMS.index("a0a0c-"),
        )
        widths: set = set()

        def gui() -> None:
            for angle in (0.0, -42.5, 45.0, -7.25, 1.0):
                state.tilt_angle_z = angle
                widths.add(round(summary_overlay_width(builder_summary_rows(state)), 3))

        render_frames(gui)
        # A readout that twitches while you drag the slider you are reading is
        # worse than one a few pixels wider than it strictly needs to be.
        self.assertEqual(len(widths), 1)

    def test_a_broken_element_is_reported_rather_than_raised(self) -> None:
        state = self._state()
        state.a_site_element = ""
        rows = builder_summary_rows(state)
        self.assertTrue(any(row.error for row in rows))


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


class RenderIndexInverseTests(unittest.TestCase):
    """``source_site_for_render_index`` undoes ``highlighted_render_indices``."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.run_magnetic_structure_calculation(structure=state.focus)
        return state

    def test_every_image_maps_back_to_the_one_site_it_images(self) -> None:
        state = self._state()
        focus = state.focus
        rendered = state.rendered_structure()
        self.assertGreater(rendered.atom_count, focus.atom_count)
        for site in range(focus.atom_count):
            images = highlighted_render_indices(rendered, focus, site)
            self.assertTrue(images)
            for image in images:
                self.assertEqual(
                    source_site_for_render_index(rendered, focus, image), site
                )

    def test_identical_structures_pass_the_index_through(self) -> None:
        state = self._state()
        focus = state.focus
        self.assertEqual(source_site_for_render_index(focus, focus, 4), 4)

    def test_out_of_range_render_indices_match_nothing(self) -> None:
        state = self._state()
        rendered = state.rendered_structure()
        self.assertEqual(
            source_site_for_render_index(rendered, state.focus, 10_000), -1
        )
        self.assertEqual(source_site_for_render_index(rendered, state.focus, -1), -1)


class ExchangeCouplingPlotTests(unittest.TestCase):
    """The pair table behind the exchange-coupling bar chart, and its filtering."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        return state

    def test_the_baseline_builds_couplings_without_a_solve(self) -> None:
        """The plot is populated on focus, not only after Magnetic Structure runs."""
        state = self._state()
        self.assertFalse(state.magnetic_solution_cache)
        self.assertTrue(state.magnetic_pair_couplings)

    def test_pairs_agree_with_the_solver_matrix_up_to_its_sign_flip(self) -> None:
        """The table is the model convention; the matrix is the negated solver one."""
        state = self._state()
        compact = {site: i for i, site in enumerate(state.magnetic_site_indices)}
        for pair in state.magnetic_pair_couplings:
            a, b = compact[pair.site_i], compact[pair.site_j]
            self.assertAlmostEqual(
                pair.j_eff, -float(state.magnetic_j_matrix[a, b]), places=12
            )

    def test_nothing_selected_shows_every_pair(self) -> None:
        state = self._state()
        self.assertEqual(state.selected_site_index, -1)
        pairs, total = visible_pair_couplings(state)
        self.assertEqual(total, len(state.magnetic_pair_couplings))
        self.assertEqual(len(pairs), total)

    def test_selecting_an_atom_keeps_only_that_atom_s_couplings(self) -> None:
        state = self._state()
        site = state.magnetic_site_indices[0]
        state.selected_site_index = site
        pairs, total = visible_pair_couplings(state)
        self.assertTrue(pairs)
        self.assertLess(total, len(state.magnetic_pair_couplings))
        for pair in pairs:
            self.assertIn(site, (pair.site_i, pair.site_j))
        expected = sum(
            1
            for pair in state.magnetic_pair_couplings
            if site in (pair.site_i, pair.site_j)
        )
        self.assertEqual(total, expected)

    def test_selecting_a_non_magnetic_atom_leaves_no_bars(self) -> None:
        """An O or La site has no couplings, and that is an empty plot, not an error."""
        state = self._state()
        symbols = state.magnetic_analysis_structure.element_symbols()
        oxygen = symbols.index("O")
        state.selected_site_index = oxygen
        self.assertEqual(visible_pair_couplings(state), ([], 0))

    def test_bars_are_ordered_by_magnitude(self) -> None:
        """To the resolution the order actually distinguishes; see
        ``exchange_bar_sort_key`` for why couplings that tie must be allowed to."""
        state = self._state()
        pairs, _total = visible_pair_couplings(state)
        magnitudes = [
            round(abs(pair.j_eff) * 1000.0, EXCHANGE_TIE_DECIMALS) for pair in pairs
        ]
        self.assertEqual(magnitudes, sorted(magnitudes, reverse=True))

    def test_the_bar_count_is_capped(self) -> None:
        """The cap keeps the strongest couplings and drops the tail, not the reverse."""
        state = self._state()
        real = list(state.magnetic_pair_couplings)
        self.assertTrue(real)
        state.magnetic_pair_couplings = [
            replace(real[0], j_eff=float(n)) for n in range(EXCHANGE_PLOT_MAX_BARS + 25)
        ]
        pairs, total = visible_pair_couplings(state)
        self.assertEqual(total, EXCHANGE_PLOT_MAX_BARS + 25)
        self.assertEqual(len(pairs), EXCHANGE_PLOT_MAX_BARS)
        self.assertEqual(pairs[0].j_eff, float(EXCHANGE_PLOT_MAX_BARS + 24))

    def test_couplings_are_cleared_with_the_landscape(self) -> None:
        state = self._state()
        self.assertTrue(state.magnetic_pair_couplings)
        state.reset_spin_landscape("gone")
        self.assertEqual(state.magnetic_pair_couplings, [])

    def test_pair_labels_are_order_independent(self) -> None:
        self.assertEqual(
            exchange_pair_label("Mn", "Fe"), exchange_pair_label("Fe", "Mn")
        )
        self.assertEqual(exchange_pair_label("Fe", "Mn"), "Fe - Mn")

    def test_pair_colors_are_opaque_in_range_and_rank_separated(self) -> None:
        categories = ["Fe - Fe", "Fe - Mn", "Mn - Mn"]
        colors = [
            exchange_pair_color(category, rank)
            for rank, category in enumerate(categories)
        ]
        for color in colors:
            self.assertEqual(len(color), 4)
            self.assertEqual(color[3], 1.0)
            self.assertTrue(all(0.0 <= channel <= 1.0 for channel in color[:3]))
        self.assertEqual(len(set(colors)), len(colors))

    def test_pair_colors_are_deterministic(self) -> None:
        self.assertEqual(
            exchange_pair_color("Fe - Mn", 1), exchange_pair_color("Fe - Mn", 1)
        )

    def test_site_labels_name_the_element_and_index(self) -> None:
        state = self._state()
        site = state.magnetic_site_indices[0]
        symbol = state.magnetic_analysis_structure.element_symbols()[site]
        self.assertEqual(exchange_site_label(state, site), f"{symbol}{site}")

    def test_tooltip_reports_the_sense_and_the_geometry(self) -> None:
        state = self._state()
        pair = max(state.magnetic_pair_couplings, key=lambda p: p.j_eff)
        tooltip = exchange_pair_tooltip(state, pair)
        # A 180-degree Fe-O-Fe bridge is antiferromagnetic, J > 0 in this convention.
        self.assertIn("AFM", tooltip)
        self.assertIn(f"{pair.j_eff * 1000.0:+.3f} meV", tooltip)
        self.assertIn(f"{pair.distance:.3f} A", tooltip)
        self.assertIn("via O", tooltip)


class ExchangeFrustrationTests(unittest.TestCase):
    """The one part of the coupling plot that depends on the spin configuration."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        return state

    def _energies(self, state: AppState) -> list[float]:
        return [
            sum(
                exchange_pair_frustration(pair, exchange_site_moments(state))
                for pair in state.magnetic_pair_couplings
            )
            for _ in (0,)
        ]

    def test_pair_contributions_sum_to_the_configuration_energy(self) -> None:
        """J_ij (m_i . m_j) summed over pairs is the model energy, exactly.

        Ties the pair table, the sign convention and the frustration measure to the
        number the solver reports: if any one of the three flips, this breaks.
        """
        state = self._state()
        for index in range(len(state.displayed_spin_configs())):
            with self.subTest(config=index):
                state.selected_spin_config_index = index
                moments = exchange_site_moments(state)
                total = sum(
                    exchange_pair_frustration(pair, moments)
                    for pair in state.magnetic_pair_couplings
                )
                self.assertAlmostEqual(
                    total, state.selected_spin_config().energy, places=9
                )

    def test_frustration_grows_with_energy(self) -> None:
        """Higher up the landscape means more couplings the ordering fights."""
        state = self._state()
        counts = []
        for index in range(len(state.displayed_spin_configs())):
            state.selected_spin_config_index = index
            moments = exchange_site_moments(state)
            counts.append(
                sum(
                    exchange_pair_frustration(pair, moments) > 0.0
                    for pair in state.magnetic_pair_couplings
                )
            )
        self.assertEqual(counts, sorted(counts))
        # The ground state of an all-AFM cubic perovskite is G, which cannot satisfy
        # every bond; F, at the top, satisfies none of them.
        self.assertGreater(counts[0], 0)
        self.assertEqual(counts[-1], len(state.magnetic_pair_couplings))

    def test_no_configuration_means_no_frustration(self) -> None:
        state = self._state()
        state.reset_spin_landscape("nothing selected")
        self.assertIsNone(exchange_site_moments(state))
        pair = PairCoupling(
            site_i=0,
            site_j=1,
            metal_i="Fe",
            metal_j="Fe",
            j_eff=1.0,
            distance=4.0,
            bridge_count=1,
            ligands=("O",),
            angles_deg=(180.0,),
        )
        self.assertEqual(exchange_pair_frustration(pair, None), 0.0)

    def test_tooltip_names_the_sense_only_when_a_configuration_says_so(self) -> None:
        state = self._state()
        pair = state.magnetic_pair_couplings[0]
        plain = exchange_pair_tooltip(state, pair, 0.0)
        self.assertNotIn("this configuration", plain)
        self.assertIn("Frustrated in this configuration",
                      exchange_pair_tooltip(state, pair, 1.0))
        self.assertIn("Satisfied in this configuration",
                      exchange_pair_tooltip(state, pair, -1.0))


class ExchangeBarOrderTests(unittest.TestCase):
    """Bars hold their positions instead of reshuffling under the cursor."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        state.two_d_plot_index = 1
        return state

    def test_equal_couplings_break_ties_on_the_atoms(self) -> None:
        """A cubic cell's six neighbours are exactly degenerate; order them anyway."""
        state = self._state()
        state.selected_site_index = state.magnetic_site_indices[0]
        pairs, _total = visible_pair_couplings(state)
        magnitudes = {round(abs(pair.j_eff), 12) for pair in pairs}
        self.assertEqual(len(magnitudes), 1, "expected degenerate couplings here")
        keys = [(pair.site_i, pair.site_j) for pair in pairs]
        self.assertEqual(keys, sorted(keys))

    def test_the_order_is_stable_across_frames(self) -> None:
        state = self._state()
        state.selected_site_index = state.magnetic_site_indices[0]
        first = [(p.site_i, p.site_j) for p in visible_pair_couplings(state)[0]]
        for _ in range(3):
            again = [(p.site_i, p.site_j) for p in visible_pair_couplings(state)[0]]
            self.assertEqual(again, first)

    def test_selecting_a_different_atom_re_establishes_the_order(self) -> None:
        state = self._state()
        state.selected_site_index = state.magnetic_site_indices[0]
        visible_pair_couplings(state)
        first_key = state._exchange_bar_order_key
        state.selected_site_index = state.magnetic_site_indices[1]
        visible_pair_couplings(state)
        self.assertNotEqual(state._exchange_bar_order_key, first_key)
        pairs, _total = visible_pair_couplings(state)
        keys = [(pair.site_i, pair.site_j) for pair in pairs]
        self.assertEqual(keys, sorted(keys))

    def test_a_rebuild_re_establishes_the_order(self) -> None:
        """New couplings mean a new sort; the frozen order is not frozen forever."""
        state = self._state()
        visible_pair_couplings(state)
        before = state._exchange_bar_order_key
        state._exchange_generation += 1
        visible_pair_couplings(state)
        self.assertNotEqual(state._exchange_bar_order_key, before)


class ExchangePathTests(unittest.TestCase):
    """The M-L-M paths drawn from a selected atom in the 3D view."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        state.two_d_plot_index = 1
        return state

    def test_paths_run_metal_to_ligand_to_metal_at_the_right_distances(self) -> None:
        """Every hop is a real M-L bond, so the path traces the actual pathway."""
        state = self._state()
        site = state.magnetic_site_indices[0]
        rendered = state.rendered_structure()
        paths = exchange_render_paths(
            state, rendered, rendered.cartesian_coords, site, use_cartesian=True
        )
        self.assertTrue(paths)
        for pair, path in paths:
            self.assertEqual(path.shape, (3, 3))
            hops = np.linalg.norm(np.diff(path, axis=0), axis=1)
            # The two hops sum to at least the metal-metal distance, and equal it
            # exactly on a straight 180-degree bridge.
            self.assertGreaterEqual(float(hops.sum()) + 1e-9, pair.distance)
            self.assertTrue(np.all(hops > 0.5))

    def test_paths_start_at_the_selected_atom(self) -> None:
        """Oriented outwards, whichever end of the stored pair the selection is.

        A site in the middle of the range, so that it is ``site_j`` of some pairs and
        ``site_i`` of others -- the lowest magnetic index is never ``site_j``, and
        would not exercise the flip at all.

        Every bridge contributes such an outgoing path. A boundary-crossing one
        contributes a second, arriving one as well, which starts out of the box
        at the image the selected atom would occupy there -- so the assertion
        is per bridge, not over the whole list.
        """
        state = self._state()
        site = state.magnetic_site_indices[len(state.magnetic_site_indices) // 2]
        rendered = state.rendered_structure()
        coords = rendered.cartesian_coords
        origin = coords[
            highlighted_render_indices(
                rendered, state.magnetic_analysis_structure, site
            )[0]
        ]
        paths = exchange_render_paths(state, rendered, coords, site, use_cartesian=True)
        self.assertTrue(any(pair.site_i == site for pair, _ in paths))
        self.assertTrue(any(pair.site_j == site for pair, _ in paths))
        for pair in exchange_pairs_for_site(state.magnetic_pair_couplings, site):
            with self.subTest(pair=(pair.site_i, pair.site_j)):
                self.assertTrue(
                    any(
                        other is pair and np.allclose(path[0], origin, atol=1e-6)
                        for other, path in paths
                    )
                )

    def test_one_path_per_bridge_and_end(self) -> None:
        """One path per bridge leaving the selection, plus any arriving ones.

        A bridge whose partner is drawn where the outgoing path already ends
        contributes nothing further -- the duplicate is dropped -- so the count
        sits between one and two per bridge.
        """
        state = self._state()
        site = state.magnetic_site_indices[0]
        rendered = state.rendered_structure()
        paths = exchange_render_paths(
            state, rendered, rendered.cartesian_coords, site, use_cartesian=True
        )
        bridges = sum(
            pair.bridge_count
            for pair in exchange_pairs_for_site(state.magnetic_pair_couplings, site)
        )
        self.assertGreaterEqual(len(paths), bridges)
        self.assertLessEqual(len(paths), 2 * bridges)
        # Nothing is drawn twice over.
        keys = {
            np.ascontiguousarray(np.round(path, 6)).tobytes() for _pair, path in paths
        }
        self.assertEqual(len(keys), len(paths))

    def test_no_selection_means_no_paths(self) -> None:
        state = self._state()
        rendered = state.rendered_structure()
        self.assertEqual(
            exchange_render_paths(
                state, rendered, rendered.cartesian_coords, -1, use_cartesian=True
            ),
            [],
        )

    def test_fractional_paths_match_the_cartesian_ones(self) -> None:
        """The view can be drawing either frame; the path has to follow it."""
        state = self._state()
        site = state.magnetic_site_indices[0]
        rendered = state.rendered_structure()
        cartesian = exchange_render_paths(
            state, rendered, rendered.cartesian_coords, site, use_cartesian=True
        )
        fractional = exchange_render_paths(
            state, rendered, rendered.fractional_coords, site, use_cartesian=False
        )
        self.assertEqual(len(cartesian), len(fractional))
        for (_pair, cart), (_other, frac) in zip(cartesian, fractional):
            self.assertTrue(
                np.allclose(cartesian_to_display(cart, rendered.lattice, False), frac)
            )

    def test_every_prominent_atom_lies_on_a_path(self) -> None:
        """Which is what keeps a bright atom from implying a coupling that is not drawn."""
        state = self._state()
        site = state.magnetic_site_indices[0]
        rendered = state.rendered_structure()
        coords = rendered.cartesian_coords
        paths = exchange_render_paths(state, rendered, coords, site, use_cartesian=True)
        prominent = exchange_prominent_render_atoms(coords, paths)
        self.assertTrue(prominent)
        points = np.concatenate([path for _pair, path in paths], axis=0)
        for index in prominent:
            self.assertLess(
                float(np.min(np.linalg.norm(points - coords[index], axis=1))), 1e-6
            )

    def test_prominent_atoms_are_the_selection_its_partners_and_the_ligands(self) -> None:
        state = self._state()
        site = state.magnetic_site_indices[0]
        rendered = state.rendered_structure()
        coords = rendered.cartesian_coords
        paths = exchange_render_paths(state, rendered, coords, site, use_cartesian=True)
        prominent = exchange_prominent_render_atoms(coords, paths)
        symbols = rendered.element_symbols()
        self.assertEqual({symbols[index] for index in prominent}, {"Fe", "O"})
        self.assertLess(len(prominent), rendered.atom_count)

    def test_every_coupled_partner_is_drawn_prominently(self) -> None:
        """One bar, one bright neighbour -- including across the cell boundary.

        A coupling to a neighbour on the far side of the cell runs out of the
        drawn box and the neighbour is drawn wrapped to the opposite face, so
        no path endpoint lands on it. It used to fade with the rest of the cell,
        which left the chart showing a bar for an atom that was not there.
        """
        state = self._state()
        rendered = state.rendered_structure()
        coords = rendered.cartesian_coords
        analysis = state.magnetic_analysis_structure
        for site in state.magnetic_site_indices:
            with self.subTest(site=site):
                state.selected_site_index = site
                paths = exchange_render_paths(
                    state, rendered, coords, site, use_cartesian=True
                )
                prominent = exchange_prominent_render_atoms(coords, paths)
                prominent |= exchange_unreached_partner_atoms(
                    state, rendered, site, prominent
                )
                partners = {
                    pair.site_j if pair.site_i == site else pair.site_i
                    for pair in exchange_pairs_for_site(
                        state.magnetic_pair_couplings, site
                    )
                }
                self.assertTrue(partners)
                for partner in partners:
                    images = set(
                        highlighted_render_indices(rendered, analysis, partner)
                    )
                    self.assertTrue(images & prominent)

    def test_a_path_arrives_at_the_image_it_couples_to(self) -> None:
        """Every path leaves the selected atom or lands on the atom it reaches.

        A coupling across the cell boundary used to show only the half running
        out of the box, with nothing arriving at the neighbour -- which is
        drawn wrapped to the opposite face.
        """
        state = self._state()
        rendered = state.rendered_structure()
        coords = rendered.cartesian_coords
        analysis = state.magnetic_analysis_structure
        for site in state.magnetic_site_indices:
            with self.subTest(site=site):
                state.selected_site_index = site
                paths = exchange_render_paths(
                    state, rendered, coords, site, use_cartesian=True
                )
                self.assertTrue(paths)
                selected_images = [
                    coords[index]
                    for index in highlighted_render_indices(rendered, analysis, site)
                ]
                for _pair, path in paths:
                    leaves = any(
                        float(np.abs(path[0] - position).max()) < 1e-6
                        for position in selected_images
                    )
                    arrives = bool(
                        (np.abs(coords - path[-1][None, :]).max(axis=1) < 1e-6).any()
                    )
                    self.assertTrue(leaves or arrives)

    def test_a_reached_partner_gains_no_extra_bright_images(self) -> None:
        """The fallback only fires for partners nothing else lit."""
        state = self._state()
        rendered = state.rendered_structure()
        coords = rendered.cartesian_coords
        site = state.magnetic_site_indices[0]
        paths = exchange_render_paths(
            state, rendered, coords, site, use_cartesian=True
        )
        prominent = exchange_prominent_render_atoms(coords, paths)
        extra = exchange_unreached_partner_atoms(state, rendered, site, prominent)
        self.assertFalse(extra & prominent)

    def test_no_paths_means_nothing_is_prominent(self) -> None:
        state = self._state()
        rendered = state.rendered_structure()
        self.assertEqual(
            exchange_prominent_render_atoms(rendered.cartesian_coords, []), set()
        )

    def test_alpha_tracks_coupling_strength(self) -> None:
        strongest = 0.05
        self.assertAlmostEqual(exchange_path_alpha(0.0, strongest), 0.18, places=6)
        self.assertAlmostEqual(exchange_path_alpha(strongest, strongest), 1.0, places=6)
        self.assertAlmostEqual(
            exchange_path_alpha(-strongest, strongest), 1.0, places=6
        )
        middle = exchange_path_alpha(strongest / 2, strongest)
        self.assertTrue(0.18 < middle < 1.0)
        # No couplings at all must not divide by zero.
        self.assertAlmostEqual(exchange_path_alpha(0.0, 0.0), 1.0, places=6)

    def test_the_nearest_path_is_found_by_distance_to_its_segments(self) -> None:
        """Hovering the middle of a bond names it, not just its endpoints."""
        first = np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]])
        second = np.array([[0.0, 60.0], [50.0, 60.0], [100.0, 60.0]])
        paths = [first, second]
        self.assertEqual(nearest_exchange_path(paths, (25.0, 2.0)), 0)
        self.assertEqual(nearest_exchange_path(paths, (75.0, 58.0)), 1)
        self.assertEqual(nearest_exchange_path(paths, (50.0, 30.0)), -1)

    def test_a_degenerate_path_still_measures(self) -> None:
        """Viewed straight down a bond, all three points project onto one pixel."""
        point = np.array([[10.0, 10.0], [10.0, 10.0], [10.0, 10.0]])
        self.assertEqual(nearest_exchange_path([point], (11.0, 11.0)), 0)
        self.assertEqual(nearest_exchange_path([point], (200.0, 200.0)), -1)


class PaneSplitTests(unittest.TestCase):
    """The draggable split between the 3D and 2D plots."""

    def test_the_fraction_sets_the_share(self) -> None:
        available = 1000.0
        usable = available - PANE_SPLITTER_THICKNESS
        top, bottom = split_pane_heights(available, 0.30)
        self.assertAlmostEqual(top + bottom, usable, places=6)
        self.assertAlmostEqual(bottom, usable * 0.30, places=6)

    def test_neither_plot_is_squeezed_below_its_minimum(self) -> None:
        available = 1000.0
        for fraction in (0.0, 0.01, 0.5, 0.99, 1.0):
            with self.subTest(fraction=fraction):
                top, bottom = split_pane_heights(available, fraction)
                self.assertGreaterEqual(top, MIN_PLOT3D_HEIGHT - 1e-9)
                self.assertGreaterEqual(bottom, MIN_TWO_D_HEIGHT - 1e-9)
                self.assertAlmostEqual(
                    top + bottom, available - PANE_SPLITTER_THICKNESS, places=6
                )

    def test_a_pane_too_short_for_both_shares_it_out(self) -> None:
        """Both stay on screen, in proportion, rather than one being pushed out."""
        available = 0.5 * (MIN_PLOT3D_HEIGHT + MIN_TWO_D_HEIGHT)
        top, bottom = split_pane_heights(available, 0.30)
        self.assertGreater(top, 0.0)
        self.assertGreater(bottom, 0.0)
        self.assertLessEqual(top + bottom, available)
        self.assertAlmostEqual(
            top / bottom, MIN_PLOT3D_HEIGHT / MIN_TWO_D_HEIGHT, places=6
        )

    def test_a_pane_with_no_room_at_all_stays_positive(self) -> None:
        """ImPlot reads a zero height as "fill the window", so a pane too small to
        measure -- the first frame after launch reports a negative one -- must not
        come back as zero."""
        for available in (-49.0, 0.0, PANE_SPLITTER_THICKNESS):
            with self.subTest(available=available):
                top, bottom = split_pane_heights(available, 0.3)
                self.assertGreater(top, 0.0)
                self.assertGreater(bottom, 0.0)

    def test_the_default_split_gives_the_3d_view_the_larger_share(self) -> None:
        state = AppState()
        top, bottom = split_pane_heights(1000.0, state.two_d_pane_fraction)
        self.assertGreater(top, bottom)

    def test_an_untouched_splitter_returns_the_fraction_unchanged(self) -> None:
        state = AppState()
        seen: list[float] = []

        def gui() -> None:
            seen.append(
                draw_pane_splitter("##probe_splitter", state.two_d_pane_fraction, 800.0)
            )

        render_frames(gui)
        self.assertTrue(seen)
        for value in seen:
            self.assertAlmostEqual(value, state.two_d_pane_fraction, places=6)

    def test_dragging_the_splitter_moves_the_split(self) -> None:
        """End to end through real ImGui frames: press the band, drag, release.

        Input is queued with ``add_mouse_*_event`` because ImGui consumes it at
        NewFrame -- setting ``io.mouse_pos`` from inside the frame would land a
        frame late and never register as a press on the button.
        """
        available = 800.0
        usable = available - PANE_SPLITTER_THICKNESS
        step, drags = 10.0, 6
        run = {"frame": 0, "fraction": 0.30}
        active_frames: list[bool] = []

        def gui() -> None:
            run["frame"] += 1
            frame = run["frame"]
            top = imgui.get_cursor_screen_pos()
            run["fraction"] = draw_pane_splitter(
                "##drag_probe", run["fraction"], available
            )
            active_frames.append(imgui.is_item_active())

            io = imgui.get_io()
            inside_y = top.y + PANE_SPLITTER_THICKNESS * 0.5
            if frame == 1:
                io.add_mouse_pos_event(top.x + 50.0, inside_y)
            elif frame == 2:
                io.add_mouse_button_event(0, True)
            elif 3 <= frame < 3 + drags:
                io.add_mouse_pos_event(
                    top.x + 50.0, inside_y - step * (frame - 2)
                )
            elif frame == 3 + drags:
                io.add_mouse_button_event(0, False)

        render_frames(gui, frames=3 + drags + 3)

        self.assertTrue(any(active_frames), "the splitter never took the press")
        self.assertFalse(active_frames[-1], "the splitter never let go")
        # Dragging up grows the 2D share by the distance travelled, as a fraction.
        self.assertAlmostEqual(
            run["fraction"], 0.30 + (step * drags) / usable, places=4
        )

    def test_dragging_past_the_end_does_not_bank_fraction(self) -> None:
        """Clamped as it goes, so a drag that runs off the pane does not have to be
        dragged back through the surplus before anything moves."""
        available = 800.0
        run = {"frame": 0, "fraction": 0.30}

        def gui() -> None:
            run["frame"] += 1
            frame = run["frame"]
            top = imgui.get_cursor_screen_pos()
            run["fraction"] = draw_pane_splitter(
                "##overshoot_probe", run["fraction"], available
            )
            io = imgui.get_io()
            inside_y = top.y + PANE_SPLITTER_THICKNESS * 0.5
            if frame == 1:
                io.add_mouse_pos_event(top.x + 50.0, inside_y)
            elif frame == 2:
                io.add_mouse_button_event(0, True)
            elif 3 <= frame <= 8:
                # Far past the top of the pane, several times over.
                io.add_mouse_pos_event(top.x + 50.0, inside_y - 900.0 * (frame - 2))
            elif frame == 9:
                io.add_mouse_button_event(0, False)

        render_frames(gui, frames=12)
        self.assertLessEqual(run["fraction"], 0.95 + 1e-9)
        self.assertGreaterEqual(run["fraction"], 0.05 - 1e-9)


class InteractiveUpdateGateTests(unittest.TestCase):
    """Live re-energization pays for itself out of the frame rate, and stops when
    the frame rate is what it is costing."""

    def test_it_is_on_by_default(self) -> None:
        state = AppState()
        self.assertTrue(state.update_spin_energies_interactively)
        self.assertTrue(state.interactive_updates_live(60.0))

    def test_a_slow_frame_rate_pauses_it(self) -> None:
        state = AppState()
        # Written against the threshold rather than a rate that happens to sit
        # under it, so tuning the pair does not quietly stop testing the pause.
        self.assertFalse(
            state.interactive_updates_live(AUTO_SPIN_UPDATE_MIN_FPS - 1.0)
        )

    def test_it_resumes_only_once_comfortably_clear(self) -> None:
        """Two thresholds, or pausing frees exactly the time that caused the pause
        and the landscape rebuilds every other frame."""
        state = AppState()
        self.assertFalse(
            state.interactive_updates_live(AUTO_SPIN_UPDATE_MIN_FPS - 1.0)
        )
        # Back over the pause threshold, but not over the resume one: still paused.
        self.assertFalse(
            state.interactive_updates_live(AUTO_SPIN_UPDATE_MIN_FPS + 1.0)
        )
        self.assertTrue(state.interactive_updates_live(AUTO_SPIN_UPDATE_RESUME_FPS))
        # And once running, it holds down to the lower threshold rather than the
        # higher one.
        self.assertTrue(
            state.interactive_updates_live(AUTO_SPIN_UPDATE_MIN_FPS + 1.0)
        )

    def test_the_thresholds_leave_room_between_them(self) -> None:
        self.assertGreater(AUTO_SPIN_UPDATE_RESUME_FPS, AUTO_SPIN_UPDATE_MIN_FPS)

    def test_switching_it_off_beats_any_frame_rate(self) -> None:
        state = AppState()
        state.update_spin_energies_interactively = False
        self.assertFalse(state.interactive_updates_live(240.0))

    def test_no_measurement_yet_is_not_treated_as_slow(self) -> None:
        """ImGui reports 0 before it has measured anything; pausing on no evidence
        would make the first edits after launch silently stale."""
        state = AppState()
        self.assertTrue(state.interactive_updates_live(0.0))

    def test_a_paused_edit_marks_the_energies_stale(self) -> None:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        before = [config.energy for config in state.spin_landscape]
        for name, value in TILT_EDIT.items():
            setattr(state, name, value)
        # Patched rather than seeded: the gate re-reads the frame rate on the call
        # inside the edit, and another test in the run may have left a live ImGui
        # context behind for it to read a healthy rate from.
        with patch("quick_mag.quick_mag_ui.current_framerate", return_value=5.0):
            state.regenerate_focus_from_builder_if_changed()
        self.assertTrue(state.spin_energies_stale)
        self.assertEqual([c.energy for c in state.spin_landscape], before)

    def test_the_framerate_reader_is_safe_without_a_context(self) -> None:
        """The gate guards model-layer code the CLI and the tests also drive."""
        with patch(
            "quick_mag.quick_mag_ui.imgui.get_current_context", return_value=None
        ):
            self.assertEqual(current_framerate(), 0.0)

    def test_the_results_panel_renders_either_side_of_the_gate(self) -> None:
        """Including the paused readout, which only appears on one side of it."""
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.run_magnetic_structure_calculation(structure=state.focus)
        for framerate in (5.0, 120.0):
            with self.subTest(framerate=framerate):
                # The panel reads the module-level APP_STATE, so a locally built
                # state has to be swapped in for it to render the solved results
                # rather than the default ones.
                with patch("quick_mag.quick_mag_ui.APP_STATE", state), patch(
                    "quick_mag.quick_mag_ui.current_framerate", return_value=framerate
                ):
                    render_frames(gui_calculation_output)
        self.assertFalse(state.interactive_updates_live(5.0))


class CustomSpinPatternTests(unittest.TestCase):
    """Orderings entered by hand, as a plane family plus a sign string."""

    def _state(self, cells: int = 3) -> AppState:
        state = AppState()
        state.update_spin_energies_interactively = True
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        if cells != 3:
            state.perovskite_supercell_x = cells
            state.perovskite_supercell_y = cells
            state.perovskite_supercell_z = cells
            state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        return state

    def test_a_new_ordering_joins_the_landscape_and_is_selected(self) -> None:
        state = self._state(cells=4)
        before = len(state.displayed_spin_configs())
        self.assertTrue(state.add_custom_spin_pattern((0, 0, 1), "+++-"))
        configs = state.displayed_spin_configs()
        self.assertEqual(len(configs), before + 1)
        self.assertIn("(001) +++-", state.custom_spin_patterns)
        # Landed on what was just added, and it is reported as itself rather than as
        # the nearest canonical ordering: (001) +++- is F with one plane in four
        # flipped, and before it was in the candidate set it read as "F, 25% defects".
        selected = configs[state.selected_spin_config_index]
        match = state.match_for_config(selected)
        self.assertIsNotNone(match)
        self.assertEqual(match.pattern.label, "(001) +++-")
        self.assertAlmostEqual(match.concentration, 0.0, places=6)
        self.assertEqual(state.label_for_config(selected), "(001) +++-")

    def test_custom_orderings_are_scored_on_unit_moments(self) -> None:
        """The magnitude is inside J; scoring on formal moments is |mu|^2 too big.

        An Fe(3+) cell would come out 25x, which puts every custom ordering below
        the real ground state and reorders the whole landscape.
        """
        state = self._state(cells=4)
        ground = state.displayed_spin_configs()[0].energy
        state.add_custom_spin_pattern((0, 0, 1), "+++-")
        self.assertAlmostEqual(
            state.displayed_spin_configs()[0].energy, ground, places=9
        )
        added = dict(state.reference_configs)["(001) +++-"]
        self.assertTrue(
            np.allclose(np.abs(np.asarray(added.all_moments)), 1.0),
            "custom orderings must carry unit moments",
        )

    def test_an_ordering_the_cell_cannot_resolve_is_refused(self) -> None:
        """A period needs one plane each; folding it back would score a lie."""
        state = self._state(cells=3)
        self.assertFalse(state.add_custom_spin_pattern((0, 0, 1), "++--"))
        self.assertIn("too few", state.custom_pattern_message)
        self.assertEqual(state.custom_spin_patterns, [])

    def test_a_canonical_ordering_re_entered_is_named_not_duplicated(self) -> None:
        state = self._state()
        self.assertFalse(state.add_custom_spin_pattern((1, 1, 1), "+-"))
        self.assertIn("G", state.custom_pattern_message)
        self.assertEqual(state.custom_spin_patterns, [])

    def test_an_ordering_equivalent_to_a_listed_one_says_so(self) -> None:
        """(123) +- and C(b) are one ordering on a 3x3x3 cell; the list cannot show
        both, so the message has to."""
        state = self._state()
        before = len(state.displayed_spin_configs())
        self.assertTrue(state.add_custom_spin_pattern((1, 2, 3), "+-"))
        self.assertIn("the same ordering as C(b)", state.custom_pattern_message)
        self.assertEqual(len(state.displayed_spin_configs()), before)

    def test_bad_input_is_refused_with_a_reason(self) -> None:
        state = self._state()
        for miller, signs in (((0, 0, 1), ""), ((0, 0, 1), "+x-"), ((0, 0, 0), "+-")):
            with self.subTest(miller=miller, signs=signs):
                self.assertFalse(state.add_custom_spin_pattern(miller, signs))
                self.assertTrue(state.custom_pattern_message)
        self.assertEqual(state.custom_spin_patterns, [])

    def test_a_period_past_the_cap_is_refused(self) -> None:
        state = self._state(cells=4)
        long_pattern = "+-" * MAX_CUSTOM_PATTERN_PERIOD
        self.assertFalse(state.add_custom_spin_pattern((0, 0, 1), long_pattern))
        self.assertIn(str(MAX_CUSTOM_PATTERN_PERIOD), state.custom_pattern_message)

    def test_removing_an_ordering_takes_it_out_of_the_landscape(self) -> None:
        state = self._state(cells=4)
        before = len(state.displayed_spin_configs())
        state.add_custom_spin_pattern((0, 0, 1), "+++-")
        state.remove_custom_spin_pattern("(001) +++-")
        self.assertEqual(state.custom_spin_patterns, [])
        self.assertEqual(len(state.displayed_spin_configs()), before)

    def test_custom_orderings_survive_a_re_energization(self) -> None:
        """Stored as a pattern, not as moments, so an edit rescores rather than drops."""
        state = self._state(cells=4)
        state.add_custom_spin_pattern((0, 0, 1), "+++-")
        apply_builder_edits(state, perovskite_a=4.2)
        self.assertIn("(001) +++-", dict(state.reference_configs))

    def test_nothing_is_custom_by_default(self) -> None:
        self.assertEqual(AppState().custom_spin_patterns, [])

    def test_the_landscape_and_the_plot_agree_on_the_name(self) -> None:
        """The row the ordering was added as, and the category the plot draws it in.

        These are separately derived -- the row comes from the reference set, the
        category from the classifier -- so it takes an assertion to hold them
        together. Before the classifier saw custom patterns, the row read
        "(001) +++-" while the plot drew the same point as an F.
        """
        state = self._state(cells=4)
        state.add_custom_spin_pattern((0, 0, 1), "+++-")
        selected = state.displayed_spin_configs()[state.selected_spin_config_index]
        categories = spin_plot_categories(state)
        self.assertIn("(001) +++-", categories)
        self.assertEqual(
            spin_plot_category(state.label_for_config(selected), categories),
            "(001) +++-",
        )
        # Exactly matched, so the list's description carries no defect fraction.
        self.assertEqual(state.described_config(selected), "(001) +++-")

    def test_a_custom_ordering_is_reported_wherever_it_appears(self) -> None:
        """One classifier behind the list, the plot labels and the 3D badge."""
        state = self._state(cells=4)
        state.add_custom_spin_pattern((0, 0, 1), "+++-")
        position = state.selected_spin_config_index
        self.assertEqual(
            state.spin_classification_labels()[position], "(001) +++-"
        )
        self.assertEqual(
            state.spin_classification_descriptions()[position], "(001) +++-"
        )

    def test_removing_an_ordering_stops_it_being_reported(self) -> None:
        """The candidate set has to shrink again, or the name outlives the row."""
        state = self._state(cells=4)
        state.add_custom_spin_pattern((0, 0, 1), "+++-")
        target = state.displayed_spin_configs()[state.selected_spin_config_index]
        state.remove_custom_spin_pattern("(001) +++-")
        self.assertNotIn("(001) +++-", spin_plot_categories(state))
        # Scored on its own now, since it is no longer in the displayed set: back to
        # the nearest canonical ordering, which is F with one plane in four flipped.
        match = state.match_for_config(target)
        self.assertEqual(match.pattern.label, "F")
        self.assertAlmostEqual(match.concentration, 0.25, places=6)

    def test_canonical_orderings_keep_their_names_and_colours(self) -> None:
        """Adding a custom pattern must not rename the landscape around it."""
        state = self._state(cells=4)
        before = state.spin_classification_labels()
        state.add_custom_spin_pattern((0, 0, 1), "+++-")
        after = state.spin_classification_labels()
        for label in before:
            self.assertIn(label, after)
        self.assertEqual(spin_class_color(state, "G"), SPIN_CLASS_COLORS["G"])

    def test_custom_categories_get_their_own_colours(self) -> None:
        """Distinct from each other and from every canonical ordering's colour."""
        state = self._state(cells=4)
        state.add_custom_spin_pattern((0, 0, 1), "+++-")
        state.add_custom_spin_pattern((0, 1, 0), "+++-")
        colors = [
            spin_class_color(state, label) for label in state.custom_spin_patterns
        ]
        self.assertEqual(len(set(colors)), len(colors))
        for color in colors:
            self.assertNotIn(color, set(SPIN_CLASS_COLORS.values()))

    def test_a_custom_colour_does_not_move_when_another_is_added(self) -> None:
        """Keyed on position in the user's list, which only ever grows at the end."""
        state = self._state(cells=4)
        state.add_custom_spin_pattern((0, 0, 1), "+++-")
        first = spin_class_color(state, "(001) +++-")
        state.add_custom_spin_pattern((0, 1, 0), "+++-")
        self.assertEqual(spin_class_color(state, "(001) +++-"), first)

    def test_an_unlisted_label_still_falls_back_to_other(self) -> None:
        state = self._state()
        self.assertEqual(
            spin_plot_category("(001) +-+-+-+-", spin_plot_categories(state)), "Other"
        )
        self.assertEqual(
            spin_class_color(state, "(001) +-+-+-+-"), SPIN_CLASS_COLORS["Other"]
        )

    def test_the_scatter_renders_with_a_custom_ordering_in_it(self) -> None:
        state = self._state(cells=4)
        state.add_custom_spin_pattern((0, 0, 1), "+++-")
        with patch("quick_mag.quick_mag_ui.APP_STATE", state):
            render_frames(lambda: gui_two_d_pane(state))


class MagnetizationReportingTests(unittest.TestCase):
    """M as a physical moment per cell, not as a count of unit-moment sites."""

    def _state(self, cells: int = 3) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        if cells != 3:
            state.perovskite_supercell_x = cells
            state.perovskite_supercell_y = cells
            state.perovskite_supercell_z = cells
            state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        return state

    def _labelled(self, state: AppState, label: str):
        for config in state.displayed_spin_configs():
            if state.label_for_config(config) == label:
                return config
        self.fail(f"no {label} configuration in the landscape")

    def test_ferromagnetic_reports_the_high_spin_moment(self) -> None:
        """Default LaFeO3: every B site is a high-spin Fe(3+), 5 muB, so F is 5/cell."""
        state = self._state()
        moment, unit = state.config_magnetization(self._labelled(state, "F"))
        self.assertAlmostEqual(moment, 5.0, places=9)
        self.assertEqual(unit, "μB/cell")

    def test_the_moment_does_not_grow_with_the_supercell(self) -> None:
        """The whole point of per-cell: the solver's own M counts sites and does."""
        for cells in (2, 3, 4):
            with self.subTest(cells=cells):
                state = self._state(cells)
                config = self._labelled(state, "F")
                self.assertAlmostEqual(
                    state.config_magnetization(config)[0], 5.0, places=9
                )
                # The number this replaces: one per magnetic site, and it does grow.
                self.assertAlmostEqual(config.magnetization, cells**3, places=9)

    def test_a_compensated_ordering_is_zero(self) -> None:
        state = self._state(cells=2)
        self.assertAlmostEqual(
            state.config_magnetization(self._labelled(state, "G"))[0], 0.0, places=9
        )

    def test_an_odd_cell_leaves_one_site_uncompensated(self) -> None:
        """3x3x3 G cannot balance: 27 sites, so one 5 muB spin is left over."""
        state = self._state(cells=3)
        moment, _ = state.config_magnetization(self._labelled(state, "G"))
        self.assertAlmostEqual(abs(moment), 5.0 / 27.0, places=9)

    def test_the_sign_is_kept(self) -> None:
        state = self._state(cells=3)
        signs = {
            np.sign(state.config_magnetization(config)[0])
            for config in state.displayed_spin_configs()
        }
        self.assertIn(-1.0, signs)
        self.assertIn(1.0, signs)

    def test_the_cell_count_is_the_b_site_grid(self) -> None:
        state = self._state(cells=4)
        self.assertEqual(state.unit_cell_count(), 64)

    def test_without_an_assignment_it_falls_back_and_says_so(self) -> None:
        """An empty unit marks the number as the solver's own, not a moment."""
        state = self._state()
        config = self._labelled(state, "F")
        state.magnetic_oxidation_assignments = []
        moment, unit = state.config_magnetization(config)
        self.assertEqual(unit, "")
        self.assertAlmostEqual(moment, config.magnetization, places=9)

    def test_without_a_cell_grid_it_reports_the_whole_structure(self) -> None:
        state = self._state()
        config = self._labelled(state, "F")
        with patch.object(AppState, "unit_cell_count", return_value=0):
            moment, unit = state.config_magnetization(config)
        self.assertEqual(unit, "μB")
        # 27 high-spin Fe(3+), undivided.
        self.assertAlmostEqual(moment, 27 * 5.0, places=9)

    def test_the_panel_renders_with_the_reordered_sections(self) -> None:
        """List, then the selection, then save, then the custom-ordering tool."""
        state = self._state()
        state.run_magnetic_structure_calculation(structure=state.focus)
        state.add_custom_spin_pattern((0, 0, 1), "+-")
        with patch("quick_mag.quick_mag_ui.APP_STATE", state):
            render_frames(gui_calculation_output)

    def test_a_shared_basis_gives_the_same_answer(self) -> None:
        """The list passes one basis into every row rather than re-reading it."""
        state = self._state()
        basis = state.magnetization_basis()
        for config in state.displayed_spin_configs():
            self.assertEqual(
                state.config_magnetization(config, basis),
                state.config_magnetization(config),
            )


class TwoDAxisPaddingTests(unittest.TestCase):
    """Where the axes sit relative to the data in the 2D pane."""

    def test_the_exchange_floor_clears_the_foot_of_the_bars(self) -> None:
        """Every bar stands on zero, so zero cannot be the axis line."""
        low, high = padded_two_d_limits(0.0, 12.0, bottom=EXCHANGE_BOTTOM_HEADROOM)
        self.assertLess(low, 0.0)
        self.assertAlmostEqual(low, -12.0 * EXCHANGE_BOTTOM_HEADROOM, places=9)
        # Enough to read as a gap rather than as anti-aliasing: on a 300 px plot
        # this is ~9 px of clear space under the baseline.
        self.assertGreater(abs(low) / (high - low), 0.05)

    def test_the_exchange_floor_is_lower_than_the_default(self) -> None:
        self.assertGreater(EXCHANGE_BOTTOM_HEADROOM, TWO_D_BOTTOM_HEADROOM)
        wide = padded_two_d_limits(0.0, 12.0, bottom=EXCHANGE_BOTTOM_HEADROOM)
        narrow = padded_two_d_limits(0.0, 12.0)
        self.assertLess(wide[0], narrow[0])
        # Only the floor moves; the headroom the corner pickers need is unchanged.
        self.assertAlmostEqual(wide[1], narrow[1], places=9)

    def test_negative_couplings_keep_their_margin_too(self) -> None:
        """A mixed FM/AFM structure pads below the lowest bar tip, not below zero."""
        low, _ = padded_two_d_limits(-4.0, 12.0, bottom=EXCHANGE_BOTTOM_HEADROOM)
        self.assertLess(low, -4.0)

    def test_a_flat_landscape_still_gets_a_span(self) -> None:
        """Every energy equal -- the padding is a fraction of a zero span."""
        for bottom in (TWO_D_BOTTOM_HEADROOM, EXCHANGE_BOTTOM_HEADROOM):
            with self.subTest(bottom=bottom):
                low, high = padded_two_d_limits(0.0, 0.0, bottom=bottom)
                self.assertLess(low, high)

    def test_the_energy_scatter_keeps_its_own_padding(self) -> None:
        """The wider floor is the coupling plot's; the scatter did not ask for it."""
        low, high = padded_two_d_limits(0.0, 1.0)
        self.assertAlmostEqual(low, -TWO_D_BOTTOM_HEADROOM, places=9)
        self.assertAlmostEqual(high, 1.0 + TWO_D_TOP_HEADROOM, places=9)


class ExchangeSelectionGatingTests(unittest.TestCase):
    """The coupling decorations belong to the coupling plot, and only to it."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        state.selected_site_index = state.magnetic_site_indices[0]
        return state

    def test_the_energy_plot_leaves_the_structure_alone(self) -> None:
        state = self._state()
        state.two_d_plot_index = 0
        self.assertEqual(exchange_selection_site(state), -1)

    def test_the_coupling_plot_picks_the_selection_up(self) -> None:
        state = self._state()
        state.two_d_plot_index = 1
        self.assertEqual(exchange_selection_site(state), state.selected_site_index)

    def test_a_non_magnetic_selection_decorates_nothing(self) -> None:
        """An O site can still be selected from the per-site list; it has no network."""
        state = self._state()
        state.two_d_plot_index = 1
        state.selected_site_index = (
            state.magnetic_analysis_structure.element_symbols().index("O")
        )
        self.assertEqual(exchange_selection_site(state), -1)

    def test_nothing_selected_decorates_nothing(self) -> None:
        state = self._state()
        state.two_d_plot_index = 1
        state.selected_site_index = -1
        self.assertEqual(exchange_selection_site(state), -1)


class SiteHoverTooltipTests(unittest.TestCase):
    """The per-site oxidation/moment readout, moved onto the atom itself."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.run_magnetic_structure_calculation(structure=state.focus)
        return state

    def test_the_tooltip_matches_the_row_the_list_used_to_show(self) -> None:
        state = self._state()
        structure = state.magnetic_analysis_structure
        assignment = state.selected_oxidation_assignment()
        moments = state.selected_spin_moments_for_structure(structure)
        rows = oxidation_site_rows(structure, assignment, site_moments=moments)
        for site in (0, structure.atom_count // 2, structure.atom_count - 1):
            with self.subTest(site=site):
                self.assertEqual(site_hover_tooltip(state, structure, site), rows[site])

    def test_every_atom_has_something_to_say(self) -> None:
        """Not only the magnetic ones -- an O site has an oxidation state too."""
        state = self._state()
        structure = state.magnetic_analysis_structure
        symbols = structure.element_symbols()
        for element in ("La", "Fe", "O"):
            site = symbols.index(element)
            with self.subTest(element=element):
                tooltip = site_hover_tooltip(state, structure, site)
                self.assertIn(element, tooltip)
                self.assertIn("ox=", tooltip)
                self.assertIn("m=", tooltip)

    def test_out_of_range_sites_say_nothing(self) -> None:
        state = self._state()
        structure = state.magnetic_analysis_structure
        self.assertEqual(site_hover_tooltip(state, structure, -1), "")
        self.assertEqual(site_hover_tooltip(state, structure, 10_000), "")

    def test_the_tooltip_survives_a_missing_assignment(self) -> None:
        """Still names the atom rather than looking like a broken tooltip."""
        state = self._state()
        structure = state.magnetic_analysis_structure
        state.magnetic_oxidation_assignments = []
        tooltip = site_hover_tooltip(state, structure, 0)
        self.assertTrue(tooltip)
        self.assertNotIn("ox=", tooltip)


class ExchangePickRestrictionTests(unittest.TestCase):
    """Once an atom is selected, only what it couples to can be clicked."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        state.two_d_plot_index = 1
        return state

    def test_candidates_are_the_selection_and_its_partners(self) -> None:
        state = self._state()
        site = state.magnetic_site_indices[len(state.magnetic_site_indices) // 2]
        rendered = state.rendered_structure()
        coords = rendered.cartesian_coords
        paths = exchange_render_paths(state, rendered, coords, site, use_cartesian=True)
        candidates = exchange_pick_candidates(coords, paths)
        analysis = state.magnetic_analysis_structure
        sites = {
            source_site_for_render_index(rendered, analysis, index)
            for index in candidates
        }
        partners = {
            pair.site_j if pair.site_i == site else pair.site_i
            for pair in exchange_pairs_for_site(state.magnetic_pair_couplings, site)
        }
        self.assertEqual(sites, partners | {site})

    def test_the_bridging_ligands_are_not_clickable(self) -> None:
        """They sit on every path but carry no couplings; selecting one empties the plot."""
        state = self._state()
        site = state.magnetic_site_indices[len(state.magnetic_site_indices) // 2]
        rendered = state.rendered_structure()
        coords = rendered.cartesian_coords
        paths = exchange_render_paths(state, rendered, coords, site, use_cartesian=True)
        candidates = exchange_pick_candidates(coords, paths)
        symbols = rendered.element_symbols()
        self.assertEqual({symbols[index] for index in candidates}, {"Fe"})
        # The ligands are still drawn prominently -- they are just not targets.
        prominent = exchange_prominent_render_atoms(coords, paths)
        self.assertIn("O", {symbols[index] for index in prominent})
        self.assertLess(len(candidates), len(prominent))

    def test_every_candidate_has_couplings_to_show(self) -> None:
        state = self._state()
        site = state.magnetic_site_indices[0]
        rendered = state.rendered_structure()
        coords = rendered.cartesian_coords
        paths = exchange_render_paths(state, rendered, coords, site, use_cartesian=True)
        analysis = state.magnetic_analysis_structure
        for index in exchange_pick_candidates(coords, paths):
            partner = source_site_for_render_index(rendered, analysis, index)
            self.assertTrue(
                exchange_pairs_for_site(state.magnetic_pair_couplings, partner)
            )

    def test_no_paths_means_nothing_to_click(self) -> None:
        state = self._state()
        rendered = state.rendered_structure()
        self.assertEqual(exchange_pick_candidates(rendered.cartesian_coords, []), [])

    def test_a_bar_names_the_lower_indexed_atom_of_its_pair(self) -> None:
        """Pairs are stored site_i < site_j, so site_i is the atom a bar click means.

        Naming the same end every time is what makes it predictable; picking "the
        one that is not selected" would send the same bar to different places
        depending on where you already were.
        """
        state = self._state()
        for pair in state.magnetic_pair_couplings[:8]:
            with self.subTest(pair=(pair.site_i, pair.site_j)):
                self.assertLess(pair.site_i, pair.site_j)
                self.assertEqual(min(pair.site_i, pair.site_j), pair.site_i)

    def test_the_projection_cache_key_tracks_every_view_change(self) -> None:
        """Reprojecting is the expensive part; a stale key would freeze the hover."""
        limits = (-4.0, 4.0, -4.0, 4.0, -4.0, 4.0)
        rotation = (0.0, 0.0, 0.0, 1.0)
        corner = imgui.ImVec2(0.0, 0.0)
        far = imgui.ImVec2(100.0, 100.0)
        base = view_projection_key(limits, rotation, 1.0, corner, far)
        self.assertEqual(base, view_projection_key(limits, rotation, 1.0, corner, far))
        for label, other in (
            ("limits", view_projection_key(
                (-5.0, 4.0, -4.0, 4.0, -4.0, 4.0), rotation, 1.0, corner, far)),
            ("rotation", view_projection_key(
                limits, (0.1, 0.0, 0.0, 1.0), 1.0, corner, far)),
            ("zoom", view_projection_key(limits, rotation, 1.5, corner, far)),
            ("rect", view_projection_key(
                limits, rotation, 1.0, corner, imgui.ImVec2(200.0, 100.0))),
        ):
            with self.subTest(changed=label):
                self.assertNotEqual(base, other)


class ExchangeIonLabelTests(unittest.TestCase):
    """The 3D hover names the ion; the bars name the atom."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        return state

    def test_the_hover_label_carries_the_oxidation_state(self) -> None:
        state = self._state()
        site = state.magnetic_site_indices[0]
        self.assertEqual(exchange_ion_label(state, site), "Fe(3+)")

    def test_the_bar_label_still_carries_the_index(self) -> None:
        """Two sites of the same ion have to be told apart on the axis."""
        state = self._state()
        site = state.magnetic_site_indices[0]
        self.assertEqual(exchange_site_label(state, site), f"Fe{site}")

    def test_the_hover_label_falls_back_without_an_assignment(self) -> None:
        state = self._state()
        site = state.magnetic_site_indices[0]
        state.magnetic_oxidation_assignments = []
        self.assertEqual(
            exchange_ion_label(state, site), exchange_site_label(state, site)
        )


class TwoDPaneTests(unittest.TestCase):
    """The 2D pane's plot picker and configuration picker."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        return state

    def test_the_spin_plot_is_the_default(self) -> None:
        self.assertEqual(AppState().two_d_plot_index, 0)
        self.assertEqual(TWO_D_PLOT_NAMES[0], "Spin energies")

    def test_both_plots_render_in_a_real_frame(self) -> None:
        """Exercises begin_plot/end_plot balance and the corner overlays."""
        state = self._state()
        for index in range(len(TWO_D_PLOT_NAMES)):
            state.two_d_plot_index = index
            render_frames(lambda: gui_two_d_pane(state))

    def test_the_exchange_plot_renders_while_filtered(self) -> None:
        state = self._state()
        state.two_d_plot_index = 1
        state.selected_site_index = state.magnetic_site_indices[0]
        render_frames(lambda: gui_two_d_pane(state))


class MagneticPickCandidateTests(unittest.TestCase):
    """Click targets for selecting a magnetic site in the 3D view."""

    def _state(self) -> AppState:
        state = AppState()
        state.sync_builder_binding()
        state.regenerate_focus_from_builder_if_changed()
        state.ensure_spin_baseline()
        return state

    def test_candidates_are_exactly_the_magnetic_sites_and_their_images(self) -> None:
        state = self._state()
        rendered = state.rendered_structure()
        candidates = magnetic_pick_candidates(state, rendered)
        self.assertTrue(candidates)
        expected = {
            index
            for site in state.magnetic_site_indices
            for index in highlighted_render_indices(
                rendered, state.magnetic_analysis_structure, site
            )
        }
        self.assertEqual(set(candidates), expected)

    def test_a_site_with_no_couplings_is_not_a_candidate(self) -> None:
        """Selecting it would replace the plot with an empty pane."""
        state = self._state()
        rendered = state.rendered_structure()
        analysis = state.magnetic_analysis_structure
        orphan = state.magnetic_site_indices[0]
        state.magnetic_pair_couplings = [
            pair
            for pair in state.magnetic_pair_couplings
            if orphan not in (pair.site_i, pair.site_j)
        ]
        state._exchange_generation += 1
        candidates = set(magnetic_pick_candidates(state, rendered))
        for image in highlighted_render_indices(rendered, analysis, orphan):
            self.assertNotIn(image, candidates)
        # The rest are untouched.
        self.assertTrue(candidates)

    def test_candidates_exclude_the_non_magnetic_atoms(self) -> None:
        """The cost of the per-frame projection is why this is not every atom."""
        state = self._state()
        rendered = state.rendered_structure()
        candidates = magnetic_pick_candidates(state, rendered)
        self.assertLess(len(candidates), rendered.atom_count)
        symbols = rendered.element_symbols()
        self.assertEqual({symbols[index] for index in candidates}, {"Fe"})

    def test_no_analysis_means_no_candidates(self) -> None:
        """Nothing analysed, nothing to pick -- and no attempt to match against None."""
        state = self._state()
        rendered = state.rendered_structure()
        state.magnetic_analysis_structure = None
        self.assertEqual(magnetic_pick_candidates(state, rendered), [])

    def test_no_magnetic_sites_means_no_candidates(self) -> None:
        state = self._state()
        rendered = state.rendered_structure()
        state.magnetic_site_indices = []
        self.assertEqual(magnetic_pick_candidates(state, rendered), [])

    def test_the_candidate_list_is_memoized(self) -> None:
        state = self._state()
        rendered = state.rendered_structure()
        first = magnetic_pick_candidates(state, rendered)
        self.assertIs(magnetic_pick_candidates(state, rendered), first)

    def test_aiming_at_a_magnetic_atom_in_a_live_plot_selects_its_site(self) -> None:
        """The whole 3D-pick chain, through ImPlot3D's own projection.

        Projection, nearest-atom, and the render-index-to-site inverse together:
        aiming at each candidate's own pixel has to land on the site it images.
        """
        state = self._state()
        rendered = state.rendered_structure()
        analysis = state.magnetic_analysis_structure
        candidates = magnetic_pick_candidates(state, rendered)
        coords = rendered.cartesian_coords
        limits = compute_plot_box_limits(coords)
        depths = view_space_depth(
            coords[candidates], limits, state.structure_rotation
        )
        outcome: dict = {}

        def gui() -> None:
            if implot3d.begin_plot("##magnetic_pick_probe", imgui.ImVec2(600.0, 600.0)):
                implot3d.setup_axes_limits(*limits, implot3d.Cond_.always)
                implot3d.setup_box_rotation(
                    implot3d.Quat(*state.structure_rotation),
                    False,
                    implot3d.Cond_.always,
                )
                pixels = candidate_pixels(coords, candidates)
                hits = 0
                for position, atom in enumerate(candidates):
                    picked = nearest_picked_atom(
                        pixels, candidates, depths, tuple(pixels[position])
                    )
                    expected = source_site_for_render_index(rendered, analysis, atom)
                    if (
                        picked >= 0
                        and source_site_for_render_index(rendered, analysis, picked)
                        == expected
                        and expected in state.magnetic_site_indices
                    ):
                        hits += 1
                outcome["hits"] = hits
                outcome["miss"] = nearest_picked_atom(
                    pixels, candidates, depths, (-500.0, -500.0)
                )
                implot3d.end_plot()

        render_frames(gui)
        self.assertEqual(outcome.get("hits"), len(candidates))
        self.assertEqual(outcome.get("miss"), -1)
