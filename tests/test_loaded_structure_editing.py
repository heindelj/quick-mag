"""What a structure loaded from a file can and cannot be edited into.

Loading used to switch the whole builder off. It now switches off only the parts
that speak about a perovskite the builder generated -- site roles, octahedral
tilts, grid-addressed defects -- and routes the rest through the cell editor,
which strains the coordinates the file actually carried instead of regenerating
them from parameters the file does not have.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The UI module imports imgui-bundle (the optional ``ui`` extra). Skip this whole
# module when it is not installed so the core suite still runs without it.
pytest.importorskip("imgui_bundle")

from quick_mag.quick_mag_ui import AppState  # noqa: E402
from quick_mag.structure import ChemicalStructure, SavedSpinConfiguration  # noqa: E402


TRICLINIC = np.array(
    [
        [4.10, 0.35, -0.20],
        [0.60, 4.35, 0.15],
        [-0.25, 0.40, 4.60],
    ]
)


def loaded_structure(lattice: np.ndarray = TRICLINIC, count: int = 6) -> ChemicalStructure:
    """A structure shaped like one that came off disk: no generation parameters."""
    rng = np.random.default_rng(0)
    fractional = rng.random((count, 3))
    return ChemicalStructure(
        name="loaded",
        lattice=lattice,
        cartesian_coords=fractional @ lattice,
        atomic_labels=["Fe"] * (count // 2) + ["O"] * (count - count // 2),
        magnetic_moments=np.zeros((count, 3)),
    )


def state_focused_on_loaded() -> tuple[AppState, ChemicalStructure]:
    state = AppState()
    structure = loaded_structure()
    state.structures.append(structure)
    state.set_focus(structure)
    state.sync_active_structure()
    # The binding baselines on the first change-check, so an edit made after this
    # is seen as an edit rather than as the bind itself.
    state.apply_cell_edits_if_changed()
    return state, structure


class CapabilityTests(unittest.TestCase):
    def test_generated_structure_keeps_the_builder_and_not_the_cell_editor(self) -> None:
        state = AppState()  # opens focused on a generated structure
        self.assertTrue(state.composition_editing_available())
        self.assertTrue(state.defect_editing_available())
        self.assertFalse(state.cell_editing_available())

    def test_loaded_structure_gets_the_cell_editor_and_not_the_builder(self) -> None:
        state, _ = state_focused_on_loaded()
        self.assertTrue(state.cell_editing_available())
        self.assertFalse(state.composition_editing_available())
        self.assertFalse(state.defect_editing_available())
        self.assertFalse(state.tilt_editing_available())

    def test_every_disabled_section_says_why(self) -> None:
        state, _ = state_focused_on_loaded()
        for section in ("composition", "tilt", "defects"):
            self.assertTrue(state.unavailable_reason(section), section)

    def test_a_generated_structure_needs_no_explanation(self) -> None:
        state = AppState()
        for section in ("composition", "tilt", "defects"):
            self.assertEqual(state.unavailable_reason(section), "")


class BindingTests(unittest.TestCase):
    def test_focusing_a_loaded_structure_seeds_the_cell_fields(self) -> None:
        state, structure = state_focused_on_loaded()
        self.assertIs(state._cell_bound_structure, structure)
        self.assertAlmostEqual(state.cell_a, float(np.linalg.norm(structure.lattice[0])))
        self.assertAlmostEqual(state.cell_c, float(np.linalg.norm(structure.lattice[2])))

    def test_focusing_a_generated_structure_clears_the_binding(self) -> None:
        state, _ = state_focused_on_loaded()
        state.set_focus(state.structures[0])
        state.sync_active_structure()
        self.assertIsNone(state._cell_bound_structure)

    def test_binding_alone_does_not_move_the_structure(self) -> None:
        """The bind must not read as an edit -- otherwise loading strains on sight."""
        state, structure = state_focused_on_loaded()
        before = structure.cartesian_coords.copy()
        state.apply_cell_constraints()
        state.apply_cell_edits_if_changed()
        self.assertTrue(np.allclose(structure.cartesian_coords, before))


class CellEditTests(unittest.TestCase):
    def test_editing_a_length_strains_at_fixed_fractional_coordinates(self) -> None:
        state, structure = state_focused_on_loaded()
        fractional = structure.fractional_coords.copy()
        state.cell_a *= 1.2
        state.apply_cell_constraints()
        state.apply_cell_edits_if_changed()
        self.assertAlmostEqual(
            float(np.linalg.norm(structure.lattice[0])), state.cell_a, places=9
        )
        self.assertTrue(np.allclose(structure.fractional_coords, fractional))

    def test_a_strain_keeps_saved_spin_configurations(self) -> None:
        """The difference from a builder regeneration: no atom is added or removed."""
        state, structure = state_focused_on_loaded()
        structure.spin_configurations.append(
            SavedSpinConfiguration(magnetic_moments=np.zeros((structure.atom_count, 3)))
        )
        state.cell_b *= 1.1
        state.apply_cell_constraints()
        state.apply_cell_edits_if_changed()
        self.assertEqual(len(structure.spin_configurations), 1)
        self.assertEqual(structure.atom_count, 6)

    def test_the_aspect_lock_drives_b_and_c_from_a(self) -> None:
        state, _ = state_focused_on_loaded()
        state.cell_lock_aspect = True
        state.capture_cell_aspect_ratio()
        ratio_b = state.cell_b / state.cell_a
        ratio_c = state.cell_c / state.cell_a
        state.cell_a *= 2.0
        state.apply_cell_constraints()
        self.assertAlmostEqual(state.cell_b / state.cell_a, ratio_b)
        self.assertAlmostEqual(state.cell_c / state.cell_a, ratio_c)

    def test_an_angle_edit_leaves_the_structure_oriented_as_it_was(self) -> None:
        state, structure = state_focused_on_loaded()
        a_axis = structure.lattice[0].copy()
        state.cell_alpha += 4.0
        state.apply_cell_constraints()
        state.apply_cell_edits_if_changed()
        # alpha is the b-c angle, so the a axis has no business moving.
        self.assertTrue(np.allclose(structure.lattice[0], a_axis))

    def test_impossible_angles_report_and_change_nothing(self) -> None:
        state, structure = state_focused_on_loaded()
        before = structure.cartesian_coords.copy()
        state.cell_alpha, state.cell_beta, state.cell_gamma = 20.0, 20.0, 170.0
        state.apply_cell_constraints()
        state.apply_cell_edits_if_changed()
        self.assertTrue(state.cell_message)
        self.assertTrue(np.allclose(structure.cartesian_coords, before))

    def test_a_refused_edit_is_not_retried_every_frame(self) -> None:
        state, structure = state_focused_on_loaded()
        state.cell_alpha, state.cell_beta, state.cell_gamma = 20.0, 20.0, 170.0
        state.apply_cell_constraints()
        state.apply_cell_edits_if_changed()
        signature = state._cell_applied_sig
        state.apply_cell_edits_if_changed()
        self.assertEqual(state._cell_applied_sig, signature)

    def test_the_cell_editor_ignores_a_generated_focus(self) -> None:
        state = AppState()
        generated = state.focus
        assert generated is not None
        before = generated.cartesian_coords.copy()
        state.cell_a *= 2.0
        state.apply_cell_edits_if_changed()
        self.assertTrue(np.allclose(generated.cartesian_coords, before))


class InvalidationTests(unittest.TestCase):
    def test_a_strain_re_energizes_saved_configurations(self) -> None:
        """Their moments survive the strain; the energies recorded beside them must not."""
        state, structure = state_focused_on_loaded()
        state.magnetic_site_indices = [0, 1, 2]
        state.magnetic_j_matrix = np.array(
            [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
        )
        moments = np.zeros((structure.atom_count, 3))
        moments[:3, 2] = [1.0, -1.0, 1.0]
        structure.spin_configurations.append(
            SavedSpinConfiguration(magnetic_moments=moments, energy=-999.0)
        )
        state.re_energize_saved_configurations(structure)
        self.assertNotAlmostEqual(structure.spin_configurations[0].energy, -999.0)

    def test_re_energizing_leaves_a_mismatched_configuration_alone(self) -> None:
        state, structure = state_focused_on_loaded()
        state.magnetic_site_indices = [0, 1]
        state.magnetic_j_matrix = np.array([[0.0, 1.0], [1.0, 0.0]])
        structure.spin_configurations.append(
            SavedSpinConfiguration(magnetic_moments=np.zeros((3, 3)), energy=-42.0)
        )
        state.re_energize_saved_configurations(structure)
        self.assertAlmostEqual(structure.spin_configurations[0].energy, -42.0)

    def test_toggling_periodicity_marks_the_energies_stale(self) -> None:
        """The checkbox writes straight onto the structure, so it must invalidate."""
        state, structure = state_focused_on_loaded()
        state.update_spin_energies_interactively = False
        state.spin_energies_stale = False
        structure.is_periodic = not structure.is_periodic
        state.invalidate_after_geometry_change(structure)
        self.assertTrue(state.spin_energies_stale)


class TilingTests(unittest.TestCase):
    def test_tiling_replicates_and_rebinds(self) -> None:
        state, structure = state_focused_on_loaded()
        state.cell_tile_x, state.cell_tile_y, state.cell_tile_z = 2, 1, 3
        state.tile_focus()
        self.assertEqual(structure.atom_count, 36)
        self.assertAlmostEqual(
            state.cell_a, float(np.linalg.norm(structure.lattice[0])), places=9
        )
        # The repeat counts go back to 1 so the next Tile does not compound silently.
        self.assertEqual(
            (state.cell_tile_x, state.cell_tile_y, state.cell_tile_z), (1, 1, 1)
        )

    def test_tiling_drops_saved_spin_configurations(self) -> None:
        """Their per-atom moments no longer match the atom list, unlike a strain."""
        state, structure = state_focused_on_loaded()
        structure.spin_configurations.append(
            SavedSpinConfiguration(magnetic_moments=np.zeros((structure.atom_count, 3)))
        )
        state.cell_tile_x = 2
        state.tile_focus()
        self.assertEqual(structure.spin_configurations, [])

    def test_tiling_leaves_the_energies_stale_while_updates_are_paused(self) -> None:
        """Tiling is what makes the exchange rebuild expensive, so it honours the gate."""
        state, _ = state_focused_on_loaded()
        state.update_spin_energies_interactively = False
        state.cell_tile_x = 2
        state.tile_focus()
        self.assertTrue(state.spin_energies_stale)

    def test_tiling_re_baselines_while_updates_are_live(self) -> None:
        """And when it does re-baseline, it must not also claim to be stale."""
        state, _ = state_focused_on_loaded()
        state.update_spin_energies_interactively = True
        state.cell_tile_x = 2
        state.tile_focus()
        self.assertFalse(state.spin_energies_stale)

    def test_tiling_by_one_says_so_and_does_nothing(self) -> None:
        state, structure = state_focused_on_loaded()
        state.tile_focus()
        self.assertEqual(structure.atom_count, 6)
        self.assertTrue(state.cell_message)

    def test_tiling_ignores_a_generated_focus(self) -> None:
        state = AppState()
        generated = state.focus
        assert generated is not None
        before = generated.atom_count
        state.cell_tile_x = 2
        state.tile_focus()
        self.assertEqual(generated.atom_count, before)


if __name__ == "__main__":
    unittest.main()
