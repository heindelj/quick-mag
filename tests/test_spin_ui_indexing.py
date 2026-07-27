from __future__ import annotations

import sys
import unittest
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
from quick_mag.quick_mag_ui import (  # noqa: E402
    AppState,
    spin_alignment_edge_segments,
    structure_with_moments,
)
from quick_mag.magnetic_moments import OxidationStateAssignment  # noqa: E402
from quick_mag.spin_solver_np import SpinConfig  # noqa: E402
from quick_mag.structure import SavedSpinConfiguration  # noqa: E402


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
        state.builder_save_name = "Saved LaFeO3"
        state.save_builder_structure()
        source_structure = state.ungrouped_structures[-1]
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

    def test_spin_solver_results_always_include_canonical_afgc_patterns(self) -> None:
        state = AppState()
        structure = state.generated_chemical_structure()
        build = state.generated_perovskite()
        params = structure.generation_parameters
        assert params is not None

        b_indices = np.asarray(build.b_site_indices, dtype=int)
        site_moments = np.zeros(structure.atom_count, dtype=float)
        site_moments[b_indices] = 5.0
        assignment = OxidationStateAssignment(
            site_oxidation_states=np.zeros(structure.atom_count, dtype=int),
            magnetic_moments=site_moments,
            total_energy=0.0,
            distributions={"Fe": {3: int(len(b_indices))}},
        )

        state.set_focus(structure)
        state.magnetic_result_structure = structure
        state.magnetic_oxidation_assignments = [assignment]
        state.magnetic_site_indices = b_indices.tolist()
        state.magnetic_j_matrix = np.eye(len(b_indices), dtype=float)

        ferromagnetic = np.full(len(b_indices), 5.0, dtype=float)
        filtered_solver_states = [
            SpinConfig(
                energy=float(-0.5 * np.sum(ferromagnetic**2)),
                all_moments=ferromagnetic,
                magnetization=float(np.sum(ferromagnetic)),
                n_unpaired=float(np.sum(np.abs(ferromagnetic))),
            )
        ]

        with patch(
            "quick_mag.quick_mag_ui.solve_for_assignment",
            return_value=(filtered_solver_states, filtered_solver_states),
        ):
            state.run_selected_oxidation_assignment(force=True)

        cached = state.cached_spin_solution()
        self.assertIsNotNone(cached)
        assert cached is not None
        _, all_states = cached
        indexing = site_indexing_from_generation_parameters(params, build)
        labels: set[str] = set()
        for config in all_states:
            moments = state.expand_spin_moments_to_structure(config.all_moments, structure)
            fractions = classify_structure_by_cubes(
                structure_with_moments(structure, moments),
                site_indexing=indexing,
            )
            if fractions is not None:
                labels.add(fractions.dominant)

        self.assertTrue({"A", "F", "G", "C"}.issubset(labels))

    def test_saved_generated_focus_uses_saved_structure_not_builder_preview(self) -> None:
        state = AppState()
        state.builder_save_name = "LaFeO3"
        state.save_builder_structure()
        structure_a = state.ungrouped_structures[-1]
        self.assertIsNotNone(structure_a)
        assert structure_a is not None

        state.a_site_element = "Sr"
        state.b_site_element = "Mn"
        state.x_site_element = "F"
        state.builder_save_name = "SrMnF3"
        state.save_builder_structure()
        structure_b = state.ungrouped_structures[-1]
        self.assertIsNotNone(structure_b)
        assert structure_b is not None

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
        state.builder_save_name = "Saved LaFeO3"
        state.save_builder_structure()
        structure = state.ungrouped_structures[-1]
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

    def test_cached_spin_solution_tracks_focused_saved_structure(self) -> None:
        state = AppState()
        state.builder_save_name = "Structure A"
        state.save_builder_structure()
        structure_a = state.ungrouped_structures[-1]
        self.assertIsNotNone(structure_a)
        assert structure_a is not None

        state.a_site_element = "Sr"
        state.b_site_element = "Mn"
        state.x_site_element = "F"
        state.builder_save_name = "Structure B"
        state.save_builder_structure()
        structure_b = state.ungrouped_structures[-1]
        self.assertIsNotNone(structure_b)
        assert structure_b is not None

        fake_config = SpinConfig(
            energy=-1.0,
            all_moments=np.ones(1, dtype=float),
            magnetization=1.0,
            n_unpaired=1.0,
        )
        state.magnetic_result_structure = structure_a
        state.magnetic_solution_cache[0] = ([], [fake_config])

        state.set_focus(structure_b)
        self.assertIsNone(state.cached_spin_solution())

        state.set_focus(structure_a)
        cached = state.cached_spin_solution()
        self.assertIsNotNone(cached)
        assert cached is not None
        self.assertEqual(len(cached[1]), 1)

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
