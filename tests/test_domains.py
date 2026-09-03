"""Domains: stacking perovskite blocks, and per-axis periodicity."""

from __future__ import annotations

import unittest
from collections import Counter

import numpy as np

from quick_mag.defects import SiteDefect, apply_defects, resolve_key_to_indices
from quick_mag.domains import (
    DomainAssigner,
    DomainSpec,
    conform_domain_to_stack,
    domain_composition,
    matching_in_plane_cells,
    stack_lattice,
    stack_oct_counts,
    validate_stack,
)
from quick_mag.generation import (
    domain_atomic_labels_for_build,
    generated_structure_from_parameters,
    stacked_structure_from_domains,
)
from quick_mag.perovskite_builder import (
    SiteKey,
    build_perovskite,
    canonical_index_of_key,
    canonical_site_counts,
    canonical_site_keys,
    periodic_axes,
)
from quick_mag.structure import ChemicalStructure, PerovskiteGenerationParameters


class PeriodicAxesTest(unittest.TestCase):
    def test_scalar_and_triple_forms_agree(self) -> None:
        self.assertEqual(periodic_axes(True), (True, True, True))
        self.assertEqual(periodic_axes(False), (False, False, False))
        self.assertEqual(periodic_axes([True, False, True]), (True, False, True))
        with self.assertRaises(ValueError):
            periodic_axes([True, False])

    def test_finite_axis_adds_only_its_closing_layer(self) -> None:
        shape = (2, 3, 4)
        n_a, n_b, n_x = canonical_site_counts(shape, (True, True, False))
        self.assertEqual(n_b, 24)
        self.assertEqual(n_a, 2 * 3 * 5)
        self.assertEqual(n_x, 3 * 24 + 2 * 3)
        keys = canonical_site_keys(shape, (True, True, False))
        self.assertEqual(len(keys), n_a + n_b + n_x)
        # Every key resolves to its own position, and nothing else does.
        for index, key in enumerate(keys):
            self.assertEqual(canonical_index_of_key(key, shape, (True, True, False)), index)
        self.assertEqual(canonical_index_of_key(SiteKey("X", 0, 0, 0, 1), shape, (True, True, False)), -1)
        self.assertGreaterEqual(canonical_index_of_key(SiteKey("X", 0, 0, 0, 5), shape, (True, True, False)), 0)

    def test_index_of_key_matches_key_list_for_every_periodicity(self) -> None:
        shape = (2, 2, 3)
        for flags in [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]:
            axes = tuple(bool(f) for f in flags)
            keys = canonical_site_keys(shape, axes)
            with self.subTest(axes=axes):
                for index, key in enumerate(keys):
                    self.assertEqual(canonical_index_of_key(key, shape, axes), index)

    def test_build_closes_finite_axis_only(self) -> None:
        build = build_perovskite(np.zeros(3), 1, 1, 1, 2.0, 2.0, 2.0, periodic=(True, True, False))
        self.assertEqual(len(build.a_sites), 2 * 2 * 3)
        self.assertEqual(len(build.x_sites), 3 * 8 + 4)
        # The closing A plane sits one full cell above the last octahedron.
        self.assertAlmostEqual(build.a_sites[:, 2].max(), 6.0)

    def test_neighbors_skip_images_along_finite_axes(self) -> None:
        structure = ChemicalStructure.with_zero_magnetic_moments(
            "t", np.eye(3) * 4.0, np.zeros((1, 3)), ["Fe"], is_periodic=(True, True, False)
        )
        self.assertTrue(structure.is_periodic)
        self.assertEqual(structure.periodic_axes, (True, True, False))
        self.assertEqual(len(structure.neighbors(0, 4.5)), 4)
        structure.set_periodic_axes((False, False, False))
        self.assertFalse(structure.is_periodic)
        self.assertEqual(len(structure.neighbors(0, 4.5)), 0)

    def test_parameters_reconcile_scalar_and_axes(self) -> None:
        params = PerovskiteGenerationParameters(
            center=np.zeros(3), n_oct_x=1, n_oct_y=1, n_oct_z=1,
            center_to_vertex_distance_x=2.0, center_to_vertex_distance_y=2.0,
            center_to_vertex_distance_z=2.0, tilt_system="a0a0a0",
            tilt_angle_x_deg=0.0, tilt_angle_y_deg=0.0, tilt_angle_z_deg=0.0,
            periodic=(True, True, False), a_site_element="La", b_site_element="Fe",
            x_site_element="O",
        )
        self.assertTrue(params.periodic)
        self.assertEqual(params.periodic_axes, (True, True, False))
        from dataclasses import replace

        rebuilt = replace(params, periodic=False)
        self.assertEqual(rebuilt.periodic_axes, (False, False, False))
        self.assertEqual(params.defect_reference_periodic(), (True, True, False))

    def test_defect_images_expand_only_along_closed_axes(self) -> None:
        shape = (2, 2, 2)
        # Authored with every axis periodic, rendered with only c closed.
        indices = resolve_key_to_indices(
            SiteKey("A", 0, 0, 0), shape, periodic=(True, True, False), expand_images=True
        )
        self.assertEqual(len(indices), 2)
        indices = resolve_key_to_indices(
            SiteKey("A", 0, 0, 0), shape, periodic=False, expand_images=True
        )
        self.assertEqual(len(indices), 8)


class DomainCompositionTest(unittest.TestCase):
    def test_follows_the_formula_mode(self) -> None:
        base = dict(a_site_element="La", b_site_element="Fe", x_site_element="O",
                    a2_site_element="Sr", b2_site_element="Co")
        self.assertEqual(domain_composition(DomainSpec(**base)), "LaFeO3")
        self.assertEqual(domain_composition(DomainSpec(formula_mode="double", **base)), "La2FeCoO6")
        self.assertEqual(domain_composition(DomainSpec(formula_mode="quadruple", **base)), "LaSr3Fe4O12")
        self.assertEqual(domain_composition(DomainSpec(formula_mode="dq", **base)), "LaSr3Fe2Co2O12")
        self.assertEqual(domain_composition(DomainSpec(formula_mode="high_entropy", **base)), "high-entropy")


class StackGeometryTest(unittest.TestCase):
    def test_octahedron_counts_sum_along_the_stacking_axis(self) -> None:
        d1 = DomainSpec(n_cells=(2, 2, 2), lattice=(3.9, 3.9, 3.9))
        d2 = DomainSpec(formula_mode="double", n_cells=(1, 1, 1), lattice=(3.9, 3.9, 4.0))
        self.assertEqual(stack_oct_counts([d1, d2], 2), (2, 2, 4))
        self.assertEqual(validate_stack([d1, d2], 2), [])
        np.testing.assert_allclose(np.diag(stack_lattice([d1, d2], 2)), [7.8, 7.8, 2 * 3.9 + 2 * 4.0])

    def test_matching_requires_an_even_grid_for_ordered_formulas(self) -> None:
        odd = DomainSpec(n_cells=(3, 3, 3))
        with self.assertRaises(ValueError):
            matching_in_plane_cells(odd, "double", 2)
        even = DomainSpec(n_cells=(4, 4, 4))
        self.assertEqual(matching_in_plane_cells(even, "double", 2), (2, 2))
        self.assertEqual(matching_in_plane_cells(even, "perovskite", 2), (4, 4))
        # Stacking along a: the plane is b x c.
        self.assertEqual(matching_in_plane_cells(DomainSpec(n_cells=(3, 4, 2)), "double", 0), (2, 1))

    def test_conform_copies_in_plane_size_and_spacing(self) -> None:
        reference = DomainSpec(n_cells=(2, 3, 4), lattice=(3.8, 3.9, 4.0))
        new = DomainSpec(formula_mode="double", n_cells=(1, 1, 5), lattice=(4.2, 4.2, 4.3))
        conform_domain_to_stack(new, reference, 1)
        self.assertEqual(new.n_cells, (1, 1, 2))
        self.assertEqual(new.lattice, (3.8, 4.2, 4.0))
        self.assertEqual(validate_stack([reference, new], 1), [])
        self.assertTrue(validate_stack([reference, DomainSpec(n_cells=(3, 3, 3))], 1))


class DomainAssignmentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bottom = DomainSpec(a_site_element="Sr", b_site_element="Ti", n_cells=(2, 2, 2), lattice=(3.9, 3.9, 3.9))
        self.top = DomainSpec(a_site_element="La", b_site_element="Fe", n_cells=(2, 2, 2), lattice=(3.9, 3.9, 4.0))

    def test_interface_plane_defaults_to_the_upper_domain(self) -> None:
        assigner = DomainAssigner([self.bottom, self.top], 2, (True, True, False))
        self.assertEqual([assigner.domain_of_plane(p) for p in range(5)], [0, 0, 1, 1, 1])
        self.assertEqual(assigner.domain_of_key(SiteKey("X", 0, 0, 1, 4)), 1)  # +c of layer 1
        self.assertEqual(assigner.domain_of_key(SiteKey("X", 0, 0, 1, 0)), 0)  # +a of layer 1
        self.assertTrue(assigner.is_interface_key(SiteKey("A", 0, 0, 2)))
        self.assertTrue(assigner.is_interface_key(SiteKey("X", 1, 1, 1, 4)))
        self.assertFalse(assigner.is_interface_key(SiteKey("A", 0, 0, 1)))
        self.assertFalse(assigner.is_interface_key(SiteKey("A", 0, 0, 0)))

    def test_interface_can_be_handed_to_the_lower_domain(self) -> None:
        self.top.interface_from_previous = True
        assigner = DomainAssigner([self.bottom, self.top], 2, (True, True, False))
        self.assertEqual([assigner.domain_of_plane(p) for p in range(5)], [0, 0, 0, 1, 1])
        self.assertEqual(assigner.domain_of_key(SiteKey("X", 0, 0, 1, 4)), 0)

    def test_periodic_stacking_axis_wraps_the_bottom_interface(self) -> None:
        self.bottom.interface_from_previous = True
        assigner = DomainAssigner([self.bottom, self.top], 2, True)
        self.assertEqual(assigner.domain_of_plane(0), 1)
        self.assertEqual(assigner.domain_of_plane(4), 1)
        self.assertTrue(assigner.is_interface_key(SiteKey("A", 0, 0, 0)))
        self.assertEqual(assigner.domain_of_key(SiteKey("X", 0, 0, 3, 4)), 1)

    def test_labels_follow_the_layers(self) -> None:
        structure = stacked_structure_from_domains(
            [self.bottom, self.top], name="film", periodic=(True, True, False)
        )
        self.assertEqual(structure.periodic_axes, (True, True, False))
        np.testing.assert_allclose(np.diag(structure.lattice), [7.8, 7.8, 2 * 3.9 + 2 * 4.0])
        z = structure.cartesian_coords[:, 2]
        by_label = {label: sorted(set(np.round(z[np.array(structure.atomic_labels) == label], 3))) for label in set(structure.atomic_labels)}
        self.assertEqual(by_label["Sr"], [0.0, 3.9])
        self.assertEqual(by_label["Ti"], [1.95, 5.85])
        self.assertEqual(by_label["La"], [7.8, 11.8, 15.8])
        self.assertEqual(by_label["Fe"], [9.8, 13.8])
        counts = Counter(structure.atomic_labels)
        self.assertEqual(counts["Sr"], 8)
        self.assertEqual(counts["La"], 12)
        self.assertEqual(counts["O"], 3 * 16 + 4)

    def test_generation_parameters_roundtrip_and_keep_domains(self) -> None:
        structure = stacked_structure_from_domains([self.bottom, self.top], name="film", periodic=True)
        params = structure.generation_parameters
        self.assertTrue(params.is_multi_domain())
        self.assertEqual(params.grid_shape(), (2, 2, 4))
        rebuilt = generated_structure_from_parameters(params, name="again", periodic=params.periodic_axes)
        np.testing.assert_allclose(rebuilt.cartesian_coords, structure.cartesian_coords)
        self.assertEqual(rebuilt.atomic_labels, structure.atomic_labels)
        # The non-periodic render used for periodic images still labels every site.
        rendered = generated_structure_from_parameters(params, name="render", periodic=False)
        self.assertEqual(rendered.atom_count, len(canonical_site_keys((2, 2, 4), False)))

    def test_double_perovskite_ordering_runs_across_domains(self) -> None:
        low = DomainSpec(formula_mode="double", b_site_element="Fe", b2_site_element="Mo", n_cells=(1, 1, 1))
        high = DomainSpec(formula_mode="double", b_site_element="Fe", b2_site_element="Mo", n_cells=(1, 1, 1))
        structure = stacked_structure_from_domains([low, high], name="d", periodic=True)
        build = build_perovskite(**structure.generation_parameters.build_kwargs())
        keys = canonical_site_keys(build.octahedra.shape, True)
        labels = domain_atomic_labels_for_build(build, periodic=True, domains=[low, high], stacking_axis=2)
        for key, label in zip(keys, labels):
            if key.role == "B":
                self.assertEqual(label, "Fe" if (key.i + key.j + key.k) % 2 == 0 else "Mo")

    def test_defects_address_the_combined_grid(self) -> None:
        structure = stacked_structure_from_domains(
            [self.bottom, self.top],
            name="film",
            periodic=True,
            defects=[SiteDefect("substitution", SiteKey("B", 0, 0, 3), "Co")],
        )
        self.assertEqual(Counter(structure.atomic_labels)["Co"], 1)
        self.assertEqual(Counter(structure.atomic_labels)["Fe"], 7)

    def test_single_domain_parameters_synthesize_a_spec(self) -> None:
        params = PerovskiteGenerationParameters(
            center=np.zeros(3), n_oct_x=3, n_oct_y=3, n_oct_z=3,
            center_to_vertex_distance_x=2.0, center_to_vertex_distance_y=2.0,
            center_to_vertex_distance_z=2.05, tilt_system="a0a0a0",
            tilt_angle_x_deg=0.0, tilt_angle_y_deg=0.0, tilt_angle_z_deg=0.0,
            periodic=True, a_site_element="La", b_site_element="Fe",
            x_site_element="O", formula_mode="double",
        )
        (spec,) = params.domain_specs()
        self.assertEqual(spec.n_cells, (2, 2, 2))
        self.assertEqual(spec.lattice, (4.0, 4.0, 4.1))
        self.assertFalse(params.is_multi_domain())


if __name__ == "__main__":
    unittest.main()
