"""Tests for magnetic orderings expressed as sign strings across lattice planes."""

import itertools
import os
import sys
import unittest

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.reference_configs import (  # noqa: E402
    builder_spin_sign,
    canonical_reference_patterns,
    plane_pattern_for,
)
from quick_mag.spin_planes import (  # noqa: E402
    CANONICAL_PLANE_PATTERNS,
    PATTERNS_BY_NAME,
    MagneticSublattice,
    PlanePattern,
    best_matching_pattern,
    defect_concentration,
    ordering_key,
    parse_plane_label,
    pattern_signs,
    patterns_for_sites,
    plane_cell_polygon,
    plane_count,
    plane_indices,
    polygon_triangles,
)

PERIOD_2_NAMES = ("F", "A(a)", "A(b)", "A(c)", "C(a)", "C(b)", "C(c)", "G")


def grid(shape):
    return MagneticSublattice.from_grid(shape)


class PlanePatternBasicsTest(unittest.TestCase):
    def test_a_pattern_rejects_a_nonsense_sign_string(self):
        for signs in ("", "+x-", "abc"):
            with self.subTest(signs=signs):
                with self.assertRaises(ValueError):
                    PlanePattern((1, 0, 0), signs)

    def test_labels_round_trip_through_plane_notation(self):
        for pattern in CANONICAL_PLANE_PATTERNS:
            with self.subTest(pattern=pattern.label):
                self.assertEqual(parse_plane_label(pattern.plane_label), 
                                 PlanePattern(pattern.miller, pattern.signs))
                self.assertEqual(plane_pattern_for(pattern.label), pattern)

    def test_a_pattern_without_a_name_displays_as_its_plane(self):
        self.assertEqual(PlanePattern((1, 1, 0), "++--").label, "(110) ++--")
        self.assertEqual(PlanePattern((1, 1, 0), "+-", "C(c)").label, "C(c)")


class CanonicalPatternsTest(unittest.TestCase):
    """The eight classical orderings are exactly the period-2 plane patterns."""

    def test_period_two_patterns_reproduce_the_classical_orderings(self):
        for shape in ((2, 2, 2), (3, 3, 3), (4, 4, 4)):
            cells = grid(shape).lattice_coords
            for name in PERIOD_2_NAMES:
                with self.subTest(shape=shape, name=name):
                    expected = np.array([builder_spin_sign(name, *c) for c in cells])
                    got = pattern_signs(cells, PATTERNS_BY_NAME[name])
                    self.assertEqual(ordering_key(got), ordering_key(expected))

    def test_they_also_hold_on_an_anisotropic_grid(self):
        # The interesting case: on a 4x2x2 the pseudocubic (111) planes are the
        # cell's (422), so anything that reads Miller indices in the cell frame gets
        # C and G wrong here while looking fine on a cubic grid.
        for shape in ((4, 2, 2), (2, 3, 4), (3, 3, 2), (5, 3, 2)):
            cells = grid(shape).lattice_coords
            for name in PERIOD_2_NAMES:
                with self.subTest(shape=shape, name=name):
                    expected = np.array([builder_spin_sign(name, *c) for c in cells])
                    got = pattern_signs(cells, PATTERNS_BY_NAME[name])
                    self.assertEqual(ordering_key(got), ordering_key(expected))

    def test_period_two_planes_generate_exactly_eight_orderings(self):
        cells = grid((4, 4, 4)).lattice_coords
        distinct = {
            ordering_key(pattern_signs(cells, PlanePattern(miller, "+-")))
            for miller in itertools.product(range(-2, 3), repeat=3)
        }
        self.assertEqual(len(distinct), 8)

        classical = {
            ordering_key(np.array([builder_spin_sign(name, *c) for c in cells]))
            for name in PERIOD_2_NAMES
        }
        self.assertEqual(distinct, classical)

    def test_e_type_is_up_up_down_down_and_is_not_a_type(self):
        cells = grid((4, 4, 4)).lattice_coords
        e_a = pattern_signs(cells, PATTERNS_BY_NAME["E(a)"])
        expected = np.array([1.0 if int(c[0]) % 4 in (0, 1) else -1.0 for c in cells])
        self.assertEqual(ordering_key(e_a), ordering_key(expected))
        # Before E was a plane pattern it was a duplicate of A(b).
        a_b = pattern_signs(cells, PATTERNS_BY_NAME["A(b)"])
        self.assertNotEqual(ordering_key(e_a), ordering_key(a_b))

    def test_a_pattern_needs_as_many_planes_as_its_string_is_long(self):
        cells = grid((2, 2, 2)).lattice_coords
        self.assertEqual(plane_count(cells, (1, 0, 0)), 2)
        self.assertNotIn("E(a)", [p.label for p in patterns_for_sites(cells)])
        self.assertIn("E(a)", [p.label for p in patterns_for_sites(grid((4, 2, 2)).lattice_coords)])

    def test_orderings_a_grid_cannot_tell_apart_are_offered_once(self):
        # A(a) and C(b) both just alternate along a on a 2x2x1 grid.
        labels = [p.label for p in patterns_for_sites(grid((2, 2, 1)).lattice_coords)]
        self.assertEqual(len(labels), len(set(labels)))
        cells = grid((2, 2, 1)).lattice_coords
        keys = [ordering_key(pattern_signs(cells, p))
                for p in patterns_for_sites(cells)]
        self.assertEqual(len(keys), len(set(keys)))


class SublatticeFrameTest(unittest.TestCase):
    def test_a_vacancy_does_not_shift_the_pattern(self):
        # Plane indices are tied to the lattice origin, so removing a site cannot
        # renumber the planes and flip every spin in the ordering.
        shape = (3, 3, 3)
        full = grid(shape)
        holed = MagneticSublattice.from_grid(
            shape, [-1 if index == 0 else index for index in range(27)]
        )
        pattern = PATTERNS_BY_NAME["G"]
        self.assertTrue(
            np.array_equal(
                pattern_signs(full.lattice_coords, pattern)[1:],
                pattern_signs(holed.lattice_coords, pattern),
            )
        )

    def test_miller_converts_into_the_cell_frame(self):
        # A pseudocubic (111) on a 3x3x3 supercell is the cell's (333).
        np.testing.assert_allclose(
            grid((3, 3, 3)).miller_in_cell((1, 1, 1)), [3.0, 3.0, 3.0]
        )
        np.testing.assert_allclose(
            grid((4, 2, 2)).miller_in_cell((1, 1, 1)), [4.0, 2.0, 2.0]
        )

    def test_vacated_cells_are_dropped_but_keep_the_others_in_place(self):
        sub = MagneticSublattice.from_grid((2, 2, 2), [0, -1, 2, 3, 4, 5, 6, 7])
        self.assertEqual(sub.size, 7)
        self.assertNotIn(1, sub.site_indices.tolist())
        # (0, 0, 1) was the vacated cell; the rest keep their own coordinates.
        self.assertEqual(sub.lattice_coords[1].tolist(), [0, 1, 0])


class DefectConcentrationTest(unittest.TestCase):
    def setUp(self):
        self.cells = grid((4, 4, 4)).lattice_coords
        self.pattern = PATTERNS_BY_NAME["G"]
        self.ideal = pattern_signs(self.cells, self.pattern)

    def test_the_ideal_and_its_global_flip_are_both_exact(self):
        for signs in (self.ideal, -self.ideal):
            match = defect_concentration(signs, self.cells, self.pattern)
            self.assertEqual(match.concentration, 0.0)
            self.assertTrue(match.is_exact)

    def test_one_flipped_spin_is_one_site_in_n(self):
        perturbed = self.ideal.copy()
        perturbed[7] *= -1
        match = defect_concentration(perturbed, self.cells, self.pattern)
        self.assertEqual(match.defect_count, 1)
        self.assertAlmostEqual(match.concentration, 1 / len(self.ideal))
        self.assertEqual(np.flatnonzero(match.mismatched).tolist(), [7])

    def test_it_never_exceeds_one_half(self):
        rng = np.random.default_rng(0)
        for trial in range(50):
            signs = np.where(rng.random(len(self.ideal)) < 0.5, -1.0, 1.0)
            for pattern in CANONICAL_PLANE_PATTERNS:
                with self.subTest(trial=trial, pattern=pattern.label):
                    self.assertLessEqual(
                        defect_concentration(signs, self.cells, pattern).concentration,
                        0.5,
                    )

    def test_sites_without_a_moment_are_not_counted(self):
        signs = self.ideal.copy()
        signs[:8] = 0.0
        match = defect_concentration(signs, self.cells, self.pattern)
        self.assertEqual(match.concentration, 0.0)
        self.assertFalse(match.mismatched.any())

    def test_a_length_mismatch_is_an_error_rather_than_a_wrong_answer(self):
        with self.assertRaises(ValueError):
            defect_concentration(self.ideal[:-1], self.cells, self.pattern)

    def test_best_match_finds_the_ordering_a_perturbation_came_from(self):
        perturbed = self.ideal.copy()
        perturbed[[1, 5, 9]] *= -1
        match = best_matching_pattern(perturbed, self.cells)
        self.assertEqual(match.pattern.label, "G")
        self.assertAlmostEqual(match.concentration, 3 / len(self.ideal))

    def test_best_match_is_none_when_nothing_carries_a_moment(self):
        self.assertIsNone(
            best_matching_pattern(np.zeros(len(self.ideal)), self.cells)
        )


class LoadedStructureTest(unittest.TestCase):
    """Orderings must be expressible for a structure with no builder provenance."""

    def test_planes_split_the_magnetic_sites_of_a_loaded_cif(self):
        pytest.importorskip("imgui_bundle")
        from pathlib import Path

        from quick_mag.cif_io import read_cif
        from quick_mag.classify_spin_structure import (
            site_indexing_from_magnetic_sublattice,
        )
        from quick_mag.reference_configs import magnetic_sublattice_for

        root = Path(__file__).resolve().parents[1]
        structure = read_cif(root / "assets" / "A_type" / "LaMnO3_222.cif")
        magnetic = [
            index
            for index, element in enumerate(structure.element_symbols())
            if element == "Mn"
        ]
        self.assertEqual(len(magnetic), 8)

        indexing = site_indexing_from_magnetic_sublattice(structure, magnetic)
        self.assertIsNotNone(indexing)
        sublattice = magnetic_sublattice_for(indexing)
        self.assertIsNotNone(sublattice)

        # A 2x2x2 Mn grid: (001) splits them 4/4, (111) spreads them over 4 planes.
        counts = np.bincount(plane_indices(sublattice.lattice_coords, (0, 0, 1)))
        self.assertEqual(counts.tolist(), [4, 4])
        self.assertEqual(plane_count(sublattice.lattice_coords, (1, 1, 1)), 4)


class PlaneGeometryTest(unittest.TestCase):
    CUBIC = np.eye(3) * 4.0
    TRICLINIC = np.array([[4.0, 0.0, 0.0], [1.0, 3.5, 0.0], [0.5, 1.0, 3.0]])

    @staticmethod
    def cartesian_normal(lattice, miller):
        normal = np.linalg.pinv(lattice) @ np.asarray(miller, dtype=float)
        return normal / np.linalg.norm(normal)

    def assert_simple_polygon(self, lattice, miller, offset, sides):
        polygon = plane_cell_polygon(lattice, miller, offset)
        self.assertEqual(len(polygon), sides)
        normal = self.cartesian_normal(lattice, miller)
        self.assertLess(float(np.ptp(polygon @ normal)), 1e-9, "not planar")
        edges = np.roll(polygon, -1, axis=0) - polygon
        turns = np.cross(edges, np.roll(edges, -1, axis=0)) @ normal
        self.assertTrue(
            bool(np.all(turns > -1e-9) or np.all(turns < 1e-9)),
            "polygon is not wound as a simple convex loop",
        )
        return polygon

    def test_a_cubic_cell_cut_flat_is_its_face(self):
        polygon = self.assert_simple_polygon(self.CUBIC, (0, 0, 1), 0.5, 4)
        self.assertAlmostEqual(float(np.ptp(polygon[:, 0])), 4.0)
        self.assertAlmostEqual(float(np.ptp(polygon[:, 1])), 4.0)

    def test_a_body_diagonal_cut_is_a_hexagon(self):
        self.assert_simple_polygon(self.CUBIC, (1, 1, 1), 1.5, 6)

    def test_it_holds_for_a_triclinic_cell(self):
        # The case that catches a transposed change of basis: on an orthogonal cell
        # the wrong normal still winds the polygon correctly.
        self.assert_simple_polygon(self.TRICLINIC, (0, 0, 1), 0.5, 4)
        self.assert_simple_polygon(self.TRICLINIC, (1, 1, 1), 1.5, 6)
        self.assert_simple_polygon(self.TRICLINIC, (1, 1, 0), 1.0, 4)

    def test_a_plane_that_misses_the_cell_draws_nothing(self):
        self.assertEqual(plane_cell_polygon(self.CUBIC, (1, 1, 1), 9.0).shape, (0, 3))
        self.assertEqual(plane_cell_polygon(self.CUBIC, (0, 0, 0), 0.0).shape, (0, 3))

    def test_triangulation_is_a_fan_over_the_polygon(self):
        polygon = plane_cell_polygon(self.CUBIC, (1, 1, 1), 1.5)
        triangles = polygon_triangles(polygon)
        self.assertEqual(triangles.shape, (3 * (len(polygon) - 2), 3))
        self.assertEqual(polygon_triangles(polygon[:2]).shape, (0, 3))


class CanonicalReferenceNamesTest(unittest.TestCase):
    def test_the_classical_eight_still_lead_the_list(self):
        self.assertEqual(
            canonical_reference_patterns((4, 4, 4))[:8],
            ("G", "C(a)", "C(b)", "C(c)", "F", "A(a)", "A(b)", "A(c)"),
        )

    def test_every_offered_name_resolves_back_to_a_pattern(self):
        for shape in ((2, 2, 2), (3, 3, 3), (4, 4, 4), (4, 2, 2)):
            for name in canonical_reference_patterns(shape):
                with self.subTest(shape=shape, name=name):
                    self.assertIsNotNone(plane_pattern_for(name))


if __name__ == "__main__":
    unittest.main()
