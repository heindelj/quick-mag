"""Lattice planes as an address for defect sites.

Pure geometry over ``SiteKey`` -- no ImGui, no structures -- so this module runs
without the optional ``ui`` extra.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quick_mag.defect_planes import (  # noqa: E402
    coerce_miller,
    fold_plane_index,
    nearest_occupied_plane,
    occupied_planes,
    plane_index_of_key,
    plane_miller_in_cell,
    plane_period,
    plane_role_counts,
    plane_role_label,
    site_key_cube_coords,
    sites_in_plane,
)
from quick_mag.perovskite_builder import (  # noqa: E402
    VERTEX_NAMES,
    SiteKey,
    canonical_site_keys,
)


class CubeCoordinateTests(unittest.TestCase):
    """Doubling the cube coordinate makes every sublattice integral."""

    def test_a_sites_land_on_even_corners(self) -> None:
        self.assertEqual(site_key_cube_coords(SiteKey("A", 0, 0, 0)), (0, 0, 0))
        self.assertEqual(site_key_cube_coords(SiteKey("A", 1, 2, 3)), (2, 4, 6))

    def test_b_sites_land_on_the_cube_centre(self) -> None:
        self.assertEqual(site_key_cube_coords(SiteKey("B", 0, 0, 0)), (1, 1, 1))
        self.assertEqual(site_key_cube_coords(SiteKey("B", 1, 2, 3)), (3, 5, 7))

    def test_each_x_vertex_steps_one_unit_off_its_b_site(self) -> None:
        # VERTEX_NAMES is ("+a","-a","+b","-b","+c","-c"): axis = vertex // 2,
        # and the even rows are the positive direction.
        expected = {
            "+a": (2, 1, 1),
            "-a": (0, 1, 1),
            "+b": (1, 2, 1),
            "-b": (1, 0, 1),
            "+c": (1, 1, 2),
            "-c": (1, 1, 0),
        }
        for vertex, name in enumerate(VERTEX_NAMES):
            with self.subTest(vertex=name):
                key = SiteKey("X", 0, 0, 0, vertex)
                self.assertEqual(site_key_cube_coords(key), expected[name])

    def test_every_canonical_site_is_integral_and_distinct(self) -> None:
        keys = canonical_site_keys((2, 2, 2), False)
        coords = [site_key_cube_coords(key) for key in keys]
        self.assertEqual(len(coords), len(set(coords)))
        for coord in coords:
            self.assertTrue(all(isinstance(value, int) for value in coord))


class PlaneIndexTests(unittest.TestCase):
    def test_the_index_is_the_dot_product(self) -> None:
        self.assertEqual(plane_index_of_key(SiteKey("A", 1, 2, 3), (1, 0, 0)), 2)
        self.assertEqual(plane_index_of_key(SiteKey("B", 0, 0, 0), (1, 1, 1)), 3)

    def test_miller_indices_must_be_a_triple(self) -> None:
        self.assertEqual(coerce_miller((1, 0, -1)), (1, 0, -1))
        with self.assertRaises(ValueError):
            coerce_miller((1, 0))

    def test_folding_is_a_no_op_without_a_period(self) -> None:
        self.assertEqual(fold_plane_index(7, 0), 7)
        self.assertEqual(fold_plane_index(7, 4), 3)
        self.assertEqual(fold_plane_index(-1, 4), 3)


class LayeringTests(unittest.TestCase):
    """The half-cube step is what reaches every sublattice."""

    GRID = (2, 2, 2)

    def test_001_alternates_ao_and_bo2_layers(self) -> None:
        planes = occupied_planes(self.GRID, True, (0, 0, 1))
        self.assertEqual(planes, [0, 1, 2, 3])
        self.assertEqual(
            [plane_role_label(self.GRID, True, (0, 0, 1), m) for m in planes],
            ["A + X", "B + X", "A + X", "B + X"],
        )
        # An AO layer: one A and one apical oxygen per cell of the layer.
        self.assertEqual(
            plane_role_counts(self.GRID, True, (0, 0, 1), 0), {"A": 4, "X": 4}
        )
        # A BO2 layer: one B and two equatorial oxygens per cell of the layer.
        self.assertEqual(
            plane_role_counts(self.GRID, True, (0, 0, 1), 1), {"B": 4, "X": 8}
        )

    def test_a_whole_cube_step_would_only_reach_one_sublattice(self) -> None:
        # Two consecutive planes never hold the same roles -- which is exactly
        # what a whole-cell step would lose by skipping every other plane.
        first = set(plane_role_counts(self.GRID, True, (0, 0, 1), 0))
        second = set(plane_role_counts(self.GRID, True, (0, 0, 1), 1))
        self.assertEqual(first, {"A", "X"})
        self.assertEqual(second, {"B", "X"})

    def test_111_alternates_ao3_and_b_layers(self) -> None:
        planes = occupied_planes(self.GRID, True, (1, 1, 1))
        self.assertEqual(
            [plane_role_counts(self.GRID, True, (1, 1, 1), m) for m in planes],
            [{"A": 4, "X": 12}, {"B": 4}, {"A": 4, "X": 12}, {"B": 4}],
        )

    def test_110_alternates_abo_and_o2_layers(self) -> None:
        planes = occupied_planes(self.GRID, True, (1, 1, 0))
        self.assertEqual(
            [plane_role_label(self.GRID, True, (1, 1, 0), m) for m in planes],
            ["A + B + X", "X", "A + B + X", "X"],
        )

    def test_a_plane_family_partitions_every_site_exactly_once(self) -> None:
        for periodic in (True, False):
            total = len(canonical_site_keys(self.GRID, periodic))
            for miller in ((0, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (1, -1, 0)):
                with self.subTest(periodic=periodic, miller=miller):
                    seen: list = []
                    for plane in occupied_planes(self.GRID, periodic, miller):
                        seen.extend(
                            sites_in_plane(self.GRID, periodic, miller, plane)
                        )
                    self.assertEqual(len(seen), len(set(seen)))
                    self.assertEqual(len(seen), total)

    def test_restricting_to_one_role_only_narrows(self) -> None:
        everything = sites_in_plane(self.GRID, True, (0, 0, 1), 1)
        b_only = sites_in_plane(self.GRID, True, (0, 0, 1), 1, role="B")
        self.assertEqual(len(b_only), 4)
        self.assertTrue(set(b_only).issubset(set(everything)))
        self.assertEqual(sites_in_plane(self.GRID, True, (0, 0, 1), 1, role="A"), [])

    def test_a_degenerate_triple_is_not_a_plane_family(self) -> None:
        self.assertEqual(occupied_planes(self.GRID, True, (0, 0, 0)), [])
        self.assertEqual(sites_in_plane(self.GRID, True, (0, 0, 0), 0), [])


class PeriodicFoldingTests(unittest.TestCase):
    """A periodic cell names the same layer twice; a finite one does not."""

    GRID = (2, 2, 2)

    def test_the_period_is_one_supercell_translation(self) -> None:
        self.assertEqual(plane_period(self.GRID, True, (0, 0, 1)), 4)
        self.assertEqual(plane_period((3, 3, 3), True, (0, 0, 1)), 6)
        self.assertEqual(plane_period(self.GRID, True, (1, 1, 1)), 4)

    def test_a_finite_build_has_no_period(self) -> None:
        self.assertEqual(plane_period(self.GRID, False, (0, 0, 1)), 0)

    def test_folding_reunites_the_apical_oxygen_with_its_a_plane(self) -> None:
        # With only the +c vertex row kept, the apical oxygen of the top cell is
        # a whole cell above the A plane it actually shares.
        top_oxygen = SiteKey("X", 0, 0, 1, 4)
        self.assertEqual(plane_index_of_key(top_oxygen, (0, 0, 1)), 4)
        self.assertEqual(
            plane_index_of_key(top_oxygen, (0, 0, 1), period=4),
            plane_index_of_key(SiteKey("A", 0, 0, 0), (0, 0, 1)),
        )
        self.assertIn(top_oxygen, sites_in_plane(self.GRID, True, (0, 0, 1), 0))

    def test_a_finite_build_keeps_its_closing_layer_separate(self) -> None:
        # Not folded: the closing layer of a cluster is a real, extra set of
        # atoms, not an image of the first one.
        self.assertEqual(occupied_planes(self.GRID, False, (0, 0, 1)), [0, 1, 2, 3, 4])

    def test_an_unoccupied_plane_index_names_nothing(self) -> None:
        # Deliberately *not* folded onto 97 % 4 == 1: a plane authored in a
        # larger supercell must be skipped, never silently relocated.
        self.assertEqual(sites_in_plane(self.GRID, True, (0, 0, 1), 97), [])
        self.assertNotIn(97, occupied_planes(self.GRID, True, (0, 0, 1)))


class SheetGeometryTests(unittest.TestCase):
    GRID = (2, 2, 2)

    def test_the_normal_picks_up_the_grid_shape(self) -> None:
        self.assertEqual(
            list(plane_miller_in_cell((2, 3, 4), (1, 1, 1))), [2.0, 3.0, 4.0]
        )
        self.assertEqual(list(plane_miller_in_cell((2, 3, 4), (0, 0, 1))), [0.0, 0.0, 4.0])

    def test_a_layer_can_sit_in_more_than_one_place(self) -> None:
        # Folding merged the A plane at z = 0 with the apical oxygens at z = 1.
        # They are the same layer, so they share an index -- but not a position,
        # which is why a sheet per index would leave half of them unmarked.
        positions = {
            0.5 * sum(a * b for a, b in zip((0, 0, 1), site_key_cube_coords(key)))
            for key in sites_in_plane(self.GRID, True, (0, 0, 1), 0)
        }
        self.assertEqual(positions, {0.0, 2.0})
        positions = {
            0.5 * sum(a * b for a, b in zip((0, 0, 1), site_key_cube_coords(key)))
            for key in sites_in_plane(self.GRID, True, (0, 0, 1), 1)
        }
        self.assertEqual(positions, {0.5})


class NearestPlaneTests(unittest.TestCase):
    def test_it_lands_on_the_closest_occupied_plane(self) -> None:
        self.assertEqual(nearest_occupied_plane([0, 2, 4], 3), 2)
        self.assertEqual(nearest_occupied_plane([0, 2, 4], 5), 4)
        self.assertEqual(nearest_occupied_plane([0, 2, 4], 2), 2)

    def test_an_empty_family_leaves_the_index_alone(self) -> None:
        self.assertEqual(nearest_occupied_plane([], 7), 7)


if __name__ == "__main__":
    unittest.main()
