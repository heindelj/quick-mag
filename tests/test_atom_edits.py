"""The selection slab, and editing a loaded structure atom by atom."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quick_mag.atom_edits import (  # noqa: E402
    SelectionSlab,
    add_proton,
    atoms_in_slab,
    cell_extent_along,
    remove_atom,
    slab_face_corners,
    slab_normal,
    slab_offset_from_distance,
    substitute_atom,
)
from quick_mag.structure import ChemicalStructure  # noqa: E402

CUBIC = np.eye(3) * 4.0
TRICLINIC = np.array(
    [
        [4.10, 0.35, -0.20],
        [0.60, 4.35, 0.15],
        [-0.25, 0.40, 4.60],
    ]
)


def layered_structure(lattice=CUBIC) -> ChemicalStructure:
    """Two atoms per layer, four layers along c (fractional z = 0, .25, .5, .75)."""
    fractional = np.array(
        [[0.0, 0.0, z] for z in (0.0, 0.25, 0.5, 0.75)]
        + [[0.5, 0.5, z] for z in (0.0, 0.25, 0.5, 0.75)]
    )
    return ChemicalStructure(
        name="layers",
        lattice=lattice,
        cartesian_coords=fractional @ lattice,
        atomic_labels=["Sr"] * 4 + ["O"] * 4,
        magnetic_moments=np.zeros((8, 3)),
    )


class SlabGeometryTests(unittest.TestCase):
    def test_lattice_directions_follow_the_lattice_not_the_axes(self) -> None:
        normal = slab_normal(TRICLINIC, (1, 0, 0))
        expected = TRICLINIC[0] / np.linalg.norm(TRICLINIC[0])
        np.testing.assert_allclose(normal, expected)
        vertex = slab_normal(TRICLINIC, (1, 1, 1))
        summed = TRICLINIC.sum(axis=0)
        np.testing.assert_allclose(vertex, summed / np.linalg.norm(summed))

    def test_zero_direction_is_no_slab(self) -> None:
        self.assertIsNone(slab_normal(CUBIC, (0, 0, 0)))
        slab = SelectionSlab(direction=(0, 0, 0))
        self.assertEqual(atoms_in_slab(np.zeros((3, 3)), CUBIC, slab), [])
        self.assertIsNone(slab_face_corners(CUBIC, slab))

    def test_extent_spans_the_cell_corners(self) -> None:
        low, high = cell_extent_along(CUBIC, np.array([0.0, 0.0, 1.0]))
        self.assertAlmostEqual(low, 0.0)
        self.assertAlmostEqual(high, 4.0)
        diagonal = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
        low, high = cell_extent_along(CUBIC, diagonal)
        self.assertAlmostEqual(high - low, 4.0 * np.sqrt(3.0))

    def test_offset_round_trips_through_distance(self) -> None:
        slab = SelectionSlab(direction=(1, 1, 0), offset=0.3)
        normal = slab_normal(CUBIC, slab.direction)
        low, high = cell_extent_along(CUBIC, normal)
        distance = low + 0.3 * (high - low)
        self.assertAlmostEqual(slab_offset_from_distance(CUBIC, slab, distance), 0.3)
        # Clamped to the cell.
        self.assertEqual(slab_offset_from_distance(CUBIC, slab, low - 5.0), 0.0)
        self.assertEqual(slab_offset_from_distance(CUBIC, slab, high + 5.0), 1.0)


class SlabSelectionTests(unittest.TestCase):
    def test_a_thin_slab_picks_one_layer(self) -> None:
        structure = layered_structure()
        slab = SelectionSlab(direction=(0, 0, 1), offset=0.5, thickness=0.5)
        picked = atoms_in_slab(structure.cartesian_coords, structure.lattice, slab)
        # Fractional z = 0.5 is atoms 2 and 6.
        self.assertEqual(picked, [2, 6])

    def test_thickness_widens_the_pick(self) -> None:
        structure = layered_structure()
        slab = SelectionSlab(direction=(0, 0, 1), offset=0.5, thickness=2.2)
        picked = atoms_in_slab(structure.cartesian_coords, structure.lattice, slab)
        self.assertEqual(picked, [1, 2, 3, 5, 6, 7])

    def test_a_layer_on_the_mid_plane_is_not_lost_to_rounding(self) -> None:
        structure = layered_structure()
        slab = SelectionSlab(direction=(0, 0, 1), offset=0.25, thickness=0.0)
        picked = atoms_in_slab(structure.cartesian_coords, structure.lattice, slab)
        self.assertEqual(picked, [1, 5])

    def test_the_same_direction_selects_the_same_layer_in_a_sheared_cell(self) -> None:
        structure = layered_structure(TRICLINIC)
        # Along c in lattice terms: the fractional-z layers project onto the
        # unit vector along c at spacings of |c|/4, but the a and b components
        # of an atom's position also project onto that vector, so the slab has
        # to be thin against the layer spacing but reach both the (0,0) and
        # (0.5,0.5) atoms of a layer. A direction [0 0 1] with the (1,1,0)
        # in-plane offset projected out is what the cell-based extent gives.
        slab = SelectionSlab(direction=(0, 0, 1), offset=0.5, thickness=1.0)
        picked = atoms_in_slab(structure.cartesian_coords, structure.lattice, slab)
        self.assertIn(2, picked)
        self.assertNotIn(0, picked)

    def test_faces_bracket_the_slab_and_cover_the_cell(self) -> None:
        slab = SelectionSlab(direction=(0, 0, 1), offset=0.5, thickness=1.0)
        low, high = slab_face_corners(CUBIC, slab)
        np.testing.assert_allclose(low[:, 2], 1.5)
        np.testing.assert_allclose(high[:, 2], 2.5)
        self.assertAlmostEqual(low[:, 0].min(), 0.0)
        self.assertAlmostEqual(low[:, 0].max(), 4.0)
        self.assertAlmostEqual(low[:, 1].min(), 0.0)
        self.assertAlmostEqual(low[:, 1].max(), 4.0)


class LoadedEditTests(unittest.TestCase):
    def test_substitution_rewrites_the_label(self) -> None:
        structure = layered_structure()
        substitute_atom(structure, 0, "Ba")
        self.assertEqual(structure.atomic_labels[0], "Ba")
        self.assertEqual(structure.element_symbols()[0], "Ba")
        with self.assertRaises(ValueError):
            substitute_atom(structure, 0, "  ")

    def test_removal_shrinks_every_array_together(self) -> None:
        structure = layered_structure()
        structure.magnetic_moments[3] = (0.0, 0.0, 2.0)
        removed = remove_atom(structure, 3)
        self.assertEqual(removed.label, "Sr")
        np.testing.assert_allclose(removed.cartesian, [0.0, 0.0, 3.0])
        np.testing.assert_allclose(removed.magnetic_moment, [0.0, 0.0, 2.0])
        self.assertEqual(structure.atom_count, 7)
        self.assertEqual(structure.cartesian_coords.shape, (7, 3))
        self.assertEqual(structure.magnetic_moments.shape, (7, 3))
        self.assertEqual(structure.atomic_labels, ["Sr"] * 3 + ["O"] * 4)

    def test_a_proton_lands_a_bond_length_from_its_host_away_from_the_cations(self) -> None:
        # One oxygen bridging two cations along x; the proton must point off-axis.
        structure = ChemicalStructure(
            name="bridge",
            lattice=np.eye(3) * 20.0,
            cartesian_coords=np.array(
                [[8.0, 10.0, 10.0], [12.0, 10.0, 10.0], [10.0, 10.0, 10.0]]
            ),
            atomic_labels=["Fe", "Fe", "O"],
            magnetic_moments=np.zeros((3, 3)),
            is_periodic=False,
        )
        index = add_proton(structure, 2)
        self.assertEqual(index, 3)
        self.assertEqual(structure.atomic_labels[3], "H")
        bond = structure.cartesian_coords[3] - structure.cartesian_coords[2]
        self.assertAlmostEqual(float(np.linalg.norm(bond)), 0.98)
        self.assertAlmostEqual(abs(float(bond[0])), 0.0, places=6)

    def test_a_proton_points_away_from_a_bent_bridge(self) -> None:
        structure = ChemicalStructure(
            name="bent",
            lattice=np.eye(3) * 20.0,
            cartesian_coords=np.array(
                [[8.0, 11.0, 10.0], [12.0, 11.0, 10.0], [10.0, 10.0, 10.0]]
            ),
            atomic_labels=["Fe", "Fe", "O"],
            magnetic_moments=np.zeros((3, 3)),
            is_periodic=False,
        )
        add_proton(structure, 2)
        bond = structure.cartesian_coords[3] - structure.cartesian_coords[2]
        self.assertLess(float(bond[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
