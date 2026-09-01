from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quick_mag.cell_edit import (  # noqa: E402
    canonical_lattice,
    cell_angles,
    cell_lengths,
    cell_parameters,
    lattice_from_parameters,
    orientation_of,
    strain_structure,
    strained_coords,
    tile_structure,
    tiled_cell,
)
from quick_mag.structure import ChemicalStructure  # noqa: E402


CUBIC = np.diag([4.0, 4.0, 4.0])
# Deliberately triclinic *and* rotated away from the canonical frame, which is
# the case an orientation-blind implementation gets wrong.
TRICLINIC = np.array(
    [
        [4.10, 0.35, -0.20],
        [0.60, 4.35, 0.15],
        [-0.25, 0.40, 4.60],
    ]
)


def structure_with(lattice: np.ndarray, coords: np.ndarray) -> ChemicalStructure:
    return ChemicalStructure(
        name="test",
        lattice=np.asarray(lattice, dtype=np.float64),
        cartesian_coords=np.asarray(coords, dtype=np.float64),
        atomic_labels=["Fe"] * len(coords),
        magnetic_moments=np.zeros((len(coords), 3)),
    )


class CellParameterTests(unittest.TestCase):
    def test_cubic_parameters(self) -> None:
        self.assertTrue(np.allclose(cell_lengths(CUBIC), (4.0, 4.0, 4.0)))
        self.assertTrue(np.allclose(cell_angles(CUBIC), (90.0, 90.0, 90.0)))

    def test_canonical_lattice_round_trips_its_parameters(self) -> None:
        params = (4.0, 5.0, 6.0, 80.0, 95.0, 110.0)
        self.assertTrue(np.allclose(cell_parameters(canonical_lattice(*params)), params))

    def test_canonical_lattice_rejects_impossible_angles(self) -> None:
        # Individually plausible, jointly describing no real cell.
        with self.assertRaises(ValueError):
            canonical_lattice(4.0, 4.0, 4.0, 20.0, 20.0, 170.0)

    def test_orientation_is_a_rotation_for_a_right_handed_cell(self) -> None:
        rotation = orientation_of(TRICLINIC)
        self.assertTrue(np.allclose(rotation @ rotation.T, np.eye(3)))
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=10)

    def test_a_left_handed_cell_keeps_its_handedness(self) -> None:
        """canonical_lattice is always right-handed, so the transform must reflect."""
        left_handed = TRICLINIC[[1, 0, 2]]
        self.assertLess(float(np.linalg.det(left_handed)), 0.0)
        transform = orientation_of(left_handed)
        self.assertTrue(np.allclose(transform @ transform.T, np.eye(3)))
        self.assertAlmostEqual(float(np.linalg.det(transform)), -1.0, places=10)
        rebuilt = lattice_from_parameters(left_handed, *cell_parameters(left_handed))
        self.assertTrue(np.allclose(rebuilt, left_handed))
        self.assertLess(float(np.linalg.det(rebuilt)), 0.0)


class LatticeFromParametersTests(unittest.TestCase):
    def test_unchanged_parameters_return_the_same_lattice(self) -> None:
        """The property the per-frame apply relies on: a no-op edit is the identity."""
        rebuilt = lattice_from_parameters(TRICLINIC, *cell_parameters(TRICLINIC))
        self.assertTrue(np.allclose(rebuilt, TRICLINIC))

    def test_angle_edit_preserves_orientation(self) -> None:
        a, b, c, alpha, beta, gamma = cell_parameters(TRICLINIC)
        edited = lattice_from_parameters(TRICLINIC, a, b, c, alpha + 3.0, beta, gamma)
        self.assertTrue(np.allclose(cell_parameters(edited)[3], alpha + 3.0))
        # The a axis is untouched by an alpha edit, so it must not have moved.
        self.assertTrue(np.allclose(edited[0], TRICLINIC[0]))

    def test_length_edit_keeps_every_angle(self) -> None:
        a, b, c, alpha, beta, gamma = cell_parameters(TRICLINIC)
        edited = lattice_from_parameters(TRICLINIC, a * 1.1, b, c, alpha, beta, gamma)
        self.assertTrue(np.allclose(cell_angles(edited), (alpha, beta, gamma)))
        self.assertTrue(np.allclose(cell_lengths(edited), (a * 1.1, b, c)))


class StrainTests(unittest.TestCase):
    def test_fractional_coordinates_are_preserved(self) -> None:
        coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [3.9, 0.1, 2.0]])
        structure = structure_with(CUBIC, coords)
        before = structure.fractional_coords.copy()
        strain_structure(structure, np.diag([4.4, 3.6, 5.0]))
        self.assertTrue(np.allclose(structure.fractional_coords, before))

    def test_strain_leaves_the_atom_list_alone(self) -> None:
        """What lets saved spin configurations survive a strain but not a tiling."""
        structure = structure_with(CUBIC, np.random.default_rng(0).random((7, 3)) * 4.0)
        labels = list(structure.atomic_labels)
        strain_structure(structure, np.diag([5.0, 5.0, 5.0]))
        self.assertEqual(structure.atom_count, 7)
        self.assertEqual(structure.atomic_labels, labels)

    def test_repeated_application_does_not_drift(self) -> None:
        structure = structure_with(TRICLINIC, np.random.default_rng(1).random((5, 3)))
        target = lattice_from_parameters(TRICLINIC, *cell_parameters(TRICLINIC))
        first = None
        for _ in range(50):
            strain_structure(structure, target)
            if first is None:
                first = structure.cartesian_coords.copy()
        self.assertTrue(np.allclose(structure.cartesian_coords, first))

    def test_degenerate_cell_is_refused(self) -> None:
        structure = structure_with(CUBIC, np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            strain_structure(structure, np.zeros((3, 3)))

    def test_strained_coords_matches_a_direct_scale(self) -> None:
        coords = np.array([[1.0, 2.0, 3.0]])
        scaled = strained_coords(CUBIC, coords, np.diag([8.0, 8.0, 8.0]))
        self.assertTrue(np.allclose(scaled, coords * 2.0))


class TilingTests(unittest.TestCase):
    def test_tiled_cell_scales_the_lattice_rows(self) -> None:
        lattice, coords, source = tiled_cell(TRICLINIC, np.zeros((1, 3)), (2, 1, 3))
        self.assertTrue(np.allclose(lattice[0], TRICLINIC[0] * 2))
        self.assertTrue(np.allclose(lattice[1], TRICLINIC[1]))
        self.assertTrue(np.allclose(lattice[2], TRICLINIC[2] * 3))
        self.assertEqual(len(coords), 6)
        self.assertEqual(len(source), 6)

    def test_tiling_repeats_labels_and_moments(self) -> None:
        structure = ChemicalStructure(
            name="test",
            lattice=CUBIC,
            cartesian_coords=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
            atomic_labels=["Fe", "O"],
            magnetic_moments=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]),
        )
        tile_structure(structure, (2, 2, 1))
        self.assertEqual(structure.atom_count, 8)
        self.assertEqual(sorted(structure.atomic_labels), ["Fe"] * 4 + ["O"] * 4)
        # Every copy of the Fe keeps the Fe's moment.
        for label, moment in zip(structure.atomic_labels, structure.magnetic_moments):
            expected = 1.0 if label == "Fe" else -1.0
            self.assertAlmostEqual(float(moment[2]), expected)

    def test_tiling_keeps_fractional_positions_within_each_image(self) -> None:
        structure = structure_with(CUBIC, np.array([[1.0, 1.0, 1.0]]))
        tile_structure(structure, (2, 1, 1))
        self.assertTrue(
            np.allclose(structure.fractional_coords, [[0.125, 0.25, 0.25], [0.625, 0.25, 0.25]])
        )

    def test_tiling_by_one_is_a_no_op(self) -> None:
        structure = structure_with(CUBIC, np.array([[1.0, 2.0, 3.0]]))
        tile_structure(structure, (1, 1, 1))
        self.assertEqual(structure.atom_count, 1)
        self.assertTrue(np.allclose(structure.lattice, CUBIC))


if __name__ == "__main__":
    unittest.main()
