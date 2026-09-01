"""How far a relaxed structure moved from the geometry it was submitted as."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quick_mag.geometry_drift import (  # noqa: E402
    builder_can_express,
    cell_lengths,
    geometry_drift,
    is_orthogonal,
)
from quick_mag.structure import ChemicalStructure  # noqa: E402


def structure(lattice, coords, labels=("Fe", "O")) -> ChemicalStructure:
    return ChemicalStructure.with_zero_magnetic_moments(
        name="s",
        lattice=np.asarray(lattice, dtype=float),
        cartesian_coords=np.asarray(coords, dtype=float),
        atomic_labels=list(labels),
    )


BASE_COORDS = [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]


class TestNoChange(unittest.TestCase):
    def test_an_identical_structure_reports_no_drift(self):
        first = structure(np.diag([4.0, 4.0, 4.0]), BASE_COORDS)
        second = structure(np.diag([4.0, 4.0, 4.0]), BASE_COORDS)
        drift = geometry_drift(first, second)

        self.assertFalse(drift.cell_changed)
        self.assertFalse(drift.atoms_moved)
        self.assertEqual(drift.rmsd, 0.0)
        self.assertAlmostEqual(drift.volume_ratio, 1.0)
        self.assertEqual(drift.headline(), "Geometry unchanged.")


class TestCellChange(unittest.TestCase):
    def test_a_cell_relaxation_is_reported_as_a_volume_change(self):
        before = structure(np.diag([4.0, 4.0, 4.0]), BASE_COORDS)
        after = structure(np.diag([3.9, 3.9, 3.9]), BASE_COORDS)
        drift = geometry_drift(before, after)

        self.assertTrue(drift.cell_changed)
        self.assertFalse(drift.atoms_moved)
        self.assertEqual(drift.lengths_before, (4.0, 4.0, 4.0))
        self.assertEqual(drift.lengths_after, (3.9, 3.9, 3.9))
        np.testing.assert_allclose(drift.length_deltas, (-0.1, -0.1, -0.1))
        self.assertAlmostEqual(drift.volume_ratio, (3.9 / 4.0) ** 3)
        self.assertIn("cell", drift.headline())

    def test_a_sheared_cell_is_flagged_as_outside_what_the_builder_can_make(self):
        # build_perovskite only ever emits diag(2dx, 2dy, 2dz), so a shear puts
        # the structure permanently beyond any choice of parameters.
        sheared = structure(
            [[4.0, 0.0, 0.0], [0.3, 4.0, 0.0], [0.0, 0.0, 4.0]], BASE_COORDS
        )
        self.assertFalse(is_orthogonal(sheared.lattice))
        self.assertFalse(builder_can_express(sheared))

        drift = geometry_drift(structure(np.diag([4.0] * 3), BASE_COORDS), sheared)
        self.assertFalse(drift.cell_is_orthogonal)


class TestAtomMotion(unittest.TestCase):
    def test_displacements_are_measured_atom_by_atom_in_shared_order(self):
        before = structure(np.diag([4.0] * 3), BASE_COORDS)
        after = structure(np.diag([4.0] * 3), [[0.0, 0.0, 0.0], [2.0, 2.0, 2.3]])
        drift = geometry_drift(before, after)

        self.assertTrue(drift.atoms_moved)
        self.assertAlmostEqual(drift.max_displacement, 0.3)
        self.assertEqual(drift.max_displacement_index, 1)
        # One atom moved 0.3 and one moved 0: rms over both.
        self.assertAlmostEqual(drift.rmsd, np.sqrt((0.3**2) / 2))

    def test_an_atom_that_crossed_a_cell_face_is_not_folded_back(self):
        # Wrapping would report a genuine 3.9 A migration as a 0.1 A wobble, and
        # the number people are reading this for is "did anything move a lot".
        before = structure(np.diag([4.0] * 3), [[0.05, 0.0, 0.0], [2.0, 2.0, 2.0]])
        after = structure(np.diag([4.0] * 3), [[3.95, 0.0, 0.0], [2.0, 2.0, 2.0]])
        drift = geometry_drift(before, after)
        self.assertAlmostEqual(drift.max_displacement, 3.9)


class TestDegenerate(unittest.TestCase):
    def test_a_different_atom_count_is_reported_as_not_comparable(self):
        before = structure(np.diag([4.0] * 3), BASE_COORDS)
        after = structure(np.diag([4.0] * 3), [[0.0, 0.0, 0.0]], labels=["Fe"])
        drift = geometry_drift(before, after)

        self.assertFalse(drift.comparable)
        self.assertIn("not comparable", drift.headline())

    def test_a_missing_side_gives_no_report_rather_than_raising(self):
        self.assertIsNone(geometry_drift(None, structure(np.diag([4.0] * 3), BASE_COORDS)))

    def test_cell_lengths_handles_a_rotated_cell(self):
        rotated = np.array([[0.0, 4.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 0.0, 5.0]])
        np.testing.assert_allclose(cell_lengths(rotated), (4.0, 4.0, 5.0))


class TestProvenanceFlag(unittest.TestCase):
    def test_a_relaxed_structure_keeps_its_parameters_but_not_its_geometry_claim(self):
        from quick_mag.remote import protocol
        from quick_mag.structure import generate_random_test_perovskite

        template, _ = generate_random_test_perovskite(np.random.default_rng(5))
        self.assertTrue(template.geometry_matches_generation)

        relaxed = protocol.structure_from_result(
            template,
            {
                "final_lattice": (np.asarray(template.lattice) * 0.98).tolist(),
                "final_coords": np.asarray(template.cartesian_coords).tolist(),
            },
        )
        # Topology kept, geometry claim dropped: that pairing is the whole point.
        self.assertIs(relaxed.generation_parameters, template.generation_parameters)
        self.assertFalse(relaxed.geometry_matches_generation)


if __name__ == "__main__":
    unittest.main()
