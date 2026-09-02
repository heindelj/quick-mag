"""Fitting the closest ideal builder structure to a relaxed one."""

from __future__ import annotations

import copy
import unittest

import numpy as np

from quick_mag.domains import DomainSpec
from quick_mag.generation import stacked_structure_from_domains
from quick_mag.perovskite_builder import glazer_tilt_orientations
from quick_mag.reconstruction import (
    ReconstructionJob,
    TiltCandidate,
    base_parameters,
    rank_candidates,
    reconstruct_ideal,
)
from quick_mag.structure import GLAZER_TILT_SYSTEMS


def _perturbed(structure, *, scale=(1.0, 1.0, 1.0), noise=0.0, shift=(0.0, 0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    relaxed = copy.deepcopy(structure)
    factors = np.asarray(scale, dtype=np.float64)
    relaxed.lattice = structure.lattice * factors[:, None]
    relaxed.cartesian_coords = (
        structure.cartesian_coords * factors
        + rng.normal(0.0, noise, structure.cartesian_coords.shape)
        + np.asarray(shift, dtype=np.float64)
    )
    relaxed.geometry_matches_generation = False
    return relaxed


class TiltOrientationsTest(unittest.TestCase):
    def test_every_axis_orientation_is_listed_once(self) -> None:
        orientations = glazer_tilt_orientations(GLAZER_TILT_SYSTEMS)
        self.assertEqual(len(orientations), len(set(orientations)))
        self.assertEqual(len(orientations), 71)
        for canonical in GLAZER_TILT_SYSTEMS:
            self.assertIn(canonical, orientations)
        self.assertIn("a-a-c+", orientations)
        self.assertIn("a0b-a0", orientations)


class RankingTest(unittest.TestCase):
    def test_a_simpler_system_wins_a_near_tie(self) -> None:
        general = TiltCandidate("a-b-c+", (7.1, 7.2, 4.2), 0.03470)
        true = TiltCandidate("a-a-c+", (7.0, 7.0, 4.0), 0.03472)
        far = TiltCandidate("a0a0a0", (0.0, 0.0, 0.0), 0.2)
        ranked = rank_candidates([general, far, true])
        self.assertEqual([c.tilt_system for c in ranked], ["a-a-c+", "a-b-c+", "a0a0a0"])


class ReconstructionTest(unittest.TestCase):
    def setUp(self) -> None:
        domain = DomainSpec(n_cells=(2, 2, 2), lattice=(4.0, 4.0, 4.0))
        self.truth = stacked_structure_from_domains(
            [domain], name="truth", periodic=True,
            tilt_system="a-a-c+", tilt_angles_deg=(7.0, 7.0, 4.0),
        )

    def test_base_parameters_read_the_cell_off_the_relaxed_structure(self) -> None:
        relaxed = _perturbed(self.truth, scale=(1.02, 1.0, 0.99))
        base = base_parameters(relaxed)
        self.assertAlmostEqual(2.0 * base.center_to_vertex_distance_x, 4.08)
        self.assertAlmostEqual(2.0 * base.center_to_vertex_distance_z, 3.96)
        self.assertEqual(base.tilt_system, "a0a0a0")

    def test_the_fit_recovers_the_tilt_system_and_angles(self) -> None:
        relaxed = _perturbed(self.truth, scale=(1.02, 1.0, 0.99), noise=0.01, shift=(0.3, 0.1, -0.2))
        result = reconstruct_ideal(relaxed)
        self.assertEqual(result.tilt_system, "a-a-c+")
        self.assertAlmostEqual(result.tilt_angles_deg[0], 7.0, delta=0.5)
        self.assertAlmostEqual(result.tilt_angles_deg[2], 4.0, delta=0.5)
        self.assertLess(result.rmsd, 0.03)
        self.assertEqual(result.atom_count, relaxed.atom_count)
        self.assertEqual(len(result.distances), relaxed.atom_count)
        # The ideal is exactly what its parameters regenerate.
        self.assertTrue(result.ideal.geometry_matches_generation)
        self.assertEqual(result.ideal.atomic_labels, relaxed.atomic_labels)

    def test_the_job_steps_one_tilt_system_at_a_time(self) -> None:
        job = ReconstructionJob(_perturbed(self.truth), tilt_systems=["a0a0a0", "a-a-c+"])
        self.assertFalse(job.step())
        self.assertEqual(job.completed, 1)
        self.assertTrue(job.step())
        self.assertIsNotNone(job.result)
        self.assertEqual(job.result.tilt_system, "a-a-c+")
        self.assertLess(job.result.rmsd, 1e-3)

    def test_a_structure_without_provenance_is_refused(self) -> None:
        loaded = copy.deepcopy(self.truth)
        loaded.generation_parameters = None
        with self.assertRaises(ValueError):
            ReconstructionJob(loaded)

    def test_a_stack_keeps_its_domains(self) -> None:
        low = DomainSpec(a_site_element="Sr", b_site_element="Ti", n_cells=(2, 2, 1), lattice=(3.9, 3.9, 3.9))
        high = DomainSpec(a_site_element="La", b_site_element="Fe", n_cells=(2, 2, 2), lattice=(3.9, 3.9, 4.0))
        truth = stacked_structure_from_domains([low, high], name="film", periodic=True, tilt_system="a0a0c-", tilt_angles_deg=(0.0, 0.0, 5.0))
        result = reconstruct_ideal(_perturbed(truth, scale=(1.0, 1.0, 1.01)), tilt_systems=["a0a0a0", "a0a0c-", "a0a0c+"])
        self.assertEqual(result.tilt_system, "a0a0c-")
        self.assertTrue(result.params.is_multi_domain())
        self.assertAlmostEqual(result.params.domains[1].lattice[2], 4.04, places=6)
        self.assertLess(result.rmsd, 1e-3)


if __name__ == "__main__":
    unittest.main()
