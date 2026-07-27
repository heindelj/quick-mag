"""Invariants for the 8-atom spin-classification enumeration."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from quick_mag.cube_spin_lookup import (  # noqa: E402
    CLASS_ORDER,
    cube_category,
    decode,
    distance_vector,
    encode,
    nearest_label,
    pure_strings,
)
from enumerate_spin_classifications import nn_coupling_fingerprint  # noqa: E402


class SpinClassificationLookupTest(unittest.TestCase):
    def setUp(self) -> None:
        self.reps = pure_strings()

    def test_encode_decode_roundtrip(self) -> None:
        for index in (0, 1, 42, 3280, 6560):
            self.assertEqual(encode(decode(index)), index)

    def test_pure_class_counts(self) -> None:
        counts = {name: len(self.reps[name]) for name in CLASS_ORDER}
        self.assertEqual(counts, {"A": 6, "C": 6, "G": 2, "F": 2})

    def test_pure_strings_distance_zero_to_own_class(self) -> None:
        for col, name in enumerate(CLASS_ORDER):
            for rep in self.reps[name]:
                dist = distance_vector(rep)
                self.assertEqual(dist[col], 0)
                self.assertIn(name, nearest_label(dist).split("/"))

    def test_single_flip_from_g_has_distance_two(self) -> None:
        g_rep = self.reps["G"][0].copy()
        g_rep[0] *= -1
        dist = distance_vector(g_rep)
        self.assertEqual(dist[CLASS_ORDER.index("G")], 2)
        # G is still the nearest pure order after a single flip.
        self.assertEqual(nearest_label(dist), "G")

    def test_flip_to_zero_costs_one(self) -> None:
        f_rep = self.reps["F"][0].copy()  # all +1
        f_rep[0] = 0
        dist = distance_vector(f_rep)
        self.assertEqual(dist[CLASS_ORDER.index("F")], 1)

    def test_all_pure_orders_share_nn_spectrum(self) -> None:
        # On the bipartite cube graph the pure collinear orders are gauge
        # equivalent, so eigenvalues cannot separate F/A/C/G.
        fingerprints = {
            nn_coupling_fingerprint(self.reps[name][0]) for name in CLASS_ORDER
        }
        self.assertEqual(len(fingerprints), 1)
        expected = tuple(np.round(np.sort([-3, -1, -1, -1, 1, 1, 1, 3]), 6))
        self.assertEqual(next(iter(fingerprints)), expected)

    def test_within_class_fingerprint_consistency(self) -> None:
        for name in CLASS_ORDER:
            fps = {nn_coupling_fingerprint(rep) for rep in self.reps[name]}
            self.assertEqual(len(fps), 1)

    def test_cube_category(self) -> None:
        self.assertEqual(cube_category(self.reps["F"][0]), "F")
        self.assertEqual(cube_category(self.reps["G"][0]), "G")
        self.assertEqual(cube_category(self.reps["A"][0]), "A")
        self.assertEqual(cube_category(self.reps["C"][0]), "C")
        # All-zero cube is equidistant to every class -> Other.
        self.assertEqual(cube_category(np.zeros(8, dtype=np.int8)), "Other")
        # One flip from G is still nearest G.
        g_rep = self.reps["G"][0].copy()
        g_rep[0] *= -1
        self.assertEqual(cube_category(g_rep), "G")


if __name__ == "__main__":
    unittest.main()
