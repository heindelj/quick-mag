"""Tests for ``ChemicalStructure.neighbors``.

The neighbour search is vectorized over (atom, periodic image) pairs because it is
called once per atom during every exchange-coupling build; these tests pin it against
the obvious naive implementation so the optimization cannot drift.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.structure import ChemicalStructure  # noqa: E402


def naive_neighbors(structure: ChemicalStructure, index: int, cutoff: float):
    """One atom at a time, one image at a time -- the reference implementation."""
    center = structure.cartesian_coords[index]
    symbols = structure.element_symbols()
    if structure.is_periodic:
        translations = structure._lattice_image_translations(cutoff)
    else:
        translations = np.zeros((1, 3), dtype=np.float64)

    found = []
    for atom in range(len(structure.cartesian_coords)):
        for translation in translations:
            position = structure.cartesian_coords[atom] + translation
            distance = float(np.linalg.norm(position - center))
            if 1e-8 < distance <= cutoff:
                found.append((atom, symbols[atom], round(distance, 9), tuple(np.round(position, 9))))
    return sorted(found)


def _structure(is_periodic: bool = True) -> ChemicalStructure:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.9, 0.0, 0.0],
            [0.0, 2.1, 0.0],
            [0.0, 0.0, 2.4],
            [1.5, 1.5, 1.5],
        ],
        dtype=np.float64,
    )
    return ChemicalStructure(
        name="test",
        lattice=np.diag([4.0, 4.5, 5.0]),
        cartesian_coords=coords,
        atomic_labels=["Fe", "O", "O", "O", "La"],
        magnetic_moments=np.zeros_like(coords),
        is_periodic=is_periodic,
    )


class NeighborsTest(unittest.TestCase):
    def _assert_matches_naive(self, structure: ChemicalStructure, cutoff: float) -> None:
        for index in range(structure.atom_count):
            actual = sorted(
                (
                    neighbor.index,
                    neighbor.symbol,
                    round(neighbor.nn_distance, 9),
                    tuple(np.round(neighbor.coords, 9)),
                )
                for neighbor in structure.neighbors(index, cutoff)
            )
            self.assertEqual(actual, naive_neighbors(structure, index, cutoff))

    def test_matches_naive_search_with_periodic_images(self):
        self._assert_matches_naive(_structure(), 3.0)

    def test_matches_naive_search_across_several_image_shells(self):
        # A cutoff wider than the cell forces multiple images along every axis.
        self._assert_matches_naive(_structure(), 7.0)

    def test_matches_naive_search_without_periodicity(self):
        self._assert_matches_naive(_structure(is_periodic=False), 3.0)

    def test_excludes_the_site_itself_but_keeps_its_images(self):
        structure = _structure()
        neighbors = structure.neighbors(0, 4.5)
        self_images = [n for n in neighbors if n.index == 0]
        self.assertTrue(self_images, "periodic images of the site are real neighbours")
        self.assertTrue(all(n.nn_distance > 1e-8 for n in self_images))

    def test_empty_result_below_the_shortest_distance(self):
        self.assertEqual(_structure().neighbors(0, 0.5), [])


class CacheInvalidationTest(unittest.TestCase):
    """``element_symbols`` and the image translations are memoized in place."""

    def test_symbols_follow_relabelled_atoms(self):
        structure = _structure()
        self.assertEqual(structure.element_symbols()[0], "Fe")
        structure.atomic_labels = ["Mn", "O", "O", "O", "La"]
        self.assertEqual(structure.element_symbols()[0], "Mn")

    def test_translations_follow_a_changed_lattice(self):
        structure = _structure()
        small_cell = structure._lattice_image_translations(9.0)
        structure.lattice = np.diag([12.0, 12.0, 12.0])
        large_cell = structure._lattice_image_translations(9.0)
        # A cell wider than the cutoff needs fewer image shells than a narrow one.
        self.assertLess(len(large_cell), len(small_cell))

    def test_translations_are_per_cutoff(self):
        structure = _structure()
        self.assertLess(
            len(structure._lattice_image_translations(3.0)),
            len(structure._lattice_image_translations(9.0)),
        )


if __name__ == "__main__":
    unittest.main()
