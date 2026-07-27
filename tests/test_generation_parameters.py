"""Round-trip tests for generation-parameter provenance on ChemicalStructure.

These exercise the key correctness property of storing generation parameters:
the build reconstructed from a structure's ``generation_parameters`` maps back
onto the structure's (shuffled, vacancy-reduced) atom order, so spin
classification uses the exact known B-site grid instead of geometry guessing.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.classify_spin_structure import (  # noqa: E402
    classify_structure,
    site_indexing_from_generation_parameters,
)
from quick_mag.structure import (  # noqa: E402
    build_from_generation_parameters,
    generate_random_test_perovskite,
)


class GenerationParameterRoundTripTest(unittest.TestCase):
    def _indexing(self, structure):
        params = structure.generation_parameters
        self.assertIsNotNone(params)
        build = build_from_generation_parameters(params)
        return params, build, site_indexing_from_generation_parameters(params, build)

    def test_field_is_populated(self):
        structure, _ = generate_random_test_perovskite(
            rng=np.random.default_rng(0), min_supercell_size=2, max_supercell_size=3
        )
        self.assertIsNotNone(structure.generation_parameters)
        self.assertEqual(structure.generation_parameters.source, "random_generator")

    def test_index_mapping_matches_labels_with_shuffle_and_vacancies(self):
        # Sweep several seeds, with shuffling and X vacancies enabled, and confirm
        # that the reconstructed A/B/X structure indices land on atoms of the
        # correct element -- this validates the inverse-permutation + vacancy map.
        for seed in range(8):
            structure, metadata = generate_random_test_perovskite(
                rng=np.random.default_rng(seed),
                min_supercell_size=2,
                max_supercell_size=4,
                shuffle_atoms=True,
                x_vacancy_fraction=0.2,
            )
            params, build, indexing = self._indexing(structure)
            labels = np.asarray(structure.atomic_labels, dtype=object)

            self.assertTrue(
                np.all(labels[indexing.a_site_indices] == params.a_site_element)
            )
            self.assertTrue(
                np.all(labels[indexing.b_site_indices] == params.b_site_element)
            )
            self.assertTrue(
                np.all(labels[indexing.x_site_indices] == params.x_site_element)
            )

            # B sites are never removed; count must equal the octahedra grid size.
            self.assertEqual(
                len(indexing.b_site_indices), int(np.prod(build.octahedra.shape))
            )
            total = (
                len(indexing.a_site_indices)
                + len(indexing.b_site_indices)
                + len(indexing.x_site_indices)
            )
            self.assertEqual(total, structure.atom_count)

            # Cross-check against the ground-truth site roles in the metadata.
            site_roles = np.asarray(metadata["site_roles"], dtype=object)
            np.testing.assert_array_equal(
                np.sort(indexing.b_site_indices),
                np.sort(np.flatnonzero(site_roles == "B")),
            )

    def test_ferromagnetic_pattern_classifies_via_stored_params(self):
        # A ferromagnetic B-site pattern is frustration-free on any (even periodic)
        # grid. If the inverse-permutation mapping were wrong, the +z moments would
        # land on A/X atoms and the B grid would read as zero -> not "F".
        structure, _ = generate_random_test_perovskite(
            rng=np.random.default_rng(3),
            min_supercell_size=2,
            max_supercell_size=3,
            shuffle_atoms=True,
            x_vacancy_fraction=0.1,
        )
        _, _, indexing = self._indexing(structure)
        structure.magnetic_moments[:] = 0.0
        structure.magnetic_moments[indexing.b_site_indices, 2] = 4.0

        result = classify_structure(structure, site_indexing=indexing)
        self.assertEqual(result.label, "F")


if __name__ == "__main__":
    unittest.main()
