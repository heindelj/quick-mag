from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quick_mag.classify_spin_structure import (  # noqa: E402
    SiteSpinClassification,
    SpinStructureClassification,
    classification_fractions,
    classify_structure,
    site_indexing_from_perovskite_build,
)
from quick_mag.perovskite_builder import build_perovskite  # noqa: E402
from quick_mag.structure import ChemicalStructure  # noqa: E402


def make_spin_structure(
    shape: tuple[int, int, int],
    sign_fn,
) -> tuple[ChemicalStructure, object]:
    nx, ny, nz = shape
    lattice_constant = 4.0
    half_edge = lattice_constant / 2.0
    build = build_perovskite(
        center=np.array([half_edge, half_edge, half_edge], dtype=float),
        n_oct_x=nx - 1,
        n_oct_y=ny - 1,
        n_oct_z=nz - 1,
        center_to_vertex_distance_x=half_edge,
        center_to_vertex_distance_y=half_edge,
        center_to_vertex_distance_z=half_edge,
        periodic=True,
    )
    coords = np.asarray(build.all_sites, dtype=float)
    labels = (
        ["La"] * len(build.a_sites)
        + ["Fe"] * len(build.b_sites)
        + ["O"] * len(build.x_sites)
    )
    moments = np.zeros_like(coords)
    b_sites = np.asarray(build.b_site_indices, dtype=int).reshape(shape)
    for i, j, k in np.ndindex(shape):
        site_index = int(b_sites[i, j, k])
        moments[site_index, 2] = 4.0 * float(sign_fn(i, j, k))

    structure = ChemicalStructure(
        name=f"synthetic_{nx}x{ny}x{nz}",
        lattice=np.diag([nx * lattice_constant, ny * lattice_constant, nz * lattice_constant]),
        cartesian_coords=coords,
        atomic_labels=labels,
        magnetic_moments=moments,
        is_periodic=True,
    )
    return structure, build


class SpinStructureClassifierTests(unittest.TestCase):
    def assert_all_site_labels(self, result, expected: str) -> None:
        self.assertEqual(result.label, expected)
        self.assertTrue(result.site_records)
        self.assertEqual({record.label for record in result.site_records}, {expected})

    def classify(self, shape: tuple[int, int, int], sign_fn):
        structure, build = make_spin_structure(shape, sign_fn)
        return classify_structure(
            structure,
            site_indexing=site_indexing_from_perovskite_build(build),
        )

    def test_f_type(self) -> None:
        result = self.classify((2, 2, 2), lambda i, j, k: 1)
        self.assert_all_site_labels(result, "F")

    def test_g_type(self) -> None:
        result = self.classify((2, 2, 2), lambda i, j, k: (-1) ** (i + j + k))
        self.assert_all_site_labels(result, "G")

    def test_a_type(self) -> None:
        result = self.classify((2, 2, 2), lambda i, j, k: (-1) ** k)
        self.assert_all_site_labels(result, "A")
        for record in result.site_records:
            self.assertEqual(record.axis_signs["a"], 1)
            self.assertEqual(record.axis_signs["b"], 1)
            self.assertEqual(record.axis_signs["c"], -1)

    def test_c_type(self) -> None:
        result = self.classify((2, 2, 2), lambda i, j, k: (-1) ** (j + k))
        self.assert_all_site_labels(result, "C")
        for record in result.site_records:
            self.assertEqual(record.axis_signs["a"], 1)
            self.assertEqual(record.axis_signs["b"], -1)
            self.assertEqual(record.axis_signs["c"], -1)

    def test_e_type_period_four_along_a(self) -> None:
        # Up-up-down-down (period 4) modulation along the a-axis.
        result = self.classify((4, 2, 2), lambda i, j, k: 1 if i % 4 in (0, 1) else -1)
        self.assert_all_site_labels(result, "E")
        for record in result.site_records:
            self.assertEqual(record.axis_signs["c"], 1)
            self.assertAlmostEqual(record.sigmas2["a"], -1.0, places=6)
            self.assertAlmostEqual(record.sigmas2["b"], 1.0, places=6)
            self.assertAlmostEqual(record.sigmas2["c"], 1.0, places=6)

    def test_old_two_by_two_plaquette_is_not_e(self) -> None:
        # The minimal 2x2x1 up-up/down-down plaquette has period 2, not 4, so it
        # is a collinear A/C-type order -- never E under the period-4 definition.
        result = self.classify(
            (2, 2, 1), lambda i, j, k: 1 if (i, j) in {(0, 0), (1, 0)} else -1
        )
        self.assertNotEqual(result.label, "E")
        self.assertNotIn("E", {record.label for record in result.site_records})

    def test_neel_along_a_is_not_e(self) -> None:
        # Period-2 Neel (up-down-up-down) along a gives distance-2 sigma2 = +1.
        result = self.classify((4, 2, 2), lambda i, j, k: (-1) ** i)
        self.assertNotIn("E", {record.label for record in result.site_records})
        for record in result.site_records:
            self.assertAlmostEqual(record.sigmas2["a"], 1.0, places=6)

    def test_uniform_e_fractions(self) -> None:
        result = self.classify((4, 2, 2), lambda i, j, k: 1 if i % 4 in (0, 1) else -1)
        fractions = classification_fractions(result)
        self.assertEqual(fractions.total, 16)
        self.assertEqual(fractions.counts["E"], 16)
        self.assertAlmostEqual(fractions.fraction("E"), 1.0)
        self.assertEqual(fractions.dominant, "E")

    def test_classification_fractions_tally(self) -> None:
        # Directly exercise the aggregation: 5 F, 3 E, 2 unknown (-> Other).
        labels = ["F"] * 5 + ["E"] * 3 + ["unknown"] * 2

        def record(label: str) -> SiteSpinClassification:
            return SiteSpinClassification(
                site_index=0,
                grid_index=(0, 0, 0),
                label=label,
                sigmas={},
                axis_signs={},
                kappa=float("nan"),
                confidence=0.0,
            )

        classification = SpinStructureClassification(
            label="F",
            site_records=[record(label) for label in labels],
        )
        fractions = classification_fractions(classification)
        self.assertEqual(fractions.total, 10)
        self.assertEqual(fractions.counts["F"], 5)
        self.assertEqual(fractions.counts["E"], 3)
        self.assertEqual(fractions.counts["Other"], 2)
        self.assertEqual(fractions.counts["A"], 0)
        self.assertEqual(fractions.dominant, "F")
        self.assertAlmostEqual(fractions.fraction("E"), 0.3)

    def test_unit_cell_size_changes_available_classification(self) -> None:
        one_cell = self.classify((1, 1, 1), lambda i, j, k: 1)
        expanded = self.classify((2, 2, 2), lambda i, j, k: (-1) ** (i + j + k))

        self.assertEqual(one_cell.label, "F")
        self.assertTrue(
            any("single-layer periodic axis" in note for note in one_cell.notes)
        )
        self.assertEqual(expanded.label, "G")


if __name__ == "__main__":
    unittest.main()
