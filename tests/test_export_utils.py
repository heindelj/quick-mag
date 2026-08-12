"""Tests for structure export: CIF + VASP magmom file writing."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.export_utils import export_structures, format_magmom_line  # noqa: E402
from quick_mag.structure import (  # noqa: E402
    ChemicalStructure,
    SavedSpinConfiguration,
)


def _structure(name: str) -> ChemicalStructure:
    return ChemicalStructure(
        name=name,
        lattice=np.eye(3) * 5.0,
        cartesian_coords=np.array(
            [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0]], dtype=np.float64
        ),
        atomic_labels=["Fe", "Fe", "O"],
        magnetic_moments=np.zeros((3, 3), dtype=np.float64),
    )


class FormatMagmomLineTest(unittest.TestCase):
    def test_collinear_writes_one_value_per_atom(self):
        moments = np.array([[0, 0, 4.0], [0, 0, -4.0], [0, 0, 0.0]])
        line = format_magmom_line(moments, collinear=True)
        values = [float(v) for v in line.split()]
        self.assertEqual(len(values), 3)
        # z-aligned spins reduce to +-m_z (dominant axis is +z here).
        self.assertAlmostEqual(values[0], 4.0)
        self.assertAlmostEqual(values[1], -4.0)
        self.assertAlmostEqual(values[2], 0.0)

    def test_noncollinear_writes_three_values_per_atom(self):
        moments = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
        line = format_magmom_line(moments, collinear=False)
        self.assertEqual(len(line.split()), 9)

    def test_all_zero_collinear(self):
        line = format_magmom_line(np.zeros((4, 3)), collinear=True)
        values = [float(v) for v in line.split()]
        self.assertEqual(values, [0.0, 0.0, 0.0, 0.0])


class ExportStructuresTest(unittest.TestCase):
    def test_export_writes_cif_and_spins(self):
        with_configs = _structure("Structure A")
        with_configs.spin_configurations = [
            SavedSpinConfiguration(
                magnetic_moments=np.array(
                    [[0, 0, 4.0], [0, 0, -4.0], [0, 0, 0.0]]
                ),
                energy=-1.0,
                collinear=True,
            ),
            SavedSpinConfiguration(
                magnetic_moments=np.array([[0, 0, 4.0], [0, 0, 4.0], [0, 0, 0.0]]),
                energy=-0.5,
                collinear=True,
            ),
        ]
        without = _structure("Structure B")  # no spin configs

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            summary = export_structures([with_configs, without], out_dir)

            self.assertTrue((out_dir / "Structure_A.cif").exists())
            self.assertTrue((out_dir / "Structure_B.cif").exists())

            spins = out_dir / "Structure_A_spins.txt"
            self.assertTrue(spins.exists())
            lines = spins.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(lines[0].split()), 3)  # collinear: 1 per atom

            # Structure B has no configs -> no spins file.
            self.assertFalse((out_dir / "Structure_B_spins.txt").exists())

            self.assertEqual(summary["structures"], 2)
            self.assertEqual(summary["spin_configs"], 2)

    def test_export_creates_missing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "nested" / "out"
            export_structures([_structure("Structure A")], out_dir)
            self.assertTrue((out_dir / "Structure_A.cif").exists())

    def test_cif_preserves_atom_order(self):
        structure = _structure("Ordered")
        with tempfile.TemporaryDirectory() as tmp:
            export_structures([structure], Path(tmp))
            from quick_mag.cif_io import read_cif

            cif = read_cif(Path(tmp) / "Ordered.cif")
            self.assertEqual(cif.element_symbols(), ["Fe", "Fe", "O"])
            # Fractional coordinates round-trip (P1, original atom order).
            np.testing.assert_allclose(
                cif.fractional_coords % 1.0,
                structure.fractional_coords % 1.0,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
