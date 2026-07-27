"""Tests for group export: CIF + VASP magmom file writing."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.export_utils import export_group, format_magmom_line  # noqa: E402
from quick_mag.structure import (  # noqa: E402
    ChemicalStructure,
    SavedSpinConfiguration,
    StructureGroup,
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


class ExportGroupTest(unittest.TestCase):
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
        group = StructureGroup(name="My Group", is_generated=True)
        group.add_structure(with_configs)
        group.add_structure(without)

        with tempfile.TemporaryDirectory() as tmp:
            summary = export_group(group, Path(tmp))
            group_dir = Path(tmp) / "My_Group"

            self.assertTrue((group_dir / "Structure_A.cif").exists())
            self.assertTrue((group_dir / "Structure_B.cif").exists())

            spins = group_dir / "Structure_A_spins.txt"
            self.assertTrue(spins.exists())
            lines = spins.read_text().strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertEqual(len(lines[0].split()), 3)  # collinear: 1 per atom

            # Structure B has no configs -> no spins file.
            self.assertFalse((group_dir / "Structure_B_spins.txt").exists())

            self.assertEqual(summary["structures"], 2)
            self.assertEqual(summary["spin_configs"], 2)

    def test_cif_preserves_atom_order(self):
        structure = _structure("Ordered")
        group = StructureGroup(name="g", is_generated=True)
        group.add_structure(structure)
        with tempfile.TemporaryDirectory() as tmp:
            export_group(group, Path(tmp))
            from quick_mag.cif_io import read_cif

            cif = read_cif(Path(tmp) / "g" / "Ordered.cif")
            self.assertEqual(cif.element_symbols(), ["Fe", "Fe", "O"])
            # Fractional coordinates round-trip (P1, original atom order).
            np.testing.assert_allclose(
                cif.fractional_coords % 1.0,
                structure.fractional_coords % 1.0,
                atol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
