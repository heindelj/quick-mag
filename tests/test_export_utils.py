"""Tests for structure export: CIF + VASP magmom file writing."""

import io
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.export_utils import (  # noqa: E402
    CIF_MIME_TYPE,
    EXPORT_ARCHIVE_NAME,
    ZIP_MIME_TYPE,
    export_bundle_bytes,
    export_structures,
    format_magmom_line,
)
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

    def test_site_magnitudes_rescale_unit_spins_to_formal_moments(self):
        # Solver output: unit spins on the two Fe sites, nothing on the O.
        moments = np.array([[0, 0, 1.0], [0, 0, -1.0], [0, 0, 0.0]])
        line = format_magmom_line(moments, collinear=True, magnitudes=[5.0, 5.0, 0.0])
        values = [float(v) for v in line.split()]
        self.assertAlmostEqual(values[0], 5.0)
        self.assertAlmostEqual(values[1], -5.0)
        self.assertAlmostEqual(values[2], 0.0)

    def test_site_magnitudes_rescale_rather_than_multiply(self):
        # Already-physical moments are rescaled to the same values, not squared.
        moments = np.array([[0, 0, 5.0], [0, 0, -5.0], [0, 0, 0.0]])
        line = format_magmom_line(moments, collinear=True, magnitudes=[5.0, 5.0, 0.0])
        values = [float(v) for v in line.split()]
        self.assertAlmostEqual(values[0], 5.0)
        self.assertAlmostEqual(values[1], -5.0)

    def test_site_magnitudes_apply_per_site_and_noncollinear(self):
        moments = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
        line = format_magmom_line(moments, collinear=False, magnitudes=[5.0, 3.0, 0.0])
        values = [float(v) for v in line.split()]
        self.assertEqual(len(values), 9)
        self.assertAlmostEqual(values[0], 5.0)
        self.assertAlmostEqual(values[5], -3.0)

    def test_mismatched_magnitude_length_is_ignored(self):
        moments = np.array([[0, 0, 1.0], [0, 0, -1.0], [0, 0, 0.0]])
        line = format_magmom_line(moments, collinear=True, magnitudes=[5.0, 5.0])
        values = [float(v) for v in line.split()]
        self.assertAlmostEqual(values[0], 1.0)
        self.assertAlmostEqual(values[1], -1.0)


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

    def test_export_writes_formal_moments_from_site_magnitudes(self):
        structure = _structure("Structure A")
        structure.spin_configurations = [
            SavedSpinConfiguration(
                magnetic_moments=np.array([[0, 0, 1.0], [0, 0, -1.0], [0, 0, 0.0]]),
                site_moment_magnitudes=np.array([5.0, 5.0, 0.0]),
                collinear=True,
            )
        ]

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            export_structures([structure], out_dir)
            line = (out_dir / "Structure_A_spins.txt").read_text().strip()

        values = [float(v) for v in line.split()]
        self.assertAlmostEqual(values[0], 5.0)
        self.assertAlmostEqual(values[1], -5.0)
        self.assertAlmostEqual(values[2], 0.0)

    def test_export_creates_missing_output_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "nested" / "out"
            export_structures([_structure("Structure A")], out_dir)
            self.assertTrue((out_dir / "Structure_A.cif").exists())

    def test_bundle_returns_a_lone_cif_as_itself(self):
        # No saved spin configurations means one file, so there is nothing to zip.
        name, payload, mime = export_bundle_bytes([_structure("Structure A")])

        self.assertEqual(name, "Structure_A.cif")
        self.assertEqual(mime, CIF_MIME_TYPE)
        with tempfile.TemporaryDirectory() as tmp:
            export_structures([_structure("Structure A")], Path(tmp))
            self.assertEqual(payload, (Path(tmp) / "Structure_A.cif").read_bytes())

    def test_bundle_zips_a_structure_that_has_spin_configurations(self):
        structure = _structure("Structure A")
        structure.spin_configurations = [
            SavedSpinConfiguration(
                magnetic_moments=np.array([[0, 0, 4.0], [0, 0, -4.0], [0, 0, 0.0]]),
                energy=-1.0,
                collinear=True,
            )
        ]

        name, payload, mime = export_bundle_bytes([structure])

        self.assertEqual(name, EXPORT_ARCHIVE_NAME)
        self.assertEqual(mime, ZIP_MIME_TYPE)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertEqual(
                sorted(archive.namelist()),
                ["Structure_A.cif", "Structure_A_spins.txt"],
            )
            spins = archive.read("Structure_A_spins.txt").decode()
            self.assertEqual(len(spins.strip().splitlines()), 1)

    def test_bundle_zips_one_cif_per_structure(self):
        names = ["Structure A", "Structure B", "Structure C"]

        _, payload, mime = export_bundle_bytes([_structure(n) for n in names])

        self.assertEqual(mime, ZIP_MIME_TYPE)
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            self.assertEqual(
                sorted(archive.namelist()),
                ["Structure_A.cif", "Structure_B.cif", "Structure_C.cif"],
            )

    def test_bundle_entries_never_carry_a_path(self):
        # A structure name is free text, so it reaches the zip through
        # sanitize_filename: separators become underscores and every entry stays a
        # bare filename that unzips into the chosen folder, not above or below it.
        structures = [_structure("La/Fe O3"), _structure("../escape")]

        _, payload, _mime = export_bundle_bytes(structures)

        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            entries = archive.namelist()
        self.assertEqual(sorted(entries), [".._escape.cif", "La_Fe_O3.cif"])
        for entry in entries:
            self.assertEqual(Path(entry).name, entry)

    def test_bundle_rejects_an_empty_export(self):
        with self.assertRaises(ValueError):
            export_bundle_bytes([])

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
