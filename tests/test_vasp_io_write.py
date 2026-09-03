"""Tests for POSCAR writing: round-trip, species blocks, and export integration."""

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.cif_io import read_cif  # noqa: E402
from quick_mag.export_utils import (  # noqa: E402
    FORMAT_CHOICES,
    export_options,
    export_structure,
    export_structures,
)
from quick_mag.structure import ChemicalStructure, SavedSpinConfiguration  # noqa: E402
from quick_mag.vasp_io import (  # noqa: E402
    grouped_by_species,
    is_grouped_by_species,
    poscar_text,
    read_poscar,
    species_blocks,
    species_grouping_permutation,
    write_poscar,
)


def _ungrouped() -> ChemicalStructure:
    """La, Fe, Mn, Fe, O -- the B block of a rocksalt double perovskite, in miniature."""
    return ChemicalStructure(
        name="LaFeMnFeO",
        lattice=np.diag([8.0, 4.0, 4.0]),
        cartesian_coords=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [3.0, 1.0, 1.0],
                [5.0, 1.0, 1.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        atomic_labels=["La", "Fe", "Mn", "Fe", "O"],
        magnetic_moments=np.array(
            [[0, 0, 0], [0, 0, 5.0], [0, 0, -4.0], [0, 0, 5.0], [0, 0, 0]],
            dtype=np.float64,
        ),
    )


class SpeciesBlockTest(unittest.TestCase):
    def test_blocks_follow_contiguous_runs(self):
        self.assertEqual(
            species_blocks(["La", "La", "Fe", "Mn", "Fe"]),
            [("La", 2), ("Fe", 1), ("Mn", 1), ("Fe", 1)],
        )

    def test_grouping_detection(self):
        self.assertFalse(is_grouped_by_species(_ungrouped()))
        self.assertTrue(is_grouped_by_species(grouped_by_species(_ungrouped())))

    def test_permutation_orders_by_first_appearance(self):
        order = species_grouping_permutation(_ungrouped())
        self.assertEqual(list(order), [0, 1, 3, 2, 4])


class GroupedBySpeciesTest(unittest.TestCase):
    def test_moments_and_spin_configs_follow_the_atoms(self):
        structure = _ungrouped()
        structure.spin_configurations.append(
            SavedSpinConfiguration(
                magnetic_moments=structure.magnetic_moments.copy(),
                site_moment_magnitudes=np.array([0.0, 5.0, 4.0, 5.0, 0.0]),
            )
        )
        grouped = grouped_by_species(structure)

        self.assertEqual(grouped.element_symbols(), ["La", "Fe", "Fe", "Mn", "O"])
        # The Mn moment moved with the Mn atom rather than staying at index 2.
        np.testing.assert_allclose(grouped.magnetic_moments[:, 2], [0, 5, 5, -4, 0])
        config = grouped.spin_configurations[0]
        np.testing.assert_allclose(config.magnetic_moments[:, 2], [0, 5, 5, -4, 0])
        np.testing.assert_allclose(config.site_moment_magnitudes, [0, 5, 5, 4, 0])

    def test_original_is_untouched(self):
        structure = _ungrouped()
        grouped_by_species(structure)
        self.assertEqual(structure.element_symbols(), ["La", "Fe", "Mn", "Fe", "O"])


class PoscarWriteTest(unittest.TestCase):
    def test_round_trip_preserves_geometry_and_order(self):
        structure = grouped_by_species(_ungrouped())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "POSCAR"
            write_poscar(structure, path)
            reloaded = read_poscar(path)

        self.assertEqual(reloaded.element_symbols(), structure.element_symbols())
        np.testing.assert_allclose(reloaded.lattice, structure.lattice, atol=1e-9)
        np.testing.assert_allclose(
            reloaded.cartesian_coords, structure.cartesian_coords, atol=1e-9
        )

    def test_atom_order_is_never_silently_changed(self):
        text = poscar_text(_ungrouped())
        lines = text.splitlines()
        # Fe appears in two blocks because the structure itself is ungrouped:
        # writing must not quietly sort it.
        self.assertEqual(lines[5].split(), ["La", "Fe", "Mn", "Fe", "O"])
        self.assertEqual(lines[6].split(), ["1", "1", "1", "1", "1"])

    def test_cartesian_mode(self):
        text = poscar_text(_ungrouped(), direct=False)
        self.assertEqual(text.splitlines()[7], "Cartesian")

    def test_comment_line(self):
        text = poscar_text(_ungrouped(), comment="relaxed with CHGNet")
        self.assertEqual(text.splitlines()[0], "relaxed with CHGNet")

    def test_direct_coordinates_are_wrapped_like_the_cif(self):
        """A boundary atom at fractional 1.0 must land at 0.0 in both files."""
        structure = ChemicalStructure(
            name="edge",
            lattice=np.diag([4.0, 4.0, 4.0]),
            cartesian_coords=np.array([[4.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
            atomic_labels=["Fe", "O"],
            magnetic_moments=np.zeros((2, 3)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export_structure(structure, out, formats=("cif", "vasp"))
            from_cif = read_cif(out / "edge.cif")
            from_vasp = read_poscar(out / "edge.vasp")
        np.testing.assert_allclose(
            from_cif.cartesian_coords, from_vasp.cartesian_coords, atol=1e-8
        )


class ExportFormatTest(unittest.TestCase):
    def test_formats_select_the_files_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export_structure(_ungrouped(), out, formats=("vasp",))
            self.assertTrue((out / "LaFeMnFeO.vasp").exists())
            self.assertFalse((out / "LaFeMnFeO.cif").exists())

    def test_group_species_reorders_every_file_together(self):
        structure = _ungrouped()
        structure.spin_configurations.append(
            SavedSpinConfiguration(magnetic_moments=structure.magnetic_moments.copy())
        )
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            export_structure(
                structure, out, formats=("cif", "vasp"), group_species=True
            )
            from_cif = read_cif(out / "LaFeMnFeO.cif")
            from_vasp = read_poscar(out / "LaFeMnFeO.vasp")
            magmoms = [
                float(value)
                for value in (out / "LaFeMnFeO_spins.txt").read_text().split()
            ]

        self.assertEqual(from_cif.element_symbols(), ["La", "Fe", "Fe", "Mn", "O"])
        self.assertEqual(from_vasp.element_symbols(), from_cif.element_symbols())
        np.testing.assert_allclose(
            from_cif.cartesian_coords, from_vasp.cartesian_coords, atol=1e-8
        )
        # The magmom line follows the same reorder, so the -4 sits on the Mn.
        np.testing.assert_allclose(magmoms, [0, 5, 5, -4, 0])

    def test_unknown_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                export_structure(_ungrouped(), Path(tmp), formats=("xyz",))

    def test_summary_counts_both_geometries(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = export_structures(
                [_ungrouped()], Path(tmp), formats=("cif", "vasp")
            )
        self.assertEqual(summary["cif"], 1)
        self.assertEqual(summary["vasp"], 1)

    def test_cli_format_flag_maps_to_export_keywords(self):
        class _Args:
            format = "both"

        self.assertEqual(
            export_options(_Args()), {"formats": ("cif", "vasp"), "group_species": True}
        )
        self.assertEqual(set(FORMAT_CHOICES), {"cif", "vasp", "both"})
        # The default stays CIF-only, so existing callers are unaffected.
        self.assertEqual(
            export_options(object()), {"formats": ("cif",), "group_species": False}
        )


if __name__ == "__main__":
    unittest.main()
