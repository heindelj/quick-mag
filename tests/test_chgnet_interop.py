"""Tests for the ASE interop layer under ``quick-mag chgnet``.

The conversions are exercised without CHGNet itself (no model is loaded), so
these run wherever ASE is installed. They are skipped otherwise, since ``ase`` is
an optional dependency of the ``chgnet`` extra.
"""

import importlib.util
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.chgnet_cli import resolve_calculation  # noqa: E402
from quick_mag.generation import generate_single_perovskite  # noqa: E402
from quick_mag.structure import ChemicalStructure  # noqa: E402

HAS_ASE = importlib.util.find_spec("ase") is not None


class ResolveCalculationTest(unittest.TestCase):
    """--sp / --opt / --fix-* map onto the four calculation modes."""

    class Args:
        def __init__(self, **kwargs):
            self.single_point = False
            self.fix_cell = False
            self.fix_atoms = False
            self.__dict__.update(kwargs)

    def test_default_optimizes_cell_and_atoms(self):
        self.assertEqual(resolve_calculation(self.Args()), "cell+atoms")

    def test_single_point(self):
        self.assertEqual(
            resolve_calculation(self.Args(single_point=True)), "single-point"
        )

    def test_fix_flags_narrow_the_optimization(self):
        self.assertEqual(resolve_calculation(self.Args(fix_cell=True)), "atoms")
        self.assertEqual(resolve_calculation(self.Args(fix_atoms=True)), "cell")

    def test_contradictory_flags_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_calculation(self.Args(fix_cell=True, fix_atoms=True))
        with self.assertRaises(ValueError):
            resolve_calculation(self.Args(single_point=True, fix_cell=True))


@unittest.skipUnless(HAS_ASE, "ase is not installed (pip install -e '.[chgnet]')")
class AseRoundTripTest(unittest.TestCase):
    def setUp(self):
        from quick_mag import chgnet_runner

        self.runner = chgnet_runner
        self.structure = generate_single_perovskite(
            "LaFeO3", a_site="La", b_site="Fe", x_site="O", a=3.9,
            n_cells_x=2, n_cells_y=2, n_cells_z=2,
        )

    def test_round_trip_preserves_geometry_and_provenance(self):
        atoms = self.runner.to_ase_atoms(self.structure)
        restored = self.runner.from_ase_atoms(atoms, template=self.structure)

        np.testing.assert_allclose(
            restored.cartesian_coords, self.structure.cartesian_coords, atol=1e-10
        )
        np.testing.assert_allclose(restored.lattice, self.structure.lattice, atol=1e-10)
        self.assertEqual(restored.atomic_labels, self.structure.atomic_labels)
        # Carried through so the solver keeps the fast B-site indexing path.
        self.assertIs(
            restored.generation_parameters, self.structure.generation_parameters
        )

    def test_oxidation_suffixes_are_stripped_for_ase(self):
        labelled = ChemicalStructure.with_zero_magnetic_moments(
            name="labelled",
            lattice=np.eye(3) * 5.0,
            cartesian_coords=np.array([[0.0, 0.0, 0.0], [2.5, 2.5, 2.5]]),
            atomic_labels=["Fe3+", "O2-"],
        )
        atoms = self.runner.to_ase_atoms(labelled)
        self.assertEqual(list(atoms.get_chemical_symbols()), ["Fe", "O"])
        # The suffixed labels come back untouched.
        restored = self.runner.from_ase_atoms(atoms, template=labelled)
        self.assertEqual(restored.atomic_labels, ["Fe3+", "O2-"])

    def test_non_periodic_structure_is_boxed_then_restored(self):
        cluster = ChemicalStructure.with_zero_magnetic_moments(
            name="cluster",
            lattice=np.eye(3) * 4.0,
            cartesian_coords=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            atomic_labels=["Fe", "O"],
            is_periodic=False,
        )
        atoms, offset = self.runner._to_ase_atoms_with_offset(cluster)
        # CHGNet needs a lattice, so the cluster runs inside a vacuum box.
        self.assertTrue(np.all(np.diag(np.asarray(atoms.get_cell())) >= 100.0))

        restored = self.runner.from_ase_atoms(atoms, template=cluster, offset=offset)
        np.testing.assert_allclose(
            restored.cartesian_coords, cluster.cartesian_coords, atol=1e-10
        )
        # The box is an artifact of the calculation, not part of the structure.
        np.testing.assert_allclose(restored.lattice, cluster.lattice, atol=1e-10)
        self.assertFalse(restored.is_periodic)

    def test_cell_relaxation_of_a_cluster_is_rejected(self):
        cluster = ChemicalStructure.with_zero_magnetic_moments(
            name="cluster",
            lattice=np.eye(3) * 4.0,
            cartesian_coords=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]),
            atomic_labels=["Fe", "O"],
            is_periodic=False,
        )
        for calculation in ("cell", "cell+atoms"):
            with self.subTest(calculation=calculation):
                with self.assertRaises(ValueError):
                    self.runner.run_chgnet_calculation(cluster, calculation)


if __name__ == "__main__":
    unittest.main()
