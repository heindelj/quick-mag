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


class _EMTWithMoments:
    """Factory for an ASE calculator that stands in for CHGNet.

    ``run_chgnet_calculation`` needs a calculator that reports energy, forces,
    stress *and* magmoms. EMT gives the first three analytically and for free;
    the fourth it does not implement at all, so it is added as zeros. That is
    enough to drive the real ASE optimizers, filters and observers through the
    runner without a 2 GB torch install -- which is the only part of the path the
    stub does not cover.
    """

    @staticmethod
    def build():
        import numpy as np
        from ase.calculators.emt import EMT

        class EMTWithMoments(EMT):
            implemented_properties = list(EMT.implemented_properties) + ["magmoms"]

            def calculate(self, atoms=None, properties=("energy",), system_changes=None):
                super().calculate(atoms, properties, system_changes)
                self.results["magmoms"] = np.zeros(len(self.atoms))

        return EMTWithMoments()


def _perturbed_copper() -> ChemicalStructure:
    """An fcc Cu cell nudged off its minimum, so there is something to relax."""
    rng = np.random.default_rng(0)
    coords = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.85, 1.85], [1.85, 0.0, 1.85], [1.85, 1.85, 0.0]]
    ) + rng.normal(scale=0.06, size=(4, 3))
    return ChemicalStructure.with_zero_magnetic_moments(
        name="Cu4", lattice=np.diag([3.7, 3.7, 3.7]),
        cartesian_coords=coords, atomic_labels=["Cu"] * 4,
    )


@unittest.skipUnless(HAS_ASE, "ase is not installed (pip install -e '.[chgnet]')")
class ProgressAndCancellationTest(unittest.TestCase):
    """The hooks the remote server drives the optimizer through."""

    def setUp(self):
        from quick_mag.chgnet_runner import run_chgnet_calculation

        self.run = run_chgnet_calculation
        self.structure = _perturbed_copper()

    def test_progress_reports_every_step_and_the_energy_falls(self):
        seen = []
        result = self.run(
            self.structure, "atoms", optimizer="LBFGS", fmax=0.05, steps=50,
            calculator=_EMTWithMoments.build(), progress=seen.append,
        )
        self.assertTrue(result.converged)
        # One event per recorded energy, including the one taken before the
        # optimizer's first step.
        self.assertEqual(len(seen), len(result.trajectory_energies))
        self.assertEqual([event["step"] for event in seen], list(range(len(seen))))
        self.assertLess(seen[-1]["energy"], seen[0]["energy"])
        self.assertLess(seen[-1]["max_force"], seen[0]["max_force"])
        # The trace is cumulative, so a client that polls late still gets all of it.
        self.assertEqual(len(seen[-1]["trajectory_energies"]), len(seen))

    def test_a_single_point_reports_once(self):
        seen = []
        self.run(
            self.structure, "single-point",
            calculator=_EMTWithMoments.build(), progress=seen.append,
        )
        self.assertEqual(len(seen), 1)

    def test_should_stop_interrupts_the_relaxation(self):
        from quick_mag.chgnet_runner import CalculationCancelled

        calls = {"n": 0}

        def stop() -> bool:
            calls["n"] += 1
            return calls["n"] > 3

        with self.assertRaises(CalculationCancelled):
            # fmax it can never reach, so only the stop check can end this.
            self.run(
                self.structure, "atoms", optimizer="FIRE", fmax=1e-9, steps=10_000,
                calculator=_EMTWithMoments.build(), should_stop=stop,
            )

    def test_the_hooks_are_optional(self):
        # The default path must be exactly what it was before the hooks existed.
        result = self.run(
            self.structure, "cell+atoms", optimizer="LBFGS", fmax=0.05, steps=30,
            calculator=_EMTWithMoments.build(),
        )
        self.assertGreater(result.steps, 0)
        self.assertEqual(len(result.magnetic_moments), 4)
