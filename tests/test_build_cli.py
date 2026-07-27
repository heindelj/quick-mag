"""Tests for the ``quick-mag build`` command-line builder.

These are pure (no imgui): they exercise scan-spec parsing, element Cartesian
expansion, high-entropy sampling, and the ``--zip`` lockstep rule by running
``build_cli.main`` into a temp directory and re-reading the written CIFs.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag import build_cli  # noqa: E402
from quick_mag.generation import (  # noqa: E402
    generate_high_entropy_perovskite,
    generated_structure_from_parameters,
)
from quick_mag.structure_utils import read_structure  # noqa: E402


class HighEntropySeedTest(unittest.TestCase):
    """The seed must be recorded as provenance so a rebuild is bit-identical."""

    def _build(self, **kwargs):
        return generate_high_entropy_perovskite(
            "HEA",
            a_sites=[("La", 1.0)],
            b_sites=[("Cr", 0.2), ("Mn", 0.2), ("Fe", 0.2), ("Co", 0.2), ("Ni", 0.2)],
            x_sites=[("O", 1.0)],
            a=3.9,
            n_cells_x=2, n_cells_y=2, n_cells_z=2,
            **kwargs,
        )

    def test_seed_defaults_to_zero(self):
        self.assertEqual(self._build().generation_parameters.high_entropy_seed, 0)

    def test_seed_is_stored_and_rebuilds_identically(self):
        for seed in (0, 1, 42):
            with self.subTest(seed=seed):
                structure = self._build(seed=seed, sample_index=1)
                params = structure.generation_parameters
                self.assertEqual(params.high_entropy_seed, seed)
                rebuilt = generated_structure_from_parameters(
                    params, name="HEA", periodic=True
                )
                self.assertEqual(
                    list(rebuilt.atomic_labels), list(structure.atomic_labels)
                )

    def test_distinct_seeds_give_distinct_occupancies(self):
        labels = {
            seed: tuple(self._build(seed=seed).atomic_labels) for seed in (0, 1, 42)
        }
        self.assertEqual(len(set(labels.values())), 3)


class ParseScanSpecTest(unittest.TestCase):
    def test_inclusive_linspace(self):
        points = build_cli.parse_scan_spec("3.8:4.2:5", name="--a")
        self.assertEqual(len(points), 5)
        self.assertAlmostEqual(points[0], 3.8)
        self.assertAlmostEqual(points[-1], 4.2)  # endpoint included

    def test_scalar_passthrough(self):
        self.assertEqual(build_cli.parse_scan_spec("4.0", name="--a"), [4.0])

    def test_integer_axis_rounds_and_dedups(self):
        points = build_cli.parse_scan_spec("1:2:4", name="--n-cells-x", integer=True)
        self.assertEqual(points, [1, 2])  # 1, 1.33, 1.67, 2 -> {1, 2}

    def test_bad_specs_raise(self):
        with self.assertRaises(ValueError):
            build_cli.parse_scan_spec("1:2", name="--a")
        with self.assertRaises(ValueError):
            build_cli.parse_scan_spec("1:2:0", name="--a")
        with self.assertRaises(ValueError):
            build_cli.parse_scan_spec("x:y:z", name="--a")


class HighEntropyDistributionTest(unittest.TestCase):
    def test_weights_and_bare_symbols(self):
        entries = build_cli.parse_high_entropy_distribution("Fe:0.5,Co", name="--b-sites")
        self.assertEqual(entries, [("Fe", 0.5), ("Co", 1.0)])

    def test_invalid_symbol_raises(self):
        with self.assertRaises(ValueError):
            build_cli.parse_high_entropy_distribution("Xx:1", name="--b-sites")


class RunBuildTest(unittest.TestCase):
    def _run(self, argv):
        code = build_cli.main(argv)
        self.assertEqual(code, 0)

    def _cifs(self, out_dir):
        return sorted(Path(out_dir).glob("*.cif"))

    def test_element_product_times_scan(self):
        with tempfile.TemporaryDirectory() as out:
            # 2 A-site x 2 B-site x 3 lattice points = 12 structures.
            self._run([
                "--formula", "perovskite",
                "--a-site", "La,Sr", "--b-site", "Fe,Co", "--x-site", "O",
                "--a", "3.8:4.2:3", "-o", out,
            ])
            cifs = self._cifs(out)
            self.assertEqual(len(cifs), 12)
            # Every written CIF must re-read as a valid structure.
            for cif in cifs:
                structure = read_structure(cif)
                self.assertGreater(structure.atom_count, 0)

    def test_num_samples_distinct_and_reproducible(self):
        args = [
            "--formula", "high_entropy",
            "--b-sites", "Fe:0.5,Co:0.3,Ni:0.2", "--x-sites", "O",
            "--num-samples", "3",
            "--n-cells-x", "2", "--n-cells-y", "2", "--n-cells-z", "2",
        ]
        with tempfile.TemporaryDirectory() as out1, tempfile.TemporaryDirectory() as out2:
            self._run(args + ["-o", out1])
            self._run(args + ["-o", out2])
            cifs1 = self._cifs(out1)
            self.assertEqual(len(cifs1), 3)

            labels = [tuple(read_structure(c).atomic_labels) for c in cifs1]
            # The three samples must differ from one another.
            self.assertEqual(len(set(labels)), 3)
            # ... and be reproducible across identical invocations.
            labels2 = [tuple(read_structure(c).atomic_labels) for c in self._cifs(out2)]
            self.assertEqual(labels, labels2)

    def test_seed_changes_the_sampled_occupancies(self):
        """A different --seed must give a different, still reproducible, family."""
        args = [
            "--formula", "high_entropy",
            "--b-sites", "Cr:0.2,Mn:0.2,Fe:0.2,Co:0.2,Ni:0.2", "--x-sites", "O",
            "--num-samples", "3",
            "--n-cells-x", "2", "--n-cells-y", "2", "--n-cells-z", "2",
        ]
        def labels(out):
            return [tuple(read_structure(c).atomic_labels) for c in self._cifs(out)]

        with tempfile.TemporaryDirectory() as d0, \
             tempfile.TemporaryDirectory() as d0_again, \
             tempfile.TemporaryDirectory() as d1:
            self._run(args + ["-o", d0])                      # default seed 0
            self._run(args + ["--seed", "0", "-o", d0_again])  # explicit 0
            self._run(args + ["--seed", "1", "-o", d1])

            # Default and explicit seed 0 agree, and seed 0 is reproducible.
            self.assertEqual(labels(d0), labels(d0_again))
            # A different seed changes every sample.
            self.assertEqual(len(labels(d1)), 3)
            for before, after in zip(labels(d0), labels(d1)):
                self.assertNotEqual(before, after)

    def test_nonzero_seed_is_tagged_in_the_filename(self):
        """Different seeds must be able to share an output directory."""
        args = [
            "--formula", "high_entropy", "--b-sites", "Fe:0.5,Co:0.5",
            "--n-cells-x", "2", "--n-cells-y", "2", "--n-cells-z", "2",
        ]
        with tempfile.TemporaryDirectory() as out:
            self._run(args + ["-o", out])                 # seed 0 -> plain name
            self._run(args + ["--seed", "3", "-o", out])  # seed 3 -> tagged name
            names = {c.stem for c in self._cifs(out)}
            self.assertEqual(names, {"HEA", "HEA_seed3"})

    def test_zip_lockstep_count(self):
        with tempfile.TemporaryDirectory() as out:
            self._run([
                "--a", "3.8:4.2:3", "--tilt-z", "0:10:3", "--zip", "-o", out,
            ])
            self.assertEqual(len(self._cifs(out)), 3)  # lockstep, not 3x3

    def test_zip_mismatched_lengths_errors(self):
        with tempfile.TemporaryDirectory() as out:
            code = build_cli.main([
                "--a", "3.8:4.2:3", "--tilt-z", "0:10:5", "--zip", "-o", out,
            ])
            self.assertEqual(code, 1)
            self.assertEqual(len(self._cifs(out)), 0)

    def test_default_cartesian_grid(self):
        with tempfile.TemporaryDirectory() as out:
            self._run([
                "--a", "3.8:4.2:3", "--tilt-z", "0:10:2", "-o", out,
            ])
            self.assertEqual(len(self._cifs(out)), 6)  # 3 x 2 grid

    def test_ordered_modes_default_to_an_even_supercell(self):
        """double/quadruple/dq must default to 2x2x2 so both species appear.

        At 1x1x1 the alternating sublattice collapses and the second species is
        dropped entirely, so the default has to follow --formula.
        """
        expected_species = {
            "double": {"La", "Fe", "Co", "O"},      # A2 B'B'' X6
            "quadruple": {"La", "Sr", "Fe", "O"},   # A A'3 B4 X12
            "dq": {"La", "Sr", "Fe", "Co", "O"},    # A A'3 B B' X12
        }
        for formula, species in expected_species.items():
            with self.subTest(formula=formula), tempfile.TemporaryDirectory() as out:
                self._run(["--formula", formula, "-o", out])
                cifs = self._cifs(out)
                self.assertEqual(len(cifs), 1)
                structure = read_structure(cifs[0])
                self.assertEqual(structure.atom_count, 40)  # 2x2x2
                self.assertEqual(set(structure.element_symbols()), species)

    def test_single_and_high_entropy_modes_still_default_to_one_cell(self):
        for formula in ("perovskite", "high_entropy"):
            with self.subTest(formula=formula), tempfile.TemporaryDirectory() as out:
                self._run(["--formula", formula, "-o", out])
                structure = read_structure(self._cifs(out)[0])
                self.assertEqual(structure.atom_count, 5)  # 1x1x1

    def test_explicit_n_cells_overrides_the_mode_default(self):
        with tempfile.TemporaryDirectory() as out:
            self._run([
                "--formula", "double",
                "--n-cells-x", "4", "--n-cells-y", "4", "--n-cells-z", "4",
                "-o", out,
            ])
            self.assertEqual(read_structure(self._cifs(out)[0]).atom_count, 320)


if __name__ == "__main__":
    unittest.main()
