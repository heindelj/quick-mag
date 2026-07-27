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
from quick_mag.structure_utils import read_structure  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
