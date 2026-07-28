"""Tests for chaining ``quick-mag`` commands with ``::``.

These need neither CHGNet nor ASE: they cover the argv splitting, the rules for
which stages may follow which, and the "only the last stage writes" contract,
using ``build :: solve`` as the end-to-end case.
"""

import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag import cli  # noqa: E402


def run_cli(*argv):
    """Run ``quick-mag`` with ``argv``, returning (exit status, combined output)."""
    stream = io.StringIO()
    with redirect_stdout(stream), redirect_stderr(stream):
        status = cli.main(list(argv))
    return status, stream.getvalue()


# A small, fast build: one 1x1x1 LaFeO3 cell.
BUILD_ARGS = ["build", "--a-site", "La", "--b-site", "Fe", "--x-site", "O", "--a", "3.9"]


class SplitChainTest(unittest.TestCase):
    def test_single_stage_is_one_segment(self):
        self.assertEqual(cli.split_chain(["solve", "a.cif"]), [["solve", "a.cif"]])

    def test_splits_on_bare_token_only(self):
        self.assertEqual(
            cli.split_chain(["build", "--name", "a::b", "::", "solve"]),
            [["build", "--name", "a::b"], ["solve"]],
        )

    def test_multiple_stages(self):
        self.assertEqual(
            cli.split_chain(["build", "::", "chgnet", "::", "solve"]),
            [["build"], ["chgnet"], ["solve"]],
        )

    def test_empty_segments_are_rejected(self):
        for argv in (
            ["::", "solve"],
            ["build", "::"],
            ["build", "::", "::", "solve"],
        ):
            with self.subTest(argv=argv):
                with self.assertRaises(ValueError):
                    cli.split_chain(argv)


class ChainValidationTest(unittest.TestCase):
    """Every rejected chain must exit nonzero and say why."""

    def _assert_rejected(self, argv, fragment):
        status, output = run_cli(*argv)
        self.assertNotEqual(status, 0)
        self.assertIn(fragment, output)

    def test_solve_must_be_last(self):
        self._assert_rejected(
            ["solve", "a.cif", "::", "build"], "must be the last stage"
        )
        self._assert_rejected(
            ["solve", "a.cif", "::", "chgnet"], "must be the last stage"
        )

    def test_build_must_be_first(self):
        self._assert_rejected(
            ["chgnet", "a.cif", "::", "build"], "cannot follow 'chgnet'"
        )

    def test_ui_never_chains(self):
        self._assert_rejected(["ui", "::", "solve", "a.cif"], "cannot be part of")

    def test_chained_stage_may_not_take_files(self):
        self._assert_rejected(
            BUILD_ARGS + ["::", "solve", "a.cif"],
            "receives its structures from the previous stage",
        )

    def test_chgnet_may_follow_build_and_precede_solve(self):
        stages = [
            cli.build_parser().parse_args(segment)
            for segment in (["build"], ["chgnet"], ["solve"])
        ]
        cli.validate_chain(stages)  # must not raise


class ChainOutputTest(unittest.TestCase):
    """Only the last stage writes, unless a stage was given an explicit -o."""

    def setUp(self):
        self._cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def test_build_alone_still_writes_its_default_directory(self):
        status, _ = run_cli(*BUILD_ARGS)
        self.assertEqual(status, 0)
        self.assertTrue(list(Path("built_structures").glob("*.cif")))

    def test_chained_build_writes_nothing(self):
        status, output = run_cli(*BUILD_ARGS, "::", "solve", "--max-configs", "1")
        self.assertEqual(status, 0)
        self.assertEqual(sorted(os.listdir(".")), [])
        # The structure really did reach the solver.
        self.assertIn("Ground state", output)

    def test_explicit_output_dir_writes_mid_chain(self):
        status, _ = run_cli(
            *BUILD_ARGS, "-o", "raw", "::", "solve", "--max-configs", "1"
        )
        self.assertEqual(status, 0)
        self.assertTrue(list(Path("raw").glob("*.cif")))


if __name__ == "__main__":
    unittest.main()
