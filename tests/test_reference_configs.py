"""Tests for the canonical reference orderings scored alongside every solve.

A and C single out one axis of the B grid. On an anisotropic cell their three
orientations are different states, so all of them must be scored — otherwise the
reference table can miss the ordering the solver actually finds.
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.classify_spin_structure import PerovskiteSiteIndexing  # noqa: E402
from quick_mag.magnetic_moments import OxidationStateAssignment  # noqa: E402
from quick_mag.reference_configs import (  # noqa: E402
    builder_spin_sign,
    canonical_reference_patterns,
    named_reference_spin_configs,
    parse_pattern,
)
from quick_mag.spin_solver import solve_for_assignment  # noqa: E402
from quick_mag.structure import ChemicalStructure  # noqa: E402

GRID_SHAPE = (2, 2, 2)


class PatternNamingTest(unittest.TestCase):
    def test_bare_names_keep_their_historical_axis(self):
        # A stacked its planes along c and C ran its chains along a before the
        # orientations were spelled out; unqualified names must not move.
        self.assertEqual(parse_pattern("A"), ("A", 2))
        self.assertEqual(parse_pattern("C"), ("C", 0))
        for i, j, k in np.ndindex(2, 2, 2):
            self.assertEqual(
                builder_spin_sign("A", i, j, k), builder_spin_sign("A(c)", i, j, k)
            )
            self.assertEqual(
                builder_spin_sign("C", i, j, k), builder_spin_sign("C(a)", i, j, k)
            )

    def test_unoriented_patterns_have_no_axis(self):
        self.assertEqual(parse_pattern("G"), ("G", -1))
        self.assertEqual(parse_pattern("F"), ("F", -1))

    def test_bad_names_are_rejected(self):
        for pattern in ("A(d)", "G(a)", "Z(b)"):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    parse_pattern(pattern)

    def test_every_orientation_is_offered(self):
        self.assertEqual(
            canonical_reference_patterns(GRID_SHAPE),
            ("G", "C(a)", "C(b)", "C(c)", "F", "A(a)", "A(b)", "A(c)"),
        )

    def test_short_axes_are_skipped(self):
        # Alternating along a length-1 axis collapses A onto F and C onto G.
        self.assertEqual(
            canonical_reference_patterns((2, 2, 1)),
            ("G", "C(a)", "C(b)", "F", "A(a)", "A(b)"),
        )

    def test_a_and_c_flip_the_same_bonds(self):
        # A(x) is antiferromagnetic along x and ferromagnetic across it; C(x) is
        # the other way round. Every bond therefore changes sign between them.
        for axis_index, axis in enumerate("abc"):
            for i, j, k in np.ndindex(2, 2, 2):
                cell = [i, j, k]
                neighbour = list(cell)
                neighbour[axis_index] = (neighbour[axis_index] + 1) % 2
                for pattern, expected in ((f"A({axis})", -1.0), (f"C({axis})", 1.0)):
                    bond = builder_spin_sign(pattern, *cell) * builder_spin_sign(
                        pattern, *neighbour
                    )
                    self.assertEqual(bond, expected, msg=f"{pattern} along {axis}")


class AnisotropicReferenceTest(unittest.TestCase):
    """On an anisotropic coupling matrix the table must contain the ground state."""

    def setUp(self):
        # A 2x2x2 B grid; sites are the first 8 atoms, in grid order.
        n_sites = 8
        self.grid = np.arange(n_sites).reshape(GRID_SHAPE)
        self.structure = ChemicalStructure.with_zero_magnetic_moments(
            name="grid",
            lattice=np.diag([7.9, 7.9, 7.9]),
            cartesian_coords=np.array(
                [
                    [3.95 * i, 3.95 * j, 3.95 * k]
                    for i, j, k in np.ndindex(*GRID_SHAPE)
                ],
                dtype=float,
            ),
            atomic_labels=["Mn"] * n_sites,
        )
        self.indexing = PerovskiteSiteIndexing(
            a_site_indices=np.array([], dtype=int),
            b_site_indices=np.arange(n_sites),
            x_site_indices=np.array([], dtype=int),
            b_grid_shape=GRID_SHAPE,
            grid_to_site=self.grid.reshape(-1),
        )
        self.assignment = OxidationStateAssignment(
            site_oxidation_states=np.full(n_sites, 3, dtype=int),
            magnetic_moments=np.ones(n_sites),
            total_energy=0.0,
            distributions={"Mn": {3: n_sites}},
        )

    def _j_matrix(self, j_per_axis):
        """Nearest-neighbour couplings with a different strength along each axis."""
        n_sites = 8
        matrix = np.zeros((n_sites, n_sites))
        for i, j, k in np.ndindex(*GRID_SHAPE):
            site = int(self.grid[i, j, k])
            for axis, coupling in enumerate(j_per_axis):
                cell = [i, j, k]
                cell[axis] = (cell[axis] + 1) % GRID_SHAPE[axis]
                neighbour = int(self.grid[tuple(cell)])
                if neighbour == site:
                    continue
                matrix[site, neighbour] = coupling
                matrix[neighbour, site] = coupling
        return matrix

    def _references(self, j_matrix):
        return named_reference_spin_configs(
            self.structure,
            self.assignment,
            j_matrix,
            list(range(8)),
            self.indexing,
        )

    def test_ground_state_appears_in_the_table_for_every_unique_axis(self):
        # J > 0 is ferromagnetic here (E = -1/2 sum J m.m), so a strong negative
        # coupling along one axis with ferromagnetic couplings across it puts the
        # ground state at A with its planes stacked along that axis.
        for axis, name in enumerate("abc"):
            with self.subTest(axis=name):
                couplings = [1.0, 1.0, 1.0]
                couplings[axis] = -3.0
                j_matrix = self._j_matrix(couplings)

                references = self._references(j_matrix)
                best_name, best = min(references, key=lambda pair: pair[1].energy)
                _base, solved = solve_for_assignment(
                    self.assignment,
                    j_matrix,
                    magnetic_site_indices=list(range(8)),
                    method="exact",
                    collinear=True,
                )
                self.assertEqual(best_name, f"A({name})")
                self.assertAlmostEqual(best.energy, solved[0].energy, places=10)

    def test_isotropic_couplings_leave_the_orientations_degenerate(self):
        references = dict(self._references(self._j_matrix([1.0, 1.0, 1.0])))
        for axis in "bc":
            self.assertAlmostEqual(
                references[f"A({axis})"].energy, references["A(a)"].energy, places=10
            )
            self.assertAlmostEqual(
                references[f"C({axis})"].energy, references["C(a)"].energy, places=10
            )


if __name__ == "__main__":
    unittest.main()
