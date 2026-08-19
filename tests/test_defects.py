"""Point defects layered onto the ideal perovskite build."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from quick_mag.classify_spin_structure import (  # noqa: E402
    classify_structure,
    site_indexing_from_generation_parameters,
)
from quick_mag.defects import (  # noqa: E402
    DEFAULT_OH_BOND_LENGTH,
    PROTON_ORIENTATION_COUNT,
    SiteDefect,
    canonicalize_key,
    compensation_hint,
    proton_direction_candidates,
    resolve_key_to_indices,
)
from quick_mag.generation import generate_single_perovskite  # noqa: E402
from quick_mag.perovskite_builder import (  # noqa: E402
    SiteKey,
    build_perovskite,
    canonical_index_of_key,
    canonical_site_counts,
    canonical_site_keys,
)
from quick_mag.reference_configs import assign_b_site_spin_pattern  # noqa: E402
from quick_mag.structure import build_from_generation_parameters  # noqa: E402

GRID_SHAPES = [(0, 0, 0), (1, 0, 0), (1, 1, 1), (2, 1, 1), (2, 2, 2)]


def _build(n_oct, periodic, *, tilt="a-b+c-", angles=(10.0, 8.0, 6.0)):
    return build_perovskite(
        center=np.array([0.3, -0.2, 0.1]),
        n_oct_x=n_oct[0],
        n_oct_y=n_oct[1],
        n_oct_z=n_oct[2],
        center_to_vertex_distance_x=2.0,
        center_to_vertex_distance_y=2.1,
        center_to_vertex_distance_z=1.9,
        tilt_system=tilt,
        tilt_angle_x_deg=angles[0],
        tilt_angle_y_deg=angles[1],
        tilt_angle_z_deg=angles[2],
        periodic=periodic,
    )


def _perovskite(defects, **kwargs):
    return generate_single_perovskite(
        "test",
        a_site="La",
        b_site="Fe",
        x_site="O",
        a=kwargs.pop("a", 4.0),
        n_cells_x=kwargs.pop("n_cells_x", 2),
        n_cells_y=kwargs.pop("n_cells_y", 2),
        n_cells_z=kwargs.pop("n_cells_z", 2),
        defects=defects,
        **kwargs,
    )


class CanonicalSiteKeyTests(unittest.TestCase):
    """The key list must stay in lockstep with the build it addresses."""

    def test_keys_reproduce_build_order_and_coordinates(self) -> None:
        for n_oct in GRID_SHAPES:
            for periodic in (True, False):
                with self.subTest(n_oct=n_oct, periodic=periodic):
                    build = _build(n_oct, periodic)
                    keys = canonical_site_keys(build.octahedra.shape, periodic)
                    self.assertEqual(len(keys), len(build.all_sites))
                    self.assertEqual(
                        sum(canonical_site_counts(build.octahedra.shape, periodic)),
                        len(keys),
                    )
                    n_a = len(build.a_sites)
                    for position, key in enumerate(keys):
                        # Closed-form index must agree with the enumeration.
                        self.assertEqual(
                            canonical_index_of_key(
                                key, build.octahedra.shape, periodic
                            ),
                            position,
                        )
                        # And the key must name the coordinate that landed there.
                        if key.role == "A":
                            expected = build.a_sites[position]
                        elif key.role == "B":
                            expected = build.b_sites[position - n_a]
                        else:
                            expected = np.asarray(
                                build.octahedra[key.i, key.j, key.k]
                            )[key.vertex]
                        np.testing.assert_allclose(
                            build.all_sites[position], expected, atol=1e-12
                        )

    def test_corner_shared_oxygen_aliases_fold_onto_one_site(self) -> None:
        shape = _build((1, 1, 1), True).octahedra.shape
        # Periodic: the -a vertex of cell 0 is the +a vertex of the last cell.
        self.assertEqual(
            canonicalize_key(SiteKey("X", 0, 0, 0, 1), shape, True),
            SiteKey("X", shape[0] - 1, 0, 0, 0),
        )
        # Finite: interior negative vertices still fold ...
        self.assertEqual(
            canonicalize_key(SiteKey("X", 1, 0, 0, 1), shape, False),
            SiteKey("X", 0, 0, 0, 0),
        )
        # ... but the low face is a site of its own.
        self.assertEqual(
            canonicalize_key(SiteKey("X", 0, 0, 0, 1), shape, False),
            SiteKey("X", 0, 0, 0, 1),
        )

    def test_out_of_range_keys_resolve_to_nothing(self) -> None:
        shape = _build((1, 1, 1), False).octahedra.shape
        self.assertIsNone(canonicalize_key(SiteKey("B", 9, 0, 0), shape, False))
        self.assertEqual(
            resolve_key_to_indices(SiteKey("B", 9, 0, 0), shape, periodic=False), []
        )
        # Periodic grids do NOT wrap arbitrary indices: doing so would silently
        # move a defect onto a different site when the supercell shrinks.
        self.assertIsNone(canonicalize_key(SiteKey("B", 9, 0, 0), shape, True))
        self.assertIsNone(canonicalize_key(SiteKey("B", 2, 0, 0), shape, True))
        # The one allowed fold: a finite build's closing A plane is the image of
        # its first, so toggling periodicity keeps that defect's meaning.
        self.assertEqual(
            canonicalize_key(SiteKey("A", shape[0], 0, 0), shape, True),
            SiteKey("A", 0, 0, 0),
        )


class PeriodicImageTests(unittest.TestCase):
    """Rendering rebuilds a periodic cell as a finite one; vacancies must follow."""

    def test_boundary_images_cover_every_copy(self) -> None:
        n_oct = (1, 1, 1)
        periodic_build = _build(n_oct, True, tilt="a0a0a0", angles=(0.0, 0.0, 0.0))
        finite_build = _build(n_oct, False, tilt="a0a0a0", angles=(0.0, 0.0, 0.0))
        shape = periodic_build.octahedra.shape
        lattice = np.diag([2 * 2.0 * shape[0], 2 * 2.1 * shape[1], 2 * 1.9 * shape[2]])
        shifts = [
            a * lattice[0] + b * lattice[1] + c * lattice[2]
            for a in (-1, 0, 1)
            for b in (-1, 0, 1)
            for c in (-1, 0, 1)
        ]
        for key in canonical_site_keys(shape, True):
            reference = periodic_build.all_sites[
                resolve_key_to_indices(key, shape, periodic=True)[0]
            ]
            expanded = set(
                resolve_key_to_indices(
                    key, shape, periodic=False, expand_images=True
                )
            )
            expected = {
                index
                for index, coord in enumerate(finite_build.all_sites)
                if any(np.allclose(coord, reference + shift, atol=1e-9) for shift in shifts)
            }
            self.assertEqual(expanded, expected, msg=f"{key}")

    def test_every_spelling_of_a_site_expands_to_the_same_images(self) -> None:
        # A key is canonicalized against the *authoring* (periodic) grid before
        # images are expanded; resolving it the finite way first would leave
        # X(0,0,0,-a) with no far-face partner and the vacancy would refill.
        shape = (2, 2, 2)
        for left, right in (
            (SiteKey("X", 1, 0, 0, 0), SiteKey("X", 0, 0, 0, 1)),
            (SiteKey("A", 0, 0, 0), SiteKey("A", 2, 0, 0)),
        ):
            with self.subTest(left=left, right=right):
                self.assertEqual(
                    sorted(
                        resolve_key_to_indices(
                            left, shape, periodic=False, expand_images=True
                        )
                    ),
                    sorted(
                        resolve_key_to_indices(
                            right, shape, periodic=False, expand_images=True
                        )
                    ),
                )

    def test_non_periodic_render_keeps_its_site_roles(self) -> None:
        # The 3D view rebuilds a periodic structure finite. Its defects were
        # applied with boundary images expanded, so the site indexing has to be
        # rebuilt the same way or the B block lands on oxygens.
        from quick_mag.generation import generated_structure_from_parameters

        cases = {
            "a-vacancy": [SiteDefect("vacancy", SiteKey("A", 0, 0, 0))],
            "x-vacancy": [SiteDefect("vacancy", SiteKey("X", 1, 0, 0, 0))],
            "b-vacancy": [SiteDefect("vacancy", SiteKey("B", 1, 0, 1))],
            "substitution+proton": [
                SiteDefect("substitution", SiteKey("B", 0, 0, 0), element="Zn"),
                SiteDefect("proton", SiteKey("X", 0, 0, 0, 2)),
            ],
        }
        for name, defects in cases.items():
            with self.subTest(case=name):
                periodic = _perovskite(defects)
                rendered = generated_structure_from_parameters(
                    periodic.generation_parameters, name="render", periodic=False
                )
                indexing = site_indexing_from_generation_parameters(
                    rendered.generation_parameters,
                    build_from_generation_parameters(rendered.generation_parameters),
                )
                labels = np.asarray(rendered.atomic_labels, dtype=object)
                self.assertLessEqual(set(labels[indexing.a_site_indices]), {"La"})
                self.assertLessEqual(set(labels[indexing.b_site_indices]), {"Fe", "Zn"})
                self.assertLessEqual(set(labels[indexing.x_site_indices]), {"O"})

    def test_finite_builds_do_not_expand_images(self) -> None:
        # A genuine cluster: vacating a corner A site must remove one atom, not 8.
        shape = _build((1, 1, 1), False).octahedra.shape
        self.assertEqual(
            len(
                resolve_key_to_indices(
                    SiteKey("A", 0, 0, 0), shape, periodic=False, expand_images=False
                )
            ),
            1,
        )


class DefectApplicationTests(unittest.TestCase):
    def test_vacancy_removes_exactly_one_site(self) -> None:
        ideal = _perovskite([])
        vacated = _perovskite([SiteDefect("vacancy", SiteKey("X", 0, 0, 0, 0))])
        self.assertEqual(vacated.atom_count, ideal.atom_count - 1)
        self.assertEqual(
            vacated.element_symbols().count("O"), ideal.element_symbols().count("O") - 1
        )
        # Every surviving site is exactly where the ideal build put it.
        for coord in vacated.cartesian_coords:
            self.assertLess(
                float(np.min(np.linalg.norm(ideal.cartesian_coords - coord, axis=1))),
                1e-9,
            )

    def test_substitution_changes_only_the_label(self) -> None:
        ideal = _perovskite([])
        swapped = _perovskite([SiteDefect("substitution", SiteKey("B", 0, 1, 0), element="Zn")])
        self.assertEqual(swapped.atom_count, ideal.atom_count)
        np.testing.assert_allclose(swapped.cartesian_coords, ideal.cartesian_coords)
        differing = [
            index
            for index, (left, right) in enumerate(
                zip(ideal.atomic_labels, swapped.atomic_labels)
            )
            if left != right
        ]
        self.assertEqual(len(differing), 1)
        self.assertEqual(swapped.atomic_labels[differing[0]], "Zn")

    def test_vacancy_wins_over_a_substitution_on_the_same_site(self) -> None:
        structure = _perovskite(
            [
                SiteDefect("substitution", SiteKey("B", 0, 0, 0), element="Zn"),
                SiteDefect("vacancy", SiteKey("B", 0, 0, 0)),
            ]
        )
        self.assertNotIn("Zn", structure.element_symbols())
        self.assertEqual(structure.atom_count, _perovskite([]).atom_count - 1)

    def test_out_of_range_defect_is_skipped_not_fatal(self) -> None:
        structure = _perovskite([SiteDefect("vacancy", SiteKey("B", 9, 9, 9))])
        self.assertEqual(structure.atom_count, _perovskite([]).atom_count)


class IdealGeometryIsRetainedTests(unittest.TestCase):
    """The load-bearing requirement: defects never freeze or accumulate geometry."""

    DEFECTS = [
        SiteDefect("vacancy", SiteKey("X", 1, 0, 1, 2)),
        SiteDefect("substitution", SiteKey("B", 0, 1, 0), element="Zn"),
    ]

    def _assert_matches_ideal_minus_defects(self, **kwargs) -> None:
        ideal = _perovskite([], **kwargs)
        defected = _perovskite(self.DEFECTS, **kwargs)
        self.assertEqual(defected.atom_count, ideal.atom_count - 1)
        # Every remaining atom sits on an *ideal* site of this geometry, so the
        # defect was re-applied to a freshly generated lattice rather than
        # carried over from the previous one.
        for coord in defected.cartesian_coords:
            self.assertLess(
                float(np.min(np.linalg.norm(ideal.cartesian_coords - coord, axis=1))),
                1e-9,
            )

    def test_defects_reapply_after_a_tilt_change(self) -> None:
        for angle in (0.0, 6.0, 12.0):
            with self.subTest(angle=angle):
                self._assert_matches_ideal_minus_defects(
                    tilt_system="a-a-a-", tilt_angles_deg=(angle, angle, angle)
                )

    def test_defects_reapply_after_a_lattice_change(self) -> None:
        for edge in (3.8, 4.0, 4.6):
            with self.subTest(a=edge):
                self._assert_matches_ideal_minus_defects(a=edge)

    def test_defects_survive_shrinking_and_regrowing_the_supercell(self) -> None:
        large = _perovskite(self.DEFECTS)
        # (1,1,1) has no cell (1,0,1), so that vacancy is out of range and skipped.
        small = _perovskite(self.DEFECTS, n_cells_x=1, n_cells_y=1, n_cells_z=1)
        self.assertEqual(
            small.atom_count,
            _perovskite([], n_cells_x=1, n_cells_y=1, n_cells_z=1).atom_count,
        )
        regrown = _perovskite(self.DEFECTS)
        self.assertEqual(regrown.atom_count, large.atom_count)
        self.assertEqual(regrown.atomic_labels, large.atomic_labels)


class ProtonPlacementTests(unittest.TestCase):
    def test_cubic_candidates_are_the_two_perpendicular_axes(self) -> None:
        build = build_perovskite(
            center=np.zeros(3),
            n_oct_x=2,
            n_oct_y=2,
            n_oct_z=2,
            center_to_vertex_distance_x=2.1,
            center_to_vertex_distance_y=2.1,
            center_to_vertex_distance_z=2.1,
            tilt_system="a0a0a0",
            periodic=True,
        )
        # The +a oxygen's proton sites are +-b and +-c: the textbook four.
        candidates = proton_direction_candidates(build.octahedra, SiteKey("X", 1, 1, 1, 0))
        self.assertEqual(len(candidates), PROTON_ORIENTATION_COUNT)
        np.testing.assert_allclose(
            np.sort(candidates, axis=0),
            np.sort(
                np.array([[0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float),
                axis=0,
            ),
            atol=1e-12,
        )

    def test_proton_sits_at_the_o_h_bond_length_from_an_oxygen(self) -> None:
        structure = _perovskite([SiteDefect("proton", SiteKey("X", 0, 0, 0, 0))])
        self.assertEqual(structure.element_symbols().count("H"), 1)
        index = structure.element_symbols().index("H")
        distances = np.linalg.norm(
            structure.cartesian_coords - structure.cartesian_coords[index], axis=1
        )
        distances[index] = np.inf
        nearest = int(np.argmin(distances))
        self.assertAlmostEqual(float(distances[nearest]), DEFAULT_OH_BOND_LENGTH, places=9)
        self.assertEqual(structure.element_symbols()[nearest], "O")

    def test_proton_avoids_cations_far_better_than_the_nearest_oxygen_rule(self) -> None:
        # The naive "point at the nearest oxygen" rule lands ~1.5 A from a B
        # cation, because those oxygens sit at 45 deg to the B-O-B axis.
        for tilt, angle in (("a0a0a0", 0.0), ("a-a-a-", 8.0), ("a-b+a-", 10.0)):
            for orientation in range(PROTON_ORIENTATION_COUNT):
                with self.subTest(tilt=tilt, orientation=orientation):
                    structure = _perovskite(
                        [
                            SiteDefect(
                                "proton",
                                SiteKey("X", 1, 1, 1, 0),
                                orientation=orientation,
                            )
                        ],
                        tilt_system=tilt,
                        tilt_angles_deg=(angle, angle, angle),
                    )
                    index = structure.element_symbols().index("H")
                    symbols = structure.element_symbols()
                    cations = [
                        position
                        for position, symbol in enumerate(symbols)
                        if symbol in ("La", "Fe")
                    ]
                    closest = float(
                        np.min(
                            np.linalg.norm(
                                structure.cartesian_coords[cations]
                                - structure.cartesian_coords[index],
                                axis=1,
                            )
                        )
                    )
                    self.assertGreater(closest, 1.5)

    def test_orientation_selects_distinct_sites_and_follows_a_tilt(self) -> None:
        positions = []
        for orientation in range(PROTON_ORIENTATION_COUNT):
            structure = _perovskite(
                [SiteDefect("proton", SiteKey("X", 0, 0, 0, 0), orientation=orientation)]
            )
            positions.append(structure.cartesian_coords[-1])
        for left in range(len(positions)):
            for right in range(left + 1, len(positions)):
                self.assertGreater(
                    float(np.linalg.norm(positions[left] - positions[right])), 1e-6
                )

        # The O-H vector rotates rigidly with its octahedron, so the bond length
        # is preserved and the proton moves with the cage.
        untilted = _perovskite([SiteDefect("proton", SiteKey("X", 0, 0, 0, 0))])
        tilted = _perovskite(
            [SiteDefect("proton", SiteKey("X", 0, 0, 0, 0))],
            tilt_system="a-a-a-",
            tilt_angles_deg=(12.0, 12.0, 12.0),
        )
        self.assertGreater(
            float(np.linalg.norm(tilted.cartesian_coords[-1] - untilted.cartesian_coords[-1])),
            1e-3,
        )

    def test_proton_on_a_vacated_oxygen_is_dropped(self) -> None:
        structure = _perovskite(
            [
                SiteDefect("vacancy", SiteKey("X", 0, 0, 0, 0)),
                SiteDefect("proton", SiteKey("X", 0, 0, 0, 0)),
            ]
        )
        self.assertNotIn("H", structure.element_symbols())

    def test_crowding_is_reported_only_when_real(self) -> None:
        from quick_mag.defects import apply_defects
        from quick_mag.structure import build_from_generation_parameters

        def warnings_for(a, tilt, angle):
            structure = _perovskite(
                [SiteDefect("proton", SiteKey("X", 1, 1, 1, 0), orientation=3)],
                a=a,
                tilt_system=tilt,
                tilt_angles_deg=(angle, angle, angle),
            )
            params = structure.generation_parameters
            build = build_from_generation_parameters(params)
            _, _, _, resolution = apply_defects(
                build,
                ["La"] * 8 + ["Fe"] * 8 + ["O"] * 24,
                periodic=True,
                stored_periodic=True,
                defects=params.defects,
                cell_origin=params.cell_origin,
            )
            return resolution.warnings

        # The host oxygen sits at the O-H bond length and must never be counted
        # as a contact, or every proton would warn.
        self.assertEqual(warnings_for(4.0, "a0a0a0", 0.0), [])
        self.assertEqual(warnings_for(4.0, "a-a-a-", 12.0), [])
        # A genuinely crowded cell does warn.
        self.assertTrue(warnings_for(2.6, "a-a-a-", 30.0))

    def test_proton_next_to_a_b_vacancy_still_places(self) -> None:
        # Directions come from the ideal cage, which exists whether or not the
        # B site it belongs to was removed -- no under-coordination fallback.
        structure = _perovskite(
            [
                SiteDefect("vacancy", SiteKey("B", 0, 0, 0)),
                SiteDefect("proton", SiteKey("X", 0, 0, 0, 0)),
            ]
        )
        self.assertEqual(structure.element_symbols().count("H"), 1)


class SiteIndexingWithDefectsTests(unittest.TestCase):
    def _indexing(self, defects):
        structure = _perovskite(defects)
        params = structure.generation_parameters
        build = build_from_generation_parameters(params)
        return structure, site_indexing_from_generation_parameters(params, build)

    def test_roles_stay_correct_for_every_defect_kind(self) -> None:
        cases = {
            "none": [],
            "a-vacancy": [SiteDefect("vacancy", SiteKey("A", 0, 0, 0))],
            "b-vacancy": [SiteDefect("vacancy", SiteKey("B", 1, 0, 1))],
            "x-vacancy": [SiteDefect("vacancy", SiteKey("X", 0, 0, 0, 0))],
            "proton": [SiteDefect("proton", SiteKey("X", 0, 0, 0, 0))],
        }
        for name, defects in cases.items():
            with self.subTest(case=name):
                structure, indexing = self._indexing(defects)
                labels = np.asarray(structure.atomic_labels, dtype=object)
                self.assertTrue(np.all(labels[indexing.a_site_indices] == "La"))
                self.assertTrue(np.all(labels[indexing.b_site_indices] == "Fe"))
                self.assertTrue(np.all(labels[indexing.x_site_indices] == "O"))
                # The B grid stays a full lattice whatever was removed.
                self.assertEqual(len(indexing.grid_to_site), 8)

    def test_b_vacancy_leaves_a_hole_in_the_grid(self) -> None:
        _, indexing = self._indexing([SiteDefect("vacancy", SiteKey("B", 1, 0, 1))])
        self.assertEqual(int(np.sum(indexing.grid_to_site < 0)), 1)
        self.assertEqual(int(np.sum(indexing.grid_present)), 7)
        self.assertEqual(len(indexing.b_site_indices), 7)

    def test_vacated_grid_cell_never_writes_to_the_last_atom(self) -> None:
        # -1 is a sentinel, not a Python negative index.
        structure, indexing = self._indexing([SiteDefect("vacancy", SiteKey("B", 1, 0, 1))])
        before = structure.magnetic_moments[-1].copy()
        assign_b_site_spin_pattern(
            structure, None, pattern="G", moment_magnitude=5.0, site_indexing=indexing
        )
        np.testing.assert_allclose(structure.magnetic_moments[-1], before)

    def test_classification_survives_a_b_vacancy(self) -> None:
        structure, indexing = self._indexing([SiteDefect("vacancy", SiteKey("B", 1, 0, 1))])
        assign_b_site_spin_pattern(
            structure, None, pattern="G", moment_magnitude=5.0, site_indexing=indexing
        )
        self.assertEqual(classify_structure(structure, site_indexing=indexing).label, "G")


class ChargeCompensationTests(unittest.TestCase):
    def _hint(self, defects):
        reference = _perovskite([]).atomic_labels
        return compensation_hint(reference, _perovskite(defects).atomic_labels)

    def test_stoichiometric_cell_is_balanced(self) -> None:
        self.assertEqual(self._hint([])[0], 0)

    def test_aliovalent_substitution_calls_for_a_proton(self) -> None:
        deficit, message = self._hint(
            [SiteDefect("substitution", SiteKey("B", 0, 0, 0), element="Zn")]
        )
        self.assertEqual(deficit, -1)
        self.assertIn("add 1 proton", message)

    def test_a_proton_restores_the_balance(self) -> None:
        deficit, _ = self._hint(
            [
                SiteDefect("substitution", SiteKey("B", 0, 0, 0), element="Zn"),
                SiteDefect("proton", SiteKey("X", 0, 0, 0, 0)),
            ]
        )
        self.assertEqual(deficit, 0)

    def test_oxygen_vacancy_is_compensated_by_reduction_not_protons(self) -> None:
        deficit, message = self._hint([SiteDefect("vacancy", SiteKey("X", 0, 0, 0, 0))])
        self.assertEqual(deficit, 2)
        self.assertIn("reducing", message)

    def test_enumerator_reduces_iron_for_an_oxygen_vacancy(self) -> None:
        from quick_mag.oxidation_state_energy import enumerate_oxidation_states_by_energy

        structure = _perovskite([SiteDefect("vacancy", SiteKey("X", 0, 0, 0, 0))])
        best = enumerate_oxidation_states_by_energy(structure.atomic_labels, 0, top_k=1)
        self.assertTrue(best)
        iron = dict(best[0][0]["Fe"])
        self.assertEqual(iron.get(2), 2)

    def test_enumerator_keeps_iron_trivalent_when_a_proton_compensates(self) -> None:
        from quick_mag.oxidation_state_energy import enumerate_oxidation_states_by_energy

        structure = _perovskite(
            [
                SiteDefect("substitution", SiteKey("B", 0, 0, 0), element="Zn"),
                SiteDefect("proton", SiteKey("X", 0, 0, 0, 0)),
            ]
        )
        best = enumerate_oxidation_states_by_energy(structure.atomic_labels, 0, top_k=1)
        self.assertTrue(best)
        self.assertEqual(dict(best[0][0]["Fe"]), {3: 7})
        self.assertEqual(dict(best[0][0]["H"]), {1: 1})


if __name__ == "__main__":
    unittest.main()
