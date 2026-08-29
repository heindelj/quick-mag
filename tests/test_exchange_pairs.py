from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quick_mag.generation import (  # noqa: E402
    generate_double_perovskite,
    generate_high_entropy_perovskite,
    generate_single_perovskite,
)
from quick_mag.ion_descriptors import structure_ion_descriptors  # noqa: E402
from quick_mag.magnetic_moments import (  # noqa: E402
    expand_distribution_to_site_assignments,
)
from quick_mag.oxidation_state_energy import (  # noqa: E402
    enumerate_oxidation_states_by_energy,
)
from quick_mag.polarization_model import (  # noqa: E402
    build_Jeff_matrix,
    build_bridges,
    default_params,
    pair_couplings,
)


def _couplings(structure):
    """(pairs, J matrix, compact index map) for a structure's best assignment."""
    ranked = enumerate_oxidation_states_by_energy(
        structure.element_symbols(), charge=0, max_mixing=2, top_k=5
    )
    assignment = expand_distribution_to_site_assignments(
        [distribution for distribution, _energy in ranked], structure
    )[0]
    descriptors = structure_ion_descriptors(structure, assignment)
    bridges = build_bridges(structure, descriptors)
    params = default_params()
    site_index = {site: i for i, site in enumerate(sorted(descriptors))}
    return (
        pair_couplings(bridges, params),
        build_Jeff_matrix(bridges, site_index, params),
        site_index,
    )


@pytest.fixture(scope="module")
def lafeo3():
    return generate_single_perovskite(
        "LaFeO3",
        a_site="La",
        b_site="Fe",
        x_site="O",
        a=4.0,
        n_cells_x=2,
        n_cells_y=2,
        n_cells_z=2,
    )


@pytest.fixture(scope="module")
def la2femno6():
    return generate_double_perovskite(
        "La2FeMnO6",
        a_site="La",
        b_site="Fe",
        b2_site="Mn",
        x_site="O",
        a=4.0,
        n_cells_x=2,
        n_cells_y=2,
        n_cells_z=2,
    )


def test_pair_couplings_sum_to_the_matrix(lafeo3):
    """Every pair's j_eff is exactly the matrix element it was summed into."""
    pairs, matrix, site_index = _couplings(lafeo3)
    assert pairs
    for pair in pairs:
        a, b = site_index[pair.site_i], site_index[pair.site_j]
        assert pair.j_eff == pytest.approx(matrix[a, b], abs=1e-12)
        assert matrix[a, b] == pytest.approx(matrix[b, a], abs=1e-12)


def test_pair_couplings_cover_exactly_the_nonzero_elements(lafeo3):
    """No bridged pair is missing, and no unbridged pair is invented."""
    pairs, matrix, site_index = _couplings(lafeo3)
    listed = {(site_index[p.site_i], site_index[p.site_j]) for p in pairs}
    nonzero = {
        (i, j)
        for i in range(matrix.shape[0])
        for j in range(i + 1, matrix.shape[0])
        if matrix[i, j] != 0.0
    }
    assert listed == nonzero


def test_pairs_are_unique_and_ordered(lafeo3):
    """One entry per pair, always with site_i < site_j."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    keys = [(p.site_i, p.site_j) for p in pairs]
    assert len(keys) == len(set(keys))
    assert all(i < j for i, j in keys)


def test_distance_matches_the_minimum_image_separation(lafeo3):
    """The law-of-cosines distance off the bridge equals the real M-M distance."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    fractional = lafeo3.fractional_coords
    for pair in pairs:
        delta = (
            fractional[pair.site_j] - fractional[pair.site_i] + 0.5
        ) % 1.0 - 0.5
        expected = float(np.linalg.norm(delta @ lafeo3.lattice))
        assert pair.distance == pytest.approx(expected, abs=1e-6)


def test_metal_symbols_follow_the_site_order(lafeo3):
    """metal_i/metal_j name the elements at site_i/site_j, not the bridge's order."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    symbols = lafeo3.element_symbols()
    for pair in pairs:
        assert pair.metal_i == symbols[pair.site_i]
        assert pair.metal_j == symbols[pair.site_j]


def test_bridge_provenance_is_recorded(lafeo3):
    """Ligands and angles are one per bridge, and bridge_count agrees."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    for pair in pairs:
        assert pair.bridge_count == len(pair.ligands) == len(pair.angles_deg)
        assert pair.bridge_count >= 1
        assert set(pair.ligands) == {"O"}
        assert all(0.0 <= angle <= 180.0 for angle in pair.angles_deg)


def test_cubic_lafeo3_nearest_neighbour_coupling_is_antiferromagnetic(lafeo3):
    """A 180-degree Fe(3+)-O-Fe(3+) bridge is AFM, which is J > 0 in this convention."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    assert all(pair.j_eff > 0.0 for pair in pairs)


def test_rock_salt_double_perovskite_couples_only_across_species(la2femno6):
    """Every B site's six bridged neighbours are B', so Fe-Mn is the only pairing.

    Fe-Fe and Mn-Mn are second neighbours in the rock-salt ordering, with no single
    ligand between them, so they are absent rather than small.
    """
    pairs, _matrix, _site_index = _couplings(la2femno6)
    pairings = {
        " - ".join(sorted((pair.metal_i, pair.metal_j))) for pair in pairs
    }
    assert pairings == {"Fe - Mn"}


def test_mixed_b_site_yields_several_element_pairings():
    """A disordered B site gives the plot more than one category to colour."""
    structure = generate_high_entropy_perovskite(
        "LaFeMnO3",
        a_sites=[("La", 1.0)],
        b_sites=[("Fe", 0.5), ("Mn", 0.5)],
        x_sites=[("O", 1.0)],
        a=4.0,
        n_cells_x=2,
        n_cells_y=2,
        n_cells_z=2,
        seed=0,
    )
    pairs, _matrix, _site_index = _couplings(structure)
    pairings = {
        " - ".join(sorted((pair.metal_i, pair.metal_j))) for pair in pairs
    }
    assert len(pairings) > 1
    assert pairings <= {"Fe - Fe", "Fe - Mn", "Mn - Mn"}


def test_pair_couplings_is_empty_without_bridges():
    """No bridges, no pairs -- and no exception."""
    assert pair_couplings([], default_params()) == []
