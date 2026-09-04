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
    bridge_J,
    bridge_J_components,
    build_Jeff_matrix,
    build_bridges,
    default_params,
    pair_couplings,
)
from quick_mag.structure import ChemicalStructure  # noqa: E402


def _bridges(structure):
    """(bridges, params, compact index map) for a structure's best assignment."""
    ranked = enumerate_oxidation_states_by_energy(
        structure.element_symbols(), charge=0, max_mixing=2, top_k=5
    )
    assignment = expand_distribution_to_site_assignments(
        [distribution for distribution, _energy in ranked], structure
    )[0]
    descriptors = structure_ion_descriptors(structure, assignment)
    return (
        build_bridges(structure, descriptors),
        default_params(),
        {site: i for i, site in enumerate(sorted(descriptors))},
    )


def _couplings(structure):
    """(pairs, J matrix, compact index map) for a structure's best assignment."""
    bridges, params, site_index = _bridges(structure)
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


@pytest.fixture(scope="module")
def lasr3mn4o12():
    """A composition charge balance forces into mixed valence: Mn averages +3.75.

    This is the double-exchange case named in ``de_active_pairs`` -- no in-window
    oxidation distribution makes Mn single-valent, so its bridges carry the term.
    """
    return generate_high_entropy_perovskite(
        "LaSr3Mn4O12",
        a_sites=[("La", 0.25), ("Sr", 0.75)],
        b_sites=[("Mn", 1.0)],
        x_sites=[("O", 1.0)],
        a=3.9,
        n_cells_x=2,
        n_cells_y=2,
        n_cells_z=2,
        seed=0,
    )


@pytest.fixture(scope="module")
def lamno3_jahn_teller():
    """LaMnO3 with the ab-plane oxygens shifted into a Jahn-Teller pattern.

    Mn(3+) is eg^1, but a perfectly cubic cage leaves the two eg orbitals
    degenerate, so no orbital is resolved and the occupied->empty channel stays
    shut. Alternating long and short Mn-O bonds in the plane resolve it -- the
    same thing relaxing the structure does, in one line and without CHGNet.
    """
    ideal = generate_single_perovskite(
        "LaMnO3",
        a_site="La",
        b_site="Mn",
        x_site="O",
        a=4.0,
        n_cells_x=2,
        n_cells_y=2,
        n_cells_z=2,
    )
    coords = np.array(ideal.cartesian_coords, dtype=float)
    for index, element in enumerate(ideal.element_symbols()):
        if element != "O":
            continue
        # A bridging oxygen sits half a cell along the axis it bridges; push it
        # off centre, in opposite senses on neighbouring bridges.
        cell = coords[index] / 4.0
        offsets = [abs(value - round(value)) > 0.25 for value in cell]
        if offsets[0]:
            coords[index][0] += 0.16 * (
                1 if int(round(cell[1] + cell[2])) % 2 == 0 else -1
            )
        elif offsets[1]:
            coords[index][1] += 0.16 * (
                1 if int(round(cell[0] + cell[2])) % 2 == 0 else -1
            )
    return ChemicalStructure(
        name="LaMnO3-jt",
        lattice=ideal.lattice,
        cartesian_coords=coords,
        atomic_labels=list(ideal.atomic_labels),
        magnetic_moments=np.array(ideal.magnetic_moments),
        is_periodic=True,
    )


def test_bridge_components_sum_to_the_bridge_coupling(lafeo3):
    """Splitting J into its channels does not change what J is."""
    bridges, params, _site_index = _bridges(lafeo3)
    assert bridges
    for bridge in bridges:
        assert sum(bridge_J_components(bridge, params)) == bridge_J(bridge, params)


def test_pair_components_sum_to_j_eff(lafeo3, lasr3mn4o12, lamno3_jahn_teller):
    """The three bars of a pair add up to its total bar, in every regime."""
    for structure in (lafeo3, lasr3mn4o12, lamno3_jahn_teller):
        pairs, _matrix, _site_index = _couplings(structure)
        assert pairs
        for pair in pairs:
            assert pair.j_se + pair.j_de + pair.j_oe == pytest.approx(
                pair.j_eff, abs=1e-12
            )


def test_half_filled_shells_are_pure_superexchange(lafeo3):
    """Fe(3+) d5: no mixed valence and no eg^1, so only the SE channel is open."""
    pairs, _matrix, _site_index = _couplings(lafeo3)
    assert pairs
    for pair in pairs:
        assert pair.j_de == 0.0
        assert pair.j_oe == 0.0
        assert pair.j_se == pair.j_eff


def test_forced_mixed_valence_opens_double_exchange(lasr3mn4o12):
    """Double exchange fires, and it is what turns a bridge ferromagnetic."""
    pairs, _matrix, _site_index = _couplings(lasr3mn4o12)
    de_pairs = [pair for pair in pairs if pair.j_de != 0.0]
    assert de_pairs
    assert all(pair.j_de < 0.0 for pair in de_pairs)
    # The AFM superexchange is still there; DE overwhelms it.
    flipped = [pair for pair in de_pairs if pair.j_se > 0.0 > pair.j_eff]
    assert flipped


def test_jahn_teller_distortion_opens_the_occupied_empty_channel(lamno3_jahn_teller):
    """Resolving the eg orbital is what switches the Kugel-Khomskii term on.

    The ideal cubic cell leaves it shut, which is why a distorted structure has to
    be found before this channel can be seen at all.
    """
    ideal_pairs, _matrix, _site_index = _couplings(
        generate_single_perovskite(
            "LaMnO3",
            a_site="La",
            b_site="Mn",
            x_site="O",
            a=4.0,
            n_cells_x=2,
            n_cells_y=2,
            n_cells_z=2,
        )
    )
    assert all(pair.j_oe == 0.0 for pair in ideal_pairs)

    pairs, _matrix, _site_index = _couplings(lamno3_jahn_teller)
    oe_pairs = [pair for pair in pairs if pair.j_oe != 0.0]
    assert oe_pairs
    assert all(pair.j_oe < 0.0 for pair in oe_pairs)
    assert all(pair.j_de == 0.0 for pair in pairs)
