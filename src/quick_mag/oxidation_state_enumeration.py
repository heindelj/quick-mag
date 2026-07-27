from __future__ import annotations

import numpy as np
from collections import Counter
from itertools import combinations
from typing import Iterator, Sequence

try:
    from .radii import SHANNON_IONIC_RADII
except ImportError:
    from quick_mag.radii import SHANNON_IONIC_RADII

OxidationStateDistribution = dict[str, list[tuple[int, int]]]

# Pauling electronegativities (en.wikipedia.org/wiki/Electronegativities_of_the_elements_(data_page)).
# Elements with no reported Pauling value (He, Ne, Ar, Rn, Fr) are omitted.
# For elements reported as a range (Pm, Eu, Yb) the midpoint is used.
ELECTRONEG: dict[str, float] = {
    'H':  2.20,
    'Li': 0.98, 'Be': 1.57, 'B':  2.04, 'C':  2.55, 'N':  3.04, 'O':  3.44, 'F':  3.98,
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P':  2.19, 'S':  2.58, 'Cl': 3.16,
    'K':  0.82, 'Ca': 1.00,
    'Sc': 1.36, 'Ti': 1.54, 'V':  1.63, 'Cr': 1.66, 'Mn': 1.55, 'Fe': 1.83, 'Co': 1.88,
    'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65,
    'Ga': 1.81, 'Ge': 2.01, 'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 3.00,
    'Rb': 0.82, 'Sr': 0.95,
    'Y':  1.22, 'Zr': 1.33, 'Nb': 1.60, 'Mo': 2.16, 'Tc': 1.90, 'Ru': 2.20, 'Rh': 2.28,
    'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69,
    'In': 1.78, 'Sn': 1.96, 'Sb': 2.05, 'Te': 2.10, 'I':  2.66, 'Xe': 2.60,
    'Cs': 0.79, 'Ba': 0.89,
    'La': 1.10, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.15, 'Sm': 1.17, 'Eu': 1.15,
    'Gd': 1.20, 'Tb': 1.10, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24, 'Tm': 1.25, 'Yb': 1.15,
    'Lu': 1.27,
    'Hf': 1.30, 'Ta': 1.50, 'W':  2.36, 'Re': 1.90, 'Os': 2.20, 'Ir': 2.20, 'Pt': 2.28,
    'Au': 2.54, 'Hg': 2.00,
    'Tl': 1.62, 'Pb': 2.33, 'Bi': 2.02, 'Po': 2.00, 'At': 2.20,
    'Ra': 0.90,
    'Ac': 1.10, 'Th': 1.30, 'Pa': 1.50, 'U':  1.38, 'Np': 1.36, 'Pu': 1.28, 'Am': 1.30,
    'Cm': 1.30, 'Bk': 1.30, 'Cf': 1.30, 'Es': 1.30, 'Fm': 1.30, 'Md': 1.30, 'No': 1.30,
    'Lr': 1.30,
}

# Per-element oxidation-state log-probabilities derived from pymatgen's
# icsd_bv.yaml occurrence counts (score = log(count / total_for_element);
# less negative = more common).
ELEMENT_OXI_SCORES: dict[str, dict[int, float]] = {
    "Ag": {1: -0.0495, 2: -3.4569, 3: -4.0894},
    "Al": {3: 0.0},
    "As": {-3: -1.1414, -2: -3.1538, -1: -3.5593, 2: -3.8106, 3: -1.7267, 5: -0.8929},
    "B": {-3: -3.6559, -2: -4.9411, 1: -4.6157, 2: -4.5616, 3: -0.0548},
    "Ba": {2: 0.0},
    "Be": {2: 0.0},
    "Bi": {1: -3.8252, 2: -4.2047, 3: -0.0938, 5: -2.941},
    "Br": {-1: -0.0249, 1: -5.6264, 3: -5.3387, 5: -4.3271, 7: -5.8087},
    "C": {-4: -2.0341, -3: -3.8586, -2: -2.5041, -1: -4.3286, 1: -4.5518, 2: -1.6286, 3: -2.3657, 4: -0.7929},
    "Ca": {2: 0.0},
    "Cd": {2: 0.0},
    "Ce": {2: -3.2027, 3: -0.2478, 4: -1.7211},
    "Cl": {-1: -0.0195, 3: -5.4065, 5: -5.3577, 7: -4.5956},
    "Co": {1: -3.0531, 2: -0.336, 3: -1.6116, 4: -3.2538},
    "Cr": {2: -1.8112, 3: -0.6585, 4: -2.8252, 5: -2.6075, 6: -1.6826},
    "Cs": {1: 0.0},
    "Cu": {1: -0.6824, 2: -0.7428, 3: -3.9737},
    "Dy": {2: -3.9627, 3: -0.0192},
    "Er": {3: 0.0},
    "Eu": {2: -0.8387, 3: -0.5661},
    "F": {-1: 0.0},
    "Fe": {1: -4.334, 2: -0.99, 3: -0.5108, 4: -4.1799},
    "Ga": {1: -4.8919, 2: -3.5569, 3: -0.0367},
    "Gd": {3: 0.0},
    "Ge": {-4: -4.6444, 2: -2.792, 3: -3.0862, 4: -0.124},
    "H": {-1: -1.024, 1: -0.4449},
    "Hf": {2: -3.2426, 4: -0.0398},
    "Hg": {1: -1.9315, 2: -0.1566},
    "Ho": {3: 0.0},
    "I": {-1: -0.1081, 1: -4.8775, 3: -5.04, 5: -2.5261, 7: -4.7822},
    "In": {1: -2.2044, 2: -3.2405, 3: -0.1619},
    "Ir": {3: -1.7373, 4: -0.803, 5: -1.1648, 6: -2.7489},
    "K": {1: 0.0},
    "La": {2: -3.9773, 3: -0.0189},
    "Li": {1: 0.0},
    "Lu": {3: 0.0},
    "Mg": {2: 0.0},
    "Mn": {2: -0.5419, 3: -1.3545, 4: -2.0555, 5: -4.5971, 6: -4.7025, 7: -4.3348},
    "Mo": {2: -3.3637, 3: -2.3743, 4: -2.7057, 5: -2.2884, 6: -0.3509},
    "N": {-3: -0.2514, -2: -3.603, -1: -3.3878, 1: -4.3651, 2: -4.976, 3: -2.7672, 5: -2.5405},
    "Na": {1: 0.0},
    "Nb": {2: -3.8273, 3: -2.6878, 4: -1.9459, 5: -0.2648},
    "Nd": {2: -3.6037, 3: -0.0276},
    "Ni": {1: -3.3814, 2: -0.2058, 3: -2.0875, 4: -3.5756},
    "O": {-2: -0.0023, -1: -6.0967},
    "P": {-3: -2.3186, -2: -4.0366, -1: -3.5376, 1: -5.3657, 2: -4.7904, 3: -4.022, 4: -3.1028, 5: -0.2497},
    "Pb": {2: -0.0917, 4: -2.4348},
    "Pd": {2: -0.1329, 3: -3.4054, 4: -2.3938},
    "Pr": {3: -0.071, 4: -2.6806},
    "Rb": {1: 0.0},
    "Re": {2: -3.9741, 3: -1.5581, 4: -1.8946, 5: -1.7123, 6: -1.8946, 7: -1.2397},
    "Rh": {3: -0.3017, 4: -1.3455},
    "Ru": {2: -2.5921, 3: -2.1401, 4: -1.1536, 5: -0.8367, 6: -2.8332},
    "S": {-2: -0.1817, -1: -3.2115, 1: -6.0593, 2: -4.332, 3: -5.0707, 4: -3.3554, 5: -6.1238, 6: -2.7028},
    "Sb": {-3: -1.7601, -2: -2.9066, -1: -3.7939, 3: -0.8976, 4: -4.9289, 5: -1.0906},
    "Sc": {1: -3.5553, 2: -2.6391, 3: -0.1054},
    "Se": {-2: -0.2654, -1: -2.5322, 1: -4.9301, 2: -4.6644, 3: -5.9861, 4: -2.2019, 6: -3.7348},
    "Si": {-4: -3.5541, -1: -5.4189, 2: -5.0442, 3: -5.0442, 4: -0.047},
    "Sm": {2: -2.7473, 3: -0.0662},
    "Sn": {2: -1.0116, 3: -3.5139, 4: -0.4999},
    "Sr": {2: 0.0},
    "Ta": {1: -3.9627, 2: -4.1859, 3: -3.5573, 4: -2.3335, 5: -0.174},
    "Tb": {1: -3.7416, 2: -3.4539, 3: -0.1911, 4: -2.1322},
    "Te": {-2: -0.806, -1: -2.5048, 1: -5.3453, 2: -4.7857, 4: -1.0583, 6: -2.1937},
    "Th": {3: -2.5572, 4: -0.0807},
    "Ti": {2: -3.0494, 3: -1.9338, 4: -0.2132},
    "Tl": {-1: -4.8916, 1: -0.1351, 3: -2.1295},
    "Tm": {2: -2.8965, 3: -0.0568},
    "U": {2: -4.346, 3: -3.3045, 4: -1.3337, 5: -2.2459, 6: -0.543},
    "V": {2: -3.3163, 3: -1.8254, 4: -1.4266, 5: -0.5755},
    "W": {2: -3.2007, 3: -3.7985, 4: -3.3061, 5: -2.7625, 6: -0.1779},
    "Y": {3: 0.0},
    "Yb": {2: -1.965, 3: -0.151},
    "Zn": {2: 0.0},
    "Zr": {2: -3.1457, 3: -2.7208, 4: -0.1153},
}


def _possible_shannon_oxidation_states(label: str) -> list[int]:
    element_radii = SHANNON_IONIC_RADII.get(label)
    if element_radii is None:
        return []

    states = {
        record.charge
        for record in element_radii.radii
        if record.source == "shannon"
    }
    return sorted(states)


def _default_possible_oxidation_states(element: str) -> list[int]:
    scored_states = sorted(ELEMENT_OXI_SCORES.get(element, {}))
    if scored_states:
        return scored_states
    return _possible_shannon_oxidation_states(element)


def _expand_possible_oxidation_states_with_shannon(
    possible_oxi_states: dict[str, list[int]],
) -> dict[str, list[int]]:
    expanded: dict[str, list[int]] = {}
    for element, states in possible_oxi_states.items():
        expanded[element] = sorted(set(states) | set(_possible_shannon_oxidation_states(element)))
    return expanded


def _positive_compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield ordered tuples of `parts` strictly-positive integers summing to `total`."""
    if parts < 1 or total < parts:
        return
    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - parts + 2):
        for rest in _positive_compositions(total - first, parts - 1):
            yield (first, *rest)


def _enumerate_element_options(
    states: list[int],
    count: int,
    max_mixing: int,
) -> list[tuple[list[tuple[int, int]], int]]:
    """Every way to split `count` atoms across 1..min(max_mixing, count) distinct states.

    Returns a list of (pairs, total_charge) where pairs is [(oxi, n), ...].
    """
    options: list[tuple[list[tuple[int, int]], int]] = []
    max_states = min(max_mixing, count, len(states))
    for k in range(1, max_states + 1):
        for chosen_states in combinations(states, k):
            for counts in _positive_compositions(count, k):
                pairs = list(zip(chosen_states, counts))
                total_charge = sum(oxi * n for oxi, n in pairs)
                options.append((pairs, total_charge))
    return options


def _find_charge_balanced_assignments(
    composition: dict[str, int],
    possible_oxi_states: dict[str, list[int]],
    max_mixing_per_element: int,
    target_charge: int,
) -> list[OxidationStateDistribution]:
    per_element_options: dict[str, list[tuple[list[tuple[int, int]], int]]] = {}
    for element, count in composition.items():
        states = sorted(set(possible_oxi_states.get(element, [])))
        if not states:
            return []

        options = _enumerate_element_options(
            states=states,
            count=count,
            max_mixing=max_mixing_per_element,
        )
        if not options:
            return []
        per_element_options[element] = options

    elements = sorted(
        composition,
        key=lambda element: (
            len(per_element_options[element]),
            composition[element],
            element,
        ),
    )
    suffix_min_charge = [0] * (len(elements) + 1)
    suffix_max_charge = [0] * (len(elements) + 1)
    for index in range(len(elements) - 1, -1, -1):
        charges = [charge for _, charge in per_element_options[elements[index]]]
        suffix_min_charge[index] = suffix_min_charge[index + 1] + min(charges)
        suffix_max_charge[index] = suffix_max_charge[index + 1] + max(charges)

    assignments: list[OxidationStateDistribution] = []

    def walk(
        element_index: int,
        running_charge: int,
        current_assignment: OxidationStateDistribution,
    ) -> None:
        if element_index == len(elements):
            if running_charge == target_charge:
                assignments.append(
                    {
                        element: list(pairs)
                        for element, pairs in current_assignment.items()
                    }
                )
            return

        element = elements[element_index]
        for pairs, option_charge in per_element_options[element]:
            next_charge = running_charge + option_charge
            remaining_min = suffix_min_charge[element_index + 1]
            remaining_max = suffix_max_charge[element_index + 1]
            if next_charge + remaining_min > target_charge:
                continue
            if next_charge + remaining_max < target_charge:
                continue

            current_assignment[element] = list(pairs)
            walk(
                element_index + 1,
                next_charge,
                current_assignment,
            )
            del current_assignment[element]

    walk(
        element_index=0,
        running_charge=0,
        current_assignment={},
    )
    return assignments


def assign_oxidation_states(
    composition: dict[str, int],
    max_mixing_per_element: int = 2,
    target_charge: int = 0,
    possible_oxi_states: dict[str, list[int]] | None = None,
) -> list[tuple[OxidationStateDistribution, float]]:
    """Enumerate charge-balanced oxidation-state distributions for a composition.

    Parameters
    ----------
    composition : dict
        Element -> atom count, e.g. ``{'Fe': 3, 'O': 4}`` for Fe3O4.
    max_mixing_per_element : int
        Maximum number of distinct oxidation states allowed per element.
    target_charge : int
        Required net charge of the assignment (default 0).
    possible_oxi_states : dict, optional
        Per-element list of candidate oxidation states. Defaults to the keys of
        ``ELEMENT_OXI_SCORES[el]`` for each element in the composition.

    Returns
    -------
    list[tuple[OxidationStateDistribution, float]]
        ``[({el: [(oxi, count), ...]}, score), ...]`` sorted by score descending.
    """
    if max_mixing_per_element < 1:
        raise ValueError("max_mixing_per_element must be at least 1.")

    if possible_oxi_states is None:
        possible_oxi_states = {
            element: _default_possible_oxidation_states(element)
            for element in composition
        }

    valid = [
        (assignment, score_assignment(assignment, composition))
        for assignment in _find_charge_balanced_assignments(
            composition=composition,
            possible_oxi_states=possible_oxi_states,
            max_mixing_per_element=max_mixing_per_element,
            target_charge=target_charge,
        )
    ]

    valid.sort(key=lambda item: item[1], reverse=True)
    return valid


def _weighted_mean_oxi(pairs: list[tuple[int, int]]) -> float:
    total = sum(n for _, n in pairs)
    return sum(oxi * n for oxi, n in pairs) / total if total else 0.0


def score_assignment(
    assignment: OxidationStateDistribution,
    composition: dict[str, int],
) -> float:
    """Combined score: per-element ICSD prior + electronegativity ordering + stabilization."""
    element_score = 0.0
    for el, pairs in assignment.items():
        total_n = sum(n for _, n in pairs)
        if total_n == 0:
            continue
        per_atom = 0.0
        for oxi, n in pairs:
            per_atom += n * ELEMENT_OXI_SCORES.get(el, {}).get(oxi, -10.0)
        element_score += per_atom / total_n

    ordering_score = electronegativity_ordering_score(assignment)
    stabilization_score = electroneg_difference_score(assignment, composition)

    return 0.4 * element_score + 0.3 * ordering_score + 0.3 * stabilization_score


def electronegativity_ordering_score(assignment: OxidationStateDistribution) -> float:
    """Score in [0, 1] for how well element-pair oxidation ordering matches electronegativity.

    Mixed-valence elements collapse to a count-weighted mean oxidation state.
    """
    mean_oxi = {el: _weighted_mean_oxi(pairs) for el, pairs in assignment.items()}
    elements = list(mean_oxi)
    n_pairs = 0
    n_consistent = 0.0

    for i in range(len(elements)):
        for j in range(i + 1, len(elements)):
            el_i, el_j = elements[i], elements[j]
            oxi_i, oxi_j = mean_oxi[el_i], mean_oxi[el_j]
            chi_i, chi_j = ELECTRONEG.get(el_i, 2.0), ELECTRONEG.get(el_j, 2.0)
            n_pairs += 1

            if chi_i > chi_j:
                if oxi_i <= oxi_j:
                    n_consistent += 1
                else:
                    violation = (oxi_i - oxi_j) * (chi_i - chi_j)
                    n_consistent += max(0.0, 1 - 0.2 * violation)
            elif chi_j > chi_i:
                if oxi_j <= oxi_i:
                    n_consistent += 1
                else:
                    violation = (oxi_j - oxi_i) * (chi_j - chi_i)
                    n_consistent += max(0.0, 1 - 0.2 * violation)
            else:
                n_consistent += 1

    return n_consistent / n_pairs if n_pairs > 0 else 1.0


def electroneg_difference_score(
    assignment: OxidationStateDistribution,
    composition: dict[str, int],
) -> float:
    """Cation oxidation states need an anion electronegativity gap to be stabilized.

    Mixed-valence elements collapse to a count-weighted mean oxidation state.
    """
    mean_oxi = {el: _weighted_mean_oxi(pairs) for el, pairs in assignment.items()}
    cations = {el: oxi for el, oxi in mean_oxi.items() if oxi > 0}
    anions = {el: oxi for el, oxi in mean_oxi.items() if oxi < 0}

    if not cations or not anions:
        return 0.5

    anion_chi = float(np.average(
        [ELECTRONEG.get(el, 2.5) for el in anions],
        weights=[composition[el] for el in anions],
    ))

    score = 0.0
    for el, oxi in cations.items():
        chi_cation = ELECTRONEG.get(el, 2.0)
        delta_chi = anion_chi - chi_cation
        required_delta = 0.4 * oxi
        if required_delta <= 0 or delta_chi >= required_delta:
            score += 1.0
        else:
            score += max(0.0, delta_chi / required_delta)

    return score / len(cations)


def enumerate_possible_oxidation_state_assignments(
    labels: Sequence[str],
    charge: int = 0,
    maximum_mixing_per_element: int = 2,
) -> list[OxidationStateDistribution]:
    """
    Returns oxidation state distributions ordered best-first.
    """
    composition = dict(Counter(labels))
    possible_oxi_states = {
        element: _default_possible_oxidation_states(element)
        for element in composition
    }
    if any(not states for states in possible_oxi_states.values()):
        return []

    while True:
        scored = assign_oxidation_states(
            composition,
            max_mixing_per_element=maximum_mixing_per_element,
            target_charge=charge,
            possible_oxi_states=possible_oxi_states,
        )
        if scored:
            return [distribution for distribution, _ in scored]

        expanded_states = _expand_possible_oxidation_states_with_shannon(possible_oxi_states)
        if expanded_states == possible_oxi_states:
            return []
        possible_oxi_states = expanded_states


# ============================================================
# EXAMPLES
# ============================================================

def _format_assignment(assignment: OxidationStateDistribution) -> str:
    parts = []
    for el, pairs in assignment.items():
        chunks = ", ".join(f"{oxi:+d}×{n}" for oxi, n in pairs)
        parts.append(f"{el}: {chunks}")
    return "  |  ".join(parts)


def _print_results(title: str, composition: dict[str, int], **kwargs) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    results = assign_oxidation_states(composition, **kwargs)
    if not results:
        print("  (no valid assignments)")
        return
    for assignment, score in results[:5]:
        print(f"  {_format_assignment(assignment)}  →  score = {score:.3f}")


if __name__ == "__main__":
    _print_results("CuFeS2 (chalcopyrite)", {'Cu': 1, 'Fe': 1, 'S': 2})
    print()
    _print_results("Fe2O3", {'Fe': 2, 'O': 3})
    print()
    _print_results("Fe3O4 (mixed valence)", {'Fe': 3, 'O': 4})
    print()
    _print_results("Cr2O3", {'Cr': 2, 'O': 3})
    print()
    _print_results("CrO3", {'Cr': 1, 'O': 3})
    print()
    _print_results("MnO4- (permanganate, target_charge=-1)", {'Mn': 1, 'O': 4}, target_charge=-1)
