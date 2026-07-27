"""Energy-based oxidation-state prediction.

Instead of an ad-hoc weighted score, rank charge-balanced oxidation-state
assignments by a physical, geometry-free energy that mimics "filling orbitals by
their energies":

    E(assignment) =  sum over cation atoms of the ionization-energy ladder
                  -  sum over anion atoms of the electron-attachment gain
                  -  gamma * charge-transfer stabilization from the electronegativity gap

Lower energy = more stable. The goal is reasonable options quickly, not exact results.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from quick_mag.element_data import electron_affinity as _electron_affinity
from quick_mag.element_data import ionization_energies as _ionization_energies

from quick_mag.oxidation_state_enumeration import (
    ELECTRONEG,
    OxidationStateDistribution,
    _default_possible_oxidation_states,
    _expand_possible_oxidation_states_with_shannon,
    _find_charge_balanced_assignments,
)

# Default electronegativity-gap weight (eV per Pauling unit per transferred
# electron). Calibrated so the test perovskites land on their textbook charges.
DEFAULT_GAMMA: float = 3.0

# Standard closed-shell oxidation state for the common anions. A geometry-free
# energy cannot derive O^2- (gas-phase O^2- is unbound; only the Madelung field
# stabilizes it), so anions are anchored to their closed-shell charge and the
# IE + electronegativity energy ranks the cation charges. Can be disabled.
STANDARD_ANION_STATES: dict[str, int] = {
    "O": -2, "S": -2, "Se": -2, "Te": -2,
    "F": -1, "Cl": -1, "Br": -1, "I": -1,
}


@lru_cache(maxsize=None)
def ionization_cost(element: str, charge: int) -> float:
    """Energy (eV) to remove ``charge`` electrons: sum of the first IEs."""
    if charge <= 0:
        return 0.0
    energies = _ionization_energies(element)
    available = [e for e in energies[:charge] if e is not None]
    if len(available) < charge:
        # Not enough tabulated IEs: extrapolate the last known step (rare, only
        # for implausibly high charges that the candidate lists won't propose).
        if not available:
            return float("inf")
        step = available[-1] - available[-2] if len(available) > 1 else available[-1]
        missing = charge - len(available)
        return float(sum(available) + missing * available[-1] + step * missing)
    return float(sum(available))


@lru_cache(maxsize=None)
def attach_gain(element: str, n_electrons: int, deep_anion_ea: float = 0.0) -> float:
    """Energy (eV) released by adding ``n_electrons`` to form an anion.

    The first electron releases the electron affinity; deeper electrons are not
    bound in the gas phase, so they contribute ``deep_anion_ea`` (the lattice /
    electronegativity field supplies their binding elsewhere in the model).
    """
    if n_electrons <= 0:
        return 0.0
    ea = _electron_affinity(element)
    first = float(ea) if ea is not None else 0.0
    return first + deep_anion_ea * (n_electrons - 1)


def assignment_energy(
    distribution: OxidationStateDistribution,
    *,
    gamma: float = DEFAULT_GAMMA,
    deep_anion_ea: float = 0.0,
) -> float:
    """Physical energy (eV) of an oxidation-state assignment; lower = more stable.

    Charge transfer is stabilized symmetrically by an electronegativity potential:
    ``+gamma * sum_atoms n * oxi * chi``. Because the assignment is charge-balanced
    (``sum n*oxi = 0``) this equals ``gamma * sum n*oxi*(chi - chi_ref)``, i.e. it
    rewards anti-correlation between charge and electronegativity — electronegative
    elements lower the energy when negative (so O prefers -2 over -1) and high-chi
    elements are penalized for being cations.
    """
    energy = 0.0
    for element, pairs in distribution.items():
        chi = ELECTRONEG.get(element, 2.0)
        for oxi, n in pairs:
            if oxi > 0:
                energy += n * ionization_cost(element, oxi)
            elif oxi < 0:
                energy -= n * attach_gain(element, -oxi, deep_anion_ea)
            # Electronegativity charge-transfer stabilization.
            energy += gamma * n * oxi * chi
    return energy


def enumerate_oxidation_states_by_energy(
    labels: Sequence[str],
    charge: int = 0,
    max_mixing: int = 2,
    *,
    gamma: float = DEFAULT_GAMMA,
    deep_anion_ea: float = 0.0,
    anchor_anions: bool = True,
    top_k: Optional[int] = None,
) -> List[Tuple[OxidationStateDistribution, float]]:
    """Energy-ranked charge-balanced oxidation-state assignments (best first).

    Drop-in alternative to ``enumerate_possible_oxidation_state_assignments`` that
    ranks by the physical energy above instead of the heuristic score. With
    ``anchor_anions`` (default), common anions are pinned to their closed-shell
    charge (``STANDARD_ANION_STATES``) and the energy ranks the cation charges.
    """
    composition = dict(Counter(labels))
    possible_oxi_states = {
        element: (
            [STANDARD_ANION_STATES[element]]
            if anchor_anions and element in STANDARD_ANION_STATES
            else _default_possible_oxidation_states(element)
        )
        for element in composition
    }
    if any(not states for states in possible_oxi_states.values()):
        return []

    assignments: List[OxidationStateDistribution] = []
    while True:
        assignments = _find_charge_balanced_assignments(
            composition=composition,
            possible_oxi_states=possible_oxi_states,
            max_mixing_per_element=max_mixing,
            target_charge=charge,
        )
        if assignments:
            break
        expanded = _expand_possible_oxidation_states_with_shannon(possible_oxi_states)
        if expanded == possible_oxi_states:
            return []
        possible_oxi_states = expanded

    scored = [
        (assignment, assignment_energy(assignment, gamma=gamma, deep_anion_ea=deep_anion_ea))
        for assignment in assignments
    ]
    scored.sort(key=lambda item: item[1])
    return scored[:top_k] if top_k is not None else scored
