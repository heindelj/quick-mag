"""Octahedral d-shell ion descriptors for the exchange model.

From an element and oxidation state, derive the octahedral d-shell picture the
exchange model needs:

    oxidation state -> d-count -> (n_t2g, n_eg) shell filling
                    -> per-shell (E, H, F) decomposition
                    -> microstates over the named d-orbitals
                    -> ion spin S = (H_t2g + H_eg) / 2

A microstate is an occupancy vector over the five named d-orbitals (electrons per
orbital, in {0, 1, 2}); the (E, H, F) label of an orbital is ``{0: "E", 1: "H",
2: "F"}``.

Microstates and the exchange model
----------------------------------
Hund filling fixes how many orbitals per shell are empty/half/full, but not which
named orbital plays which role — every distinct assignment is a degenerate
microstate (Mn3+ t2g^3 eg^1 has two: the lone eg electron in dz2 or in dx2-y2).

**The exchange model does not currently consume these.** It takes the closed-form
average over them instead — ``polarization_model.occupancy_vector``, which puts
H_t2g/3 on each t2g orbital and H_eg/2 on each eg orbital. For degenerate
microstates that is exactly their mean, so nothing is lost relative to a uniform
weighting; what is lost is orbital-order information, which the model recovers
geometrically through ``polarization_model.eg_orbital_director`` instead.

``enumerate_shell_microstates`` and ``IonDescriptor.microstates`` are kept as the
hook for per-microstate weighting (Boltzmann weights, or resolving each microstate
into its own coupling and combining afterwards). Until something consumes them,
they are built and discarded on every solve — cheap (<= 6 per ion), but do not
mistake their presence for orbital resolution in the couplings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product as cartesian_product
from typing import Dict, List, Tuple

from quick_mag.electron_configurations import get_ionic_electron_configuration

# Named d-orbitals per shell. Order is fixed and used as the microstate key set.
T2G_ORBITALS: Tuple[str, ...] = ("dxy", "dxz", "dyz")
EG_ORBITALS: Tuple[str, ...] = ("dz2", "dx2-y2")
D_ORBITALS: Tuple[str, ...] = T2G_ORBITALS + EG_ORBITALS

# Transition-metal d-block elements that carry a partially-fillable d shell.
TRANSITION_METALS: frozenset[str] = frozenset(
    {
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    }
)

# Bridging anions that mediate superexchange.
ANION_SPECIES: frozenset[str] = frozenset(
    {"O", "S", "N", "F", "Cl", "Br", "I", "Se", "Te"}
)

Microstate = Dict[str, int]


def d_electron_count(element: str, oxidation_state: int) -> int:
    """Number of d electrons for the ion (highest-n d subshell of its ionic config)."""
    configuration = get_ionic_electron_configuration(element, oxidation_state)
    d_subshells = [(n, count) for n, orbital, count in configuration if orbital == "d"]
    if not d_subshells:
        return 0
    # Valence d shell = the highest principal quantum number carrying d electrons.
    _, count = max(d_subshells, key=lambda item: item[0])
    if not 0 <= count <= 10:
        raise ValueError(f"Unphysical d-count {count} for {element}{oxidation_state:+d}.")
    return count


def shell_filling(d_count: int, spin: str = "HS") -> Tuple[int, int]:
    """Return ``(n_t2g, n_eg)`` for a d-count. High-spin by default."""
    if not 0 <= d_count <= 10:
        raise ValueError(f"d_count must be in [0, 10], got {d_count}.")
    spin = spin.upper()
    if spin == "HS":
        # Hund filling: t2g^3 eg^2 singly, then pair t2g before eg.
        if d_count <= 3:
            return d_count, 0
        if d_count <= 5:
            return 3, d_count - 3
        if d_count <= 8:
            return d_count - 2, 2
        return 6, d_count - 6
    if spin == "LS":
        # Strong field: fill t2g completely before touching eg.
        n_t2g = min(d_count, 6)
        return n_t2g, d_count - n_t2g
    raise ValueError(f"spin must be 'HS' or 'LS', got {spin!r}.")


def shell_ehf(n_shell: int, degeneracy: int) -> Tuple[int, int, int]:
    """(E, H, F) for ``n_shell`` electrons in a shell of orbital degeneracy ``g``."""
    empty = max(degeneracy - n_shell, 0)
    full = max(n_shell - degeneracy, 0)
    half = degeneracy - empty - full
    return empty, half, full


def _canonical_shell_occupancy(n_shell: int, degeneracy: int) -> List[int]:
    """Hund canonical per-orbital occupancy: doubly filled first, then singly, then empty."""
    empty, half, full = shell_ehf(n_shell, degeneracy)
    return [2] * full + [1] * half + [0] * empty


def enumerate_shell_microstates(
    n_shell: int, orbitals: Tuple[str, ...]
) -> List[Microstate]:
    """All distinct label assignments of a shell to its named orbitals.

    The count equals the multinomial ``g! / (E! H! F!)``. These microstates are
    degenerate under Hund filling; see the module docstring for why the exchange
    model averages over them rather than weighting them individually.
    """
    occupancy = _canonical_shell_occupancy(n_shell, len(orbitals))
    seen: set[Tuple[int, ...]] = set()
    microstates: List[Microstate] = []
    for permutation in permutations(occupancy):
        if permutation in seen:
            continue
        seen.add(permutation)
        microstates.append(dict(zip(orbitals, permutation)))
    return microstates


@dataclass
class IonDescriptor:
    """Octahedral d-shell descriptor for a single transition-metal ion."""

    element: str
    oxidation_state: int
    d_count: int
    spin_state: str
    n_t2g: int
    n_eg: int
    ehf_t2g: Tuple[int, int, int]
    ehf_eg: Tuple[int, int, int]
    spin: float
    # Degenerate Hund microstates. Nothing consumes these yet — the exchange model
    # uses their average via ``polarization_model.occupancy_vector``; see the
    # module docstring.
    microstates: List[Microstate] = field(default_factory=list)

    @property
    def n_microstates(self) -> int:
        return len(self.microstates)

    def __repr__(self) -> str:
        return (
            f"IonDescriptor({self.element}{self.oxidation_state:+d}, d{self.d_count} "
            f"{self.spin_state}, t2g^{self.n_t2g} eg^{self.n_eg}, S={self.spin:g}, "
            f"{self.n_microstates} microstate(s))"
        )


def ion_descriptor(
    element: str, oxidation_state: int, spin: str = "HS"
) -> IonDescriptor:
    """Build the full ion descriptor for ``element`` in the given oxidation state."""
    spin = spin.upper()
    d_count = d_electron_count(element, oxidation_state)
    n_t2g, n_eg = shell_filling(d_count, spin)
    ehf_t2g = shell_ehf(n_t2g, len(T2G_ORBITALS))
    ehf_eg = shell_ehf(n_eg, len(EG_ORBITALS))
    total_spin = (ehf_t2g[1] + ehf_eg[1]) / 2.0

    t2g_states = enumerate_shell_microstates(n_t2g, T2G_ORBITALS)
    eg_states = enumerate_shell_microstates(n_eg, EG_ORBITALS)
    microstates = [
        {**t2g_state, **eg_state}
        for t2g_state, eg_state in cartesian_product(t2g_states, eg_states)
    ]

    return IonDescriptor(
        element=element,
        oxidation_state=int(oxidation_state),
        d_count=d_count,
        spin_state=spin,
        n_t2g=n_t2g,
        n_eg=n_eg,
        ehf_t2g=ehf_t2g,
        ehf_eg=ehf_eg,
        spin=total_spin,
        microstates=microstates,
    )


def structure_ion_descriptors(
    structure, assignment, spin: str = "HS"
) -> Dict[int, IonDescriptor]:
    """Map each transition-metal site of an ``OxidationStateAssignment`` to its descriptor.

    ``assignment`` only needs a ``site_oxidation_states`` array aligned with
    ``structure``. Non-transition-metal sites (A-site cations, anions) are skipped.
    """
    site_ox = assignment.site_oxidation_states
    descriptors: Dict[int, IonDescriptor] = {}
    for index, symbol in enumerate(structure.element_symbols()):
        if symbol not in TRANSITION_METALS:
            continue
        descriptors[index] = ion_descriptor(symbol, int(site_ox[index]), spin)
    return descriptors
