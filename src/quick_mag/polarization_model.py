"""Exchange-polarization superexchange model (docs/exchange_polarization_model.md).

Each metal i polarizes the p channels of a bridging ligand L with intensity

    mu_{i->L,p} = kappa_i * f_iL(r) * sum_a mbar_{i,a} * W_{a,p}

where ``W`` are squared Slater-Koster pd channel weights (``sk_table``) evaluated
in a bridge-adapted orthonormal p frame, ``mbar_{i,a}`` is the shell-averaged net
unpaired spin per d orbital, and ``f_iL(r) = exp(-alpha_iL (r - R0_iL))`` is a
metal-ligand damping with ``alpha_iL = sqrt(alpha_metal * alpha_ligand)`` and
``R0_iL`` the sum of Shannon crystal radii (not fit).

The ligand's net per-channel polarization for a collinear configuration is
``P_p = sigma_i mu_{i,p} + sigma_j mu_{j,p}`` (sigma = +-1). A Pauli cost
``E_A = w_L sum_p P_p^2`` penalizes shared channels (AFM) and a ligand Hund term
``E_B = -J_H^L sum_{p != p'} P_p P_p'`` rewards orthogonal-channel polarization
(FM). The configuration-dependent part per bridge is

    J_bridge = 2 (w_L + J_H^L) (mu_i . mu_j) - 2 J_H^L (sum_p mu_{i,p}) (sum_p mu_{j,p})

Sign convention throughout: J > 0 is AFM, and the model energy of a collinear
configuration is ``E = +1/2 sigma^T J sigma`` with sigma in {+1, 0, -1}. Spin
magnitude lives inside mu (via ``sum_a mbar_a``); do NOT multiply by nominal
moments.

Orbital resolution: ``mbar_a`` is the per-orbital net unpaired spin from the Hund
shell filling (t2g H/3 on each t2g orbital, eg H/2 on each eg orbital), evaluated
in each site's own octahedral frame (``local_octahedral_frame``). This equals the
average over degenerate microstates, since J is bilinear in the occupancies.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from quick_mag.ion_descriptors import IonDescriptor, ANION_SPECIES
from quick_mag.oxidation_state_energy import enumerate_oxidation_states_by_energy
from quick_mag.radii import SHANNON_IONIC_RADII
import quick_mag.sk_table as sk_table
T2G_INDICES = tuple(sk_table.D_ORBITALS.index(o) for o in ("dxy", "dxz", "dyz"))
EG_INDICES = tuple(sk_table.D_ORBITALS.index(o) for o in ("dz2", "dx2-y2"))

# Formal anion charge used for the ligand Shannon-radius lookup.
ANION_CHARGES: Dict[str, int] = {
    "O": -2, "S": -2, "Se": -2, "Te": -2, "N": -3,
    "F": -1, "Cl": -1, "Br": -1, "I": -1,
}

# IonDescriptor.spin_state -> Shannon table spin-state label.
_SHANNON_SPIN_LABEL = {"HS": "High Spin", "LS": "Low Spin"}


@dataclass
class PolarizationParameters:
    """Fit parameters of the polarization model.

    ``alpha`` holds per-element damping exponents for metals AND ligands;
    ``w_ligand`` is the ligand Pauli stiffness (gauge: w_O pinned to 1 by the
    fit); ``jh_ligand`` the ligand Hund coupling; ``gamma_pi`` a global pi/sigma
    polarization amplitude ratio (1.0 = no hierarchy).
    """

    kappa: Dict[str, float] = field(default_factory=dict)
    alpha: Dict[str, float] = field(default_factory=dict)
    w_ligand: Dict[str, float] = field(default_factory=dict)
    jh_ligand: Dict[str, float] = field(default_factory=dict)
    # Per-element double-exchange amplitude factor tau (Anderson-Hasegawa).
    # A DE-active bridge gets hopping amplitude tau_i * tau_j (product
    # combination rule), so same-element pairs see tau^2 and cross-element
    # resonant pairs get a combination for free. Default 0 = term off.
    t_de: Dict[str, float] = field(default_factory=dict)
    # Per-element eg orbital-order amplitude factor (Kugel-Khomskii FM term for
    # orbitally-degenerate eg^1 / eg^3 ions such as Mn3+; product rule tau_i*tau_j
    # like t_de). HAND-CALIBRATED, not fit — the d5-heavy training set cannot
    # constrain it. Default 0 = term off. See ``eg_order_fm_factor``.
    t_eg: Dict[str, float] = field(default_factory=dict)
    gamma_pi: float = 1.0
    kappa_default: float = 0.1
    alpha_default: float = 2.0
    w_default: float = 1.0
    jh_default: float = 0.1
    t_de_default: float = 0.0
    t_eg_default: float = 0.0

    def get_kappa(self, element: str) -> float:
        return self.kappa.get(element, self.kappa_default)

    def get_alpha(self, element: str) -> float:
        return self.alpha.get(element, self.alpha_default)

    def get_w(self, ligand: str) -> float:
        return self.w_ligand.get(ligand, self.w_default)

    def get_jh(self, ligand: str) -> float:
        return self.jh_ligand.get(ligand, self.jh_default)

    def get_t_de(self, element: str) -> float:
        return self.t_de.get(element, self.t_de_default)

    def get_t_eg(self, element: str) -> float:
        return self.t_eg.get(element, self.t_eg_default)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def bridge_frame(u_i: np.ndarray, u_j: np.ndarray, *, tol: float = 1e-6) -> np.ndarray:
    """Bridge-adapted orthonormal p frame (rows e1, e2, e3) for one M-L-M bridge.

    e1 = u_i (metal i's bond direction), e3 normal to the bridge plane, e2
    completes the right-handed set (u_j lies in span(e1, e2)). For (near-)linear
    bridges the plane normal is undefined; fall back to a deterministic global
    axis choice — irrelevant for orbitally isotropic (d5) ions.
    """
    e1 = np.asarray(u_i, dtype=float)
    cross = np.cross(e1, np.asarray(u_j, dtype=float))
    norm = np.linalg.norm(cross)
    if norm > tol:
        e3 = cross / norm
    else:
        axis = np.zeros(3)
        axis[int(np.argmin(np.abs(e1)))] = 1.0
        e3 = axis - (axis @ e1) * e1
        e3 /= np.linalg.norm(e3)
    e2 = np.cross(e3, e1)
    return np.array([e1, e2, e3])


# Charge-balanced oxidation distributions within this energy of the best one
# (eV/atom, from the geometry-free oxidation energy model) count as accessible
# when deciding whether mixed valence is forced. Calibrated so that resonant
# hole-transfer variants (~0.1-0.2 eV/atom, e.g. Fe-mixed vs Co-mixed in
# LaSr3FeCoO12) are included while unphysical A-site charge states (La2+, Sr+;
# >0.8 eV/atom) are excluded.
DE_ENERGY_WINDOW: float = 0.3


@lru_cache(maxsize=None)
def _de_active_pairs_cached(
    composition_key: Tuple[Tuple[str, int], ...],
    tm_key: Tuple[str, ...],
) -> frozenset:
    from itertools import combinations

    labels = [element for element, count in composition_key for _ in range(count)]
    try:
        ranked = enumerate_oxidation_states_by_energy(labels, top_k=None)
    except Exception:
        ranked = []
    if not ranked:
        return frozenset()
    cutoff = ranked[0][1] + DE_ENERGY_WINDOW * len(labels)
    single_valent = [
        {el for el in tm_key if len(dist.get(el, [])) == 1}
        for dist, energy in ranked
        if energy <= cutoff
    ]
    pairs = set()
    for element in tm_key:
        if not any(element in s for s in single_valent):
            pairs.add(frozenset((element,)))
    for el_a, el_b in combinations(tm_key, 2):
        possible_a = any(el_a in s for s in single_valent)
        possible_b = any(el_b in s for s in single_valent)
        both = any(el_a in s and el_b in s for s in single_valent)
        if possible_a and possible_b and not both:
            pairs.add(frozenset((el_a, el_b)))
    return frozenset(pairs)


def de_active_pairs(structure, tm_elements) -> frozenset:
    """Element pairs whose bridges carry double exchange, gated by charge balance.

    A pair (E, E') is DE-active when no charge-balanced oxidation distribution
    within ``DE_ENERGY_WINDOW`` of the best one makes both elements single-valent:

    - E == E': mixed valence is *forced* (e.g. Mn averaging +3.75 in
      LaSr3Mn4O12) — the classic same-element DE case.
    - E != E': each element *can* be single-valent, but never both at once —
      the itinerant carrier resonates between them (e.g. Fe-mixed <-> Co-mixed
      in LaSr3FeCoO12), the cross-element DE case.

    Materials with an accessible all-single-valent distribution (stoichiometric
    perovskites, hematite, goethite) get no flags, so spurious mixed-valence
    *assignments* from moment matching can no longer switch the term on.
    Composition-cached: repeated geometries of one material cost one enumeration.
    """
    from collections import Counter

    composition = Counter(structure.element_symbols())
    return _de_active_pairs_cached(
        tuple(sorted(composition.items())), tuple(sorted(tm_elements))
    )


def local_octahedral_frame(
    structure,
    site_index: int,
    *,
    ligand_cutoff: float = 3.0,
    min_pair_dot: float = -0.7,
) -> np.ndarray:
    """Local octahedral d-frame of a metal site from its ligand cage.

    Greedily pairs (near-)antiparallel M-L directions; each pair's difference
    vector is one octahedral axis. The stacked axes are symmetrized to the
    nearest orthonormal set (polar decomposition). Returns the rotation matrix
    whose *columns* are the local axes in global coordinates — the ``rotation``
    argument of ``sk_table.pd_amplitudes``.

    Axis labeling/sign is arbitrary, which is harmless: the shell-averaged
    occupancies weight all t2g (and all eg) orbitals equally, and those shell
    sums are invariant under relabeling the octahedral axes. Falls back to the
    identity (global frame) when fewer than two trans ligand pairs exist
    (non-octahedral or under-coordinated sites).
    """
    center = structure.cartesian_coords[site_index]
    directions: List[np.ndarray] = []
    for neighbor in structure.neighbors(site_index, ligand_cutoff):
        if neighbor.symbol not in ANION_SPECIES:
            continue
        vec = neighbor.coords - center
        norm = np.linalg.norm(vec)
        if norm > 1e-8:
            directions.append(vec / norm)

    axes: List[np.ndarray] = []
    used: set = set()
    while len(axes) < 3:
        best_pair = None
        best_dot = min_pair_dot
        for i in range(len(directions)):
            if i in used:
                continue
            for j in range(i + 1, len(directions)):
                if j in used:
                    continue
                dot = float(directions[i] @ directions[j])
                if dot < best_dot:
                    best_dot = dot
                    best_pair = (i, j)
        if best_pair is None:
            break
        i, j = best_pair
        used.update(best_pair)
        axis = directions[i] - directions[j]
        axes.append(axis / np.linalg.norm(axis))

    if len(axes) < 2:
        return np.eye(3)
    if len(axes) == 2:
        third = np.cross(axes[0], axes[1])
        axes.append(third / np.linalg.norm(third))
    u_svd, _, vt = np.linalg.svd(np.array(axes))
    return (u_svd @ vt).T


def is_eg_active(descriptor: IonDescriptor) -> bool:
    """True when the eg shell is singly occupied or singly holed (orbital freedom).

    ``n_eg in {1, 3}`` — one eg electron (e.g. Mn3+ d4 HS) or one eg hole (d9) —
    is the Jahn-Teller / orbital-ordering case the eg-order term targets. ``n_eg``
    of 0 (empty) or 2 (full) has no eg orbital degree of freedom.
    """
    return descriptor.n_eg in (1, 3)


def eg_orbital_director(
    structure, site_index: int, *, ligand_cutoff: float = 3.0
) -> Optional[np.ndarray]:
    """Unit vector along the Jahn-Teller-elongated octahedral axis of a site.

    For an eg^1 ion the occupied eg orbital (a d(3z^2-r^2)-type lobe) points along
    the *long* metal-ligand bonds, so the longest bond direction is the orbital
    director. The structural distortion in a relaxed geometry therefore encodes
    the orbital order directly — no orbital relaxation needed. Sign is irrelevant
    downstream (only ``(director . bond)^2`` is used). Returns None if the site
    has no ligand neighbours.
    """
    center = structure.cartesian_coords[site_index]
    bonds = [
        (float(nbr.nn_distance), nbr.coords - center)
        for nbr in structure.neighbors(site_index, ligand_cutoff)
        if nbr.symbol in ANION_SPECIES
    ]
    if not bonds:
        return None
    _, longest = max(bonds, key=lambda b: b[0])
    return longest / np.linalg.norm(longest)


def eg_order_fm_factor(
    director_i: np.ndarray, u_iL: np.ndarray,
    director_j: np.ndarray, u_jL: np.ndarray,
) -> float:
    """FM weight of the eg orbital-order term for one bridge, in [0, 1].

    ``g = (director . bond)^2`` is the sigma-activity of a site's occupied eg
    orbital on its metal-ligand bond. The Kugel-Khomskii FM channel opens when
    exactly one of the two orbitals points along the bridge (half-filled sigma on
    one side, empty on the other):

        factor = g_i + g_j - 2 g_i g_j = g_i(1 - g_j) + g_j(1 - g_i)

    which is ~1 for antiferro orbital order across the bond (one lobe on-axis, the
    other orthogonal -> in-plane FM in LaMnO3) and ~0 when both lobes point along
    the bond (ferro order, AFM handled by the base terms) or neither does (out-of-
    plane bond in LaMnO3 -> stays AFM).
    """
    g_i = float(director_i @ u_iL) ** 2
    g_j = float(director_j @ u_jL) ** 2
    return g_i + g_j - 2.0 * g_i * g_j


@lru_cache(maxsize=None)
def _shannon_crystal_radius(
    element: str, charge: int, spin_label: Optional[str]
) -> float:
    """Mean Shannon crystal radius with fallback: exact (VI, spin) -> (VI) ->
    any coordination at that charge -> any entry for the element."""
    entry = SHANNON_IONIC_RADII[element]
    for kwargs in (
        {"charge": charge, "coordination": "VI", "spin_state": spin_label},
        {"charge": charge, "coordination": "VI"},
        {"charge": charge},
        {},
    ):
        records = entry.find(**kwargs)
        if records:
            return sum(r.crystal_radius for r in records) / len(records)
    raise KeyError(f"No Shannon radius for {element} (charge {charge}).")


def r0_metal_ligand(
    metal: str, oxidation_state: int, spin_state: str, ligand: str
) -> float:
    """Reference M-L bond length: sum of Shannon crystal radii (Angstrom)."""
    metal_radius = _shannon_crystal_radius(
        metal, int(oxidation_state), _SHANNON_SPIN_LABEL.get(spin_state.upper())
    )
    ligand_radius = _shannon_crystal_radius(ligand, ANION_CHARGES.get(ligand, -2), None)
    return metal_radius + ligand_radius


def occupancy_vector(descriptor: IonDescriptor) -> np.ndarray:
    """Shell-averaged net unpaired spin per d orbital, ordered as sk_table.D_ORBITALS.

    H_t2g/3 on each t2g orbital and H_eg/2 on each eg orbital — the exact average
    over degenerate Hund microstates.
    """
    m = np.zeros(5)
    m[list(T2G_INDICES)] = descriptor.ehf_t2g[1] / 3.0
    m[list(EG_INDICES)] = descriptor.ehf_eg[1] / 2.0
    return m


@dataclass(frozen=True)
class BridgeGeometry:
    """One M-L-M bridge with precomputed per-end channel intensity vectors.

    ``Bsig_*``/``Bpi_*`` are ``sum_a mbar_a W_{a,p}`` (3,) in the bridge frame;
    everything downstream is elementwise in these plus the damping scalars.
    """

    site_i: int
    site_j: int
    ligand_index: int
    metal_i: str
    metal_j: str
    ligand: str
    r_iL: float
    r_jL: float
    R0_iL: float
    R0_jL: float
    Bsig_i: np.ndarray
    Bpi_i: np.ndarray
    Bsig_j: np.ndarray
    Bpi_j: np.ndarray
    u_iL: np.ndarray = None
    u_jL: np.ndarray = None
    frame: np.ndarray = None
    rot_i: np.ndarray = None   # local d-frame rotations (None = global frame)
    rot_j: np.ndarray = None
    # True when the end elements form a DE-active pair (charge-balance-forced
    # mixed valence, same-element or cross-element resonant; see de_active_pairs).
    de_active: bool = False
    # eg orbital-order geometry: both ends eg-active, and the FM factor
    # g_i + g_j - 2 g_i g_j in [0,1] (>0 for antiferro orbital order across the
    # bond, i.e. one occupied eg lobe points along the bond and the other does
    # not). See ``eg_order_fm_factor`` and ``bridge_J``.
    eg_active: bool = False
    eg_fm_factor: float = 0.0

    @property
    def cos_theta(self) -> float:
        return float(self.u_iL @ self.u_jL)


def _end_intensities(
    m_occ: np.ndarray,
    u: np.ndarray,
    frame: np.ndarray,
    rotation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(Bsig, Bpi) channel intensity 3-vectors for one bridge end."""
    w_sigma, w_pi = sk_table.channel_weights(u, frame, rotation=rotation)
    return m_occ @ w_sigma, m_occ @ w_pi


def build_bridges(
    structure,
    descriptors: Dict[int, IonDescriptor],
    *,
    de_pairs: Optional[frozenset] = None,
    anion_bond_cutoff: float = 3.0,
    super_cutoff: float = 7.0,
) -> List[BridgeGeometry]:
    """Enumerate M-L-M bridges between descriptor sites with actual bond vectors.

    For each bridging anion, pairs its transition-metal neighbours and records the
    image-resolved M-L unit vectors, precomputing the channel intensities. A metal
    pair sharing several ligands (edge/face sharing) yields one bridge per ligand.

    Each site's d orbitals are evaluated in its own octahedral frame
    (``local_octahedral_frame``), so tilted octahedra keep their eg/t2g character.

    ``de_pairs`` overrides the charge-balance double-exchange gate (a frozenset
    of frozensets of element symbols; pass ``frozenset()`` to disable DE). When
    None it is computed via ``de_active_pairs``.
    """
    occupancies = {
        site: occupancy_vector(descriptor)
        for site, descriptor in descriptors.items()
    }
    rotations: Dict[int, np.ndarray] = {
        site: local_octahedral_frame(structure, site, ligand_cutoff=anion_bond_cutoff)
        for site in descriptors
    }

    # eg orbital-order directors (JT-elongated axis) for eg-active sites only.
    eg_directors: Dict[int, np.ndarray] = {
        site: eg_orbital_director(structure, site, ligand_cutoff=anion_bond_cutoff)
        for site, descriptor in descriptors.items()
        if is_eg_active(descriptor)
    }

    if de_pairs is None:
        de_pairs = de_active_pairs(
            structure, {d.element for d in descriptors.values()}
        )
    symbols = structure.element_symbols()
    anion_indices = [
        i for i, symbol in enumerate(symbols) if symbol in ANION_SPECIES
    ]

    bridges: List[BridgeGeometry] = []
    for ligand_index in anion_indices:
        center_x = structure.cartesian_coords[ligand_index]
        neighbors = [
            nbr
            for nbr in structure.neighbors(ligand_index, anion_bond_cutoff)
            if nbr.index in descriptors
        ]
        for k1 in range(len(neighbors)):
            for k2 in range(k1 + 1, len(neighbors)):
                nbr_i, nbr_j = neighbors[k1], neighbors[k2]
                r_i, r_j = float(nbr_i.nn_distance), float(nbr_j.nn_distance)
                if r_i + r_j > super_cutoff or r_i < 1e-8 or r_j < 1e-8:
                    continue
                u_i = (nbr_i.coords - center_x) / r_i
                u_j = (nbr_j.coords - center_x) / r_j
                frame = bridge_frame(u_i, u_j)
                rot_i = rotations.get(nbr_i.index)
                rot_j = rotations.get(nbr_j.index)
                bsig_i, bpi_i = _end_intensities(
                    occupancies[nbr_i.index], u_i, frame, rotation=rot_i
                )
                bsig_j, bpi_j = _end_intensities(
                    occupancies[nbr_j.index], u_j, frame, rotation=rot_j
                )
                desc_i = descriptors[nbr_i.index]
                desc_j = descriptors[nbr_j.index]
                ligand = symbols[ligand_index]
                dir_i = eg_directors.get(nbr_i.index)
                dir_j = eg_directors.get(nbr_j.index)
                eg_active = dir_i is not None and dir_j is not None
                eg_fm_factor = (
                    eg_order_fm_factor(dir_i, u_i, dir_j, u_j) if eg_active else 0.0
                )
                bridges.append(
                    BridgeGeometry(
                        site_i=nbr_i.index,
                        site_j=nbr_j.index,
                        ligand_index=ligand_index,
                        metal_i=desc_i.element,
                        metal_j=desc_j.element,
                        ligand=ligand,
                        r_iL=r_i,
                        r_jL=r_j,
                        R0_iL=r0_metal_ligand(
                            desc_i.element, desc_i.oxidation_state, desc_i.spin_state, ligand
                        ),
                        R0_jL=r0_metal_ligand(
                            desc_j.element, desc_j.oxidation_state, desc_j.spin_state, ligand
                        ),
                        Bsig_i=bsig_i,
                        Bpi_i=bpi_i,
                        Bsig_j=bsig_j,
                        Bpi_j=bpi_j,
                        u_iL=u_i,
                        u_jL=u_j,
                        frame=frame,
                        rot_i=rot_i,
                        rot_j=rot_j,
                        de_active=(
                            frozenset((desc_i.element, desc_j.element)) in de_pairs
                        ),
                        eg_active=eg_active,
                        eg_fm_factor=eg_fm_factor,
                    )
                )
    return bridges


# ---------------------------------------------------------------------------
# mu and J assembly
# ---------------------------------------------------------------------------


def bridge_J(bridge: BridgeGeometry, params: PolarizationParameters) -> float:
    """Signed coupling of one bridge (J > 0 AFM).

    Superexchange (Terms A/B), plus on DE-active bridges an Anderson-Hasegawa
    double-exchange term ``-tau_i * tau_j * f_i * f_j * cos^2(theta)`` (FM;
    sigma-carrier transfer maximal at 180 degrees, zero at 90), plus on eg-active
    bridges a Kugel-Khomskii orbital-order term ``-t_eg_i * t_eg_j * fm_factor``
    (FM for antiferro eg orbital order across the bond; undamped, since the effect
    is set by orbital overlap geometry, not bond length — this is what lets the
    JT-elongated in-plane bonds of LaMnO3 be strongly FM). All product-rule
    combined: ``x_i * x_j`` for same-element ``x^2`` / cross-element geometric.
    """
    alpha_ligand = params.get_alpha(bridge.ligand)
    damp_i = math.exp(
        -math.sqrt(params.get_alpha(bridge.metal_i) * alpha_ligand)
        * (bridge.r_iL - bridge.R0_iL)
    )
    damp_j = math.exp(
        -math.sqrt(params.get_alpha(bridge.metal_j) * alpha_ligand)
        * (bridge.r_jL - bridge.R0_jL)
    )
    g2 = params.gamma_pi ** 2
    mu_i = params.get_kappa(bridge.metal_i) * damp_i * (bridge.Bsig_i + g2 * bridge.Bpi_i)
    mu_j = params.get_kappa(bridge.metal_j) * damp_j * (bridge.Bsig_j + g2 * bridge.Bpi_j)
    w = params.get_w(bridge.ligand)
    jh = params.get_jh(bridge.ligand)
    dot = float(mu_i @ mu_j)
    total = 2.0 * (w + jh) * dot - 2.0 * jh * float(mu_i.sum() * mu_j.sum())
    if bridge.de_active:
        t_pair = params.get_t_de(bridge.metal_i) * params.get_t_de(bridge.metal_j)
        if t_pair != 0.0:
            total -= t_pair * damp_i * damp_j * bridge.cos_theta ** 2
    if bridge.eg_active:
        t_eg = params.get_t_eg(bridge.metal_i) * params.get_t_eg(bridge.metal_j)
        if t_eg != 0.0:
            total -= t_eg * bridge.eg_fm_factor
    return total


def bridge_J_for_occupancies(
    bridge: BridgeGeometry,
    m_i: np.ndarray,
    m_j: np.ndarray,
    params: PolarizationParameters,
) -> float:
    """Coupling of ``bridge`` re-evaluated with explicit per-orbital occupancies.

    Used by tests (microstate-average equivalence) and toy studies; requires the
    bridge to carry its ``u_iL``/``u_jL``/``frame`` debug fields.
    """
    bsig_i, bpi_i = _end_intensities(
        np.asarray(m_i, dtype=float), bridge.u_iL, bridge.frame, rotation=bridge.rot_i
    )
    bsig_j, bpi_j = _end_intensities(
        np.asarray(m_j, dtype=float), bridge.u_jL, bridge.frame, rotation=bridge.rot_j
    )
    from dataclasses import replace

    return bridge_J(
        replace(bridge, Bsig_i=bsig_i, Bpi_i=bpi_i, Bsig_j=bsig_j, Bpi_j=bpi_j),
        params,
    )


def build_Jeff_matrix(
    bridges: List[BridgeGeometry],
    site_index: Dict[int, int],
    params: PolarizationParameters,
) -> np.ndarray:
    """Symmetric J matrix over compact magnetic-site indices (J > 0 AFM).

    Model energy of a collinear configuration: ``E = +1/2 sigma^T J sigma``.
    """
    n = len(site_index)
    matrix = np.zeros((n, n))
    for bridge in bridges:
        if bridge.site_i not in site_index or bridge.site_j not in site_index:
            continue
        value = bridge_J(bridge, params)
        a, b = site_index[bridge.site_i], site_index[bridge.site_j]
        matrix[a, b] += value
        matrix[b, a] += value
    return matrix


def effective_couplings(
    structure,
    descriptors: Dict[int, IonDescriptor],
    params: PolarizationParameters,
    *,
    de_pairs: Optional[frozenset] = None,
    anion_bond_cutoff: float = 3.0,
    super_cutoff: float = 7.0,
) -> Dict[Tuple[int, int], float]:
    """Per-pair effective couplings ``{(site_i, site_j): J}`` (J > 0 AFM)."""
    couplings: Dict[Tuple[int, int], float] = {}
    for bridge in build_bridges(
        structure,
        descriptors,
        de_pairs=de_pairs,
        anion_bond_cutoff=anion_bond_cutoff,
        super_cutoff=super_cutoff,
    ):
        key = tuple(sorted((bridge.site_i, bridge.site_j)))
        couplings[key] = couplings.get(key, 0.0) + bridge_J(bridge, params)
    return couplings


def to_solver_couplings(j_eff: np.ndarray) -> np.ndarray:
    """Convert to the spin-solver convention.

    ``spin_solver_np`` minimizes ``H = -1/2 sum_ij J_ij m_i m_j`` with J > 0 FM,
    so hand it ``-J_eff`` — and feed it UNIT moments (+-1). Spin magnitude is
    already inside mu; nominal 2S moments would double-count it.
    """
    return -np.asarray(j_eff)


# ---------------------------------------------------------------------------
# Parameter (de)serialization
# ---------------------------------------------------------------------------

#: The single bundled fitted + hand-calibrated parameter set (the only model).
DEFAULT_PARAMS_PATH = Path(__file__).with_name("exchange_params") / "polarization_params.json"


def params_from_dict(data: Dict) -> PolarizationParameters:
    """Build ``PolarizationParameters`` from a parsed JSON dict."""
    fields = {
        "kappa", "alpha", "w_ligand", "jh_ligand", "t_de", "t_eg", "gamma_pi",
        "kappa_default", "alpha_default", "w_default", "jh_default",
        "t_de_default", "t_eg_default",
    }
    return PolarizationParameters(**{k: v for k, v in data.items() if k in fields})


def load_params(path) -> PolarizationParameters:
    """Load the fitted parameter set from a JSON file path."""
    return params_from_dict(json.loads(Path(path).read_text()))


def default_params() -> PolarizationParameters:
    """The bundled fitted + hand-calibrated default model parameters."""
    return load_params(DEFAULT_PARAMS_PATH)
