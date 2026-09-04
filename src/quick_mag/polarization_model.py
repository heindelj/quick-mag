"""Exchange-polarization superexchange model (docs/theory/magnetism-model.md).

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
average over degenerate microstates, since J is bilinear in the occupancies. For
eg^1 sites the shell average is replaced by the *coherent* eg orbital selected by
a point-charge crystal field (``crystal_field_eg_orbital``), which also opens the
occupied->empty FM channel of ``bridge_J_components``.
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
from quick_mag.oxidation_state_energy import min_energies_with_single_valence
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

# eg splitting (e/A^3, the units of the k=2 field tensor) below which a site counts
# as undistorted, so its orbital order is undetermined and the occupied->empty FM
# channel stays off. An ideal octahedron gives exactly 0 (~1e-17 in floating point);
# a 0.02 A elongation already gives ~1e-3, so this only catches cages with no
# distortion at all -- an idealized builder seed, not a relaxed structure.
# See ``crystal_field_eg_orbital``.
_CF_DEGENERATE_TOL = 1e-12


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
    # Per-element occupied->empty FM amplitude factor (Kugel-Khomskii FM channel
    # for eg^1 Hund-active-core ions such as Mn3+; product rule j_fm_i*j_fm_j like
    # t_de). An electron hopping into a neighbour's empty eg orbital Hund-aligns
    # with that site's core spins -> FM (see docs/theory/magnetism-model.md).
    # HAND-CALIBRATED, not fit — the d5-heavy training set cannot constrain it.
    # Default 0 = channel off.
    j_fm: Dict[str, float] = field(default_factory=dict)
    gamma_pi: float = 1.0
    kappa_default: float = 0.1
    alpha_default: float = 2.0
    w_default: float = 1.0
    jh_default: float = 0.1
    t_de_default: float = 0.0
    j_fm_default: float = 0.0

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

    def get_j_fm(self, element: str) -> float:
        return self.j_fm.get(element, self.j_fm_default)


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
    # "Is element E (or pair E, E') ever single-valent within the window?" is a
    # constrained minimum-energy question, answered by one DP each — enumerating
    # every in-window assignment is combinatorially hopeless for large cells.
    groups = [frozenset((element,)) for element in tm_key] + [
        frozenset(pair) for pair in combinations(tm_key, 2)
    ]
    try:
        result = min_energies_with_single_valence(labels, groups)
    except Exception:
        result = None
    if result is None:
        return frozenset()
    global_min, constrained_min = result
    cutoff = global_min + DE_ENERGY_WINDOW * len(labels)

    pairs = set()
    for element in tm_key:
        if constrained_min[frozenset((element,))] > cutoff:
            pairs.add(frozenset((element,)))
    for el_a, el_b in combinations(tm_key, 2):
        possible_a = constrained_min[frozenset((el_a,))] <= cutoff
        possible_b = constrained_min[frozenset((el_b,))] <= cutoff
        both = constrained_min[frozenset((el_a, el_b))] <= cutoff
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
    is the Jahn-Teller / orbital-ordering case. ``n_eg`` of 0 (empty) or 2 (full)
    has no eg orbital degree of freedom.
    """
    return descriptor.n_eg in (1, 3)


def is_occ_empty_active(descriptor: IonDescriptor) -> bool:
    """True for an eg^1 ion with a Hund-active core (the occupied->empty FM case).

    One eg electron and one empty eg orbital, plus a spin-polarized core (t2g here)
    for the hopped electron to Hund-align with. An electron hopping into a
    neighbour's empty eg orbital then lowers energy for parallel spins -> FM.
    Excludes eg^3 (a lone hole; filling it leaves no other unpaired spin, so no
    Hund bonus — the FM channel is not electron-hole symmetric; see
    docs/theory/magnetism-model.md).
    """
    return descriptor.n_eg == 1


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

    Heuristic; superseded by ``crystal_field_eg_orbital`` (a k=2 point-charge
    crystal-field integral) which is continuous, balances all bonds, and vanishes
    for an undistorted octahedron. See docs/theory/magnetism-model.md.
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


def crystal_field_eg_orbital(
    structure,
    site_index: int,
    rotation: np.ndarray,
    *,
    hole: bool = False,
    ligand_cutoff: float = 3.0,
    charges: Optional[Dict[str, int]] = None,
) -> Optional[np.ndarray]:
    """Occupied (or hole) eg orbital of a site from a k=2 point-charge crystal field.

    Builds the ligand-field tensor ``F = sum_L (q_L / R_L^3)(3 u u^T - I)/2`` in the
    site's local octahedral frame (``rotation``, local->global) and returns the
    coefficient 2-vector on ``(dz2, dx2-y2)`` of the lower eg eigenvector
    (``hole=False``, eg^1 electron) or the upper one (``hole=True``, eg^3 hole).
    Returns None if the site has no ligand neighbours, or if its cage is
    undistorted (see below) — either way the caller falls back to the shell average.

    The eg block is closed form. The k=2 crystal-field matrix element is
    ``H_ab = int (n^T Q_a n)(n^T F n)(n^T Q_b n) dOmega``, a degree-6 polynomial in
    n whose sphere average is exact via the isotropic 6th-moment tensor; since
    ``Q_a``, ``Q_b`` and ``F`` are all symmetric traceless, every pairing with an
    intra-tensor contraction drops out and the eight surviving ones each give the
    same trace:

        <H_ab> = (8/105) tr(Q_a F Q_b)

    Both eg tensors are diagonal, so only ``diag(F)`` survives the trace and (using
    ``tr F = 0``) the eg block reduces, up to the positive constant, to

        [[p, q], [q, -p]],   p = F_zz,   q = (F_yy - F_xx)/sqrt(3)

    whose eigenvectors are the half-angle pair of ``phi = atan2(q, p)``. ``phi`` is
    the usual eg pseudospin angle: ``phi = pi`` is a z-elongated site (occupied
    orbital dz2), ``phi = 0`` a z-compressed one (dx2-y2).

    The k=2 field vanishes for an ideal octahedron, so this is exactly the
    distortion-induced orbital order and ``r = hypot(p, q)`` is the splitting. At
    ``r = 0`` the two eg orbitals are degenerate and there is no orbital order to
    report, so this returns None rather than an arbitrary member of the degenerate
    pair: ``build_bridges`` then leaves the site on its shell-averaged occupancy with
    the FM channel off, which is how every non-orbital ion class is already handled.
    See docs/theory/magnetism-model.md.
    """
    charges = ANION_CHARGES if charges is None else charges
    center = structure.cartesian_coords[site_index]
    F = np.zeros((3, 3))
    have = False
    for nbr in structure.neighbors(site_index, ligand_cutoff):
        if nbr.symbol not in ANION_SPECIES:
            continue
        d = nbr.coords - center
        r = float(np.linalg.norm(d))
        if r < 1e-8:
            continue
        u = d / r
        q = abs(charges.get(nbr.symbol, -2))
        F += (q / r ** 3) * (3.0 * np.outer(u, u) - np.eye(3)) / 2.0
        have = True
    if not have:
        return None
    rotation = np.asarray(rotation)
    f_xx, f_yy, f_zz = np.diag(rotation.T @ F @ rotation)
    p = float(f_zz)
    q = float(f_yy - f_xx) / math.sqrt(3.0)
    if math.hypot(p, q) < _CF_DEGENERATE_TOL:
        return None                             # degenerate: no orbital order
    half = 0.5 * math.atan2(q, p)
    if hole:
        return np.array([math.cos(half), math.sin(half)])
    return np.array([-math.sin(half), math.cos(half)])


def coherent_eg_intensity(
    coeff: np.ndarray,
    u: np.ndarray,
    frame: np.ndarray,
    rotation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(Bsig, Bpi) channel intensities of one *coherent* d orbital (coeff on the
    5 real d orbitals, local frame), for the occupied->empty channel.

    Unlike ``_end_intensities`` (an incoherent per-orbital sum over occupancies),
    this squares the *coherent* amplitude of a single superposition orbital, so the
    cross-terms that carry orbital directionality are retained:

        Bsig[p] = (A_sigma^psi * (e_p . u))^2 ,  Bpi[p] = ((sum_a c_a t_pi_a) . e_p)^2
    """
    a_sigma, t_pi = sk_table.pd_amplitudes(u, rotation=rotation)
    coeff = np.asarray(coeff, dtype=float)
    a_sig_psi = float(coeff @ a_sigma)
    t_pi_psi = coeff @ t_pi                       # (3,)
    c = np.asarray(frame) @ np.asarray(u)         # (3,) components of u on frame
    return (a_sig_psi * c) ** 2, (t_pi_psi @ np.asarray(frame).T) ** 2


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

    H_t2g/3 on each t2g orbital and H_eg/2 on each eg orbital. Only singly occupied
    orbitals carry net spin, so this is the exact average of "is orbital a singly
    occupied?" over the degenerate Hund microstates in
    ``descriptor.microstates`` — closed form, which is why those microstates are
    never enumerated here (Mn3+ -> [1, 1, 1, 0.5, 0.5]).

    The average is spherically symmetric within each shell, so the couplings it
    feeds carry no orbital-order information. For eg^1 Hund-active-core sites
    ``build_bridges`` replaces the averaged eg part with the *coherent* orbital
    picked out by the crystal field (``crystal_field_eg_orbital``); elsewhere
    replacing this with a non-uniform weighting over ``descriptor.microstates`` is
    the intended extension point if orbital order should come from the electronic
    structure rather than the geometry.
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
    # Occupied-orbital channel intensities (per bridge frame, (3,)). For an eg^1
    # Hund-active-core ion the eg part is the *coherent* CF-ordered orbital (see
    # build_bridges); otherwise the shell-averaged occupancy.
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
    # Occupied->empty FM channel: both ends eg^1 Hund-active-core, with the *empty*
    # (CF) eg orbital's channel intensities on each end. ``bridge_J`` adds
    # -j_fm_i j_fm_j (mu_occ_i . mu_emp_j + mu_emp_i . mu_occ_j). See
    # docs/theory/magnetism-model.md.
    oe_active: bool = False
    Bsig_emp_i: np.ndarray = None
    Bpi_emp_i: np.ndarray = None
    Bsig_emp_j: np.ndarray = None
    Bpi_emp_j: np.ndarray = None

    @property
    def cos_theta(self) -> float:
        return float(self.u_iL @ self.u_jL)


def _end_intensities(
    m_occ: np.ndarray,
    u: np.ndarray,
    frame: np.ndarray,
    rotation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """(Bsig, Bpi) channel intensity vectors for one bridge end."""
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

    ``de_pairs`` overrides the charge-balance double-exchange gate. When
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

    # Occupied->empty FM channel setup for eg^1 Hund-active-core sites: the coherent
    # CF-ordered occupied eg orbital replaces the averaged eg part of the occupancy,
    # and the orthogonal (empty) eg orbital is the FM acceptor. t2g stays averaged.
    eg_mask = np.zeros(5)
    eg_mask[list(EG_INDICES)] = 1.0
    psi_occ: Dict[int, np.ndarray] = {}
    psi_emp: Dict[int, np.ndarray] = {}
    t2g_occ: Dict[int, np.ndarray] = {}
    h_eg: Dict[int, float] = {}
    for site, descriptor in descriptors.items():
        if not is_occ_empty_active(descriptor):
            continue
        ev = crystal_field_eg_orbital(
            structure, site, rotations[site], hole=False, ligand_cutoff=anion_bond_cutoff
        )
        if ev is None:
            continue
        occ5 = np.zeros(5); occ5[EG_INDICES[0]] = ev[0]; occ5[EG_INDICES[1]] = ev[1]
        emp5 = np.zeros(5); emp5[EG_INDICES[0]] = -ev[1]; emp5[EG_INDICES[1]] = ev[0]
        psi_occ[site] = occ5
        psi_emp[site] = emp5
        t2g_occ[site] = occupancies[site] * (1.0 - eg_mask)   # eg zeroed
        h_eg[site] = float(descriptor.ehf_eg[1])

    def _end_channels(idx, u, frame, rot):
        """(Bsig_occ, Bpi_occ, Bsig_emp, Bpi_emp, oe_active) for one bridge end."""
        if idx in psi_occ:
            bst, bpt = _end_intensities(t2g_occ[idx], u, frame, rotation=rot)
            bso, bpo = coherent_eg_intensity(psi_occ[idx], u, frame, rotation=rot)
            bse, bpe = coherent_eg_intensity(psi_emp[idx], u, frame, rotation=rot)
            he = h_eg[idx]
            return bst + he * bso, bpt + he * bpo, he * bse, he * bpe, True
        bs, bp = _end_intensities(occupancies[idx], u, frame, rotation=rot)
        return bs, bp, None, None, False

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
                bsig_i, bpi_i, bse_i, bpe_i, oe_i = _end_channels(
                    nbr_i.index, u_i, frame, rot_i
                )
                bsig_j, bpi_j, bse_j, bpe_j, oe_j = _end_channels(
                    nbr_j.index, u_j, frame, rot_j
                )
                oe_active = oe_i and oe_j
                desc_i = descriptors[nbr_i.index]
                desc_j = descriptors[nbr_j.index]
                ligand = symbols[ligand_index]
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
                        oe_active=oe_active,
                        Bsig_emp_i=bse_i,
                        Bpi_emp_i=bpe_i,
                        Bsig_emp_j=bse_j,
                        Bpi_emp_j=bpe_j,
                    )
                )
    return bridges


# ---------------------------------------------------------------------------
# mu and J assembly
# ---------------------------------------------------------------------------


def bridge_J_components(
    bridge: BridgeGeometry, params: PolarizationParameters
) -> Tuple[float, float, float]:
    """One bridge's coupling split into its three channels (J > 0 AFM).

    Returns ``(J_SE, J_DE, J_OE)``:

    - ``J_SE``, superexchange (Terms A/B), carried by every bridge.
    - ``J_DE``, on DE-active bridges an Anderson-Hasegawa double-exchange term
      ``-tau_i * tau_j * f_i * f_j * cos^2(theta)`` (FM; sigma-carrier transfer
      maximal at 180 degrees, zero at 90).
    - ``J_OE``, on occupied->empty-active bridges (eg^1 Hund-active-core on both
      ends) a Kugel-Khomskii FM channel ``-j_fm_i j_fm_j (mu_occ_i . mu_emp_j +
      mu_emp_i . mu_occ_j)`` (an electron hopping into a neighbour's empty eg
      orbital Hund-aligns with that site's core -> FM; quadratic in the
      intensities, like Term A).

    The two FM channels are product-rule combined, ``x_i * x_j`` for the
    per-element factors, so an unparameterized element drops the channel to zero.
    Split out rather than summed on the spot because the UI plots the three
    against their total: which mechanism makes a pair FM is the question the
    coupling plot is there to answer.
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
    kap_i = params.get_kappa(bridge.metal_i) * damp_i
    kap_j = params.get_kappa(bridge.metal_j) * damp_j
    mu_i = kap_i * (bridge.Bsig_i + g2 * bridge.Bpi_i)
    mu_j = kap_j * (bridge.Bsig_j + g2 * bridge.Bpi_j)
    w = params.get_w(bridge.ligand)
    jh = params.get_jh(bridge.ligand)
    dot = float(mu_i @ mu_j)
    j_se = 2.0 * (w + jh) * dot - 2.0 * jh * float(mu_i.sum() * mu_j.sum())
    j_de = 0.0
    if bridge.de_active:
        t_pair = params.get_t_de(bridge.metal_i) * params.get_t_de(bridge.metal_j)
        if t_pair != 0.0:
            j_de = -(t_pair * damp_i * damp_j * bridge.cos_theta ** 2)
    j_oe = 0.0
    if bridge.oe_active:
        j_fm = params.get_j_fm(bridge.metal_i) * params.get_j_fm(bridge.metal_j)
        if j_fm != 0.0:
            mu_emp_i = kap_i * (bridge.Bsig_emp_i + g2 * bridge.Bpi_emp_i)
            mu_emp_j = kap_j * (bridge.Bsig_emp_j + g2 * bridge.Bpi_emp_j)
            j_oe = -(j_fm * float(mu_i @ mu_emp_j + mu_emp_i @ mu_j))
    return j_se, j_de, j_oe


def bridge_J(bridge: BridgeGeometry, params: PolarizationParameters) -> float:
    """Signed coupling of one bridge (J > 0 AFM): the sum of its three channels.

    See ``bridge_J_components`` for what those channels are.
    """
    j_se, j_de, j_oe = bridge_J_components(bridge, params)
    return j_se + j_de + j_oe


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


@dataclass(frozen=True)
class PairCoupling:
    """The total coupling between one pair of metal sites, with its provenance.

    ``build_Jeff_matrix`` sums the bridges of a pair into a single matrix element
    and throws away everything that says *why*. This keeps that: which elements
    are coupled, how far apart they are, and through how many ligands at what
    angles -- enough to label and explain a bar in the UI's coupling plot.

    ``j_eff`` is in the model convention (J > 0 AFM), the same as the matrix.
    """

    site_i: int  # structure atom index; site_i < site_j
    site_j: int
    metal_i: str
    metal_j: str
    j_eff: float
    distance: float  # metal-metal distance, Angstrom
    bridge_count: int
    ligands: Tuple[str, ...]
    angles_deg: Tuple[float, ...]  # M-L-M angle, one per bridge
    ligand_indices: Tuple[int, ...] = ()
    #: ``bridge_J_components`` summed over the pair's bridges, so the three add up
    #: to ``j_eff`` (up to float rounding: ``j_eff`` is accumulated bridge by
    #: bridge, exactly as the matrix element is). Defaulted so a caller that only
    #: cares about the total can still build one.
    j_se: float = 0.0  # superexchange; the only channel that can be AFM
    j_de: float = 0.0  # double exchange (<= 0, FM)
    j_oe: float = 0.0  # occupied->empty Kugel-Khomskii (<= 0, FM)
    #: Cartesian M-L-M paths, ``(bridge_count, 3, 3)``, ordered
    #: ``[site_i, ligand, site_j]``. Empty when built without coordinates. These are
    #: the *image* positions the bridge actually used, so a path that crosses a cell
    #: boundary is contiguous rather than leaping back across the cell.
    paths: np.ndarray = None


def pair_couplings(
    bridges: List[BridgeGeometry],
    params: PolarizationParameters,
    cartesian_coords: Optional[np.ndarray] = None,
) -> List[PairCoupling]:
    """Per-pair couplings, summed over bridges exactly as ``build_Jeff_matrix`` does.

    Same accumulation, so ``j_eff`` here and the corresponding matrix element are
    the same number by construction rather than by coincidence. The three channels
    of ``bridge_J_components`` are accumulated alongside it, so a pair carries not
    just how strongly it couples but which mechanism did it.

    The geometry is read off the bridge instead of being recomputed from the
    structure: ``u_iL`` and ``u_jL`` both point outwards from the ligand, so
    ``cos_theta`` is the cosine of the M-L-M angle, the law of cosines closes the
    triangle for the metal-metal distance, and stepping ``r_iL`` and ``r_jL`` back
    along those directions recovers the two metals' positions -- as *images*, which
    is what makes a path across a cell boundary contiguous. Pairs bridged more than
    once give the same distance from every bridge, up to rounding; the first is kept.

    ``cartesian_coords`` is the structure's coordinates, needed only to place the
    ligand that anchors each path. Without it the couplings still carry everything
    but ``paths``.
    """
    totals: Dict[Tuple[int, int], float] = {}
    components: Dict[Tuple[int, int], List[float]] = {}
    metals: Dict[Tuple[int, int], Tuple[str, str]] = {}
    distances: Dict[Tuple[int, int], float] = {}
    ligands: Dict[Tuple[int, int], List[str]] = {}
    ligand_indices: Dict[Tuple[int, int], List[int]] = {}
    angles: Dict[Tuple[int, int], List[float]] = {}
    paths: Dict[Tuple[int, int], List[np.ndarray]] = {}

    for bridge in bridges:
        forward = bridge.site_i < bridge.site_j
        key = (
            (bridge.site_i, bridge.site_j) if forward else (bridge.site_j, bridge.site_i)
        )
        j_se, j_de, j_oe = bridge_J_components(bridge, params)
        # The total is accumulated from the bridge's own sum, the same quantity
        # and in the same order as ``build_Jeff_matrix`` accumulates it, so
        # ``j_eff`` stays the matrix element bit for bit.
        totals[key] = totals.get(key, 0.0) + (j_se + j_de + j_oe)
        running = components.setdefault(key, [0.0, 0.0, 0.0])
        running[0] += j_se
        running[1] += j_de
        running[2] += j_oe
        cos_theta = min(max(bridge.cos_theta, -1.0), 1.0)
        if key not in metals:
            metals[key] = (
                (bridge.metal_i, bridge.metal_j)
                if forward
                else (bridge.metal_j, bridge.metal_i)
            )
            distances[key] = math.sqrt(
                max(
                    bridge.r_iL**2
                    + bridge.r_jL**2
                    - 2.0 * bridge.r_iL * bridge.r_jL * cos_theta,
                    0.0,
                )
            )
        ligands.setdefault(key, []).append(bridge.ligand)
        ligand_indices.setdefault(key, []).append(bridge.ligand_index)
        angles.setdefault(key, []).append(math.degrees(math.acos(cos_theta)))
        if cartesian_coords is not None:
            centre = np.asarray(cartesian_coords[bridge.ligand_index], dtype=np.float64)
            ends = (
                (centre + bridge.u_iL * bridge.r_iL, centre + bridge.u_jL * bridge.r_jL)
                if forward
                else (centre + bridge.u_jL * bridge.r_jL, centre + bridge.u_iL * bridge.r_iL)
            )
            paths.setdefault(key, []).append(
                np.array([ends[0], centre, ends[1]], dtype=np.float64)
            )

    return [
        PairCoupling(
            site_i=key[0],
            site_j=key[1],
            metal_i=metals[key][0],
            metal_j=metals[key][1],
            j_eff=totals[key],
            distance=distances[key],
            bridge_count=len(ligands[key]),
            ligands=tuple(ligands[key]),
            angles_deg=tuple(angles[key]),
            ligand_indices=tuple(ligand_indices[key]),
            j_se=components[key][0],
            j_de=components[key][1],
            j_oe=components[key][2],
            paths=(
                np.array(paths[key], dtype=np.float64)
                if key in paths
                else np.zeros((0, 3, 3), dtype=np.float64)
            ),
        )
        for key in totals
    ]


def to_solver_couplings(j_eff: np.ndarray) -> np.ndarray:
    """Convert to the spin-solver convention.

    ``spin_solver`` minimizes ``H = -1/2 sum_ij J_ij m_i m_j`` with J > 0 FM,
    so hand it ``-J_eff`` and feed it UNIT moments (+-1). Spin magnitude is
    already inside mu.
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
        "kappa", "alpha", "w_ligand", "jh_ligand", "t_de", "j_fm", "gamma_pi",
        "kappa_default", "alpha_default", "w_default", "jh_default",
        "t_de_default", "j_fm_default",
    }
    return PolarizationParameters(**{k: v for k, v in data.items() if k in fields})


def load_params(path) -> PolarizationParameters:
    """Load the fitted parameter set from a JSON file path."""
    return params_from_dict(json.loads(Path(path).read_text()))


def default_params() -> PolarizationParameters:
    """The bundled fitted + hand-calibrated default model parameters."""
    return load_params(DEFAULT_PARAMS_PATH)
