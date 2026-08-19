"""Point defects layered onto an ideal perovskite build.

The builder is a pure function of :class:`PerovskiteGenerationParameters`: every
edit regenerates the structure from scratch through ``build_perovskite``, which
always emits a complete, perfectly ordered ``(A, B, X)`` lattice. Defects are
therefore kept *out* of the generator and applied as a post-pass here.

That buys the property the builder needs: the idealized structure stays the
source of truth. Changing a tilt angle, a lattice constant, or the supercell size
rebuilds the ideal geometry and re-applies the same defect list on top, so
defects never freeze the geometry and never accumulate.

It works because a defect addresses a site by a
:class:`~quick_mag.perovskite_builder.SiteKey` -- a grid address -- rather than an
array index. Two wrinkles follow from that, both handled by
:func:`resolve_key_to_indices`:

* **Aliasing.** Corner-shared oxygens have two names: the ``-a`` vertex of cell
  ``i`` is the same atom as the ``+a`` vertex of cell ``i-1``.
  :func:`canonicalize_key` folds every alias onto the canonical representative,
  so the user can name an oxygen by whichever octahedron they are looking at.
* **Periodic images.** The 3D view renders a periodic structure by rebuilding it
  non-periodically, which adds the closing boundary layer -- a corner A site gains
  up to 8 copies. A vacancy has to remove all of them, or the hole visibly fills
  back in at the cell edge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from quick_mag.perovskite_builder import (
    VERTEX_NAMES,
    SiteKey,
    a_site_grid_counts,
    canonical_index_of_key,
    canonical_site_counts,
    canonical_site_keys,
)

DEFECT_KINDS: Tuple[str, ...] = ("vacancy", "substitution", "proton")
LATTICE_ROLES: Tuple[str, ...] = ("A", "B", "X")
PROTON_ROLE = "H"
PROTON_SYMBOL = "H"

# Standard O-H distance for a proton trapped on a perovskite oxygen.
DEFAULT_OH_BOND_LENGTH = 0.98
# An oxygen has four symmetry-equivalent proton sites: the two octahedral axes
# perpendicular to its own, in both directions.
PROTON_ORIENTATION_COUNT = 4
# Below this, a placed proton is reported as clashing with its surroundings.
MIN_CONTACT_DISTANCE = 1.30


@dataclass
class SiteDefect:
    """One point defect addressed by a stable grid key.

    ``kind`` is ``"vacancy"`` (remove the site), ``"substitution"`` (rewrite its
    element), or ``"proton"`` (add an interstitial H on the named oxygen, for
    charge compensation). ``element`` is the replacement symbol for a
    substitution. ``orientation`` selects among the four equivalent proton sites
    around a host oxygen.
    """

    kind: str
    site: SiteKey
    element: str = ""
    orientation: int = 0
    bond_length: float = DEFAULT_OH_BOND_LENGTH

    def __post_init__(self) -> None:
        self.kind = str(self.kind).strip().lower()
        if self.kind not in DEFECT_KINDS:
            raise ValueError(
                f"Unknown defect kind '{self.kind}'; expected one of {DEFECT_KINDS}."
            )
        self.site = coerce_site_key(self.site)
        self.element = str(self.element).strip()
        self.orientation = int(self.orientation)
        self.bond_length = float(self.bond_length)
        if self.kind == "substitution" and not self.element:
            raise ValueError("A substitution defect needs a replacement element.")
        if self.kind == "proton":
            if self.site.role != "X":
                raise ValueError(
                    "A proton attaches to an X (anion) site; "
                    f"got role '{self.site.role}'."
                )
            self.element = self.element or PROTON_SYMBOL

    def signature(self) -> Tuple:
        """Hashable form, for the builder's change-detection signature."""
        return (
            self.kind,
            tuple(self.site),
            self.element,
            self.orientation,
            round(self.bond_length, 6),
        )


@dataclass
class DefectResolution:
    """Where each canonical site ended up once the defect list was applied."""

    # canonical build index -> index in the emitted structure, or -1 if vacated.
    canonical_to_structure: np.ndarray
    # Ascending canonical indices that survived.
    kept_canonical: np.ndarray
    # canonical index -> replacement element.
    substitutions: Dict[int, str]
    proton_coords: np.ndarray
    proton_host_canonical: np.ndarray
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def coerce_site_key(key) -> SiteKey:
    """Accept a ``SiteKey``, a tuple, or a mapping and return a ``SiteKey``."""
    if isinstance(key, SiteKey):
        role, i, j, k, vertex = key
    elif isinstance(key, dict):
        role = key.get("role")
        i, j, k = key.get("i", 0), key.get("j", 0), key.get("k", 0)
        vertex = key.get("vertex", 0)
    else:
        parts = tuple(key)
        if len(parts) == 4:
            role, i, j, k = parts
            vertex = 0
        elif len(parts) == 5:
            role, i, j, k, vertex = parts
        else:
            raise ValueError(
                "A site key is (role, i, j, k) or (role, i, j, k, vertex); "
                f"got {len(parts)} fields."
            )
    role = str(role).strip().upper()
    if role not in LATTICE_ROLES:
        raise ValueError(f"Unknown site role '{role}'; expected one of {LATTICE_ROLES}.")
    vertex = int(vertex)
    if role != "X":
        vertex = 0
    elif not 0 <= vertex < 6:
        raise ValueError(f"X-site vertex row {vertex} is outside 0..5.")
    return SiteKey(role, int(i), int(j), int(k), vertex)


def site_key_display(key) -> str:
    """Human-readable form, e.g. ``"X(1,0,1) +b"``."""
    key = coerce_site_key(key)
    label = f"{key.role}({key.i},{key.j},{key.k})"
    return f"{label} {VERTEX_NAMES[key.vertex]}" if key.role == "X" else label


def canonicalize_key(key, grid_shape, periodic: bool) -> Optional[SiteKey]:
    """Fold ``key`` onto the canonical site it names, or None if out of range.

    Wrapping is deliberately narrow. A periodic lattice has no boundary, so it is
    tempting to reduce every index modulo the grid -- but then shrinking the
    supercell would silently move a defect onto a *different* site instead of
    dropping it, and the user's coordinates would be quietly rewritten. Only the
    two folds that name the *same* atom are allowed:

    * the corner-shared oxygen alias, where the ``-a`` vertex of cell ``i`` is the
      ``+a`` vertex of cell ``i-1`` (wrapping to the last cell when periodic);
    * the closing A-site plane a finite build adds at ``i == nx``, which is the
      image of the ``i == 0`` plane, so a periodicity toggle keeps its meaning.

    Anything else outside the grid returns None and is skipped -- which is how a
    defect survives shrinking and then re-growing the supercell intact.
    """
    key = coerce_site_key(key)
    nx, ny, nz = (int(value) for value in grid_shape)
    bounds = a_site_grid_counts(grid_shape, periodic) if key.role == "A" else (nx, ny, nz)
    cell = [key.i, key.j, key.k]
    vertex = key.vertex

    if key.role == "A" and periodic:
        # The closing plane of a finite build folds onto the first one.
        cell = [
            0 if value == bound else value for value, bound in zip(cell, bounds)
        ]

    if any(value < 0 or value >= bound for value, bound in zip(cell, bounds)):
        return None

    if key.role == "X" and vertex % 2 == 1:
        axis = vertex // 2
        if cell[axis] > 0:
            cell[axis] -= 1
            vertex = axis * 2
        elif periodic:
            cell[axis] = bounds[axis] - 1
            vertex = axis * 2
        # Otherwise it is the low face of a finite build: a site in its own right.

    resolved = SiteKey(key.role, cell[0], cell[1], cell[2], vertex)
    if canonical_index_of_key(resolved, grid_shape, periodic) < 0:
        return None
    return resolved


def resolve_key_to_indices(
    key,
    grid_shape,
    *,
    periodic: bool,
    expand_images: bool = False,
) -> List[int]:
    """Canonical build indices named by ``key``; empty when it is out of range.

    Normally one index. ``expand_images`` -- set when a *stored periodic*
    structure is being rebuilt non-periodically for rendering -- also returns the
    boundary copies that the finite build adds, so a vacancy does not visibly fill
    back in at the cell edge. A corner A site has up to 8 copies; a boundary
    oxygen has 2; a B site always has exactly 1.
    """
    if expand_images and not periodic:
        # Canonicalize against the *authoring* (periodic) grid before expanding, so
        # every spelling of a site expands to the same image set. Resolving the
        # finite way first would leave, say, X(0,0,0,-a) as a low-face site with no
        # far-face partner, and the vacancy would refill at the cell edge.
        periodic_key = canonicalize_key(key, grid_shape, True)
        if periodic_key is None:
            return []
        keys = _boundary_images(periodic_key, grid_shape)
    else:
        resolved = canonicalize_key(key, grid_shape, periodic)
        if resolved is None:
            return []
        keys = [resolved]
    indices = []
    for candidate in keys:
        index = canonical_index_of_key(candidate, grid_shape, periodic)
        if index >= 0 and index not in indices:
            indices.append(index)
    return indices


def _boundary_images(key: SiteKey, grid_shape) -> List[SiteKey]:
    """Copies of ``key`` that a finite build adds to close the periodic cell."""
    nx, ny, nz = (int(value) for value in grid_shape)
    if key.role == "B":
        return [key]
    if key.role == "A":
        # The finite A grid has one extra plane per axis, imaging the i == 0 one.
        counts = (nx, ny, nz)
        options = []
        for value, count in zip((key.i, key.j, key.k), counts):
            options.append([value, count] if value == 0 else [value])
        return [
            SiteKey("A", i, j, k)
            for i in options[0]
            for j in options[1]
            for k in options[2]
        ]
    # An X site on the far face is imaged by the low-face vertex of the opposite
    # row; the other two axes add no oxygens, so there is at most one extra copy.
    axis = key.vertex // 2
    counts = (nx, ny, nz)
    cell = [key.i, key.j, key.k]
    if cell[axis] != counts[axis] - 1:
        return [key]
    mirrored = list(cell)
    mirrored[axis] = 0
    return [key, SiteKey("X", mirrored[0], mirrored[1], mirrored[2], axis * 2 + 1)]


# ---------------------------------------------------------------------------
# Protons
# ---------------------------------------------------------------------------


def proton_direction_candidates(octahedra: np.ndarray, key: SiteKey) -> np.ndarray:
    """The four O-H directions available on the oxygen named by ``key``.

    A proton trapped in an oxide sits on an oxygen at ~0.98 A, pointing *away*
    from the two cations it bridges -- so the O-H vector lies in the plane
    perpendicular to the B-O-B axis. Taking that plane from the host octahedron's
    own two other axes gives exactly four candidates, which in the cubic limit are
    the textbook +-b, +-c sites of an a-axis oxygen.

    Deriving the directions from the octahedron rather than from a neighbour
    search is what makes them rigidly tilt-covariant: they rotate with the cage,
    so ``orientation`` names the same physical site before and after a tilt edit.
    A neighbour-ranked ordering would silently renumber itself whenever the
    geometry changed.

    Returned in the fixed order ``(+e_p, -e_p, +e_q, -e_q)`` for the two axes
    ``p < q`` other than the host vertex's own.
    """
    vertices = np.asarray(octahedra[key.i, key.j, key.k], dtype=np.float64)
    host_axis = key.vertex // 2
    directions: List[np.ndarray] = []
    for axis in (value for value in range(3) if value != host_axis):
        edge = vertices[2 * axis] - vertices[2 * axis + 1]
        norm = float(np.linalg.norm(edge))
        if norm < 1e-9:
            continue
        edge = edge / norm
        directions.extend((edge, -edge))
    return np.asarray(directions, dtype=np.float64).reshape(-1, 3)


def proton_position(
    octahedra: np.ndarray,
    key: SiteKey,
    host_coord: np.ndarray,
    *,
    orientation: int = 0,
    bond_length: float = DEFAULT_OH_BOND_LENGTH,
) -> Optional[np.ndarray]:
    """Cartesian position of a proton on the oxygen at ``host_coord``.

    ``host_coord`` is passed separately because the structure's coordinates are
    shifted by the cell origin while ``octahedra`` is in build coordinates; only
    the *direction* is taken from the cage, and directions are translation
    invariant.
    """
    candidates = proton_direction_candidates(octahedra, key)
    if not len(candidates):
        return None
    direction = candidates[int(orientation) % len(candidates)]
    return np.asarray(host_coord, dtype=np.float64) + float(bond_length) * direction


def proton_contact_distance(
    cartesian_coords: np.ndarray,
    proton_index: int,
    host_index: int,
) -> Tuple[float, int]:
    """``(distance, index)`` of the nearest atom other than the proton's own host.

    The host oxygen is excluded by construction: it sits at the O-H bond length,
    which is shorter than any contact threshold and would otherwise be flagged
    every single time.
    """
    coords = np.asarray(cartesian_coords, dtype=np.float64).reshape(-1, 3)
    deltas = np.linalg.norm(coords - coords[proton_index], axis=1)
    deltas[proton_index] = np.inf
    deltas[host_index] = np.inf
    nearest = int(np.argmin(deltas))
    return float(deltas[nearest]), nearest


def contact_warnings(
    cartesian_coords: np.ndarray,
    atomic_labels: Sequence[str],
    proton_hosts: Sequence[Tuple[int, int]],
    *,
    min_distance: float = MIN_CONTACT_DISTANCE,
) -> List[str]:
    """Report protons that landed unphysically close to another atom.

    ``proton_hosts`` pairs each proton's structure index with its host oxygen's.
    The builder leaves A cations at ideal, undisplaced positions, so a strongly
    tilted cell can crowd a proton in a way a relaxed structure would not. That is
    a property of the idealization rather than something to design around, so it
    is reported and left alone -- silently nudging the proton would break the
    guarantee that an orientation index names a fixed site.
    """
    messages: List[str] = []
    for proton_index, host_index in proton_hosts:
        distance, nearest = proton_contact_distance(
            cartesian_coords, proton_index, host_index
        )
        if distance < min_distance:
            messages.append(
                f"Proton is {distance:.2f} A from {atomic_labels[nearest]} "
                f"(below {min_distance:.2f} A); the ideal lattice leaves A cations "
                "undisplaced, so a relaxed structure would have more room."
            )
    return messages


# ---------------------------------------------------------------------------
# Applying a defect list
# ---------------------------------------------------------------------------


def resolve_defects(
    build,
    *,
    periodic: bool,
    stored_periodic: bool,
    defects: Sequence[SiteDefect],
) -> DefectResolution:
    """Work out what ``defects`` does to one canonical build.

    ``periodic`` is the build being produced; ``stored_periodic`` is the
    periodicity the defects were authored against. They differ only on the
    rendering path, which rebuilds a periodic structure as a finite cluster --
    that is when boundary images have to be expanded.

    Vacancies, then substitutions, then protons, in that fixed order, so the order
    entries appear in the list never changes the result.
    """
    grid_shape = build.octahedra.shape
    n_a, n_b, n_x = canonical_site_counts(grid_shape, periodic)
    total = n_a + n_b + n_x
    expand_images = bool(stored_periodic) and not bool(periodic)
    warnings: List[str] = []

    vacated: set[int] = set()
    substitutions: Dict[int, str] = {}
    proton_rows: List[Tuple[int, SiteKey, SiteDefect]] = []

    def resolve(defect: SiteDefect) -> List[int]:
        indices = resolve_key_to_indices(
            defect.site,
            grid_shape,
            periodic=periodic,
            expand_images=expand_images,
        )
        if not indices:
            warnings.append(
                f"{defect.kind} at {site_key_display(defect.site)} is outside this "
                "supercell and was skipped."
            )
        return indices

    for defect in defects:
        if defect.kind == "vacancy":
            vacated.update(resolve(defect))
    for defect in defects:
        if defect.kind != "substitution":
            continue
        for index in resolve(defect):
            if index in vacated:
                warnings.append(
                    f"{site_key_display(defect.site)} is both vacant and "
                    "substituted; the vacancy wins."
                )
                continue
            if index in substitutions and substitutions[index] != defect.element:
                warnings.append(
                    f"{site_key_display(defect.site)} is substituted more than "
                    f"once; using {defect.element}."
                )
            substitutions[index] = defect.element
    for defect in defects:
        if defect.kind != "proton":
            continue
        for index in resolve(defect):
            if index in vacated:
                warnings.append(
                    f"Proton on {site_key_display(defect.site)} was dropped: its "
                    "host site is vacant."
                )
                continue
            resolved = canonicalize_key(defect.site, grid_shape, periodic)
            if resolved is None:
                continue
            proton_rows.append((index, resolved, defect))

    vacancy_mask = np.zeros(total, dtype=bool)
    if vacated:
        vacancy_mask[np.fromiter(sorted(vacated), dtype=np.int64, count=len(vacated))] = True
    kept = np.flatnonzero(~vacancy_mask)
    canonical_to_structure = np.full(total, -1, dtype=np.int64)
    canonical_to_structure[kept] = np.arange(len(kept), dtype=np.int64)

    proton_coords: List[np.ndarray] = []
    proton_hosts: List[int] = []
    for index, resolved, defect in proton_rows:
        # The host's own vertex row may differ from the canonical key when a
        # boundary image was expanded; take it from the build coordinates.
        host_coord = build.all_sites[index]
        position = proton_position(
            build.octahedra,
            resolved,
            host_coord,
            orientation=defect.orientation,
            bond_length=defect.bond_length,
        )
        if position is None:
            warnings.append(
                f"Proton on {site_key_display(defect.site)} has no usable O-H "
                "direction and was skipped."
            )
            continue
        proton_coords.append(position)
        proton_hosts.append(index)

    return DefectResolution(
        canonical_to_structure=canonical_to_structure,
        kept_canonical=kept,
        substitutions=substitutions,
        proton_coords=(
            np.asarray(proton_coords, dtype=np.float64).reshape(-1, 3)
            if proton_coords
            else np.zeros((0, 3), dtype=np.float64)
        ),
        proton_host_canonical=np.asarray(proton_hosts, dtype=np.int64),
        warnings=warnings,
    )


def apply_defects(
    build,
    atomic_labels: Sequence[str],
    *,
    periodic: bool,
    stored_periodic: bool,
    defects: Sequence[SiteDefect],
    cell_origin=None,
) -> Tuple[np.ndarray, List[str], List[str], DefectResolution]:
    """Layer ``defects`` onto a canonical build.

    Returns ``(cartesian_coords, atomic_labels, site_roles, resolution)`` with the
    coordinates already shifted by ``cell_origin``. ``site_roles`` is ``"A"``,
    ``"B"`` or ``"X"`` per surviving lattice site and ``"H"`` per added proton --
    protons are appended after the X block, so the ``[A, B, X]`` block layout the
    rest of the package relies on is preserved.
    """
    keys = canonical_site_keys(build.octahedra.shape, periodic)
    labels = [str(label) for label in atomic_labels]
    if len(labels) != len(keys):
        raise ValueError(
            "apply_defects expects the canonical (A, B, X) build: "
            f"{len(keys)} grid keys but {len(labels)} labels."
        )
    resolution = resolve_defects(
        build,
        periodic=periodic,
        stored_periodic=stored_periodic,
        defects=defects,
    )
    for index, element in resolution.substitutions.items():
        labels[index] = element

    kept = resolution.kept_canonical
    coords = [np.asarray(build.all_sites, dtype=np.float64)[kept]]
    out_labels = [labels[index] for index in kept]
    out_roles = [keys[index].role for index in kept]
    if len(resolution.proton_coords):
        coords.append(resolution.proton_coords)
        out_labels.extend([PROTON_SYMBOL] * len(resolution.proton_coords))
        out_roles.extend([PROTON_ROLE] * len(resolution.proton_coords))

    cartesian = np.vstack(coords).astype(np.float64).reshape(-1, 3)
    if cell_origin is not None:
        cartesian = cartesian - np.asarray(cell_origin, dtype=np.float64).reshape(3)
    n_protons = len(resolution.proton_coords)
    if n_protons:
        # Protons are the trailing block, so their structure indices follow the
        # surviving lattice sites in the order their hosts were recorded.
        first_proton = len(cartesian) - n_protons
        resolution.warnings.extend(
            contact_warnings(
                cartesian,
                out_labels,
                [
                    (first_proton + offset, int(resolution.canonical_to_structure[host]))
                    for offset, host in enumerate(resolution.proton_host_canonical)
                ],
            )
        )
    return cartesian, out_labels, out_roles, resolution


def vacated_b_cells(
    grid_shape,
    periodic: bool,
    defects: Sequence[SiteDefect],
) -> set:
    """B-grid cells removed by ``defects``; those octahedra are not drawn."""
    cells = set()
    for defect in defects:
        if defect.kind != "vacancy" or coerce_site_key(defect.site).role != "B":
            continue
        resolved = canonicalize_key(defect.site, grid_shape, periodic)
        if resolved is not None:
            cells.add((resolved.i, resolved.j, resolved.k))
    return cells


# ---------------------------------------------------------------------------
# Charge bookkeeping
# ---------------------------------------------------------------------------


def reference_oxidation_states(atomic_labels: Sequence[str]) -> Dict[str, float]:
    """Per-element oxidation state of the best assignment for a composition.

    Mixed-valent elements collapse to their mean, which is what makes this usable
    as a *reference*: it is the charge each element carries when the cell is
    stoichiometric.
    """
    from quick_mag.oxidation_state_energy import enumerate_oxidation_states_by_energy

    labels = [str(label) for label in atomic_labels]
    if not labels:
        return {}
    try:
        found = enumerate_oxidation_states_by_energy(labels, 0, top_k=1)
    except Exception:
        found = []
    if not found:
        return {}
    states: Dict[str, float] = {}
    for element, pairs in found[0][0].items():
        total = sum(oxidation * count for oxidation, count in pairs)
        population = sum(count for _, count in pairs)
        if population:
            states[element] = total / population
    return states


def _fallback_oxidation_state(element: str) -> float:
    """Most probable oxidation state for an element absent from the reference."""
    from quick_mag.oxidation_state_enumeration import ELEMENT_OXI_SCORES

    scores = ELEMENT_OXI_SCORES.get(element)
    if not scores:
        return 0.0
    return float(max(scores.items(), key=lambda item: item[1])[0])


def compensation_hint(
    reference_labels: Sequence[str],
    defected_labels: Sequence[str],
) -> Tuple[int, str]:
    """How far the defected cell drifts from the stoichiometric charge balance.

    Asking whether the defected cell "balances at zero" is not useful -- it almost
    always does, by pushing a cation off its normal valence. ``LaFeO3`` with one
    Fe replaced by Zn balances only by promoting another Fe to Fe(4+); the same
    cell plus one proton balances with every Fe at 3+. So the question that
    matters is the one the user is actually asking: *holding the elements at the
    charges they carry in the stoichiometric cell, how far off is this one?*

    Returns ``(nominal_charge, message)``. A negative nominal charge is a cation
    deficit and calls for protons; a positive one (an oxygen vacancy, say) is
    absorbed by reducing cations, which the oxidation-state enumerator does on its
    own -- so no proton is wanted there.
    """
    states = reference_oxidation_states(reference_labels)
    if not states:
        return 0, "No reference oxidation states; charge unknown."
    from quick_mag.structure import _ELEMENT_SYMBOL_RE

    total = 0.0
    for label in defected_labels:
        match = _ELEMENT_SYMBOL_RE.match(str(label))
        symbol = match.group(1) if match else str(label)
        total += states.get(symbol) or _fallback_oxidation_state(symbol)
    nominal = int(round(total))
    if nominal == 0:
        return 0, "Charge balanced against the stoichiometric cell."
    if nominal < 0:
        protons = -nominal
        plural = "s" if protons > 1 else ""
        return nominal, (
            f"Net {nominal:+d} vs. the stoichiometric cell: add {protons} "
            f"proton{plural} to compensate."
        )
    plural = "s" if nominal > 1 else ""
    return nominal, (
        f"Net {nominal:+d} vs. the stoichiometric cell: compensated by reducing "
        f"{nominal} cation{plural}."
    )
