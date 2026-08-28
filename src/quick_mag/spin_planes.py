"""Collinear magnetic orderings as sign strings across families of lattice planes.

A collinear ordering assigns +1 or -1 to every magnetic site. Every ordering this
package cares about has the same shape: sort the magnetic sites into parallel
planes, then walk those planes in order applying a repeating sign string. The
plane family is named by Miller indices and the string by its ``+``/``-`` pattern,
so ``((1, 1, 1), "+-")`` is "alternate the sign on successive (111) planes" -- which
is G-type.

That is not a reinterpretation of the perovskite orderings, it *is* them. The eight
period-2 patterns over low Miller indices are exactly F, the three A orientations,
the three C orientations, and G, with nothing left over:

    F    (000)      A(a) (100)   A(b) (010)   A(c) (001)
    G    (111)      C(a) (011)   C(b) (101)   C(c) (110)

Two things come from saying it this way. The site -> plane map needs only fractional
coordinates, so an ordering can be written down for any structure rather than only
for a perovskite B-site grid; and a real configuration can be compared against an
ideal one site by site, which turns "is this G-type" into "how far from G-type is
this" -- see :func:`defect_concentration`.

Longer strings extend the family: ``"++--"`` is the up-up-down-down modulation, so
``((1, 0, 0), "++--")`` is E-type along a.

**Which frame the Miller indices are in matters.** They index the *magnetic
sublattice's* own lattice, not the structure's cell. Those coincide only when the
cell holds one magnetic site per axis repeat; on a 4x2x2 supercell the pseudocubic
(111) planes are the cell's (422) planes, and reading (111) in the cell frame would
mix up sites that G-type puts on opposite spins. :class:`MagneticSublattice` carries
the change of basis, and :meth:`MagneticSublattice.miller_in_cell` converts an index
back to the cell frame for drawing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


AXIS_NAMES = ("a", "b", "c")

# Sites whose plane coordinate differs by less than this are on the same plane.
# Generous next to the ~1e-16 spread a builder structure actually shows, and still
# far below the 1/n spacing between neighbouring planes of any cell worth drawing.
DEFAULT_PLANE_TOL = 1e-6


def format_miller(miller: tuple[int, int, int]) -> str:
    """``(110)`` for single digits; comma-separated once an index needs a sign.

    Public because the defect-plane overlay labels its sheets the same way -- and
    ``(1-10)`` for ``(1, -1, 0)`` is not a Miller index anyone can read.
    """
    if all(0 <= value <= 9 for value in miller):
        return "({}{}{})".format(*miller)
    return "({}, {}, {})".format(*miller)


@dataclass(frozen=True)
class PlanePattern:
    """One collinear ordering: a sign string repeated across a plane family.

    ``name`` is the classical label where one exists (``"G"``, ``"A(c)"``); patterns
    without one fall back to plane notation, so a novel ordering is still nameable.
    """

    miller: tuple[int, int, int]
    signs: str
    name: str = ""

    def __post_init__(self) -> None:
        miller = tuple(int(value) for value in self.miller)
        if len(miller) != 3:
            raise ValueError("Miller indices must have three components.")
        object.__setattr__(self, "miller", miller)
        signs = str(self.signs)
        if not signs or set(signs) - {"+", "-"}:
            raise ValueError(f"Sign string must be non-empty '+'/'-', got {self.signs!r}.")
        object.__setattr__(self, "signs", signs)

    @property
    def period(self) -> int:
        return len(self.signs)

    @property
    def label(self) -> str:
        return self.name or f"{format_miller(self.miller)} {self.signs}"

    @property
    def plane_label(self) -> str:
        """Plane notation, whether or not the pattern also has a classical name."""
        return f"{format_miller(self.miller)} {self.signs}"


def plane_ordinals(
    fractional_coords: np.ndarray,
    miller: Sequence[int],
    *,
    tol: float = DEFAULT_PLANE_TOL,
) -> np.ndarray:
    """Group sites into planes of the ``miller`` family, numbered along the normal.

    Cell-frame only, and for *drawing*: it numbers the planes the sites happen to
    occupy, in order, which is what you want when deciding where to put a sheet.
    It is deliberately not what :func:`plane_indices` does -- numbering occupied
    planes consecutively silently merges distinct planes of the family when the
    occupancy is uneven, which is fine for drawing and wrong for signs.
    """
    coords = np.asarray(fractional_coords, dtype=np.float64).reshape(-1, 3)
    if coords.shape[0] == 0:
        return np.zeros(0, dtype=int)

    distance = coords @ np.asarray(miller, dtype=np.float64).reshape(3)
    order = np.argsort(distance, kind="stable")
    ordinals = np.empty(coords.shape[0], dtype=int)

    current = 0
    # Compared against the plane's first member, not the previous site, so a long run
    # of near-ties cannot drift a plane apart one tolerance at a time.
    reference = float(distance[order[0]])
    ordinals[order[0]] = 0
    for position in order[1:]:
        if float(distance[position]) - reference > tol:
            current += 1
            reference = float(distance[position])
        ordinals[position] = current
    return ordinals


def plane_offsets(
    fractional_coords: np.ndarray,
    miller: Sequence[int],
    *,
    tol: float = DEFAULT_PLANE_TOL,
) -> np.ndarray:
    """The plane coordinate of each occupied plane, in ordinal order.

    What :func:`plane_cell_polygon` needs to draw the planes the sites actually lie
    on, rather than the whole infinite family.
    """
    coords = np.asarray(fractional_coords, dtype=np.float64).reshape(-1, 3)
    if coords.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    distance = coords @ np.asarray(miller, dtype=np.float64).reshape(3)
    ordinals = plane_ordinals(coords, miller, tol=tol)
    return np.array(
        [float(distance[ordinals == value].mean()) for value in range(ordinals.max() + 1)],
        dtype=np.float64,
    )


def signs_from_ordinals(
    ordinals: np.ndarray,
    signs: str,
    *,
    phase: int = 0,
) -> np.ndarray:
    """Apply a sign string cyclically to plane indices."""
    lookup = np.array([1.0 if char == "+" else -1.0 for char in signs], dtype=np.float64)
    return lookup[(np.asarray(ordinals, dtype=int) + int(phase)) % lookup.size]


@dataclass(frozen=True)
class MagneticSublattice:
    """Magnetic sites as integer coordinates in the lattice they themselves form.

    Plane patterns are defined against *this* frame. ``basis`` records how it sits
    in the structure's cell -- its rows are the sublattice's primitive vectors in
    the cell's fractional coordinates -- which is what lets a pattern's Miller index
    be converted back to the cell for drawing.
    """

    lattice_coords: np.ndarray  # (N, 3) integer coordinates, one row per site
    basis: np.ndarray  # (3, 3) sublattice vectors in cell-fractional coordinates
    site_indices: np.ndarray  # (N,) indices into the structure's atom list

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "lattice_coords", np.asarray(self.lattice_coords, dtype=int).reshape(-1, 3)
        )
        object.__setattr__(self, "basis", np.asarray(self.basis, dtype=np.float64).reshape(3, 3))
        object.__setattr__(
            self, "site_indices", np.asarray(self.site_indices, dtype=int).reshape(-1)
        )

    @property
    def size(self) -> int:
        return int(self.lattice_coords.shape[0])

    @classmethod
    def from_grid(
        cls,
        grid_shape: Sequence[int],
        grid_to_site: Sequence[int] | None = None,
    ) -> "MagneticSublattice":
        """Build from a perovskite-style B-site grid.

        ``grid_to_site`` maps each cell to a structure atom index, with ``-1`` for a
        cell whose site was removed by a vacancy; those cells are dropped, so the
        surviving sites keep their true lattice coordinates.
        """
        shape = [max(1, int(size)) for size in grid_shape]
        cells = np.array(
            [
                (i, j, k)
                for i in range(shape[0])
                for j in range(shape[1])
                for k in range(shape[2])
            ],
            dtype=int,
        )
        if grid_to_site is None:
            sites = np.arange(cells.shape[0], dtype=int)
        else:
            sites = np.asarray(grid_to_site, dtype=int).reshape(-1)
            if sites.size != cells.shape[0]:
                raise ValueError(
                    f"grid_to_site has {sites.size} entries for a {shape} grid."
                )
        present = sites >= 0
        return cls(
            lattice_coords=cells[present],
            basis=np.diag(1.0 / np.asarray(shape, dtype=np.float64)),
            site_indices=sites[present],
        )

    def miller_in_cell(self, miller: Sequence[int]) -> np.ndarray:
        """``miller`` re-expressed in the structure cell's frame.

        A site at sublattice coordinate ``u`` sits at ``s = u @ basis``, so a plane
        ``miller . u`` is the cell-frame plane ``(inv(basis) @ miller) . s``. On a
        4x2x2 grid that turns the pseudocubic (111) into the cell's (422).
        """
        return np.linalg.pinv(self.basis) @ np.asarray(miller, dtype=np.float64).reshape(3)


def plane_indices(lattice_coords: np.ndarray, miller: Sequence[int]) -> np.ndarray:
    """Which plane of the ``miller`` family each site is on, numbered from 0.

    In the sublattice frame the projection is an integer, so this is just the dot
    product -- no binning, no tolerance, and correct whatever the cell's shape.

    The index is left tied to the lattice origin rather than rebased onto the
    lowest occupied plane. Rebasing would make the sign a site removed elsewhere in
    the cell, which is how a vacancy ends up silently flipping every spin in the
    reference ordering it generates.
    """
    coords = np.asarray(lattice_coords, dtype=np.float64).reshape(-1, 3)
    if coords.shape[0] == 0:
        return np.zeros(0, dtype=int)
    return np.rint(coords @ np.asarray(miller, dtype=np.float64).reshape(3)).astype(int)


def pattern_signs(
    lattice_coords: np.ndarray,
    pattern: "PlanePattern",
    *,
    phase: int = 0,
) -> np.ndarray:
    """The ideal +-1 per site for ``pattern``."""
    return signs_from_ordinals(
        plane_indices(lattice_coords, pattern.miller), pattern.signs, phase=phase
    )


# Listed in the order the UI shows them: the period-2 family in its established
# legend order, then the period-4 family. The three axis-aligned period-4 patterns
# are named E after this package's existing definition of E-type -- an axis carrying
# an up-up-down-down modulation (see SiteSpinClassification.sigmas2). The diagonal
# ones have no classical name and display as plane notation.
# F sits on a single plane taking a single sign. Writing it "+" rather than "+-"
# keeps "how many planes does this pattern need" equal to the string length for
# every pattern, which is what patterns_for_sites filters on.
_PERIOD_2 = (
    ((1, 1, 1), "+-", "G"),
    ((0, 1, 1), "+-", "C(a)"),
    ((1, 0, 1), "+-", "C(b)"),
    ((1, 1, 0), "+-", "C(c)"),
    ((0, 0, 0), "+", "F"),
    ((1, 0, 0), "+-", "A(a)"),
    ((0, 1, 0), "+-", "A(b)"),
    ((0, 0, 1), "+-", "A(c)"),
)
# (0, 0, 0) is omitted: a single plane takes signs[0] whatever the string, so it
# would just be F again.
_PERIOD_4 = (
    ((1, 0, 0), "++--", "E(a)"),
    ((0, 1, 0), "++--", "E(b)"),
    ((0, 0, 1), "++--", "E(c)"),
    ((0, 1, 1), "++--", ""),
    ((1, 0, 1), "++--", ""),
    ((1, 1, 0), "++--", ""),
    ((1, 1, 1), "++--", ""),
)

CANONICAL_PLANE_PATTERNS: tuple[PlanePattern, ...] = tuple(
    PlanePattern(miller, signs, name) for miller, signs, name in _PERIOD_2 + _PERIOD_4
)

PATTERNS_BY_NAME: dict[str, PlanePattern] = {
    pattern.name: pattern for pattern in CANONICAL_PLANE_PATTERNS if pattern.name
}


def plane_count(lattice_coords: np.ndarray, miller: Sequence[int]) -> int:
    """How many planes of the ``miller`` family the sites are spread over."""
    indices = plane_indices(lattice_coords, miller)
    return 0 if indices.size == 0 else int(indices.max()) - int(indices.min()) + 1


def parse_plane_label(text: str) -> PlanePattern | None:
    """Inverse of :attr:`PlanePattern.plane_label` -- ``"(011) ++--"`` back to a pattern.

    Lets a pattern without a classical name survive a round trip through a label,
    which is how the unnamed orderings reach code that passes patterns around by
    string (the reference scorer, saved configurations, the CLI).
    """
    body, _, signs = str(text).strip().partition(")")
    signs = signs.strip()
    body = body.strip()
    if not body.startswith("(") or not signs or set(signs) - {"+", "-"}:
        return None
    digits = body[1:].strip()
    try:
        if "," in digits:
            values = [int(part) for part in digits.split(",")]
        else:
            values = [int(char) for char in digits.split()[0]] if digits.split() else []
    except ValueError:
        return None
    if len(values) != 3:
        return None
    return PlanePattern((values[0], values[1], values[2]), signs)


def ordering_key(signs: np.ndarray) -> tuple[float, ...]:
    """Hashable identity of a sign assignment, up to a global spin flip."""
    values = tuple(float(value) for value in np.asarray(signs, dtype=np.float64).ravel())
    return min(values, tuple(-value for value in values))


def patterns_for_sites(
    lattice_coords: np.ndarray,
    patterns: Sequence[PlanePattern] | None = None,
) -> tuple[PlanePattern, ...]:
    """The patterns these sites can actually tell apart.

    Two filters, both of which matter on a small or anisotropic cell:

    * a pattern needs at least as many planes as its string is long -- alternating
      across a single plane is only F, and up-up-down-down across two planes is only
      ferromagnetic;
    * patterns that land on the *same* assignment for these particular sites are
      collapsed to the first. On a 1x2x2 B grid, for instance, G and C(a) are
      literally the same state, and scoring both would put one ordering in the
      reference set twice under two names.

    Order is preserved, so the canonical listing decides which name a collapsed
    group keeps.
    """
    candidates = CANONICAL_PLANE_PATTERNS if patterns is None else tuple(patterns)
    kept: list[PlanePattern] = []
    seen: set[tuple[float, ...]] = set()
    for pattern in candidates:
        if plane_count(lattice_coords, pattern.miller) < pattern.period:
            continue
        key = ordering_key(pattern_signs(lattice_coords, pattern))
        if key in seen:
            continue
        seen.add(key)
        kept.append(pattern)
    return tuple(kept)


@dataclass(frozen=True)
class PatternMatch:
    """How close a configuration is to one ideal ordering."""

    pattern: PlanePattern
    concentration: float
    mismatched: np.ndarray  # bool per site, True where the spin disagrees
    phase: int
    flipped: bool

    @property
    def is_exact(self) -> bool:
        return self.defect_count == 0

    @property
    def defect_count(self) -> int:
        return int(np.count_nonzero(self.mismatched))


def build_plane_index(
    lattice_coords: np.ndarray,
    patterns: Sequence[PlanePattern],
) -> tuple[np.ndarray, ...]:
    """Plane index per site, per pattern, for one fixed set of sites.

    Depends only on where the sites are, not on their spins, so scoring a whole
    landscape against the patterns should compute this once rather than once per
    configuration.
    """
    return tuple(plane_indices(lattice_coords, pattern.miller) for pattern in patterns)


def defect_concentration(
    actual_signs: np.ndarray,
    lattice_coords: np.ndarray,
    pattern: PlanePattern,
    *,
    plane_index: np.ndarray | None = None,
) -> PatternMatch:
    """How far ``actual_signs`` is from ``pattern``, as a fraction of magnetic sites.

    Minimized over the global spin flip and over every phase of the sign string,
    since neither changes which ordering a configuration *is*. That bounds the result
    at 0.5: past halfway, flipping the comparison is the better match.

    Sites with no moment take no part -- they are neither matched nor counted -- so
    the denominator is the number of sites actually carrying a spin. A configuration
    with no moments at all reports 0.0 against every pattern; callers that need to
    distinguish that case should check :func:`best_matching_pattern`, which returns
    ``None`` for it.
    """
    actual = np.sign(np.asarray(actual_signs, dtype=np.float64).reshape(-1))
    coords = np.asarray(lattice_coords, dtype=np.float64).reshape(-1, 3)
    if actual.size != coords.shape[0]:
        raise ValueError(
            f"{actual.size} spins but {coords.shape[0]} sites -- they must correspond."
        )

    active = actual != 0.0
    n_active = int(np.count_nonzero(active))
    mismatched = np.zeros(actual.size, dtype=bool)
    if n_active == 0:
        return PatternMatch(pattern, 0.0, mismatched, 0, False)

    if plane_index is None:
        ordinals = plane_indices(coords[active], pattern.miller)
    else:
        ordinals = np.asarray(plane_index, dtype=int).reshape(-1)[active]
    observed = actual[active]

    best_count: int | None = None
    best: tuple[np.ndarray, int, bool] | None = None
    for phase in range(pattern.period):
        ideal = signs_from_ordinals(ordinals, pattern.signs, phase=phase)
        for flipped in (False, True):
            candidate = observed != (-ideal if flipped else ideal)
            count = int(np.count_nonzero(candidate))
            if best_count is None or count < best_count:
                best_count, best = count, (candidate, phase, flipped)

    assert best is not None and best_count is not None
    candidate, phase, flipped = best
    mismatched[active] = candidate
    return PatternMatch(pattern, best_count / n_active, mismatched, phase, flipped)


def best_matching_pattern(
    actual_signs: np.ndarray,
    lattice_coords: np.ndarray,
    patterns: Sequence[PlanePattern] | None = None,
    *,
    plane_index: Sequence[np.ndarray] | None = None,
) -> PatternMatch | None:
    """The nearest ideal ordering, or ``None`` when no site carries a moment.

    Ties go to whichever pattern comes first in ``patterns``, so the canonical order
    decides between orderings that describe a configuration equally well.
    """
    actual = np.asarray(actual_signs, dtype=np.float64).reshape(-1)
    if not np.any(np.sign(actual) != 0.0):
        return None
    candidates = CANONICAL_PLANE_PATTERNS if patterns is None else tuple(patterns)
    if not candidates:
        return None
    if plane_index is None:
        plane_index = build_plane_index(lattice_coords, candidates)
    matches = (
        defect_concentration(actual, lattice_coords, pattern, plane_index=index)
        for pattern, index in zip(candidates, plane_index)
    )
    return min(matches, key=lambda match: match.concentration)


# ----------------------------------------------------------------------------
# Drawing a plane family
# ----------------------------------------------------------------------------

# The twelve edges of the unit cell as index pairs into the eight corners, ordered
# (i, j, k) with k fastest -- the same corner order unit_cell_vertices uses.
_CELL_CORNERS = np.array(
    [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.float64
)
_CELL_EDGES = tuple(
    (a, b)
    for a in range(8)
    for b in range(a + 1, 8)
    # Corners differing in exactly one coordinate share an edge.
    if int(np.count_nonzero(_CELL_CORNERS[a] != _CELL_CORNERS[b])) == 1
)


def plane_cell_polygon(
    lattice: np.ndarray,
    miller: Sequence[int],
    offset: float,
    *,
    tol: float = 1e-9,
) -> np.ndarray:
    """Cartesian polygon where one plane of the family cuts the unit cell.

    The plane is ``miller . s == offset`` in fractional coordinates. Returns the
    convex cross-section wound consistently in-plane, or an empty ``(0, 3)`` array
    when the plane misses the cell (or ``miller`` is degenerate).
    """
    normal = np.asarray(miller, dtype=np.float64).reshape(3)
    if not np.any(normal):
        return np.zeros((0, 3), dtype=np.float64)

    height = _CELL_CORNERS @ normal - float(offset)
    points: list[np.ndarray] = []
    for start, stop in _CELL_EDGES:
        h0, h1 = float(height[start]), float(height[stop])
        if abs(h0) <= tol:
            points.append(_CELL_CORNERS[start])
        if abs(h1) <= tol:
            points.append(_CELL_CORNERS[stop])
        if abs(h0) <= tol or abs(h1) <= tol or (h0 > 0) == (h1 > 0):
            continue
        fraction = h0 / (h0 - h1)
        points.append(_CELL_CORNERS[start] + fraction * (_CELL_CORNERS[stop] - _CELL_CORNERS[start]))
    if len(points) < 3:
        return np.zeros((0, 3), dtype=np.float64)

    cartesian = np.asarray(points, dtype=np.float64) @ np.asarray(lattice, dtype=np.float64)
    # Corner hits arrive once per edge that meets there, so deduplicate.
    kept: list[np.ndarray] = []
    for point in cartesian:
        if not any(float(np.linalg.norm(point - other)) <= 1e-7 for other in kept):
            kept.append(point)
    if len(kept) < 3:
        return np.zeros((0, 3), dtype=np.float64)

    polygon = np.asarray(kept, dtype=np.float64)
    centroid = polygon.mean(axis=0)
    relative = polygon - centroid
    # An in-plane basis, taken from the widest spread so it cannot be degenerate.
    first = relative[int(np.argmax(np.linalg.norm(relative, axis=1)))]
    first = first / np.linalg.norm(first)
    # r = s @ lattice, so a plane 'normal . s == offset' has cartesian normal
    # inv(lattice) @ normal -- not its transpose, which only agrees for an
    # orthogonal cell and mis-winds the polygon for anything else.
    plane_normal = np.linalg.pinv(np.asarray(lattice, dtype=np.float64)) @ normal
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    second = np.cross(plane_normal, first)
    angles = np.arctan2(relative @ second, relative @ first)
    return polygon[np.argsort(angles)]


def polygon_triangles(polygon: np.ndarray) -> np.ndarray:
    """Fan-triangulate a convex polygon into a flat ``(3T, 3)`` vertex array.

    The layout ``implot3d.plot_triangle`` consumes, matching how the octahedra are
    already drawn.
    """
    points = np.asarray(polygon, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 3:
        return np.zeros((0, 3), dtype=np.float64)
    fan = [
        vertex
        for index in range(1, points.shape[0] - 1)
        for vertex in (points[0], points[index], points[index + 1])
    ]
    return np.asarray(fan, dtype=np.float64)
