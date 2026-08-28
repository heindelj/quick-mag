"""Lattice planes as a way of *naming* the sites a defect goes on.

A defect is stored as a :class:`~quick_mag.perovskite_builder.SiteKey` -- a grid
address -- and that is not going to change. What this module adds is a way to
*find* one: instead of an ordinal ("site 37 of 95"), name a Miller plane family
and step through its members, and the sites lying in the current plane are the
candidates.

The one thing that makes it work is the doubled cube coordinate. In units of the
cube edge, measured from the cell origin (the ``A`` site of cell ``(0,0,0)``),
``build_perovskite`` puts

======  ==========================  ==========================
site    cube coordinate             doubled cube coordinate
======  ==========================  ==========================
A       ``(i, j, k)``               ``(2i, 2j, 2k)``
B       ``(i+1/2, j+1/2, k+1/2)``   ``(2i+1, 2j+1, 2k+1)``
X       B +- 1/2 on one axis        B +- 1 on one axis
======  ==========================  ==========================

so *doubling* makes every site an integer point, and a plane family ``(hkl)`` is
exactly ``h*u + k*v + l*w == m`` over the integers -- no tolerance, no binning,
and the planes step by *half* a cube edge. That half step is the whole point: a
whole-cell step would only ever reach one sublattice, whereas the half step
alternates. For ``(001)`` the even planes hold the A sites and the apical
oxygens -- the ``AO`` layer -- and the odd ones hold the B sites and the
equatorial oxygens -- the ``BO2`` layer.

Contrast :mod:`quick_mag.spin_planes`, which numbers planes over the *magnetic
sublattice* alone. That frame cannot express a plane through the A or X sites at
all, which is why this is a separate module rather than a reuse.
"""

from __future__ import annotations

from math import gcd
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from quick_mag.defects import coerce_site_key
from quick_mag.perovskite_builder import SiteKey, canonical_site_keys

# Role captions for the plane, in the order a caption lists them.
_ROLE_ORDER: Tuple[str, ...] = ("A", "B", "X")


def coerce_miller(miller: Sequence[int]) -> Tuple[int, int, int]:
    """``miller`` as three ints. Non-integers are rejected rather than rounded."""
    values = tuple(int(value) for value in tuple(miller)[:3])
    if len(values) != 3:
        raise ValueError(f"Miller indices are (h, k, l); got {len(values)} of them.")
    return values


def site_key_cube_coords(key) -> Tuple[int, int, int]:
    """Doubled cube coordinate of one site, measured from the cell origin.

    Doubled so that the B centres and the octahedron vertices -- which sit on
    halves of the cube edge -- come out integral along with the A corners.
    """
    key = coerce_site_key(key)
    if key.role == "A":
        return (2 * key.i, 2 * key.j, 2 * key.k)
    coords = [2 * key.i + 1, 2 * key.j + 1, 2 * key.k + 1]
    if key.role == "X":
        # VERTEX_NAMES is ("+a","-a","+b","-b","+c","-c"), matching the rows of
        # ``build_perovskite``'s vertex_offsets: axis = vertex // 2, and the even
        # rows are the positive direction.
        axis = key.vertex // 2
        coords[axis] += 1 if key.vertex % 2 == 0 else -1
    return (coords[0], coords[1], coords[2])


def plane_period(grid_shape, periodic: bool, miller: Sequence[int]) -> int:
    """How far along the family one supercell translation moves; 0 when finite.

    Under periodic boundaries a plane and the plane one cell along the normal are
    the *same* layer, and the canonical key set does name both: with only the
    ``+a``/``+b``/``+c`` vertex rows kept, the apical oxygen of the top cell lands
    a whole cell above the A plane it actually shares. Translating by the
    supercell vector along an axis shifts the plane index by ``2 * n_axis * h``,
    so the aliases repeat with the gcd of those three shifts and folding by it
    puts each layer back together.

    A finite build has no such aliasing -- its closing layer is a genuinely
    separate set of atoms -- so this returns 0 and nothing is folded.
    """
    if not periodic:
        return 0
    h, k, l = coerce_miller(miller)
    nx, ny, nz = (int(value) for value in grid_shape)
    shifts = [abs(2 * nx * h), abs(2 * ny * k), abs(2 * nz * l)]
    nonzero = [shift for shift in shifts if shift]
    if not nonzero:
        return 0
    period = 0
    for shift in nonzero:
        period = gcd(period, shift)
    return period


def fold_plane_index(plane: int, period: int) -> int:
    """``plane`` reduced into ``[0, period)``, or unchanged when ``period`` is 0."""
    period = int(period)
    return int(plane) % period if period > 0 else int(plane)


def plane_index_of_key(key, miller: Sequence[int], *, period: int = 0) -> int:
    """Which plane of the ``miller`` family a site lies on.

    Exact integer arithmetic in the doubled cube frame, so two sites share a
    plane if and only if this returns the same number for both. Pass ``period``
    from :func:`plane_period` to fold periodic aliases together.
    """
    h, k, l = coerce_miller(miller)
    u, v, w = site_key_cube_coords(key)
    return fold_plane_index(h * u + k * v + l * w, period)


def _candidate_keys(grid_shape, periodic: bool) -> List[SiteKey]:
    """Every canonical site of the build, in build order.

    Going through :func:`canonical_site_keys` rather than generating coordinates
    here means the closing A layer and the boundary X faces of a finite build are
    included exactly when the build has them.
    """
    return canonical_site_keys(grid_shape, periodic)


def sites_in_plane(
    grid_shape,
    periodic: bool,
    miller: Sequence[int],
    plane: int,
    *,
    role: str = "",
) -> List[SiteKey]:
    """Canonical sites lying in one plane, in build order.

    ``role`` optionally narrows to one of ``A``/``B``/``X``; the empty string
    keeps every role, which is the interesting case -- an ``AO`` plane holds both
    A sites and oxygens, and either may be the one the user wants.
    """
    if not any(coerce_miller(miller)):
        return []
    wanted = str(role).strip().upper()
    period = plane_period(grid_shape, periodic, miller)
    # Only the *keys* are folded, never the plane being asked for. Folding the
    # request too would quietly relocate a plane authored in a larger supercell
    # onto whichever layer it happens to be congruent to in the smaller one --
    # the same silent rewrite ``canonicalize_key`` refuses to do for site
    # addresses. An index with no sites simply has none, and the panel says so.
    plane = int(plane)
    return [
        key
        for key in _candidate_keys(grid_shape, periodic)
        if (not wanted or key.role == wanted)
        and plane_index_of_key(key, miller, period=period) == plane
    ]


def occupied_planes(
    grid_shape,
    periodic: bool,
    miller: Sequence[int],
    *,
    role: str = "",
) -> List[int]:
    """The plane indices that hold at least one site, ascending.

    This is what bounds the plane slider, so it can only ever name a plane the
    lattice actually has sites on. Planes of the family that fall between two
    layers of atoms are simply absent from the list rather than being offered and
    drawn empty.
    """
    if not any(coerce_miller(miller)):
        return []
    wanted = str(role).strip().upper()
    period = plane_period(grid_shape, periodic, miller)
    return sorted(
        {
            plane_index_of_key(key, miller, period=period)
            for key in _candidate_keys(grid_shape, periodic)
            if not wanted or key.role == wanted
        }
    )


def plane_role_counts(
    grid_shape,
    periodic: bool,
    miller: Sequence[int],
    plane: int,
) -> Dict[str, int]:
    """How many sites of each role the plane holds. Absent roles are omitted."""
    counts: Dict[str, int] = {}
    for key in sites_in_plane(grid_shape, periodic, miller, plane):
        counts[key.role] = counts.get(key.role, 0) + 1
    return counts


def plane_role_label(
    grid_shape,
    periodic: bool,
    miller: Sequence[int],
    plane: int,
) -> str:
    """A caption naming what the plane cuts, e.g. ``"A + X"``.

    This is what makes the slider legible: stepping through ``(001)`` reads as
    ``A + X`` / ``B + X`` alternating, which is the ``AO``/``BO2`` layering, not
    an anonymous run of numbers.
    """
    counts = plane_role_counts(grid_shape, periodic, miller, plane)
    present = [role for role in _ROLE_ORDER if counts.get(role)]
    if not present:
        return "empty"
    return " + ".join(present)


def plane_miller_in_cell(grid_shape, miller: Sequence[int]) -> np.ndarray:
    """``miller`` re-expressed in the supercell's fractional frame.

    A cube coordinate is ``n_axis`` times the cell fraction along that axis, so
    the family's normal picks up the grid shape. Only the *direction* is set
    here; where along it a given plane sits has to be measured from the sites
    (the builder's coordinates carry an origin offset the lattice does not).
    """
    h, k, l = coerce_miller(miller)
    nx, ny, nz = (int(value) for value in grid_shape)
    return np.array([h * nx, k * ny, l * nz], dtype=np.float64)


def nearest_occupied_plane(planes: Iterable[int], plane: int) -> int:
    """The member of ``planes`` closest to ``plane``, or ``plane`` if empty.

    Used when a Miller change strands the stored plane index: the slider lands on
    the nearest real plane instead of jumping to the start of the family.
    """
    candidates = list(planes)
    if not candidates:
        return int(plane)
    return min(candidates, key=lambda value: (abs(value - int(plane)), value))
