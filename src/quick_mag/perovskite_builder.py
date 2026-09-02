from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from quick_mag.tilt_systems import apply_glazer_tilts


@dataclass(frozen=True)
class PerovskiteBuild:
    all_sites: np.ndarray
    a_site_indices: np.ndarray
    b_site_indices: np.ndarray
    x_site_indices: np.ndarray
    a_sites: np.ndarray
    b_sites: np.ndarray
    x_sites: np.ndarray
    octahedra: np.ndarray


# Octahedron vertex rows, in the order ``build_perovskite`` lays them out.
VERTEX_NAMES: tuple[str, ...] = ("+a", "-a", "+b", "-b", "+c", "-c")

# The vertex rows kept for every cell; the odd rows survive only on the three
# boundary faces of a non-periodic build (see ``canonical_site_keys``).
_KEPT_VERTEX_ROWS: tuple[int, ...] = (0, 2, 4)


class SiteKey(NamedTuple):
    """A stable address for one built site, independent of array order.

    Site *positions* move whenever a tilt angle or lattice constant changes, and
    site *indices* shift whenever the supercell is resized -- but the grid address
    of, say, "the +a oxygen of octahedron (1, 0, 2)" never changes. That is what
    lets ``quick_mag.defects`` pin a defect to a site across a rebuild.

    ``vertex`` is the octahedron vertex row (see ``VERTEX_NAMES``) for X sites and
    is unused, always 0, for A and B sites.
    """

    role: str
    i: int
    j: int
    k: int
    vertex: int = 0


PeriodicSpec = "bool | tuple[bool, bool, bool] | list[bool]"


def periodic_axes(periodic) -> tuple[bool, bool, bool]:
    """Normalize a periodicity flag to one boolean per lattice axis.

    Every builder function accepts either form: a single bool applies to all three
    axes, a length-3 sequence names the axes individually. This is what lets a
    structure be periodic in the plane of a film and finite along its growth
    direction.
    """
    if isinstance(periodic, (bool, np.bool_, int)) and not isinstance(periodic, (tuple, list)):
        flag = bool(periodic)
        return flag, flag, flag
    values = tuple(bool(value) for value in periodic)
    if len(values) != 3:
        raise ValueError("periodic must be a bool or a sequence of three bools.")
    return values  # type: ignore[return-value]


def a_site_grid_counts(grid_shape, periodic) -> tuple[int, int, int]:
    """A-site grid shape; one extra layer closes each finite (non-periodic) axis."""
    nx, ny, nz = (int(value) for value in grid_shape)
    px, py, pz = periodic_axes(periodic)
    return nx + (0 if px else 1), ny + (0 if py else 1), nz + (0 if pz else 1)


def canonical_site_keys(grid_shape, periodic) -> list[SiteKey]:
    """Grid keys for a build, in the order ``build_perovskite`` stacks its sites.

    This is the single definition of the canonical site order: the two
    ``index_deduplicated_*`` functions below generate their coordinates by walking
    this list, so key ``n`` always addresses ``build.all_sites[n]`` and the two
    cannot drift apart.

    ``grid_shape`` is the octahedron grid, i.e. ``build.octahedra.shape``.
    """
    nx, ny, nz = (int(value) for value in grid_shape)
    count_x, count_y, count_z = a_site_grid_counts(grid_shape, periodic)

    keys = [
        SiteKey("A", i, j, k)
        for i in range(count_x)
        for j in range(count_y)
        for k in range(count_z)
    ]
    keys += [
        SiteKey("B", i, j, k)
        for i in range(nx)
        for j in range(ny)
        for k in range(nz)
    ]
    keys += [
        SiteKey("X", i, j, k, vertex)
        for i in range(nx)
        for j in range(ny)
        for k in range(nz)
        for vertex in _KEPT_VERTEX_ROWS
    ]
    px, py, pz = periodic_axes(periodic)
    # The faces whose corner-shared oxygens have no owning cell to the negative
    # side, so they are not covered by the +a/+b/+c rows above. Only a finite
    # axis has such a face.
    if not px:
        keys += [SiteKey("X", 0, j, k, 1) for j in range(ny) for k in range(nz)]
    if not py:
        keys += [SiteKey("X", i, 0, k, 3) for i in range(nx) for k in range(nz)]
    if not pz:
        keys += [SiteKey("X", i, j, 0, 5) for i in range(nx) for j in range(ny)]
    return keys


def canonical_site_counts(grid_shape, periodic) -> tuple[int, int, int]:
    """``(n_a, n_b, n_x)`` for a fully occupied build."""
    count_x, count_y, count_z = a_site_grid_counts(grid_shape, periodic)
    nx, ny, nz = (int(value) for value in grid_shape)
    px, py, pz = periodic_axes(periodic)
    n_a = count_x * count_y * count_z
    n_b = nx * ny * nz
    n_x = 3 * n_b
    if not px:
        n_x += ny * nz
    if not py:
        n_x += nx * nz
    if not pz:
        n_x += nx * ny
    return n_a, n_b, n_x


def canonical_index_of_key(key: SiteKey, grid_shape, periodic) -> int:
    """Position of ``key`` in :func:`canonical_site_keys`, or -1 if it has none.

    Closed form, so resolving a defect does not cost a full key enumeration. Keys
    outside the grid, and the odd X vertex rows that are corner-shared duplicates
    rather than canonical sites, return -1 -- callers fold those onto their
    canonical representative first (``defects.canonicalize_key``).
    """
    role, i, j, k, vertex = (
        key.role,
        int(key.i),
        int(key.j),
        int(key.k),
        int(key.vertex),
    )
    nx, ny, nz = (int(value) for value in grid_shape)
    count_x, count_y, count_z = a_site_grid_counts(grid_shape, periodic)
    n_a = count_x * count_y * count_z
    n_b = nx * ny * nz

    if role == "A":
        if not (0 <= i < count_x and 0 <= j < count_y and 0 <= k < count_z):
            return -1
        return (i * count_y + j) * count_z + k
    if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
        return -1
    if role == "B":
        return n_a + (i * ny + j) * nz + k
    if role != "X":
        return -1
    if vertex in _KEPT_VERTEX_ROWS:
        return n_a + n_b + 3 * ((i * ny + j) * nz + k) + vertex // 2
    px, py, pz = periodic_axes(periodic)
    base = n_a + n_b + 3 * n_b
    if vertex == 1:
        return base + j * nz + k if (i == 0 and not px) else -1
    if not px:
        base += ny * nz
    if vertex == 3:
        return base + i * nz + k if (j == 0 and not py) else -1
    if not py:
        base += nx * nz
    if vertex == 5:
        return base + i * ny + j if (k == 0 and not pz) else -1
    return -1


def index_deduplicated_a_sites(
    origin: np.ndarray,
    step_x,
    step_y,
    step_z,
    nx: int,
    ny: int,
    nz: int,
    periodic,
) -> np.ndarray:
    """A-site positions in canonical order.

    Each ``step_*`` is a scalar edge length or a per-cell sequence of them (see
    ``build_perovskite``): the A plane at index ``m`` sits after the first ``m``
    cells along that axis.
    """
    planes = [
        layer_positions(step, count + 1)
        for step, count in ((step_x, nx), (step_y, ny), (step_z, nz))
    ]
    origin = np.asarray(origin, dtype=float)
    return np.array(
        [
            origin + np.array([planes[0][key.i], planes[1][key.j], planes[2][key.k]])
            for key in canonical_site_keys((nx, ny, nz), periodic)
            if key.role == "A"
        ],
        dtype=float,
    ).reshape(-1, 3)


def expand_per_cell(value, count: int, name: str = "value") -> np.ndarray:
    """``value`` as a length-``count`` float array; a scalar is broadcast."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size == 1:
        return np.full(int(count), float(array[0]))
    if array.size != int(count):
        raise ValueError(
            f"{name} must be a scalar or have one entry per cell ({count}); got {array.size}."
        )
    return array.copy()


def layer_positions(step, count: int) -> np.ndarray:
    """Cumulative offsets of ``count`` planes separated by ``step`` per cell.

    ``step`` is a scalar or one entry per cell (``count - 1`` of them); plane 0 is
    at 0 and plane ``m`` is the sum of the first ``m`` cell edges.
    """
    edges = expand_per_cell(step, max(int(count) - 1, 0), "step")
    return np.concatenate(([0.0], np.cumsum(edges)))


def index_deduplicated_x_sites(octahedra: np.ndarray, periodic) -> np.ndarray:
    x_sites = [
        np.asarray(octahedra[key.i, key.j, key.k], dtype=float)[key.vertex].copy()
        for key in canonical_site_keys(octahedra.shape, periodic)
        if key.role == "X"
    ]
    return np.asarray(x_sites, dtype=float).reshape(-1, 3)


def active_tilt_axes(tilt_system: str) -> tuple[bool, bool, bool]:
    if len(tilt_system) != 6:
        raise ValueError(f"Unexpected Glazer tilt system format: {tilt_system}")

    return (
        tilt_system[1] != "0",
        tilt_system[3] != "0",
        tilt_system[5] != "0",
    )


def active_glazer_parameter_axes(tilt_system: str) -> tuple[bool, bool, bool]:
    """Return which pseudocubic axes are the independent Glazer selectors."""
    parsed = parse_glazer_tilt_system(tilt_system)
    selectors = [False, False, False]
    seen_labels: set[int] = set()
    for axis, (label, sign) in enumerate(parsed):
        if sign == 0 or label in seen_labels:
            continue
        selectors[axis] = True
        seen_labels.add(label)
    return (selectors[0], selectors[1], selectors[2])


def parse_glazer_tilt_system(
    tilt_system: str,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if len(tilt_system) != 6:
        raise ValueError(f"Unexpected Glazer tilt system format: {tilt_system}")

    label_map = {"a": 0, "b": 1, "c": 2}
    sign_map = {"+": 1, "-": -1, "0": 0}
    parsed = []
    for axis in range(3):
        label = tilt_system[2 * axis]
        sign = tilt_system[2 * axis + 1]
        if label not in label_map:
            raise ValueError(f"Unexpected Glazer magnitude label: {label}")
        if sign not in sign_map:
            raise ValueError(f"Unexpected Glazer tilt sign: {sign}")
        parsed.append((label_map[label], sign_map[sign]))
    return tuple(parsed)  # type: ignore[return-value]


def format_glazer_tilt_system(parsed) -> str:
    """The Glazer string for a parsed ``((label, sign), ...)`` triple.

    Magnitude labels are re-lettered by the axis a magnitude first appears on,
    so a permutation of ``a+b-b-`` that puts the in-phase axis last reads
    ``a-a-c+`` rather than an arbitrary relettering.
    """
    letters: dict[int, str] = {}
    parts = []
    sign_chars = {1: "+", -1: "-", 0: "0"}
    for axis, (label, sign) in enumerate(parsed):
        if sign == 0:
            parts.append("a0")
            continue
        if label not in letters:
            letters[label] = "abc"[axis]
        parts.append(f"{letters[label]}{sign_chars[int(sign)]}")
    return "".join(parts)


def glazer_tilt_orientations(tilt_systems) -> list[str]:
    """Every distinct axis permutation of ``tilt_systems``, canonical spellings first.

    Glazer's 23 systems are written with a conventional axis order (the
    in-phase axis first, say), but a real cell tilts about whichever axes it
    likes: a Pnma perovskite with its long axis along c is ``a-a-c+``, which is
    ``a+b-b-`` turned round. Fitting a tilt system to a relaxed structure has
    to consider those orientations too.
    """
    from itertools import permutations

    seen: list[str] = []
    for tilt_system in tilt_systems:
        parsed = parse_glazer_tilt_system(tilt_system)
        if tilt_system not in seen:
            seen.append(tilt_system)
        for order in permutations(range(3)):
            candidate = format_glazer_tilt_system([parsed[index] for index in order])
            if candidate not in seen:
                seen.append(candidate)
    return seen


def canonicalize_glazer_tilt_angles_deg(
    tilt_system: str,
    tilt_angle_x_deg: float,
    tilt_angle_y_deg: float,
    tilt_angle_z_deg: float,
) -> tuple[float, float, float]:
    """Apply Glazer constraints to per-axis tilt angles.

    - axes marked with ``0`` are forced to ``0.0``
    - repeated magnitude labels share the same angle
    """
    parsed = parse_glazer_tilt_system(tilt_system)
    axis_angles_deg = np.array(
        [tilt_angle_x_deg, tilt_angle_y_deg, tilt_angle_z_deg],
        dtype=float,
    )

    canonical = np.zeros(3, dtype=float)
    label_to_selector_angle: dict[int, float] = {}
    for axis, (label, sign) in enumerate(parsed):
        if sign == 0:
            continue
        if label not in label_to_selector_angle:
            label_to_selector_angle[label] = float(axis_angles_deg[axis])
        canonical[axis] = label_to_selector_angle[label]

    return (
        float(canonical[0]),
        float(canonical[1]),
        float(canonical[2]),
    )


def glazer_active_parameter_labels(tilt_system: str) -> tuple[int, ...]:
    """Return the unique active Glazer magnitude labels in axis order."""
    parsed = parse_glazer_tilt_system(tilt_system)
    labels: list[int] = []
    seen: set[int] = set()
    for label, sign in parsed:
        if sign == 0 or label in seen:
            continue
        labels.append(label)
        seen.add(label)
    return tuple(labels)


def glazer_phi_from_degrees(
    tilt_system: str,
    tilt_angle_x_deg: float,
    tilt_angle_y_deg: float,
    tilt_angle_z_deg: float,
) -> np.ndarray:
    tilt = parse_glazer_tilt_system(tilt_system)
    constrained_angles_deg = canonicalize_glazer_tilt_angles_deg(
        tilt_system,
        tilt_angle_x_deg,
        tilt_angle_y_deg,
        tilt_angle_z_deg,
    )
    axis_angles_rad = np.deg2rad(np.array(constrained_angles_deg, dtype=float))

    phi = np.zeros(3, dtype=float)
    counts = np.zeros(3, dtype=int)
    for axis, (label, sign) in enumerate(tilt):
        if sign == 0:
            continue
        phi[label] += axis_angles_rad[axis]
        counts[label] += 1

    for label in range(3):
        if counts[label] > 0:
            phi[label] /= counts[label]

    return phi


def octahedron_triangle_vertices(
    octahedra: np.ndarray,
    skip_cells: set | None = None,
) -> np.ndarray:
    """Triangle soup for the octahedral cages.

    ``skip_cells`` names grid cells whose B centre was removed by a vacancy;
    those cages are not drawn, since there is no octahedron without its centre.
    A cage that has merely lost a *vertex* is still drawn -- a five-coordinate
    site reads more clearly as a defect than a missing cage would.
    """
    face_rows = np.array(
        [
            4, 0, 2,
            4, 2, 1,
            4, 1, 3,
            4, 3, 0,
            5, 2, 0,
            5, 1, 2,
            5, 3, 1,
            5, 0, 3,
        ],
        dtype=int,
    )

    skip = skip_cells or set()
    flattened_octahedra = [
        np.asarray(octahedra[index], dtype=float)
        for index in np.ndindex(octahedra.shape)
        if tuple(int(value) for value in index) not in skip
    ]
    if not flattened_octahedra:
        return np.empty((0, 3), dtype=float)

    triangles = np.empty((len(flattened_octahedra) * len(face_rows), 3), dtype=float)
    for octahedron_index, vertices in enumerate(flattened_octahedra):
        if vertices.shape != (6, 3):
            raise ValueError("Each octahedron must be a (6, 3) array.")
        start = octahedron_index * len(face_rows)
        stop = start + len(face_rows)
        triangles[start:stop] = vertices[face_rows]
    return triangles


def build_perovskite(
    center: np.ndarray,
    n_oct_x: int = 0,
    n_oct_y: int = 0,
    n_oct_z: int = 0,
    center_to_vertex_distance_x=0.5,
    center_to_vertex_distance_y=0.5,
    center_to_vertex_distance_z=0.5,
    tilt_system: str = "a0a0a0",
    tilt_angle_x_deg: float = 0.0,
    tilt_angle_y_deg: float = 0.0,
    tilt_angle_z_deg: float = 0.0,
    periodic=False,
) -> PerovskiteBuild:
    """Lay out a corner-sharing octahedral grid and apply Glazer tilts.

    ``center`` is the centre of octahedron (0, 0, 0). Each
    ``center_to_vertex_distance_*`` is a scalar, or one value per octahedron
    along that axis (``n_oct_* + 1`` of them): that is how a multi-domain build
    gives each domain its own spacing along the stacking direction while the
    corner-shared vertices still coincide exactly. ``periodic`` is a bool or a
    per-axis triple; a finite axis gains the closing A plane and X face.
    """
    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise ValueError("center must be a length-3 coordinate.")

    if min(n_oct_x, n_oct_y, n_oct_z) < 0:
        raise ValueError("n_oct_x, n_oct_y, and n_oct_z must be non-negative.")
    periodic = periodic_axes(periodic)

    (
        tilt_angle_x_deg,
        tilt_angle_y_deg,
        tilt_angle_z_deg,
    ) = canonicalize_glazer_tilt_angles_deg(
        tilt_system,
        tilt_angle_x_deg,
        tilt_angle_y_deg,
        tilt_angle_z_deg,
    )
    tilt = parse_glazer_tilt_system(tilt_system)
    phi = glazer_phi_from_degrees(
        tilt_system,
        tilt_angle_x_deg=tilt_angle_x_deg,
        tilt_angle_y_deg=tilt_angle_y_deg,
        tilt_angle_z_deg=tilt_angle_z_deg,
    )

    nx, ny, nz = n_oct_x + 1, n_oct_y + 1, n_oct_z + 1
    half_x = expand_per_cell(center_to_vertex_distance_x, nx, "center_to_vertex_distance_x")
    half_y = expand_per_cell(center_to_vertex_distance_y, ny, "center_to_vertex_distance_y")
    half_z = expand_per_cell(center_to_vertex_distance_z, nz, "center_to_vertex_distance_z")
    step_x, step_y, step_z = 2.0 * half_x, 2.0 * half_y, 2.0 * half_z
    # Direction-only for the tilt rotations; the magnitudes are per cell.
    lattice_vectors = np.diag([step_x[0], step_y[0], step_z[0]])
    supercell_vectors = np.diag([step_x.sum(), step_y.sum(), step_z.sum()])

    # Octahedron m along an axis sits after the first m cell edges, measured
    # from the low face of cell 0.
    origin_x = layer_positions(step_x, nx + 1)
    origin_y = layer_positions(step_y, ny + 1)
    origin_z = layer_positions(step_z, nz + 1)
    cell_origin = center - np.array([half_x[0], half_y[0], half_z[0]], dtype=float)

    centers_grid = np.empty((nx, ny, nz, 3), dtype=float)
    octahedra = np.empty((nx, ny, nz), dtype=object)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                oct_center = cell_origin + np.array(
                    [origin_x[i] + half_x[i], origin_y[j] + half_y[j], origin_z[k] + half_z[k]],
                    dtype=float,
                )
                vertex_offsets = np.array(
                    [
                        [half_x[i], 0.0, 0.0],
                        [-half_x[i], 0.0, 0.0],
                        [0.0, half_y[j], 0.0],
                        [0.0, -half_y[j], 0.0],
                        [0.0, 0.0, half_z[k]],
                        [0.0, 0.0, -half_z[k]],
                    ],
                    dtype=float,
                )
                centers_grid[i, j, k] = oct_center
                octahedra[i, j, k] = oct_center + vertex_offsets

    tilted_octahedra = apply_glazer_tilts(
        octahedra,
        tilt=tilt,
        phi=phi,
        lattice_vectors=lattice_vectors,
        collapse=True,
        periodic=periodic,
        supercell_vectors=supercell_vectors,
    )

    centers_B = centers_grid.reshape(-1, 3)
    vertices_A = index_deduplicated_a_sites(
        origin=cell_origin,
        step_x=step_x,
        step_y=step_y,
        step_z=step_z,
        nx=nx,
        ny=ny,
        nz=nz,
        periodic=periodic,
    )
    vertices_X = index_deduplicated_x_sites(tilted_octahedra, periodic=periodic)

    all_sites = np.vstack((vertices_A, centers_B, vertices_X))
    a_site_indices = np.arange(len(vertices_A), dtype=int)
    b_site_indices = np.arange(len(vertices_A), len(vertices_A) + len(centers_B), dtype=int)
    x_site_indices = np.arange(
        len(vertices_A) + len(centers_B),
        len(all_sites),
        dtype=int,
    )

    return PerovskiteBuild(
        all_sites=all_sites,
        a_site_indices=a_site_indices,
        b_site_indices=b_site_indices,
        x_site_indices=x_site_indices,
        a_sites=vertices_A,
        b_sites=centers_B,
        x_sites=vertices_X,
        octahedra=tilted_octahedra,
    )
