from __future__ import annotations

from dataclasses import dataclass

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


def index_deduplicated_a_sites(
    origin: np.ndarray,
    step_x: float,
    step_y: float,
    step_z: float,
    nx: int,
    ny: int,
    nz: int,
    periodic: bool,
) -> np.ndarray:
    count_x = nx if periodic else nx + 1
    count_y = ny if periodic else ny + 1
    count_z = nz if periodic else nz + 1
    a_sites = np.empty((count_x * count_y * count_z, 3), dtype=float)

    cursor = 0
    for i in range(count_x):
        for j in range(count_y):
            for k in range(count_z):
                a_sites[cursor] = origin + np.array(
                    [i * step_x, j * step_y, k * step_z],
                    dtype=float,
                )
                cursor += 1
    return a_sites


def index_deduplicated_x_sites(octahedra: np.ndarray, periodic: bool) -> np.ndarray:
    nx, ny, nz = octahedra.shape
    x_sites: list[np.ndarray] = []

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                vertices = np.asarray(octahedra[i, j, k], dtype=float)
                x_sites.append(vertices[0].copy())
                x_sites.append(vertices[2].copy())
                x_sites.append(vertices[4].copy())

    if not periodic:
        for j in range(ny):
            for k in range(nz):
                x_sites.append(np.asarray(octahedra[0, j, k], dtype=float)[1].copy())
        for i in range(nx):
            for k in range(nz):
                x_sites.append(np.asarray(octahedra[i, 0, k], dtype=float)[3].copy())
        for i in range(nx):
            for j in range(ny):
                x_sites.append(np.asarray(octahedra[i, j, 0], dtype=float)[5].copy())

    return np.asarray(x_sites, dtype=float)


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


def octahedron_triangle_vertices(octahedra: np.ndarray) -> np.ndarray:
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

    flattened_octahedra = [
        np.asarray(octahedra[index], dtype=float) for index in np.ndindex(octahedra.shape)
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
    center_to_vertex_distance_x: float = 0.5,
    center_to_vertex_distance_y: float = 0.5,
    center_to_vertex_distance_z: float = 0.5,
    tilt_system: str = "a0a0a0",
    tilt_angle_x_deg: float = 0.0,
    tilt_angle_y_deg: float = 0.0,
    tilt_angle_z_deg: float = 0.0,
    periodic: bool = False,
) -> PerovskiteBuild:
    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise ValueError("center must be a length-3 coordinate.")

    if min(n_oct_x, n_oct_y, n_oct_z) < 0:
        raise ValueError("n_oct_x, n_oct_y, and n_oct_z must be non-negative.")

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

    step_x = 2.0 * center_to_vertex_distance_x
    step_y = 2.0 * center_to_vertex_distance_y
    step_z = 2.0 * center_to_vertex_distance_z
    lattice_vectors = np.array(
        [
            [step_x, 0.0, 0.0],
            [0.0, step_y, 0.0],
            [0.0, 0.0, step_z],
        ],
        dtype=float,
    )

    nx, ny, nz = n_oct_x + 1, n_oct_y + 1, n_oct_z + 1
    centers_grid = np.empty((nx, ny, nz, 3), dtype=float)
    octahedra = np.empty((nx, ny, nz), dtype=object)

    vertex_offsets = np.array(
        [
            [center_to_vertex_distance_x, 0.0, 0.0],
            [-center_to_vertex_distance_x, 0.0, 0.0],
            [0.0, center_to_vertex_distance_y, 0.0],
            [0.0, -center_to_vertex_distance_y, 0.0],
            [0.0, 0.0, center_to_vertex_distance_z],
            [0.0, 0.0, -center_to_vertex_distance_z],
        ],
        dtype=float,
    )
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                oct_center = center + np.array(
                    [step_x * i, step_y * j, step_z * k], dtype=float
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
    )

    centers_B = centers_grid.reshape(-1, 3)
    cell_origin = center - np.array(
        [
            center_to_vertex_distance_x,
            center_to_vertex_distance_y,
            center_to_vertex_distance_z,
        ],
        dtype=float,
    )
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
