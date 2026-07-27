"""
Glazer tilt system implementation for perovskite octahedra.

Tilt system representation (tuple form):
    tilt = ((m_a, s_a), (m_b, s_b), (m_c, s_c))
where
    m_alpha : int  -- magnitude *label* (0, 1, 2). Equal labels => equal magnitudes.
                      Convention: label 0 -> 'a', 1 -> 'b', 2 -> 'c'.
    s_alpha : int  -- sign of the tilt: +1 (in-phase), -1 (anti-phase), 0 (none).

Rotation axes:
    The three pseudocubic rotation axes are the normalized lattice directions
    a_hat, b_hat, c_hat (rows of `lattice_vectors`, unit-normalized).
    Rotations are proper orthogonal rotations in Cartesian space, so
    octahedra remain rigid. For an orthogonal lattice this reduces to the
    standard rotations about Cartesian x, y, z.

Vertex storage (per octahedron):
    (6, 3) array, ordered as:
        row 0: +a vertex
        row 1: -a vertex
        row 2: +b vertex
        row 3: -b vertex
        row 4: +c vertex
        row 5: -c vertex

Supercell storage:
    `octahedra` is a 3D object array of shape (Nx, Ny, Nz). Each entry is a
    (6, 3) float array of Cartesian vertex positions.
"""

from __future__ import annotations
import numpy as np


def tilt_sign(sigma: int, axis: int, i: int, j: int, k: int) -> int:
    """Sign of the rotation about `axis` for the octahedron at cell (i,j,k)."""
    if sigma == 0:
        return 0
    if sigma == +1:
        other_sum = (i + j + k) - (i, j, k)[axis]
        return 1 if other_sum % 2 == 0 else -1
    if sigma == -1:
        return 1 if (i + j + k) % 2 == 0 else -1
    raise ValueError(f"sigma must be -1, 0, or +1; got {sigma}")


def _axis_angle_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    """Rotation matrix for angle `theta` about unit-vector `axis`."""
    if theta == 0.0:
        return np.eye(3)
    x, y, z = axis
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,    y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s,  c + z*z*C ],
    ])


def composite_rotation(axes_hat: np.ndarray,
                       theta_a: float,
                       theta_b: float,
                       theta_c: float) -> np.ndarray:
    """
    Composite rotation R = R_c(theta_c) @ R_b(theta_b) @ R_a(theta_a),
    where R_alpha rotates about the unit vector axes_hat[alpha].
    """
    Ra = _axis_angle_matrix(axes_hat[0], theta_a)
    Rb = _axis_angle_matrix(axes_hat[1], theta_b)
    Rc = _axis_angle_matrix(axes_hat[2], theta_c)
    return Rc @ Rb @ Ra


def resolve_magnitudes(tilt, phi):
    """Map magnitude labels to actual angles."""
    return tuple(phi[tilt[a][0]] for a in range(3))


def apply_glazer_tilts(octahedra: np.ndarray,
                       tilt,
                       phi,
                       lattice_vectors: np.ndarray,
                       collapse: bool = True,
                       periodic: bool = True) -> np.ndarray:
    """
    Apply a Glazer tilt system to a supercell of octahedra.

    Parameters
    ----------
    octahedra : ndarray, shape (Nx, Ny, Nz), dtype=object
        Each entry is a (6, 3) array of Cartesian vertex positions.
    tilt : ((m_a, s_a), (m_b, s_b), (m_c, s_c))
        Glazer tilt system in tuple form.
    phi : sequence or dict
        Magnitudes in radians, indexed by magnitude label.
    lattice_vectors : (3, 3) ndarray
        Single-unit-cell lattice vectors as ROWS. Their normalized directions
        define the rotation axes. May be non-orthogonal.
    collapse : bool, default True
        If True, average shared-oxygen positions across neighbors.
    periodic : bool, default True
        If True, the supercell is treated as periodic. If False, boundary
        faces have no partner and are left untouched by the collapse pass.

    Returns
    -------
    rotated : ndarray, shape (Nx, Ny, Nz), dtype=object
        New supercell. The input is not modified.
    """
    Nx, Ny, Nz = octahedra.shape
    L = np.asarray(lattice_vectors, dtype=float)
    if L.shape != (3, 3):
        raise ValueError(f"lattice_vectors must be (3,3); got {L.shape}")

    norms = np.linalg.norm(L, axis=1)
    if np.any(norms == 0):
        raise ValueError("lattice_vectors must all have non-zero length")
    axes_hat = L / norms[:, None]

    phi_a, phi_b, phi_c = resolve_magnitudes(tilt, phi)
    s_sigma = (tilt[0][1], tilt[1][1], tilt[2][1])

    rotated = np.empty_like(octahedra)
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                verts = np.asarray(octahedra[i, j, k], dtype=float)
                center = verts.mean(axis=0)

                ta = tilt_sign(s_sigma[0], 0, i, j, k) * phi_a
                tb = tilt_sign(s_sigma[1], 1, i, j, k) * phi_b
                tc = tilt_sign(s_sigma[2], 2, i, j, k) * phi_c

                R = composite_rotation(axes_hat, ta, tb, tc)
                rotated[i, j, k] = (verts - center) @ R.T + center

    if collapse:
        rotated = collapse_shared_vertices(rotated, lattice_vectors,
                                           periodic=periodic)
    return rotated


def collapse_shared_vertices(octahedra: np.ndarray,
                             lattice_vectors: np.ndarray,
                             periodic: bool = True) -> np.ndarray:
    """
    Average duplicated corner-shared oxygens across neighbors.

    Parameters
    ----------
    octahedra : (Nx, Ny, Nz) object array of (6,3) arrays.
    lattice_vectors : (3, 3) ndarray, rows = a, b, c.
    periodic : bool
        If True, neighbors wrap with a supercell translation. If False,
        boundary faces are left untouched.

    Returns
    -------
    New supercell. Input not modified.
    """
    Nx, Ny, Nz = octahedra.shape
    L = np.asarray(lattice_vectors, dtype=float)

    out = np.empty_like(octahedra)
    for idx in np.ndindex(Nx, Ny, Nz):
        out[idx] = np.array(octahedra[idx], dtype=float, copy=True)

    pairs = [(0, 1, (1, 0, 0), 0),
             (2, 3, (0, 1, 0), 1),
             (4, 5, (0, 0, 1), 2)]
    Ns = (Nx, Ny, Nz)

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                for row_pos, row_neg, (di, dj, dk), axis in pairs:
                    raws = (i + di, j + dj, k + dk)
                    wraps = raws[axis] // Ns[axis]

                    if wraps != 0 and not periodic:
                        continue

                    ni, nj, nk = raws[0] % Nx, raws[1] % Ny, raws[2] % Nz
                    shift = wraps * Ns[axis] * L[axis]

                    v_here = out[i, j, k][row_pos]
                    v_there = out[ni, nj, nk][row_neg] + shift
                    mean = 0.5 * (v_here + v_there)

                    out[i, j, k][row_pos] = mean
                    out[ni, nj, nk][row_neg] = mean - shift

    return out