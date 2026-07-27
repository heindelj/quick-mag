from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from quick_mag.perovskite_builder import (
    canonicalize_glazer_tilt_angles_deg,
    glazer_active_parameter_labels,
    parse_glazer_tilt_system,
)

def cart_to_frac(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert Cartesian positions to fractional coordinates."""
    return positions @ np.linalg.inv(cell)


def frac_to_cart(frac: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert fractional coordinates to Cartesian coordinates."""
    return frac @ cell


def min_image_frac_delta(df: np.ndarray) -> np.ndarray:
    """Wrap fractional deltas into [-0.5, 0.5)."""
    return df - np.round(df)


def min_image_cart_distances(
    pos_a: np.ndarray,
    pos_b: np.ndarray,
    cell: np.ndarray,
) -> np.ndarray:
    """Pairwise Cartesian distances under the minimum-image convention."""
    fa = cart_to_frac(pos_a, cell)
    fb = cart_to_frac(pos_b, cell)
    df = fa[:, None, :] - fb[None, :, :]
    df = min_image_frac_delta(df)
    dc = df @ cell
    return np.linalg.norm(dc, axis=-1)


def estimate_origin_shift(
    frac_positions: np.ndarray,
    n: tuple[int, int, int],
    n_bins: int = 32,
) -> np.ndarray:
    """Estimate the fractional origin shift for a grid of shape `n`."""
    nx, ny, nz = n
    folded = (frac_positions * np.array([nx, ny, nz])) % 1.0
    histogram, edges = np.histogramdd(
        folded,
        bins=(n_bins, n_bins, n_bins),
        range=((0, 1), (0, 1), (0, 1)),
    )
    peak_idx = np.unravel_index(np.argmax(histogram), histogram.shape)
    centers = [(edges[d][:-1] + edges[d][1:]) / 2.0 for d in range(3)]
    peak_uvw = np.array([centers[d][peak_idx[d]] for d in range(3)])
    return peak_uvw / np.array([nx, ny, nz])


def padded_hungarian_match(
    cost: np.ndarray,
    defect_penalty: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve a defect-tolerant assignment problem."""
    n_grid, n_candidates = cost.shape
    padded_size = n_grid + n_candidates
    padded_cost = np.full((padded_size, padded_size), 0.0)
    padded_cost[:n_grid, :n_candidates] = cost
    padded_cost[:n_grid, n_candidates:] = defect_penalty
    padded_cost[n_grid:, :n_candidates] = defect_penalty

    row_ind, col_ind = linear_sum_assignment(padded_cost)
    grid_to_candidate = np.full(n_grid, -1, dtype=int)
    candidate_to_grid = np.full(n_candidates, -1, dtype=int)
    for row, col in zip(row_ind, col_ind):
        if row < n_grid and col < n_candidates:
            grid_to_candidate[row] = col
            candidate_to_grid[col] = row
    total_cost = padded_cost[row_ind, col_ind].sum()
    return grid_to_candidate, candidate_to_grid, float(total_cost)


def candidate_supercell_dims(
    cell: np.ndarray,
    d_bb_min: float = 3.4,
    d_bb_max: float = 4.6,
) -> list[tuple[int, int, int]]:
    """Enumerate plausible B-site supercell dimensions."""
    lengths = np.linalg.norm(cell, axis=1)
    per_axis = []
    for length in lengths:
        n_lo = max(1, int(np.floor(length / d_bb_max)))
        n_hi = max(n_lo, int(np.ceil(length / d_bb_min)))
        per_axis.append(list(range(n_lo, n_hi + 1)))
    return [(a, b, c) for a in per_axis[0] for b in per_axis[1] for c in per_axis[2]]


@dataclass
class BSiteMatch:
    n: tuple[int, int, int]
    origin_cart: np.ndarray
    grid_frac: np.ndarray
    grid_cart: np.ndarray
    matched_target_indices: np.ndarray
    grid_to_target: np.ndarray
    rmsd: float
    coverage: float
    score: float


def _build_grid(n: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = n
    ii, jj, kk = np.meshgrid(
        np.arange(nx),
        np.arange(ny),
        np.arange(nz),
        indexing="ij",
    )
    return np.stack([ii / nx, jj / ny, kk / nz], axis=-1).reshape(-1, 3)


def match_b_sites(
    cell: np.ndarray,
    positions: np.ndarray,
    d_bb_min: float = 3.4,
    d_bb_max: float = 4.6,
    defect_distance: float = 1.2,
    refine_iters: int = 2,
    histogram_bins: int = 32,
) -> BSiteMatch:
    """Identify the B sublattice in a target perovskite."""
    cell = np.asarray(cell, dtype=float)
    positions = np.asarray(positions, dtype=float)
    frac = cart_to_frac(positions, cell) % 1.0
    penalty = defect_distance**2
    best: BSiteMatch | None = None

    for n in candidate_supercell_dims(cell, d_bb_min, d_bb_max):
        shift = estimate_origin_shift(frac, n, n_bins=histogram_bins)
        grid_frac_base = _build_grid(n)

        for _ in range(refine_iters + 1):
            grid_frac = (grid_frac_base + shift) % 1.0
            grid_cart = frac_to_cart(grid_frac, cell)
            dist = min_image_cart_distances(grid_cart, positions, cell)
            cost = dist**2
            grid_to_candidate, _, _ = padded_hungarian_match(cost, penalty)
            matched_mask = grid_to_candidate >= 0
            if not matched_mask.any():
                break
            target_frac_matched = frac[grid_to_candidate[matched_mask]]
            base_matched = grid_frac_base[matched_mask]
            df = min_image_frac_delta(target_frac_matched - base_matched)
            updated_shift = (base_matched + df).mean(axis=0) - base_matched.mean(axis=0)
            updated_shift = updated_shift % 1.0
            if np.allclose(updated_shift, shift, atol=1e-5):
                shift = updated_shift
                break
            shift = updated_shift

        grid_frac = (grid_frac_base + shift) % 1.0
        grid_cart = frac_to_cart(grid_frac, cell)
        dist = min_image_cart_distances(grid_cart, positions, cell)
        cost = dist**2
        grid_to_candidate, _, total_cost = padded_hungarian_match(cost, penalty)
        matched_mask = grid_to_candidate >= 0
        if not matched_mask.any():
            continue
        matched_dists = dist[matched_mask, grid_to_candidate[matched_mask]]
        rmsd = float(np.sqrt((matched_dists**2).mean()))
        coverage = matched_mask.sum() / len(grid_to_candidate)
        origin_cart = frac_to_cart(np.array([shift]), cell)[0]

        candidate = BSiteMatch(
            n=n,
            origin_cart=origin_cart,
            grid_frac=grid_frac,
            grid_cart=grid_cart,
            matched_target_indices=grid_to_candidate[matched_mask],
            grid_to_target=grid_to_candidate,
            rmsd=rmsd,
            coverage=coverage,
            score=-float(total_cost),
        )
        if best is None or candidate.score > best.score:
            best = candidate

    if best is None:
        raise RuntimeError("No (nx, ny, nz) candidate produced any matches.")
    return best


def identify_x_candidates(
    cell: np.ndarray,
    positions: np.ndarray,
    b_match: BSiteMatch,
    excluded_indices: np.ndarray | None = None,
    midpoint_tolerance: float = 0.5,
) -> np.ndarray:
    """Return target-atom indices that lie close to B-B midpoints."""
    if excluded_indices is None:
        excluded_indices = b_match.matched_target_indices

    excluded = set(int(index) for index in excluded_indices)
    nx, ny, nz = b_match.n
    grid_frac = b_match.grid_frac % 1.0
    midpoint_offsets = np.array(
        [
            [0.5 / nx, 0.0, 0.0],
            [0.0, 0.5 / ny, 0.0],
            [0.0, 0.0, 0.5 / nz],
        ],
        dtype=float,
    )
    # Build the expected periodic X-site manifold explicitly from the B grid.
    # This avoids the ambiguous minimum-image midpoint when an axis has only two
    # B layers, where q - p = 0.5 can collapse one whole oxygen family.
    midpoints_frac = np.concatenate(
        [(grid_frac + offset) % 1.0 for offset in midpoint_offsets],
        axis=0,
    )
    midpoints_cart = frac_to_cart(midpoints_frac, cell)

    candidate_indices = np.array(
        [index for index in range(len(positions)) if index not in excluded],
        dtype=int,
    )
    if len(candidate_indices) == 0:
        return np.array([], dtype=int)

    candidate_positions = positions[candidate_indices]
    step_lengths = np.linalg.norm(cell, axis=1) / np.array([nx, ny, nz], dtype=float)
    defect_distance = max(float(midpoint_tolerance), 0.2 * float(np.min(step_lengths)))
    dists = min_image_cart_distances(midpoints_cart, candidate_positions, cell)
    midpoint_to_candidate, _, _ = padded_hungarian_match(dists**2, defect_distance**2)
    matched_mask = midpoint_to_candidate >= 0
    return np.sort(candidate_indices[midpoint_to_candidate[matched_mask]])


def kabsch_rotation(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Find the rotation minimizing ||P @ R - Q||_F."""
    hessian = P.T @ Q
    U, _, Vt = np.linalg.svd(hessian)
    sign = np.sign(np.linalg.det(Vt.T @ U.T))
    correction = np.eye(3)
    correction[2, 2] = sign
    return U @ correction @ Vt


def rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """Return a rotation vector in axis-angle form."""
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3)
    axis = np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]],
        dtype=float,
    )
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1.0))
        return np.real(eigvecs[:, idx]) * theta
    return axis / norm * theta


def extract_octahedron_rotations(
    cell: np.ndarray,
    positions: np.ndarray,
    b_match: BSiteMatch,
    x_candidate_indices: np.ndarray,
    ideal_half_steps: tuple[float, float, float],
) -> np.ndarray:
    """Extract an approximate rotation vector for each BO6 octahedron."""
    half_x, half_y, half_z = ideal_half_steps
    ideal_offsets = np.array(
        [
            [half_x, 0.0, 0.0],
            [-half_x, 0.0, 0.0],
            [0.0, half_y, 0.0],
            [0.0, -half_y, 0.0],
            [0.0, 0.0, half_z],
            [0.0, 0.0, -half_z],
        ],
        dtype=float,
    )

    b_positions = b_match.grid_cart
    x_positions = positions[x_candidate_indices]
    rot_vecs = np.full((len(b_positions), 3), np.nan, dtype=float)

    dists = min_image_cart_distances(b_positions, x_positions, cell)
    b_frac = cart_to_frac(b_positions, cell)
    x_frac = cart_to_frac(x_positions, cell)
    delta_frac = min_image_frac_delta(x_frac[None, :, :] - b_frac[:, None, :])
    delta_cart = delta_frac @ cell

    for b_index in range(len(b_positions)):
        if dists.shape[1] < 6:
            continue
        nearest = np.argsort(dists[b_index])[:6]
        actual_offsets = delta_cart[b_index, nearest]
        cost = np.linalg.norm(
            ideal_offsets[:, None, :] - actual_offsets[None, :, :],
            axis=-1,
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        ordered_actual = actual_offsets[col_ind[np.argsort(row_ind)]]
        rot_vecs[b_index] = rotation_matrix_to_axis_angle(
            kabsch_rotation(ideal_offsets, ordered_actual)
        )

    return rot_vecs

def glazer_effective_parameter_values(
    tilt_system: str,
    tilt_angles_deg: tuple[float, float, float],
) -> tuple[float, ...]:
    """Return one effective signed angle per unique active Glazer label."""
    parsed = parse_glazer_tilt_system(tilt_system)
    axis_angles = canonicalize_glazer_tilt_angles_deg(
        tilt_system,
        tilt_angles_deg[0],
        tilt_angles_deg[1],
        tilt_angles_deg[2],
    )
    values: list[float] = []
    seen_labels: set[int] = set()
    for axis, (label, sign) in enumerate(parsed):
        if sign == 0 or label in seen_labels:
            continue
        values.append(float(axis_angles[axis]))
        seen_labels.add(label)
    return tuple(values)


def is_valid_tilt_solution(
    tilt_system: str,
    tilt_angles_deg: tuple[float, float, float],
    *,
    zero_tol_deg: float = 1e-6,
) -> bool:
    """Reject non-cubic Glazer classes whose active shared angles collapse to zero."""
    active_values = glazer_effective_parameter_values(tilt_system, tilt_angles_deg)
    if not active_values:
        return True
    return all(abs(value) > zero_tol_deg for value in active_values)


def refine_tilts(
    cell: np.ndarray,
    target_positions: np.ndarray,
    b_match: BSiteMatch,
    x_candidate_indices: np.ndarray,
    builder: Callable,
    tilt_system: str,
    initial_angles_deg: tuple[float, ...] = (0.5, 5.0),
    defect_distance: float = 1.0,
    half_steps: tuple[float, float, float] | None = None,
    angle_bound_deg: float = 15.0,
) -> dict:
    """Refine the tilt angles for a given Glazer tilt system.

    Returns both an optimization RMSD (used by the optimizer) and a 
    matched-only RMSD (used for reporting and for tie-breaking).
    """
    nx, ny, nz = b_match.n
    if half_steps is None:
        idx_origin = 0
        idx_x1 = ny * nz
        idx_y1 = nz
        idx_z1 = 1
        half_x = min_image_cart_distances(
            b_match.grid_cart[[idx_origin]],
            b_match.grid_cart[[idx_x1]],
            cell,
        )[0, 0] / 2.0
        half_y = min_image_cart_distances(
            b_match.grid_cart[[idx_origin]],
            b_match.grid_cart[[idx_y1]],
            cell,
        )[0, 0] / 2.0
        half_z = min_image_cart_distances(
            b_match.grid_cart[[idx_origin]],
            b_match.grid_cart[[idx_z1]],
            cell,
        )[0, 0] / 2.0
        half_steps = (half_x, half_y, half_z)

    half_x, half_y, half_z = half_steps
    x_target = target_positions[x_candidate_indices]
    builder_center = b_match.grid_cart[0]
    parsed_tilt = parse_glazer_tilt_system(tilt_system)
    active_labels = glazer_active_parameter_labels(tilt_system)
    n_active = len(active_labels)
    label_to_parameter_index = {
        label: parameter_index
        for parameter_index, label in enumerate(active_labels)
    }

    def expand_angles(label_angles_deg: np.ndarray) -> tuple[float, float, float]:
        full = np.zeros(3, dtype=float)
        for axis_index, (label, sign) in enumerate(parsed_tilt):
            if sign == 0:
                continue
            full[axis_index] = float(label_angles_deg[label_to_parameter_index[label]])
        return canonicalize_glazer_tilt_angles_deg(
            tilt_system,
            float(full[0]),
            float(full[1]),
            float(full[2]),
        )

    def predicted_x_sites(free_angles: np.ndarray) -> np.ndarray:
        ax, ay, az = expand_angles(free_angles)
        build = builder(
            center=builder_center,
            n_oct_x=nx - 1,
            n_oct_y=ny - 1,
            n_oct_z=nz - 1,
            center_to_vertex_distance_x=half_x,
            center_to_vertex_distance_y=half_y,
            center_to_vertex_distance_z=half_z,
            tilt_system=tilt_system,
            tilt_angle_x_deg=ax,
            tilt_angle_y_deg=ay,
            tilt_angle_z_deg=az,
            periodic=True,
        )
        return np.asarray(build.x_sites, dtype=float)

    def assignment_distances(
        free_angles: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        predicted = predicted_x_sites(free_angles)
        if len(x_target) == 0:
            return (
                np.full(len(predicted), defect_distance, dtype=float),
                np.zeros(len(predicted), dtype=bool),
            )
        dists = min_image_cart_distances(predicted, x_target, cell)
        predicted_to_target, _, _ = padded_hungarian_match(dists**2, defect_distance**2)
        matched_mask = predicted_to_target >= 0
        assigned = np.full(len(predicted), defect_distance, dtype=float)
        if matched_mask.any():
            rows = np.flatnonzero(matched_mask)
            cols = predicted_to_target[matched_mask]
            assigned[rows] = dists[rows, cols]
        return assigned, matched_mask

    def residuals(free_angles: np.ndarray) -> np.ndarray:
        nearest, _ = assignment_distances(free_angles)
        # Smooth cap for stable optimization.
        return nearest / np.sqrt(1.0 + (nearest / defect_distance) ** 2)

    def matched_rmsd(
        free_angles: np.ndarray,
    ) -> tuple[float, float, float]:
        """Return (rmsd_full, rmsd_matched, matched_fraction)."""
        nearest, matched_mask = assignment_distances(free_angles)
        rmsd_full = float(np.sqrt(np.mean(nearest**2)))
        if matched_mask.any():
            rmsd_matched = float(np.sqrt(np.mean(nearest[matched_mask] ** 2)))
        else:
            rmsd_matched = float("inf")
        matched_fraction = float(np.mean(matched_mask))
        return rmsd_full, rmsd_matched, matched_fraction

    if n_active == 0:
        rmsd_full, rmsd_matched, matched_fraction = matched_rmsd(
            np.array([], dtype=float)
        )
        canonical_angles = canonicalize_glazer_tilt_angles_deg(
            tilt_system,
            0.0,
            0.0,
            0.0,
        )
        return {
            "tilt_system": tilt_system,
            "angles_deg": canonical_angles,
            "rmsd": rmsd_full,
            "rmsd_matched": rmsd_matched,
            "matched_fraction": matched_fraction,
            "n_active": 0,
            "scipy_result": None,
        }

    best_payload = None
    best_key = (float("inf"), float("inf"), float("inf"))
    bounds = (
        -angle_bound_deg * np.ones(n_active),
        angle_bound_deg * np.ones(n_active),
    )

    for init in initial_angles_deg:
        x0 = np.full(n_active, init, dtype=float)
        try:
            result = least_squares(
                residuals, x0,
                method="trf",
                bounds=bounds,
                max_nfev=200,
            )
        except Exception:
            continue
        rmsd_full, rmsd_matched, matched_fraction = matched_rmsd(result.x)
        key = (-matched_fraction, rmsd_matched, rmsd_full)
        if key < best_key:
            best_key = key
            best_payload = (result, rmsd_full, rmsd_matched, matched_fraction)

    if best_payload is None:
        raise RuntimeError(f"All multi-starts failed for {tilt_system}")

    result, rmsd_full, rmsd_matched, matched_fraction = best_payload
    canonical_angles = expand_angles(result.x)
    return {
        "tilt_system": tilt_system,
        "angles_deg": canonical_angles,
        "rmsd": rmsd_full,
        "rmsd_matched": rmsd_matched,
        "matched_fraction": matched_fraction,
        "n_active": n_active,
        "scipy_result": result,
    }


ALL_GLAZER_SYSTEMS = [
    "a0a0a0",
    "a0a0c+",
    "a0a0c-",
    "a0b+c+",
    "a0b+b+",
    "a0b+c-",
    "a0b+b-",
    "a0b-c-",
    "a0b-b-",
    "a+b+c+",
    "a+b+b+",
    "a+a+a+",
    "a+b+c-",
    "a+a+c-",
    "a+b+b-",
    "a+a+a-",
    "a+b-c-",
    "a+a-c-",
    "a+b-b-",
    "a+a-a-",
    "a-b-c-",
    "a-b-b-",
    "a-a-a-",
]


@dataclass
class XSiteFitResult:
    tilt_system: str
    angles_deg: tuple[float, float, float]
    rmsd: float
    matched_fraction: float
    x_candidate_indices: np.ndarray
    near_zero_solutions: list[dict]
    all_attempts: list[dict]


def fit_x_sites(
    cell: np.ndarray,
    target_positions: np.ndarray,
    b_match: BSiteMatch,
    builder: Callable,
    midpoint_tolerance: float = 0.5,
    defect_distance: float = 1.0,
    allowed_systems: list[str] | None = None,
    rmsd_threshold: float = 0.01,
    tie_tolerance: float = 0.02,
    initial_angles_deg: tuple[float, ...] = (-5.0, -0.5, 0.5, 5.0),
    near_zero_rmsd_threshold: float = 1e-5,
    zero_tilt_tol_deg: float = 1e-6,
) -> XSiteFitResult:
    """Sweep tilt systems; prefer the lowest valid RMSD solution."""
    if allowed_systems is None:
        allowed_systems = ALL_GLAZER_SYSTEMS

    x_indices = identify_x_candidates(
        cell, target_positions, b_match, midpoint_tolerance=midpoint_tolerance
    )
    if len(x_indices) == 0:
        raise RuntimeError("No X candidates found; check midpoint_tolerance.")

    nx, ny, nz = b_match.n
    idx_origin = 0
    idx_x1 = ny * nz
    idx_y1 = nz
    idx_z1 = 1
    half_steps = (
        min_image_cart_distances(
            b_match.grid_cart[[idx_origin]], b_match.grid_cart[[idx_x1]], cell
        )[0, 0] / 2.0,
        min_image_cart_distances(
            b_match.grid_cart[[idx_origin]], b_match.grid_cart[[idx_y1]], cell
        )[0, 0] / 2.0,
        min_image_cart_distances(
            b_match.grid_cart[[idx_origin]], b_match.grid_cart[[idx_z1]], cell
        )[0, 0] / 2.0,
    )

    attempts: list[dict] = []
    best: dict | None = None
    coverage_tolerance = 0.01
    for tilt_system in allowed_systems:
        try:
            result = refine_tilts(
                cell, target_positions, b_match, x_indices,
                builder, tilt_system,
                initial_angles_deg=initial_angles_deg,
                defect_distance=defect_distance,
                half_steps=half_steps,
            )
        except Exception as exc:
            attempts.append({"tilt_system": tilt_system, "error": str(exc)})
            continue
        result["valid_classification"] = is_valid_tilt_solution(
            tilt_system,
            result["angles_deg"],
            zero_tol_deg=zero_tilt_tol_deg,
        )
        result["constraint_rank"] = len(glazer_active_parameter_labels(tilt_system))
        attempts.append(result)
        if not result["valid_classification"]:
            continue

        # Prefer the lowest matched RMSD first, then use coverage and model
        # simplicity only as tie-breakers.
        if best is None:
            best = result
        else:
            improvement = best["rmsd_matched"] - result["rmsd_matched"]
            if improvement > tie_tolerance:
                best = result
            elif abs(improvement) <= tie_tolerance:
                coverage_gain = result["matched_fraction"] - best["matched_fraction"]
                if coverage_gain > coverage_tolerance:
                    best = result
                elif abs(coverage_gain) <= coverage_tolerance:
                    full_improvement = best["rmsd"] - result["rmsd"]
                    if full_improvement > tie_tolerance:
                        best = result
                    elif abs(full_improvement) <= tie_tolerance:
                        if result["constraint_rank"] < best["constraint_rank"]:
                            best = result
                        elif result["constraint_rank"] == best["constraint_rank"]:
                            if result["n_active"] < best["n_active"]:
                                best = result
                            elif result["n_active"] == best["n_active"]:
                                if result["tilt_system"] < best["tilt_system"]:
                                    best = result
            elif result["rmsd_matched"] < best["rmsd_matched"]:
                best = result

    if best is None:
        raise RuntimeError("All tilt-system refinements failed.")

    candidate_solutions = [
        attempt
        for attempt in attempts
        if "error" not in attempt
        and attempt["valid_classification"]
        and attempt["matched_fraction"] >= best["matched_fraction"] - coverage_tolerance
    ]

    near_zero_solutions = [
        attempt
        for attempt in candidate_solutions
        if attempt["rmsd_matched"] <= near_zero_rmsd_threshold
    ]
    near_zero_solutions.sort(
        key=lambda attempt: (
            attempt["rmsd_matched"],
            attempt["rmsd"],
            attempt["constraint_rank"],
            attempt["n_active"],
            attempt["tilt_system"],
        )
    )

    selected = best
    if near_zero_solutions:
        selected = near_zero_solutions[0]

    return XSiteFitResult(
        tilt_system=selected["tilt_system"],
        angles_deg=selected["angles_deg"],
        rmsd=selected["rmsd_matched"],   # report matched-only for interpretability
        matched_fraction=selected["matched_fraction"],
        x_candidate_indices=x_indices,
        near_zero_solutions=near_zero_solutions,
        all_attempts=attempts,
    )
