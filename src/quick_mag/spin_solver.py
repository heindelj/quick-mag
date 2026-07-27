import math
import random
import numpy as np
from dataclasses import dataclass

from itertools import combinations
from typing import Optional, List, Tuple, Sequence

from quick_mag.magnetic_moments import (
    OxidationStateAssignment,
    format_oxidation_distribution,
)

NO_NONZERO_MAGNETIC_MOMENTS_MESSAGE = (
    "The assigned oxidation state has no nonzero magnetic moments, so there is nothing to do."
)

@dataclass
class SpinConfig:
    """One evaluated spin configuration."""
    energy: float
    all_moments: np.ndarray       # signed moments, shape (n_atoms,) or (n_atoms, 3)
    n_flips: int = 0              # 0 for base states
    magnetization: float = 0.0    # net signed magnetization
    n_unpaired: float = 0.0       # num unpaired moments
    ox_assignment_id: int = 0     # which oxidation-state assignment
    degeneracy: int = 1           # number of energy-degenerate spin configs collapsed into this one

def _canonical_moment_vector(moments: np.ndarray, atol: float = 1e-8) -> np.ndarray:
    """
    Canonicalize a collinear spin vector so configs related by global
    spin inversion map to the same representative.
    """
    arr = np.asarray(moments, dtype=float)
    if arr.ndim == 1:
        for value in arr:
            if abs(value) <= atol:
                continue
            return arr if value > 0 else -arr
        return arr

    if arr.ndim == 2 and arr.shape[1] == 3:
        for vec in arr:
            if np.linalg.norm(vec) <= atol:
                continue
            for component in vec:
                if abs(component) <= atol:
                    continue
                return arr if component > 0 else -arr
        return arr

    raise ValueError(f"Unsupported moment array shape for canonicalization: {arr.shape}")


def _canonical_moment_key(moments: np.ndarray, decimals: int = 6) -> tuple[float, ...]:
    """
    Hashable canonical representative for collinear moments up to global inversion.

    This lets us deduplicate configurations in O(1) average time instead of
    repeatedly scanning previously accepted states.
    """
    canon = _canonical_moment_vector(np.asarray(moments, dtype=float))
    rounded = np.round(canon, decimals=decimals)
    return tuple(float(value) for value in rounded.reshape(-1))


def _canonicalize_first_spin_up(moments: np.ndarray, magnetic_indices: Sequence[int]) -> np.ndarray:
    """
    Canonicalize a configuration by forcing the first magnetic spin to point up.
    Configurations related by global inversion then share the same representative.
    """
    arr = np.asarray(moments, dtype=float).copy()
    if not magnetic_indices:
        return arr
    first_idx = magnetic_indices[0]
    if arr.ndim == 1:
        if arr[first_idx] < 0:
            arr *= -1.0
        return arr

    if arr.ndim == 2 and arr.shape[1] == 3:
        first_vec = arr[first_idx]
        for component in first_vec:
            if abs(component) <= 1e-8:
                continue
            if component < 0:
                arr *= -1.0
            break
        return arr

    raise ValueError(f"Unsupported moment array shape for canonicalization: {arr.shape}")


def compute_config_energy(J_matrix: np.ndarray, moments: np.ndarray) -> float:
    """Heisenberg energy  E = -Σ_{i<j} J_ij (m_i · m_j)  (J > 0 = FM, J < 0 = AFM)."""
    J = np.asarray(J_matrix, dtype=float)
    m = np.asarray(moments, dtype=float)
    if m.ndim == 1:
        pairwise = np.outer(m, m)
    elif m.ndim == 2 and m.shape[1] == 3:
        pairwise = m @ m.T
    else:
        raise ValueError(f"Unsupported moment array shape for energy computation: {tuple(m.shape)}")
    return float(-0.5 * np.sum(J * pairwise))


def _spin_stats(moments: np.ndarray) -> Tuple[float, float]:
    """(net magnetization magnitude, total unsigned moment)."""
    arr = np.asarray(moments, dtype=float)
    if arr.ndim == 1:
        return float(np.sum(arr)), float(np.sum(np.abs(arr)))
    if arr.ndim == 2 and arr.shape[1] == 3:
        return float(np.linalg.norm(np.sum(arr, axis=0))), float(np.sum(np.linalg.norm(arr, axis=1)))
    raise ValueError(f"Unsupported moment array shape for spin statistics: {arr.shape}")

class _Adam:
    """Minimal Adam optimizer over a flat parameter vector."""
    def __init__(self, shape, lr=0.05, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = np.zeros(shape, dtype=float)
        self.v = np.zeros(shape, dtype=float)
        self.t = 0

    def step(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.beta1 * self.m + (1.0 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1.0 - self.beta2) * (grad * grad)
        m_hat = self.m / (1.0 - self.beta1 ** self.t)
        v_hat = self.v / (1.0 - self.beta2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

def _moments_from_angles(
    theta: np.ndarray,
    phi: np.ndarray,
    magnitudes: np.ndarray,
    magnetic_indices: Sequence[int],
    n_atoms: int,
) -> np.ndarray:
    """Build full 3D moment vectors, anchoring the first magnetic moment along +z."""
    moments = np.zeros((n_atoms, 3), dtype=float)
    if not magnetic_indices:
        return moments

    anchor_idx = magnetic_indices[0]
    moments[anchor_idx, 2] = magnitudes[anchor_idx]

    if len(magnetic_indices) == 1:
        return moments

    opt_indices = np.asarray(magnetic_indices[1:], dtype=int)
    mags = magnitudes[opt_indices]
    sin_theta = np.sin(theta)
    moments[opt_indices, 0] = mags * sin_theta * np.cos(phi)
    moments[opt_indices, 1] = mags * sin_theta * np.sin(phi)
    moments[opt_indices, 2] = mags * np.cos(theta)
    return moments


def _moments_from_collinear_params(
    collinear_params: np.ndarray,
    magnitudes: np.ndarray,
    magnetic_indices: Sequence[int],
) -> np.ndarray:
    """Build 1D collinear moment vector with the first active spin fixed up."""
    moments = np.zeros_like(magnitudes)
    if not magnetic_indices:
        return moments

    anchor_idx = magnetic_indices[0]
    moments[anchor_idx] = magnitudes[anchor_idx]
    if len(magnetic_indices) <= 1:
        return moments

    opt_indices = np.asarray(magnetic_indices[1:], dtype=int)
    moments[opt_indices] = magnitudes[opt_indices] * np.tanh(collinear_params)
    return moments


def _rotation_matrix_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 1e-12 or abs(angle) <= 1e-12:
        return np.eye(3)

    x, y, z = axis / axis_norm
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ])


def _rotation_to_align_with_positive_z(vector: np.ndarray) -> np.ndarray:
    target = np.array([0.0, 0.0, 1.0], dtype=float)
    vec = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return np.eye(3)

    unit_vec = vec / norm
    dot = float(np.clip(np.dot(unit_vec, target), -1.0, 1.0))
    if dot >= 1.0 - 1e-12:
        return np.eye(3)
    if dot <= -1.0 + 1e-12:
        return _rotation_matrix_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi)

    axis = np.cross(unit_vec, target)
    angle = math.acos(dot)
    return _rotation_matrix_from_axis_angle(axis, angle)


def _snap_moments_to_z_axis(moments: np.ndarray, magnetic_indices: Sequence[int]) -> np.ndarray:
    """
    Rotate the configuration so the first magnetic moment lies on +z, then snap
    each magnetic moment to either +z or -z according to which pole it is
    closer to. Non-magnetic sites remain zero.
    """
    arr = np.asarray(moments, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        return arr.copy()
    if not magnetic_indices:
        return arr.copy()

    rotated = arr @ _rotation_to_align_with_positive_z(arr[magnetic_indices[0]]).T
    snapped = np.zeros_like(rotated)
    for atom_idx in magnetic_indices:
        vec = rotated[atom_idx]
        magnitude = np.linalg.norm(vec)
        if magnitude <= 1e-12:
            continue
        snapped[atom_idx, 2] = magnitude if vec[2] >= 0.0 else -magnitude
    return _canonical_moment_vector(snapped)

def sort_and_rank(
    configs: List[SpinConfig],
    ox_assignments: Optional[List[OxidationStateAssignment]] = None,
    energy_tol: float = 1e-6,
) -> List[SpinConfig]:
    """Deduplicate up to global spin inversion and sort by (ox_state_energy, spin_energy)."""

    def _sort_key(c: SpinConfig):
        # Primary: oxidation-state model energy
        ox_energy = 0.0
        if ox_assignments and 0 <= c.ox_assignment_id < len(ox_assignments):
            ox_energy = ox_assignments[c.ox_assignment_id].total_energy
        # Secondary: spin configuration energy
        return (ox_energy, c.energy)

    configs.sort(key=_sort_key)
    unique: List[SpinConfig] = []
    seen_keys: set[tuple[float, ...]] = set()
    for c in configs:
        key = _canonical_moment_key(c.all_moments)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique.append(c)
    return unique


def collect_optimizer_states(
    J_matrix: np.ndarray,
    magnitudes: np.ndarray,
    *,
    n_trials: int = 100,
    n_steps: int = 500,
    lr: float = 0.05,
    energy_tol: float = 1e-6,
    patience: int = 5,
    device: str = "cpu",  # accepted for backward compat; ignored
) -> List[SpinConfig]:
    """Collinear 1D optimizer over spin signs with fixed moment magnitudes."""
    del device  # numpy-only; arg kept for API compatibility
    magnitudes = np.asarray(magnitudes, dtype=float)
    n_atoms = magnitudes.shape[0]
    magnetic_mask = np.abs(magnitudes) > 1e-8
    magnetic_indices = np.flatnonzero(magnetic_mask).tolist()
    n_magnetic = len(magnetic_indices)

    if n_magnetic == 0:
        m = np.zeros(n_atoms, dtype=float)
        return [SpinConfig(energy=0.0, all_moments=m)]

    J = np.asarray(J_matrix, dtype=float)
    # Symmetrize for safe gradient computation (E = -1/2 m^T J m → ∇_m E = -J_sym m)
    J_sym = 0.5 * (J + J.T)
    found: List[SpinConfig] = []

    if n_magnetic == 1:
        final = np.zeros(n_atoms, dtype=float)
        final[magnetic_indices[0]] = float(magnitudes[magnetic_indices[0]])
        e = compute_config_energy(J_matrix, final)
        mag, n_unp = _spin_stats(final)
        return [SpinConfig(energy=e, all_moments=final, magnetization=mag, n_unpaired=n_unp)]

    opt_indices = np.asarray(magnetic_indices[1:], dtype=int)
    opt_mags = magnitudes[opt_indices]
    n_free = n_magnetic - 1

    for trial in range(n_trials):
        initial_signs = 2.0 * np.random.randint(0, 2, size=n_free).astype(float) - 1.0
        collinear_params = np.arctanh(0.99 * initial_signs)
        adam = _Adam(collinear_params.shape, lr=lr)

        best_loss, stale = float("inf"), 0
        for _ in range(n_steps):
            # Forward: build moments and energy
            tanh_p = np.tanh(collinear_params)
            moments_1d = np.zeros(n_atoms, dtype=float)
            moments_1d[magnetic_indices[0]] = magnitudes[magnetic_indices[0]]
            moments_1d[opt_indices] = opt_mags * tanh_p

            Jm = J_sym @ moments_1d        # shape (n_atoms,)
            energy = -0.5 * float(moments_1d @ Jm)

            # Gradient: dE/dm_i = -(J_sym m)_i ; dm_i/dp_i = μ_i (1 - tanh²(p_i))
            dE_dm_opt = -Jm[opt_indices]
            grad = dE_dm_opt * opt_mags * (1.0 - tanh_p * tanh_p)

            collinear_params = adam.step(collinear_params, grad)

            if energy < best_loss - energy_tol:
                best_loss, stale = energy, 0
            else:
                stale += 1
                if stale >= patience:
                    break

        final = _moments_from_collinear_params(collinear_params, magnitudes, magnetic_indices)
        quantized_final = np.zeros_like(final)
        for atom_idx in magnetic_indices:
            sign = 1.0 if final[atom_idx] >= 0.0 else -1.0
            quantized_final[atom_idx] = abs(float(magnitudes[atom_idx])) * sign
        final = _canonicalize_first_spin_up(quantized_final, magnetic_indices)

        final_energy = compute_config_energy(J_matrix, final)
        mag, n_unp = _spin_stats(final)
        found.append(SpinConfig(energy=final_energy, all_moments=final,
                                magnetization=mag, n_unpaired=n_unp))

        if (trial + 1) % 25 == 0:
            print(f"\r    1D optimizer trial {trial+1}/{n_trials}", end="", flush=True)

    if n_trials:
        print(f"\r    1D optimizer: {n_trials} trials completed              ")

    return sort_and_rank(found, energy_tol=energy_tol)


def collect_3d_optimizer_states(
    J_matrix: np.ndarray,
    magnitudes: np.ndarray,
    *,
    n_trials: int = 100,
    n_steps: int = 500,
    lr: float = 0.05,
    energy_tol: float = 1e-6,
    patience: int = 5,
    device: str = "cpu",  # accepted for backward compat; ignored
) -> List[SpinConfig]:
    """Full 3D optimizer over theta/phi with fixed moment magnitudes."""
    del device
    magnitudes = np.asarray(magnitudes, dtype=float)
    n_atoms = magnitudes.shape[0]
    magnetic_mask = np.abs(magnitudes) > 1e-8
    magnetic_indices = np.flatnonzero(magnetic_mask).tolist()
    n_magnetic = len(magnetic_indices)

    if n_magnetic == 0:
        m = np.zeros((n_atoms, 3), dtype=float)
        return [SpinConfig(energy=0.0, all_moments=m)]

    J = np.asarray(J_matrix, dtype=float)
    J_sym = 0.5 * (J + J.T)
    found: List[SpinConfig] = []

    if n_magnetic == 1:
        final = np.zeros((n_atoms, 3), dtype=float)
        final[magnetic_indices[0], 2] = float(magnitudes[magnetic_indices[0]])
        e = compute_config_energy(J_matrix, final)
        mag, n_unp = _spin_stats(final)
        return [SpinConfig(energy=e, all_moments=final, magnetization=mag, n_unpaired=n_unp)]

    opt_indices = np.asarray(magnetic_indices[1:], dtype=int)
    opt_mags = magnitudes[opt_indices]      # shape (n_free,)
    n_free = n_magnetic - 1

    for trial in range(n_trials):
        theta = math.pi * np.random.rand(n_free)
        phi = 2.0 * math.pi * np.random.rand(n_free)
        adam_theta = _Adam(theta.shape, lr=lr)
        adam_phi = _Adam(phi.shape, lr=lr)

        best_loss, stale = float("inf"), 0
        for _ in range(n_steps):
            sin_t = np.sin(theta)
            cos_t = np.cos(theta)
            sin_p = np.sin(phi)
            cos_p = np.cos(phi)

            # Build full moment array
            moments_3d = np.zeros((n_atoms, 3), dtype=float)
            moments_3d[magnetic_indices[0], 2] = magnitudes[magnetic_indices[0]]
            moments_3d[opt_indices, 0] = opt_mags * sin_t * cos_p
            moments_3d[opt_indices, 1] = opt_mags * sin_t * sin_p
            moments_3d[opt_indices, 2] = opt_mags * cos_t

            # Energy
            Jm = J_sym @ moments_3d                  # shape (n_atoms, 3)
            energy = -0.5 * float(np.sum(moments_3d * Jm))

            # ∇_{m_i} E = -(J_sym m)_i  (a 3-vector); want only free sites.
            dE_dm = -Jm[opt_indices]                 # (n_free, 3)

            # dm/dθ = μ (cosθ cosφ, cosθ sinφ, -sinθ)
            dm_dtheta = np.stack(
                (opt_mags * cos_t * cos_p,
                 opt_mags * cos_t * sin_p,
                 -opt_mags * sin_t),
                axis=1,
            )
            # dm/dφ = μ (-sinθ sinφ, sinθ cosφ, 0)
            dm_dphi = np.stack(
                (-opt_mags * sin_t * sin_p,
                 opt_mags * sin_t * cos_p,
                 np.zeros_like(theta)),
                axis=1,
            )

            grad_theta = np.sum(dE_dm * dm_dtheta, axis=1)
            grad_phi = np.sum(dE_dm * dm_dphi, axis=1)

            theta = adam_theta.step(theta, grad_theta)
            phi = adam_phi.step(phi, grad_phi)

            # Wrap angles like the torch version did
            theta = np.remainder(theta, math.pi)
            phi = np.remainder(phi, 2.0 * math.pi)

            if energy < best_loss - energy_tol:
                best_loss, stale = energy, 0
            else:
                stale += 1
                if stale >= patience:
                    break

        final = _moments_from_angles(theta, phi, magnitudes, magnetic_indices, n_atoms)
        final = _canonical_moment_vector(final)

        final_energy = compute_config_energy(J_matrix, final)
        snapped_final = _snap_moments_to_z_axis(final, magnetic_indices)
        snapped_energy = compute_config_energy(J_matrix, snapped_final)
        if snapped_energy <= final_energy + energy_tol:
            final = snapped_final
            final_energy = snapped_energy

        mag, n_unp = _spin_stats(final)
        found.append(SpinConfig(energy=final_energy, all_moments=final,
                                magnetization=mag, n_unpaired=n_unp))

        if (trial + 1) % 25 == 0:
            print(f"\r    3D optimizer trial {trial+1}/{n_trials}", end="", flush=True)

    if n_trials:
        print(f"\r    3D optimizer: {n_trials} trials completed              ")

    return sort_and_rank(found, energy_tol=energy_tol)

def exact_ising_solver_as_configs(
    J_matrix: np.ndarray,
    magnitudes: np.ndarray,
) -> List[SpinConfig]:
    """Enumerate all 2^n_magnetic Ising configurations (small systems). [2]"""
    magnitudes = np.asarray(magnitudes, dtype=float)
    n_atoms = magnitudes.shape[0]
    magnetic_mask = np.abs(magnitudes) > 1e-8
    mag_idx = np.flatnonzero(magnetic_mask).tolist()
    n_mag = len(mag_idx)

    if n_mag == 0:
        return [SpinConfig(energy=0.0, all_moments=np.zeros(n_atoms))]

    if n_mag > 20:
        print(f"  WARNING: 2^{n_mag} = {2**n_mag:,} configurations — this will be slow.")

    mags_np = magnitudes.copy()
    configs: List[SpinConfig] = []
    for bits in range(2**n_mag):
        m = mags_np.copy()
        for k, idx in enumerate(mag_idx):
            if bits & (1 << k):
                m[idx] = -m[idx]
        e = compute_config_energy(J_matrix, m)
        mag, n_unp = _spin_stats(m)
        configs.append(SpinConfig(energy=e, all_moments=m,
                                  magnetization=mag, n_unpaired=n_unp))
    return sort_and_rank(configs)

def generate_flip_states(
    base_states: List[SpinConfig],
    magnitudes: np.ndarray,
    J_matrix: np.ndarray,
    *,
    max_order: int = 2,
    max_configs: Optional[int] = 75_000,
) -> List[SpinConfig]:
    """
    Random spin-flip perturbations up to *max_order* simultaneous flips from
    each base state, sampled from the canonical space where the first magnetic
    spin is always kept pointing up. [2]
    """
    magnitudes = np.asarray(magnitudes, dtype=float)
    magnetic_mask = np.abs(magnitudes) > 1e-8
    mag_idx = np.flatnonzero(magnetic_mask).tolist()
    n_magnetic = len(mag_idx)

    if n_magnetic == 0:
        return []

    flippable_idx = mag_idx[1:]
    n_flippable = len(flippable_idx)

    max_order = min(max_order, n_flippable)
    if max_order <= 0:
        return []

    flip_configs: List[SpinConfig] = []
    seen_keys: set[tuple[float, ...]] = {
        _canonical_moment_key(_canonicalize_first_spin_up(state.all_moments, mag_idx))
        for state in base_states
    }

    per_base_total = sum(math.comb(n_flippable, k) for k in range(1, max_order + 1))
    total_possible = len(base_states) * per_base_total
    sampled_from_total = False

    def _iter_flip_tasks():
        for base_idx, _base_state in enumerate(base_states):
            for k in range(1, max_order + 1):
                for combo in combinations(flippable_idx, k):
                    yield (base_idx, k, combo)

    if max_configs is not None and max_configs > 0 and total_possible > max_configs:
        sampled_from_total = True
        sampled_tasks: List[Tuple[int, int, Tuple[int, ...]]] = []
        for seen_count, task in enumerate(_iter_flip_tasks(), start=1):
            if len(sampled_tasks) < max_configs:
                sampled_tasks.append(task)
            else:
                replace_idx = random.randint(1, seen_count)
                if replace_idx <= max_configs:
                    sampled_tasks[replace_idx - 1] = task
        tasks: Sequence[Tuple[int, int, Tuple[int, ...]]] = sampled_tasks
    else:
        tasks = list(_iter_flip_tasks())
        random.shuffle(tasks)

    selected_total = len(tasks)
    if sampled_from_total:
        print(f"  Sampling {selected_total:,} flip configurations "
              f"from {total_possible:,} total candidates")
    else:
        print(f"  Evaluating all {selected_total:,} flip configurations")

    processed_count = 0
    unique_count = 0
    for base_idx, k, combo in tasks:
        processed_count += 1
        base_state = base_states[base_idx]
        new_moments = _canonicalize_first_spin_up(base_state.all_moments, mag_idx)
        for atom_idx in combo:
            new_moments[atom_idx] = -new_moments[atom_idx]
        new_moments = _canonicalize_first_spin_up(new_moments, mag_idx)

        energy = compute_config_energy(J_matrix, new_moments)
        mag, n_unp = _spin_stats(new_moments)
        config = SpinConfig(
            energy=energy, all_moments=new_moments,
            n_flips=k,
            magnetization=mag, n_unpaired=n_unp,
            ox_assignment_id=base_state.ox_assignment_id,
        )
        key = _canonical_moment_key(config.all_moments)
        if key in seen_keys:
            if processed_count % 25000 == 0:
                if sampled_from_total:
                    progress = (
                        f"\r    processed {processed_count:,} / {selected_total:,} sampled "
                        f"flip configurations (kept {unique_count:,} unique from {total_possible:,} total)"
                    )
                else:
                    progress = (
                        f"\r    processed {processed_count:,} / {selected_total:,} "
                        f"flip configurations (kept {unique_count:,} unique)"
                    )
                print(progress, end="", flush=True)
            continue
        seen_keys.add(key)
        flip_configs.append(config)
        unique_count += 1

        if processed_count % 25000 == 0:
            if sampled_from_total:
                progress = (
                    f"\r    processed {processed_count:,} / {selected_total:,} sampled "
                    f"flip configurations (kept {unique_count:,} unique from {total_possible:,} total)"
                )
            else:
                progress = (
                    f"\r    processed {processed_count:,} / {selected_total:,} "
                    f"flip configurations (kept {unique_count:,} unique)"
                )
            print(progress, end="", flush=True)

    if processed_count:
        if sampled_from_total:
            print(
                f"\r    processed {processed_count:,} / {selected_total:,} sampled "
                f"flip configurations (kept {unique_count:,} unique from {total_possible:,} total)          "
            )
        else:
            print(
                f"\r    processed {processed_count:,} / {selected_total:,} "
                f"flip configurations (kept {unique_count:,} unique)          "
            )

    return flip_configs

def solve_for_assignment(
    assignment: OxidationStateAssignment,
    J_matrix: np.ndarray,
    *,
    magnetic_site_indices: Optional[Sequence[int]] = None,
    method: str = "optimizer",
    collinear: bool = True,
    n_trials: int = 20,
    n_steps: int = 250,
    lr: float = 0.05,
    energy_tol: float = 1e-4,
    patience: int = 5,
    max_flip_order: int = 2,
    device: str = "cpu",
    max_flip_configs: Optional[int] = 75_000,
) -> Tuple[List[SpinConfig], List[SpinConfig]]:
    """
    Full spin-solve pipeline for one oxidation-state assignment.

    `method="optimizer"` runs either the default 1D collinear relaxation
    (`collinear=True`) or the full 3D vector solver (`collinear=False`).
    Use `method="exact"` for exhaustive Ising enumeration on small systems.

    Returns
    -------
    base_states : list[SpinConfig]
    all_configs : list[SpinConfig]   (base + flips, sorted & deduped)
    """
    dist_str = format_oxidation_distribution(assignment.distributions)
    if magnetic_site_indices is None:
        magnetic_site_indices = np.flatnonzero(assignment.magnetic_moments).tolist()

    magnetic_site_indices = list(magnetic_site_indices)
    magnitudes_np = np.asarray(assignment.magnetic_moments, dtype=float)[magnetic_site_indices]
    J_matrix = np.asarray(J_matrix, dtype=float)
    n_atoms = len(magnitudes_np)
    if J_matrix.shape != (n_atoms, n_atoms):
        raise ValueError(
            f"J_matrix shape {tuple(J_matrix.shape)} does not match the selected magnetic-site count {n_atoms}."
        )

    print(f"\n{'─' * 80}")
    print(f"  ASSIGNMENT:  {dist_str}  (model E = {assignment.total_energy:.3f})")
    print(f"{'─' * 80}")

    magnitudes = magnitudes_np
    mag_mask = np.abs(magnitudes) > 1e-8
    n_mag = int(np.count_nonzero(mag_mask))

    print(f"  Magnetic sites with exchange couplings: {n_atoms}")
    print(f"  Nonzero magnetic moments in this assignment: {n_mag} / {n_atoms}")
    if n_mag == 0:
        raise ValueError(NO_NONZERO_MAGNETIC_MOMENTS_MESSAGE)
    unique_mags = sorted(set(f"{v:.2f}" for v in magnitudes[mag_mask].tolist()))
    print(f"  Unique |μ|: {', '.join(unique_mags)} μ_B")

    # --- Phase 1: base states ---
    if method == "optimizer":
        solver_fn = collect_optimizer_states if collinear else collect_3d_optimizer_states
        base_states = solver_fn(
            J_matrix, magnitudes,
            n_trials=n_trials, n_steps=n_steps, lr=lr,
            energy_tol=energy_tol, patience=patience, device=device,
        )
    elif method == "exact":
        base_states = exact_ising_solver_as_configs(J_matrix, magnitudes)
    else:
        raise ValueError(
            f"Unknown spin-solver method {method!r}. "
            "Expected 'optimizer' or 'exact'."
        )

    for c in base_states:
        c.ox_assignment_id = 0

    base_sorted = sort_and_rank(base_states)
    print(f"  Found {len(base_sorted)} unique base state(s)")

    all_configs: List[SpinConfig] = list(base_sorted)

    # --- Phase 2: spin flips ---
    if max_flip_order > 0 and n_mag > 0:
        flips = generate_flip_states(
            base_sorted, magnitudes, J_matrix,
            max_order=max_flip_order, max_configs=max_flip_configs,
        )
        for c in flips:
            c.ox_assignment_id = 0
        all_configs.extend(flips)

    all_sorted = sort_and_rank(all_configs)
    if all_sorted:
        print(f"  Best energy for this assignment: {all_sorted[0].energy:.6f}")

    return base_sorted, all_sorted
