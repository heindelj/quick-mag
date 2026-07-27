"""Slater-Koster p-d angular factors in tensor form.

Each real d orbital ``a`` is represented by its unit-normalized symmetric traceless
quadrupole tensor ``Q_a`` (Tr Q = 0, Tr Q^2 = 1). For a metal-ligand bond unit
vector ``u`` (ligand -> metal), the signed p-amplitude of orbital ``a`` splits
exactly into a sigma part (parallel to ``u``) and a pi part (perpendicular):

    A_sigma(a, u) = sqrt(3/2) * u^T Q_a u          (scalar)
    t_pi(a, u)    = sqrt(2) * [Q_a u - (u^T Q_a u) u]   (vector, perpendicular to u)

The polarization model consumes squared (intensity) weights: the induced ligand
spin polarization scales like the hopping squared, so channel weights are
sign-blind. Squaring is done per sigma/pi channel (incoherently) after projecting
the signed amplitudes onto a bridge-adapted orthonormal p frame.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

# Orbital order matches ion_descriptors.D_ORBITALS.
D_ORBITALS: Tuple[str, ...] = ("dxy", "dxz", "dyz", "dz2", "dx2-y2")


def _pair_tensor(a: int, b: int) -> np.ndarray:
    t = np.zeros((3, 3))
    t[a, b] = t[b, a] = 1.0 / math.sqrt(2.0)
    return t


D_ORBITAL_TENSORS: np.ndarray = np.stack(
    [
        _pair_tensor(0, 1),                                  # dxy
        _pair_tensor(0, 2),                                  # dxz
        _pair_tensor(1, 2),                                  # dyz
        np.diag([-1.0, -1.0, 2.0]) / math.sqrt(6.0),         # dz2
        np.diag([1.0, -1.0, 0.0]) / math.sqrt(2.0),          # dx2-y2
    ]
)


def pd_amplitudes(
    u: np.ndarray, rotation: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Signed SK pd amplitudes of all five d orbitals for bond direction ``u``.

    ``u`` must be a unit vector in the global frame. ``rotation`` optionally
    rotates the d-orbital frame (site local frame -> global): ``Q -> R Q R^T``.
    Returns ``(A_sigma, t_pi)`` with shapes ``(5,)`` and ``(5, 3)``; ``t_pi`` rows
    are perpendicular to ``u``.
    """
    u = np.asarray(u, dtype=float)
    tensors = D_ORBITAL_TENSORS
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        tensors = np.einsum("ab,nbc,dc->nad", rotation, tensors, rotation)
    qu = tensors @ u                          # (5, 3)
    uqu = qu @ u                              # (5,)
    a_sigma = math.sqrt(1.5) * uqu
    t_pi = math.sqrt(2.0) * (qu - np.outer(uqu, u))
    return a_sigma, t_pi


def channel_weights(
    u: np.ndarray,
    frame: np.ndarray,
    rotation: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Squared (intensity) channel weights of each d orbital on the p frame.

    ``frame`` is a 3x3 matrix whose *rows* are the orthonormal ligand p channels
    (e_1, e_2, e_3). Returns ``(W_sigma, W_pi)``, each ``(5, 3)`` and >= 0:

        W_sigma[a, p] = (A_sigma(a, u) * (e_p . u))**2
        W_pi[a, p]    = (e_p . t_pi(a, u))**2

    Squaring is per channel (incoherent in sigma/pi), so the weights are blind to
    the sign of ``u`` and of the p lobes; ``sum_p (W_sigma + W_pi)[a]`` is
    frame-independent (= A_sigma^2 + |t_pi|^2).
    """
    frame = np.asarray(frame, dtype=float)
    a_sigma, t_pi = pd_amplitudes(u, rotation=rotation)
    c = frame @ np.asarray(u, dtype=float)    # (3,) components of u on the frame
    w_sigma = np.outer(a_sigma, c) ** 2
    w_pi = (t_pi @ frame.T) ** 2
    return w_sigma, w_pi
