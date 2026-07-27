"""Lookup table for classifying a 2x2x2 cube of collinear spins as A/C/G/F.

A cube of 8 B-site spins (each in {+1, 0, -1}, taken along +/-z) is encoded as a
base-3 integer in [0, 6561). We precompute, for every such string, the L1
distance to the nearest pure A/C/G/F ordering and the resulting category, so
classifying a real cube is a single array lookup.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

# Fixed class order for the four distance columns.
CLASS_ORDER = ("A", "C", "G", "F")

# base-3 digit d in {0,1,2} maps to SPIN_VALUES[d].
SPIN_VALUES = (-1, 0, 1)

# Cube site index n = 4*i + 2*j + k over (i, j, k) in {0,1}^3 (axes a, b, c).
CUBE_COORDS = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]

NUM_STRINGS = 3 ** 8


def encode(spins: np.ndarray) -> int:
    """Length-8 spin string ({-1,0,1}) -> base-3 integer in [0, 6561)."""
    index = 0
    for value in spins:
        index = index * 3 + (int(value) + 1)
    return index


def decode(index: int) -> np.ndarray:
    """Inverse of :func:`encode`."""
    digits = np.empty(8, dtype=np.int8)
    for position in range(7, -1, -1):
        index, digit = divmod(index, 3)
        digits[position] = digit - 1
    return digits


@lru_cache(maxsize=1)
def pure_strings() -> dict[str, tuple[np.ndarray, ...]]:
    """Fully-occupied (+/-1) representatives of each pure order.

    Axis orientation and the global spin flip collapse into one class, so a
    class's distance is the minimum over all its representatives.
    """
    reps: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_ORDER}
    for sign in (1, -1):
        reps["F"].append(np.array([sign for _ in CUBE_COORDS], dtype=np.int8))
        reps["G"].append(
            np.array(
                [sign * (-1) ** (i + j + k) for (i, j, k) in CUBE_COORDS],
                dtype=np.int8,
            )
        )
        for axis in range(3):
            reps["A"].append(
                np.array(
                    [sign * (-1) ** coord[axis] for coord in CUBE_COORDS],
                    dtype=np.int8,
                )
            )
            other = [a for a in range(3) if a != axis]
            reps["C"].append(
                np.array(
                    [
                        sign * (-1) ** (coord[other[0]] + coord[other[1]])
                        for coord in CUBE_COORDS
                    ],
                    dtype=np.int8,
                )
            )
    return {name: tuple(strings) for name, strings in reps.items()}


def distance_vector(spins: np.ndarray) -> np.ndarray:
    """L1 distance to the nearest representative of each class, in CLASS_ORDER."""
    reps = pure_strings()
    return np.array(
        [
            min(int(np.abs(spins - rep).sum()) for rep in reps[name])
            for name in CLASS_ORDER
        ],
        dtype=np.int8,
    )


def nearest_label(distances: np.ndarray) -> str:
    """Class(es) at minimum distance, e.g. 'G' or 'A/G' on a tie."""
    best = int(np.min(distances))
    winners = [CLASS_ORDER[i] for i in range(len(CLASS_ORDER)) if distances[i] == best]
    return "/".join(winners)


@lru_cache(maxsize=1)
def category_table() -> tuple[str, ...]:
    """Per-string category: the single nearest class, or 'Other' on a tie.

    A cube is classified as its unique nearest pure order; ambiguous cubes (a
    tie between classes, including the all-zero cube) fall into 'Other'.
    """
    categories: list[str] = []
    for index in range(NUM_STRINGS):
        distances = distance_vector(decode(index))
        best = int(distances.min())
        winners = [
            CLASS_ORDER[i] for i in range(len(CLASS_ORDER)) if distances[i] == best
        ]
        categories.append(winners[0] if len(winners) == 1 else "Other")
    return tuple(categories)


def cube_category(spins: np.ndarray) -> str:
    """Classify one cube string ({-1,0,1}^8) as 'A'/'C'/'G'/'F' or 'Other'."""
    return category_table()[encode(spins)]
