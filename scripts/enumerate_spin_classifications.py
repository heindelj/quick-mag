"""Enumerate and archive the 8-atom (2x2x2 cube) spin-classification table.

The reusable core (string encoding, pure A/C/G/F orders, L1 distances, category
lookup) lives in ``src/cube_spin_lookup.py`` so the application and this script
share one implementation. This script enumerates all 3^8 = 6561 cube strings,
saves the lookup table to ``data/spin_classification_8.npz``, and cross-checks
the grouping with the nearest-neighbour spin-coupling matrix.

Finding: on the bipartite cube graph (Q3) every unfrustrated collinear order is
gauge-equivalent to plain Q3, so ALL of F/A/C/G share one spectrum
{-3,-1,-1,-1,1,1,1,3} -- eigenvalues cannot separate the pure orders. The L1
distance-vector is the real classifier; the eigenvalue spectrum is only a
(coarser) symmetry invariant.

E-type is intentionally NOT handled here: its primitive cell is 18 atoms (four
cubes joined in x and y) and is deferred until the A/C/G/F lookup is in place.

Run:  python scripts/enumerate_spin_classifications.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quick_mag.cube_spin_lookup import (  # noqa: E402
    CLASS_ORDER,
    CUBE_COORDS,
    SPIN_VALUES,
    decode,
    distance_vector,
    encode,
    nearest_label,
    pure_strings,
)


def cube_edges() -> list[tuple[int, int]]:
    """The 12 nearest-neighbour edges of the periodic 2x2x2 cube (graph Q3)."""
    coord_to_index = {coord: n for n, coord in enumerate(CUBE_COORDS)}
    edges: set[tuple[int, int]] = set()
    for coord in CUBE_COORDS:
        site = coord_to_index[coord]
        for axis in range(3):
            neighbor = list(coord)
            neighbor[axis] ^= 1  # the single distinct neighbour along a size-2 axis
            other = coord_to_index[tuple(neighbor)]
            edges.add((min(site, other), max(site, other)))
    return sorted(edges)


CUBE_EDGES = cube_edges()


def nn_coupling_fingerprint(spins: np.ndarray) -> tuple[float, ...]:
    """Sorted, rounded eigenvalues of the NN spin-coupling matrix on the cube."""
    matrix = np.zeros((8, 8), dtype=np.float64)
    for a, b in CUBE_EDGES:
        coupling = float(spins[a] * spins[b])
        matrix[a, b] = coupling
        matrix[b, a] = coupling
    eigvals = np.linalg.eigvalsh(matrix)
    return tuple(np.round(np.sort(eigvals), 6))


def build_table() -> dict[str, object]:
    reps = pure_strings()
    total = 3 ** 8
    distances = np.empty((total, len(CLASS_ORDER)), dtype=np.int8)
    labels: list[str] = [""] * total
    fingerprints: list[tuple[float, ...]] = [()] * total

    for index in range(total):
        spins = decode(index)
        dist = distance_vector(spins)
        distances[index] = dist
        labels[index] = nearest_label(dist)
        fingerprints[index] = nn_coupling_fingerprint(spins)

    return {
        "reps": reps,
        "distances": distances,
        "labels": labels,
        "fingerprints": fingerprints,
    }


def summarize(table: dict[str, object]) -> None:
    reps = table["reps"]
    distances: np.ndarray = table["distances"]  # type: ignore[assignment]
    labels: list[str] = table["labels"]  # type: ignore[assignment]
    fingerprints: list[tuple[float, ...]] = table["fingerprints"]  # type: ignore[assignment]
    total = distances.shape[0]

    print(f"Enumerated {total} cube strings (3^8).")
    print("\nPure representatives per class:")
    for name in CLASS_ORDER:
        print(f"  {name}: {len(reps[name])} strings")

    pure_per_class = {name: 0 for name in CLASS_ORDER}
    for index in range(total):
        for col, name in enumerate(CLASS_ORDER):
            if distances[index, col] == 0:
                pure_per_class[name] += 1
    print("\nStrings at distance 0 to each class (recovered pure orders):")
    for name in CLASS_ORDER:
        print(f"  {name}: {pure_per_class[name]}")

    by_distance: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for index in range(total):
        by_distance[tuple(int(v) for v in distances[index])].append(index)
    print(f"\nDistinct distance-vector groups: {len(by_distance)}")

    distinct_fingerprints = len(set(fingerprints))
    consistent = all(
        len({fingerprints[i] for i in members}) == 1
        for members in by_distance.values()
    )
    print(f"Distinct NN-coupling eigenvalue fingerprints: {distinct_fingerprints}")
    print(
        "NN-coupling fingerprint constant within every distance group: "
        f"{consistent}"
    )

    pure_fp = {
        name: nn_coupling_fingerprint(reps[name][0]) for name in CLASS_ORDER
    }
    all_pure_same = len(set(pure_fp.values())) == 1
    print("\nFinding -- NN-coupling eigenvalues do NOT distinguish the pure orders:")
    for name in CLASS_ORDER:
        print(f"  {name}: {tuple(float(v) for v in pure_fp[name])}")
    if all_pure_same:
        print(
            "  All of F/A/C/G share the Q3 spectrum {-3,-1,-1,-1,1,1,1,3}: on the\n"
            "  bipartite cube graph every unfrustrated collinear order is gauge-\n"
            "  equivalent to plain Q3, so a pairwise-product matrix cannot separate\n"
            "  them by eigenvalue. The L1 distance-vector is the actual classifier;\n"
            "  the eigenvalue spectrum is only a (coarser) symmetry-invariant."
        )

    label_counts: dict[str, int] = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    print("\nNearest-class label distribution (top 12):")
    for label, count in sorted(label_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:12]:
        print(f"  {label:<8} {count}")


def save_table(table: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        distances=table["distances"],
        labels=np.array(table["labels"], dtype=object),
        class_order=np.array(CLASS_ORDER, dtype=object),
        spin_values=np.array(SPIN_VALUES, dtype=np.int8),
        cube_coords=np.array(CUBE_COORDS, dtype=np.int8),
    )
    print(f"\nSaved lookup table -> {path}")


def validate(table: dict[str, object]) -> None:
    """Internal sanity asserts on the enumerated table."""
    reps = table["reps"]
    distances: np.ndarray = table["distances"]  # type: ignore[assignment]

    for col, name in enumerate(CLASS_ORDER):
        for rep in reps[name]:
            assert distances[encode(rep)][col] == 0, f"{name} rep not at distance 0"

    expected = {"A": 6, "C": 6, "G": 2, "F": 2}
    for col, name in enumerate(CLASS_ORDER):
        count = int((distances[:, col] == 0).sum())
        assert count == expected[name], f"{name}: {count} pure strings, expected {expected[name]}"

    g_rep = reps["G"][0].copy()
    g_rep[0] *= -1
    assert distances[encode(g_rep)][CLASS_ORDER.index("G")] == 2

    for name in CLASS_ORDER:
        fps = {nn_coupling_fingerprint(rep) for rep in reps[name]}
        assert len(fps) == 1, f"{name} representatives disagree on NN fingerprint"
    print("Internal validation asserts passed.")


def main() -> None:
    table = build_table()
    validate(table)
    summarize(table)
    output = Path(__file__).resolve().parents[1] / "data" / "spin_classification_8.npz"
    save_table(table, output)


if __name__ == "__main__":
    main()
