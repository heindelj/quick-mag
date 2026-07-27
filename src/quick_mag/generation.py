"""UI-free structure generation from perovskite generation parameters.

These helpers reconstruct a :class:`ChemicalStructure` purely from a
:class:`PerovskiteGenerationParameters` object (the provenance stored on every
structure built in the UI). They live here, separate from the imgui UI module,
so standalone scripts and the package API can rebuild structures without pulling
in any GUI dependencies.

The label assignment is deterministic for every formula mode, including the
hash-seeded high-entropy sampling, so a structure rebuilt from its parameters is
identical (lattice, coordinates, and atomic labels) to the one the UI saved.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np

from quick_mag.element_data import is_valid_symbol
from quick_mag.perovskite_builder import PerovskiteBuild
from quick_mag.structure import (
    ChemicalStructure,
    PerovskiteGenerationParameters,
    build_from_generation_parameters,
)


def normalize_element_symbol(raw_symbol: str) -> str:
    symbol = raw_symbol.strip()
    if not symbol:
        raise ValueError("Element symbols cannot be empty.")
    return symbol[0].upper() + symbol[1:].lower()


def normalized_distribution(
    entries: list[tuple[str, float]],
    *,
    site_name: str,
) -> list[tuple[str, float]]:
    if not entries:
        raise ValueError(
            f"{site_name}-site high-entropy distribution needs at least one row."
        )

    normalized_entries: list[tuple[str, float]] = []
    total = 0.0
    for raw_symbol, raw_fraction in entries:
        symbol = normalize_element_symbol(raw_symbol)
        if not is_valid_symbol(symbol):
            raise ValueError(
                f"{site_name}-site element '{raw_symbol}' is not a valid element symbol."
            )
        fraction = max(0.0, float(raw_fraction))
        total += fraction
        normalized_entries.append((symbol, fraction))

    if total <= 0.0:
        raise ValueError(
            f"{site_name}-site high-entropy fractions must include a positive value."
        )
    return [(symbol, fraction / total) for symbol, fraction in normalized_entries]


def deterministic_seed(*parts: object) -> int:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(repr(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "little", signed=False)


def sampled_distribution_labels(
    entries: list[tuple[str, float]],
    count: int,
    *,
    seed_parts: tuple[object, ...],
) -> list[str]:
    if count <= 0:
        return []
    normalized = normalized_distribution(entries, site_name=str(seed_parts[0]))
    symbols = [symbol for symbol, _ in normalized]
    probabilities = np.asarray([fraction for _, fraction in normalized], dtype=float)
    rng = np.random.default_rng(deterministic_seed(*seed_parts, normalized, count))
    choices = rng.choice(len(symbols), size=count, p=probabilities)
    return [symbols[int(index)] for index in choices]


def a_site_grid_shape_for_build(
    build: PerovskiteBuild,
    periodic: bool,
) -> tuple[int, int, int]:
    nx, ny, nz = (int(value) for value in build.octahedra.shape)
    if periodic:
        return nx, ny, nz
    return nx + 1, ny + 1, nz + 1


def formula_atomic_labels_for_build(
    build: PerovskiteBuild,
    *,
    periodic: bool,
    formula_mode: str,
    a_site_element: str,
    b_site_element: str,
    x_site_element: str,
    a2_site_element: str,
    b2_site_element: str,
    high_entropy_a_sites: list[tuple[str, float]],
    high_entropy_b_sites: list[tuple[str, float]],
    high_entropy_x_sites: list[tuple[str, float]],
    high_entropy_sample_index: int = 0,
) -> list[str]:
    formula_mode = str(formula_mode)
    if formula_mode == "high_entropy":
        sample = int(high_entropy_sample_index)
        a_labels = sampled_distribution_labels(
            high_entropy_a_sites,
            len(build.a_sites),
            seed_parts=("A", build.octahedra.shape, periodic, sample),
        )
        b_labels = sampled_distribution_labels(
            high_entropy_b_sites,
            len(build.b_sites),
            seed_parts=("B", build.octahedra.shape, periodic, sample),
        )
        x_labels = sampled_distribution_labels(
            high_entropy_x_sites,
            len(build.x_sites),
            seed_parts=("X", build.octahedra.shape, periodic, sample),
        )
        return a_labels + b_labels + x_labels

    a_symbol = normalize_element_symbol(a_site_element)
    b_symbol = normalize_element_symbol(b_site_element)
    x_symbol = normalize_element_symbol(x_site_element)
    for site_name, symbol in (
        ("A", a_symbol),
        ("B", b_symbol),
        ("X", x_symbol),
    ):
        if not is_valid_symbol(symbol):
            raise ValueError(
                f"{site_name}-site element '{symbol}' is not a valid element symbol."
            )

    if formula_mode in ("quadruple", "dq"):
        a2_symbol = normalize_element_symbol(a2_site_element)
        if not is_valid_symbol(a2_symbol):
            raise ValueError(
                f"A'-site element '{a2_site_element}' is not a valid element symbol."
            )
        a_shape = a_site_grid_shape_for_build(build, periodic)
        a_labels = []
        for i, j, k in np.ndindex(a_shape):
            a_labels.append(a_symbol if i % 2 == j % 2 == k % 2 else a2_symbol)
    else:
        a_labels = [a_symbol] * len(build.a_sites)

    if formula_mode in ("double", "dq"):
        b2_symbol = normalize_element_symbol(b2_site_element)
        if not is_valid_symbol(b2_symbol):
            site_label = "B'" if formula_mode == "dq" else "B''"
            raise ValueError(
                f"{site_label}-site element '{b2_site_element}' is not a valid element symbol."
            )
        b_labels = []
        for i, j, k in np.ndindex(build.octahedra.shape):
            b_labels.append(b_symbol if (i + j + k) % 2 == 0 else b2_symbol)
    else:
        b_labels = [b_symbol] * len(build.b_sites)

    x_labels = [x_symbol] * len(build.x_sites)
    return a_labels + b_labels + x_labels


def formula_atomic_labels_from_parameters(
    build: PerovskiteBuild,
    params: PerovskiteGenerationParameters,
) -> list[str]:
    return formula_atomic_labels_for_build(
        build,
        periodic=bool(params.periodic),
        formula_mode=getattr(params, "formula_mode", "perovskite"),
        a_site_element=params.a_site_element,
        b_site_element=params.b_site_element,
        x_site_element=params.x_site_element,
        a2_site_element=getattr(params, "a2_site_element", "Sr"),
        b2_site_element=getattr(params, "b2_site_element", "Co"),
        high_entropy_a_sites=list(
            getattr(params, "high_entropy_a_sites", [("La", 1.0)])
        ),
        high_entropy_b_sites=list(
            getattr(params, "high_entropy_b_sites", [("Fe", 1.0)])
        ),
        high_entropy_x_sites=list(
            getattr(params, "high_entropy_x_sites", [("O", 1.0)])
        ),
        high_entropy_sample_index=int(
            getattr(params, "high_entropy_sample_index", 0)
        ),
    )


def generated_structure_from_parameters(
    params: PerovskiteGenerationParameters,
    *,
    name: str,
    periodic: bool,
) -> ChemicalStructure:
    build = build_from_generation_parameters(replace(params, periodic=periodic))
    # This function always emits the canonical, fully-occupied (A, B, X) build in
    # order, so the atom-order metadata must describe *that* build, not whatever
    # was stored on ``params``. Toggling ``periodic`` (e.g. for periodic-image
    # rendering) changes the A/X site counts, so carrying the old permutation /
    # site_roles / vacancy fields over would leave them inconsistent with the
    # produced structure (an IndexError waiting to happen in spin classification).
    na, nb, nx = len(build.a_sites), len(build.b_sites), len(build.x_sites)
    params_for_build = replace(
        params,
        periodic=periodic,
        site_roles=["A"] * na + ["B"] * nb + ["X"] * nx,
        permutation=np.arange(na + nb + nx, dtype=np.int64),
        x_vacancy_fraction=0.0,
        x_removed_count=0,
        removed_x_site_indices=np.zeros(0, dtype=np.int64),
    )
    lattice_a = 2.0 * float(params_for_build.center_to_vertex_distance_x)
    lattice_b = 2.0 * float(params_for_build.center_to_vertex_distance_y)
    lattice_c = 2.0 * float(params_for_build.center_to_vertex_distance_z)
    lattice = np.array(
        [
            [(int(params_for_build.n_oct_x) + 1) * lattice_a, 0.0, 0.0],
            [0.0, (int(params_for_build.n_oct_y) + 1) * lattice_b, 0.0],
            [0.0, 0.0, (int(params_for_build.n_oct_z) + 1) * lattice_c],
        ],
        dtype=np.float64,
    )
    cartesian_coords = np.vstack((build.a_sites, build.b_sites, build.x_sites)).astype(
        np.float64
    )
    cartesian_coords -= np.asarray(params_for_build.cell_origin, dtype=np.float64)
    atomic_labels = formula_atomic_labels_from_parameters(build, params_for_build)
    return ChemicalStructure.with_zero_magnetic_moments(
        name=name,
        lattice=lattice,
        cartesian_coords=cartesian_coords,
        atomic_labels=atomic_labels,
        is_periodic=periodic,
        generation_parameters=params_for_build,
    )


# ---------------------------------------------------------------------------
# High-level builders
#
# These are thin, mode-specific wrappers over generated_structure_from_parameters.
# Each takes only the arguments meaningful for its perovskite type and fills in
# the rest of PerovskiteGenerationParameters with the builder's conventions
# (center at the origin, cubic-by-default lattice). They are what the
# auto-generated build scripts call, so users see the simplest sensible API.
#
# ``a``/``b``/``c`` are the single-octahedron cell edge lengths (Angstrom); ``b``
# and ``c`` default to ``a`` for a cubic cell. ``n_cells_*`` is the number of
# octahedra (B-site grid points) along each axis, so the full cell edge is
# ``n_cells * a``.
# ---------------------------------------------------------------------------


def _perovskite_structure(
    *,
    name: str,
    formula_mode: str,
    a_site: str,
    b_site: str,
    x_site: str,
    a: float,
    b: float | None,
    c: float | None,
    n_cells_x: int,
    n_cells_y: int,
    n_cells_z: int,
    tilt_system: str,
    tilt_angles_deg: tuple[float, float, float],
    periodic: bool,
    a2_site: str = "Sr",
    b2_site: str = "Co",
    high_entropy_a_sites: list[tuple[str, float]] | None = None,
    high_entropy_b_sites: list[tuple[str, float]] | None = None,
    high_entropy_x_sites: list[tuple[str, float]] | None = None,
    high_entropy_sample_index: int = 0,
) -> ChemicalStructure:
    edge_a = float(a)
    edge_b = float(a if b is None else b)
    edge_c = float(a if c is None else c)
    half = np.array([edge_a, edge_b, edge_c], dtype=np.float64) / 2.0
    center = np.zeros(3, dtype=np.float64)
    tilt_x, tilt_y, tilt_z = tilt_angles_deg

    he_kwargs: dict = {}
    if high_entropy_a_sites is not None:
        he_kwargs["high_entropy_a_sites"] = high_entropy_a_sites
    if high_entropy_b_sites is not None:
        he_kwargs["high_entropy_b_sites"] = high_entropy_b_sites
    if high_entropy_x_sites is not None:
        he_kwargs["high_entropy_x_sites"] = high_entropy_x_sites

    params = PerovskiteGenerationParameters(
        center=center,
        n_oct_x=max(1, int(n_cells_x)) - 1,
        n_oct_y=max(1, int(n_cells_y)) - 1,
        n_oct_z=max(1, int(n_cells_z)) - 1,
        center_to_vertex_distance_x=0.5 * edge_a,
        center_to_vertex_distance_y=0.5 * edge_b,
        center_to_vertex_distance_z=0.5 * edge_c,
        tilt_system=tilt_system,
        tilt_angle_x_deg=float(tilt_x),
        tilt_angle_y_deg=float(tilt_y),
        tilt_angle_z_deg=float(tilt_z),
        periodic=periodic,
        a_site_element=a_site,
        b_site_element=b_site,
        x_site_element=x_site,
        formula_mode=formula_mode,
        a2_site_element=a2_site,
        b2_site_element=b2_site,
        high_entropy_sample_index=int(high_entropy_sample_index),
        cell_origin=center - half,
        source="perovskite_builder",
        **he_kwargs,
    )
    return generated_structure_from_parameters(params, name=name, periodic=periodic)


def generate_single_perovskite(
    name: str,
    *,
    a_site: str,
    b_site: str,
    x_site: str,
    a: float,
    b: float | None = None,
    c: float | None = None,
    n_cells_x: int = 1,
    n_cells_y: int = 1,
    n_cells_z: int = 1,
    tilt_system: str = "a0a0a0",
    tilt_angles_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    periodic: bool = True,
) -> ChemicalStructure:
    """Build a simple ABX3 perovskite."""
    return _perovskite_structure(
        name=name,
        formula_mode="perovskite",
        a_site=a_site,
        b_site=b_site,
        x_site=x_site,
        a=a,
        b=b,
        c=c,
        n_cells_x=n_cells_x,
        n_cells_y=n_cells_y,
        n_cells_z=n_cells_z,
        tilt_system=tilt_system,
        tilt_angles_deg=tilt_angles_deg,
        periodic=periodic,
    )


def generate_double_perovskite(
    name: str,
    *,
    a_site: str,
    b_site: str,
    b2_site: str,
    x_site: str,
    a: float,
    b: float | None = None,
    c: float | None = None,
    n_cells_x: int = 2,
    n_cells_y: int = 2,
    n_cells_z: int = 2,
    tilt_system: str = "a0a0a0",
    tilt_angles_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    periodic: bool = True,
) -> ChemicalStructure:
    """Build an A2 B'B'' X6 double perovskite (rock-salt B-site ordering).

    ``b_site``/``b2_site`` are the two B-site species; an even ``n_cells_*`` is
    needed for a consistent rock-salt pattern (default 2x2x2).
    """
    return _perovskite_structure(
        name=name,
        formula_mode="double",
        a_site=a_site,
        b_site=b_site,
        b2_site=b2_site,
        x_site=x_site,
        a=a,
        b=b,
        c=c,
        n_cells_x=n_cells_x,
        n_cells_y=n_cells_y,
        n_cells_z=n_cells_z,
        tilt_system=tilt_system,
        tilt_angles_deg=tilt_angles_deg,
        periodic=periodic,
    )


def generate_quadruple_perovskite(
    name: str,
    *,
    a_site: str,
    a2_site: str,
    b_site: str,
    x_site: str,
    a: float,
    b: float | None = None,
    c: float | None = None,
    n_cells_x: int = 2,
    n_cells_y: int = 2,
    n_cells_z: int = 2,
    tilt_system: str = "a0a0a0",
    tilt_angles_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    periodic: bool = True,
) -> ChemicalStructure:
    """Build an A A'3 B4 X12 quadruple perovskite (A-site ordering).

    ``a_site``/``a2_site`` are the two A-site species; an even ``n_cells_*`` is
    needed for a consistent A-site pattern (default 2x2x2).
    """
    return _perovskite_structure(
        name=name,
        formula_mode="quadruple",
        a_site=a_site,
        a2_site=a2_site,
        b_site=b_site,
        x_site=x_site,
        a=a,
        b=b,
        c=c,
        n_cells_x=n_cells_x,
        n_cells_y=n_cells_y,
        n_cells_z=n_cells_z,
        tilt_system=tilt_system,
        tilt_angles_deg=tilt_angles_deg,
        periodic=periodic,
    )


def generate_dq_perovskite(
    name: str,
    *,
    a_site: str,
    a2_site: str,
    b_site: str,
    b2_site: str,
    x_site: str,
    a: float,
    b: float | None = None,
    c: float | None = None,
    n_cells_x: int = 2,
    n_cells_y: int = 2,
    n_cells_z: int = 2,
    tilt_system: str = "a0a0a0",
    tilt_angles_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    periodic: bool = True,
) -> ChemicalStructure:
    """Build an A A'3 B B' X12 doubly-ordered (DQ) perovskite.

    Combines A-site (``a_site``/``a2_site``) and B-site (``b_site``/``b2_site``)
    ordering; an even ``n_cells_*`` is needed (default 2x2x2).
    """
    return _perovskite_structure(
        name=name,
        formula_mode="dq",
        a_site=a_site,
        a2_site=a2_site,
        b_site=b_site,
        b2_site=b2_site,
        x_site=x_site,
        a=a,
        b=b,
        c=c,
        n_cells_x=n_cells_x,
        n_cells_y=n_cells_y,
        n_cells_z=n_cells_z,
        tilt_system=tilt_system,
        tilt_angles_deg=tilt_angles_deg,
        periodic=periodic,
    )


def generate_high_entropy_perovskite(
    name: str,
    *,
    a_sites: list[tuple[str, float]],
    b_sites: list[tuple[str, float]],
    x_sites: list[tuple[str, float]],
    a: float,
    b: float | None = None,
    c: float | None = None,
    n_cells_x: int = 1,
    n_cells_y: int = 1,
    n_cells_z: int = 1,
    tilt_system: str = "a0a0a0",
    tilt_angles_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    periodic: bool = True,
    sample_index: int = 0,
) -> ChemicalStructure:
    """Build a high-entropy perovskite from per-site (element, fraction) mixes.

    ``a_sites``/``b_sites``/``x_sites`` are lists of ``(symbol, fraction)`` pairs
    (fractions are normalized). Site occupancies are sampled deterministically
    from these distributions for a given grid, so the result is reproducible.
    ``sample_index`` selects the occupancy realization: distinct indices give
    distinct-but-reproducible weighted draws (use it to enumerate samples).
    """
    return _perovskite_structure(
        name=name,
        formula_mode="high_entropy",
        a_site=a_sites[0][0],
        b_site=b_sites[0][0],
        x_site=x_sites[0][0],
        a=a,
        b=b,
        c=c,
        n_cells_x=n_cells_x,
        n_cells_y=n_cells_y,
        n_cells_z=n_cells_z,
        tilt_system=tilt_system,
        tilt_angles_deg=tilt_angles_deg,
        periodic=periodic,
        high_entropy_a_sites=list(a_sites),
        high_entropy_b_sites=list(b_sites),
        high_entropy_x_sites=list(x_sites),
        high_entropy_sample_index=int(sample_index),
    )
