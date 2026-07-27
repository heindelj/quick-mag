from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from quick_mag.cube_spin_lookup import cube_category


AXIS_NAMES = ("a", "b", "c")

# Canonical perovskite spin orders plus an "Other" bucket for everything else
# (unknown / noncollinear / canted). Used for per-cell fractions and scatter color.
SPIN_CATEGORIES = ("F", "A", "C", "G", "E", "Other")


def spin_category(label: str) -> str:
    """Map a raw per-site/structure label to one of ``SPIN_CATEGORIES``."""
    return label if label in ("F", "A", "C", "G", "E") else "Other"


@dataclass(frozen=True)
class PerovskiteSiteIndexing:
    """Site-role indices for a perovskite B-site grid."""

    a_site_indices: np.ndarray
    b_site_indices: np.ndarray
    x_site_indices: np.ndarray
    b_grid_shape: tuple[int, int, int] | None = None
    grid_to_site: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "a_site_indices", np.asarray(self.a_site_indices, dtype=int)
        )
        object.__setattr__(
            self, "b_site_indices", np.asarray(self.b_site_indices, dtype=int)
        )
        object.__setattr__(
            self, "x_site_indices", np.asarray(self.x_site_indices, dtype=int)
        )
        if self.grid_to_site is not None:
            object.__setattr__(
                self, "grid_to_site", np.asarray(self.grid_to_site, dtype=int)
            )

    @classmethod
    def from_perovskite_build(cls, build) -> "PerovskiteSiteIndexing":
        """Create indexing from the existing ``PerovskiteBuild`` object."""
        b_grid_shape = tuple(int(value) for value in build.octahedra.shape)
        return cls(
            a_site_indices=np.asarray(build.a_site_indices, dtype=int),
            b_site_indices=np.asarray(build.b_site_indices, dtype=int),
            x_site_indices=np.asarray(build.x_site_indices, dtype=int),
            b_grid_shape=b_grid_shape,
            grid_to_site=np.asarray(build.b_site_indices, dtype=int),
        )


@dataclass
class SiteSpinClassification:
    site_index: int
    grid_index: tuple[int, int, int] | None
    label: str
    sigmas: dict[str, float]
    axis_signs: dict[str, int | None]
    kappa: float
    confidence: float
    notes: list[str] = field(default_factory=list)
    canted_flag: bool = False
    # Distance-2 neighbour correlation per axis (block-staggered eta_E proxy);
    # an axis with sigma2 == -1 indicates an up-up-down-down (E-type) modulation.
    sigmas2: dict[str, float] = field(default_factory=dict)


@dataclass
class SpinStructureClassification:
    label: str
    site_records: list[SiteSpinClassification]
    notes: list[str] = field(default_factory=list)


@dataclass
class SpinClassificationFractions:
    """Per-cell classification tally for a structure (one vote per B-site)."""

    counts: dict[str, int]
    total: int

    def fraction(self, label: str) -> float:
        if self.total <= 0:
            return 0.0
        return self.counts.get(label, 0) / self.total

    @property
    def dominant(self) -> str:
        if self.total <= 0:
            return "Other"
        # Tie-break by the canonical SPIN_CATEGORIES order.
        return max(
            SPIN_CATEGORIES,
            key=lambda name: (self.counts.get(name, 0), -SPIN_CATEGORIES.index(name)),
        )


def classification_fractions(
    classification: SpinStructureClassification,
) -> SpinClassificationFractions:
    """Tally per-site labels into per-category counts."""
    counts = {name: 0 for name in SPIN_CATEGORIES}
    total = 0
    for record in classification.site_records:
        counts[spin_category(record.label)] += 1
        total += 1
    return SpinClassificationFractions(counts=counts, total=total)


def classify_structure_by_cubes(
    structure,
    *,
    site_indexing: "PerovskiteSiteIndexing | None" = None,
    b_grid_shape: tuple[int, int, int] | None = None,
    grid_to_site: Sequence[int] | None = None,
    moment_eps: float = 1e-3,
) -> SpinClassificationFractions | None:
    """Classify every 2x2x2 B-site cube via the precomputed lookup table.

    Slides a 2x2x2 window over the B-site grid (one cube per origin), reads each
    cube's 8 spins as signs of their z-moment (``+1/0/-1``), and looks up the
    nearest A/C/G/F order (ties / no-moment cubes are "Other"). Returns the
    per-cube distribution, or ``None`` if the grid is too small to form a cube
    (any dimension < 2). Periodic structures wrap the window; non-periodic ones
    only use windows that stay inside the grid.

    Cube site order matches the lookup table: index ``4*di + 2*dj + dk`` over the
    local offsets ``(di, dj, dk) in {0,1}^3``.
    """
    indexing = _normalize_indexing(
        site_indexing=site_indexing,
        b_site_indices=None if site_indexing is not None else np.array([], dtype=int),
        x_site_indices=None,
        b_grid_shape=b_grid_shape,
        grid_to_site=grid_to_site,
    )
    if indexing.b_grid_shape is None or indexing.grid_to_site is None:
        return None

    grid_shape = tuple(int(value) for value in indexing.b_grid_shape)
    if any(size < 2 for size in grid_shape):
        return None

    grid = np.asarray(indexing.grid_to_site, dtype=int).reshape(grid_shape)
    moments = np.asarray(structure.magnetic_moments, dtype=float)
    if moments.ndim == 1:
        z_component = moments
    else:
        z_component = moments[:, 2]

    def spin_sign(site_index: int) -> int:
        if site_index < 0 or site_index >= len(z_component):
            return 0
        value = float(z_component[site_index])
        if value > moment_eps:
            return 1
        if value < -moment_eps:
            return -1
        return 0

    periodic = bool(getattr(structure, "is_periodic", True))
    nx, ny, nz = grid_shape
    origins_x = range(nx) if periodic else range(nx - 1)
    origins_y = range(ny) if periodic else range(ny - 1)
    origins_z = range(nz) if periodic else range(nz - 1)

    offsets = [(di, dj, dk) for di in (0, 1) for dj in (0, 1) for dk in (0, 1)]
    counts = {name: 0 for name in SPIN_CATEGORIES}
    total = 0
    for ox in origins_x:
        for oy in origins_y:
            for oz in origins_z:
                cube = np.empty(8, dtype=np.int8)
                for position, (di, dj, dk) in enumerate(offsets):
                    site = int(
                        grid[(ox + di) % nx, (oy + dj) % ny, (oz + dk) % nz]
                    )
                    cube[position] = spin_sign(site)
                counts[cube_category(cube)] += 1
                total += 1

    if total == 0:
        return None
    return SpinClassificationFractions(counts=counts, total=total)


def site_indexing_from_perovskite_build(build) -> PerovskiteSiteIndexing:
    """Convenience wrapper for callers that prefer a function API."""
    return PerovskiteSiteIndexing.from_perovskite_build(build)


def site_indexing_from_generated_order(
    *,
    a_site_count: int,
    b_grid_shape: tuple[int, int, int],
    x_site_count: int,
) -> PerovskiteSiteIndexing:
    """Build site-role indices for structures ordered as A sites, B sites, X sites."""
    a_count = int(a_site_count)
    b_count = int(np.prod(np.asarray(b_grid_shape, dtype=int)))
    x_count = int(x_site_count)
    b_start = a_count
    x_start = b_start + b_count
    return PerovskiteSiteIndexing(
        a_site_indices=np.arange(0, a_count, dtype=int),
        b_site_indices=np.arange(b_start, x_start, dtype=int),
        x_site_indices=np.arange(x_start, x_start + x_count, dtype=int),
        b_grid_shape=tuple(int(value) for value in b_grid_shape),
        grid_to_site=np.arange(b_start, x_start, dtype=int),
    )


def site_indexing_from_magnetic_sublattice(
    structure,
    magnetic_indices: Sequence[int],
    *,
    min_coverage: float = 0.999,
    max_rmsd: float = 1.0,
) -> "PerovskiteSiteIndexing | None":
    """Recover a B-site grid from the magnetic sublattice geometry alone.

    For a loaded structure with no builder provenance, fit its magnetic sites
    (the perovskite B sublattice) onto an ``(nx, ny, nz)`` grid via
    ``match_structure.match_b_sites`` so ``classify_structure_by_cubes`` can run
    without ``generation_parameters``. ``grid_to_site`` is returned in the same
    grid ordering the classifier expects (row-major over the ``(i, j, k)`` cells).

    Returns ``None`` when the magnetic sites do not form a clean perovskite grid
    (non-perovskite ordering, partial coverage), so cube classification is only
    attempted when it is actually meaningful.
    """
    from quick_mag.match_structure import match_b_sites

    mag = np.array(sorted({int(index) for index in magnetic_indices}), dtype=int)
    if mag.size < 8:  # smallest classifiable grid is 2x2x2
        return None

    lattice = np.asarray(structure.lattice, dtype=float)
    positions = np.asarray(structure.cartesian_coords, dtype=float)[mag]
    try:
        b_match = match_b_sites(lattice, positions)
    except Exception:
        return None

    nx, ny, nz = (int(value) for value in b_match.n)
    grid_to_target = np.asarray(b_match.grid_to_target, dtype=int)
    if nx * ny * nz != mag.size or grid_to_target.size != mag.size:
        return None
    if np.any(grid_to_target < 0):
        return None
    if b_match.coverage < min_coverage or b_match.rmsd > max_rmsd:
        return None

    grid_to_site = mag[grid_to_target]  # subset position -> full-structure index
    return PerovskiteSiteIndexing(
        a_site_indices=np.array([], dtype=int),
        b_site_indices=mag,
        x_site_indices=np.array([], dtype=int),
        b_grid_shape=(nx, ny, nz),
        grid_to_site=grid_to_site,
    )


def site_indexing_from_generation_parameters(params, build) -> PerovskiteSiteIndexing:
    """Site-role indices in *structure atom order* for a generated perovskite.

    ``build`` is the canonical (A, B, X) build reconstructed from ``params`` (see
    ``structure.build_from_generation_parameters``). The original structure was
    built as the contiguous block ``[A, B, X_kept]`` and then reordered by
    ``params.permutation`` (identity for the builder path). Applying the inverse
    permutation to each contiguous block yields the structure positions; the B
    block is unaffected by X vacancies, so the B-site grid maps exactly.
    """
    na = len(build.a_sites)
    nb = len(build.b_sites)
    perm = np.asarray(params.permutation, dtype=int).reshape(-1)
    total = int(perm.size)
    if total == 0:
        total = na + nb + len(build.x_sites)
        perm = np.arange(total, dtype=int)
    inverse = np.empty(total, dtype=int)
    inverse[perm] = np.arange(total, dtype=int)

    nx_kept = total - na - nb
    a_indices = inverse[np.arange(0, na, dtype=int)]
    b_indices = inverse[np.arange(na, na + nb, dtype=int)]
    x_indices = inverse[np.arange(na + nb, na + nb + nx_kept, dtype=int)]
    shape = build.octahedra.shape
    grid_shape = (int(shape[0]), int(shape[1]), int(shape[2]))
    return PerovskiteSiteIndexing(
        a_site_indices=a_indices,
        b_site_indices=b_indices,
        x_site_indices=x_indices,
        b_grid_shape=grid_shape,
        grid_to_site=b_indices,
    )


def classify_structure(
    structure,
    *,
    site_indexing: PerovskiteSiteIndexing | None = None,
    b_site_indices: Sequence[int] | None = None,
    x_site_indices: Sequence[int] | None = None,
    b_grid_shape: tuple[int, int, int] | None = None,
    grid_to_site: Sequence[int] | None = None,
    site_oxidation_states: Sequence[float] | None = None,
    moment_eps: float = 1e-3,
    sigma_tol: float = 0.3,
    collinear_tol: float = 0.05,
    noncollinear_tol: float = 0.3,
) -> SpinStructureClassification:
    """Classify common perovskite B-site spin structures.

    The preferred path is to pass ``site_indexing`` from a generated
    ``PerovskiteBuild`` or from the B-site matching code.  The classifier then
    uses the B-site grid directly instead of rediscovering pseudocubic axes from
    distances.
    """
    indexing = _normalize_indexing(
        site_indexing=site_indexing,
        b_site_indices=b_site_indices,
        x_site_indices=x_site_indices,
        b_grid_shape=b_grid_shape,
        grid_to_site=grid_to_site,
    )
    notes: list[str] = []
    if indexing.b_grid_shape is None or indexing.grid_to_site is None:
        raise ValueError(
            "classify_structure requires b_grid_shape and grid_to_site, either "
            "through site_indexing or explicit keyword arguments."
        )

    moments = np.asarray(structure.magnetic_moments, dtype=float)
    if moments.ndim == 1:
        full_moments = np.zeros((len(moments), 3), dtype=float)
        full_moments[:, 2] = moments
        moments = full_moments
    if moments.ndim != 2 or moments.shape[1] != 3:
        raise ValueError("structure.magnetic_moments must have shape (N, 3).")

    grid_shape = tuple(int(value) for value in indexing.b_grid_shape)
    grid_to_site_array = np.asarray(indexing.grid_to_site, dtype=int).reshape(-1)
    expected_b_count = int(np.prod(np.asarray(grid_shape, dtype=int)))
    if len(grid_to_site_array) != expected_b_count:
        raise ValueError(
            "grid_to_site length must match the product of b_grid_shape."
        )

    grid_to_site_array = grid_to_site_array.reshape(grid_shape)
    b_site_indices_array = np.asarray(indexing.b_site_indices, dtype=int)
    if len(b_site_indices_array) < 1:
        return SpinStructureClassification(
            label="unknown",
            site_records=[],
            notes=["insufficient B sites"],
        )

    units, active_mask = _unit_moments(moments, moment_eps=moment_eps)
    inactive_b_sites = [
        int(site_index)
        for site_index in b_site_indices_array
        if site_index < len(active_mask) and not bool(active_mask[site_index])
    ]
    if inactive_b_sites:
        notes.append(
            "zero-or-missing B-site moments: "
            + ", ".join(str(index) for index in inactive_b_sites[:8])
        )

    records: list[SiteSpinClassification] = []
    for grid_index in np.ndindex(grid_shape):
        site_index = int(grid_to_site_array[grid_index])
        record = _classify_site(
            site_index=site_index,
            grid_index=tuple(int(value) for value in grid_index),
            grid_to_site=grid_to_site_array,
            grid_shape=grid_shape,
            units=units,
            active_mask=active_mask,
            sigma_tol=sigma_tol,
            collinear_tol=collinear_tol,
            noncollinear_tol=noncollinear_tol,
        )
        records.append(record)

    label = _overall_label(records)
    if any(value <= 1 for value in grid_shape):
        constrained_axes = [
            AXIS_NAMES[axis] for axis, size in enumerate(grid_shape) if size <= 1
        ]
        notes.append(
            "single-layer periodic axis: " + ", ".join(constrained_axes)
        )
    return SpinStructureClassification(label=label, site_records=records, notes=notes)


def _normalize_indexing(
    *,
    site_indexing: PerovskiteSiteIndexing | None,
    b_site_indices: Sequence[int] | None,
    x_site_indices: Sequence[int] | None,
    b_grid_shape: tuple[int, int, int] | None,
    grid_to_site: Sequence[int] | None,
) -> PerovskiteSiteIndexing:
    if site_indexing is not None:
        return site_indexing
    if b_site_indices is None:
        raise ValueError("b_site_indices are required when site_indexing is omitted.")

    b_indices = np.asarray(b_site_indices, dtype=int)
    grid_sites = np.asarray(grid_to_site, dtype=int) if grid_to_site is not None else b_indices
    return PerovskiteSiteIndexing(
        a_site_indices=np.array([], dtype=int),
        b_site_indices=b_indices,
        x_site_indices=(
            np.asarray(x_site_indices, dtype=int)
            if x_site_indices is not None
            else np.array([], dtype=int)
        ),
        b_grid_shape=b_grid_shape,
        grid_to_site=grid_sites,
    )


def _unit_moments(
    moments: np.ndarray,
    *,
    moment_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(moments, axis=1)
    active = norms > moment_eps
    units = np.zeros_like(moments, dtype=float)
    units[active] = moments[active] / norms[active, None]
    return units, active


def _classify_site(
    *,
    site_index: int,
    grid_index: tuple[int, int, int],
    grid_to_site: np.ndarray,
    grid_shape: tuple[int, int, int],
    units: np.ndarray,
    active_mask: np.ndarray,
    sigma_tol: float,
    collinear_tol: float,
    noncollinear_tol: float,
) -> SiteSpinClassification:
    sigmas: dict[str, float] = {}
    sigmas2: dict[str, float] = {}
    axis_signs: dict[str, int | None] = {}
    notes: list[str] = []
    neighbor_dots: list[float] = []

    if site_index >= len(active_mask) or not bool(active_mask[site_index]):
        return SiteSpinClassification(
            site_index=site_index,
            grid_index=grid_index,
            label="unknown",
            sigmas={axis_name: float("nan") for axis_name in AXIS_NAMES},
            axis_signs={axis_name: None for axis_name in AXIS_NAMES},
            kappa=float("nan"),
            confidence=0.0,
            notes=["zero-moment B site"],
            sigmas2={axis_name: float("nan") for axis_name in AXIS_NAMES},
        )

    for axis, axis_name in enumerate(AXIS_NAMES):
        axis_size = grid_shape[axis]
        if axis_size <= 1:
            sigmas[axis_name] = 1.0
            sigmas2[axis_name] = 1.0
            axis_signs[axis_name] = 1
            notes.append(f"axis-{axis_name}-single-layer")
            continue

        dots: list[float] = []
        for step in (-1, 1):
            neighbor_grid = list(grid_index)
            neighbor_grid[axis] = (neighbor_grid[axis] + step) % axis_size
            neighbor_site = int(grid_to_site[tuple(neighbor_grid)])
            if neighbor_site >= len(active_mask) or not bool(active_mask[neighbor_site]):
                notes.append(f"axis-{axis_name}-inactive-neighbor")
                continue
            dot_value = float(np.dot(units[site_index], units[neighbor_site]))
            dots.append(dot_value)
            neighbor_dots.append(dot_value)

        if dots:
            sigma = float(np.mean(dots))
            sigmas[axis_name] = sigma
            axis_signs[axis_name] = _snap_sigma(sigma, sigma_tol=sigma_tol)
        else:
            sigmas[axis_name] = float("nan")
            axis_signs[axis_name] = None
            notes.append(f"axis-{axis_name}-missing")

        # Distance-2 correlation (block-staggered E-type probe). Only meaningful
        # for an axis that can host an up-up-down-down period-4 modulation.
        sigmas2[axis_name] = _distance2_sigma(
            site_index=site_index,
            grid_index=grid_index,
            grid_to_site=grid_to_site,
            axis=axis,
            axis_size=axis_size,
            units=units,
            active_mask=active_mask,
        )

    if neighbor_dots:
        kappa = float(1.0 - np.mean(np.square(neighbor_dots)))
    else:
        kappa = float("nan")

    e_axes = [
        axis_name
        for axis_name in AXIS_NAMES
        if _snap_sigma(sigmas2[axis_name], sigma_tol=sigma_tol) == -1
    ]
    label = _label_from_axis_signs(axis_signs)
    if e_axes:
        label = "E"
        notes.append("E-modulation axis: " + ", ".join(e_axes))
    elif np.isfinite(kappa) and kappa > noncollinear_tol:
        label = "noncollinear"
    elif (
        label == "unknown"
        and np.isfinite(kappa)
        and kappa < collinear_tol
        and _has_any_intermediate_axis(sigmas, sigma_tol=sigma_tol)
    ):
        label = "canted"

    canted_flag = bool(np.isfinite(kappa) and 0.0 < kappa < 0.1)
    return SiteSpinClassification(
        site_index=site_index,
        grid_index=grid_index,
        label=label,
        sigmas=sigmas,
        axis_signs=axis_signs,
        kappa=kappa,
        confidence=_confidence(sigmas, axis_signs),
        notes=notes,
        canted_flag=canted_flag,
        sigmas2=sigmas2,
    )


def _distance2_sigma(
    *,
    site_index: int,
    grid_index: tuple[int, int, int],
    grid_to_site: np.ndarray,
    axis: int,
    axis_size: int,
    units: np.ndarray,
    active_mask: np.ndarray,
) -> float:
    """Mean dot product with the distance-2 neighbours along ``axis``.

    For a collinear chain this is +1 for FM/Neel/A/C/G and -1 only for the
    up-up-down-down (E-type) modulation. Period-4 ordering needs an axis of at
    least 4 cells; shorter axes can never be E and return +1 (or NaN if a
    neighbour is inactive).
    """
    if axis_size < 4:
        return 1.0
    dots: list[float] = []
    for step in (-2, 2):
        neighbor_grid = list(grid_index)
        neighbor_grid[axis] = (neighbor_grid[axis] + step) % axis_size
        neighbor_site = int(grid_to_site[tuple(neighbor_grid)])
        if neighbor_site >= len(active_mask) or not bool(active_mask[neighbor_site]):
            continue
        dots.append(float(np.dot(units[site_index], units[neighbor_site])))
    if not dots:
        return float("nan")
    return float(np.mean(dots))


def _snap_sigma(value: float, *, sigma_tol: float) -> int | None:
    if not np.isfinite(value):
        return None
    if value > 1.0 - sigma_tol:
        return 1
    if value < -1.0 + sigma_tol:
        return -1
    if abs(value) < sigma_tol:
        return 0
    return None


def _label_from_axis_signs(axis_signs: dict[str, int | None]) -> str:
    signs = [axis_signs[axis_name] for axis_name in AXIS_NAMES]
    if any(sign is None or sign == 0 for sign in signs):
        return "unknown"
    n_positive = sum(1 for sign in signs if sign == 1)
    n_negative = sum(1 for sign in signs if sign == -1)
    if n_positive == 3:
        return "F"
    if n_negative == 3:
        return "G"
    if n_positive == 2 and n_negative == 1:
        return "A"
    if n_positive == 1 and n_negative == 2:
        return "C"
    return "unknown"


def _has_any_intermediate_axis(
    sigmas: dict[str, float],
    *,
    sigma_tol: float,
) -> bool:
    for value in sigmas.values():
        if not np.isfinite(value):
            continue
        snapped = _snap_sigma(value, sigma_tol=sigma_tol)
        if snapped is None:
            return True
    return False


def _confidence(
    sigmas: dict[str, float],
    axis_signs: dict[str, int | None],
) -> float:
    scores: list[float] = []
    for axis_name, value in sigmas.items():
        sign = axis_signs[axis_name]
        if sign is None or not np.isfinite(value):
            continue
        target = float(sign)
        scores.append(max(0.0, 1.0 - abs(value - target)))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _overall_label(records: Sequence[SiteSpinClassification]) -> str:
    """Dominant per-cell category (most common per-site label).

    Ties are broken by the canonical ``SPIN_CATEGORIES`` order so the result is
    deterministic. This is the label used to color the energy-landscape scatter.
    """
    if not records:
        return "Other"
    counts = Counter(spin_category(record.label) for record in records)
    return max(
        SPIN_CATEGORIES,
        key=lambda name: (counts.get(name, 0), -SPIN_CATEGORIES.index(name)),
    )
