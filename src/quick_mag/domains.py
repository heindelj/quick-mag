"""Domains: stacking several perovskite blocks into one structure.

A *domain* is one perovskite block -- a formula, a composition and an extent --
and a structure is an ordered stack of them along one lattice direction, the
*stacking axis*. The stack is a single corner-sharing octahedral grid: the
octahedron count along the stacking axis is the sum over domains, the two
in-plane counts are shared by every domain (they have to match, see
:func:`matching_in_plane_cells`), and the Glazer tilt system is a property of
the whole grid rather than of any one domain.

Where two domains meet, the grid has an AO-type plane: the A sites at that
plane index plus the bridging X atoms of the octahedra just below it. Those
atoms are the *interface* and they belong to exactly one of the two domains --
:attr:`DomainSpec.interface_from_previous` says which. With it False (the
default) a domain's low interface plane carries its own composition, so a
LaFeO3 block grown on SrTiO3 starts with a LaO layer; with it True the SrO
layer of the block below is the interface and the LaFeO3 block starts at FeO2.

The whole assignment is a function of the canonical grid key of a site (see
``perovskite_builder.canonical_site_keys``) and nothing else, which is what lets
defects keep addressing the combined grid exactly as they did a single block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from quick_mag.perovskite_builder import SiteKey, periodic_axes

FORMULA_UNIT_FACTORS: Dict[str, int] = {
    "perovskite": 1,
    "double": 2,
    "quadruple": 2,
    "dq": 2,
    "high_entropy": 1,
}

AXIS_NAMES: Tuple[str, str, str] = ("a", "b", "c")


def formula_unit_factor(formula_mode: str) -> int:
    """Octahedra per primitive cell edge for a formula: 2 for the ordered modes."""
    return FORMULA_UNIT_FACTORS.get(str(formula_mode), 1)


@dataclass
class DomainSpec:
    """One perovskite block of a stacked structure.

    ``n_cells`` counts primitive cells per axis, as the builder's supercell
    boxes do; the octahedron count is ``n_cells * unit_factor`` (see
    :meth:`oct_counts`). ``lattice`` is the single-octahedron edge length per
    axis. In-plane entries of both are constrained to match the rest of the stack;
    only the stacking-axis entries are the domain's own.
    """

    formula_mode: str = "perovskite"
    a_site_element: str = "La"
    b_site_element: str = "Fe"
    x_site_element: str = "O"
    a2_site_element: str = "Sr"
    b2_site_element: str = "Co"
    high_entropy_a_sites: List[Tuple[str, float]] = field(
        default_factory=lambda: [("La", 1.0)]
    )
    high_entropy_b_sites: List[Tuple[str, float]] = field(
        default_factory=lambda: [("Fe", 1.0)]
    )
    high_entropy_x_sites: List[Tuple[str, float]] = field(
        default_factory=lambda: [("O", 1.0)]
    )
    high_entropy_sample_index: int = 0
    high_entropy_seed: int = 0
    n_cells: Tuple[int, int, int] = (3, 3, 3)
    lattice: Tuple[float, float, float] = (4.0, 4.0, 4.0)
    # Whether the AO plane on this domain's low side takes the previous
    # domain's composition (True) or this domain's own (False). For domain 0 it
    # refers to the last domain, across the periodic wrap, and is moot when the
    # stacking axis is finite.
    interface_from_previous: bool = False
    name: str = ""

    def __post_init__(self) -> None:
        self.formula_mode = str(self.formula_mode)
        self.n_cells = tuple(max(1, int(value)) for value in self.n_cells)  # type: ignore[assignment]
        self.lattice = tuple(float(value) for value in self.lattice)  # type: ignore[assignment]
        self.high_entropy_a_sites = [(str(e), float(f)) for e, f in self.high_entropy_a_sites]
        self.high_entropy_b_sites = [(str(e), float(f)) for e, f in self.high_entropy_b_sites]
        self.high_entropy_x_sites = [(str(e), float(f)) for e, f in self.high_entropy_x_sites]
        self.interface_from_previous = bool(self.interface_from_previous)

    @property
    def unit_factor(self) -> int:
        return formula_unit_factor(self.formula_mode)

    def oct_counts(self) -> Tuple[int, int, int]:
        """Octahedra per axis: primitive cells scaled by the formula's unit factor."""
        factor = self.unit_factor
        return tuple(int(value) * factor for value in self.n_cells)  # type: ignore[return-value]

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            formula_mode=self.formula_mode,
            a_site_element=self.a_site_element,
            b_site_element=self.b_site_element,
            x_site_element=self.x_site_element,
            a2_site_element=self.a2_site_element,
            b2_site_element=self.b2_site_element,
            high_entropy_a_sites=list(self.high_entropy_a_sites),
            high_entropy_b_sites=list(self.high_entropy_b_sites),
            high_entropy_x_sites=list(self.high_entropy_x_sites),
            high_entropy_sample_index=int(self.high_entropy_sample_index),
            high_entropy_seed=int(self.high_entropy_seed),
            n_cells=tuple(self.n_cells),
            lattice=tuple(self.lattice),
            interface_from_previous=bool(self.interface_from_previous),
            name=self.name,
        )

    def signature(self) -> Tuple[object, ...]:
        """Hashable content key, for change detection."""
        return (
            self.formula_mode,
            self.a_site_element.strip(),
            self.b_site_element.strip(),
            self.x_site_element.strip(),
            self.a2_site_element.strip(),
            self.b2_site_element.strip(),
            tuple(self.high_entropy_a_sites),
            tuple(self.high_entropy_b_sites),
            tuple(self.high_entropy_x_sites),
            int(self.high_entropy_sample_index),
            int(self.high_entropy_seed),
            tuple(self.n_cells),
            tuple(round(value, 6) for value in self.lattice),
            bool(self.interface_from_previous),
        )


def coerce_domain(value) -> DomainSpec:
    if isinstance(value, DomainSpec):
        return value
    if isinstance(value, dict):
        return DomainSpec(**value)
    raise TypeError(f"cannot interpret {value!r} as a DomainSpec")


# ---------------------------------------------------------------------------
# Stack geometry
# ---------------------------------------------------------------------------


def in_plane_axes(stacking_axis: int) -> Tuple[int, int]:
    """The two axes perpendicular to ``stacking_axis``, in order."""
    axis = int(stacking_axis)
    if axis not in (0, 1, 2):
        raise ValueError("stacking_axis must be 0, 1 or 2.")
    return tuple(other for other in range(3) if other != axis)  # type: ignore[return-value]


def stack_oct_counts(domains: Sequence[DomainSpec], stacking_axis: int) -> Tuple[int, int, int]:
    """Octahedra per axis of the combined grid.

    The stacking axis sums over domains; the in-plane axes are taken from the
    first domain, after :func:`validate_stack` has checked they all agree.
    """
    if not domains:
        raise ValueError("a structure needs at least one domain.")
    counts = list(domains[0].oct_counts())
    axis = int(stacking_axis)
    counts[axis] = sum(domain.oct_counts()[axis] for domain in domains)
    return tuple(counts)  # type: ignore[return-value]


def domain_offsets(domains: Sequence[DomainSpec], stacking_axis: int) -> List[int]:
    """First octahedron layer of each domain along the stacking axis."""
    offsets: List[int] = []
    total = 0
    for domain in domains:
        offsets.append(total)
        total += domain.oct_counts()[int(stacking_axis)]
    return offsets


def stack_half_lengths(
    domains: Sequence[DomainSpec], stacking_axis: int
) -> Tuple[Any, Any, Any]:
    """``center_to_vertex_distance_*`` arguments for ``build_perovskite``.

    Scalars in plane; one entry per octahedron layer along the stacking axis, so
    each domain keeps its own spacing there.
    """
    axis = int(stacking_axis)
    halves: List[Any] = [0.5 * value for value in domains[0].lattice]
    per_layer: List[float] = []
    for domain in domains:
        per_layer.extend([0.5 * domain.lattice[axis]] * domain.oct_counts()[axis])
    halves[axis] = np.asarray(per_layer, dtype=np.float64)
    return halves[0], halves[1], halves[2]


def stack_lattice(domains: Sequence[DomainSpec], stacking_axis: int) -> np.ndarray:
    """Diagonal supercell lattice of the stack (octahedron count times edge)."""
    counts = stack_oct_counts(domains, stacking_axis)
    axis = int(stacking_axis)
    edges = [counts[index] * domains[0].lattice[index] for index in range(3)]
    edges[axis] = sum(domain.oct_counts()[axis] * domain.lattice[axis] for domain in domains)
    return np.diag(np.asarray(edges, dtype=np.float64))


# ---------------------------------------------------------------------------
# Matching rules
# ---------------------------------------------------------------------------


def matching_in_plane_cells(
    reference: DomainSpec, formula_mode: str, stacking_axis: int
) -> Tuple[int, int]:
    """In-plane primitive-cell counts a ``formula_mode`` domain needs to sit on ``reference``.

    The octahedron counts have to agree; a formula with unit factor 2 (the
    ordered double/quadruple modes) therefore needs an even octahedron count in
    plane, so a 3x3 single perovskite has no matching double perovskite.
    Raises ``ValueError`` with the reason when there is none.
    """
    factor = formula_unit_factor(formula_mode)
    first, second = in_plane_axes(stacking_axis)
    octs = reference.oct_counts()
    cells = []
    for axis in (first, second):
        if octs[axis] % factor != 0:
            raise ValueError(
                f"{formula_mode} needs an even octahedron count in plane, but the "
                f"neighbouring domain is {octs[first]}x{octs[second]} along "
                f"{AXIS_NAMES[first]}x{AXIS_NAMES[second]}."
            )
        cells.append(octs[axis] // factor)
    return cells[0], cells[1]


def conform_domain_to_stack(
    domain: DomainSpec, reference: DomainSpec, stacking_axis: int
) -> DomainSpec:
    """``domain`` with its in-plane size and lattice constants taken from ``reference``.

    Its stacking-axis extent and spacing are left alone. Raises ``ValueError``
    when the formula cannot match the reference's in-plane size.
    """
    first, second = in_plane_axes(stacking_axis)
    cells_first, cells_second = matching_in_plane_cells(
        reference, domain.formula_mode, stacking_axis
    )
    n_cells = list(domain.n_cells)
    n_cells[first], n_cells[second] = cells_first, cells_second
    lattice = list(domain.lattice)
    lattice[first], lattice[second] = reference.lattice[first], reference.lattice[second]
    domain.n_cells = tuple(n_cells)  # type: ignore[assignment]
    domain.lattice = tuple(lattice)  # type: ignore[assignment]
    return domain


def validate_stack(domains: Sequence[DomainSpec], stacking_axis: int) -> List[str]:
    """Reasons the stack is inconsistent; empty when it can be built."""
    problems: List[str] = []
    if not domains:
        return ["a structure needs at least one domain."]
    first, second = in_plane_axes(stacking_axis)
    reference = domains[0]
    ref_octs = reference.oct_counts()
    for index, domain in enumerate(domains[1:], start=1):
        octs = domain.oct_counts()
        for axis in (first, second):
            if octs[axis] != ref_octs[axis]:
                problems.append(
                    f"domain {index + 1} spans {octs[axis]} octahedra along "
                    f"{AXIS_NAMES[axis]} but domain 1 spans {ref_octs[axis]}."
                )
            if abs(domain.lattice[axis] - reference.lattice[axis]) > 1e-9:
                problems.append(
                    f"domain {index + 1} has {AXIS_NAMES[axis]} = {domain.lattice[axis]:.4f} "
                    f"but domain 1 has {reference.lattice[axis]:.4f}."
                )
    return problems


# ---------------------------------------------------------------------------
# Site -> domain assignment
# ---------------------------------------------------------------------------


class DomainAssigner:
    """Maps canonical grid keys of the combined build to domain indices."""

    def __init__(self, domains: Sequence[DomainSpec], stacking_axis: int, periodic) -> None:
        self.domains = list(domains)
        self.axis = int(stacking_axis)
        self.periodic = periodic_axes(periodic)[self.axis]
        self.offsets = domain_offsets(self.domains, self.axis)
        self.total = stack_oct_counts(self.domains, self.axis)[self.axis]
        # layer -> domain, one entry per octahedron layer along the axis.
        self._layer_domain = np.empty(self.total, dtype=np.int64)
        for index, domain in enumerate(self.domains):
            start = self.offsets[index]
            self._layer_domain[start : start + domain.oct_counts()[self.axis]] = index

    def domain_of_layer(self, layer: int) -> int:
        """Domain owning the octahedron (BO2) layer at ``layer``."""
        return int(self._layer_domain[int(layer) % self.total])

    def domain_of_plane(self, plane: int) -> int:
        """Domain owning the AO plane at index ``plane`` along the axis.

        Plane ``m`` sits between octahedron layers ``m - 1`` and ``m``. Inside a
        domain it is that domain's; at a domain's low face it is the domain's
        unless ``interface_from_previous`` hands it to the block below.
        """
        plane = int(plane)
        if plane >= self.total:
            if self.periodic:
                plane -= self.total
            else:
                # The closing plane of a finite axis belongs to the top domain.
                return len(self.domains) - 1
        upper = self.domain_of_layer(plane)
        if plane != self.offsets[upper]:
            return upper
        # A domain boundary.
        if upper == 0 and not self.periodic:
            return 0
        if self.domains[upper].interface_from_previous:
            return (upper - 1) % len(self.domains)
        return upper

    def domain_of_key(self, key: SiteKey) -> int:
        position = (key.i, key.j, key.k)[self.axis]
        if key.role == "B":
            return self.domain_of_layer(position)
        if key.role == "A":
            return self.domain_of_plane(position)
        if key.role == "X":
            vertex_axis = key.vertex // 2
            if vertex_axis != self.axis:
                # In-plane oxygen: part of this layer's BO2 sheet.
                return self.domain_of_layer(position)
            if key.vertex % 2 == 0:
                # The +axis vertex is the bridging oxygen of the AO plane above.
                return self.domain_of_plane(position + 1)
            # The -axis vertex only exists canonically on a finite low face.
            return self.domain_of_plane(position)
        return self.domain_of_layer(position)

    def is_interface_key(self, key: SiteKey) -> bool:
        """Whether ``key`` lies in an AO plane shared by two domains."""
        if len(self.domains) < 2:
            return False
        position = (key.i, key.j, key.k)[self.axis]
        if key.role == "A":
            plane = position
        elif key.role == "X" and key.vertex // 2 == self.axis:
            plane = position + 1 if key.vertex % 2 == 0 else position
        else:
            return False
        if plane >= self.total:
            if not self.periodic:
                return False
            plane -= self.total
        if plane == 0:
            return self.periodic
        return plane in self.offsets

    def domains_of_keys(self, keys: Sequence[SiteKey]) -> np.ndarray:
        return np.asarray([self.domain_of_key(key) for key in keys], dtype=np.int64)
