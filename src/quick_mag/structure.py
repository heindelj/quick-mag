from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np


# Leading element symbol of an atomic label, e.g. "Fe" from "Fe" or "Fe2+".
_ELEMENT_SYMBOL_RE = re.compile(r"([A-Z][a-z]?)")


class Neighbor(NamedTuple):
    """A periodic neighbor of a site (drop-in for pymatgen's neighbor objects).

    ``coords`` is the cartesian position of the (possibly periodic-image)
    neighbor; ``index`` is the neighbor's site index in the original structure.
    """

    index: int
    coords: np.ndarray
    nn_distance: float
    symbol: str

from quick_mag.perovskite_builder import (
    active_tilt_axes,
    build_perovskite,
    canonicalize_glazer_tilt_angles_deg,
)


GLAZER_TILT_SYSTEMS = [
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
class PerovskiteGenerationParameters:
    """Provenance for a generated perovskite structure.

    Stores exactly the inputs needed to (a) deterministically re-call
    ``build_perovskite(...)`` and (b) map the canonical (A, B, X) build order onto
    the structure's atom order. ``ChemicalStructure`` objects loaded from a file
    carry ``generation_parameters=None`` instead.
    """

    # --- build_perovskite(...) inputs (deterministic regeneration) ---
    center: np.ndarray
    n_oct_x: int
    n_oct_y: int
    n_oct_z: int
    center_to_vertex_distance_x: float
    center_to_vertex_distance_y: float
    center_to_vertex_distance_z: float
    tilt_system: str
    tilt_angle_x_deg: float
    tilt_angle_y_deg: float
    tilt_angle_z_deg: float
    periodic: bool

    # --- composition ---
    a_site_element: str
    b_site_element: str
    x_site_element: str
    formula_mode: str = "perovskite"
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
    # Selects which high-entropy occupancy realization to sample. Distinct indices
    # give distinct-but-reproducible draws (respecting the site weights); 0 is the
    # canonical single-sample build.
    high_entropy_sample_index: int = 0

    # --- spin pattern (builder path; random generator uses "None") ---
    spin_pattern: str = "None"
    spin_moment_magnitude: float = 0.0

    # --- X vacancy info (random path; builder path leaves these at 0) ---
    x_vacancy_fraction: float = 0.0
    x_removed_count: int = 0
    # Indices into the canonical (pre-shuffle, fully occupied) X-site list removed.
    removed_x_site_indices: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )

    # --- atom-order mapping from canonical build order to structure order ---
    # site_roles[p] in {"A", "B", "X"} for structure position p.
    site_roles: List[str] = field(default_factory=list)
    # permutation[p] = canonical (post-vacancy) index that landed at structure
    # position p. Identity for the unshuffled builder path.
    permutation: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.int64)
    )

    # --- provenance / origin offset ---
    # cell_origin is subtracted from build coords to get structure coords
    # (builder path); zeros for the random generator.
    cell_origin: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    source: str = "perovskite_builder"

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=np.float64).reshape(3)
        self.cell_origin = np.asarray(self.cell_origin, dtype=np.float64).reshape(3)
        self.removed_x_site_indices = np.asarray(
            self.removed_x_site_indices, dtype=np.int64
        ).reshape(-1)
        self.permutation = np.asarray(self.permutation, dtype=np.int64).reshape(-1)
        self.site_roles = list(self.site_roles)
        self.tilt_system = str(self.tilt_system)
        self.formula_mode = str(self.formula_mode)
        self.high_entropy_sample_index = int(self.high_entropy_sample_index)
        self.high_entropy_a_sites = [
            (str(element), float(fraction))
            for element, fraction in self.high_entropy_a_sites
        ]
        self.high_entropy_b_sites = [
            (str(element), float(fraction))
            for element, fraction in self.high_entropy_b_sites
        ]
        self.high_entropy_x_sites = [
            (str(element), float(fraction))
            for element, fraction in self.high_entropy_x_sites
        ]

    def build_kwargs(self) -> Dict[str, Any]:
        """Exact keyword arguments for ``build_perovskite(...)``."""
        return dict(
            center=self.center,
            n_oct_x=self.n_oct_x,
            n_oct_y=self.n_oct_y,
            n_oct_z=self.n_oct_z,
            center_to_vertex_distance_x=self.center_to_vertex_distance_x,
            center_to_vertex_distance_y=self.center_to_vertex_distance_y,
            center_to_vertex_distance_z=self.center_to_vertex_distance_z,
            tilt_system=self.tilt_system,
            tilt_angle_x_deg=self.tilt_angle_x_deg,
            tilt_angle_y_deg=self.tilt_angle_y_deg,
            tilt_angle_z_deg=self.tilt_angle_z_deg,
            periodic=self.periodic,
        )


@dataclass
class SavedSpinConfiguration:
    """A magnetic configuration saved onto a structure (one of possibly several).

    ``magnetic_moments`` is the full (N, 3) per-atom moment array. ``collinear``
    records the solver mode that produced it and drives the export width
    (1 value/atom when collinear, else 3).
    """

    magnetic_moments: np.ndarray
    energy: float = 0.0
    magnetization: float = 0.0
    classification: str = ""
    collinear: bool = True

    def __post_init__(self) -> None:
        self.magnetic_moments = np.asarray(
            self.magnetic_moments, dtype=np.float64
        ).reshape(-1, 3)


@dataclass(eq=False)
class ChemicalStructure:
    name: str
    lattice: np.ndarray
    cartesian_coords: np.ndarray
    atomic_labels: List[str]
    magnetic_moments: np.ndarray
    is_periodic: bool = True
    generation_parameters: Optional[PerovskiteGenerationParameters] = None
    spin_configurations: List["SavedSpinConfiguration"] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.lattice = np.asarray(self.lattice, dtype=np.float64)
        self.cartesian_coords = np.asarray(self.cartesian_coords, dtype=np.float64)
        self.atomic_labels = list(self.atomic_labels)
        self.magnetic_moments = np.asarray(self.magnetic_moments, dtype=np.float64)

        if self.lattice.shape != (3, 3):
            raise ValueError("lattice must be a (3, 3) matrix.")
        if self.cartesian_coords.ndim != 2 or self.cartesian_coords.shape[1] != 3:
            raise ValueError("cartesian_coords must be an (N, 3) array.")
        if len(self.atomic_labels) != len(self.cartesian_coords):
            raise ValueError("atomic_labels length must match the number of sites.")
        if self.magnetic_moments.shape != self.cartesian_coords.shape:
            raise ValueError("magnetic_moments must be an (N, 3) array.")

    @property
    def atom_count(self) -> int:
        return len(self.atomic_labels)

    @property
    def fractional_coords(self) -> np.ndarray:
        return np.linalg.solve(self.lattice.T, self.cartesian_coords.T).T

    def grouped_coords(self, use_cartesian: bool = True) -> Dict[str, np.ndarray]:
        coords = self.cartesian_coords if use_cartesian else self.fractional_coords
        grouped: Dict[str, List[np.ndarray]] = {}
        for label, coord in zip(self.atomic_labels, coords):
            grouped.setdefault(label, []).append(coord)
        normalized: Dict[str, np.ndarray] = {}
        for label, site_coords in grouped.items():
            coord_array = np.asarray(site_coords, dtype=np.float64)
            normalized[label] = coord_array.reshape(-1, 3)
        return normalized

    def element_symbols(self) -> List[str]:
        """Per-site element symbols with any oxidation-state suffix stripped.

        Mirrors pymatgen's ``site.specie.symbol`` (e.g. ``"Fe"`` from ``"Fe2+"``).
        """
        symbols: List[str] = []
        for label in self.atomic_labels:
            match = _ELEMENT_SYMBOL_RE.match(label)
            symbols.append(match.group(1) if match else label)
        return symbols

    def neighbors(self, index: int, cutoff: float) -> List[Neighbor]:
        """Neighbors of site ``index`` within ``cutoff`` (Angstrom).

        Periodic-aware replacement for pymatgen's ``Structure.get_neighbors``:
        enumerates lattice images out to the cutoff and returns every atom image
        within range (excluding the site itself). Non-periodic structures use a
        single image.
        """
        center = self.cartesian_coords[index]
        coords = self.cartesian_coords
        symbols = self.element_symbols()

        if self.is_periodic:
            translations = self._lattice_image_translations(cutoff)
        else:
            translations = np.zeros((1, 3), dtype=np.float64)

        result: List[Neighbor] = []
        for j in range(len(coords)):
            positions = coords[j] + translations
            distances = np.linalg.norm(positions - center, axis=1)
            within = (distances > 1e-8) & (distances <= cutoff)
            for t in np.nonzero(within)[0]:
                result.append(
                    Neighbor(
                        index=j,
                        coords=positions[t],
                        nn_distance=float(distances[t]),
                        symbol=symbols[j],
                    )
                )
        return result

    def _lattice_image_translations(self, cutoff: float) -> np.ndarray:
        """Cartesian translation vectors for every lattice image within ``cutoff``."""
        lattice = self.lattice
        volume = abs(float(np.linalg.det(lattice)))
        counts = []
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            face_area = float(np.linalg.norm(np.cross(lattice[j], lattice[k])))
            spacing = volume / face_area if face_area > 0 else np.inf
            counts.append(int(np.ceil(cutoff / spacing)) if spacing > 0 else 0)

        offsets = [
            a * lattice[0] + b * lattice[1] + c * lattice[2]
            for a in range(-counts[0], counts[0] + 1)
            for b in range(-counts[1], counts[1] + 1)
            for c in range(-counts[2], counts[2] + 1)
        ]
        return np.asarray(offsets, dtype=np.float64)

    def equivalent_atoms(self, cutoff: float = 3.0) -> np.ndarray:
        """Approximate symmetry orbits: an orbit id per site.

        Pure-python stand-in for spglib/pymatgen's ``equivalent_atoms``. Sites are
        grouped by a local fingerprint of (element, sorted (rounded neighbor
        distance, neighbor element) pairs); sites sharing a fingerprint share an
        orbit id. This is approximate but pymatgen-free and web-safe.
        """
        symbols = self.element_symbols()
        orbit_of: Dict[object, int] = {}
        result = np.empty(self.atom_count, dtype=np.int64)
        for i in range(self.atom_count):
            fingerprint = tuple(
                sorted(
                    (round(neighbor.nn_distance, 2), neighbor.symbol)
                    for neighbor in self.neighbors(i, cutoff)
                )
            )
            key = (symbols[i], fingerprint)
            if key not in orbit_of:
                orbit_of[key] = len(orbit_of)
            result[i] = orbit_of[key]
        return result

    @classmethod
    def with_zero_magnetic_moments(
        cls,
        name: str,
        lattice: np.ndarray,
        cartesian_coords: np.ndarray,
        atomic_labels: List[str],
        is_periodic: bool = True,
        generation_parameters: Optional[PerovskiteGenerationParameters] = None,
    ) -> "ChemicalStructure":
        coords = np.asarray(cartesian_coords, dtype=np.float64)
        return cls(
            name=name,
            lattice=np.asarray(lattice, dtype=np.float64),
            cartesian_coords=coords,
            atomic_labels=list(atomic_labels),
            magnetic_moments=np.zeros_like(coords, dtype=np.float64),
            is_periodic=is_periodic,
            generation_parameters=generation_parameters,
        )


@dataclass
class StructureGroup:
    name: str
    is_generated: bool
    structures: List[ChemicalStructure] = field(default_factory=list)

    def structure_names(self) -> List[str]:
        return [structure.name for structure in self.structures]

    def add_structure(self, structure: ChemicalStructure) -> int:
        self.structures.append(structure)
        return len(self.structures) - 1

    def replace_structure(self, index: int, structure: ChemicalStructure) -> None:
        self.structures[index] = structure

    def get_structure(self, index: int) -> ChemicalStructure | None:
        if 0 <= index < len(self.structures):
            return self.structures[index]
        return None

    def find_structure_index(self, name: str) -> int:
        for index, structure in enumerate(self.structures):
            if structure.name == name:
                return index
        return -1


def match_structure_to_build_parameters(structure: ChemicalStructure):
    pass


def build_from_generation_parameters(params: PerovskiteGenerationParameters):
    """Deterministically reconstruct the canonical ``PerovskiteBuild``.

    The returned build is in canonical, fully-occupied (A, B, X) order, identical
    to the one originally produced during generation. To express its site indices
    in a structure's (possibly shuffled, vacancy-reduced) atom order, pair this
    with ``site_indexing_from_generation_parameters`` in ``classify_spin_structure``.
    """
    return build_perovskite(**params.build_kwargs())


def generate_random_test_perovskite(
    rng: np.random.Generator | None = None,
    *,
    min_supercell_size: int = 1,
    max_supercell_size: int = 4,
    periodic: bool = True,
    include_tilts: bool = True,
    shuffle_atoms: bool = True,
    x_vacancy_fraction: float = 0.0,
) -> tuple[ChemicalStructure, Dict[str, Any]]:
    if min_supercell_size < 1:
        raise ValueError("min_supercell_size must be at least 1.")
    if max_supercell_size < min_supercell_size:
        raise ValueError("max_supercell_size must be greater than or equal to min_supercell_size.")
    if not 0.0 <= x_vacancy_fraction <= 0.8:
        raise ValueError("x_vacancy_fraction must be between 0.0 and 0.8.")

    rng = np.random.default_rng() if rng is None else rng

    supercell_shape = rng.integers(
        min_supercell_size,
        max_supercell_size + 1,
        size=3,
        endpoint=False,
    )
    supercell_shape = np.asarray(supercell_shape, dtype=np.int64)
    replications = supercell_shape - 1

    lattice_abc = rng.uniform(5.5, 6.5, size=3)
    a_site_element = str(rng.choice(np.array(["Cs", "Rb", "K", "Ba"], dtype=object)))
    b_site_element = str(rng.choice(np.array(["Pb", "Sn", "Ti", "Zr"], dtype=object)))
    x_site_element = str(rng.choice(np.array(["I", "Br", "Cl", "O"], dtype=object)))
    tilt_system = "a0a0a0"
    if include_tilts and np.all(supercell_shape >= 2):
        tilt_system = str(rng.choice(np.array(GLAZER_TILT_SYSTEMS, dtype=object)))
    active_x, active_y, active_z = active_tilt_axes(tilt_system)
    raw_tilt_angles_deg = (
        float(rng.uniform(-14.0, 14.0)) if active_x else 0.0,
        float(rng.uniform(-14.0, 14.0)) if active_y else 0.0,
        float(rng.uniform(-14.0, 14.0)) if active_z else 0.0,
    )
    tilt_angles_deg = canonicalize_glazer_tilt_angles_deg(
        tilt_system,
        raw_tilt_angles_deg[0],
        raw_tilt_angles_deg[1],
        raw_tilt_angles_deg[2],
    )
    half_edge_lengths = 0.5 * lattice_abc
    center = half_edge_lengths.copy()

    build = build_perovskite(
        center=center,
        n_oct_x=int(replications[0]),
        n_oct_y=int(replications[1]),
        n_oct_z=int(replications[2]),
        center_to_vertex_distance_x=float(half_edge_lengths[0]),
        center_to_vertex_distance_y=float(half_edge_lengths[1]),
        center_to_vertex_distance_z=float(half_edge_lengths[2]),
        tilt_system=tilt_system,
        tilt_angle_x_deg=tilt_angles_deg[0],
        tilt_angle_y_deg=tilt_angles_deg[1],
        tilt_angle_z_deg=tilt_angles_deg[2],
        periodic=periodic,
    )

    x_sites = np.asarray(build.x_sites, dtype=np.float64)
    original_x_site_count = len(x_sites)
    x_removed_count = 0
    removed_x_indices = np.zeros(0, dtype=np.int64)
    if x_vacancy_fraction > 0.0 and original_x_site_count > 0:
        x_removed_count = min(
            int(np.floor(original_x_site_count * x_vacancy_fraction)),
            int(np.floor(original_x_site_count * 0.8)),
        )
        if x_removed_count > 0:
            retained_x_mask = np.ones(original_x_site_count, dtype=bool)
            removed_x_indices = rng.choice(
                original_x_site_count,
                size=x_removed_count,
                replace=False,
            )
            retained_x_mask[removed_x_indices] = False
            x_sites = x_sites[retained_x_mask]

    cartesian_coords = np.vstack((build.a_sites, build.b_sites, x_sites)).astype(np.float64)
    atomic_labels = np.array(
        [a_site_element] * len(build.a_sites)
        + [b_site_element] * len(build.b_sites)
        + [x_site_element] * len(x_sites),
        dtype=object,
    )
    site_roles = np.array(
        ["A"] * len(build.a_sites) + ["B"] * len(build.b_sites) + ["X"] * len(x_sites),
        dtype=object,
    )
    permutation = np.arange(len(cartesian_coords), dtype=int)
    if shuffle_atoms:
        permutation = rng.permutation(len(cartesian_coords))
        cartesian_coords = cartesian_coords[permutation]
        atomic_labels = atomic_labels[permutation]
        site_roles = site_roles[permutation]
    structure_name = (
        "random_perovskite_"
        f"{int(supercell_shape[0])}x{int(supercell_shape[1])}x{int(supercell_shape[2])}"
    )
    generation_parameters = PerovskiteGenerationParameters(
        center=center,
        n_oct_x=int(replications[0]),
        n_oct_y=int(replications[1]),
        n_oct_z=int(replications[2]),
        center_to_vertex_distance_x=float(half_edge_lengths[0]),
        center_to_vertex_distance_y=float(half_edge_lengths[1]),
        center_to_vertex_distance_z=float(half_edge_lengths[2]),
        tilt_system=tilt_system,
        tilt_angle_x_deg=tilt_angles_deg[0],
        tilt_angle_y_deg=tilt_angles_deg[1],
        tilt_angle_z_deg=tilt_angles_deg[2],
        periodic=periodic,
        a_site_element=a_site_element,
        b_site_element=b_site_element,
        x_site_element=x_site_element,
        spin_pattern="None",
        spin_moment_magnitude=0.0,
        x_vacancy_fraction=float(x_vacancy_fraction),
        x_removed_count=int(x_removed_count),
        removed_x_site_indices=np.asarray(removed_x_indices, dtype=np.int64).copy(),
        site_roles=site_roles.tolist(),
        permutation=np.asarray(permutation, dtype=np.int64).copy(),
        cell_origin=np.zeros(3, dtype=np.float64),
        source="random_generator",
    )
    structure = ChemicalStructure.with_zero_magnetic_moments(
        name=structure_name,
        lattice=np.diag(supercell_shape * lattice_abc).astype(np.float64),
        cartesian_coords=cartesian_coords,
        atomic_labels=atomic_labels.tolist(),
        is_periodic=periodic,
        generation_parameters=generation_parameters,
    )
    metadata: Dict[str, Any] = {
        "name": structure_name,
        "supercell_shape": tuple(int(value) for value in supercell_shape),
        "replications": tuple(int(value) for value in replications),
        "lattice_abc": tuple(float(value) for value in lattice_abc),
        "site_elements": (a_site_element, b_site_element, x_site_element),
        "tilt_system": tilt_system,
        "tilt_angles_deg": tilt_angles_deg,
        "raw_tilt_angles_deg": raw_tilt_angles_deg,
        "x_vacancy_fraction": float(x_vacancy_fraction),
        "x_removed_count": int(x_removed_count),
        "x_retained_count": len(x_sites),
        "original_x_site_count": int(original_x_site_count),
        "periodic": periodic,
        "site_roles": site_roles.tolist(),
        "permutation": permutation.tolist(),
        "atom_count": structure.atom_count,
        "a_site_count": len(build.a_sites),
        "b_site_count": len(build.b_sites),
        "x_site_count": len(x_sites),
    }
    return structure, metadata


if __name__ == "__main__":
    from quick_mag.match_structure import ALL_GLAZER_SYSTEMS, fit_x_sites, BSiteMatch

    def run_demo_pass(
        *,
        title: str,
        rng: np.random.Generator,
        with_vacancies: bool,
        n_cases: int = 5,
    ) -> None:
        print(title)
        for case_index in range(n_cases):
            x_vacancy_fraction = 0.0
            if with_vacancies:
                x_vacancy_fraction = float(rng.uniform(0.0, 0.25))

            structure, metadata = generate_random_test_perovskite(
                rng=rng,
                min_supercell_size=2,
                max_supercell_size=4,
                periodic=True,
                include_tilts=True,
                shuffle_atoms=True,
                x_vacancy_fraction=x_vacancy_fraction,
            )
            site_roles = np.asarray(metadata["site_roles"], dtype=object)
            true_a_indices = np.flatnonzero(site_roles == "A")
            true_b_indices = np.flatnonzero(site_roles == "B")
            true_x_indices = np.flatnonzero(site_roles == "X")
            true_b_frac = structure.fractional_coords[true_b_indices] % 1.0
            b_order = np.lexsort(
                (true_b_frac[:, 2], true_b_frac[:, 1], true_b_frac[:, 0])
            )
            ordered_b_indices = true_b_indices[b_order]
            ordered_b_frac = true_b_frac[b_order]
            ordered_b_cart = structure.cartesian_coords[ordered_b_indices]

            b_match = BSiteMatch(
                n=tuple(int(value) for value in metadata["supercell_shape"]),
                origin_cart=ordered_b_cart[0].copy(),
                grid_frac=ordered_b_frac,
                grid_cart=ordered_b_cart,
                matched_target_indices=ordered_b_indices,
                grid_to_target=ordered_b_indices.copy(),
                rmsd=0.0,
                coverage=1.0,
                score=0.0,
            )

            x_fit = fit_x_sites(
                structure.lattice,
                structure.cartesian_coords,
                b_match,
                build_perovskite,
                allowed_systems=ALL_GLAZER_SYSTEMS,
            )
            recovered_b_indices = np.asarray(b_match.matched_target_indices, dtype=int)
            recovered_x_indices = np.asarray(x_fit.x_candidate_indices, dtype=int)
            recovered_a_indices = np.setdiff1d(
                np.arange(structure.atom_count, dtype=int),
                np.union1d(recovered_b_indices, recovered_x_indices),
            )

            def overlap_fraction(recovered: np.ndarray, truth: np.ndarray) -> float:
                if len(truth) == 0:
                    return 1.0 if len(recovered) == 0 else 0.0
                return float(len(np.intersect1d(recovered, truth)) / len(truth))

            print(f"Case {case_index + 1}:")
            print(f"  name: {metadata['name']}")
            print(f"  supercell_shape: {metadata['supercell_shape']}")
            print(f"  lattice_abc: {metadata['lattice_abc']}")
            print(f"  site_elements: {metadata['site_elements']}")
            print(
                f"  tilt_system: expected={metadata['tilt_system']} "
                f"recovered={x_fit.tilt_system}"
            )
            print(
                "  tilt_angles_deg: "
                f"expected={tuple(round(value, 3) for value in metadata['tilt_angles_deg'])} "
                f"recovered={tuple(round(value, 3) for value in x_fit.angles_deg)}"
            )
            print(f"  systems_tried: {len(x_fit.all_attempts)} / {len(ALL_GLAZER_SYSTEMS)}")
            near_zero_summary = [
                (
                    attempt["tilt_system"],
                    tuple(round(value, 3) for value in attempt["angles_deg"]),
                    round(float(attempt["rmsd_matched"]), 6),
                )
                for attempt in x_fit.near_zero_solutions
            ]
            print(f"  near_zero_solutions: {near_zero_summary}")
            print(
                "  x_vacancies: "
                f"fraction={metadata['x_vacancy_fraction']:.3f} "
                f"removed={metadata['x_removed_count']} "
                f"retained={metadata['x_retained_count']}/{metadata['original_x_site_count']}"
            )
            print(
                "  site_counts: "
                f"A={metadata['a_site_count']}/{len(recovered_a_indices)} "
                f"B={metadata['b_site_count']}/{len(recovered_b_indices)} "
                f"X={metadata['x_site_count']}/{len(recovered_x_indices)}"
            )
            print(
                "  site_recall: "
                f"A={overlap_fraction(recovered_a_indices, true_a_indices):.3f} "
                f"B={overlap_fraction(recovered_b_indices, true_b_indices):.3f} "
                f"X={overlap_fraction(recovered_x_indices, true_x_indices):.3f}"
            )
            print("  b_sites: using generated ground-truth B sublattice")
            print(f"  x_fit_rmsd: {x_fit.rmsd:.6f}")
            print(f"  atom_count: {metadata['atom_count']}")
            print()

    rng = np.random.default_rng(7)
    run_demo_pass(title="Pass 1: No Vacancies", rng=rng, with_vacancies=False)
    run_demo_pass(title="Pass 2: X Vacancies", rng=rng, with_vacancies=True)
