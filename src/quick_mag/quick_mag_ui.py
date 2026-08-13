from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from imgui_bundle import (
    __bundle_pyodide__,
    hello_imgui,
    imgui,
    immapp,
    implot,
    implot3d,
    portable_file_dialogs as pfd,
)
from quick_mag.analysis import crystal_radius_for_rendering
from quick_mag.classify_spin_structure import (
    SPIN_CATEGORIES,
    PerovskiteSiteIndexing,
    classify_structure_by_cubes,
    site_indexing_from_generation_parameters,
    site_indexing_from_magnetic_sublattice,
    site_indexing_from_perovskite_build,
)
from quick_mag.cif_io import read_cif
from quick_mag.constants import ELEMENT_RENDER_COLORS, LIGANDS
from quick_mag.element_data import is_valid_symbol
from quick_mag.ion_descriptors import structure_ion_descriptors
from quick_mag.magnetic_moments import (
    OxidationStateAssignment,
    expand_distribution_to_site_assignments,
    format_oxidation_distribution,
)
from quick_mag.oxidation_state_energy import enumerate_oxidation_states_by_energy
from quick_mag.polarization_model import (
    build_Jeff_matrix,
    build_bridges,
    default_params,
    to_solver_couplings,
)
from quick_mag.reference_configs import named_reference_spin_configs
from quick_mag.perovskite_builder import (
    PerovskiteBuild,
    active_glazer_parameter_axes,
    active_tilt_axes,
    build_perovskite,
    canonicalize_glazer_tilt_angles_deg,
    octahedron_triangle_vertices,
)
from quick_mag.structure import (
    ChemicalStructure,
    PerovskiteGenerationParameters,
    SavedSpinConfiguration,
    build_from_generation_parameters,
)
from quick_mag.export_utils import export_structure, export_structures
from quick_mag.generation import (
    formula_atomic_labels_for_build,
    generated_structure_from_parameters,
    normalize_element_symbol,
    normalized_distribution,
)
from quick_mag.spin_solver import (
    SpinConfig,
    canonical_moment_key,
    compute_config_energy,
    solve_for_assignment,
    sort_and_rank,
)
from quick_mag.vasp_io import parse_poscar


def _find_assets_dir() -> Path:
    """Locate the bundled ``assets/`` sample-geometry directory.

    The module lives at ``.../quick_mag/quick_mag_ui.py`` but ``assets/`` sits
    outside the package: at the repo root when running in-place, and staged at
    ``/app/assets`` (i.e. ``parent.parent``) in the Pyodide web build. Probe the
    likely roots and return the first that exists; fall back to the web layout.
    """
    module_dir = Path(__file__).resolve().parent
    candidates = [
        module_dir.parent / "assets",        # web build: /app/quick_mag -> /app/assets
        module_dir.parent.parent / "assets",  # in-repo: src/quick_mag -> repo/assets
        Path.cwd() / "assets",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[0]


ASSETS_DIR = _find_assets_dir()
SAMPLE_GEOMETRY = ASSETS_DIR / "goethite_ZnH81_121.vasp"

NO_ASSIGNMENT_MESSAGE = (
    "No valid oxidation-state assignments were found. "
    "The material may be metallic, have several mixed oxidation states, "
    "or require a different total charge than the value provided."
)
NO_EXCHANGE_COUPLINGS_MESSAGE = (
    "No exchange couplings were found. The structure has no bridged "
    "transition-metal sites for the exchange-polarization model to couple."
)
IS_PYODIDE = bool(__bundle_pyodide__)


def _drain_browser_uploads(state: "AppState") -> None:
    """Load any geometry files the browser staged into the Pyodide FS.

    The web bootstrap (``web/index.html``) writes uploaded/dropped files under
    ``/app/uploads`` and pushes their paths onto ``window.quickMagPendingUploads``.
    We snapshot and clear that queue each frame, then hand each path to the
    existing :meth:`AppState.load_geometry` (which already dispatches ``.cif`` vs
    VASP by suffix), so uploads reuse the desktop loader's error/focus handling.
    """
    if not IS_PYODIDE:
        return
    try:
        import js  # type: ignore
    except ImportError:
        return
    pending = getattr(js.window, "quickMagPendingUploads", None)
    if not pending or not len(pending):
        return
    paths = [str(pending[i]) for i in range(len(pending))]
    pending.length = 0  # clear the JS-side queue so we don't reload next frame
    for path in paths:
        state.load_geometry(Path(path))


def _open_browser_file_picker() -> None:
    """Trigger the hidden ``<input type=file>`` in the web page (Pyodide only)."""
    if not IS_PYODIDE:
        return
    try:
        import js  # type: ignore
    except ImportError:
        return
    element = js.document.getElementById("geometry-upload")
    if element is not None:
        element.click()


DEFAULT_ELEMENT_RENDER_COLOR = (0.72, 0.72, 0.72, 1.0)
DEFAULT_ATOM_RENDER_RADIUS = 0.8
LIGAND_RADIUS_SCALE = 0.4
SPHERE_LATITUDE_SEGMENTS = 8
SPHERE_LONGITUDE_SEGMENTS = 16
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
MAGNETIC_STRUCTURE_STEPS = [
    "Oxidation state analysis",
    "Exchange coupling assignment (GKA)",
    "Spin solve",
]
SPIN_SOLVER_METHODS = ["optimizer", "exact"]
SPIN_UP_COLOR = (0.92, 0.12, 0.10, 1.0)
SPIN_DOWN_COLOR = (0.10, 0.30, 0.95, 1.0)
# Fixed legend order. A and C each single out an axis, so their three orientations are
# separate states that split apart on a distorted cell; they are shaded within a family
# so the legend still reads as "the A's" and "the C's" at a glance.
SPIN_PLOT_CATEGORIES = [
    "G",
    "C(a)",
    "C(b)",
    "C(c)",
    "F",
    "A(a)",
    "A(b)",
    "A(c)",
    "Other",
]
# Per-classification colors (RGBA); anything that is not exactly a reference is gray.
SPIN_CLASS_COLORS = {
    "F": (0.90, 0.20, 0.20, 1.0),
    "G": (0.95, 0.60, 0.10, 1.0),
    "A(a)": (0.45, 0.68, 0.98, 1.0),
    "A(b)": (0.20, 0.45, 0.90, 1.0),
    "A(c)": (0.10, 0.24, 0.62, 1.0),
    "C(a)": (0.48, 0.85, 0.52, 1.0),
    "C(b)": (0.20, 0.70, 0.30, 1.0),
    "C(c)": (0.08, 0.44, 0.18, 1.0),
    # Bare names appear when a short grid axis collapses an orientation, and "E" comes
    # from the CLI's per-site classifier.
    "A": (0.20, 0.45, 0.90, 1.0),
    "C": (0.20, 0.70, 0.30, 1.0),
    "E": (0.65, 0.30, 0.80, 1.0),
    "Other": (0.55, 0.55, 0.55, 1.0),
}
SPIN_ALIGNMENT_COLORS = {
    "aligned": (0.18, 0.72, 0.28, 1.0),
    "anti-aligned": (0.92, 0.20, 0.16, 1.0),
}
FORMULA_MODES = [
    "Perovskite (ABX3)",
    "Double Perovskite (A2B'B''X6)",
    "Quadruple Perovskite (AA'3B4X12)",
    "DQ Perovskite (AA'3BB'X12)",
    "High-Entropy",
]
FORMULA_MODE_KEYS = ("perovskite", "double", "quadruple", "dq", "high_entropy")
FORMULA_MODE_UNIT_FACTORS = {
    "perovskite": 1,
    "double": 2,
    "quadruple": 2,
    "dq": 2,
    "high_entropy": 1,
}


@dataclass
class GeometryData:
    path: Path
    title: str
    lattice: np.ndarray
    species: List[str]
    counts: List[int]
    fractional_coords: np.ndarray
    cartesian_coords: np.ndarray
    species_labels: List[str]
    coordinate_mode: str

    @property
    def atom_count(self) -> int:
        return len(self.species_labels)

    @property
    def formula(self) -> str:
        return " ".join(
            f"{element}{count}" for element, count in zip(self.species, self.counts)
        )

    def grouped_coords(self, use_cartesian: bool) -> Dict[str, np.ndarray]:
        coords = self.cartesian_coords if use_cartesian else self.fractional_coords
        grouped: Dict[str, np.ndarray] = {}
        cursor = 0
        for element, count in zip(self.species, self.counts):
            grouped[element] = ensure_xyz_array(coords[cursor : cursor + count])
            cursor += count
        return grouped

    def as_chemical_structure(self, is_periodic: bool) -> ChemicalStructure:
        return ChemicalStructure.with_zero_magnetic_moments(
            name=self.path.stem,
            lattice=self.lattice,
            cartesian_coords=self.cartesian_coords,
            atomic_labels=self.species_labels,
            is_periodic=is_periodic,
        )

    @classmethod
    def from_chemical_structure(
        cls,
        structure: ChemicalStructure,
        path: Path,
        *,
        coordinate_mode: str = "fractional",
    ) -> "GeometryData":
        """Wrap a ``ChemicalStructure`` (e.g. from a CIF) for the loader info panel.

        ``species``/``counts`` are contiguous element runs in the structure's atom
        order (VASP-block semantics), matching ``parse_vasp``.
        """
        labels = structure.element_symbols()
        species: List[str] = []
        counts: List[int] = []
        for label in labels:
            if species and species[-1] == label:
                counts[-1] += 1
            else:
                species.append(label)
                counts.append(1)
        return cls(
            path=Path(path),
            title=structure.name or Path(path).stem,
            lattice=np.asarray(structure.lattice, dtype=np.float32),
            species=species,
            counts=counts,
            fractional_coords=np.asarray(structure.fractional_coords, dtype=np.float32),
            cartesian_coords=np.asarray(structure.cartesian_coords, dtype=np.float32),
            species_labels=list(labels),
            coordinate_mode=coordinate_mode,
        )


def parse_vasp(path: Path) -> GeometryData:
    data = parse_poscar(path.read_text(), title_fallback=path.stem)
    return GeometryData(
        path=path,
        title=data.title,
        lattice=data.lattice.astype(np.float32),
        species=data.species,
        counts=data.counts,
        fractional_coords=data.fractional_coords.astype(np.float32),
        cartesian_coords=data.cartesian_coords.astype(np.float32),
        species_labels=data.species_labels,
        coordinate_mode=data.coordinate_mode,
    )


def compute_plot_box_limits(
    coords: np.ndarray,
    padding_scale: float = 1.8,
    axis_extents: np.ndarray | None = None,
) -> Tuple[float, float, float, float, float, float]:
    if axis_extents is not None and axis_extents.size:
        mins = (coords - axis_extents).min(axis=0)
        maxs = (coords + axis_extents).max(axis=0)
    else:
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
    center = 0.5 * (mins + maxs)
    span = np.maximum(maxs - mins, 1e-3)
    half_extent = 0.5 * max(float(span.max()), 1.0) * padding_scale
    return (
        float(center[0] - half_extent),
        float(center[0] + half_extent),
        float(center[1] - half_extent),
        float(center[1] + half_extent),
        float(center[2] - half_extent),
        float(center[2] + half_extent),
    )


def clamp_min(value: float, v_min: float) -> float:
    return max(v_min, value)


def ensure_xyz_array(coords: np.ndarray) -> np.ndarray:
    coord_array = np.asarray(coords, dtype=np.float64)
    if coord_array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return coord_array.reshape(-1, 3)


@lru_cache(maxsize=1)
def unit_sphere_template() -> Tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]

    for latitude_index in range(1, SPHERE_LATITUDE_SEGMENTS):
        polar_angle = np.pi * latitude_index / SPHERE_LATITUDE_SEGMENTS
        sin_polar = float(np.sin(polar_angle))
        cos_polar = float(np.cos(polar_angle))
        for longitude_index in range(SPHERE_LONGITUDE_SEGMENTS):
            azimuth = 2.0 * np.pi * longitude_index / SPHERE_LONGITUDE_SEGMENTS
            vertices.append(
                (
                    sin_polar * float(np.cos(azimuth)),
                    sin_polar * float(np.sin(azimuth)),
                    cos_polar,
                )
            )

    vertices.append((0.0, 0.0, -1.0))
    south_pole_index = len(vertices) - 1

    def ring_start(latitude_index: int) -> int:
        return 1 + (latitude_index - 1) * SPHERE_LONGITUDE_SEGMENTS

    triangles: list[tuple[int, int, int]] = []

    first_ring_start = ring_start(1)
    for longitude_index in range(SPHERE_LONGITUDE_SEGMENTS):
        next_longitude = (longitude_index + 1) % SPHERE_LONGITUDE_SEGMENTS
        triangles.append(
            (
                0,
                first_ring_start + next_longitude,
                first_ring_start + longitude_index,
            )
        )

    for latitude_index in range(1, SPHERE_LATITUDE_SEGMENTS - 1):
        current_ring_start = ring_start(latitude_index)
        next_ring_start = ring_start(latitude_index + 1)
        for longitude_index in range(SPHERE_LONGITUDE_SEGMENTS):
            next_longitude = (longitude_index + 1) % SPHERE_LONGITUDE_SEGMENTS
            current = current_ring_start + longitude_index
            current_next = current_ring_start + next_longitude
            below = next_ring_start + longitude_index
            below_next = next_ring_start + next_longitude
            triangles.append((current, below, current_next))
            triangles.append((current_next, below, below_next))

    last_ring_start = ring_start(SPHERE_LATITUDE_SEGMENTS - 1)
    for longitude_index in range(SPHERE_LONGITUDE_SEGMENTS):
        next_longitude = (longitude_index + 1) % SPHERE_LONGITUDE_SEGMENTS
        triangles.append(
            (
                last_ring_start + longitude_index,
                last_ring_start + next_longitude,
                south_pole_index,
            )
        )

    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(triangles, dtype=np.uint32),
    )


def structure_site_oxidation_states(
    state: "AppState",
    structure: ChemicalStructure,
) -> np.ndarray | None:
    assignment = state.selected_oxidation_assignment()
    reference_structure = state.magnetic_result_structure
    if assignment is None or reference_structure is None:
        return None
    if len(assignment.site_oxidation_states) != structure.atom_count:
        return None
    if structure.atomic_labels != reference_structure.atomic_labels:
        return None
    if structure.name == reference_structure.name:
        return assignment.site_oxidation_states
    if np.allclose(structure.lattice, reference_structure.lattice, atol=1e-6) and np.allclose(
        structure.cartesian_coords,
        reference_structure.cartesian_coords,
        atol=1e-6,
    ):
        return assignment.site_oxidation_states
    return None


def structure_atom_render_radii(
    structure: ChemicalStructure,
    site_oxidation_states: np.ndarray | None,
    *,
    render_with_ionic_radius: bool,
) -> np.ndarray:
    fe3_reference = crystal_radius_for_rendering("Fe", 3).crystal_radius
    max_cation_radius = (
        fe3_reference if fe3_reference is not None else DEFAULT_ATOM_RENDER_RADIUS
    )
    ligand_radius = LIGAND_RADIUS_SCALE * max_cation_radius
    radii = np.empty(structure.atom_count, dtype=np.float64)
    for atom_index, element in enumerate(structure.atomic_labels):
        oxidation_state = None
        if site_oxidation_states is not None:
            oxidation_state = int(site_oxidation_states[atom_index])
        resolved_radius = crystal_radius_for_rendering(element, oxidation_state)
        base_radius = (
            resolved_radius.crystal_radius
            if resolved_radius.crystal_radius is not None
            else DEFAULT_ATOM_RENDER_RADIUS
        )
        if render_with_ionic_radius:
            radii[atom_index] = base_radius
        elif element in LIGANDS:
            radii[atom_index] = ligand_radius
        else:
            radii[atom_index] = min(base_radius, max_cation_radius)
    return radii


def sphere_axis_extents(
    radii: np.ndarray,
    lattice: np.ndarray,
    use_cartesian: bool,
) -> np.ndarray:
    if radii.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if use_cartesian:
        return np.repeat(radii[:, None], 3, axis=1)

    inverse_lattice = np.linalg.inv(lattice)
    axis_scale = np.linalg.norm(inverse_lattice, axis=0)
    return radii[:, None] * axis_scale[None, :]


def build_sphere_mesh(
    centers: np.ndarray,
    radii: np.ndarray,
    lattice: np.ndarray,
    *,
    use_cartesian: bool,
) -> implot3d.Mesh:
    unit_vertices, unit_triangles = unit_sphere_template()
    base_vertices = unit_vertices
    if not use_cartesian:
        base_vertices = unit_vertices @ np.linalg.inv(lattice)

    points: list[implot3d.Point] = []
    idx: list[int] = []
    for center, radius in zip(centers, radii):
        if radius <= 0.0:
            continue
        sphere_vertices = center + radius * base_vertices
        vertex_offset = len(points)
        points.extend(
            implot3d.Point(float(vertex[0]), float(vertex[1]), float(vertex[2]))
            for vertex in sphere_vertices
        )
        idx.extend((unit_triangles + vertex_offset).ravel().tolist())
    return implot3d.Mesh(points=points, idx=idx)


def spin_signs_from_moments(
    moments: np.ndarray | None,
    *,
    eps: float = 1e-8,
) -> np.ndarray | None:
    if moments is None:
        return None
    array = np.asarray(moments, dtype=np.float64)
    if array.ndim == 1:
        signs = np.zeros(array.shape[0], dtype=np.int8)
        signs[array > eps] = 1
        signs[array < -eps] = -1
        return signs
    if array.ndim != 2 or array.shape[1] != 3:
        return None

    norms = np.linalg.norm(array, axis=1)
    active = np.flatnonzero(norms > eps)
    signs = np.zeros(array.shape[0], dtype=np.int8)
    if active.size == 0:
        return signs
    reference = array[active[0]] / norms[active[0]]
    dots = array @ reference
    signs[dots > eps] = 1
    signs[dots < -eps] = -1
    return signs


def moments_as_vectors(moments: np.ndarray, n_atoms: int) -> np.ndarray:
    array = np.asarray(moments, dtype=np.float64)
    if array.ndim == 1:
        vectors = np.zeros((n_atoms, 3), dtype=np.float64)
        site_count = min(n_atoms, len(array))
        vectors[:site_count, 2] = array[:site_count]
        return vectors
    if array.ndim == 2 and array.shape[1] == 3:
        vectors = np.zeros((n_atoms, 3), dtype=np.float64)
        site_count = min(n_atoms, array.shape[0])
        vectors[:site_count] = array[:site_count]
        return vectors
    return np.zeros((n_atoms, 3), dtype=np.float64)


def structure_with_moments(
    structure: ChemicalStructure,
    moments: np.ndarray,
) -> ChemicalStructure:
    return ChemicalStructure(
        name=structure.name,
        lattice=np.array(structure.lattice, dtype=np.float64, copy=True),
        cartesian_coords=np.array(structure.cartesian_coords, dtype=np.float64, copy=True),
        atomic_labels=list(structure.atomic_labels),
        magnetic_moments=moments_as_vectors(moments, structure.atom_count),
        is_periodic=structure.is_periodic,
        generation_parameters=structure.generation_parameters,
    )


def formula_key_from_index(index: int) -> str:
    if 0 <= int(index) < len(FORMULA_MODE_KEYS):
        return FORMULA_MODE_KEYS[int(index)]
    return "perovskite"


def formula_index_from_key(key: str) -> int:
    try:
        return FORMULA_MODE_KEYS.index(str(key))
    except ValueError:
        return 0


def formula_unit_factor(key: str) -> int:
    return FORMULA_MODE_UNIT_FACTORS.get(str(key), 1)


def octahedron_triangles_for_generated_structure(
    structure: ChemicalStructure,
    build: PerovskiteBuild,
) -> np.ndarray:
    triangles = octahedron_triangle_vertices(build.octahedra)
    if triangles.size == 0:
        return triangles
    params = getattr(structure, "generation_parameters", None)
    if params is None:
        return triangles
    return triangles - np.asarray(params.cell_origin, dtype=np.float64)


def structures_match_geometry(
    left: ChemicalStructure | None,
    right: ChemicalStructure | None,
    *,
    atol: float = 1e-6,
) -> bool:
    if left is None or right is None:
        return False
    if left.atom_count != right.atom_count:
        return False
    if left.atomic_labels != right.atomic_labels:
        return False
    return bool(
        np.allclose(left.lattice, right.lattice, atol=atol)
        and np.allclose(left.cartesian_coords, right.cartesian_coords, atol=atol)
    )


def recovered_site_indexing_from_magnetic_sites(structure: ChemicalStructure):
    """B-site grid recovered from a builder-less structure's transition-metal sites.

    Lets loaded structures (``generation_parameters is None``) still get A/F/C/G
    cube classifications when their magnetic sublattice forms a perovskite grid.
    Returns None when it does not (non-perovskite ordering).
    """
    from quick_mag.ion_descriptors import TRANSITION_METALS

    magnetic_indices = [
        index
        for index, symbol in enumerate(structure.element_symbols())
        if symbol in TRANSITION_METALS
    ]
    try:
        return site_indexing_from_magnetic_sublattice(structure, magnetic_indices)
    except Exception:
        return None


def cube_fractions_for_structure(
    structure: ChemicalStructure,
    build: PerovskiteBuild,
    *,
    site_indexing=None,
):
    """Per-cube A/C/G/F distribution (or None if too small) via the lookup table."""
    try:
        if site_indexing is None:
            site_indexing = site_indexing_from_perovskite_build(build)
        return classify_structure_by_cubes(structure, site_indexing=site_indexing)
    except Exception:
        return None


def format_oxidation_assignment_label(
    assignment: OxidationStateAssignment,
    index: int,
) -> str:
    return (
        f"{index + 1}. {format_oxidation_distribution(assignment.distributions)} "
        f"[E={assignment.total_energy:.3f}]"
    )


def format_oxidation_assignment_details(
    structure: ChemicalStructure,
    assignment: OxidationStateAssignment,
    *,
    site_moments: np.ndarray | None = None,
    max_sites: int = 48,
) -> str:
    lines = [
        f"Distribution: {format_oxidation_distribution(assignment.distributions)}",
        f"Model energy: {assignment.total_energy:.3f}",
        "",
        "Per-site oxidation states and moments:",
    ]
    moment_vectors = (
        moments_as_vectors(site_moments, structure.atom_count)
        if site_moments is not None
        else np.zeros((structure.atom_count, 3), dtype=np.float64)
    )
    site_count = min(structure.atom_count, len(assignment.site_oxidation_states), max_sites)
    for site_index in range(site_count):
        moment = moment_vectors[site_index]
        lines.append(
            f"{site_index + 1:>3}. {structure.atomic_labels[site_index]:<2}  "
            f"ox={int(assignment.site_oxidation_states[site_index]):+d}  "
            f"m=({moment[0]:+.2f}, {moment[1]:+.2f}, {moment[2]:+.2f})"
        )
    remaining = structure.atom_count - site_count
    if remaining > 0:
        lines.append("")
        lines.append(f"... {remaining} more site(s)")
    return "\n".join(lines)


def unit_cell_vertices(lattice: np.ndarray, use_cartesian: bool) -> np.ndarray:
    fractional_vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    if use_cartesian:
        return fractional_vertices @ lattice
    return fractional_vertices


def plot_unit_cell(lattice: np.ndarray, use_cartesian: bool) -> None:
    vertices = unit_cell_vertices(lattice, use_cartesian)
    edges = [
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 3),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for edge_index, (start, stop) in enumerate(edges):
        edge_points = np.linspace(vertices[start], vertices[stop], num=24, dtype=np.float64)
        xs = np.ascontiguousarray(edge_points[:, 0], dtype=np.float64)
        ys = np.ascontiguousarray(edge_points[:, 1], dtype=np.float64)
        zs = np.ascontiguousarray(edge_points[:, 2], dtype=np.float64)
        spec = implot3d.Spec(
            marker=implot3d.Marker_.circle,
            marker_size=1.8,
            marker_fill_color=(0.88, 0.88, 0.88, 1.0),
            marker_line_color=(0.88, 0.88, 0.88, 1.0),
            fill_alpha=0.95,
        )
        implot3d.plot_scatter(f"##unit_cell_edge_{edge_index}", xs, ys, zs, spec=spec)


def spin_alignment_edge_segments(
    coords: np.ndarray,
    build: PerovskiteBuild,
    moments: np.ndarray | None,
    *,
    dot_tol: float = 1e-6,
) -> dict[str, list[np.ndarray]]:
    if moments is None:
        return {"aligned": [], "anti-aligned": []}

    moment_vectors = moments_as_vectors(moments, coords.shape[0])
    b_grid = np.asarray(build.b_site_indices, dtype=int).reshape(build.octahedra.shape)
    segments: dict[str, list[np.ndarray]] = {"aligned": [], "anti-aligned": []}

    for grid_index in np.ndindex(b_grid.shape):
        site_index = int(b_grid[grid_index])
        site_vector = moment_vectors[site_index]
        if np.linalg.norm(site_vector) <= dot_tol:
            continue
        for axis in range(3):
            if b_grid.shape[axis] <= 1 or grid_index[axis] + 1 >= b_grid.shape[axis]:
                continue
            neighbor_index = list(grid_index)
            neighbor_index[axis] += 1
            neighbor_site = int(b_grid[tuple(neighbor_index)])
            neighbor_vector = moment_vectors[neighbor_site]
            if np.linalg.norm(neighbor_vector) <= dot_tol:
                continue
            dot_value = float(np.dot(site_vector, neighbor_vector))
            if dot_value > dot_tol:
                label = "aligned"
            elif dot_value < -dot_tol:
                label = "anti-aligned"
            else:
                continue
            segments[label].append(
                np.linspace(
                    coords[site_index],
                    coords[neighbor_site],
                    num=16,
                    dtype=np.float64,
                )
            )
    return segments


def spin_alignment_edge_counts(
    coords: np.ndarray,
    build: PerovskiteBuild,
    moments: np.ndarray | None,
) -> dict[str, int]:
    return {
        label: len(edge_segments)
        for label, edge_segments in spin_alignment_edge_segments(
            coords,
            build,
            moments,
        ).items()
    }


def _segments_to_line_coords(
    edge_segments: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if not edge_segments:
        return None
    separator = np.full((1, 3), np.nan, dtype=np.float64)
    stacked = np.vstack(
        [part for segment in edge_segments for part in (segment, separator)]
    )
    return (
        np.ascontiguousarray(stacked[:, 0], dtype=np.float64),
        np.ascontiguousarray(stacked[:, 1], dtype=np.float64),
        np.ascontiguousarray(stacked[:, 2], dtype=np.float64),
    )


def plot_classification_lattice(
    coords: np.ndarray,
    build: PerovskiteBuild,
    moments: np.ndarray | None,
    *,
    line_width: float = 3.0,
) -> None:
    for label, edge_segments in spin_alignment_edge_segments(coords, build, moments).items():
        line_coords = _segments_to_line_coords(edge_segments)
        if line_coords is None:
            continue
        spec = implot3d.Spec(
            line_color=SPIN_ALIGNMENT_COLORS[label],
            marker=implot3d.Marker_.none,
            line_weight=line_width,
        )
        xs, ys, zs = line_coords
        legend_label = (
            "NN aligned (green)"
            if label == "aligned"
            else "NN anti-aligned (red)"
        )
        implot3d.plot_line(legend_label, xs, ys, zs, spec=spec)


# The AppState fields the builder owns. "New structure" restores these to their
# dataclass defaults; keep the list in sync with builder_fields_signature().
BUILDER_FIELD_NAMES: Tuple[str, ...] = (
    "treat_as_periodic",
    "formula_mode",
    "perovskite_type",
    "a_site_element",
    "b_site_element",
    "x_site_element",
    "a2_site_element",
    "b2_site_element",
    "high_entropy_a_site_elements",
    "high_entropy_a_site_fractions",
    "high_entropy_b_site_elements",
    "high_entropy_b_site_fractions",
    "high_entropy_x_site_elements",
    "high_entropy_x_site_fractions",
    "perovskite_rep_x",
    "perovskite_rep_y",
    "perovskite_rep_z",
    "lattice_a",
    "lattice_b",
    "lattice_c",
    "perovskite_tilt_system",
    "tilt_angle_x",
    "tilt_angle_y",
    "tilt_angle_z",
    "perovskite_center",
)


@dataclass
class AppState:
    geometry_path: str = str(SAMPLE_GEOMETRY)
    geometry: GeometryData | None = None
    load_error: str = ""
    status_message: str = ""
    structures: List[ChemicalStructure] = field(default_factory=list)
    # The single "active structure" focus. Always one of ``structures``: the app
    # creates a default structure at startup and never leaves the list empty.
    focus: ChemicalStructure | None = None
    # Index into focus.spin_configurations to display (-1 = use the structure's own moments).
    active_saved_spin_index: int = -1
    use_cartesian: bool = True
    render_with_ionic_radius: bool = False
    show_legend: bool = True
    show_spin_classifications: bool = False
    show_octahedra: bool = True
    show_unit_cell: bool = True
    treat_as_periodic: bool = True
    render_periodic_images: bool = True
    perovskite_center: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    formula_mode: int = 0
    perovskite_type: int = 0
    a_site_element: str = "La"
    b_site_element: str = "Fe"
    x_site_element: str = "O"
    a2_site_element: str = "Sr"
    b2_site_element: str = "Co"
    high_entropy_a_site_elements: List[str] = field(default_factory=lambda: ["La", "Sr"])
    high_entropy_a_site_fractions: List[float] = field(default_factory=lambda: [0.5, 0.5])
    high_entropy_b_site_elements: List[str] = field(default_factory=lambda: ["Fe", "Co"])
    high_entropy_b_site_fractions: List[float] = field(default_factory=lambda: [0.5, 0.5])
    high_entropy_x_site_elements: List[str] = field(default_factory=lambda: ["O"])
    high_entropy_x_site_fractions: List[float] = field(default_factory=lambda: [1.0])
    perovskite_rep_x: int = 1
    perovskite_rep_y: int = 1
    perovskite_rep_z: int = 1
    lattice_a: float = 4.0
    lattice_b: float = 4.0
    lattice_c: float = 4.0
    perovskite_tilt_system: int = 0
    tilt_angle_x: float = 0.0
    tilt_angle_y: float = 0.0
    tilt_angle_z: float = 0.0
    magnetic_solver_method: int = 0
    magnetic_solver_collinear: bool = True
    magnetic_solver_trials: int = 20
    magnetic_solver_steps: int = 250
    magnetic_solver_learning_rate: float = 0.05
    magnetic_solver_energy_tolerance: float = 1e-4
    magnetic_solver_patience: int = 5
    magnetic_solver_max_flip_order: int = 2
    magnetic_solver_max_flip_configs: int = 75000
    last_calculation_method_name: str = ""
    magnetic_result_structure_name: str = ""
    magnetic_result_structure: ChemicalStructure | None = None
    magnetic_analysis_structure: "ChemicalStructure | None" = None
    magnetic_oxidation_assignments: List[OxidationStateAssignment] = field(default_factory=list)
    selected_oxidation_assignment_index: int = 0
    selected_spin_config_index: int = 0
    magnetic_site_indices: List[int] = field(default_factory=list)
    magnetic_j_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )
    magnetic_solution_cache: Dict[int, Tuple[List[Any], List[Any]]] = field(default_factory=dict)
    # The plotted spin-energy landscape. These configurations persist across builder
    # edits: only their energies are recomputed against the new J matrix, so the plot
    # tracks the structure instead of resetting. New configurations come only from a
    # user-requested solve.
    spin_landscape: List[SpinConfig] = field(default_factory=list)
    reference_configs: List[Tuple[str, SpinConfig]] = field(default_factory=list)
    spin_plot_max_configs: int = 100
    baseline_status: str = ""
    # The structure the landscape belongs to. Held by identity rather than id() so a
    # reallocated object at the same address cannot be mistaken for it.
    _baseline_structure: ChemicalStructure | None = None
    magnetic_oxidation_status: str = (
        "Run Magnetic Structure to see oxidation-state analysis."
    )
    magnetic_spin_status: str = "Run Magnetic Structure to see spin-solver results."
    spin_save_message: str = ""
    magnetic_result_collinear: bool = True
    export_directory: str = ""
    export_message: str = ""
    active_structure: ChemicalStructure | None = None
    _pending_structure_delete: Any = None
    _rename_target: Any = None
    _rename_buffer: str = ""
    _rename_request: bool = False
    _builder_bound_id: int | None = None
    _builder_applied_sig: Tuple[object, ...] | None = None
    _last_formula_mode: int = 0
    _last_plot_signature: Tuple[object, ...] | None = None
    _spin_plot_axis_solution: Any = None

    def __post_init__(self) -> None:
        # The app always has exactly one active structure; seed it from the
        # builder defaults so building and solving work with no save step.
        self.create_new_structure()
        # Seeds the spin landscape with the canonical reference orderings.
        self.sync_active_structure()

    # ------------------------------------------------------------------
    # Active-structure focus model
    # ------------------------------------------------------------------
    def is_builder_active(self) -> bool:
        # The builder is bound to the focused structure whenever that structure
        # carries generation parameters, so builder edits regenerate it in place.
        # Structures loaded from a file have no provenance and stay read-only.
        return self.focus_has_generated_provenance()

    def focus_has_generated_provenance(self) -> bool:
        return (
            self.focus is not None
            and getattr(self.focus, "generation_parameters", None) is not None
        )

    def magnetic_results_match_focus(self) -> bool:
        return self.focus is not None and self.magnetic_result_structure is self.focus

    def set_focus(self, structure: ChemicalStructure | None) -> None:
        self.focus = structure
        self.active_saved_spin_index = -1

    def reset_builder_to_defaults(self) -> None:
        """Restore every builder-owned field to its dataclass default."""
        defaults = {item.name: item for item in fields(AppState)}
        for name in BUILDER_FIELD_NAMES:
            spec = defaults[name]
            if spec.default_factory is not MISSING:  # type: ignore[misc]
                setattr(self, name, spec.default_factory())  # type: ignore[misc]
            elif spec.default is not MISSING:
                setattr(self, name, spec.default)
        self._last_formula_mode = self.formula_mode

    def builder_fields_signature(self) -> Tuple[object, ...]:
        return (
            self.treat_as_periodic,
            self.render_periodic_images,
            self.formula_mode,
            self.perovskite_type,
            self.perovskite_rep_x,
            self.perovskite_rep_y,
            self.perovskite_rep_z,
            round(self.lattice_a, 6),
            round(self.lattice_b, 6),
            round(self.lattice_c, 6),
            self.a_site_element.strip(),
            self.b_site_element.strip(),
            self.x_site_element.strip(),
            self.a2_site_element.strip(),
            self.b2_site_element.strip(),
            self.high_entropy_signature(),
            self.perovskite_tilt_system,
            round(self.tilt_angle_x, 6),
            round(self.tilt_angle_y, 6),
            round(self.tilt_angle_z, 6),
            round(float(self.perovskite_center[0]), 6),
            round(float(self.perovskite_center[1]), 6),
            round(float(self.perovskite_center[2]), 6),
        )

    def load_generation_parameters_into_builder(
        self, params: PerovskiteGenerationParameters
    ) -> None:
        formula_mode = getattr(params, "formula_mode", "perovskite")
        self.formula_mode = formula_index_from_key(formula_mode)
        self._last_formula_mode = self.formula_mode
        factor = formula_unit_factor(formula_mode)
        self.perovskite_rep_x = max(0, (int(params.n_oct_x) + 1) // factor - 1)
        self.perovskite_rep_y = max(0, (int(params.n_oct_y) + 1) // factor - 1)
        self.perovskite_rep_z = max(0, (int(params.n_oct_z) + 1) // factor - 1)
        self.lattice_a = float(params.center_to_vertex_distance_x) * 2.0
        self.lattice_b = float(params.center_to_vertex_distance_y) * 2.0
        self.lattice_c = float(params.center_to_vertex_distance_z) * 2.0
        if params.tilt_system in GLAZER_TILT_SYSTEMS:
            self.perovskite_tilt_system = GLAZER_TILT_SYSTEMS.index(params.tilt_system)
        self.tilt_angle_x = float(params.tilt_angle_x_deg)
        self.tilt_angle_y = float(params.tilt_angle_y_deg)
        self.tilt_angle_z = float(params.tilt_angle_z_deg)
        self.a_site_element = params.a_site_element
        self.b_site_element = params.b_site_element
        self.x_site_element = params.x_site_element
        self.a2_site_element = getattr(params, "a2_site_element", self.a2_site_element)
        self.b2_site_element = getattr(params, "b2_site_element", self.b2_site_element)
        self.set_high_entropy_entries("A", getattr(params, "high_entropy_a_sites", []))
        self.set_high_entropy_entries("B", getattr(params, "high_entropy_b_sites", []))
        self.set_high_entropy_entries("X", getattr(params, "high_entropy_x_sites", []))
        self.perovskite_center = np.asarray(params.center, dtype=np.float32).copy()
        self.treat_as_periodic = bool(params.periodic)
        if (
            abs(self.lattice_a - self.lattice_b) < 1e-6
            and abs(self.lattice_a - self.lattice_c) < 1e-6
        ):
            self.perovskite_type = 0
        elif abs(self.lattice_a - self.lattice_b) < 1e-6:
            self.perovskite_type = 1
        else:
            self.perovskite_type = 2

    def sync_builder_binding(self) -> None:
        """Bind the builder fields to a focused generated structure (idempotent)."""
        focus = self.focus
        if focus is not None and focus.generation_parameters is not None:
            if self._builder_bound_id != id(focus):
                self.load_generation_parameters_into_builder(focus.generation_parameters)
                self._builder_bound_id = id(focus)
                # Baseline is established on the next regen-check, after the
                # builder widgets/constraints have run, to avoid a spurious edit.
                self._builder_applied_sig = None
        else:
            self._builder_bound_id = None
            self._builder_applied_sig = None

    def regenerate_focus_from_builder_if_changed(self) -> None:
        """When bound to a generated structure, apply builder edits to it in place."""
        focus = self.focus
        if (
            focus is None
            or focus.generation_parameters is None
            or self._builder_bound_id != id(focus)
        ):
            return
        signature = self.builder_fields_signature()
        if self._builder_applied_sig is None:
            self._builder_applied_sig = signature  # baseline, no regeneration
            return
        if signature == self._builder_applied_sig:
            return
        try:
            regenerated = self.generated_chemical_structure()
        except ValueError:
            return
        focus.lattice = regenerated.lattice
        focus.cartesian_coords = regenerated.cartesian_coords
        focus.atomic_labels = regenerated.atomic_labels
        focus.magnetic_moments = regenerated.magnetic_moments
        focus.generation_parameters = regenerated.generation_parameters
        focus.spin_configurations.clear()
        self.active_saved_spin_index = -1
        if self.magnetic_result_structure is focus:
            # Solver output belongs to the old geometry; the landscape's
            # configurations survive and get re-energized by the baseline below.
            self.clear_solver_results()
        self._builder_applied_sig = signature
        self.prepare_spin_baseline(focus)

    def index_of(self, structure: ChemicalStructure) -> int:
        for index, item in enumerate(self.structures):
            if item is structure:
                return index
        return -1

    def unique_structure_name(self, base: str) -> str:
        """``base``, or ``base (2)``, ``base (3)``, ... if that name is taken."""
        stem = base.strip() or "Structure"
        if not any(item.name == stem for item in self.structures):
            return stem
        suffix = 2
        while any(item.name == f"{stem} ({suffix})" for item in self.structures):
            suffix += 1
        return f"{stem} ({suffix})"

    def rename_structure(self, structure: ChemicalStructure, name: str) -> None:
        candidate = name.strip()
        if not candidate or candidate == structure.name:
            return
        # Exclude the structure itself so renaming "X" to "X " does not yield "X (2)".
        others = [item for item in self.structures if item is not structure]
        unique = candidate
        if any(item.name == unique for item in others):
            suffix = 2
            while any(item.name == f"{candidate} ({suffix})" for item in others):
                suffix += 1
            unique = f"{candidate} ({suffix})"
        structure.name = unique
        if self.magnetic_result_structure is structure:
            self.magnetic_result_structure_name = unique

    def remove_structure(self, structure: ChemicalStructure) -> None:
        index = self.index_of(structure)
        if index < 0:
            return
        self.structures.pop(index)
        if self.magnetic_result_structure is structure:
            self.clear_magnetic_results()
        if self.focus is structure:
            # Fall back to the neighbour that took its place; the list is never
            # left empty, so a fresh default replaces the last structure.
            if self.structures:
                self.set_focus(self.structures[min(index, len(self.structures) - 1)])
            else:
                self.create_new_structure()

    def builder_enabled(self) -> bool:
        return self.is_builder_active()

    def clear_solver_results(
        self,
        *,
        oxidation_status: str = "Run Magnetic Structure to see oxidation-state analysis.",
        spin_status: str = "Run Magnetic Structure to see spin-solver results.",
    ) -> None:
        """Drop solver output but keep the plotted landscape.

        The landscape is re-energized against the new J matrix rather than discarded,
        so builder edits move the reference points instead of emptying the plot.
        """
        self.magnetic_result_structure_name = ""
        self.magnetic_result_structure = None
        self.magnetic_analysis_structure = None
        self.magnetic_result_collinear = True
        self.magnetic_oxidation_assignments = []
        self.selected_oxidation_assignment_index = 0
        self.selected_spin_config_index = 0
        self.magnetic_site_indices = []
        self.magnetic_j_matrix = np.zeros((0, 0), dtype=np.float64)
        self.magnetic_solution_cache = {}
        self.magnetic_oxidation_status = oxidation_status
        self.magnetic_spin_status = spin_status

    def clear_magnetic_results(
        self,
        *,
        oxidation_status: str = "Run Magnetic Structure to see oxidation-state analysis.",
        spin_status: str = "Run Magnetic Structure to see spin-solver results.",
    ) -> None:
        """Full reset, including the plotted landscape (structure deleted / replaced)."""
        self.clear_solver_results(
            oxidation_status=oxidation_status, spin_status=spin_status
        )
        self.reset_spin_landscape()
        self._baseline_structure = None

    def selected_oxidation_assignment(self) -> OxidationStateAssignment | None:
        if not self.magnetic_oxidation_assignments:
            self.selected_oxidation_assignment_index = 0
            self.selected_spin_config_index = 0
            return None
        self.selected_oxidation_assignment_index = min(
            max(self.selected_oxidation_assignment_index, 0),
            len(self.magnetic_oxidation_assignments) - 1,
        )
        return self.magnetic_oxidation_assignments[self.selected_oxidation_assignment_index]

    def selected_spin_config(self) -> Any | None:
        configs = self.displayed_spin_configs()
        if not configs:
            self.selected_spin_config_index = 0
            return None
        self.selected_spin_config_index = min(
            max(self.selected_spin_config_index, 0),
            len(configs) - 1,
        )
        return configs[self.selected_spin_config_index]

    def spin_classification_labels(self) -> List[str]:
        """Label per displayed config: its reference name, or "Other".

        This is an exact match against the canonical orderings (up to global spin
        inversion), not a similarity score -- a configuration is called A(c) only if
        it *is* A(c). Most of a solved landscape is legitimately "Other".
        """
        label_map = self.reference_label_map()
        if not label_map:
            return ["Other"] * len(self.displayed_spin_configs())
        return [
            label_map.get(canonical_moment_key(config.all_moments), "Other")
            for config in self.displayed_spin_configs()
        ]

    def expand_spin_moments_to_structure(
        self,
        moments: np.ndarray,
        structure: ChemicalStructure,
    ) -> np.ndarray:
        array = np.asarray(moments, dtype=np.float64)
        if array.ndim == 1 and len(array) == structure.atom_count:
            return moments_as_vectors(array, structure.atom_count)
        if array.ndim == 2 and array.shape == (structure.atom_count, 3):
            return moments_as_vectors(array, structure.atom_count)

        site_indices = list(self.magnetic_site_indices)
        if array.ndim == 1 and len(array) == len(site_indices):
            vectors = np.zeros((structure.atom_count, 3), dtype=np.float64)
            for compact_index, site_index in enumerate(site_indices):
                if 0 <= site_index < structure.atom_count:
                    vectors[site_index, 2] = array[compact_index]
            return vectors

        if array.ndim == 2 and array.shape[1] == 3 and array.shape[0] == len(site_indices):
            vectors = np.zeros((structure.atom_count, 3), dtype=np.float64)
            for compact_index, site_index in enumerate(site_indices):
                if 0 <= site_index < structure.atom_count:
                    vectors[site_index] = array[compact_index]
            return vectors

        return moments_as_vectors(array, structure.atom_count)

    def resolved_site_indexing(
        self, structure: ChemicalStructure
    ) -> PerovskiteSiteIndexing | None:
        """B-site grid for ``structure``, from builder provenance or recovered from it."""
        params = getattr(structure, "generation_parameters", None)
        build = self.generated_build_for_structure(structure)
        if params is not None and build is not None:
            return site_indexing_from_generation_parameters(params, build)
        return recovered_site_indexing_from_magnetic_sites(structure)

    def compute_reference_configs(
        self,
        structure: ChemicalStructure,
        assignment: OxidationStateAssignment,
    ) -> list[tuple[str, SpinConfig]]:
        """The canonical orderings (G, C(a..c), F, A(a..c)) and their single-point energies.

        Orientations of any grid axis shorter than two cells are dropped by
        ``canonical_reference_patterns``, so a slab or a single-cell grid simply gets
        the subset it can actually distinguish.
        """
        if self.magnetic_j_matrix.size == 0 or not self.magnetic_site_indices:
            return []
        site_indexing = self.resolved_site_indexing(structure)
        if site_indexing is None or site_indexing.b_site_indices.size == 0:
            return []
        try:
            return named_reference_spin_configs(
                structure,
                assignment,
                self.magnetic_j_matrix,
                self.magnetic_site_indices,
                site_indexing,
            )
        except Exception:
            return []

    def reference_label_map(self) -> Dict[Tuple[float, ...], str]:
        """Canonical moment key -> reference name, for exact-match labelling."""
        return {
            canonical_moment_key(config.all_moments): name
            for name, config in self.reference_configs
        }

    def label_for_config(self, config: SpinConfig) -> str:
        """Reference name of ``config``, or "Other" when it is not a canonical ordering."""
        return self.reference_label_map().get(
            canonical_moment_key(config.all_moments), "Other"
        )

    def refresh_landscape_energies(self) -> None:
        """Re-evaluate every retained configuration against the current J matrix.

        A single point per configuration -- no optimization. Configurations whose
        length no longer matches the magnetic-site count belong to a different cell
        (a replication change) and are dropped.
        """
        n_mag = len(self.magnetic_site_indices)
        if n_mag == 0 or self.magnetic_j_matrix.size == 0:
            self.spin_landscape = []
            return

        def reenergized(config: SpinConfig) -> SpinConfig | None:
            moments = np.asarray(config.all_moments, dtype=np.float64)
            if moments.shape[0] != n_mag:
                return None
            try:
                energy = compute_config_energy(self.magnetic_j_matrix, moments)
            except ValueError:
                return None
            return replace(config, energy=float(energy))

        retained = [
            updated
            for updated in (reenergized(config) for config in self.spin_landscape)
            if updated is not None
        ]
        references = [
            updated
            for updated in (reenergized(config) for _, config in self.reference_configs)
            if updated is not None
        ]

        # The cap is a total, and references always claim their slots first: they are
        # the point of the plot, so only the solved remainder is truncated.
        cap = max(int(self.spin_plot_max_configs), len(references))
        merged = sort_and_rank(references + retained)
        reference_keys = {canonical_moment_key(c.all_moments) for c in references}
        others = [
            config
            for config in merged
            if canonical_moment_key(config.all_moments) not in reference_keys
        ]
        self.spin_landscape = sort_and_rank(
            references + others[: cap - len(references)]
        )

    def prepare_spin_baseline(self, structure: ChemicalStructure) -> None:
        """Rebuild J for ``structure`` and re-energize the landscape. No solving.

        This runs whenever the active structure changes, so the plot always shows the
        canonical orderings at their current energies. The expensive part is the
        exchange build; the per-configuration energies are a matrix product each.
        """
        # Recorded up front so a structure that cannot be analysed is not retried on
        # every frame; only a change of focus or an explicit edit re-runs this.
        self._baseline_structure = structure
        try:
            labels = structure.element_symbols()
            ranked = enumerate_oxidation_states_by_energy(labels, charge=0, max_mixing=2)
            if not ranked:
                self.reset_spin_landscape(NO_ASSIGNMENT_MESSAGE)
                return
            assignments = expand_distribution_to_site_assignments(
                [distribution for distribution, _energy in ranked],
                structure,
            )
            if not assignments:
                self.reset_spin_landscape(NO_ASSIGNMENT_MESSAGE)
                return

            # The landscape is a result attached to this structure, so the output pane
            # and the save button treat it exactly like solver output.
            self.magnetic_analysis_structure = structure
            self.magnetic_result_structure = structure
            self.magnetic_result_structure_name = structure.name
            self.magnetic_oxidation_assignments = assignments
            self.selected_oxidation_assignment_index = min(
                max(self.selected_oxidation_assignment_index, 0), len(assignments) - 1
            )
            assignment = assignments[self.selected_oxidation_assignment_index]
            solver_assignment = self.build_unit_moment_assignment(assignment)
            if not self.build_exchange_couplings_for_assignment(assignment):
                self.reset_spin_landscape(NO_EXCHANGE_COUPLINGS_MESSAGE)
                return

            self.reference_configs = self.compute_reference_configs(
                structure, solver_assignment
            )
            self.refresh_landscape_energies()
            self.baseline_status = (
                ""
                if self.reference_configs
                else "No canonical reference orderings for this structure."
            )
        except Exception as exc:  # keep the UI alive on any analysis failure
            self.reset_spin_landscape(f"Reference-configuration setup failed: {exc}")

    def ensure_spin_baseline(self) -> None:
        """Seed a fresh baseline when the focus moves to a different structure.

        The landscape and the solver cache belong to one structure. They persist
        across *edits* of that structure, but a different structure starts over --
        otherwise a same-sized neighbour would inherit its predecessor's solved
        configurations, since re-energizing only checks the magnetic-site count.
        """
        focus = self.focus
        if focus is None:
            return
        if self._baseline_structure is focus:
            return
        self.spin_landscape = []
        self.magnetic_solution_cache = {}
        self.prepare_spin_baseline(focus)

    def reset_spin_landscape(self, status: str = "") -> None:
        """Empty the landscape and everything derived from it.

        The assignments and J matrix go too: leaving the previous structure's values
        behind would show analysis for a structure that is no longer active. Leaves
        ``_baseline_structure`` alone -- callers that want the baseline recomputed
        clear it themselves.
        """
        self.spin_landscape = []
        self.reference_configs = []
        self.magnetic_oxidation_assignments = []
        self.selected_oxidation_assignment_index = 0
        self.selected_spin_config_index = 0
        self.magnetic_site_indices = []
        self.magnetic_j_matrix = np.zeros((0, 0), dtype=np.float64)
        self.baseline_status = status

    def displayed_spin_configs(self) -> List[SpinConfig]:
        """Configurations shown in the plot, the results list, and the 3D view.

        Solver output is merged into ``spin_landscape``, so this is the one accessor
        everything reads -- which is what makes reference points as interactive as
        solved ones.
        """
        if self.focus is None or self.magnetic_analysis_structure is not self.focus:
            return []
        return self.spin_landscape

    def merge_solver_states_into_landscape(self, all_states: list[SpinConfig]) -> None:
        """Fold a completed solve into the persistent landscape, honouring the cap."""
        cap = max(int(self.spin_plot_max_configs), 1)
        self.spin_landscape = list(all_states)[:cap]
        self.refresh_landscape_energies()

    def save_selected_spin_configuration(self) -> None:
        config = self.selected_spin_config()
        structure = self.magnetic_result_structure
        if config is None or structure is None:
            self.spin_save_message = "Run Magnetic Structure and select a configuration first."
            return

        moments = self.expand_spin_moments_to_structure(config.all_moments, structure)

        # Same exact label the plot uses, so a saved config is named A(c) only when it
        # really is A(c) rather than merely closest to it.
        classification = self.label_for_config(config)

        # Read collinearity off the configuration itself: the landscape can outlive
        # the solve that produced it, so the solver's flag may no longer describe it.
        collinear = np.asarray(config.all_moments, dtype=np.float64).ndim == 1

        structure.spin_configurations.append(
            SavedSpinConfiguration(
                magnetic_moments=np.array(moments, dtype=np.float64, copy=True),
                energy=float(config.energy),
                magnetization=float(config.magnetization),
                classification=classification,
                collinear=collinear,
            )
        )
        self.spin_save_message = (
            f"Saved configuration #{len(structure.spin_configurations)} "
            f"to '{structure.name}'."
        )

    def resolved_export_directory(self) -> Path | None:
        directory = self.export_directory.strip()
        if not directory:
            self.export_message = "Choose an export folder first."
            return None
        return Path(directory).expanduser()

    def export_active_structure(self) -> None:
        target = self.resolved_export_directory()
        if target is None:
            return
        structure = self.focus
        if structure is None:
            self.export_message = "No active structure to export."
            return
        try:
            target.mkdir(parents=True, exist_ok=True)
            summary = export_structure(structure, target)
        except Exception as exc:
            self.export_message = f"Export failed: {exc}"
            return
        self.export_message = (
            f"Exported '{structure.name}' with {summary['spin_configs']} spin "
            f"configuration(s) to {target}."
        )

    def export_all_structures(self) -> None:
        target = self.resolved_export_directory()
        if target is None:
            return
        try:
            summary = export_structures(list(self.structures), target)
        except Exception as exc:
            self.export_message = f"Export failed: {exc}"
            return
        self.export_message = (
            f"Exported {summary['structures']} structure(s), "
            f"{summary['spin_configs']} spin configuration(s) to {target}."
        )

    def displayed_saved_spin_configuration(self) -> SavedSpinConfiguration | None:
        """The saved spin config selected in the Active Structure tree, if any."""
        focus = self.focus
        if focus is None or self.active_saved_spin_index < 0:
            return None
        configs = focus.spin_configurations
        if not (0 <= self.active_saved_spin_index < len(configs)):
            return None
        return configs[self.active_saved_spin_index]

    def displayed_saved_spin_moments(
        self,
        structure: ChemicalStructure,
    ) -> np.ndarray | None:
        """Moments of the saved spin config selected in the Active Structure tree."""
        focus = self.focus
        if focus is None or self.active_saved_spin_index < 0:
            return None
        configs = focus.spin_configurations
        if not (0 <= self.active_saved_spin_index < len(configs)):
            return None
        saved_moments = moments_as_vectors(
            configs[self.active_saved_spin_index].magnetic_moments,
            focus.atom_count,
        )
        if structures_match_geometry(focus, structure):
            return moments_as_vectors(saved_moments, structure.atom_count)
        return self.remap_generated_moments_to_structure(
            focus,
            structure,
            saved_moments,
        )

    def remap_generated_moments_to_structure(
        self,
        source_structure: ChemicalStructure,
        target_structure: ChemicalStructure,
        source_moments: np.ndarray,
    ) -> np.ndarray | None:
        if structures_match_geometry(source_structure, target_structure):
            return moments_as_vectors(source_moments, target_structure.atom_count)

        source_params = getattr(source_structure, "generation_parameters", None)
        target_params = getattr(target_structure, "generation_parameters", None)
        if source_params is None or target_params is None:
            return None

        source_build = self.generated_build_for_structure(source_structure)
        target_build = self.generated_build_for_structure(target_structure)
        if source_build is None or target_build is None:
            return None
        if source_build.octahedra.shape != target_build.octahedra.shape:
            return None

        source_vectors = moments_as_vectors(source_moments, source_structure.atom_count)
        target_vectors = np.zeros((target_structure.atom_count, 3), dtype=np.float64)
        source_b_grid = np.asarray(source_build.b_site_indices, dtype=int).reshape(
            source_build.octahedra.shape
        )
        target_b_grid = np.asarray(target_build.b_site_indices, dtype=int).reshape(
            target_build.octahedra.shape
        )
        for grid_index in np.ndindex(source_b_grid.shape):
            source_site = int(source_b_grid[grid_index])
            target_site = int(target_b_grid[grid_index])
            if (
                0 <= source_site < len(source_vectors)
                and 0 <= target_site < len(target_vectors)
            ):
                target_vectors[target_site] = source_vectors[source_site]
        return target_vectors

    def selected_spin_moments_for_structure(
        self,
        structure: ChemicalStructure,
    ) -> np.ndarray | None:
        config = self.selected_spin_config()
        reference = self.magnetic_result_structure
        if config is None or reference is None:
            return None
        source_moments = self.expand_spin_moments_to_structure(config.all_moments, reference)
        if structures_match_geometry(reference, structure):
            return moments_as_vectors(source_moments, structure.atom_count)
        return self.remap_generated_moments_to_structure(
            reference,
            structure,
            source_moments,
        )

    def build_exchange_couplings_for_assignment(
        self,
        assignment: OxidationStateAssignment,
    ) -> bool:
        """Build the exchange-polarization J matrix for the selected assignment.

        The polarization J depends on each transition-metal site's d-shell
        descriptor (oxidation state + spin state), so it is rebuilt per
        assignment. Sets ``magnetic_j_matrix`` / ``magnetic_site_indices`` and
        returns True on success, or sets a status message and returns False.
        """
        structure = self.magnetic_analysis_structure
        if structure is None:
            # No structure to rebuild from: keep any J matrix already provided
            # (e.g. set directly in tests) rather than discarding it.
            if self.magnetic_j_matrix.size and self.magnetic_site_indices:
                return True
            self.magnetic_j_matrix = np.zeros((0, 0), dtype=np.float64)
            self.magnetic_site_indices = []
            self.magnetic_spin_status = "No structure is available for exchange-coupling analysis."
            return False

        try:
            descriptors = structure_ion_descriptors(structure, assignment)
            magnetic_sites = sorted(descriptors)
            bridges = build_bridges(structure, descriptors) if descriptors else []
            if not descriptors or not bridges:
                self.magnetic_j_matrix = np.zeros((0, 0), dtype=np.float64)
                self.magnetic_site_indices = []
                self.magnetic_spin_status = NO_EXCHANGE_COUPLINGS_MESSAGE
                return False

            params = default_params()
            site_index = {site: i for i, site in enumerate(magnetic_sites)}
            j_eff = build_Jeff_matrix(bridges, site_index, params)
            # ``to_solver_couplings`` maps to the spin solver's sign convention;
            # spin magnitude is already inside J_eff, so the solver is fed UNIT
            # (+-1) moments (see build_unit_moment_assignment).
            self.magnetic_j_matrix = to_solver_couplings(j_eff)
            self.magnetic_site_indices = magnetic_sites
        except Exception as exc:
            self.magnetic_j_matrix = np.zeros((0, 0), dtype=np.float64)
            self.magnetic_site_indices = []
            self.magnetic_spin_status = f"Exchange-coupling assignment failed: {exc}"
            return False
        return True

    def run_selected_oxidation_assignment(self, *, force: bool = False) -> None:
        assignment = self.selected_oxidation_assignment()
        if assignment is None:
            self.magnetic_spin_status = "No oxidation-state assignment is selected."
            return

        cache_key = self.selected_oxidation_assignment_index
        if not force and cache_key in self.magnetic_solution_cache:
            # Rebuild J for the selected assignment and restore that assignment's
            # configurations, re-energized against it -- the cached energies belong
            # to whichever assignment was solved last.
            if not self.build_exchange_couplings_for_assignment(assignment):
                return
            structure = self.magnetic_analysis_structure
            if structure is not None:
                self.reference_configs = self.compute_reference_configs(
                    structure, self.build_unit_moment_assignment(assignment)
                )
            _, cached_states = self.magnetic_solution_cache[cache_key]
            self.merge_solver_states_into_landscape(cached_states)
            self.magnetic_spin_status = ""
            return

        if not self.build_exchange_couplings_for_assignment(assignment):
            self.magnetic_solution_cache.pop(cache_key, None)
            return

        self.selected_spin_config_index = 0
        # Spin magnitude is baked into the polarization J matrix, so the solver
        # operates on unit (+-1) spins; a unit-moment copy encodes which sites
        # carry a moment without double-counting magnitude.
        solver_assignment = self.build_unit_moment_assignment(assignment)

        max_flip_configs = (
            None
            if self.magnetic_solver_max_flip_configs <= 0
            else self.magnetic_solver_max_flip_configs
        )
        try:
            base_states, all_states = solve_for_assignment(
                solver_assignment,
                self.magnetic_j_matrix,
                magnetic_site_indices=self.magnetic_site_indices,
                method=SPIN_SOLVER_METHODS[self.magnetic_solver_method],
                collinear=self.magnetic_solver_collinear,
                n_trials=self.magnetic_solver_trials,
                n_steps=self.magnetic_solver_steps,
                lr=self.magnetic_solver_learning_rate,
                energy_tol=self.magnetic_solver_energy_tolerance,
                patience=self.magnetic_solver_patience,
                max_flip_order=self.magnetic_solver_max_flip_order,
                max_flip_configs=max_flip_configs,
            )
        except Exception as exc:
            self.magnetic_solution_cache.pop(cache_key, None)
            self.magnetic_spin_status = (
                f"Spin solve failed for oxidation-state assignment "
                f"{cache_key + 1}: {exc}"
            )
            return

        # The references are recomputed here because the selected oxidation assignment
        # (and therefore J) may have changed since the last baseline.
        structure = self.magnetic_analysis_structure
        if structure is not None:
            self.reference_configs = self.compute_reference_configs(
                structure, solver_assignment
            )
        self.merge_solver_states_into_landscape(all_states)
        self.magnetic_solution_cache[cache_key] = (base_states, list(self.spin_landscape))
        self.magnetic_spin_status = ""

    @staticmethod
    def build_unit_moment_assignment(
        assignment: OxidationStateAssignment,
    ) -> OxidationStateAssignment:
        """Copy of ``assignment`` with unit (+-1 magnitude) magnetic moments.

        Sites carrying a nonzero predicted moment become magnitude 1; the rest
        stay 0. Used for solving against the exchange-polarization J matrix,
        which already encodes spin magnitude.
        """
        moments = np.asarray(assignment.magnetic_moments, dtype=np.float64)
        unit = (np.abs(moments) > 1e-8).astype(np.float64)
        return replace(assignment, magnetic_moments=unit)

    def run_magnetic_structure_calculation(
        self,
        *,
        structure: ChemicalStructure,
    ) -> None:
        self.last_calculation_method_name = "Magnetic Structure"
        self.clear_solver_results(
            oxidation_status="Running oxidation-state analysis...",
            spin_status="Running magnetic structure workflow...",
        )
        self.magnetic_result_structure_name = structure.name
        self.magnetic_result_structure = structure
        self.magnetic_result_collinear = self.magnetic_solver_collinear

        try:
            self.magnetic_analysis_structure = structure
            labels = structure.element_symbols()
            ranked = enumerate_oxidation_states_by_energy(
                labels,
                charge=0,
                max_mixing=2,
            )
            if not ranked:
                self.magnetic_oxidation_status = NO_ASSIGNMENT_MESSAGE
                self.magnetic_spin_status = "Spin solve skipped because no oxidation-state assignments were found."
                return

            assignments = expand_distribution_to_site_assignments(
                [distribution for distribution, _energy in ranked],
                structure,
            )
            if not assignments:
                self.magnetic_oxidation_status = NO_ASSIGNMENT_MESSAGE
                self.magnetic_spin_status = "Spin solve skipped because no site-resolved assignments were produced."
                return

            self.magnetic_oxidation_assignments = assignments
            self.selected_oxidation_assignment_index = 0
            self.magnetic_oxidation_status = ""
        except Exception as exc:
            self.clear_magnetic_results(
                oxidation_status=f"Magnetic Structure setup failed: {exc}",
                spin_status="Spin solve was not started.",
            )
            self.last_calculation_method_name = "Magnetic Structure"
            self.magnetic_result_structure_name = structure.name
            self.magnetic_result_structure = structure
            return

        # The exchange-polarization J matrix depends on the chosen oxidation-state
        # assignment (via each site's d-shell descriptor), so it is built per
        # assignment inside run_selected_oxidation_assignment rather than once here.
        self.run_selected_oxidation_assignment(force=True)

    def load_geometry(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        try:
            if resolved.suffix.lower() == ".cif":
                geometry = GeometryData.from_chemical_structure(
                    read_cif(resolved), resolved
                )
            else:
                geometry = parse_vasp(resolved)
        except Exception as exc:
            self.load_error = str(exc)
            self.status_message = ""
            return

        self.geometry = geometry
        self.geometry_path = str(geometry.path)
        self.load_error = ""
        # A loaded file becomes a structure in its own right and takes focus. It
        # carries no generation parameters, so the builder stays disabled for it.
        loaded = geometry.as_chemical_structure(is_periodic=True)
        loaded.name = self.unique_structure_name(geometry.path.stem or "loaded")
        self.structures.append(loaded)
        self.set_focus(loaded)
        self.status_message = f"Loaded {geometry.path.name} with {geometry.atom_count} atoms."

    def apply_perovskite_constraints(self) -> None:
        self.formula_mode = min(max(int(self.formula_mode), 0), len(FORMULA_MODES) - 1)
        self.perovskite_rep_x = max(0, self.perovskite_rep_x)
        self.perovskite_rep_y = max(0, self.perovskite_rep_y)
        self.perovskite_rep_z = max(0, self.perovskite_rep_z)
        self.lattice_a = clamp_min(self.lattice_a, 2.0)
        self.lattice_b = clamp_min(self.lattice_b, 2.0)
        self.lattice_c = clamp_min(self.lattice_c, 2.0)
        self.ensure_high_entropy_rows()

        if self.perovskite_type == 0:
            self.lattice_b = self.lattice_a
            self.lattice_c = self.lattice_a
        elif self.perovskite_type == 1:
            self.lattice_b = self.lattice_a

        self.tilt_angle_x = min(max(self.tilt_angle_x, -45.0), 45.0)
        self.tilt_angle_y = min(max(self.tilt_angle_y, -45.0), 45.0)
        self.tilt_angle_z = min(max(self.tilt_angle_z, -45.0), 45.0)

        if not self.tilt_system_available():
            self.perovskite_tilt_system = 0
            self.tilt_angle_x = 0.0
            self.tilt_angle_y = 0.0
            self.tilt_angle_z = 0.0
            return

        active_x, active_y, active_z = active_tilt_axes(
            GLAZER_TILT_SYSTEMS[self.perovskite_tilt_system]
        )
        if not active_x:
            self.tilt_angle_x = 0.0
        if not active_y:
            self.tilt_angle_y = 0.0
        if not active_z:
            self.tilt_angle_z = 0.0

        constrained_angles = canonicalize_glazer_tilt_angles_deg(
            GLAZER_TILT_SYSTEMS[self.perovskite_tilt_system],
            self.tilt_angle_x,
            self.tilt_angle_y,
            self.tilt_angle_z,
        )
        self.tilt_angle_x = constrained_angles[0]
        self.tilt_angle_y = constrained_angles[1]
        self.tilt_angle_z = constrained_angles[2]

    def tilt_system_available(self) -> bool:
        effective_x, effective_y, effective_z = self.effective_oct_counts()
        return (
            effective_x >= 1
            and effective_y >= 1
            and effective_z >= 1
        )

    def formula_key(self) -> str:
        return formula_key_from_index(self.formula_mode)

    def default_replications_for_formula(self) -> tuple[int, int, int]:
        if self.formula_key() in ("double", "quadruple", "dq"):
            return (0, 0, 0)
        return (1, 1, 1)

    def apply_default_replications_for_formula(self) -> None:
        (
            self.perovskite_rep_x,
            self.perovskite_rep_y,
            self.perovskite_rep_z,
        ) = self.default_replications_for_formula()

    def apply_default_composition_for_formula(self) -> None:
        if self.formula_key() != "dq":
            return
        self.a_site_element = "Ca"
        self.a2_site_element = "Cu"
        self.b_site_element = "Fe"
        self.b2_site_element = "Re"
        self.x_site_element = "O"

    def apply_defaults_for_formula(self) -> None:
        self.apply_default_replications_for_formula()
        self.apply_default_composition_for_formula()

    def formula_unit_factor(self) -> int:
        return formula_unit_factor(self.formula_key())

    def effective_n_oct(self, replications: int) -> int:
        return (max(0, int(replications)) + 1) * self.formula_unit_factor() - 1

    def effective_oct_counts(self) -> tuple[int, int, int]:
        return (
            self.effective_n_oct(self.perovskite_rep_x),
            self.effective_n_oct(self.perovskite_rep_y),
            self.effective_n_oct(self.perovskite_rep_z),
        )

    def high_entropy_entries(self, site: str) -> list[tuple[str, float]]:
        if site == "A":
            return list(
                zip(
                    self.high_entropy_a_site_elements,
                    self.high_entropy_a_site_fractions,
                )
            )
        if site == "B":
            return list(
                zip(
                    self.high_entropy_b_site_elements,
                    self.high_entropy_b_site_fractions,
                )
            )
        return list(
            zip(
                self.high_entropy_x_site_elements,
                self.high_entropy_x_site_fractions,
            )
        )

    def set_high_entropy_entries(
        self,
        site: str,
        entries: list[tuple[str, float]],
    ) -> None:
        if not entries:
            return
        elements = [str(element) for element, _ in entries]
        fractions = [max(0.0, float(fraction)) for _, fraction in entries]
        if site == "A":
            self.high_entropy_a_site_elements = elements
            self.high_entropy_a_site_fractions = fractions
        elif site == "B":
            self.high_entropy_b_site_elements = elements
            self.high_entropy_b_site_fractions = fractions
        else:
            self.high_entropy_x_site_elements = elements
            self.high_entropy_x_site_fractions = fractions

    def ensure_high_entropy_rows(self) -> None:
        defaults = {
            "A": ("La", 1.0),
            "B": ("Fe", 1.0),
            "X": ("O", 1.0),
        }
        for site, default in defaults.items():
            entries = self.high_entropy_entries(site)
            if not entries:
                self.set_high_entropy_entries(site, [default])
                continue
            self.set_high_entropy_entries(
                site,
                [(element, max(0.0, float(fraction))) for element, fraction in entries],
            )

    def high_entropy_signature(self) -> tuple[tuple[tuple[str, float], ...], ...]:
        return (
            tuple(
                (element.strip(), round(float(fraction), 6))
                for element, fraction in self.high_entropy_entries("A")
            ),
            tuple(
                (element.strip(), round(float(fraction), 6))
                for element, fraction in self.high_entropy_entries("B")
            ),
            tuple(
                (element.strip(), round(float(fraction), 6))
                for element, fraction in self.high_entropy_entries("X")
            ),
        )

    def half_edge_lengths(self) -> Tuple[float, float, float]:
        self.apply_perovskite_constraints()
        return (
            0.5 * self.lattice_a,
            0.5 * self.lattice_b,
            0.5 * self.lattice_c,
        )

    def builder_cell_origin(self) -> np.ndarray:
        half_a, half_b, half_c = self.half_edge_lengths()
        return np.asarray(self.perovskite_center, dtype=np.float64) - np.array(
            [half_a, half_b, half_c],
            dtype=np.float64,
        )

    def builder_supercell_lattice(self) -> np.ndarray:
        self.apply_perovskite_constraints()
        effective_x, effective_y, effective_z = self.effective_oct_counts()
        return np.array(
            [
                [(effective_x + 1) * self.lattice_a, 0.0, 0.0],
                [0.0, (effective_y + 1) * self.lattice_b, 0.0],
                [0.0, 0.0, (effective_z + 1) * self.lattice_c],
            ],
            dtype=np.float64,
        )

    def _normalize_element_symbol(self, raw_symbol: str) -> str:
        return normalize_element_symbol(raw_symbol)

    def validated_builder_elements(self) -> tuple[str, str, str]:
        site_labels = (
            ("A", self.a_site_element),
            ("B", self.b_site_element),
            ("X", self.x_site_element),
        )
        validated_symbols: list[str] = []
        for site_name, raw_symbol in site_labels:
            symbol = self._normalize_element_symbol(raw_symbol)
            if not is_valid_symbol(symbol):
                raise ValueError(
                    f"{site_name}-site element '{raw_symbol}' is not a valid element symbol."
                )
            validated_symbols.append(symbol)
        return validated_symbols[0], validated_symbols[1], validated_symbols[2]

    def atomic_labels_for_build(
        self,
        build: PerovskiteBuild,
        *,
        periodic: bool,
    ) -> list[str]:
        return formula_atomic_labels_for_build(
            build,
            periodic=periodic,
            formula_mode=self.formula_key(),
            a_site_element=self.a_site_element,
            b_site_element=self.b_site_element,
            x_site_element=self.x_site_element,
            a2_site_element=self.a2_site_element,
            b2_site_element=self.b2_site_element,
            high_entropy_a_sites=self.high_entropy_entries("A"),
            high_entropy_b_sites=self.high_entropy_entries("B"),
            high_entropy_x_sites=self.high_entropy_entries("X"),
        )

    def generated_perovskite(self) -> PerovskiteBuild:
        return self.generated_perovskite_with_periodicity(self.treat_as_periodic)

    def generated_perovskite_with_periodicity(self, periodic: bool) -> PerovskiteBuild:
        half_a, half_b, half_c = self.half_edge_lengths()
        effective_x, effective_y, effective_z = self.effective_oct_counts()
        return build_perovskite(
            center=self.perovskite_center,
            n_oct_x=effective_x,
            n_oct_y=effective_y,
            n_oct_z=effective_z,
            center_to_vertex_distance_x=half_a,
            center_to_vertex_distance_y=half_b,
            center_to_vertex_distance_z=half_c,
            tilt_system=GLAZER_TILT_SYSTEMS[self.perovskite_tilt_system],
            tilt_angle_x_deg=self.tilt_angle_x,
            tilt_angle_y_deg=self.tilt_angle_y,
            tilt_angle_z_deg=self.tilt_angle_z,
            periodic=periodic,
        )

    def generated_chemical_structure(self) -> ChemicalStructure:
        return self.generated_chemical_structure_with_periodicity(self.treat_as_periodic)

    def generated_chemical_structure_with_periodicity(
        self,
        periodic: bool,
    ) -> ChemicalStructure:
        build = self.generated_perovskite_with_periodicity(periodic)
        lattice = self.builder_supercell_lattice()
        cell_origin = self.builder_cell_origin()
        if self.formula_key() == "high_entropy":
            a_symbol = normalized_distribution(
                self.high_entropy_entries("A"), site_name="A"
            )[0][0]
            b_symbol = normalized_distribution(
                self.high_entropy_entries("B"), site_name="B"
            )[0][0]
            x_symbol = normalized_distribution(
                self.high_entropy_entries("X"), site_name="X"
            )[0][0]
        else:
            a_symbol, b_symbol, x_symbol = self.validated_builder_elements()

        cartesian_coords = np.vstack((build.a_sites, build.b_sites, build.x_sites)).astype(
            np.float64
        )
        cartesian_coords -= cell_origin
        atomic_labels = self.atomic_labels_for_build(build, periodic=periodic)
        structure = ChemicalStructure.with_zero_magnetic_moments(
            name="Builder preview",
            lattice=lattice,
            cartesian_coords=cartesian_coords,
            atomic_labels=atomic_labels,
            is_periodic=periodic,
        )

        na, nb, nx = len(build.a_sites), len(build.b_sites), len(build.x_sites)
        half_a, half_b, half_c = self.half_edge_lengths()
        effective_x, effective_y, effective_z = self.effective_oct_counts()
        structure.generation_parameters = PerovskiteGenerationParameters(
            center=np.asarray(self.perovskite_center, dtype=np.float64),
            n_oct_x=effective_x,
            n_oct_y=effective_y,
            n_oct_z=effective_z,
            center_to_vertex_distance_x=half_a,
            center_to_vertex_distance_y=half_b,
            center_to_vertex_distance_z=half_c,
            tilt_system=GLAZER_TILT_SYSTEMS[self.perovskite_tilt_system],
            tilt_angle_x_deg=self.tilt_angle_x,
            tilt_angle_y_deg=self.tilt_angle_y,
            tilt_angle_z_deg=self.tilt_angle_z,
            periodic=periodic,
            a_site_element=a_symbol,
            b_site_element=b_symbol,
            x_site_element=x_symbol,
            formula_mode=self.formula_key(),
            a2_site_element=(
                self._normalize_element_symbol(self.a2_site_element)
                if self.formula_key() in ("quadruple", "dq")
                else self.a2_site_element.strip()
            ),
            b2_site_element=(
                self._normalize_element_symbol(self.b2_site_element)
                if self.formula_key() in ("double", "dq")
                else self.b2_site_element.strip()
            ),
            high_entropy_a_sites=(
                normalized_distribution(self.high_entropy_entries("A"), site_name="A")
                if self.formula_key() == "high_entropy"
                else self.high_entropy_entries("A")
            ),
            high_entropy_b_sites=(
                normalized_distribution(self.high_entropy_entries("B"), site_name="B")
                if self.formula_key() == "high_entropy"
                else self.high_entropy_entries("B")
            ),
            high_entropy_x_sites=(
                normalized_distribution(self.high_entropy_entries("X"), site_name="X")
                if self.formula_key() == "high_entropy"
                else self.high_entropy_entries("X")
            ),
            spin_pattern="None",
            spin_moment_magnitude=0.0,
            x_vacancy_fraction=0.0,
            x_removed_count=0,
            removed_x_site_indices=np.zeros(0, dtype=np.int64),
            site_roles=(["A"] * na + ["B"] * nb + ["X"] * nx),
            permutation=np.arange(na + nb + nx, dtype=np.int64),
            cell_origin=cell_origin,
            source="perovskite_builder",
        )
        return structure

    def generated_build_for_structure(
        self,
        structure: ChemicalStructure,
    ) -> PerovskiteBuild | None:
        params = getattr(structure, "generation_parameters", None)
        if params is not None:
            return build_from_generation_parameters(params)
        # Fallback for legacy structures with no stored provenance: regenerate
        # from the live builder UI and match geometry.
        try:
            generated_structure = self.generated_chemical_structure()
            build = self.generated_perovskite()
        except ValueError:
            return None
        if structures_match_geometry(generated_structure, structure):
            return build
        return None

    def current_structure(self) -> ChemicalStructure | None:
        """The active structure. Builder edits are applied to it in place."""
        return self.focus

    def rendered_structure(self) -> ChemicalStructure | None:
        if (
            self.focus_has_generated_provenance()
            and self.render_periodic_images
            and self.focus is not None
            and self.focus.is_periodic
        ):
            return generated_structure_from_parameters(
                self.focus.generation_parameters,
                name=self.focus.name,
                periodic=False,
            )
        return self.focus

    def focus_is_loaded(self) -> bool:
        """True when the focus is a structure with no generation parameters."""
        return (
            self.focus is not None
            and getattr(self.focus, "generation_parameters", None) is None
        )

    def sync_active_structure(self) -> None:
        self.sync_builder_binding()
        self.active_structure = self.current_structure()
        # Idempotent: only fires when the focus moved to a different structure.
        self.ensure_spin_baseline()

    def create_new_structure(self) -> None:
        """Add a structure built from the default builder settings and focus it."""
        self.reset_builder_to_defaults()
        structure = self.generated_chemical_structure()
        structure.name = self.unique_structure_name(
            f"Structure {len(self.structures) + 1}"
        )
        self.structures.append(structure)
        self.set_focus(structure)
        # Force sync_builder_binding to rebind (and re-baseline) on the next frame.
        self._builder_bound_id = None
        self._builder_applied_sig = None

    def run_selected_calculation(self) -> None:
        structure = self.focus
        if structure is None:
            return
        self.run_magnetic_structure_calculation(structure=structure)

    def refresh_plot_view_generation(self, signature: Tuple[object, ...]) -> bool:
        if signature != self._last_plot_signature:
            self._last_plot_signature = signature
            return True
        return False

APP_STATE = AppState()


def axis_length_control(label: str, value: float, enabled: bool, linked_note: str = "") -> float:
    if not enabled:
        imgui.begin_disabled()

    _, value = imgui.input_float(f"{label} (A)", value, 0.1, 1.0, "%.3f")

    if not enabled:
        imgui.end_disabled()
        if linked_note:
            imgui.same_line()
            imgui.text_disabled(linked_note)

    return clamp_min(value, 2.0)


def tilt_angle_control(label: str, value: float, enabled: bool) -> float:
    if not enabled:
        imgui.begin_disabled()

    _, value = imgui.slider_float(label, value, -45.0, 45.0, "%.1f deg")

    if not enabled:
        imgui.end_disabled()
        imgui.same_line()
        imgui.text_disabled("inactive in selected tilt system")

    return min(max(value, -45.0), 45.0)


def high_entropy_site_controls(state: AppState, site: str, label: str) -> None:
    elements = {
        "A": state.high_entropy_a_site_elements,
        "B": state.high_entropy_b_site_elements,
        "X": state.high_entropy_x_site_elements,
    }[site]
    fractions = {
        "A": state.high_entropy_a_site_fractions,
        "B": state.high_entropy_b_site_fractions,
        "X": state.high_entropy_x_site_fractions,
    }[site]

    fraction_column_x = 125.0
    remove_column_x = 260.0

    imgui.text(label)
    imgui.text_disabled("Elements")
    imgui.same_line(fraction_column_x)
    imgui.text_disabled("Fraction")
    imgui.same_line(remove_column_x)
    imgui.text_disabled("Remove")

    remove_index = -1
    for index in range(len(elements)):
        imgui.push_id(f"he_{site}_{index}")
        imgui.push_item_width(110)
        _, elements[index] = imgui.input_text("##element", elements[index])
        imgui.pop_item_width()
        imgui.same_line(fraction_column_x)
        imgui.push_item_width(120)
        _, fractions[index] = imgui.input_float(
            "##fraction",
            fractions[index],
            0.0,
            0.0,
            "%.3f",
        )
        fractions[index] = max(0.0, float(fractions[index]))
        imgui.pop_item_width()
        imgui.same_line(remove_column_x)
        if imgui.button("-##remove"):
            remove_index = index
        imgui.pop_id()

    if remove_index >= 0 and len(elements) > 1:
        elements.pop(remove_index)
        fractions.pop(remove_index)

    if imgui.button(f"+##add_he_{site}"):
        elements.append("O" if site == "X" else ("Fe" if site == "B" else "La"))
        fractions.append(0.0)
    imgui.same_line()
    imgui.text("Add")

    total = sum(max(0.0, float(value)) for value in fractions)
    try:
        normalized = normalized_distribution(state.high_entropy_entries(site), site_name=site)
        summary = ", ".join(
            f"{element} {fraction:.3f}" for element, fraction in normalized
        )
        imgui.text_disabled(f"Total {total:.3f}; normalized: {summary}")
    except ValueError as exc:
        imgui.push_style_color(imgui.Col_.text, (0.95, 0.35, 0.35, 1.0))
        imgui.text_wrapped(str(exc))
        imgui.pop_style_color()


def gui_controls() -> None:
    state = APP_STATE
    _drain_browser_uploads(state)
    state.sync_builder_binding()
    state.apply_perovskite_constraints()
    state.magnetic_solver_trials = max(0, state.magnetic_solver_trials)
    state.magnetic_solver_steps = max(0, state.magnetic_solver_steps)
    state.magnetic_solver_learning_rate = max(0.0, state.magnetic_solver_learning_rate)
    state.magnetic_solver_energy_tolerance = max(0.0, state.magnetic_solver_energy_tolerance)
    state.magnetic_solver_patience = max(0, state.magnetic_solver_patience)
    state.magnetic_solver_max_flip_order = max(0, state.magnetic_solver_max_flip_order)

    if imgui.button("New structure", size=(140, 0)):
        state.create_new_structure()
    focus_name = "-" if state.focus is None else state.focus.name
    imgui.text(f"Active structure: {focus_name}")
    # text_disabled does not wrap, so dim a wrapped block by hand.
    imgui.push_style_color(imgui.Col_.text, imgui.get_style().color_(imgui.Col_.text_disabled))
    imgui.text_wrapped(
        "Builder edits apply to the active structure. Choose it in the Active "
        "Structure panel."
    )
    imgui.pop_style_color()
    imgui.separator()

    imgui.spacing()
    if imgui.collapsing_header(
        "Perovskite builder##builder_panel", imgui.TreeNodeFlags_.default_open.value
    ):
        # Capture once: editing can change builder_enabled() mid-frame, so the
        # begin/end_disabled pair must use the same value.
        builder_disabled = not state.builder_enabled()
        if builder_disabled:
            imgui.text_wrapped(
                "This structure is not editable in the builder (loading a file "
                "decouples it). Select a generated structure to edit, or press "
                "New structure."
            )
            imgui.begin_disabled()

        imgui.text("Formula")
        imgui.push_item_width(250)
        formula_changed, state.formula_mode = imgui.combo(
            "##formula_mode",
            state.formula_mode,
            FORMULA_MODES,
        )
        imgui.pop_item_width()
        state.apply_perovskite_constraints()
        if formula_changed and state.formula_mode != state._last_formula_mode:
            state.apply_defaults_for_formula()
            state.apply_perovskite_constraints()
            state._last_formula_mode = state.formula_mode

        imgui.spacing()
        _, state.treat_as_periodic = imgui.checkbox(
            "Treat structure as periodic", state.treat_as_periodic
        )

        imgui.spacing()
        if imgui.collapsing_header(
            "Atoms##builder_atoms_panel", imgui.TreeNodeFlags_.default_open.value
        ):
            imgui.push_item_width(90)
            if state.formula_key() == "high_entropy":
                high_entropy_site_controls(state, "A", "A sites")
                imgui.spacing()
                high_entropy_site_controls(state, "B", "B sites")
                imgui.spacing()
                high_entropy_site_controls(state, "X", "X sites")
            else:
                _, state.a_site_element = imgui.input_text("A-site", state.a_site_element)
                if state.formula_key() in ("quadruple", "dq"):
                    _, state.a2_site_element = imgui.input_text(
                        "A'-site", state.a2_site_element
                    )
                _, state.b_site_element = imgui.input_text("B-site", state.b_site_element)
                if state.formula_key() in ("double", "dq"):
                    b2_label = "B'-site" if state.formula_key() == "dq" else "B''-site"
                    _, state.b2_site_element = imgui.input_text(
                        b2_label, state.b2_site_element
                    )
                _, state.x_site_element = imgui.input_text("X-site", state.x_site_element)
            imgui.pop_item_width()

            try:
                preview_build = state.generated_perovskite()
                preview_labels = state.atomic_labels_for_build(
                    preview_build,
                    periodic=state.treat_as_periodic,
                )
                preview_counts: dict[str, int] = {}
                for symbol in preview_labels:
                    preview_counts[symbol] = preview_counts.get(symbol, 0) + 1
                count_summary = ", ".join(
                    f"{symbol}: {count}" for symbol, count in sorted(preview_counts.items())
                )
                imgui.text_wrapped(f"Validated atoms: {count_summary}")
            except ValueError as exc:
                imgui.push_style_color(imgui.Col_.text, (0.95, 0.35, 0.35, 1.0))
                imgui.text_wrapped(str(exc))
                imgui.pop_style_color()

        imgui.spacing()
        if imgui.collapsing_header("Lattice##builder_lattice_panel"):
            _, state.perovskite_rep_x = imgui.input_int(
                "Replications x", state.perovskite_rep_x, 1, 10
            )
            _, state.perovskite_rep_y = imgui.input_int(
                "Replications y", state.perovskite_rep_y, 1, 10
            )
            _, state.perovskite_rep_z = imgui.input_int(
                "Replications z", state.perovskite_rep_z, 1, 10
            )
            state.perovskite_rep_x = max(0, state.perovskite_rep_x)
            state.perovskite_rep_y = max(0, state.perovskite_rep_y)
            state.perovskite_rep_z = max(0, state.perovskite_rep_z)
            state.apply_perovskite_constraints()
            imgui.spacing()
            imgui.text("Lattice constants")
            state.lattice_a = axis_length_control("a", state.lattice_a, enabled=True)
            state.lattice_b = axis_length_control(
                "b",
                state.lattice_b,
                enabled=state.perovskite_type == 2,
                linked_note="linked to a" if state.perovskite_type in (0, 1) else "",
            )
            state.lattice_c = axis_length_control(
                "c",
                state.lattice_c,
                enabled=state.perovskite_type in (1, 2),
                linked_note="linked to a" if state.perovskite_type == 0 else "",
            )

            imgui.spacing()
            imgui.text("Perovskite type")
            if imgui.radio_button("Cubic##perovskite_type", state.perovskite_type == 0):
                state.perovskite_type = 0
            imgui.same_line()
            if imgui.radio_button("Tetragonal##perovskite_type", state.perovskite_type == 1):
                state.perovskite_type = 1
            imgui.same_line()
            if imgui.radio_button(
                "Orthorhombic##perovskite_type", state.perovskite_type == 2
            ):
                state.perovskite_type = 2
            state.apply_perovskite_constraints()

        imgui.spacing()
        if imgui.collapsing_header("Tilt system##perovskite_tilt_panel"):
            tilt_controls_enabled = state.tilt_system_available()
            if not tilt_controls_enabled:
                imgui.text_wrapped(
                    "You need at least one replication in each direction to use the tilt systems."
                )
                imgui.spacing()

            if not tilt_controls_enabled:
                imgui.begin_disabled()

            imgui.text("Glazer notation")
            imgui.push_item_width(170)
            _, state.perovskite_tilt_system = imgui.combo(
                "##perovskite_tilt_system",
                state.perovskite_tilt_system,
                GLAZER_TILT_SYSTEMS,
            )
            imgui.pop_item_width()

            state.apply_perovskite_constraints()
            active_x, active_y, active_z = active_glazer_parameter_axes(
                GLAZER_TILT_SYSTEMS[state.perovskite_tilt_system]
            )

            imgui.spacing()
            imgui.text("Tilt angles")
            state.tilt_angle_x = tilt_angle_control(
                "Tilt a (deg)", state.tilt_angle_x, active_x
            )
            state.tilt_angle_y = tilt_angle_control(
                "Tilt b (deg)", state.tilt_angle_y, active_y
            )
            state.tilt_angle_z = tilt_angle_control(
                "Tilt c (deg)", state.tilt_angle_z, active_z
            )

            if not tilt_controls_enabled:
                imgui.end_disabled()

        active_build = state.generated_perovskite()
        a_site_count = len(active_build.a_sites)
        b_site_count = len(active_build.b_sites)
        x_site_count = len(active_build.x_sites)
        imgui.spacing()
        imgui.text(
            f"a = {state.lattice_a:.3f} A, b = {state.lattice_b:.3f} A, c = {state.lattice_c:.3f} A"
        )
        imgui.text(
            f"Active structure: {'periodic' if state.treat_as_periodic else 'non-periodic'}"
        )
        try:
            labels = state.atomic_labels_for_build(
                active_build,
                periodic=state.treat_as_periodic,
            )
            role_counts = {
                "A": labels[:a_site_count],
                "B": labels[a_site_count : a_site_count + b_site_count],
                "X": labels[a_site_count + b_site_count :],
            }
            for role, role_labels in role_counts.items():
                counts: dict[str, int] = {}
                for symbol in role_labels:
                    counts[symbol] = counts.get(symbol, 0) + 1
                summary = ", ".join(
                    f"{symbol}: {count}" for symbol, count in sorted(counts.items())
                )
                imgui.text(f"{role} sites ({summary})")
        except ValueError as exc:
            imgui.text(f"A sites: {a_site_count}")
            imgui.text(f"B sites: {b_site_count}")
            imgui.text(f"X sites: {x_site_count}")
            imgui.push_style_color(imgui.Col_.text, (0.95, 0.35, 0.35, 1.0))
            imgui.text_wrapped(str(exc))
            imgui.pop_style_color()
        imgui.text(
            f"Tilt system: {GLAZER_TILT_SYSTEMS[state.perovskite_tilt_system]}"
        )
        imgui.text(
            "Tilt angles: "
            f"a = {state.tilt_angle_x:.1f} deg, "
            f"b = {state.tilt_angle_y:.1f} deg, "
            f"c = {state.tilt_angle_z:.1f} deg"
        )

        if builder_disabled:
            imgui.end_disabled()

    # Builder edits update the active structure in place.
    state.regenerate_focus_from_builder_if_changed()
    state.sync_active_structure()

    imgui.spacing()
    if imgui.collapsing_header("Calculate"):
        if state.focus is not None:
            imgui.text(f"Target: {state.focus.name}")
        imgui.spacing()

        imgui.text("Workflow")
        imgui.separator()
        for step in MAGNETIC_STRUCTURE_STEPS:
            imgui.bullet_text(step)

        imgui.spacing()
        if imgui.collapsing_header("Solver Settings"):
            solver_settings_changed = False
            imgui.push_item_width(220)
            changed, state.magnetic_solver_method = imgui.combo(
                "Solver method",
                state.magnetic_solver_method,
                SPIN_SOLVER_METHODS,
            )
            solver_settings_changed = solver_settings_changed or changed
            changed, state.magnetic_solver_collinear = imgui.checkbox(
                "Collinear solve",
                state.magnetic_solver_collinear,
            )
            solver_settings_changed = solver_settings_changed or changed
            changed, state.magnetic_solver_trials = imgui.input_int(
                "Trials",
                state.magnetic_solver_trials,
                1,
                10,
            )
            solver_settings_changed = solver_settings_changed or changed
            changed, state.magnetic_solver_steps = imgui.input_int(
                "Steps",
                state.magnetic_solver_steps,
                10,
                100,
            )
            solver_settings_changed = solver_settings_changed or changed
            changed, state.magnetic_solver_learning_rate = imgui.input_float(
                "Learning rate",
                state.magnetic_solver_learning_rate,
                0.001,
                0.01,
                "%.6f",
            )
            solver_settings_changed = solver_settings_changed or changed
            changed, state.magnetic_solver_energy_tolerance = imgui.input_float(
                "Energy tolerance",
                state.magnetic_solver_energy_tolerance,
                1e-5,
                1e-4,
                "%.6g",
            )
            solver_settings_changed = solver_settings_changed or changed
            changed, state.magnetic_solver_patience = imgui.input_int(
                "Patience",
                state.magnetic_solver_patience,
                1,
                5,
            )
            solver_settings_changed = solver_settings_changed or changed
            changed, state.magnetic_solver_max_flip_order = imgui.input_int(
                "Max flip order",
                state.magnetic_solver_max_flip_order,
                1,
                5,
            )
            solver_settings_changed = solver_settings_changed or changed
            changed, state.magnetic_solver_max_flip_configs = imgui.input_int(
                "Max flip configs",
                state.magnetic_solver_max_flip_configs,
                1000,
                10000,
            )
            solver_settings_changed = solver_settings_changed or changed
            plot_cap_changed, state.spin_plot_max_configs = imgui.input_int(
                "Max plotted configurations",
                state.spin_plot_max_configs,
                10,
                100,
            )
            state.spin_plot_max_configs = max(1, state.spin_plot_max_configs)
            imgui.pop_item_width()
            imgui.text_disabled("Set max flip configs to 0 or less to represent no limit.")
            imgui.text_disabled(
                "Plotted configurations are kept across structure edits and "
                "re-evaluated; reference orderings are always kept."
            )
            if plot_cap_changed:
                state.refresh_landscape_energies()
            if solver_settings_changed:
                state.magnetic_solution_cache = {}
                state.selected_spin_config_index = 0
                if state.magnetic_oxidation_assignments:
                    state.magnetic_spin_status = (
                        "Solver settings changed. Re-run Magnetic Structure or "
                        "solve the selected oxidation state again to refresh results."
                    )

        imgui.spacing()
        if imgui.button("Run Magnetic Structure", size=(180, 0)):
            state.run_selected_calculation()

    imgui.spacing()
    if imgui.collapsing_header("Rendering"):
        _, state.show_unit_cell = imgui.checkbox("Draw unit cell", state.show_unit_cell)
        _, state.show_spin_classifications = imgui.checkbox(
            "Show spin classifications",
            state.show_spin_classifications,
        )
        if state.focus_is_loaded():
            _, state.use_cartesian = imgui.checkbox(
                "Plot cartesian coordinates", state.use_cartesian
            )
        if state.focus_has_generated_provenance():
            _, state.render_periodic_images = imgui.checkbox(
                "Render periodic images", state.render_periodic_images
            )
            active_periodic = state.focus.is_periodic if state.focus is not None else False
            if not active_periodic:
                imgui.same_line()
                imgui.text_disabled("inactive for non-periodic real structures")
        if state.focus_has_generated_provenance():
            _, state.show_octahedra = imgui.checkbox(
                "Render octahedra", state.show_octahedra
            )
        _, state.render_with_ionic_radius = imgui.checkbox(
            "Render with ionic radius",
            state.render_with_ionic_radius,
        )
        _, state.show_legend = imgui.checkbox("Show species legend", state.show_legend)
        if state.render_with_ionic_radius:
            imgui.text_disabled(
                "Using oxidation-state or Shannon-radius lookups directly."
            )
        else:
            imgui.text_disabled(
                "Ligands use 40% of Fe3+, and other atoms are capped at the Fe3+ radius."
            )

    imgui.spacing()
    if imgui.collapsing_header("Geometry loader"):
        if IS_PYODIDE:
            imgui.text_wrapped(
                "Upload a .cif or VASP/POSCAR file, or drag one onto the 3D view."
            )
            if imgui.button("Upload geometry file..."):
                _open_browser_file_picker()
            imgui.spacing()
        else:
            imgui.push_item_width(-1)
            _, state.geometry_path = imgui.input_text("Geometry path", state.geometry_path)
            imgui.pop_item_width()

            if imgui.button("Load path"):
                state.load_geometry(Path(state.geometry_path))
            imgui.same_line()

        if imgui.button("Load sample asset"):
            state.load_geometry(SAMPLE_GEOMETRY)

        if state.load_error:
            imgui.push_style_color(imgui.Col_.text, (0.95, 0.35, 0.35, 1.0))
            imgui.text_wrapped(state.load_error)
            imgui.pop_style_color()
        elif state.status_message:
            imgui.text_wrapped(state.status_message)

        if state.geometry is not None:
            geometry = state.geometry
            imgui.spacing()
            imgui.text("Loaded structure")
            imgui.separator()
            imgui.text_wrapped(geometry.title)
            imgui.text(f"Formula: {geometry.formula}")
            imgui.text(f"Atoms: {geometry.atom_count}")
            imgui.text(f"Species: {len(geometry.species)}")
            imgui.text(f"File mode: {geometry.coordinate_mode}")
            if state.focus_is_loaded():
                imgui.text(
                    f"Current view: {'cartesian' if state.use_cartesian else 'fractional'}"
                )
            for element, count in zip(geometry.species, geometry.counts):
                imgui.bullet_text(f"{element}: {count}")

def format_cell_classification_counts(fractions) -> str:
    """One-line per-category cell counts, e.g. 'F: 12  A: 0  C: 4  G: 0  E: 8'."""
    return "  ".join(
        f"{name}: {fractions.counts.get(name, 0)}" for name in SPIN_CATEGORIES
    )


def gui_calculation_output() -> None:
    state = APP_STATE

    imgui.text("Magnetic Structure Results")
    imgui.separator()

    # Results belong to whatever structure is currently focused. If the focus has
    # moved elsewhere, prompt rather than showing stale results for another structure.
    if not state.magnetic_results_match_focus():
        imgui.text_wrapped(
            "Run Magnetic Structure on the active structure to see results here."
        )
        return
    if not state.magnetic_oxidation_assignments:
        imgui.text_wrapped(state.baseline_status or state.magnetic_oxidation_status)
        return

    assignment_labels = [
        format_oxidation_assignment_label(assignment, index)
        for index, assignment in enumerate(state.magnetic_oxidation_assignments)
    ]
    state.selected_oxidation_assignment_index = min(
        max(state.selected_oxidation_assignment_index, 0),
        len(assignment_labels) - 1,
    )

    imgui.text(f"Structure: {state.magnetic_result_structure_name}")
    imgui.text(f"Assignments: {len(state.magnetic_oxidation_assignments)}")
    imgui.push_item_width(-1)
    changed, state.selected_oxidation_assignment_index = imgui.combo(
        "Oxidation assignment",
        state.selected_oxidation_assignment_index,
        assignment_labels,
    )
    imgui.pop_item_width()
    if changed:
        state.selected_spin_config_index = 0
        state.run_selected_oxidation_assignment()

    selected_assignment = state.selected_oxidation_assignment()
    result_structure = state.magnetic_result_structure
    if selected_assignment is not None and result_structure is not None:
        imgui.text(
            f"Exchange matrix: {state.magnetic_j_matrix.shape[0]} x "
            f"{state.magnetic_j_matrix.shape[1]}"
            if state.magnetic_j_matrix.size
            else "Exchange matrix unavailable"
        )

    imgui.spacing()
    imgui.separator()

    # --- Spin solver section ---
    if imgui.button("Solve selected oxidation state", size=(220, 0)):
        state.run_selected_oxidation_assignment(force=True)
    all_states = state.displayed_spin_configs()

    selected_config = None
    selected_moments = None
    fractions = None
    if selected_assignment is None:
        imgui.text_wrapped("No oxidation-state assignment is selected.")
    elif not all_states:
        imgui.text_wrapped(state.magnetic_spin_status)
    else:
        selected_config = state.selected_spin_config()
        if selected_config is not None and result_structure is not None:
            selected_moments = state.expand_spin_moments_to_structure(
                selected_config.all_moments, result_structure
            )
            result_build = state.generated_build_for_structure(result_structure)
            if result_build is not None:
                classified_structure = structure_with_moments(
                    result_structure, selected_moments
                )
                result_params = classified_structure.generation_parameters
                result_site_indexing = (
                    site_indexing_from_generation_parameters(result_params, result_build)
                    if result_params is not None
                    else None
                )
                fractions = cube_fractions_for_structure(
                    classified_structure,
                    result_build,
                    site_indexing=result_site_indexing,
                )

        # Per-classification cell counts, above the spin-solver results.
        if fractions is not None:
            imgui.spacing()
            imgui.text(f"Magnetic cells ({fractions.total}) by classification:")
            imgui.text(format_cell_classification_counts(fractions))

        imgui.spacing()
        if selected_config is None:
            imgui.text_wrapped("No spin configurations were returned.")
        else:
            imgui.text(f"Energy: {selected_config.energy:.6f}")
            imgui.text(f"Magnetization: {selected_config.magnetization:.3f}")
            imgui.text(f"Ordering: {state.label_for_config(selected_config)}")

            can_save = result_structure is not None
            if not can_save:
                imgui.begin_disabled()
            if imgui.button("Save magnetic configuration", size=(220, 0)):
                state.save_selected_spin_configuration()
            if not can_save:
                imgui.end_disabled()
            if result_structure is not None:
                imgui.text(
                    f"Saved configurations for '{result_structure.name}': "
                    f"{len(result_structure.spin_configurations)}"
                )
            if state.spin_save_message:
                imgui.text_wrapped(state.spin_save_message)

        imgui.separator()
        list_height = max(120.0, imgui.get_content_region_avail().y * 0.4)
        imgui.text(f"Spin configurations ({len(all_states)})")
        if imgui.begin_child("##spin_config_list", (0.0, list_height), True):
            state.selected_spin_config_index = min(
                max(state.selected_spin_config_index, 0),
                max(len(all_states) - 1, 0),
            )
            config_labels = state.spin_classification_labels()
            for index, config in enumerate(all_states):
                ordering = config_labels[index] if index < len(config_labels) else "Other"
                label = (
                    f"{index + 1:>3}. E={config.energy:.6f}  "
                    f"M={config.magnetization:.3f}  {ordering}##spin_config_{index}"
                )
                clicked, _ = imgui.selectable(
                    label, state.selected_spin_config_index == index
                )
                if clicked:
                    state.selected_spin_config_index = index
        imgui.end_child()

    # --- Per-atom oxidation states + moments (magnetic and non-magnetic) ---
    if selected_assignment is not None and result_structure is not None:
        imgui.separator()
        imgui.text_wrapped(
            format_oxidation_assignment_details(
                result_structure,
                selected_assignment,
                site_moments=selected_moments,
            )
        )


def structure_plot_view() -> Tuple[str, np.ndarray, str, bool]:
    state = APP_STATE
    state.sync_active_structure()
    structure = state.rendered_structure()
    if structure is None:
        raise ValueError("No structure is currently focused.")

    use_cartesian = (not state.focus_is_loaded()) or state.use_cartesian
    coords = structure.cartesian_coords if use_cartesian else structure.fractional_coords
    title = structure.name if state.focus is None else state.focus.name
    return (
        title,
        coords,
        "A" if use_cartesian else "fractional",
        use_cartesian,
    )


def structure_signature(state: AppState) -> Tuple[object, ...]:
    state.sync_active_structure()
    structure = state.focus
    return (
        "structure",
        id(structure),
        structure.name if structure is not None else "",
        structure.atom_count if structure is not None else 0,
        state.active_saved_spin_index,
        state.render_periodic_images if state.focus_has_generated_provenance() else False,
        state.show_octahedra if state.focus_has_generated_provenance() else False,
        tuple(np.round(structure.lattice.flatten(), 6)) if structure is not None else (),
        tuple(np.round(structure.cartesian_coords.flatten(), 6)) if structure is not None else (),
    )


def spin_plot_category(label: str) -> str:
    """Scatter-plot category for a label; anything without a color renders as Other."""
    return label if label in SPIN_CLASS_COLORS else "Other"


def plot_spin_energy_scatter(state: "AppState") -> None:
    """2D ImPlot pane: ΔE-from-ground-state vs rank, colored by exact reference match.

    The points persist across builder edits -- only their energies are recomputed --
    so this tracks the structure instead of resetting. Clicking the nearest point
    selects that spin configuration, mirroring a click in the spin-results list.
    """
    configs = state.displayed_spin_configs()

    if not configs:
        imgui.text_disabled(
            state.baseline_status
            or "Build or select a structure to see its reference configurations."
        )
    elif not state.magnetic_solution_cache:
        imgui.text_disabled(
            "Reference configurations only - run Magnetic Structure for the full landscape."
        )

    if not implot.begin_plot("Spin energy landscape##spin_energy", size=(-1, -1)):
        return

    implot.setup_axes("Rank", "ΔE")

    if configs:
        labels = state.spin_classification_labels()
        ranks = np.arange(1, len(configs) + 1, dtype=np.float64)
        e0 = float(configs[0].energy)
        delta_e = np.array(
            [float(config.energy) - e0 for config in configs], dtype=np.float64
        )
        categories = [
            spin_plot_category(labels[index] if index < len(labels) else "Other")
            for index in range(len(configs))
        ]
        categories_arr = np.array(categories, dtype=object)

        # Refit when the landscape's shape changes -- a new solve, or energies that
        # moved after a builder edit -- then leave the axes free to pan/zoom. Keyed on
        # content rather than list identity because the list is rebuilt on every
        # re-energization, which would otherwise refit on every frame.
        e_max = float(delta_e.max())
        span = e_max  # delta_e is measured from the ground state, so e_min == 0.
        if span <= 1e-12:
            y_lo, y_hi = -0.5, 0.5
        else:
            y_margin = span * 0.04
            y_lo, y_hi = -y_margin, e_max + y_margin
        axis_key = (len(configs), round(e_max, 6))
        if axis_key != state._spin_plot_axis_solution:
            state._spin_plot_axis_solution = axis_key
            # x starts just left of rank 1 and y dips slightly below 0 so the
            # lowest-ranked, lowest-energy point is clearly visible.
            implot.setup_axis_limits(
                implot.ImAxis_.x1, 0.95, len(configs) + 0.05, implot.Cond_.always
            )
            implot.setup_axis_limits(
                implot.ImAxis_.y1, y_lo, y_hi, implot.Cond_.always
            )

        for category in SPIN_PLOT_CATEGORIES:
            mask = categories_arr == category
            # Only show a classification in the legend if it actually occurs.
            if int(mask.sum()) == 0:
                continue
            color = imgui.ImVec4(*SPIN_CLASS_COLORS[category])
            spec = implot.Spec()
            spec.marker = implot.Marker_.circle
            spec.marker_size = 4.0
            spec.marker_fill_color = color
            spec.marker_line_color = color
            spec.line_color = color
            xs = np.ascontiguousarray(ranks[mask], dtype=np.float64)
            ys = np.ascontiguousarray(delta_e[mask], dtype=np.float64)
            implot.plot_scatter(category, xs, ys, spec)

        # Nearest point to the cursor (pixel space), recomputed each frame so the
        # hovered point can be ringed and clicking it selects that configuration.
        hovered_index = -1
        if implot.is_plot_hovered():
            mouse = imgui.get_mouse_pos()
            nearest_dist_sq = 12.0 * 12.0
            for index in range(len(configs)):
                pixel = implot.plot_to_pixels(
                    float(ranks[index]), float(delta_e[index])
                )
                dx = pixel.x - mouse.x
                dy = pixel.y - mouse.y
                dist_sq = dx * dx + dy * dy
                if dist_sq <= nearest_dist_sq:
                    nearest_dist_sq = dist_sq
                    hovered_index = index
            if hovered_index >= 0 and imgui.is_mouse_clicked(0):
                state.selected_spin_config_index = hovered_index

        no_legend = int(implot.ItemFlags_.no_legend)

        # Selected marker (white ring); excluded from the legend.
        selected_index = state.selected_spin_config_index
        if 0 <= selected_index < len(configs):
            spec = implot.Spec()
            spec.marker = implot.Marker_.circle
            spec.marker_size = 8.0
            spec.marker_fill_color = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
            spec.marker_line_color = imgui.ImVec4(1.0, 1.0, 1.0, 1.0)
            spec.flags = no_legend
            implot.plot_scatter(
                "##selected_point",
                np.array([ranks[selected_index]], dtype=np.float64),
                np.array([delta_e[selected_index]], dtype=np.float64),
                spec,
            )

        # Hovered marker (yellow ring); also excluded from the legend.
        if 0 <= hovered_index < len(configs):
            spec = implot.Spec()
            spec.marker = implot.Marker_.circle
            spec.marker_size = 11.0
            spec.marker_fill_color = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
            spec.marker_line_color = imgui.ImVec4(1.0, 0.92, 0.16, 1.0)
            spec.flags = no_legend
            implot.plot_scatter(
                "##hovered_point",
                np.array([ranks[hovered_index]], dtype=np.float64),
                np.array([delta_e[hovered_index]], dtype=np.float64),
                spec,
            )

    implot.end_plot()


def gui_structure_view() -> None:
    state = APP_STATE
    state.sync_active_structure()
    real_structure = state.active_structure
    rendered_structure = state.rendered_structure()
    if real_structure is None or rendered_structure is None:
        imgui.text_wrapped(
            "Select a structure in the Active Structure panel, or edit the "
            "Builder preview, to populate the 3D view."
        )
        return

    title, coords, axis_label, use_cartesian = structure_plot_view()
    assert real_structure is not None
    structure = rendered_structure

    # Moments to display, in priority order: a saved spin config selected in the
    # tree, then the builder/solver moments matching the focus, then the
    # structure's own moments.
    selected_spin_moments = state.displayed_saved_spin_moments(structure)
    showing_saved_config = selected_spin_moments is not None
    if selected_spin_moments is None:
        selected_spin_moments = state.selected_spin_moments_for_structure(structure)

    site_oxidation_states = structure_site_oxidation_states(state, structure)
    atom_radii = structure_atom_render_radii(
        structure,
        site_oxidation_states,
        render_with_ionic_radius=state.render_with_ionic_radius,
    )
    flags = implot3d.Flags_.equal.value
    if not state.show_legend:
        flags |= implot3d.Flags_.no_legend.value

    alignment_counts: dict[str, int] | None = None
    rendered_build = state.generated_build_for_structure(structure)
    # The badge names the exact ordering of the configuration actually drawn, matching
    # the plot legend, rather than the nearest-neighbour similarity vote it used to
    # show. A saved config picked in the tree carries the label it was saved with.
    spin_ordering: str | None = None
    if showing_saved_config:
        saved = state.displayed_saved_spin_configuration()
        spin_ordering = (saved.classification or "Other") if saved is not None else None
    else:
        selected_config = state.selected_spin_config()
        if selected_config is not None:
            spin_ordering = state.label_for_config(selected_config)
    if rendered_build is not None and selected_spin_moments is not None:
        alignment_counts = spin_alignment_edge_counts(
            structure.cartesian_coords, rendered_build, selected_spin_moments
        )
    imgui.text(f"3D atomic spheres from {title} ({axis_label} coordinates)")

    if state.show_spin_classifications and spin_ordering is not None:
        imgui.text(f"Spin ordering: {spin_ordering}")
        if alignment_counts is not None:
            imgui.text(
                "Visible NN edges: "
                f"{alignment_counts['aligned']} aligned, "
                f"{alignment_counts['anti-aligned']} anti-aligned"
            )
    displayed_spin_moments = (
        selected_spin_moments
        if selected_spin_moments is not None
        else structure.magnetic_moments
    )
    displayed_spin_signs = spin_signs_from_moments(displayed_spin_moments)
    imgui.separator()

    structure_changed = state.refresh_plot_view_generation(structure_signature(state))
    plot_coords = coords
    plot_axis_extents = sphere_axis_extents(atom_radii, structure.lattice, use_cartesian)
    if state.show_unit_cell:
        unit_cell_coords = unit_cell_vertices(structure.lattice, axis_label == "A")
        plot_coords = np.vstack((plot_coords, unit_cell_coords))
        plot_axis_extents = np.vstack(
            (
                plot_axis_extents,
                np.zeros((unit_cell_coords.shape[0], 3), dtype=np.float64),
            )
        )
    plot_limits = compute_plot_box_limits(
        plot_coords,
        axis_extents=plot_axis_extents,
    )
    plot_id = "Atomic coordinates##structure_view"

    # Reserve the lower portion of the pane for the 2D spin-energy scatter.
    available = imgui.get_content_region_avail()
    energy_pane_height = max(180.0, available.y * 0.30)
    plot3d_height = max(160.0, available.y - energy_pane_height - 8.0)

    if implot3d.begin_plot(plot_id, size=(-1, plot3d_height), flags=flags):
        axis_flags = implot3d.AxisFlags_.no_grid_lines.value
        implot3d.setup_axes(
            "a",
            "b",
            "c",
            axis_flags,
            axis_flags,
            axis_flags,
        )
        axes_limit_condition = (
            implot3d.Cond_.always if structure_changed else implot3d.Cond_.once
        )
        implot3d.setup_axes_limits(*plot_limits, axes_limit_condition)

        if state.show_unit_cell:
            plot_unit_cell(structure.lattice, use_cartesian=axis_label == "A")

        if (
            state.show_spin_classifications
            and rendered_build is not None
            and selected_spin_moments is not None
        ):
            plot_classification_lattice(coords, rendered_build, selected_spin_moments)

        if (
            state.show_octahedra
            and rendered_build is not None
            and getattr(structure, "generation_parameters", None) is not None
        ):
            triangle_vertices = octahedron_triangles_for_generated_structure(
                structure,
                rendered_build,
            )
            xs_tri = np.ascontiguousarray(triangle_vertices[:, 0], dtype=np.float64)
            ys_tri = np.ascontiguousarray(triangle_vertices[:, 1], dtype=np.float64)
            zs_tri = np.ascontiguousarray(triangle_vertices[:, 2], dtype=np.float64)
            spec_tri = implot3d.Spec(
                fill_color=implot3d.get_colormap_color(0),
                line_color=implot3d.get_colormap_color(1),
                marker=implot3d.Marker_.none,
                fill_alpha=0.28,
            )
            implot3d.plot_triangle("Octahedra", xs_tri, ys_tri, zs_tri, spec=spec_tri)

        grouped_indices: Dict[tuple[str, tuple[float, float, float, float]], List[int]] = {}
        for atom_index, element in enumerate(structure.atomic_labels):
            color = ELEMENT_RENDER_COLORS.get(element, DEFAULT_ELEMENT_RENDER_COLOR)
            label = element
            if (
                displayed_spin_signs is not None
                and atom_index < len(displayed_spin_signs)
                and displayed_spin_signs[atom_index] != 0
            ):
                if displayed_spin_signs[atom_index] > 0:
                    label = "Spin up"
                    color = SPIN_UP_COLOR
                else:
                    label = "Spin down"
                    color = SPIN_DOWN_COLOR
            grouped_indices.setdefault((label, color), []).append(atom_index)

        for (label, color), element_indices in grouped_indices.items():
            element_coords = ensure_xyz_array(coords[element_indices])
            element_radii = np.asarray(atom_radii[element_indices], dtype=np.float64)
            if element_coords.shape[0] == 0:
                continue
            mesh = build_sphere_mesh(
                element_coords,
                element_radii,
                structure.lattice,
                use_cartesian=use_cartesian,
            )
            spec = implot3d.Spec(
                fill_color=color,
                line_color=color,
                fill_alpha=0.92,
                flags=implot3d.MeshFlags_.no_lines.value,
            )
            implot3d.plot_mesh(label, mesh, spec=spec)

        implot3d.end_plot()

    imgui.separator()
    plot_spin_energy_scatter(state)


def gui_export() -> None:
    state = APP_STATE
    imgui.text("Export structures to disk")
    imgui.text_wrapped(
        "Writes one CIF per structure plus '<name>_spins.txt' (VASP magmoms, one "
        "line per saved magnetic configuration) into <folder>/."
    )
    imgui.separator()

    imgui.push_item_width(220)
    _, state.export_directory = imgui.input_text(
        "Output folder",
        state.export_directory,
    )
    imgui.pop_item_width()
    imgui.same_line()
    if imgui.button("Browse..."):
        try:
            selection = pfd.select_folder("Select export folder").result()
        except Exception as exc:
            state.export_message = f"Folder dialog failed: {exc}"
            selection = ""
        if selection:
            state.export_directory = selection

    if imgui.button("Export active structure", size=(180, 0)):
        state.export_active_structure()
    if imgui.button("Export all structures", size=(180, 0)):
        state.export_all_structures()

    if state.export_message:
        imgui.spacing()
        imgui.text_wrapped(state.export_message)


def _active_structure_leaf(
    state: "AppState",
    structure: ChemicalStructure,
    registry: list,
    *,
    selected: bool,
) -> None:
    """A structure row that has no saved spin configs (rendered as a leaf)."""
    reg_id = len(registry)
    registry.append(structure)
    clicked, _ = imgui.selectable(f"{structure.name}##struct{reg_id}", selected)
    if clicked:
        state.set_focus(structure)
    _structure_context_menu(state, structure)


def _structure_context_menu(state: "AppState", structure: ChemicalStructure) -> None:
    if imgui.begin_popup_context_item():
        if imgui.menu_item("Rename", "", False)[0]:
            state._rename_target = structure
            state._rename_buffer = structure.name
            state._rename_request = True
        if imgui.menu_item("Delete structure", "", False)[0]:
            state._pending_structure_delete = structure
        imgui.end_popup()


def _render_structure_with_configs(
    state: "AppState",
    structure: ChemicalStructure,
    registry: list,
) -> None:
    reg_id = len(registry)
    registry.append(structure)
    flags = (
        imgui.TreeNodeFlags_.open_on_arrow.value
        | imgui.TreeNodeFlags_.span_full_width.value
    )
    if structure is state.focus and state.active_saved_spin_index < 0:
        flags |= imgui.TreeNodeFlags_.selected.value
    opened = imgui.tree_node_ex(f"{structure.name}##struct{reg_id}", flags)
    if imgui.is_item_clicked() and not imgui.is_item_toggled_open():
        state.set_focus(structure)
    _structure_context_menu(state, structure)
    if opened:
        for config_index, config in enumerate(structure.spin_configurations):
            label = (
                f"#{config_index + 1}  {config.classification or '?'}  "
                f"E={config.energy:.4f}##cfg{reg_id}_{config_index}"
            )
            is_active = (
                structure is state.focus
                and state.active_saved_spin_index == config_index
            )
            clicked, _ = imgui.selectable(label, is_active)
            if clicked:
                state.set_focus(structure)
                state.active_saved_spin_index = config_index
        imgui.tree_pop()


def _render_structure_row(
    state: "AppState", structure: ChemicalStructure, registry: list
) -> None:
    if structure.spin_configurations:
        _render_structure_with_configs(state, structure, registry)
    else:
        _active_structure_leaf(
            state,
            structure,
            registry,
            selected=(structure is state.focus and state.active_saved_spin_index < 0),
        )


RENAME_POPUP_ID = "Rename structure##rename_structure"


def _rename_structure_popup(state: "AppState") -> None:
    """Right-click rename. open_popup and begin_popup share the pane's ID scope."""
    if state._rename_request:
        imgui.open_popup(RENAME_POPUP_ID)
        state._rename_request = False
    if not imgui.begin_popup(RENAME_POPUP_ID):
        return
    if imgui.is_window_appearing():
        imgui.set_keyboard_focus_here()
    imgui.push_item_width(200)
    entered, state._rename_buffer = imgui.input_text(
        "##rename_field",
        state._rename_buffer,
        imgui.InputTextFlags_.enter_returns_true.value
        | imgui.InputTextFlags_.auto_select_all.value,
    )
    imgui.pop_item_width()
    commit = entered
    if imgui.button("Rename"):
        commit = True
    imgui.same_line()
    if imgui.button("Cancel"):
        imgui.close_current_popup()
        commit = False
    if commit:
        if state._rename_target is not None:
            state.rename_structure(state._rename_target, state._rename_buffer)
        imgui.close_current_popup()
    imgui.end_popup()


def gui_active_structure() -> None:
    state = APP_STATE

    if imgui.button("New structure##active_structure"):
        state.create_new_structure()
    imgui.text_disabled("Right-click a structure to rename or delete it.")
    imgui.separator()

    registry: list = []
    for structure in list(state.structures):
        _render_structure_row(state, structure, registry)

    _rename_structure_popup(state)

    # Apply the deferred context-menu deletion after the list render.
    if state._pending_structure_delete is not None:
        state.remove_structure(state._pending_structure_delete)
        state._pending_structure_delete = None


def create_docking_splits() -> List[hello_imgui.DockingSplit]:
    split_left = hello_imgui.DockingSplit()
    split_left.initial_dock = "MainDockSpace"
    split_left.new_dock = "ControlsSpace"
    split_left.direction = imgui.Dir.left
    split_left.ratio = 0.20

    split_right = hello_imgui.DockingSplit()
    split_right.initial_dock = "MainDockSpace"
    split_right.new_dock = "CalculationOutputSpace"
    split_right.direction = imgui.Dir.right
    split_right.ratio = 0.30

    split_export = hello_imgui.DockingSplit()
    split_export.initial_dock = "ControlsSpace"
    split_export.new_dock = "ExportSpace"
    split_export.direction = imgui.Dir.down
    split_export.ratio = 0.30

    split_active = hello_imgui.DockingSplit()
    split_active.initial_dock = "CalculationOutputSpace"
    split_active.new_dock = "ActiveStructureSpace"
    split_active.direction = imgui.Dir.up
    split_active.ratio = 0.45
    return [split_left, split_right, split_export, split_active]


def create_dockable_windows() -> List[hello_imgui.DockableWindow]:
    controls = hello_imgui.DockableWindow()
    controls.label = "Controls"
    controls.dock_space_name = "ControlsSpace"
    controls.gui_function = gui_controls

    structure = hello_imgui.DockableWindow()
    structure.label = "Structure View"
    structure.dock_space_name = "MainDockSpace"
    structure.gui_function = gui_structure_view

    calculation_output = hello_imgui.DockableWindow()
    calculation_output.label = "Calculation Output"
    calculation_output.dock_space_name = "CalculationOutputSpace"
    calculation_output.gui_function = gui_calculation_output

    export = hello_imgui.DockableWindow()
    export.label = "Export"
    export.dock_space_name = "ExportSpace"
    export.gui_function = gui_export

    active = hello_imgui.DockableWindow()
    active.label = "Active Structure"
    active.dock_space_name = "ActiveStructureSpace"
    active.gui_function = gui_active_structure

    windows = [controls, structure, calculation_output, export, active]
    # Every panel is part of the fixed layout, so hide the tab close button.
    for window in windows:
        window.can_be_closed = False
    return windows


def create_runner_params() -> hello_imgui.RunnerParams:
    params = hello_imgui.RunnerParams()
    params.app_window_params.window_title = "Quick Mag"
    params.app_window_params.window_geometry.size = (1760, 1160)

    # The desktop and Pyodide wheels can expose slightly different RunnerParams fields.
    # Prefer clearing prior layout state when available; otherwise disable ini persistence.
    if hasattr(params, "ini_clear_previous_settings"):
        params.ini_clear_previous_settings = True
    elif hasattr(params, "ini_disable"):
        params.ini_disable = True

    params.imgui_window_params.default_imgui_window_type = (
        hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
    )
    if hasattr(params.imgui_window_params, "show_menu_bar"):
        params.imgui_window_params.show_menu_bar = True
    if hasattr(params.imgui_window_params, "show_menu_view"):
        params.imgui_window_params.show_menu_view = True

    params.docking_params.docking_splits = create_docking_splits()
    params.docking_params.dockable_windows = create_dockable_windows()
    if hasattr(params.docking_params, "layout_condition"):
        params.docking_params.layout_condition = (
            hello_imgui.DockingLayoutCondition.application_start
        )
    return params


def ensure_pyodide_runner_patch() -> None:
    try:
        import js  # type: ignore  # noqa: F401
    except ImportError:
        return

    try:
        from imgui_bundle.pyodide_patch_runners import pyodide_do_patch_runners
    except Exception:
        return

    pyodide_do_patch_runners()


def main() -> None:
    ensure_pyodide_runner_patch()
    add_ons = immapp.AddOnsParams()
    add_ons.with_implot = True
    add_ons.with_implot3d = True
    immapp.run(runner_params=create_runner_params(), add_ons_params=add_ons)


if __name__ == "__main__":
    main()
