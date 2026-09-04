from __future__ import annotations

import base64
import copy
import math
import time
from collections import OrderedDict
from dataclasses import MISSING, dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from quick_mag.atom_edits import (
    SelectionSlab,
    VacatedAtom,
    add_proton,
    append_atom,
    atoms_in_slab,
    remove_atom,
    slab_center_distance,
    slab_face_corners,
    slab_normal,
    slab_offset_from_distance,
    substitute_atom,
)
from quick_mag.defects import (
    PROTON_ORIENTATION_COUNT,
    SiteDefect,
    SiteKey,
    apply_defects,
    canonicalize_key,
    coerce_site_key,
    compensation_hint,
    resolve_defects,
    site_key_display,
    vacated_b_cells,
)
from quick_mag.classify_spin_structure import (
    PerovskiteSiteIndexing,
    site_indexing_from_generation_parameters,
    site_indexing_from_magnetic_sublattice,
)
from quick_mag.cell_edit import (
    MAX_CELL_ANGLE,
    MIN_CELL_ANGLE,
    MIN_CELL_LENGTH,
    cell_parameters,
    lattice_from_parameters,
    strain_structure,
    tile_structure,
)
from quick_mag.cif_io import read_cif
from quick_mag.geometry_drift import GeometryDrift, geometry_drift
from quick_mag.remote import protocol as remote_protocol
from quick_mag.remote.client import DEFAULT_URL as REMOTE_DEFAULT_URL, RemoteClient
from quick_mag.constants import ELEMENT_RENDER_COLORS, LIGANDS
from quick_mag.element_data import is_valid_symbol
from quick_mag.ion_descriptors import structure_ion_descriptors
from quick_mag.magnetic_moments import (
    OxidationStateAssignment,
    expand_group_assignments_to_site_assignments,
    format_oxidation_distribution,
)
from quick_mag.oxidation_overrides import (
    OxidationOverrides,
    assignment_with_overrides,
    resolve_overrides as resolve_oxidation_overrides,
)
from quick_mag.oxidation_state_energy import enumerate_site_group_assignments
from quick_mag.polarization_model import (
    PairCoupling,
    build_Jeff_matrix,
    build_bridges,
    default_params,
    pair_couplings,
    to_solver_couplings,
)
from quick_mag.reference_configs import (
    magnetic_sublattice_for,
    named_reference_spin_configs,
)
from quick_mag.spin_planes import (
    CANONICAL_PLANE_PATTERNS,
    MagneticSublattice,
    PatternMatch,
    PlanePattern,
    best_matching_pattern,
    build_plane_index,
    format_miller,
    parse_plane_label,
    patterns_for_sites,
    plane_cell_polygon,
    plane_count,
    plane_indices,
    polygon_triangles,
    signs_from_ordinals,
)
from quick_mag.domains import (
    AXIS_NAMES as DOMAIN_AXIS_NAMES,
    DomainSpec,
    conform_domain_to_stack,
    domain_composition,
    in_plane_axes,
    stack_half_lengths,
    stack_lattice,
    stack_oct_counts,
    validate_stack,
)
from quick_mag.perovskite_builder import (
    PerovskiteBuild,
    active_glazer_parameter_axes,
    active_tilt_axes,
    build_perovskite,
    canonical_site_keys,
    canonicalize_glazer_tilt_angles_deg,
    octahedron_triangle_vertices,
    parse_glazer_tilt_system,
    periodic_axes,
)
from quick_mag.structure import (
    ChemicalStructure,
    PerovskiteGenerationParameters,
    SavedSpinConfiguration,
    build_from_generation_parameters,
)
from quick_mag.export_utils import (
    export_bundle_bytes,
    export_structure,
    export_structures,
)
from quick_mag.generation import (
    domain_atomic_labels_for_build,
    formula_atomic_labels_for_build,
    formula_atomic_labels_from_parameters,
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

# Remote compute. The lists are indices into imgui combos, so their order is part
# of the UI rather than of the protocol.
REMOTE_CALCULATIONS = list(remote_protocol.CALCULATIONS)
# What the combo shows for each protocol key: a single point is not an
# optimization, and the labels say so rather than leaving it to the reader.
REMOTE_CALCULATION_LABELS = {
    "single-point": "Single point (energy only)",
    "atoms": "Optimize atoms (cell fixed)",
    "cell": "Optimize cell (atoms fixed)",
    "cell+atoms": "Optimize cell + atoms",
}
REMOTE_OPTIMIZERS = list(remote_protocol.OPTIMIZERS)
REMOTE_CALCULATION_HINTS = {
    "single-point": "Energy, forces and |m| at the geometry as it stands.",
    "atoms": "Relax the atomic positions, hold the lattice fixed.",
    "cell": "Relax the lattice, hold the atomic positions fixed.",
    "cell+atoms": "Relax both. The usual choice.",
}
RELAXATION_TRACE_COLOR = (0.45, 0.75, 1.00, 1.0)
REMOTE_STATUS_COLORS = {
    remote_protocol.STATUS_QUEUED: (0.70, 0.74, 0.82, 1.0),
    remote_protocol.STATUS_RUNNING: (0.45, 0.75, 1.00, 1.0),
    remote_protocol.STATUS_DONE: (0.45, 0.85, 0.60, 1.0),
    remote_protocol.STATUS_ERROR: (0.95, 0.45, 0.45, 1.0),
    remote_protocol.STATUS_CANCELLED: (0.80, 0.70, 0.45, 1.0),
}


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


def _drain_remote_jobs(state: "AppState") -> None:
    """Advance every in-flight remote calculation by one frame.

    Called once a frame beside :func:`_drain_browser_uploads`, and for the same
    reason: the work happens somewhere else and its results have to be collected
    without the loop ever waiting. The client does the rate limiting -- this is
    cheap on the frames where nothing has arrived, which is nearly all of them.

    Only jobs that reached a terminal state on this frame are returned, so each
    relaxed structure is added exactly once.
    """
    client = state.remote_client_if_any()
    if client is None:
        return
    for job in client.poll():
        state.collect_remote_job(job)


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


def _download_via_browser(filename: str, payload: bytes, mime_type: str) -> bool:
    """Hand ``payload`` to the browser as a download. False when unavailable.

    The mirror image of the upload path: the JavaScript lives in ``web/index.html``
    (as ``window.quickMagDownload``) and Python calls into it, so the page owns all
    the DOM work. Base64 rather than raw bytes because the same call has to carry
    both the text CIFs and the binary zip across the FFI boundary.

    A browser tab still running a cached ``index.html`` from before this existed has
    no such function; that returns False so the caller can say so rather than raise.
    """
    if not IS_PYODIDE:
        return False
    try:
        import js  # type: ignore
    except ImportError:
        return False
    download = getattr(js.window, "quickMagDownload", None)
    if download is None:
        return False
    download(filename, base64.b64encode(payload).decode("ascii"), mime_type)
    return True


DEFAULT_ELEMENT_RENDER_COLOR = (0.72, 0.72, 0.72, 1.0)
DEFAULT_ATOM_RENDER_RADIUS = 0.8
LIGAND_RADIUS_SCALE = 0.4
SPHERE_LATITUDE_SEGMENTS = 8
SPHERE_LONGITUDE_SEGMENTS = 16
# The 3D view's cost is close to linear in total vertex count, so a big cell steps
# down to a coarser sphere. Measured on a 1315-atom render: 8x16 (114 verts/atom)
# is 41 ms a frame, 6x12 is 23 ms, 5x10 is 17 ms. At that density each sphere is a
# few pixels across and the facets are invisible; anything small enough to inspect
# closely keeps the full tessellation.
SPHERE_DETAIL_LEVELS: Tuple[Tuple[int | None, Tuple[int, int]], ...] = (
    (400, (SPHERE_LATITUDE_SEGMENTS, SPHERE_LONGITUDE_SEGMENTS)),
    (1200, (6, 12)),
    (None, (5, 10)),
)


def sphere_detail_for(atom_count: int) -> Tuple[int, int]:
    """Sphere tessellation to draw a structure of this size with."""
    for limit, detail in SPHERE_DETAIL_LEVELS:
        if limit is None or atom_count <= limit:
            return detail
    return SPHERE_DETAIL_LEVELS[-1][1]
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
# Vacancies are drawn in vivid fuchsia at the radius of the species that is
# missing. No element's CPK colour is anywhere near this hue, so a hole can never
# be mistaken for an atom -- white would collide with hydrogen.
VACANCY_RENDER_COLOR = (1.0, 0.11, 0.81, 1.0)
# Ring drawn around the site selected in the per-site table. Reuses the vacancy
# fuchsia: nothing else in the view is near that hue, and the yellow it used to be
# now collides with the spin-down colour below.
SITE_HIGHLIGHT_COLOR = VACANCY_RENDER_COLOR
SPIN_UP_COLOR = (0.10, 0.80, 0.78, 1.0)
SPIN_DOWN_COLOR = (0.95, 0.85, 0.15, 1.0)
# Ring drawn around a magnetic site whose spin disagrees with the matched ideal
# ordering. Deliberately not the picked-site colour: both rings can be on screen at
# once and they mean different things.
SPIN_DEFECT_RING_COLOR = (1.0, 0.35, 0.10, 1.0)
# Ring drawn around every atom picked by hand, in every view. Green because the
# other rings are taken -- white for the cursor, fuchsia for the site the
# oxidation table is on, orange for a spin that disagrees -- and a pick has to be
# tellable from all three. Drawn a little wider than the rest so that hovering a
# picked atom shows both rings rather than one over the other.
PICKED_ATOM_RING_COLOR = (0.30, 1.00, 0.45, 1.0)
PICKED_ATOM_RING_SCALE = 2.2
# Translucent sheets for the Miller-plane overlay, and the most it will draw before
# the view turns into a solid block.
MILLER_PLANE_ALPHA = 0.16
MILLER_PLANE_NEUTRAL_COLOR = (0.60, 0.72, 0.95, 1.0)
# Octahedral cages: a translucent blue body with thin white edges.
OCTAHEDRON_FILL_COLOR = (0.30, 0.48, 0.86, 1.0)
OCTAHEDRON_EDGE_COLOR = (0.40, 0.6, 1.0, 1.0)
OCTAHEDRON_EDGE_WEIGHT = 0.75
OCTAHEDRON_ALPHA = 0.28
# How close the cursor has to be to an atom's centre to pick it, and the screen
# distance within which two atoms count as a tie and depth decides instead.
PICK_RADIUS_PIXELS = 16.0
PICK_HOVER_COLOR = (1.0, 1.0, 1.0, 0.95)
# A hand-set oxidation state, and one the net charge says does not balance. The
# first is amber rather than red: an edit is not an error, it is just not the
# model's answer any more, and the panel has to be able to say which sites are
# yours at a glance.
OXIDATION_EDITED_COLOR = (1.0, 0.78, 0.28, 1.0)
OXIDATION_UNBALANCED_COLOR = (1.0, 0.45, 0.40, 1.0)
# Range the oxidation-state box accepts. Wide enough for every ion anyone has a
# use for and narrow enough that a slipped keystroke cannot ask the
# electron-configuration tables for something absurd.
MIN_EDITABLE_OXIDATION_STATE = -8
MAX_EDITABLE_OXIDATION_STATE = 9
# The structure summary floated over the corner of the 3D view. The box is held
# at least as wide as a tilt line at full deflection so that dragging a tilt
# slider does not make the readout beside it twitch.
SUMMARY_OVERLAY_BG_ALPHA = 0.72
SUMMARY_WIDEST_TILT_ROW = (
    "Tilt angles: a = -45.0 deg, b = -45.0 deg, c = -45.0 deg"
)
# A symbol that no element table knows. Not an error -- just a note.
UNKNOWN_ELEMENT_COLOR = (0.95, 0.78, 0.25, 1.0)
PICK_TIE_PIXELS = 6.0
# Atoms outside the selection slab, or off the selected exchange paths, are
# kept as faint translucent context rather than hidden: the structure around
# a defect is most of what you are judging it against. The cages get the same
# treatment from a lower baseline -- they are translucent to begin with, so the
# faded ones go most of the way to nothing. The slab's own atoms draw fully
# opaque, so the layer being worked in reads solid.
FADED_ATOM_ALPHA = 0.14
PROMINENT_ATOM_ALPHA = 0.97
FADED_OCTAHEDRON_ALPHA = 0.05
MAX_DRAWN_MILLER_PLANES = 24
# The selection slab: two translucent faces the cell's width, and the arrow
# along its normal that says which way "up" the [u v w] direction is.
SLAB_FACE_COLOR = (0.60, 0.72, 0.95, 1.0)
SLAB_FACE_HOVER_COLOR = (0.80, 0.88, 1.00, 1.0)
SLAB_FACE_ALPHA = 0.18
SLAB_ARROW_COLOR = (1.0, 0.85, 0.1, 1.0)
SLAB_ARROW_HEAD_PX = 12.0
# Unit-cell wireframe, drawn as one NaN-separated line rather than sampled points.
UNIT_CELL_LINE_COLOR = (0.88, 0.88, 0.88, 1.0)
# The cell of the domain the builder is editing, drawn over the structure so the
# block being worked on is the one that stands out.
DOMAIN_CELL_LINE_COLOR = (1.0, 0.85, 0.1, 1.0)
DOMAIN_CELL_LINE_WEIGHT = 2.0
UNIT_CELL_LINE_WEIGHT = 1.5
# Two configurations count as degenerate when their energies agree to this much;
# matches the deduplication tolerance in spin_solver.sort_and_rank.
DEGENERACY_ENERGY_TOL = 1e-6
# Padding around the structure in the 3D view at zoom 1.0, and the zoom limits. The
# view box is recomputed every frame to keep the cell centred, so zoom is applied here
# rather than through ImPlot3D's own (axis-limit based) zoom.
STRUCTURE_PLOT_PADDING = 1.8
STRUCTURE_ZOOM_RANGE = (0.25, 6.0)
# ImPlot3D's own opening pose, so taking the rotation over does not change the view the
# app starts in and double right-click still lands back on it. Theirs is written
# (-0.513269, -0.212596, -0.318184, 0.76819) in implot3d_internal.h, six figures that come
# out a few 1e-7 short of unit length; these are the same numbers normalised, because the
# view is composed onto this on every drag and only an exact rotation is left untouched by
# a turn and its inverse.
DEFAULT_STRUCTURE_ROTATION = (
    -0.5132692413564485,
    -0.2125960999698317,
    -0.3181841496208815,
    0.7681903612289271,
)
STRUCTURE_ROTATE_RADIANS_PER_PIXEL = math.radians(0.6)
# How far one press of the turn buttons rotates the view.
STRUCTURE_TURN_STEP_DEGREES = 5.0
# The screen-space axes the x/y/z buttons offer, written in ImPlot3D's view frame: x to
# the right, y up, z into the screen. ImPlot3D's own z points the other way, out at the
# viewer, which is why the third entry is negated.
#
# Note those three directions are a left-handed set (x cross y points *out* of the screen,
# not into it), so "right-hand rule about z" is ambiguous if read off the labels. The turn
# direction is therefore taken as the right-hand rule about each axis's physical
# direction, which is what the viewer actually sees: +5 about z spins the picture
# clockwise, +5 about x brings the top of the cell towards you.
SCREEN_TURN_AXES = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, -1.0),
)
SCREEN_TURN_AXIS_TOOLTIPS = (
    "Screen x: to the right.\n+ tips the top of the cell towards you.",
    "Screen y: up.\n+ swings the near face to the right.",
    "Screen z: into the screen.\n+ spins the picture clockwise.",
)
# Time constant of the swing the a/b/c buttons animate, in seconds.
STRUCTURE_ALIGN_TIME_CONSTANT = 0.09
# The a, b, c triad drawn in the corner of the 3D view. Its arms are renormalised to a
# fixed pixel length, so the widget reads the same at every zoom and cell size.
AXIS_WIDGET_COLORS = (
    (0.94, 0.38, 0.38, 1.0),
    (0.45, 0.85, 0.48, 1.0),
    (0.46, 0.62, 0.98, 1.0),
)
AXIS_WIDGET_ARM_PX = 26.0
# Where the letter sits, as a multiple of the arm length.
AXIS_WIDGET_LABEL_SCALE = 1.28
# Inset of the triad's origin from the plot's bottom-left corner. The draw list is clipped
# to the plot rect, so this has to clear an arm's whole reach -- arm length times the label
# scale, plus the letter itself -- or a letter pointing into the corner loses its head.
AXIS_WIDGET_MARGIN_PX = 46.0
# Moment that draws a sphere at its full element radius when sizing atoms by spin. Fixed
# rather than taken from the structure so a spin-5 Fe is the same size in every cell.
SPIN_RADIUS_REFERENCE_MOMENT = 5.0
# How many configurations the landscape keeps in reserve behind the plotted subset, so
# that re-enabling degenerate points restores them. Re-energizing this many costs ~11 ms
# for a 64-site cell, well under the exchange rebuild it rides along with.
SPIN_LANDSCAPE_POOL_LIMIT = 2000
# Fixed legend order. A and C each single out an axis, so their three orientations are
# separate states that split apart on a distorted cell; they are shaded within a family
# so the legend still reads as "the A's" and "the C's" at a glance.
SPIN_PLOT_CATEGORIES = [pattern.label for pattern in CANONICAL_PLANE_PATTERNS] + ["Other"]
# Past this, the nearest pattern is no better than chance (the concentration cannot
# exceed 0.5, since beyond halfway the flipped comparison wins) and the configuration
# is reported as "Other" rather than as a badly-matched ordering.
MAX_MATCH_DEFECT_CONCENTRATION = 0.25
# The two plots the 2D pane can show, in dropdown order.
TWO_D_PLOT_NAMES = [
    "Spin energies",
    "Exchange couplings",
    "Relaxation",
    "CHGNet energy",
]
TWO_D_PLOT_RELAXATION = 2
TWO_D_PLOT_LIVE_ENERGY = 3
# The live CHGNet single-point energy of the active structure. Asked for at most
# this often, and only once the previous answer is back: the server has one
# worker, and queueing a second request behind an unanswered one would only
# make every later point arrive later. Points beyond the cap scroll off.
LIVE_ENERGY_INTERVAL_SECONDS = 0.5
LIVE_ENERGY_MAX_POINTS = 50
LIVE_ENERGY_COLOR = (0.55, 0.95, 0.65, 1.0)
# A cell with hundreds of magnetic sites has thousands of coupled pairs, far more
# bars than there are pixels. Sorted by |J|, so the cap keeps the couplings that
# actually decide the ordering and drops a tail of near-zero ones.
EXCHANGE_PLOT_MAX_BARS = 200
# Half-width, in bar-position units, of a bar's click/hover target in the
# unfiltered view. Matches the 0.65 bar size used when plotting, so the hit area is
# the bar you can see.
EXCHANGE_BAR_HALF_WIDTH = 0.325
# With one atom selected, a pair is drawn as four bars -- its total and the three
# channels of ``bridge_J_components`` -- so the reader can see *which* mechanism
# made a coupling FM rather than only how strong it came out. The group spans
# +-0.4 about the pair's position: four bars of 0.19 at these offsets, with a hair
# of air between them and between neighbouring groups.
EXCHANGE_COMPONENT_BAR_SIZE = 0.19
EXCHANGE_COMPONENT_OFFSETS = (-0.3, -0.1, 0.1, 0.3)
EXCHANGE_GROUP_HALF_WIDTH = 0.4
# The three channels, in the offset order above after the total. Fixed colours,
# unlike the total's element blend: a channel is the same physics whatever the
# chemistry. Green/purple/blue sit clear of both EXCHANGE_FRUSTRATED_COLOR and the
# element blends the total bar can take.
EXCHANGE_SE_COLOR = (0.35, 0.78, 0.42, 1.0)
EXCHANGE_DE_COLOR = (0.62, 0.47, 0.80, 1.0)
EXCHANGE_OE_COLOR = (0.33, 0.60, 0.92, 1.0)
EXCHANGE_COMPONENT_SERIES = (
    ("Superexchange", EXCHANGE_SE_COLOR),
    ("Double exchange", EXCHANGE_DE_COLOR),
    ("Occupied-empty", EXCHANGE_OE_COLOR),
)
# Gap between a 2D plot's data area and the controls floated in its corners, so they
# read as sitting inside the plot rather than pinned to its frame.
TWO_D_OVERLAY_INSET = 6.0
# Extra fraction of the data span left empty above it, so the corner dropdowns
# float over blank plot instead of over the tallest bar or point. Sized against a
# combo's height in a pane at its usual share of the Structure View; a taller pane
# only gets more clearance, and the overlays are translucent where it falls short.
# The ring marking a frustrated bar sits on the bar's tip and needs room of its own
# above it, which is why this is not simply the combo's height.
TWO_D_TOP_HEADROOM = 0.52
# Default fraction of the span left below the data. Enough to keep the lowest point
# off the axis line without spending pane on emptiness.
TWO_D_BOTTOM_HEADROOM = 0.04
# The exchange bars all stand on y = 0, so the default margin puts every bar's foot
# on the x axis and the row of feet is unreadable -- a short bar in particular cannot
# be told from no bar. This drops the floor far enough that the baseline is visibly
# above the axis and each bar has a foot to see.
EXCHANGE_BOTTOM_HEADROOM = 0.12
# Ring on the tip of a bar the selected spin configuration disagrees with.
EXCHANGE_FRUSTRATED_COLOR = (1.0, 0.92, 0.16, 1.0)
# The M-L-M paths drawn from a selected atom. White so they read against every
# element colour, with the coupling's strength carried entirely by the alpha.
EXCHANGE_PATH_COLOR = (1.0, 1.0, 1.0, 1.0)
EXCHANGE_PATH_WEIGHT = 2.4
EXCHANGE_PATH_HOVER_WEIGHT = 4.5
# The weakest path still has to be visible, and the strongest must not wash out the
# atoms behind it.
EXCHANGE_PATH_MIN_ALPHA = 0.18
EXCHANGE_PATH_MAX_ALPHA = 1.0
# How near the cursor has to come to a path, in pixels, to hover it.
EXCHANGE_PATH_PICK_PIXELS = 7.0
# Decimal places of meV that two couplings must differ by before the bar order
# treats them as different rather than as a tie. See ``exchange_bar_sort_key``.
EXCHANGE_TIE_DECIMALS = 9
# Longest hand-entered sign string. Not a limit of the model -- PlanePattern takes
# any period -- but of what a cell can express: a pattern needs one plane per
# character, and a period the structure cannot resolve folds back onto a shorter
# one and is scored as an ordering it is not. Eight covers every pattern a
# perovskite supercell of a usable size can hold.
MAX_CUSTOM_PATTERN_PERIOD = 8
# Live re-energization pauses below this frame rate and resumes above the second,
# higher one. Two thresholds rather than one: pausing frees exactly the time that
# caused the drop, so a single threshold would rebuild the landscape every other
# frame. 10 fps is where a slider drag stops tracking the cursor badly enough to
# be worth giving up the live landscape for.
AUTO_SPIN_UPDATE_MIN_FPS = 10.0
AUTO_SPIN_UPDATE_RESUME_FPS = 20.0
# The grab band between the 3D and 2D plots, and the least either may be squeezed
# to. Below these a plot is not a smaller plot, it is an unreadable one.
PANE_SPLITTER_THICKNESS = 8.0
MIN_PLOT3D_HEIGHT = 160.0
MIN_TWO_D_HEIGHT = 120.0
# Width of the grip mark drawn at the centre of the splitter.
PANE_SPLITTER_GRIP_WIDTH = 40.0
# Stands in for the paused frame-rate readout while live updates are keeping up, so
# the row is the same width either way and the panel never reflows around it. Wide
# enough for a three-digit rate; spaces rather than an empty string because ImGui
# gives a zero-width item no space at all.
PAUSED_FPS_PLACEHOLDER = " " * len("000 fps")
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
    # The up-up-down-down family: E where it has a classical name, one shared colour
    # for the diagonal members, which are rarer and read as one group.
    "E(a)": (0.78, 0.42, 0.90, 1.0),
    "E(b)": (0.65, 0.30, 0.80, 1.0),
    "E(c)": (0.48, 0.18, 0.62, 1.0),
    "(011) ++--": (0.85, 0.55, 0.35, 1.0),
    "(101) ++--": (0.85, 0.55, 0.35, 1.0),
    "(110) ++--": (0.85, 0.55, 0.35, 1.0),
    "(111) ++--": (0.85, 0.55, 0.35, 1.0),
    "Other": (0.55, 0.55, 0.55, 1.0),
}
# Colours for hand-entered orderings, cycled by position in the user's list. Teal,
# pink, gold, brown -- hues the canonical set above leaves free, so a custom ordering
# is never mistaken for an A (blue), a C (green) or an E (purple) in the legend.
CUSTOM_SPIN_CLASS_COLORS = (
    (0.10, 0.75, 0.72, 1.0),
    (0.95, 0.40, 0.65, 1.0),
    (0.85, 0.78, 0.15, 1.0),
    (0.60, 0.40, 0.22, 1.0),
)
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


def plot_axis_directions(
    lattice: np.ndarray,
    use_cartesian: bool,
    plot_limits: Tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Unit directions of the lattice a, b, c vectors inside the plot's normalized box.

    In fractional mode the plot axes *are* a, b and c, so the raw directions are the
    identity; in Cartesian mode they are the lattice rows themselves. Either way each
    component is divided by its own axis range first, because ImPlot3D maps every axis
    onto the same box independently -- the box happens to be a cube today, so this is a
    no-op, but it is what keeps the directions honest if that ever changes.
    """
    raw = np.eye(3, dtype=np.float64)
    if use_cartesian:
        raw = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    limits = np.asarray(plot_limits, dtype=np.float64).reshape(3, 2)
    ranges = np.maximum(limits[:, 1] - limits[:, 0], 1e-12)
    scaled = raw / ranges[None, :]
    norms = np.linalg.norm(scaled, axis=1)
    directions = np.zeros((3, 3), dtype=np.float64)
    usable = norms > 1e-12
    directions[usable] = scaled[usable] / norms[usable, None]
    return directions


def _perpendicular_to(vector: np.ndarray) -> np.ndarray:
    """Some unit vector orthogonal to ``vector``, for degenerate lattices."""
    axis = int(np.argmin(np.abs(vector)))
    candidate = np.cross(vector, np.eye(3, dtype=np.float64)[axis])
    norm = float(np.linalg.norm(candidate))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0])
    return candidate / norm


def _rotation_matrix_to_quaternion(
    matrix: np.ndarray,
) -> Tuple[float, float, float, float]:
    """(x, y, z, w) for a rotation matrix, branching on the largest diagonal term."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quaternion = np.array([x, y, z, w], dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1e-12)
    return tuple(float(value) for value in quaternion)  # type: ignore[return-value]


def view_rotation_for_axis(
    axis_directions: np.ndarray,
    axis_index: int,
    sign: int = 1,
) -> Tuple[float, float, float, float]:
    """(x, y, z, w) rotation that points the camera straight down the given axis.

    ImPlot3D's rotation carries box coordinates into a view frame where +x is screen
    right, +y is screen up and +z points out of the screen at the viewer, so looking
    down an axis means rotating it onto +z. ``sign`` picks which end faces the viewer:
    +1 for the axis itself, -1 for the opposite face.

    The remaining freedom is the roll, fixed the crystallographic way: **c** points up
    the screen, except when c is the axis being looked down, where **b** takes over.
    Viewing down c therefore gives the usual ab-plane picture with a to the right, and
    viewing down b puts a to the left, which is what a right-handed cell forces once c
    is up.
    """
    directions = np.asarray(axis_directions, dtype=np.float64).reshape(3, 3)
    axis_index = axis_index % 3
    forward = directions[axis_index]
    if float(np.linalg.norm(forward)) < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    forward = forward / float(np.linalg.norm(forward))
    if sign < 0:
        forward = -forward

    up = directions[1 if axis_index == 2 else 2]
    up = up - float(up @ forward) * forward
    if float(np.linalg.norm(up)) < 1e-9:
        # The two axes are collinear (or the up one is degenerate); any roll will do.
        up = _perpendicular_to(forward)
    up = up / float(np.linalg.norm(up))
    right = np.cross(up, forward)

    # Rows, not columns: the matrix maps a box vector to its view-space components.
    return _rotation_matrix_to_quaternion(np.vstack((right, up, forward)))


def view_axis_alignment_sign(
    rotation: Tuple[float, float, float, float],
    axis_directions: np.ndarray,
    axis_index: int,
    *,
    tolerance: float = 0.995,
) -> int:
    """Which end of the axis to bring towards the viewer: +1, or -1 to look from behind.

    Pressing the same button twice turns the cell around, so both faces of a plane are
    one click apart. The answer is read off the pose rather than remembered, so having
    rotated away and come back, the button aligns the near way round instead of flipping
    to a side you were not looking at.
    """
    direction = np.asarray(axis_directions, dtype=np.float64).reshape(3, 3)[
        axis_index % 3
    ]
    if float(np.linalg.norm(direction)) < 1e-12:
        return 1
    turned = implot3d.Quat(*rotation) * implot3d.Point(*direction)
    # +z is out of the screen, so a z component at full length means this axis is already
    # pointing at the viewer and the click is asking for the opposite face.
    return -1 if turned.z / float(np.linalg.norm(direction)) > tolerance else 1


def rotation_after_drag(
    rotation: Tuple[float, float, float, float],
    dx: float,
    dy: float,
) -> Tuple[float, float, float, float]:
    """Trackball step: turn about the on-screen axis perpendicular to the drag.

    ImPlot3D's own rotation is a turntable whose pole is the plot's c axis, so it runs
    out of freedom exactly where the alignment buttons put you -- looking down c, a
    horizontal drag rolls the picture in place instead of orbiting it, and looking down
    b both drags turn about nearly the same screen axis. A trackball has no pole: the
    structure follows the cursor identically from every orientation.
    """
    length = math.hypot(dx, dy)
    if length < 1e-9:
        return rotation
    # Dragging right swings the near face right, about the screen's up axis; dragging
    # down tips it down, about the screen's right axis. Mouse dy already points down.
    step = implot3d.Quat(
        length * STRUCTURE_ROTATE_RADIANS_PER_PIXEL,
        implot3d.Point(dy / length, dx / length, 0.0),
    )
    # Left-multiplied: the rotation carries box coordinates into view space, so composing
    # on that side turns the structure in the screen's frame rather than in its own.
    turned = (step * implot3d.Quat(*rotation)).normalized()
    return (turned.x, turned.y, turned.z, turned.w)


def rotation_after_screen_turn(
    rotation: Tuple[float, float, float, float],
    axis: Sequence[float],
    degrees: float,
) -> Tuple[float, float, float, float]:
    """Turn the view about a direction fixed in screen space.

    The axis stays put on screen while the structure turns under it, which is what makes
    stepping by a fixed angle useful -- and, for the screen normal, is the one turn a
    trackball cannot make at all, since a drag only ever rotates about an axis lying in
    the screen plane.
    """
    step = implot3d.Quat(math.radians(degrees), implot3d.Point(*axis))
    turned = (step * implot3d.Quat(*rotation)).normalized()
    return (turned.x, turned.y, turned.z, turned.w)


def rotation_after_alignment_step(
    rotation: Tuple[float, float, float, float],
    target: Tuple[float, float, float, float],
    delta_time: float,
) -> Tuple[Tuple[float, float, float, float], bool]:
    """One eased step of the a/b/c swing. Returns the pose and whether it has arrived.

    Exponential rather than a fixed number of frames, so the swing takes the same wall
    time whether the app is redrawing at 60 fps or idling at 10.
    """
    current = implot3d.Quat(*rotation)
    goal = implot3d.Quat(*target)
    fraction = 1.0 - math.exp(-max(delta_time, 0.0) / STRUCTURE_ALIGN_TIME_CONSTANT)
    stepped = implot3d.Quat.slerp(current, goal, min(max(fraction, 0.0), 1.0))
    # Quaternions double-cover rotations, so q and -q are the same pose: compare with
    # the absolute dot product or the last few degrees never register as arrived.
    if abs(stepped.dot(goal)) > 1.0 - 1e-7:
        return target, True
    stepped = stepped.normalized()
    return (stepped.x, stepped.y, stepped.z, stepped.w), False


def clamp_min(value: float, v_min: float) -> float:
    return max(v_min, value)


def ensure_xyz_array(coords: np.ndarray) -> np.ndarray:
    coord_array = np.asarray(coords, dtype=np.float64)
    if coord_array.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return coord_array.reshape(-1, 3)


@lru_cache(maxsize=len(SPHERE_DETAIL_LEVELS) + 1)
def unit_sphere_template(
    latitude_segments: int = SPHERE_LATITUDE_SEGMENTS,
    longitude_segments: int = SPHERE_LONGITUDE_SEGMENTS,
) -> Tuple[np.ndarray, np.ndarray]:
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]

    for latitude_index in range(1, latitude_segments):
        polar_angle = np.pi * latitude_index / latitude_segments
        sin_polar = float(np.sin(polar_angle))
        cos_polar = float(np.cos(polar_angle))
        for longitude_index in range(longitude_segments):
            azimuth = 2.0 * np.pi * longitude_index / longitude_segments
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
        return 1 + (latitude_index - 1) * longitude_segments

    triangles: list[tuple[int, int, int]] = []

    first_ring_start = ring_start(1)
    for longitude_index in range(longitude_segments):
        next_longitude = (longitude_index + 1) % longitude_segments
        triangles.append(
            (
                0,
                first_ring_start + next_longitude,
                first_ring_start + longitude_index,
            )
        )

    for latitude_index in range(1, latitude_segments - 1):
        current_ring_start = ring_start(latitude_index)
        next_ring_start = ring_start(latitude_index + 1)
        for longitude_index in range(longitude_segments):
            next_longitude = (longitude_index + 1) % longitude_segments
            current = current_ring_start + longitude_index
            current_next = current_ring_start + next_longitude
            below = next_ring_start + longitude_index
            below_next = next_ring_start + next_longitude
            triangles.append((current, below, current_next))
            triangles.append((current_next, below, below_next))

    last_ring_start = ring_start(latitude_segments - 1)
    for longitude_index in range(longitude_segments):
        next_longitude = (longitude_index + 1) % longitude_segments
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


def visible_role_indices(
    structure: ChemicalStructure,
    *,
    show_a: bool,
    show_b: bool,
    show_x: bool,
) -> set[int]:
    """Atom indices to draw, given which site roles are switched on.

    Roles come from the builder provenance; a structure loaded from a file has none,
    so everything is drawn rather than guessing which atoms are the A sublattice.
    Interstitial protons follow the X sites they attach to.
    """
    params = getattr(structure, "generation_parameters", None)
    roles = None if params is None else getattr(params, "site_roles", None)
    if roles is None or len(roles) != structure.atom_count:
        return set(range(structure.atom_count))
    allowed = {"A": show_a, "B": show_b, "X": show_x, "H": show_x}
    return {
        index
        for index, role in enumerate(roles)
        if allowed.get(str(role), True)
    }


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


def element_render_radii(
    atomic_labels: Sequence[str],
    site_oxidation_states: np.ndarray | None,
    *,
    render_with_ionic_radius: bool,
) -> np.ndarray:
    """Sphere radius per label. Shared so a vacancy matches the atom it replaces."""
    fe3_reference = crystal_radius_for_rendering("Fe", 3).crystal_radius
    max_cation_radius = (
        fe3_reference if fe3_reference is not None else DEFAULT_ATOM_RENDER_RADIUS
    )
    ligand_radius = LIGAND_RADIUS_SCALE * max_cation_radius
    radii = np.empty(len(atomic_labels), dtype=np.float64)
    for atom_index, element in enumerate(atomic_labels):
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


# Radii are a pure function of the labels and the oxidation states, but the lookup
# runs per atom through the Shannon tables -- several milliseconds a frame on a
# thousand-atom cell, for an answer that only changes when the structure does.
ATOM_RADII_CACHE_LIMIT = 8
_ATOM_RADII_CACHE: "OrderedDict[Tuple[object, ...], np.ndarray]" = OrderedDict()


def structure_atom_render_radii(
    structure: ChemicalStructure,
    site_oxidation_states: np.ndarray | None,
    *,
    render_with_ionic_radius: bool,
) -> np.ndarray:
    key = (
        tuple(structure.atomic_labels),
        None
        if site_oxidation_states is None
        else _array_signature(np.asarray(site_oxidation_states)),
        bool(render_with_ionic_radius),
    )
    cached = _ATOM_RADII_CACHE.get(key)
    if cached is not None:
        _ATOM_RADII_CACHE.move_to_end(key)
        return cached
    radii = element_render_radii(
        structure.atomic_labels,
        site_oxidation_states,
        render_with_ionic_radius=render_with_ionic_radius,
    )
    radii.flags.writeable = False
    _ATOM_RADII_CACHE[key] = radii
    while len(_ATOM_RADII_CACHE) > ATOM_RADII_CACHE_LIMIT:
        _ATOM_RADII_CACHE.popitem(last=False)
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


def _array_signature(*arrays: np.ndarray) -> Tuple[object, ...]:
    """Hashable content key for numpy arrays.

    Deliberately content-addressed rather than a revision counter: nothing can
    mutate a structure in a way this misses. It is also almost free -- under a
    microsecond for a 320-atom cell, against the milliseconds each cached rebuild
    costs.
    """
    return tuple(
        (array.shape, array.dtype.str, array.tobytes())
        for array in (np.ascontiguousarray(item) for item in arrays)
    )


# The 3D view re-issues every draw call each frame, and materializing the ~36k
# implot3d.Point objects of a 320-atom cell is ~10 ms of that. Rotating the view does
# not move a single coordinate, so a mesh keyed on its inputs is reused until the
# structure actually changes. A handful of entries covers one mesh per element group
# plus the vacancy markers.
SPHERE_MESH_CACHE_LIMIT = 24
# Plane cross-sections, cached the same way and for the same reason: the geometry
# only moves when the cell or the ordering does, but it is re-issued every frame.
PLANE_POLYGON_CACHE_LIMIT = 64
_PLANE_POLYGON_CACHE: "OrderedDict[Tuple[object, ...], Tuple[np.ndarray, np.ndarray]]" = (
    OrderedDict()
)
# Live memo slots per AppState. The 3D view asks about two structures in a frame (the
# focus and the non-periodic rebuild it draws), so a slot per name alone would have the
# two evicting each other every frame; slots carry the structure's identity instead.
RENDER_CACHE_SLOT_LIMIT = 16
_SPHERE_MESH_CACHE: "OrderedDict[Tuple[object, ...], implot3d.Mesh]" = OrderedDict()


def build_sphere_mesh(
    centers: np.ndarray,
    radii: np.ndarray,
    lattice: np.ndarray,
    *,
    use_cartesian: bool,
    detail: Tuple[int, int] | None = None,
) -> implot3d.Mesh:
    """One mesh holding a sphere per center, cached on its inputs.

    The result is shared with the cache, so callers must treat it as read-only.
    They must also never read ``mesh.points`` back: that copies the whole vertex
    list out of C++ and costs more than rebuilding a small mesh would.
    """
    centers = ensure_xyz_array(np.asarray(centers, dtype=np.float64))
    radii = np.asarray(radii, dtype=np.float64)
    segments = detail or (SPHERE_LATITUDE_SEGMENTS, SPHERE_LONGITUDE_SEGMENTS)
    key = (_array_signature(centers, radii, lattice), bool(use_cartesian), segments)
    cached = _SPHERE_MESH_CACHE.get(key)
    if cached is not None:
        _SPHERE_MESH_CACHE.move_to_end(key)
        return cached

    unit_vertices, unit_triangles = unit_sphere_template(*segments)
    base_vertices = unit_vertices
    if not use_cartesian:
        base_vertices = unit_vertices @ np.linalg.inv(lattice)

    drawn = np.flatnonzero(radii > 0.0)
    if drawn.size == 0:
        mesh = implot3d.Mesh(points=[], idx=[])
    else:
        # Every sphere's vertices in one broadcast rather than a Python loop per
        # atom; what is left is the unavoidable Point construction.
        vertices = (
            centers[drawn][:, None, :]
            + radii[drawn][:, None, None] * base_vertices[None, :, :]
        ).reshape(-1, 3)
        offsets = np.arange(drawn.size, dtype=np.int64) * base_vertices.shape[0]
        idx = (
            unit_triangles.astype(np.int64)[None, :, :] + offsets[:, None, None]
        ).ravel()
        mesh = implot3d.Mesh(
            points=[implot3d.Point(x, y, z) for x, y, z in vertices.tolist()],
            idx=idx.tolist(),
        )

    _SPHERE_MESH_CACHE[key] = mesh
    while len(_SPHERE_MESH_CACHE) > SPHERE_MESH_CACHE_LIMIT:
        _SPHERE_MESH_CACHE.popitem(last=False)
    return mesh


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


def spin_moment_magnitudes(
    moments: np.ndarray | None,
    n_atoms: int,
) -> np.ndarray | None:
    """|m| per atom, zero-padded to ``n_atoms``.

    The moment array can be shorter than the atom list -- the same mismatch
    ``moments_as_vectors`` below absorbs -- and the atoms past its end are simply not
    magnetic.
    """
    if moments is None:
        return None
    array = np.asarray(moments, dtype=np.float64)
    if array.ndim == 1:
        site_magnitudes = np.abs(array)
    elif array.ndim == 2 and array.shape[1] == 3:
        site_magnitudes = np.linalg.norm(array, axis=1)
    else:
        return None
    magnitudes = np.zeros(n_atoms, dtype=np.float64)
    site_count = min(n_atoms, site_magnitudes.shape[0])
    magnitudes[:site_count] = site_magnitudes[:site_count]
    return magnitudes


def spin_scaled_render_radii(
    radii: np.ndarray,
    magnitudes: np.ndarray | None,
    *,
    reference: float = SPIN_RADIUS_REFERENCE_MOMENT,
) -> np.ndarray:
    """Sphere radii scaled by moment magnitude, against a fixed reference moment.

    A spin-5 ion draws at its full element radius and a spin-1 ion at a fifth of it, so
    the ratio between two sites is the ratio of their moments. Sites with no moment come
    out at radius zero, which ``build_sphere_mesh`` skips -- non-magnetic atoms drop out
    of the view rather than dwarfing the magnetic sublattice. Moments above the reference
    (Gd3+ at 7, say) are left to exceed the element radius.
    """
    base = np.asarray(radii, dtype=np.float64)
    if magnitudes is None:
        return base
    scale = np.asarray(magnitudes, dtype=np.float64)
    if scale.shape != base.shape or not np.any(scale > 0.0):
        # Nothing is magnetic (or the arrays disagree): leave the view as it was rather
        # than blanking it.
        return base
    return base * (scale / max(float(reference), 1e-9))


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


def _energy_groups(
    configs: List[SpinConfig], tol: float = DEGENERACY_ENERGY_TOL
) -> List[List[int]]:
    """Indices of ``configs`` grouped by equal energy. Assumes energy-sorted input."""
    groups: List[List[int]] = []
    for index, config in enumerate(configs):
        if groups and abs(config.energy - configs[groups[-1][0]].energy) <= tol:
            groups[-1].append(index)
        else:
            groups.append([index])
    return groups


def annotate_degeneracy(configs: List[SpinConfig]) -> List[SpinConfig]:
    """Record how many distinct configurations share each energy.

    The exchange model is highly symmetric, so a single energy routinely covers many
    configurations; carrying the count lets the UI say so instead of implying that a
    collapsed list is the whole landscape.
    """
    annotated = list(configs)
    for group in _energy_groups(annotated):
        for index in group:
            annotated[index] = replace(annotated[index], degeneracy=len(group))
    return annotated


def collapse_degenerate_configs(
    configs: List[SpinConfig],
    reference_keys: set[Tuple[float, ...]],
) -> List[SpinConfig]:
    """One representative per distinct energy, keeping every reference ordering.

    References are exempt because several of them are degenerate by construction on an
    undistorted cell -- collapsing those would hide exactly the splitting the plot
    exists to show.
    """
    kept: List[SpinConfig] = []
    for group in _energy_groups(configs):
        members = [configs[index] for index in group]
        references = [
            config
            for config in members
            if canonical_moment_key(config.all_moments) in reference_keys
        ]
        kept.extend(references or members[:1])
    return kept


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
    *,
    keep_cells: set | None = None,
    drop_cells: set | None = None,
) -> np.ndarray:
    """Cage triangles for a build, already shifted into the structure's frame.

    ``keep_cells`` restricts the draw to those grid cells and ``drop_cells``
    excludes them, which is how the cages are split into the ones a focused plane
    cuts and the ones it does not -- the two halves are drawn at different alpha.
    Both are on top of the cages a vacancy has already removed the centre from.
    """
    params = getattr(structure, "generation_parameters", None)
    skip_cells = (
        vacated_b_cells(
            build.octahedra.shape,
            params.periodic_axes,
            list(getattr(params, "defects", [])),
        )
        if params is not None
        else set()
    )
    skip_cells = set(skip_cells)
    if drop_cells:
        skip_cells |= set(drop_cells)
    if keep_cells is not None:
        every = set(
            tuple(int(value) for value in index)
            for index in np.ndindex(build.octahedra.shape)
        )
        skip_cells |= every - set(keep_cells)
    triangles = octahedron_triangle_vertices(build.octahedra, skip_cells=skip_cells)
    if triangles.size == 0 or params is None:
        return triangles
    return triangles - np.asarray(params.cell_origin, dtype=np.float64)


def vacancy_render_sites(
    structure: ChemicalStructure,
) -> tuple[np.ndarray, List[str]]:
    """Ideal positions of the vacated sites, and the element each is missing.

    A vacancy has no atom to draw, so its position has to come back from the ideal
    build the defects were subtracted from. Labelling each hole with the species
    that *would* occupy it lets the marker be drawn at that species' radius, so an
    oxygen vacancy is exactly the size of the oxygens around it.

    Resolved the same way the structure itself was built -- including boundary
    images on the non-periodic render -- so every copy of a vacated site is
    marked, not just the one inside the home cell.
    """
    params = getattr(structure, "generation_parameters", None)
    if params is None or not getattr(params, "defects", None):
        return np.zeros((0, 3), dtype=np.float64), []
    try:
        build = build_from_generation_parameters(params)
        resolution = resolve_defects(
            build,
            periodic=params.periodic_axes,
            stored_periodic=params.defect_reference_periodic(),
            defects=list(params.defects),
        )
    except ValueError:
        return np.zeros((0, 3), dtype=np.float64), []
    vacated = np.flatnonzero(resolution.canonical_to_structure < 0)
    if not len(vacated):
        return np.zeros((0, 3), dtype=np.float64), []
    ideal_labels = formula_atomic_labels_from_parameters(build, params)
    coords = np.asarray(build.all_sites, dtype=np.float64)[vacated] - np.asarray(
        params.cell_origin, dtype=np.float64
    )
    return coords, [ideal_labels[index] for index in vacated]


def vacancy_render_radii(
    vacancy_labels: Sequence[str],
    structure: ChemicalStructure,
    atom_radii: np.ndarray,
    *,
    render_with_ionic_radius: bool,
) -> np.ndarray:
    """Marker radius per vacancy: whatever its element is *actually* drawn at.

    Copied from a surviving atom of the same element rather than recomputed, so
    the hole matches its neighbours even when the atoms are being drawn at ionic
    radii from a solved oxidation state (which a vacancy has no way to look up).
    Falls back to the neutral radius when the element has been vacated entirely.
    """
    if not len(vacancy_labels):
        return np.zeros(0, dtype=np.float64)
    drawn: Dict[str, float] = {}
    for atom_index, element in enumerate(structure.atomic_labels):
        drawn.setdefault(element, float(atom_radii[atom_index]))
    fallback = element_render_radii(
        vacancy_labels, None, render_with_ionic_radius=render_with_ionic_radius
    )
    return np.array(
        [
            drawn.get(element, float(fallback[position]))
            for position, element in enumerate(vacancy_labels)
        ],
        dtype=np.float64,
    )


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


def current_framerate() -> float:
    """ImGui's smoothed frame rate, or 0.0 where there is no live context.

    The frame-rate gate on live re-energization is a property of the running app,
    and the model-layer code it guards is also driven straight from the CLI and the
    tests, where there is no ImGui at all. Zero reads as "no measurement", which
    the gate treats as "carry on" rather than as "too slow".
    """
    if imgui.get_current_context() is None:
        return 0.0
    return float(imgui.get_io().framerate)


def oxidation_site_rows(
    structure: ChemicalStructure,
    assignment: OxidationStateAssignment,
    *,
    site_moments: np.ndarray | None = None,
) -> List[str]:
    """One label per atom: element, oxidation state, and moment vector.

    Every site is listed. Kept as the batch form of ``oxidation_site_row`` for the
    CLI and the tests; the UI reads one site at a time, off the atom under the
    cursor.
    """
    moment_vectors = (
        moments_as_vectors(site_moments, structure.atom_count)
        if site_moments is not None
        else np.zeros((structure.atom_count, 3), dtype=np.float64)
    )
    site_count = min(structure.atom_count, len(assignment.site_oxidation_states))
    return [
        oxidation_site_row(structure, assignment, index, moment_vectors[index])
        for index in range(site_count)
    ]


def oxidation_site_row(
    structure: ChemicalStructure,
    assignment: OxidationStateAssignment,
    site_index: int,
    moment: Sequence[float],
) -> str:
    """One atom as ``  1. Fe  ox=+3  m=(+0.00, +0.00, +5.00)``."""
    return (
        f"{site_index + 1:>3}. {structure.atomic_labels[site_index]:<2}  "
        f"ox={int(assignment.site_oxidation_states[site_index]):+d}  "
        f"m=({moment[0]:+.2f}, {moment[1]:+.2f}, {moment[2]:+.2f})"
    )


def site_hover_tooltip(
    state: "AppState",
    structure: ChemicalStructure,
    site_index: int,
) -> str:
    """What an atom says when the cursor is on it in the default view.

    The oxidation state and moment that used to be a per-atom row in the results
    panel. Read off the atom itself rather than off a list, so there is no matching
    a row against the picture by eye.
    """
    if not 0 <= site_index < structure.atom_count:
        return ""
    assignment = state.selected_oxidation_assignment()
    moments = state.selected_spin_moments_for_structure(structure)
    vectors = (
        moments_as_vectors(moments, structure.atom_count)
        if moments is not None
        else np.zeros((structure.atom_count, 3), dtype=np.float64)
    )
    if assignment is None or site_index >= len(assignment.site_oxidation_states):
        # No assignment to read a charge from; the element and moment still say
        # something, and saying nothing at all would look like a broken tooltip.
        moment = vectors[site_index]
        return (
            f"{site_index + 1:>3}. {structure.atomic_labels[site_index]}  "
            f"m=({moment[0]:+.2f}, {moment[1]:+.2f}, {moment[2]:+.2f})"
        )
    return oxidation_site_row(structure, assignment, site_index, vectors[site_index])


def highlighted_render_indices(
    rendered: ChemicalStructure,
    source: ChemicalStructure,
    site_index: int,
    *,
    tolerance: float = 1e-6,
) -> List[int]:
    """Atoms of ``rendered`` that are the site ``site_index`` of ``source``.

    The 3D view may be drawing a non-periodic rebuild of the focused structure,
    which has a different atom count and order, so the selected site is matched by
    fractional position instead of by index. Matching modulo the cell means every
    periodic image of the chosen atom lights up, which is what you want when the
    boundary layer is being drawn.
    """
    if not 0 <= site_index < source.atom_count:
        return []
    if rendered is source:
        return [site_index]
    try:
        target = source.fractional_coords[site_index]
        rendered_fractional = rendered.fractional_coords
    except np.linalg.LinAlgError:
        return []
    delta = (rendered_fractional - target + 0.5) % 1.0 - 0.5
    return [int(index) for index in np.flatnonzero(np.all(np.abs(delta) < tolerance, axis=1))]


def source_site_for_render_index(
    rendered: ChemicalStructure,
    source: ChemicalStructure,
    render_index: int,
    *,
    tolerance: float = 1e-6,
) -> int:
    """The site of ``source`` that the rendered atom ``render_index`` is, or -1.

    The inverse of ``highlighted_render_indices``, and matched the same way: the
    3D view may be drawing a non-periodic rebuild with a different atom count and
    order, so position modulo the cell is the only thing the two share. Every
    periodic image of a site maps back to the one site it is an image of.
    """
    if not 0 <= render_index < rendered.atom_count:
        return -1
    if rendered is source:
        return render_index
    try:
        target = rendered.fractional_coords[render_index]
        source_fractional = source.fractional_coords
    except np.linalg.LinAlgError:
        return -1
    delta = (source_fractional - target + 0.5) % 1.0 - 0.5
    matches = np.flatnonzero(np.all(np.abs(delta) < tolerance, axis=1))
    return int(matches[0]) if len(matches) else -1


def magnetic_pick_candidates(
    state: "AppState",
    rendered: ChemicalStructure,
) -> List[int]:
    """Rendered atoms with couplings to show, as click targets in the 3D view.

    Magnetic sites, less any that are coupled to nothing -- an isolated moment, or
    one whose neighbours are all non-magnetic. Selecting one of those would replace
    the plot with an empty pane, which is not an answer to the question the click
    asked.

    Memoized: this is a fractional-coordinate match per magnetic site, and the 3D
    view rebuilds it every frame otherwise. The key covers everything that can
    change which atoms qualify -- a different render, a different analysed
    structure, a new landscape (which is what a change of magnetic sites rides in
    on), or rebuilt couplings.
    """
    analysis = state.magnetic_analysis_structure
    if analysis is None or not state.magnetic_site_indices:
        return []

    def build() -> List[int]:
        coupled = set()
        for pair in state.magnetic_pair_couplings:
            coupled.add(pair.site_i)
            coupled.add(pair.site_j)
        indices: List[int] = []
        for site in state.magnetic_site_indices:
            if site in coupled:
                indices.extend(highlighted_render_indices(rendered, analysis, site))
        return indices

    return state._cached(
        "magnetic_pick_candidates",
        (
            id(rendered),
            id(analysis),
            state._landscape_generation,
            state._exchange_generation,
        ),
        build,
    )


def cartesian_to_display(
    points: np.ndarray, lattice: np.ndarray, use_cartesian: bool
) -> np.ndarray:
    """Cartesian points in the frame the 3D view is currently drawing in."""
    points = np.asarray(points, dtype=np.float64)
    if use_cartesian:
        return points
    flat = points.reshape(-1, 3)
    return np.linalg.solve(lattice.T, flat.T).T.reshape(points.shape)


def exchange_selection_site(state: "AppState") -> int:
    """The atom whose couplings the 3D view should be decorating, or -1.

    Gated on the coupling plot being the one on screen: the fading, the paths and
    the click that sets all this off are answers to a question only that plot asks,
    and leaving them on behind the energy landscape would be decoration with
    nothing to read it against.
    """
    if state.two_d_plot_index != 1:
        return -1
    site = state.selected_site_index
    if site < 0 or state.magnetic_analysis_structure is None:
        return -1
    return site if site in state.magnetic_site_indices else -1


def exchange_prominent_render_atoms(
    display_coords: np.ndarray,
    paths: Sequence[Tuple[PairCoupling, np.ndarray]],
    *,
    tolerance: float = 1e-6,
) -> set[int]:
    """Rendered atoms that a drawn exchange path passes through.

    Read off the paths rather than off the couplings, so that a bright atom always
    has a path running through it. Going the other way -- taking the coupled sites
    and lighting up every periodic image of each -- lights up images on the far side
    of the cell that no drawn path reaches, which reads as a coupling that is not
    there.

    Everything else fades back, so the network stands out of the cell instead of
    being buried in it.
    """
    if not len(paths):
        return set()
    points = np.concatenate([path for _pair, path in paths], axis=0)
    coords = np.asarray(display_coords, dtype=np.float64)
    # (n_points, n_atoms): small enough to do flat, since the paths of one atom are
    # a few dozen points at most.
    close = np.all(
        np.abs(coords[None, :, :] - points[:, None, :]) < tolerance, axis=2
    )
    return {int(index) for index in np.flatnonzero(close.any(axis=0))}


def exchange_unreached_partner_atoms(
    state: "AppState",
    rendered: ChemicalStructure,
    site: int,
    reached: set[int],
) -> set[int]:
    """Images of coupled partners that no drawn path endpoint landed on.

    A coupling to a neighbour across the cell boundary runs out of the drawn
    box, and the neighbour itself is drawn wrapped to the opposite face -- so
    the path's endpoint has no atom on it and the neighbour faded away with the
    rest of the cell, even though its bar sits right there in the chart.

    Only partners that nothing else lit are added, so the ordinary case -- a
    path that ends on the atom it points at -- keeps the tighter reading of
    ``exchange_prominent_render_atoms``, where a bright atom has a path through
    it. When a partner is unreachable that way, every image of it lights, the
    same convention the hover and selection rings already use.
    """
    analysis = state.magnetic_analysis_structure
    if site < 0 or analysis is None:
        return set()
    extra: set[int] = set()
    for pair in exchange_pairs_for_site(state.magnetic_pair_couplings, site):
        partner = pair.site_j if pair.site_i == site else pair.site_i
        images = set(highlighted_render_indices(rendered, analysis, partner))
        if images and not (images & reached):
            extra |= images
    return extra


def exchange_pick_candidates(
    display_coords: np.ndarray,
    paths: Sequence[Tuple[PairCoupling, np.ndarray]],
    *,
    tolerance: float = 1e-6,
) -> List[int]:
    """Rendered atoms that are an end of some drawn path: the metals, not the ligands.

    While one atom's couplings are on show, these are the only atoms worth clicking
    -- the selected atom itself, to clear the selection, and the atoms it actually
    couples to, to walk to. Clicking anything else would jump to an atom whose
    couplings share nothing with what is on screen.
    """
    if not len(paths):
        return []
    # The ends only. The bridging ligand in the middle carries no couplings of its
    # own and selecting it would empty the plot.
    ends = np.concatenate([path[[0, -1]] for _pair, path in paths], axis=0)
    coords = np.asarray(display_coords, dtype=np.float64)
    close = np.all(np.abs(coords[None, :, :] - ends[:, None, :]) < tolerance, axis=2)
    return [int(index) for index in np.flatnonzero(close.any(axis=0))]


def view_projection_key(
    plot_limits: Sequence[float],
    rotation: Sequence[float],
    zoom: float,
    rect_min: Any,
    rect_max: Any,
) -> Tuple[float, ...]:
    """Everything that moves a point on screen, as a cache key.

    Projecting every atom costs a pybind call each, ~10 ms at 320 atoms, and the
    hover test needs all of them. It only has to be redone when the view moves,
    though -- dragging the cursor across a still structure reprojects nothing.
    """
    return (
        tuple(float(value) for value in plot_limits)
        + tuple(float(value) for value in rotation)
        + (float(zoom), rect_min.x, rect_min.y, rect_max.x, rect_max.y)
    )


def exchange_render_paths(
    state: "AppState",
    rendered: ChemicalStructure,
    display_coords: np.ndarray,
    site: int,
    *,
    use_cartesian: bool,
) -> List[Tuple[PairCoupling, np.ndarray]]:
    """One (coupling, 3-point path) per bridge, per drawn image of either end.

    The paths come off the bridges in the analysed structure's frame. The view may
    be drawing several periodic images of the selected atom, and the selection ring
    already marks all of them, so each path is repeated at each image -- translated
    by that image's offset from the base site. Anchoring them all at one image
    instead would say the other images were different atoms.

    The same path is also anchored at each drawn image of the *partner*. A
    coupling across the cell boundary otherwise only shows the half that leaves
    the selected atom and runs out of the box, with nothing arriving at the
    neighbour it actually reaches -- which is drawn wrapped to the opposite
    face. Anchoring at the partner draws that arriving half. Where both
    anchorings land on the same polyline -- the ordinary coupling that stays
    inside the cell -- the duplicate is dropped.
    """
    analysis = state.magnetic_analysis_structure
    if site < 0 or analysis is None:
        return []
    lattice = rendered.lattice
    images = [
        np.asarray(display_coords[image], dtype=np.float64)
        for image in highlighted_render_indices(rendered, analysis, site)
    ]
    if not images:
        return []

    paths: List[Tuple[PairCoupling, np.ndarray]] = []
    seen: set[bytes] = set()

    def add(pair: PairCoupling, path: np.ndarray) -> None:
        key = np.ascontiguousarray(np.round(path, 6)).tobytes()
        if key in seen:
            return
        seen.add(key)
        paths.append((pair, path))

    for pair in exchange_pairs_for_site(state.magnetic_pair_couplings, site):
        if pair.paths is None or not len(pair.paths):
            continue
        # Drawn outwards from the selected atom, so a path always starts where the
        # eye already is; the stored order runs site_i -> ligand -> site_j.
        oriented = pair.paths if pair.site_i == site else pair.paths[:, ::-1, :]
        display = cartesian_to_display(oriented, lattice, use_cartesian)
        # Shifted so the path *starts* on the drawn atom, rather than by that
        # image's offset from the base site: a bridge that itself crosses the
        # boundary stores its near end at an image, and assuming otherwise left
        # the line floating clear of the atom it was supposed to leave.
        for anchor in images:
            for path in display:
                add(pair, path + (anchor - path[0]))
        # ...and arriving at every drawn image of the other end. The shift comes
        # from the path's own endpoint rather than the partner's base position:
        # a bridge that crosses the boundary already carries that offset.
        partner = pair.site_j if pair.site_i == site else pair.site_i
        for image in highlighted_render_indices(rendered, analysis, partner):
            target = np.asarray(display_coords[image], dtype=np.float64)
            for path in display:
                add(pair, path + (target - path[-1]))
    return paths


def exchange_path_alpha(j_eff: float, strongest: float) -> float:
    """Path opacity from coupling strength, relative to the strongest one shown.

    Linear in |J| rather than in its rank: the question a path answers is "how much
    does this one matter compared to that one", and ranks flatten exactly the
    difference worth seeing.
    """
    if strongest <= 1e-15:
        return EXCHANGE_PATH_MAX_ALPHA
    fraction = min(abs(j_eff) / strongest, 1.0)
    return EXCHANGE_PATH_MIN_ALPHA + fraction * (
        EXCHANGE_PATH_MAX_ALPHA - EXCHANGE_PATH_MIN_ALPHA
    )


def point_to_segment_distance(
    point: Tuple[float, float],
    start: Sequence[float],
    end: Sequence[float],
) -> float:
    """Distance from a point to a line segment, in whatever units it is given."""
    px, py = point
    ax, ay = float(start[0]), float(start[1])
    bx, by = float(end[0]), float(end[1])
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = min(max(((px - ax) * dx + (py - ay) * dy) / length_sq, 0.0), 1.0)
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def nearest_exchange_path(
    pixel_paths: Sequence[np.ndarray],
    mouse: Tuple[float, float],
    *,
    radius_pixels: float = EXCHANGE_PATH_PICK_PIXELS,
) -> int:
    """Index of the path nearest the cursor, or -1 when none is near enough.

    Distance to the *segments*, not to the three points: a path is mostly the two
    lines between them, and hovering a bond's middle is the natural way to ask
    about it. Takes projected pixels for the same reason ``nearest_picked_atom``
    does -- the projection needs a live plot, the choice does not.
    """
    best_index, best_distance = -1, radius_pixels
    for index, pixels in enumerate(pixel_paths):
        distance = min(
            point_to_segment_distance(mouse, pixels[step], pixels[step + 1])
            for step in range(len(pixels) - 1)
        )
        if distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def view_space_depth(
    coords: np.ndarray,
    plot_limits: Sequence[float],
    rotation: Tuple[float, float, float, float],
) -> np.ndarray:
    """How near the viewer each point is, in the view frame. Larger is nearer.

    The plot box rotation carries box coordinates into view space, so normalizing
    into the box and applying the same quaternion puts the camera's own axes back
    on the data. Only the ordering matters here -- this breaks the tie when two
    atoms project onto nearly the same pixel and one is in front of the other.
    """
    points = ensure_xyz_array(np.asarray(coords, dtype=np.float64))
    if points.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    limits = np.asarray(plot_limits, dtype=np.float64).reshape(3, 2)
    center = limits.mean(axis=1)
    half = np.maximum((limits[:, 1] - limits[:, 0]) * 0.5, 1e-12)
    normalized = (points - center) / half
    quaternion = implot3d.Quat(*rotation)
    return np.array(
        [
            (quaternion * implot3d.Point(float(x), float(y), float(z))).z
            for x, y, z in normalized
        ],
        dtype=np.float64,
    )


def candidate_pixels(display_coords: np.ndarray, candidates: Sequence[int]) -> np.ndarray:
    """Screen position of each candidate. Only valid inside a begin_plot block."""
    pixels = np.empty((len(candidates), 2), dtype=np.float64)
    for position, atom in enumerate(candidates):
        pixel = implot3d.plot_to_pixels(
            float(display_coords[atom][0]),
            float(display_coords[atom][1]),
            float(display_coords[atom][2]),
        )
        pixels[position] = (pixel.x, pixel.y)
    return pixels


def nearest_picked_atom(
    pixels: np.ndarray,
    candidates: Sequence[int],
    depths: np.ndarray,
    mouse: Tuple[float, float],
    *,
    radius_pixels: float = PICK_RADIUS_PIXELS,
) -> int:
    """The candidate atom under the cursor, or -1 when the cursor is over none.

    Nearest in screen space, with depth as the tie-break so a click lands on the
    atom in front rather than one hidden behind it -- viewed edge-on, a plane's
    own atoms line up almost exactly, and screen distance alone would pick
    arbitrarily between them.

    The tie is deliberately coarse rather than exact. Depth only decides between
    atoms within ``PICK_TIE_PIXELS`` of each other, so an atom directly under the
    cursor still wins against a nearer one that is merely close by -- otherwise
    the frontmost atom in the plane would swallow clicks aimed past it.

    Takes projected pixels rather than projecting itself, which keeps it a pure
    function: the projection needs a live plot, and the choice does not.
    """
    best_index, best_key = -1, None
    for position, atom in enumerate(candidates):
        distance = math.hypot(
            float(pixels[position][0]) - mouse[0],
            float(pixels[position][1]) - mouse[1],
        )
        if distance > radius_pixels:
            continue
        key = (round(distance / PICK_TIE_PIXELS), -float(depths[position]))
        if best_key is None or key < best_key:
            best_index, best_key = atom, key
    return best_index


def draw_site_highlight_rings(
    display_coords: np.ndarray,
    axis_extents: np.ndarray,
    indices: Sequence[int],
    *,
    scale: float = 1.9,
    color: tuple[float, float, float, float] = SITE_HIGHLIGHT_COLOR,
) -> None:
    """Ring the given atoms in screen space.

    Drawn onto the plot's draw list after projecting each centre through
    ``plot_to_pixels``, so the ring always faces the viewer however the structure
    is rotated -- a 3D circle would foreshorten to a line edge-on. The radius
    comes from projecting the atom's own extent, so it tracks zoom.
    """
    if not len(indices):
        return
    draw_list = implot3d.get_plot_draw_list()
    packed = imgui.IM_COL32(
        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255
    )
    for index in indices:
        if not 0 <= index < len(display_coords):
            continue
        centre = np.asarray(display_coords[index], dtype=np.float64)
        centre_px = implot3d.plot_to_pixels(centre[0], centre[1], centre[2])
        extent = np.asarray(axis_extents[index], dtype=np.float64)
        radius_px = 0.0
        for axis in range(3):
            offset = centre.copy()
            offset[axis] += extent[axis]
            edge_px = implot3d.plot_to_pixels(offset[0], offset[1], offset[2])
            radius_px = max(
                radius_px, float(np.hypot(edge_px.x - centre_px.x, edge_px.y - centre_px.y))
            )
        draw_list.add_circle(centre_px, max(radius_px * scale, 6.0), packed, 48, 2.5)


def draw_axis_orientation_widget(
    axis_directions: np.ndarray,
    plot_limits: Tuple[float, float, float, float, float, float],
) -> None:
    """An a/b/c triad in the corner of the plot, showing how the cell is oriented.

    Screen space, like ``draw_site_highlight_rings``: each arm is the pixel direction of
    a step along that lattice vector, taken through ``plot_to_pixels`` so it follows the
    rotation without this code needing the camera quaternion. ImPlot3D is orthographic,
    so that pixel delta is linear in the plot delta and its direction does not depend on
    the step size -- renormalising to a fixed pixel length is therefore free, and is what
    keeps the widget the same size at every zoom. An axis pointing straight at the viewer
    has no direction left to draw, and becomes a dot.
    """
    limits = np.asarray(plot_limits, dtype=np.float64).reshape(3, 2)
    centre = limits.mean(axis=1)
    ranges = np.maximum(limits[:, 1] - limits[:, 0], 1e-12)
    centre_px = implot3d.plot_to_pixels(centre[0], centre[1], centre[2])

    rect_pos, rect_size = implot3d.get_plot_rect_pos(), implot3d.get_plot_rect_size()
    origin = imgui.ImVec2(
        rect_pos.x + AXIS_WIDGET_MARGIN_PX,
        rect_pos.y + rect_size.y - AXIS_WIDGET_MARGIN_PX,
    )
    draw_list = implot3d.get_plot_draw_list()
    for axis, label in enumerate("abc"):
        color = AXIS_WIDGET_COLORS[axis]
        packed = imgui.IM_COL32(
            int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255
        )
        step = centre + np.asarray(axis_directions[axis], dtype=np.float64) * (
            0.25 * ranges
        )
        tip_px = implot3d.plot_to_pixels(step[0], step[1], step[2])
        delta_x, delta_y = tip_px.x - centre_px.x, tip_px.y - centre_px.y
        length = float(np.hypot(delta_x, delta_y))
        if length < 1e-3:
            draw_list.add_circle_filled(origin, 3.5, packed, 12)
            continue
        arm = imgui.ImVec2(
            origin.x + delta_x / length * AXIS_WIDGET_ARM_PX,
            origin.y + delta_y / length * AXIS_WIDGET_ARM_PX,
        )
        draw_list.add_line(origin, arm, packed, 2.0)
        text_size = imgui.calc_text_size(label)
        draw_list.add_text(
            imgui.ImVec2(
                origin.x
                + delta_x / length * AXIS_WIDGET_ARM_PX * AXIS_WIDGET_LABEL_SCALE
                - text_size.x * 0.5,
                origin.y
                + delta_y / length * AXIS_WIDGET_ARM_PX * AXIS_WIDGET_LABEL_SCALE
                - text_size.y * 0.5,
            ),
            packed,
            label,
        )


def render_to_focus_indices(
    rendered: ChemicalStructure,
    source: ChemicalStructure,
    *,
    tolerance: float = 1e-6,
) -> np.ndarray:
    """For every rendered atom, the site of ``source`` it is an image of, or -1.

    The whole-structure form of ``source_site_for_render_index``, vectorised:
    the selection can hold hundreds of atoms and asking about them one at a time
    would cost a pass over the structure each. Position modulo the cell is the
    only thing the two structures share, and it is what is matched.
    """
    if rendered is source:
        return np.arange(rendered.atom_count, dtype=np.int64)
    result = np.full(rendered.atom_count, -1, dtype=np.int64)
    if rendered.atom_count == 0 or source.atom_count == 0:
        return result
    try:
        rendered_fractional = rendered.fractional_coords
        source_fractional = source.fractional_coords
    except np.linalg.LinAlgError:
        return result
    delta = (
        rendered_fractional[:, None, :] - source_fractional[None, :, :] + 0.5
    ) % 1.0 - 0.5
    close = np.all(np.abs(delta) < tolerance, axis=2)
    hit = close.any(axis=1)
    result[hit] = np.argmax(close[hit], axis=1)
    return result


def to_display_frame(
    cartesian: np.ndarray, lattice: np.ndarray, use_cartesian: bool
) -> np.ndarray:
    """Cartesian points in whatever frame the 3D view is drawing in."""
    points = np.asarray(cartesian, dtype=np.float64).reshape(-1, 3)
    if use_cartesian or points.shape[0] == 0:
        return points
    return np.linalg.solve(np.asarray(lattice, dtype=np.float64).T, points.T).T


def plot_selection_slab(
    lattice: np.ndarray,
    slab: SelectionSlab,
    *,
    use_cartesian: bool,
    hovered: bool,
) -> List[np.ndarray]:
    """Draw the slab as two faces joined at their corners; returns the faces.

    The faces are returned in the display frame so the caller can project them
    for the hover test. Faces are fans without their internal edges, the same
    way the Miller sheets are drawn, with a closed border each and the four
    joining edges so the slab reads as a box with a thickness rather than as
    two unrelated sheets.
    """
    faces = slab_face_corners(lattice, slab)
    if faces is None:
        return []
    color = SLAB_FACE_HOVER_COLOR if hovered else SLAB_FACE_COLOR
    display = [to_display_frame(face, lattice, use_cartesian) for face in faces]
    for position, quad in enumerate(display):
        triangles = np.vstack((quad[[0, 1, 2]], quad[[0, 2, 3]]))
        implot3d.plot_triangle(
            "Selection slab" if position == 0 else "##selection_slab_face_1",
            np.ascontiguousarray(triangles[:, 0]),
            np.ascontiguousarray(triangles[:, 1]),
            np.ascontiguousarray(triangles[:, 2]),
            spec=implot3d.Spec(
                fill_color=color,
                line_color=color,
                marker=implot3d.Marker_.none,
                fill_alpha=SLAB_FACE_ALPHA,
                flags=implot3d.TriangleFlags_.no_lines.value,
            ),
        )
    edges = [np.vstack((quad, quad[:1])) for quad in display]
    edges += [
        np.vstack((display[0][corner], display[1][corner])) for corner in range(4)
    ]
    line_coords = _segments_to_line_coords(edges)
    if line_coords is not None:
        xs, ys, zs = line_coords
        implot3d.plot_line(
            "##selection_slab_edges",
            xs,
            ys,
            zs,
            spec=implot3d.Spec(
                line_color=color, marker=implot3d.Marker_.none, line_weight=1.5
            ),
        )
    return display


def slab_arrow_endpoints(
    lattice: np.ndarray, slab: SelectionSlab
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Cartesian tail and tip of the arrow along the slab's normal.

    The tail sits at the slab's centre; the tip a third of the cell's extent
    along the normal beyond it, so the arrow is legible at any thickness and
    never leaves the neighbourhood of the slab it belongs to.
    """
    normal = slab_normal(lattice, slab.direction_tuple())
    center_distance = slab_center_distance(lattice, slab)
    if normal is None or center_distance is None:
        return None
    rows = np.asarray(lattice, dtype=np.float64).reshape(3, 3)
    center = rows.sum(axis=0) * 0.5
    tail = center + (center_distance - float(np.dot(center, normal))) * normal
    length = 0.33 * float(np.linalg.norm(rows.sum(axis=0)))
    return tail, tail + length * normal


def draw_slab_arrow(
    lattice: np.ndarray,
    slab: SelectionSlab,
    *,
    use_cartesian: bool,
    color: tuple[float, float, float, float] = SLAB_ARROW_COLOR,
) -> None:
    """The slab's normal as an arrow: a line in the plot, a head in screen space.

    The shaft is a plot item so it is occluded and foreshortened like any other
    line; the head is drawn on the draw list so it is a fixed pixel size and
    always faces the viewer, the same treatment the site rings get.
    """
    ends = slab_arrow_endpoints(lattice, slab)
    if ends is None:
        return
    tail, tip = (to_display_frame(point, lattice, use_cartesian)[0] for point in ends)
    shaft = np.vstack((tail, tip))
    implot3d.plot_line(
        "##selection_slab_arrow",
        np.ascontiguousarray(shaft[:, 0]),
        np.ascontiguousarray(shaft[:, 1]),
        np.ascontiguousarray(shaft[:, 2]),
        spec=implot3d.Spec(
            line_color=color, marker=implot3d.Marker_.none, line_weight=3.0
        ),
    )
    tail_px = implot3d.plot_to_pixels(float(tail[0]), float(tail[1]), float(tail[2]))
    tip_px = implot3d.plot_to_pixels(float(tip[0]), float(tip[1]), float(tip[2]))
    delta_x, delta_y = tip_px.x - tail_px.x, tip_px.y - tail_px.y
    length = float(np.hypot(delta_x, delta_y))
    packed = imgui.IM_COL32(
        int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), 255
    )
    draw_list = implot3d.get_plot_draw_list()
    if length < 1e-3:
        # Pointing straight at (or away from) the viewer: no direction left to
        # draw, so the head becomes a dot at the tip.
        draw_list.add_circle_filled(tip_px, SLAB_ARROW_HEAD_PX * 0.5, packed, 16)
        return
    unit_x, unit_y = delta_x / length, delta_y / length
    base_x = tip_px.x - unit_x * SLAB_ARROW_HEAD_PX
    base_y = tip_px.y - unit_y * SLAB_ARROW_HEAD_PX
    half = SLAB_ARROW_HEAD_PX * 0.5
    draw_list.add_triangle_filled(
        tip_px,
        imgui.ImVec2(base_x - unit_y * half, base_y + unit_x * half),
        imgui.ImVec2(base_x + unit_y * half, base_y - unit_x * half),
        packed,
    )


def point_in_convex_polygon(point: Tuple[float, float], polygon: np.ndarray) -> bool:
    """Whether a screen point lies inside a convex polygon given as pixel corners."""
    corners = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    if corners.shape[0] < 3:
        return False
    sign = 0
    for start in range(corners.shape[0]):
        ax, ay = corners[start]
        bx, by = corners[(start + 1) % corners.shape[0]]
        cross = (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)
        if abs(cross) < 1e-9:
            continue
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            return False
    return True


def slab_faces_hovered(
    display_faces: Sequence[np.ndarray], mouse: Tuple[float, float]
) -> bool:
    """Whether the cursor is over either face of the drawn slab."""
    for face in display_faces:
        pixels = candidate_pixels(face, range(len(face)))
        if point_in_convex_polygon(mouse, pixels):
            return True
    return False


def slab_pixels_per_angstrom(
    lattice: np.ndarray, slab: SelectionSlab, *, use_cartesian: bool
) -> Tuple[float, float] | None:
    """The screen vector one Angstrom along the slab's normal projects to.

    ImPlot3D is orthographic, so this is constant across the plot and a mouse
    delta dotted with it says how far along the normal the slab was dragged.
    None when the normal points straight at the viewer and a drag means
    nothing.
    """
    ends = slab_arrow_endpoints(lattice, slab)
    normal = slab_normal(lattice, slab.direction_tuple())
    if ends is None or normal is None:
        return None
    tail = ends[0]
    tail_d, step_d = (
        to_display_frame(point, lattice, use_cartesian)[0]
        for point in (tail, tail + normal)
    )
    tail_px = implot3d.plot_to_pixels(float(tail_d[0]), float(tail_d[1]), float(tail_d[2]))
    step_px = implot3d.plot_to_pixels(float(step_d[0]), float(step_d[1]), float(step_d[2]))
    vector = (step_px.x - tail_px.x, step_px.y - tail_px.y)
    if float(np.hypot(*vector)) < 1e-3:
        return None
    return vector


def slab_offset_after_drag(
    lattice: np.ndarray,
    slab: SelectionSlab,
    start_offset: float,
    mouse_delta: Tuple[float, float],
    pixels_per_angstrom: Tuple[float, float],
) -> float:
    """Where a drag of ``mouse_delta`` pixels leaves the slab's offset.

    The delta is projected onto the normal's own screen direction, so dragging
    across the normal does nothing and dragging along it moves the slab by
    exactly the distance the cursor covered -- the slab stays under the hand.
    """
    px, py = pixels_per_angstrom
    scale = px * px + py * py
    if scale < 1e-9:
        return float(start_offset)
    distance_change = (mouse_delta[0] * px + mouse_delta[1] * py) / scale
    start = SelectionSlab(
        direction=slab.direction_tuple(),
        offset=float(start_offset),
        thickness=float(slab.thickness),
    )
    center = slab_center_distance(lattice, start)
    if center is None:
        return float(start_offset)
    return slab_offset_from_distance(lattice, slab, center + distance_change)


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


# The twelve edges of the cell, as index pairs into ``unit_cell_vertices``.
UNIT_CELL_EDGES: Tuple[Tuple[int, int], ...] = (
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
)


def plot_box_wireframe(
    vertices: np.ndarray,
    *,
    label: str,
    color=UNIT_CELL_LINE_COLOR,
    weight: float = UNIT_CELL_LINE_WEIGHT,
) -> None:
    """A box as a wireframe: one line item, NaN-separated between edges.

    Real lines rather than a scatter of sampled points -- the dots read as a dashed
    box at anything but one particular zoom, and cost twelve plot items per frame.
    ``vertices`` are the eight corners in ``unit_cell_vertices`` order.
    """
    line_coords = _segments_to_line_coords(
        [vertices[[start, stop]] for start, stop in UNIT_CELL_EDGES]
    )
    if line_coords is None:
        return
    xs, ys, zs = line_coords
    spec = implot3d.Spec(
        marker=implot3d.Marker_.none,
        line_color=color,
        line_weight=weight,
    )
    implot3d.plot_line(label, xs, ys, zs, spec=spec)


def plot_unit_cell(lattice: np.ndarray, use_cartesian: bool) -> None:
    plot_box_wireframe(unit_cell_vertices(lattice, use_cartesian), label="##unit_cell")


def domain_cell_vertices(
    structure: ChemicalStructure, domain_index: int, use_cartesian: bool
) -> np.ndarray | None:
    """Corners of one domain's block of a stacked structure, or None.

    The block spans the whole cell in plane and its own slice along the
    stacking axis. Worked out in fractional coordinates of the *ideal* stack and
    applied to the structure's actual lattice, so it stays attached to a relaxed
    structure whose cell has moved.
    """
    params = getattr(structure, "generation_parameters", None)
    if params is None or not params.is_multi_domain():
        return None
    domains = params.domains
    axis = int(params.stacking_axis)
    if not 0 <= int(domain_index) < len(domains):
        return None
    thickness = [
        domain.oct_counts()[axis] * domain.lattice[axis] for domain in domains
    ]
    total = float(sum(thickness))
    if total <= 0.0:
        return None
    start = float(sum(thickness[: int(domain_index)])) / total
    stop = start + thickness[int(domain_index)] / total
    lower = np.zeros(3)
    upper = np.ones(3)
    lower[axis], upper[axis] = start, stop
    corners = np.array(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [lower[0], upper[1], lower[2]],
            [upper[0], upper[1], lower[2]],
            [lower[0], lower[1], upper[2]],
            [upper[0], lower[1], upper[2]],
            [lower[0], upper[1], upper[2]],
            [upper[0], upper[1], upper[2]],
        ],
        dtype=np.float64,
    )
    if use_cartesian:
        return corners @ np.asarray(structure.lattice, dtype=np.float64)
    return corners


def plot_miller_planes(
    lattice: np.ndarray,
    miller: Sequence[float],
    offsets: Sequence[float],
    *,
    use_cartesian: bool,
    colors: Sequence[tuple[float, float, float, float]] | None = None,
    legend_label: str = "Ordering planes",
    id_prefix: str = "ordering_plane",
    alpha: float = MILLER_PLANE_ALPHA,
) -> None:
    """Draw one translucent sheet per offset where the plane family cuts the cell.

    ``colors`` tints each sheet individually -- the spin colour its sign string
    assigns, so a magnetic ordering reads as alternating sheets. Omit it for a plain
    overlay in one neutral tint.

    Each sheet is a triangle fan for the fill and a separate closed line for the
    border: drawing the fan with lines on would web the sheet with the fan's internal
    edges, which read as creases that are not there.

    Drawn in the same frame as the unit cell: cartesian when the view is, otherwise
    straight fractional coordinates.

    ``id_prefix`` namespaces the hidden item labels. Every sheet after the first
    is drawn under a ``##``-prefixed label, and ImPlot3D keys items by that label
    -- so two overlays drawn in one frame need different prefixes or the second
    one's sheets land on top of the first one's items.
    """
    frame = np.asarray(lattice, dtype=np.float64) if use_cartesian else np.eye(3)
    normal = np.asarray(miller, dtype=np.float64)
    drawn = 0
    for position, offset in enumerate(offsets):
        if drawn >= MAX_DRAWN_MILLER_PLANES:
            break
        key = (_array_signature(frame, normal), round(float(offset), 9))
        cached = _PLANE_POLYGON_CACHE.get(key)
        if cached is None:
            polygon = plane_cell_polygon(frame, normal, float(offset))
            cached = (polygon, polygon_triangles(polygon))
            _PLANE_POLYGON_CACHE[key] = cached
            while len(_PLANE_POLYGON_CACHE) > PLANE_POLYGON_CACHE_LIMIT:
                _PLANE_POLYGON_CACHE.popitem(last=False)
        else:
            _PLANE_POLYGON_CACHE.move_to_end(key)
        polygon, triangles = cached
        if triangles.shape[0] == 0:
            continue
        color = (
            MILLER_PLANE_NEUTRAL_COLOR
            if colors is None
            else colors[position % len(colors)]
        )
        implot3d.plot_triangle(
            legend_label if drawn == 0 else f"##{id_prefix}_{position}",
            np.ascontiguousarray(triangles[:, 0]),
            np.ascontiguousarray(triangles[:, 1]),
            np.ascontiguousarray(triangles[:, 2]),
            spec=implot3d.Spec(
                fill_color=color,
                line_color=color,
                marker=implot3d.Marker_.none,
                fill_alpha=alpha,
                flags=implot3d.TriangleFlags_.no_lines.value,
            ),
        )
        # The outline, as a closed loop back to the first vertex.
        border = np.vstack((polygon, polygon[:1]))
        implot3d.plot_line(
            f"##{id_prefix}_edge_{position}",
            np.ascontiguousarray(border[:, 0]),
            np.ascontiguousarray(border[:, 1]),
            np.ascontiguousarray(border[:, 2]),
            spec=implot3d.Spec(
                line_color=color,
                marker=implot3d.Marker_.none,
                line_weight=1.5,
            ),
        )
        drawn += 1


def spin_alignment_edge_segments(
    coords: np.ndarray,
    b_grid: np.ndarray | None,
    moments: np.ndarray | None,
    *,
    dot_tol: float = 1e-6,
) -> dict[str, list[np.ndarray]]:
    """Nearest-neighbour B-B bonds, split by whether their moments agree.

    ``b_grid`` maps grid cell -> structure atom index, with ``-1`` for a cell
    whose B site was removed by a vacancy; those cells contribute no bonds.
    """
    if moments is None or b_grid is None:
        return {"aligned": [], "anti-aligned": []}

    moment_vectors = moments_as_vectors(moments, coords.shape[0])
    b_grid = np.asarray(b_grid, dtype=int)
    segments: dict[str, list[np.ndarray]] = {"aligned": [], "anti-aligned": []}

    for grid_index in np.ndindex(b_grid.shape):
        site_index = int(b_grid[grid_index])
        if site_index < 0:
            continue
        site_vector = moment_vectors[site_index]
        if np.linalg.norm(site_vector) <= dot_tol:
            continue
        for axis in range(3):
            if b_grid.shape[axis] <= 1 or grid_index[axis] + 1 >= b_grid.shape[axis]:
                continue
            neighbor_index = list(grid_index)
            neighbor_index[axis] += 1
            neighbor_site = int(b_grid[tuple(neighbor_index)])
            if neighbor_site < 0:
                continue
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
    b_grid: np.ndarray | None,
    moments: np.ndarray | None,
) -> dict[str, int]:
    return {
        label: len(edge_segments)
        for label, edge_segments in spin_alignment_edge_segments(
            coords,
            b_grid,
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
    b_grid: np.ndarray | None,
    moments: np.ndarray | None,
    *,
    line_width: float = 3.0,
) -> None:
    for label, edge_segments in spin_alignment_edge_segments(coords, b_grid, moments).items():
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
    "periodic_axes_flags",
    "builder_domains",
    "stacking_axis",
    "active_domain_index",
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
    "perovskite_supercell_x",
    "perovskite_supercell_y",
    "perovskite_supercell_z",
    "lattice_a",
    "lattice_b",
    "lattice_c",
    "perovskite_tilt_system",
    "tilt_angle_x",
    "tilt_angle_y",
    "tilt_angle_z",
    "perovskite_center",
    "defect_entries",
)

# Defect panel vocabulary. Kinds and role filters are combo indices into these.
# There is no "Vacancy" kind. A substitution with no element named *is* one, so
# offering both meant two ways to say the same thing, and a blank box that had to
# be explained as a mistake rather than read as a choice. Emptying the element box
# is now how a site is emptied, and ``defect_for_entry`` turns it back into the
# vacancy the model still has. Protons stay: an interstitial is not a replacement.
# Prefix of the note left when an in-plane edit is refused; cleared by the next
# in-plane edit the stack accepts.
_IN_PLANE_KEPT = "In-plane size kept as it was: "

DEFECT_KIND_LABELS: Tuple[str, ...] = ("Substitute", "Proton (H)")
DEFECT_KIND_KEYS: Tuple[str, ...] = ("substitution", "proton")


@dataclass
class DefectEntry:
    """One defect on one site, complete in itself.

    ``element`` is the replacement symbol for a substitution -- blank means the
    site is vacated -- and ``orientation`` picks among the four equivalent
    proton sites; each is ignored by the kind that has no use for it. The grid
    address is never clamped to the current supercell.
    """

    site: SiteKey
    kind: int = 0
    element: str = ""
    orientation: int = 0

    def kind_key(self) -> str:
        return DEFECT_KIND_KEYS[int(self.kind) % len(DEFECT_KIND_KEYS)]

    def signature(self) -> Tuple[object, ...]:
        return (tuple(self.site), int(self.kind), self.element, int(self.orientation))


@dataclass
class AtomRow:
    """One site of the focused structure as the Atoms panel lists it.

    ``ref`` is the stable handle a selection holds: the grid key of a built
    site (``("site", role, i, j, k, vertex)``), the host key of a built proton
    (``("proton", ...)``), or the rounded position of a loaded atom
    (``("pos", x, y, z)``). ``index`` is the atom's position in the focus, or
    -1 for a vacated site, which is still a row -- a vacancy has to be listed to
    be undone. ``ideal_element`` is what the site holds when nothing has been
    done to it: the ideal lattice's species for a built site, the label that
    was removed for a loaded vacancy.
    """

    ref: Tuple[object, ...]
    index: int
    element: str
    ideal_element: str
    cartesian: np.ndarray
    key: SiteKey | None = None
    host_key: SiteKey | None = None
    role: str = ""

    @property
    def vacant(self) -> bool:
        return self.index < 0

    @property
    def edited(self) -> bool:
        """Whether the row differs from the ideal lattice (or the loaded file)."""
        return self.vacant or self.role == "H" or self.element != self.ideal_element


@dataclass(frozen=True)
class EnergySample:
    """One CHGNet single-point energy, as the energy pane plots it.

    ``per_atom`` is what is drawn: structures of different sizes share the
    pane once single-point jobs feed it, and total energies would not compare.
    ``live`` separates the toggle's automatic samples from the single points
    asked for by hand.
    """

    energy: float
    atom_count: int
    label: str
    live: bool
    stamp: float

    @property
    def per_atom(self) -> float:
        return float(self.energy) / self.atom_count if self.atom_count else float(self.energy)


@dataclass(frozen=True)
class DomainBoundary:
    """One interface of the active domain, as the builder shows it.

    ``upper_index`` is the domain whose ``interface_from_previous`` flag decides
    the plane's owner; ``side`` says which face of the active domain it is.
    """

    upper_index: int
    neighbour_index: int
    side: str  # "below" or "above"
    owned: bool
    editable: bool
    neighbour_label: str


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
    # Ring the magnetic sites that disagree with the matched ideal ordering -- the
    # per-site picture behind the defect concentration.
    show_spin_defect_rings: bool = False
    # Translucent sheets on the planes of the selected configuration's ordering.
    show_miller_planes: bool = False
    # Site roles drawn in the 3D view, toggled beside the spin view options.
    show_a_sites: bool = True
    show_b_sites: bool = True
    show_x_sites: bool = True
    # Recolour magnetic atoms by the sign of their moment. Off by default: the
    # element colours are what most of the work is done against, and the spin
    # colouring is only meaningful once a configuration has been chosen.
    color_atoms_by_spin: bool = False
    show_octahedra: bool = True
    show_unit_cell: bool = True
    # The block of the domain being edited, in yellow over a stacked structure.
    show_domain_cell: bool = True
    # Periodicity per lattice axis. ``treat_as_periodic`` (below) reads this
    # back as the per-axis triple every grid helper accepts.
    periodic_axes_flags: Tuple[bool, bool, bool] = (True, True, True)
    # The domains of the structure being built, in stacking order. The
    # composition / supercell / lattice fields on this state are a *buffer* for
    # ``builder_domains[active_domain_index]``: the widgets edit the buffer and
    # ``store_active_domain`` writes it back every frame, so with a single
    # domain the buffer simply is the structure, as it always was. Everything
    # not in a DomainSpec -- tilts, periodicity, defects, the centre -- is
    # global to the stack.
    builder_domains: List[DomainSpec] = field(default_factory=lambda: [DomainSpec()])
    stacking_axis: int = 2
    active_domain_index: int = 0
    # A domain edit the stack could not accept (a double perovskite on an odd
    # in-plane grid, say), shown under the domain row.
    domain_message: str = ""
    _in_plane_refused_at: Tuple[object, ...] | None = None
    # Which domain each structure was last being edited on, by id(), so
    # switching focus between structures comes back to the same block.
    _structure_domain_choice: Dict[int, int] = field(default_factory=dict, repr=False)
    # Whether the active domain's block is outlined in the view. Picking a domain
    # in the Structure panel turns it on; clicking the structure row itself (the
    # whole structure, not one block of it) turns it off.
    domain_highlight_active: bool = False
    render_periodic_images: bool = True
    perovskite_center: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float32)
    )
    formula_mode: int = 0
    # A formula the user picked that would discard their edits, held while the
    # confirmation dialog is up. -1 when nothing is pending.
    pending_formula_mode: int = -1
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
    # Point defects, one flat list. Grid addresses are stored raw and never
    # clamped: a site that falls outside a shrunken supercell is skipped while
    # out of range and comes back intact when the cell grows again.
    defect_entries: List[DefectEntry] = field(default_factory=list)
    defect_message: str = ""
    # Atoms picked by hand, as row refs (see ``AtomRow.ref``). Kept apart from
    # the slab's contents, which are recomputed from its geometry every frame:
    # the two combine into what is active, and these alone are the ones the
    # panel lists first, so a pick survives the slab moving off it.
    atom_selection: set = field(default_factory=set)
    # The selection slab, and whether it is in play. Off, the panel lists every
    # atom; on, its contents join the hand picks.
    selection_slab: SelectionSlab = field(default_factory=SelectionSlab)
    slab_enabled: bool = False
    # A drag of the slab in the 3D view, in progress: the offset it started
    # from and the mouse position it started at.
    _slab_drag: Tuple[float, float, float] | None = None
    # Atoms removed from a loaded structure, by id() of the structure, so a
    # vacancy can be listed and put back. A built structure needs none of this:
    # its vacancies are defect entries and its ideal positions come from the
    # build.
    _loaded_vacancies: Dict[int, List[VacatedAtom]] = field(
        default_factory=dict, repr=False
    )
    # What a relabelled atom of a loaded structure used to be, by id() of the
    # structure and the atom's row ref, so the row can say what it was and
    # typing the original back reads as an un-edit.
    _loaded_relabels: Dict[int, Dict[Tuple[object, ...], str]] = field(
        default_factory=dict, repr=False
    )
    # The row under the cursor in the Atoms panel, ringed in the 3D view. Read
    # and cleared by the view, which is drawn before the panel, so the ring
    # follows the cursor one frame behind and cannot outlive the panel.
    atom_hover_ref: Tuple[object, ...] | None = None
    # The row the panel is editing (its element box has focus), so the box
    # keeps its text while the row is being typed in.
    atom_edit_ref: Tuple[object, ...] | None = None
    atom_edit_text: str = ""
    # A refused edit, shown under the panel.
    atom_edit_message: str = ""
    # Which feature the 3D view is decorating for: "exchange" or "plain". The
    # exchange-coupling plot claims the view when an atom's couplings are
    # selected; anything else gives the plain structure back.
    structure_view_focus: str = "plain"
    # Supercell size in primitive cells per axis: 1 is the primitive cell. The
    # default is 3 so the app opens on a 3x3x3 grid -- comfortably above the two
    # cells per axis the A/C/G reference orderings need, and closer to the cells
    # actually worked with. The ordered formula modes double the grid themselves,
    # so they default to 2 instead (see default_supercell_for_formula).
    perovskite_supercell_x: int = 3
    perovskite_supercell_y: int = 3
    perovskite_supercell_z: int = 3
    lattice_a: float = 4.0
    lattice_b: float = 4.0
    lattice_c: float = 4.0
    # The cell editor's own six parameters, for a structure with no builder
    # provenance. Deliberately *not* the lattice_a/b/c above: those are one
    # octahedron's cube edge and mean nothing without a grid to repeat over,
    # while these are the cell vectors the file actually carries. They are a
    # buffer bound to the focus by sync_cell_binding, not a source of truth --
    # the structure's own lattice is that.
    cell_a: float = 4.0
    cell_b: float = 4.0
    cell_c: float = 4.0
    cell_alpha: float = 90.0
    cell_beta: float = 90.0
    cell_gamma: float = 90.0
    # Move a, b and c together, keeping the cell's shape. Stands in for the
    # cubic/tetragonal/orthorhombic radios, which describe a perovskite the
    # builder is generating rather than a cell that arrived from a file.
    cell_lock_aspect: bool = False
    # Tiling is a button, not a live edit: it changes the atom count, so it
    # cannot be re-derived from a signature the way a strain can.
    cell_tile_x: int = 1
    cell_tile_y: int = 1
    cell_tile_z: int = 1
    cell_message: str = ""
    perovskite_tilt_system: int = 0
    tilt_angle_x: float = 0.0
    tilt_angle_y: float = 0.0
    tilt_angle_z: float = 0.0
    # Net charge of the cell for oxidation-state enumeration. 0 is right for the
    # usual defect chemistry (an O vacancy is compensated by reducing cations),
    # but a deliberately charged supercell needs to say so.
    magnetic_net_charge: int = 0
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
    # Always the single lowest-energy assignment the model produced, held in a list
    # only because everything downstream already reads it out of one. Choosing
    # between ranked assignments is gone: the ones past the first were routinely
    # mixed-valence in ways nothing in the cell justified, and picking among them by
    # flipping through a combo was guessing. What the model is confident about is
    # its ranking's head; everything else is now an explicit edit.
    magnetic_oxidation_assignments: List[OxidationStateAssignment] = field(default_factory=list)
    selected_oxidation_assignment_index: int = 0
    # Oxidation states set by hand, layered over the assignment above. See
    # ``quick_mag.oxidation_overrides`` for what the two scopes mean.
    oxidation_overrides: OxidationOverrides = field(default_factory=OxidationOverrides)
    # Bumped on every edit. The effective assignment is memoized per frame and this
    # is what tells the memo that an edit landed -- the override dicts are mutated
    # in place, so their identity says nothing.
    oxidation_override_generation: int = 0
    # What the oxidation-state box holds, and which atom it was seeded from. The
    # box is a staging value rather than a live binding: it commits when the edit
    # is finished, so half-typed digits never reach the structure and never
    # rebuild the exchange matrix on the way to a number the user has not
    # finished writing.
    oxidation_edit_site: int = -1
    oxidation_edit_value: int = 0
    # An edit that could not be applied, shown under the list. Failing soft here
    # matters more than most places: this is a text box wired to a rebuild of the
    # whole exchange matrix, and an app that dies on a bad value loses the session.
    oxidation_edit_message: str = ""
    # True while the hand-set states have no charge-balanced completion and are
    # therefore applied verbatim, leaving the cell's net charge to drift.
    oxidation_solve_unbalanced: bool = False
    # The atom under the cursor in a per-atom plot (the drift plot, say), ringed
    # in the 3D view. Read and cleared by the view, which is drawn before the
    # panels -- so the ring follows the cursor one frame behind, which is
    # invisible, and cannot outlive the panel that set it by more than that.
    oxidation_hover_site: int = -1
    selected_spin_config_index: int = 0
    # Atom picked in the per-site oxidation/moment list, ringed in the 3D view.
    # -1 is "nothing selected"; indexes the analysed structure, not the render.
    selected_site_index: int = -1
    # The atom a click on the hovered exchange bar would select, ringed white
    # in the 3D view while the bar is hovered. Written by the bar chart at the
    # end of a frame and read by the 3D plot at the start of the next -- the
    # one-frame lag is invisible. -1 when no bar is hovered.
    exchange_hover_site: int = -1
    magnetic_site_indices: List[int] = field(default_factory=list)
    magnetic_j_matrix: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 0), dtype=np.float64)
    )
    # The same couplings, one entry per bridged pair, with the element symbols,
    # distances and ligands the matrix drops. Built alongside the matrix; this is
    # what the exchange-coupling plot draws.
    magnetic_pair_couplings: List[PairCoupling] = field(default_factory=list)
    magnetic_solution_cache: Dict[int, Tuple[List[Any], List[Any]]] = field(default_factory=dict)
    # The plotted spin-energy landscape. These configurations persist across builder
    # edits: only their energies are recomputed against the new J matrix, so the plot
    # tracks the structure instead of resetting. New configurations come only from a
    # user-requested solve.
    spin_landscape: List[SpinConfig] = field(default_factory=list)
    # The subset of the pool actually plotted: the pool survives so that toggling
    # ``plot_degenerate_configs`` back on can restore what it hid.
    spin_display_configs: List[SpinConfig] = field(default_factory=list)
    reference_configs: List[Tuple[str, SpinConfig]] = field(default_factory=list)
    # Orderings entered by hand, as plane-notation labels ("(011) ++--"). Held as
    # patterns rather than as moments so they are rescored against the current J
    # alongside the canonical ones, and so they survive a builder edit or a resize
    # of the cell that would invalidate a fixed set of moments.
    custom_spin_patterns: List[str] = field(default_factory=list)
    custom_pattern_miller: List[int] = field(default_factory=lambda: [0, 0, 1])
    custom_pattern_signs: str = "+-"
    custom_pattern_message: str = ""
    # Re-energizing the landscape after a builder edit means rebuilding the
    # oxidation assignments and the exchange matrix -- tens to hundreds of
    # milliseconds on a large cell, on every frame of a slider drag. On by
    # default, but only while the view can afford it: ``interactive_updates_live``
    # pauses it when the frame rate falls too far, and edits made while it is
    # paused mark the energies stale until "Refresh energies" or a solve runs.
    update_spin_energies_interactively: bool = True
    # Whether the frame-rate gate is currently letting it run. Held across frames
    # because the decision has hysteresis -- see ``interactive_updates_live``.
    _interactive_updates_live: bool = True
    spin_energies_stale: bool = False
    spin_plot_max_configs: int = 100
    # The model produces many configurations at identical energies, which crowd the
    # plot; off, the cap is spent on distinct energies instead.
    plot_degenerate_configs: bool = False
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
    # The cell editor binds the same way the builder does, and for the same
    # reason: the widgets need somewhere to hold a half-typed number that is not
    # yet a valid cell.
    # Held by identity rather than ``id()``, so a reallocated object at the same
    # address cannot be mistaken for the structure the fields were seeded from --
    # loaded structures are exactly the ones the user creates and deletes by hand.
    _cell_bound_structure: ChemicalStructure | None = None
    _cell_applied_sig: Tuple[object, ...] | None = None
    # b:a and c:a frozen when the aspect lock went on, so locking preserves the
    # shape the user is looking at rather than snapping to a cube.
    _cell_aspect_ratio: Tuple[float, float] = (1.0, 1.0)
    _last_formula_mode: int = 0
    structure_zoom: float = 1.0
    # Set by the a/b/c buttons above the 3D view, turned into a rotation target by the
    # next frame's plot setup (which is where the lattice directions are known) and
    # cleared there.
    pending_view_axis: int | None = None
    # The 3D view's orientation is ours rather than ImPlot3D's -- see
    # ``rotation_after_drag`` for why -- so it lives here, as (x, y, z, w), and is pushed
    # into the plot every frame. ``structure_rotation_target`` is the pose an alignment
    # button asked for while the swing towards it is still running.
    structure_rotation: Tuple[float, float, float, float] = DEFAULT_STRUCTURE_ROTATION
    structure_rotation_target: Tuple[float, float, float, float] | None = None
    _structure_drag_active: bool = False
    # Which of the screen x/y/z axes the turn buttons step about, as an index into
    # SCREEN_TURN_AXES. Starts on screen x -- tipping the cell towards or away from
    # the viewer, which is the step most often wanted from a settled view.
    screen_turn_axis_index: int = 0
    _spin_plot_axis_solution: Any = None
    _relaxation_plot_axis_key: Any = None
    _exchange_plot_axis_key: Any = None
    # Which plot the 2D pane is showing, as an index into TWO_D_PLOT_NAMES.
    two_d_plot_index: int = 0
    # Share of the Structure View the 2D plot gets, dragged on the splitter between
    # the two. A fraction rather than a height so it survives the pane resizing.
    two_d_pane_fraction: float = 0.30
    # Bumped whenever the pair table is rebuilt, so anything derived from it knows
    # to recompute without having to compare the couplings themselves.
    _exchange_generation: int = 0
    # The bar order, fixed when an atom is selected rather than resorted every
    # frame. See ``visible_pair_couplings``.
    _exchange_bar_order: Tuple[Tuple[int, int], ...] = ()
    _exchange_bar_order_key: Any = None
    # The spin arrangement the user is looking at, remembered by its moments rather
    # than by its position. The landscape is re-sorted by energy every time it is
    # re-energized, so a builder edit moves a configuration up or down the list --
    # holding the index would silently swap which arrangement is on screen.
    _selected_spin_key: Tuple[float, ...] | None = None
    # Single-slot memos for the per-frame rebuilds the 3D view and the builder panel
    # would otherwise repeat every frame. Each entry is (key, value); a key mismatch
    # rebuilds. See _structure_signature for why these are content-addressed.
    _render_cache: "OrderedDict[object, Tuple[object, Any]]" = field(
        default_factory=OrderedDict, repr=False
    )
    # Bumped whenever the plotted landscape is rebuilt. Keying the pattern-match
    # cache on this rather than on the configurations themselves keeps the key O(1)
    # -- hashing 216 moments per configuration per frame was itself material.
    _landscape_generation: int = 0

    # ------------------------------------------------------------------
    # Remote compute: CHGNet on a machine that has the hardware for it.
    # ------------------------------------------------------------------
    # Always a loopback address, even when the calculation runs on a cluster: the
    # deployment detail is an SSH tunnel, not a hostname the app has to know. That
    # is also what makes this reachable from the web build at all, where an
    # https:// page may talk to 127.0.0.1 but not to an arbitrary host.
    remote_url: str = REMOTE_DEFAULT_URL
    remote_token: str = ""
    remote_calculation_index: int = REMOTE_CALCULATIONS.index("cell+atoms")
    remote_optimizer_index: int = REMOTE_OPTIMIZERS.index("LBFGS")
    remote_fmax: float = 0.05
    remote_steps: int = 50
    remote_message: str = ""
    # Focus the relaxed structure as soon as it lands. Off for a batch, where the
    # view would otherwise jump every time one finished.
    remote_focus_on_arrival: bool = True
    # Which job's energy trace is plotted. A key rather than an index: the list
    # shifts as jobs are cleared, and an index would silently swap the plot.
    remote_selected_job_key: str = ""
    # CHGNet's per-site |m| magnitudes, keyed by id() of the structure they belong
    # to. Deliberately not ChemicalStructure.magnetic_moments: those carry signed
    # spins from the solver, and these are unsigned diagnostics that would quietly
    # corrupt every spin feature if they were written there.
    chgnet_moments: Dict[int, np.ndarray] = field(default_factory=dict, repr=False)
    # How far each relaxed structure ended up from the geometry it was submitted
    # as, keyed the same way. This is what stands in for builder parameters that
    # a relaxation has invalidated: not what the structure is, but how far it is
    # from what the builder would make.
    relaxation_drift: Dict[int, GeometryDrift] = field(default_factory=dict, repr=False)
    # Built on first use rather than at startup: constructing it probes for the
    # browser bridge, and most sessions never submit anything.
    _remote_client: Any = field(default=None, repr=False)
    # The structure-list entry a submitted relaxation occupies while it runs,
    # keyed by the job's client key. A copy of what was submitted, so it can be
    # looked at meanwhile; the result takes its place in the list when it lands.
    _remote_placeholders: Dict[str, ChemicalStructure] = field(
        default_factory=dict, repr=False
    )

    # The live CHGNet energy: a single-point on the active structure, requested
    # at most every LIVE_ENERGY_INTERVAL_SECONDS and never while one is out.
    # ``live_energy_points`` is one EnergySample per structure an energy came
    # back for -- live samples while the toggle is on, and every single-point
    # job -- oldest first, capped at LIVE_ENERGY_MAX_POINTS.
    live_energy_enabled: bool = False
    live_energy_points: List["EnergySample"] = field(default_factory=list)
    live_energy_last: float | None = None
    live_energy_message: str = ""
    _live_energy_job: Any = field(default=None, repr=False)
    _live_energy_sent_at: float = 0.0
    # Content signature of the structure the last energy was computed for. An
    # unchanged structure re-plots that energy rather than asking again.
    _live_energy_signature: Any = None
    _live_energy_pending_signature: Any = None

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

    def focus_has_generation_parameters(self) -> bool:
        """Provenance of any kind: this structure knows its A/B/X site roles.

        True for a relaxed structure too. What the parameters describe after a
        relaxation is the *topology* -- atom order, the B-site grid -- which is
        what site indexing, the octahedral mesh and the periodic-image rebuild
        all read, and none of which a relaxation changes.
        """
        return (
            self.focus is not None
            and getattr(self.focus, "generation_parameters", None) is not None
        )

    def focus_has_generated_provenance(self) -> bool:
        """Provenance the builder can still *drive*: geometry included.

        Stricter than the above, and the difference is the whole point. A relaxed
        structure keeps its parameters but its coordinates are no longer what they
        rebuild, so every path that would regenerate geometry from them has to
        stop here -- otherwise the first builder edit silently replaces the
        relaxation with the ideal cell it started from.
        """
        return self.focus_has_generation_parameters() and bool(
            getattr(self.focus, "geometry_matches_generation", True)
        )

    def focus_is_relaxed_from_builder(self) -> bool:
        """Has builder provenance, but a geometry the builder can no longer make."""
        return (
            self.focus_has_generation_parameters()
            and not self.focus_has_generated_provenance()
        )

    # The builder panel used to be one switch: provenance or nothing. It is now a
    # question per section, because the sections need different things. Editing a
    # cell needs no knowledge of the structure at all; naming the A-site element
    # needs to know which atoms *are* A sites, and only the builder knows that.
    def cell_editing_available(self) -> bool:
        """Whether the cell editor drives the focus.

        Every structure has a cell, but a builder-generated one is a function of
        its parameters -- straining it would be undone by the next regeneration --
        so the cell editor takes exactly the structures the builder does not
        drive: the ones loaded from a file, and the ones a relaxation has moved
        off their generated geometry. For the latter this is also the only place
        the *actual* lattice constants are shown, the builder's own fields having
        gone stale the moment the cell relaxed.
        """
        return self.focus is not None and not self.focus_has_generated_provenance()

    def composition_editing_available(self) -> bool:
        """A/B/X element fields. They name site roles, which come from the builder."""
        return self.focus_has_generated_provenance()

    def tilt_editing_available(self) -> bool:
        """Glazer tilts, which need the octahedral network the builder generates."""
        return self.focus_has_generated_provenance() and self.tilt_system_available()

    def defect_editing_available(self) -> bool:
        """Defect placement, which addresses sites by grid key.

        A loaded structure has no grid keys: recovering them means fitting its
        sublattices back onto a perovskite grid, which is not done yet.
        """
        return self.focus_has_generated_provenance()

    def unavailable_reason(self, section: str) -> str:
        """Why ``section`` is greyed out for the current focus, or ""."""
        if self.focus is None:
            return "No structure is active."
        if self.focus_is_relaxed_from_builder():
            return (
                "This structure was relaxed, so its coordinates are no longer the "
                "ones the builder parameters generate. Editing them here would "
                "rebuild the ideal geometry and discard the relaxation. The cell "
                "editor above shows its actual lattice."
            )
        if not self.focus_is_loaded():
            return ""
        return {
            "composition": (
                "Element fields name the A, B and X sublattices, which a structure "
                "loaded from a file does not record. Its atoms are editable as "
                "geometry, not as site roles."
            ),
            "tilt": (
                "Tilt systems rotate the octahedral network, which is not inferred "
                "from a loaded file. The cell itself is editable above."
            ),
            "defects": (
                "Defects address a site by its grid position, which a loaded "
                "structure does not carry. Recovering it means fitting the "
                "sublattices back onto a perovskite grid."
            ),
        }.get(section, "")

    def magnetic_results_match_focus(self) -> bool:
        return self.focus is not None and self.magnetic_result_structure is self.focus

    def set_focus(self, structure: ChemicalStructure | None) -> None:
        self.focus = structure
        self.active_saved_spin_index = -1
        # Focusing a structure is about the whole thing; a domain outline is
        # asked for again by picking a domain in the panel.
        self.domain_highlight_active = False

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
        self.store_active_domain()
        return (
            tuple(self.periodic_axes_flags),
            self.render_periodic_images,
            self.formula_mode,
            self.perovskite_type,
            self.perovskite_supercell_x,
            self.perovskite_supercell_y,
            self.perovskite_supercell_z,
            round(self.lattice_a, 6),
            round(self.lattice_b, 6),
            round(self.lattice_c, 6),
            self.a_site_element.strip(),
            self.b_site_element.strip(),
            self.x_site_element.strip(),
            self.a2_site_element.strip(),
            self.b2_site_element.strip(),
            self.high_entropy_signature(),
            tuple(domain.signature() for domain in self.builder_domains),
            int(self.stacking_axis),
            self.perovskite_tilt_system,
            round(self.tilt_angle_x, 6),
            round(self.tilt_angle_y, 6),
            round(self.tilt_angle_z, 6),
            round(float(self.perovskite_center[0]), 6),
            round(float(self.perovskite_center[1]), 6),
            round(float(self.perovskite_center[2]), 6),
            self.defects_signature(),
        )

    # ------------------------------------------------------------------
    # Periodicity
    # ------------------------------------------------------------------
    @property
    def treat_as_periodic(self) -> Tuple[bool, bool, bool]:
        """Per-axis periodicity of the structure being built.

        A triple rather than a bool, which every grid helper accepts directly;
        ``any(...)`` is the whole-structure reading.
        """
        return tuple(self.periodic_axes_flags)  # type: ignore[return-value]

    @treat_as_periodic.setter
    def treat_as_periodic(self, value) -> None:
        self.periodic_axes_flags = periodic_axes(value)

    def builder_is_periodic(self) -> bool:
        return any(self.periodic_axes_flags)

    # ------------------------------------------------------------------
    # Domains
    # ------------------------------------------------------------------
    def domain_count(self) -> int:
        return len(self.builder_domains)

    def active_domain(self) -> DomainSpec:
        if not self.builder_domains:
            self.builder_domains = [DomainSpec()]
        self.active_domain_index = min(
            max(int(self.active_domain_index), 0), len(self.builder_domains) - 1
        )
        return self.builder_domains[self.active_domain_index]

    def store_active_domain(self) -> DomainSpec:
        """Write the builder's buffer fields into the active DomainSpec."""
        domain = self.active_domain()
        domain.formula_mode = self.formula_key()
        domain.a_site_element = self.a_site_element
        domain.b_site_element = self.b_site_element
        domain.x_site_element = self.x_site_element
        domain.a2_site_element = self.a2_site_element
        domain.b2_site_element = self.b2_site_element
        domain.high_entropy_a_sites = self.high_entropy_entries("A")
        domain.high_entropy_b_sites = self.high_entropy_entries("B")
        domain.high_entropy_x_sites = self.high_entropy_entries("X")
        domain.n_cells = (
            max(1, int(self.perovskite_supercell_x)),
            max(1, int(self.perovskite_supercell_y)),
            max(1, int(self.perovskite_supercell_z)),
        )
        domain.lattice = (float(self.lattice_a), float(self.lattice_b), float(self.lattice_c))
        return domain

    def load_domain_into_buffer(self, domain: DomainSpec) -> None:
        """Seed the builder's buffer fields from ``domain``."""
        self.formula_mode = formula_index_from_key(domain.formula_mode)
        self._last_formula_mode = self.formula_mode
        self.a_site_element = domain.a_site_element
        self.b_site_element = domain.b_site_element
        self.x_site_element = domain.x_site_element
        self.a2_site_element = domain.a2_site_element
        self.b2_site_element = domain.b2_site_element
        self.set_high_entropy_entries("A", list(domain.high_entropy_a_sites))
        self.set_high_entropy_entries("B", list(domain.high_entropy_b_sites))
        self.set_high_entropy_entries("X", list(domain.high_entropy_x_sites))
        (
            self.perovskite_supercell_x,
            self.perovskite_supercell_y,
            self.perovskite_supercell_z,
        ) = (int(value) for value in domain.n_cells)
        self.lattice_a, self.lattice_b, self.lattice_c = (float(v) for v in domain.lattice)
        if (
            abs(self.lattice_a - self.lattice_b) < 1e-6
            and abs(self.lattice_a - self.lattice_c) < 1e-6
        ):
            self.perovskite_type = 0
        elif abs(self.lattice_a - self.lattice_b) < 1e-6:
            self.perovskite_type = 1
        else:
            self.perovskite_type = 2

    def select_domain(self, index: int) -> None:
        """Make domain ``index`` the one the builder fields edit."""
        self.store_active_domain()
        index = min(max(int(index), 0), len(self.builder_domains) - 1)
        if index == self.active_domain_index:
            return
        self.active_domain_index = index
        self.load_domain_into_buffer(self.builder_domains[index])
        if self.focus is not None:
            self._structure_domain_choice[id(self.focus)] = index

    def domain_index_for_structure(self, structure: ChemicalStructure) -> int:
        """The domain a structure's dropdown shows: live for the focus, remembered otherwise."""
        if structure is self.focus and self._builder_bound_id == id(structure):
            return int(self.active_domain_index)
        return int(self._structure_domain_choice.get(id(structure), 0))

    def choose_structure_domain(self, structure: ChemicalStructure, index: int) -> None:
        """The structure panel's dropdown: focus ``structure`` and edit domain ``index``."""
        self._structure_domain_choice[id(structure)] = int(index)
        if structure is not self.focus:
            self.set_focus(structure)
            self.sync_active_structure()
        if self._builder_bound_id == id(structure):
            self.select_domain(index)
        self.domain_highlight_active = True

    def add_domain(self) -> bool:
        """Grow the stack by one domain, a copy of the last one, and select it.

        The copy already matches in plane, so the new block is a continuation
        of the stack until its formula or composition is changed. Its own
        extent along the stacking axis starts at one primitive cell.
        """
        self.store_active_domain()
        template = self.builder_domains[-1]
        new_domain = DomainSpec(**template.to_dict())
        n_cells = list(new_domain.n_cells)
        n_cells[int(self.stacking_axis)] = 1
        new_domain.n_cells = tuple(n_cells)  # type: ignore[assignment]
        new_domain.interface_from_previous = False
        new_domain.name = ""
        try:
            conform_domain_to_stack(new_domain, self.builder_domains[0], self.stacking_axis)
        except ValueError as exc:
            self.domain_message = str(exc)
            return False
        self.builder_domains.append(new_domain)
        self.active_domain_index = len(self.builder_domains) - 1
        self.load_domain_into_buffer(new_domain)
        self.domain_message = ""
        self.domain_highlight_active = True
        return True

    def remove_domain(self, index: int) -> None:
        if len(self.builder_domains) <= 1:
            return
        index = min(max(int(index), 0), len(self.builder_domains) - 1)
        del self.builder_domains[index]
        self.active_domain_index = min(self.active_domain_index, len(self.builder_domains) - 1)
        self.load_domain_into_buffer(self.builder_domains[self.active_domain_index])

    def set_active_domain_formula(self, mode: int) -> bool:
        """Change the active domain's formula, keeping it matched in plane.

        Only meaningful for a stack; a lone domain goes through
        ``apply_formula_change``, which resets the whole builder. Fails, with
        the reason in ``domain_message``, when the new formula cannot sit on
        the neighbouring domains' in-plane grid.
        """
        domain = self.store_active_domain()
        key = formula_key_from_index(mode)
        others = [d for d in self.builder_domains if d is not domain]
        reference = others[0] if others else None
        candidate = DomainSpec(**domain.to_dict())
        candidate.formula_mode = key
        if reference is not None:
            try:
                conform_domain_to_stack(candidate, reference, self.stacking_axis)
            except ValueError as exc:
                self.domain_message = str(exc)
                return False
        domain.formula_mode = key
        domain.n_cells = candidate.n_cells
        domain.lattice = candidate.lattice
        self.load_domain_into_buffer(domain)
        self.apply_default_composition_for_formula()
        self.domain_message = ""
        return True

    def conform_stack_to_active_domain(self) -> None:
        """Propagate the active domain's in-plane size and spacing to the rest.

        The in-plane grid is one grid for the whole stack, so an edit to it on
        any domain is an edit to every domain. When some other domain cannot
        follow -- its formula needs an even octahedron count the new size does
        not have -- the edit is refused and the active domain is put back to
        what the others still agree on.
        """
        if len(self.builder_domains) < 2:
            return
        active = self.store_active_domain()
        others = [d for d in self.builder_domains if d is not active]
        try:
            for other in others:
                conform_domain_to_stack(other, active, self.stacking_axis)
            # The refusal note stays up until an in-plane edit is accepted.
            if (
                self.domain_message.startswith(_IN_PLANE_KEPT)
                and self._in_plane_signature() != self._in_plane_refused_at
            ):
                self.domain_message = ""
        except ValueError as exc:
            self.domain_message = f"{_IN_PLANE_KEPT}{exc}"
            conform_domain_to_stack(active, others[0], self.stacking_axis)
            self.load_domain_into_buffer(active)
            self._in_plane_refused_at = self._in_plane_signature()
        self.store_active_domain()

    def _in_plane_signature(self) -> Tuple[object, ...]:
        first, second = in_plane_axes(self.stacking_axis)
        domain = self.store_active_domain()
        return (
            domain.n_cells[first],
            domain.n_cells[second],
            round(domain.lattice[first], 6),
            round(domain.lattice[second], 6),
        )

    def set_stacking_axis(self, axis: int) -> bool:
        axis = int(axis)
        if axis == int(self.stacking_axis):
            return True
        previous = int(self.stacking_axis)
        self.stacking_axis = axis
        try:
            for domain in self.builder_domains[1:]:
                conform_domain_to_stack(domain, self.builder_domains[0], axis)
        except ValueError as exc:
            self.stacking_axis = previous
            self.domain_message = str(exc)
            return False
        self.load_domain_into_buffer(self.active_domain())
        self.domain_message = ""
        return True

    def stack_problems(self) -> List[str]:
        self.store_active_domain()
        return validate_stack(self.builder_domains, self.stacking_axis)

    def active_domain_boundaries(self) -> List["DomainBoundary"]:
        """The two interfaces of the active domain: with the block below, then above.

        Each is one AO plane with exactly one owner, and the owner is stored as
        the upper domain's ``interface_from_previous``; this view turns that
        single flag into a per-domain "owns it" reading for whichever side of
        the plane the builder is looking at.
        """
        count = len(self.builder_domains)
        if count < 2:
            return []
        index = int(self.active_domain_index)
        periodic = bool(self.periodic_axes_flags[int(self.stacking_axis)])
        boundaries: List[DomainBoundary] = []
        # Below: the plane at this domain's low face.
        below_editable = index > 0 or periodic
        boundaries.append(
            DomainBoundary(
                upper_index=index,
                neighbour_index=(index - 1) % count,
                side="below",
                owned=below_editable and not self.builder_domains[index].interface_from_previous,
                editable=below_editable,
                neighbour_label=f"Domain {((index - 1) % count) + 1}",
            )
        )
        # Above: the plane at the next domain's low face.
        above_editable = index < count - 1 or periodic
        upper = (index + 1) % count
        boundaries.append(
            DomainBoundary(
                upper_index=upper,
                neighbour_index=upper,
                side="above",
                owned=above_editable and bool(self.builder_domains[upper].interface_from_previous),
                editable=above_editable,
                neighbour_label=f"Domain {upper + 1}",
            )
        )
        return boundaries

    def set_interface_ownership(self, boundary: "DomainBoundary", owned: bool) -> None:
        """Give the active domain (or take from it) the interface ``boundary``.

        Ownership is exclusive by construction -- the plane has one flag -- so
        handing it to this domain is the same act as taking it from the
        neighbour.
        """
        upper = self.builder_domains[boundary.upper_index]
        if boundary.side == "below":
            # This domain is the upper one; it owns the plane unless the flag
            # hands it to the previous domain.
            upper.interface_from_previous = not bool(owned)
        else:
            # The neighbour above is the upper one; the flag hands the plane
            # down to this domain.
            upper.interface_from_previous = bool(owned)

    def domain_label(self, index: int) -> str:
        domain = self.builder_domains[index]
        if domain.name:
            return domain.name
        if domain.formula_mode == "high_entropy":
            return f"Domain {index + 1} (HE)"
        return f"Domain {index + 1} ({domain_composition(domain)})"

    # ------------------------------------------------------------------
    # Point defects
    # ------------------------------------------------------------------
    def defect_entry_count(self) -> int:
        return len(self.defect_entries)

    def remove_defect_entry(self, index: int) -> None:
        if not 0 <= index < len(self.defect_entries):
            return
        del self.defect_entries[index]

    def ensure_defect_entries(self) -> None:  # noqa: D401
        """Keep every entry's enums in range.

        Grid addresses are deliberately *not* clamped to the current supercell:
        an entry that falls outside it is skipped by the resolver with a warning
        and comes back untouched when the cell grows again. Clamping would
        quietly rewrite the user's coordinates on a transient shrink.
        """
        for entry in self.defect_entries:
            entry.site = coerce_site_key(entry.site)
            entry.kind = min(max(int(entry.kind), 0), len(DEFECT_KIND_KEYS) - 1)
            entry.element = str(entry.element)
            entry.orientation = min(
                max(int(entry.orientation), 0), PROTON_ORIENTATION_COUNT - 1
            )

    def defect_grid_shape(self) -> Tuple[int, int, int]:
        """Octahedron grid the defects are addressed over."""
        return tuple(value + 1 for value in self.effective_oct_counts())

    def substitution_entry_for(self, key: SiteKey) -> DefectEntry | None:
        """The substitution (or blank-element vacancy) entry on ``key``, if any."""
        for entry in self.defect_entries:
            if entry.site == key and entry.kind_key() == "substitution":
                return entry
        return None

    def proton_entry_for(self, key: SiteKey) -> DefectEntry | None:
        """The proton entry riding on the oxygen ``key``, if any."""
        for entry in self.defect_entries:
            if entry.site == key and entry.kind_key() == "proton":
                return entry
        return None

    def defect_for_entry(self, entry: DefectEntry) -> SiteDefect | None:
        """One entry as a ``SiteDefect``, or None if it cannot be built.

        A substitution whose element has not been typed yet is built as a
        *vacancy*: emptying the element box is how a site is emptied, and an
        atom that vanishes is the clearest possible sign the edit landed.
        """
        kind = entry.kind_key()
        element = entry.element.strip()
        if kind == "substitution" and not element:
            kind, element = "vacancy", ""
        try:
            return SiteDefect(
                kind=kind,
                site=entry.site,
                element=element,
                orientation=entry.orientation,
            )
        except ValueError:
            return None

    def builder_defects(self) -> List[SiteDefect]:
        """Every entry as the defect list the builder applies."""
        defects: List[SiteDefect] = []
        for entry in self.defect_entries:
            defect = self.defect_for_entry(entry)
            if defect is not None:
                defects.append(defect)
        return defects

    def add_compensating_protons(self, count: int) -> None:
        """Add ``count`` protons on oxygens next to the aliovalent defects.

        Hosts are the oxygens of the substituted site's own octahedron -- where a
        proton actually localizes, next to the charge it is compensating. They
        arrive as ordinary entries, so the user can retune an orientation or
        delete them afterwards.
        """
        grid_shape = self.defect_grid_shape()
        periodic = self.treat_as_periodic
        # An oxygen that already hosts a proton, or that is itself vacant, is not
        # available as a host.
        taken = {
            canonicalize_key(defect.site, grid_shape, periodic)
            for defect in self.builder_defects()
            if defect.kind in ("proton", "vacancy")
        }
        candidates: List[SiteKey] = []
        for defect in self.builder_defects():
            if defect.kind != "substitution":
                continue
            site = defect.site
            for vertex in range(6):
                resolved = canonicalize_key(
                    SiteKey("X", site.i, site.j, site.k, vertex), grid_shape, periodic
                )
                if resolved is not None and resolved not in taken:
                    candidates.append(resolved)
                    taken.add(resolved)
        hosts = candidates[: max(0, int(count))]
        for host in hosts:
            self.defect_entries.append(
                DefectEntry(
                    site=host,
                    kind=DEFECT_KIND_KEYS.index("proton"),
                    element="H",
                )
            )

    def defects_signature(self) -> Tuple[object, ...]:
        return tuple(defect.signature() for defect in self.builder_defects())

    def set_defect_rows(self, defects: Sequence[SiteDefect]) -> None:
        """Unpack a stored defect list back into panel entries."""
        self.defect_entries = []
        for defect in defects:
            # The panel has no vacancy kind; a stored one is a substitution whose
            # element box is empty, which is what produced it in the first place.
            kind = "substitution" if defect.kind == "vacancy" else defect.kind
            element = "" if defect.kind == "vacancy" else defect.element
            self.defect_entries.append(
                DefectEntry(
                    site=defect.site,
                    kind=DEFECT_KIND_KEYS.index(kind),
                    element=element,
                    orientation=defect.orientation,
                )
            )

    # ------------------------------------------------------------------
    # Atom table, selection and per-atom edits
    # ------------------------------------------------------------------
    def atom_edit_mode(self) -> str:
        """How the focus is edited atom by atom: "builder", "loaded" or "none".

        A structure the builder can still drive is edited through its defect
        list and regenerated; anything else with atoms -- a file, a relaxation --
        is edited in place. The distinction is the builder's, not the panel's:
        the rows look and behave the same either way.
        """
        if self.focus is None:
            return "none"
        if self.focus_has_generated_provenance():
            return "builder"
        return "loaded"

    def _loaded_vacancy_list(self, structure: ChemicalStructure) -> List[VacatedAtom]:
        return self._loaded_vacancies.setdefault(id(structure), [])

    def atom_table(self) -> List[AtomRow]:
        """Every site of the focus, occupied or vacated, one row each.

        Cached on the structure's content and its defects: a builder edit
        regenerates the focus, which is what moves the rows, and nothing else
        does. Rows carry a stable ``ref`` -- the grid key for a built site, the
        position for a loaded atom -- so a selection survives the renumbering
        that removing an atom causes.
        """
        focus = self.focus
        if focus is None:
            return []
        mode = self.atom_edit_mode()
        vacancies = self._loaded_vacancy_list(focus) if mode == "loaded" else []
        relabels = self._loaded_relabels.get(id(focus), {}) if mode == "loaded" else {}
        return self._cached(
            "atom_table",
            (
                mode,
                self._structure_signature(focus),
                tuple(focus.atomic_labels),
                len(vacancies),
                tuple(
                    tuple(np.round(vacancy.cartesian, 6)) for vacancy in vacancies
                ),
                tuple(sorted(relabels.items())),
            ),
            lambda: (
                self._build_builder_atom_table(focus)
                if mode == "builder"
                else self._build_loaded_atom_table(focus, vacancies, relabels)
            ),
        )

    def _build_builder_atom_table(self, focus: ChemicalStructure) -> List[AtomRow]:
        params = getattr(focus, "generation_parameters", None)
        if params is None:
            return self._build_loaded_atom_table(focus, [], {})
        try:
            build = build_from_generation_parameters(params)
            resolution = resolve_defects(
                build,
                periodic=params.periodic_axes,
                stored_periodic=params.defect_reference_periodic(),
                defects=list(params.defects),
            )
            ideal_labels = formula_atomic_labels_from_parameters(build, params)
        except (ValueError, KeyError):
            return self._build_loaded_atom_table(focus, [], {})
        grid_shape = build.octahedra.shape
        keys = canonical_site_keys(grid_shape, params.periodic_axes)
        mapping = resolution.canonical_to_structure
        origin = np.asarray(params.cell_origin, dtype=np.float64)
        ideal = np.asarray(build.all_sites, dtype=np.float64) - origin
        symbols = focus.element_symbols()
        rows: List[AtomRow] = []
        for canonical, key in enumerate(keys):
            if canonical >= len(mapping):
                break
            index = int(mapping[canonical])
            if index >= len(symbols):
                index = -1
            rows.append(
                AtomRow(
                    ref=("site",) + tuple(key),
                    index=index,
                    element=symbols[index] if index >= 0 else "",
                    ideal_element=(
                        ideal_labels[canonical] if canonical < len(ideal_labels) else ""
                    ),
                    cartesian=ideal[canonical].copy(),
                    key=key,
                    role=key.role,
                )
            )
        # Protons are the trailing block of the emitted structure, one per host
        # in the order the hosts were recorded.
        first_proton = int(len(resolution.kept_canonical))
        coords = np.asarray(focus.cartesian_coords, dtype=np.float64)
        for offset, host in enumerate(resolution.proton_host_canonical):
            index = first_proton + offset
            if not 0 <= index < len(symbols) or not 0 <= int(host) < len(keys):
                continue
            host_key = keys[int(host)]
            rows.append(
                AtomRow(
                    ref=("proton",) + tuple(host_key),
                    index=index,
                    element=symbols[index],
                    ideal_element=symbols[index],
                    cartesian=coords[index].copy(),
                    key=None,
                    host_key=host_key,
                    role="H",
                )
            )
        return rows

    @staticmethod
    def _position_ref(cartesian: np.ndarray) -> Tuple[object, ...]:
        return ("pos",) + tuple(
            float(value) for value in np.round(np.asarray(cartesian, dtype=np.float64), 3)
        )

    def _build_loaded_atom_table(
        self,
        focus: ChemicalStructure,
        vacancies: Sequence[VacatedAtom],
        relabels: Dict[Tuple[object, ...], str],
    ) -> List[AtomRow]:
        symbols = focus.element_symbols()
        coords = np.asarray(focus.cartesian_coords, dtype=np.float64).reshape(-1, 3)
        rows = []
        for index in range(len(symbols)):
            ref = self._position_ref(coords[index])
            rows.append(
                AtomRow(
                    ref=ref,
                    index=index,
                    element=symbols[index],
                    ideal_element=relabels.get(ref, symbols[index]),
                    cartesian=coords[index].copy(),
                )
            )
        for vacancy in vacancies:
            rows.append(
                AtomRow(
                    ref=self._position_ref(vacancy.cartesian),
                    index=-1,
                    element="",
                    ideal_element=str(vacancy.label),
                    cartesian=np.asarray(vacancy.cartesian, dtype=np.float64).copy(),
                )
            )
        return rows

    def vacancy_markers(
        self, structure: ChemicalStructure
    ) -> Tuple[np.ndarray, List[str]]:
        """Positions of the vacated sites of ``structure``, and what each is missing.

        A built structure reads them off the ideal build (every boundary image
        included); a loaded one off the record of what was removed from it.
        """
        params = getattr(structure, "generation_parameters", None)
        if params is not None and getattr(structure, "geometry_matches_generation", True):
            return vacancy_render_sites(structure)
        vacancies = self._loaded_vacancies.get(id(structure), [])
        if not vacancies:
            return np.zeros((0, 3), dtype=np.float64), []
        return (
            np.asarray([vacancy.cartesian for vacancy in vacancies], dtype=np.float64),
            [str(vacancy.label) for vacancy in vacancies],
        )

    def atom_row_for_index(self, index: int) -> AtomRow | None:
        """The row of the focus atom ``index``, or None."""
        if index < 0:
            return None
        for row in self.atom_table():
            if row.index == index:
                return row
        return None

    def atom_row_for_ref(self, ref) -> AtomRow | None:
        for row in self.atom_table():
            if row.ref == ref:
                return row
        return None

    # -- selection -----------------------------------------------------
    def slab_rows(self) -> List[AtomRow]:
        """Rows whose position lies inside the selection slab, when it is on."""
        if not self.slab_enabled or self.focus is None:
            return []
        rows = self.atom_table()
        if not rows:
            return []
        coords = np.asarray([row.cartesian for row in rows], dtype=np.float64)
        inside = atoms_in_slab(coords, self.focus.lattice, self.selection_slab)
        return [rows[position] for position in inside]

    def active_rows(self) -> List[AtomRow]:
        """Everything the selection holds: the slab's atoms plus the hand picks.

        In table order rather than click order, so the list reads like the
        structure rather than like a history. This is what the 3D view fades
        to, and what the panel's "Active" list is drawn from.
        """
        picked = set(self.atom_selection)
        if self.slab_enabled:
            picked |= {row.ref for row in self.slab_rows()}
        return [row for row in self.atom_table() if row.ref in picked]

    def picked_rows(self) -> List[AtomRow]:
        """The hand-picked atoms alone, which the panel floats above the rest.

        Clicking an atom the slab already holds does not change what is active
        -- it was in the selection either way -- but it does pull the atom out
        into this list, where it stays while the slab moves on. That is how a
        handful of atoms is kept together long enough to edit them as a group.
        Table order again, for the same reason as ``active_rows``.
        """
        if not self.atom_selection:
            return []
        picked = set(self.atom_selection)
        return [row for row in self.atom_table() if row.ref in picked]

    def selection_active(self) -> bool:
        return bool(self.atom_selection) or bool(self.slab_enabled)

    def active_refs(self) -> set:
        return {row.ref for row in self.active_rows()}

    def toggle_atom_selection(self, ref, *, additive: bool = False) -> None:
        """Click on an atom: toggle it, or with ``additive`` only ever add it."""
        if ref in self.atom_selection:
            if not additive:
                self.atom_selection.discard(ref)
        else:
            self.atom_selection.add(ref)

    def clear_picked_atoms(self) -> None:
        """Drop the hand picks, leaving the slab and its contents as they are."""
        self.atom_selection.clear()

    def reset_slab_selection(self) -> None:
        """Put the slab away: the whole cell is drawn, and pickable, again.

        The hand picks are untouched. They were picked out of the cell, not out
        of the slab, and there is nowhere else to keep them.
        """
        self.slab_enabled = False

    def prune_atom_selection(self) -> None:
        """Drop refs that no longer name a row (a resized supercell, say)."""
        if not self.atom_selection:
            return
        live = {row.ref for row in self.atom_table()}
        self.atom_selection &= live

    # -- edits -----------------------------------------------------------
    def set_atom_element(self, row: AtomRow, element: str) -> None:
        """Make the site of ``row`` hold ``element``; an empty symbol empties it.

        The one operation behind the element box, the delete button and the
        restore button: what the site ends up holding is all that is said, and
        this works out whether that is a substitution, a vacancy, or the ideal
        lattice back again.
        """
        element = str(element).strip()
        mode = self.atom_edit_mode()
        if mode == "builder":
            self._set_builder_atom_element(row, element)
        elif mode == "loaded":
            self._set_loaded_atom_element(row, element)

    def _set_builder_atom_element(self, row: AtomRow, element: str) -> None:
        if row.role == "H" and row.host_key is not None:
            # A proton is an interstitial: it can be taken away, not replaced.
            if not element:
                entry = self.proton_entry_for(row.host_key)
                if entry is not None:
                    self.defect_entries.remove(entry)
            elif element != "H":
                self.atom_edit_message = "A proton can be removed but not substituted."
            return
        if row.key is None:
            return
        entry = self.substitution_entry_for(row.key)
        if element == row.ideal_element:
            if entry is not None:
                self.defect_entries.remove(entry)
            return
        if entry is None:
            self.defect_entries.append(
                DefectEntry(site=row.key, kind=DEFECT_KIND_KEYS.index("substitution"))
            )
            entry = self.defect_entries[-1]
        entry.element = element

    def atom_count_edits_available(self) -> bool:
        """Whether atoms may be removed from or added to the focus.

        Not on a relaxed builder structure: its generation parameters still
        describe its site numbering -- the spin features index by it -- and a
        removal would silently break that. Relabelling keeps the numbering.
        """
        return self.focus is not None and not self.focus_is_relaxed_from_builder()

    def _set_loaded_atom_element(self, row: AtomRow, element: str) -> None:
        focus = self.focus
        if focus is None:
            return
        if (row.index < 0 or not element) and not self.atom_count_edits_available():
            self.atom_edit_message = (
                "A relaxed structure keeps its atom count; relabel instead."
            )
            return
        vacancies = self._loaded_vacancy_list(focus)
        if row.index < 0:
            # A vacancy row: restoring puts the atom back where it was.
            if not element:
                return
            match = next(
                (
                    vacancy
                    for vacancy in vacancies
                    if self._position_ref(vacancy.cartesian) == row.ref
                ),
                None,
            )
            if match is None:
                return
            vacancies.remove(match)
            append_atom(focus, element, match.cartesian, match.magnetic_moment)
            if element != str(match.label):
                # Filled with something else: a substitution of what was there.
                self._loaded_relabels.setdefault(id(focus), {})[row.ref] = str(
                    match.label
                )
            self.after_loaded_atom_edit(focus, count_changed=True)
            return
        if not 0 <= row.index < focus.atom_count:
            return
        if not element:
            removed = remove_atom(focus, row.index)
            # The hole remembers what the file had there, not the relabel.
            removed.label = row.ideal_element or removed.label
            self._loaded_relabels.get(id(focus), {}).pop(row.ref, None)
            vacancies.append(removed)
            self.after_loaded_atom_edit(focus, count_changed=True)
            return
        if element == focus.element_symbols()[row.index]:
            return
        relabels = self._loaded_relabels.setdefault(id(focus), {})
        original = relabels.setdefault(row.ref, row.element)
        if element == original:
            del relabels[row.ref]
        substitute_atom(focus, row.index, element)
        self.after_loaded_atom_edit(focus, count_changed=False)

    def add_proton_to_atom(self, row: AtomRow) -> None:
        """Attach a proton to the oxygen of ``row``."""
        if row.index < 0 or row.element != "O":
            self.atom_edit_message = "A proton attaches to an oxygen."
            return
        mode = self.atom_edit_mode()
        if mode == "builder":
            if row.key is None or row.key.role != "X":
                self.atom_edit_message = "A proton attaches to an oxygen (X) site."
                return
            if self.proton_entry_for(row.key) is not None:
                self.atom_edit_message = "That oxygen already carries a proton."
                return
            self.defect_entries.append(
                DefectEntry(
                    site=row.key, kind=DEFECT_KIND_KEYS.index("proton"), element="H"
                )
            )
        elif mode == "loaded" and self.focus is not None:
            if not self.atom_count_edits_available():
                self.atom_edit_message = "A relaxed structure keeps its atom count."
                return
            add_proton(self.focus, row.index)
            self.after_loaded_atom_edit(self.focus, count_changed=True)

    def after_loaded_atom_edit(
        self, structure: ChemicalStructure, *, count_changed: bool
    ) -> None:
        """What an in-place edit to a loaded structure's atoms costs.

        Relabelling changes the chemistry every result was computed for, so the
        solver output and the oxidation assignment go; the saved configurations
        are moments on sites that still exist, so they stay and are re-energized.
        Removing or adding an atom renumbers the sites as well, which is what
        ``tile_focus`` already deals with, and is dealt with the same way here.
        """
        self.atom_edit_message = ""
        if count_changed:
            structure.spin_configurations.clear()
            self.oxidation_overrides.drop_atom_scope()
            self.oxidation_override_generation += 1
            self.active_saved_spin_index = -1
            self.selected_site_index = -1
        if self.magnetic_result_structure is structure:
            self.clear_solver_results()
        # Through the same gate as a builder edit, so a paused session does not
        # pay for the exchange rebuild on every keystroke.
        if self.interactive_updates_live():
            self.prepare_spin_baseline(structure)
            if not count_changed:
                self.re_energize_saved_configurations(structure)
            self.spin_energies_stale = False
        else:
            self.spin_energies_stale = True

    def load_generation_parameters_into_builder(
        self, params: PerovskiteGenerationParameters
    ) -> None:
        self.builder_domains = [
            DomainSpec(**domain.to_dict()) for domain in params.domain_specs()
        ]
        self.stacking_axis = int(getattr(params, "stacking_axis", 2))
        self.active_domain_index = 0
        self.domain_message = ""
        self.load_domain_into_buffer(self.builder_domains[0])
        if params.tilt_system not in GLAZER_TILT_SYSTEMS:
            # A loaded structure can carry any axis orientation of a tilt
            # system (a-a-c+ for a+b-b- turned round). The combo lists the 23
            # canonical spellings; an extra one is appended so the structure's
            # own system stays selectable rather than silently replaced.
            try:
                parse_glazer_tilt_system(params.tilt_system)
            except ValueError:
                pass
            else:
                GLAZER_TILT_SYSTEMS.append(str(params.tilt_system))
        if params.tilt_system in GLAZER_TILT_SYSTEMS:
            self.perovskite_tilt_system = GLAZER_TILT_SYSTEMS.index(params.tilt_system)
        self.tilt_angle_x = float(params.tilt_angle_x_deg)
        self.tilt_angle_y = float(params.tilt_angle_y_deg)
        self.tilt_angle_z = float(params.tilt_angle_z_deg)
        self.set_defect_rows(list(getattr(params, "defects", [])))
        self.perovskite_center = np.asarray(params.center, dtype=np.float32).copy()
        self.treat_as_periodic = params.periodic_axes

    def sync_builder_binding(self) -> None:
        """Bind the builder fields to a focused generated structure (idempotent).

        Gated on ``focus_has_generated_provenance`` rather than on the parameters
        alone: binding to a relaxed structure would load the pre-relaxation
        numbers into the builder fields, where they would misreport the geometry
        on screen and regenerate over it at the first edit.
        """
        focus = self.focus
        if focus is not None and self.focus_has_generated_provenance():
            if self._builder_bound_id != id(focus):
                self.load_generation_parameters_into_builder(focus.generation_parameters)
                self._builder_bound_id = id(focus)
                remembered = self._structure_domain_choice.get(id(focus), 0)
                if 0 < remembered < len(self.builder_domains):
                    self.select_domain(remembered)
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
            or not self.focus_has_generated_provenance()
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
        if regenerated.atom_count != focus.atom_count:
            # Resizing the supercell, or adding a vacancy, renumbers everything. The
            # per-atom oxidation edits were statements about indices in the old
            # numbering; the propagating ones are geometric and are the ones meant to
            # survive exactly this.
            self.oxidation_overrides.drop_atom_scope()
            self.oxidation_override_generation += 1
            # The site picked for the exchange plot was an index in the old
            # numbering too; left alone it would ring whatever slid into its slot.
            self.selected_site_index = -1
            self.oxidation_edit_site = -1
        focus.lattice = regenerated.lattice
        focus.cartesian_coords = regenerated.cartesian_coords
        focus.atomic_labels = regenerated.atomic_labels
        focus.magnetic_moments = regenerated.magnetic_moments
        focus.generation_parameters = regenerated.generation_parameters
        focus.is_periodic = regenerated.is_periodic
        focus.periodic_axes = regenerated.periodic_axes
        focus.spin_configurations.clear()
        self.active_saved_spin_index = -1
        if self.magnetic_result_structure is focus:
            # Solver output belongs to the old geometry; the landscape's
            # configurations survive and get re-energized by the baseline below.
            self.clear_solver_results()
        self._builder_applied_sig = signature
        if self.interactive_updates_live():
            self.prepare_spin_baseline(focus)
            self.spin_energies_stale = False
        else:
            # The landscape holds its last energies and the results panel says so.
            # Nothing downstream indexes past a length check -- oxidation_site_rows
            # clamps, structure_site_oxidation_states rejects a length mismatch, and
            # moments_as_vectors pads -- so an edit that changes the atom count is
            # safe to leave stale until the user asks for a refresh.
            self.spin_energies_stale = True

    # ------------------------------------------------------------------
    # Cell editing (structures with no builder provenance)
    # ------------------------------------------------------------------
    def load_cell_parameters_into_editor(self, structure: ChemicalStructure) -> None:
        """Seed the cell editor's fields from a structure's actual lattice."""
        try:
            a, b, c, alpha, beta, gamma = cell_parameters(structure.lattice)
        except ValueError:
            return
        self.cell_a, self.cell_b, self.cell_c = a, b, c
        self.cell_alpha, self.cell_beta, self.cell_gamma = alpha, beta, gamma
        self._cell_aspect_ratio = (b / a, c / a) if a > 0.0 else (1.0, 1.0)
        self.cell_tile_x = 1
        self.cell_tile_y = 1
        self.cell_tile_z = 1
        self.cell_message = ""

    def sync_cell_binding(self) -> None:
        """Bind the cell fields to a focused loaded structure (idempotent)."""
        focus = self.focus
        if focus is not None and self.cell_editing_available():
            if self._cell_bound_structure is not focus:
                self.load_cell_parameters_into_editor(focus)
                self._cell_bound_structure = focus
                # Baselined on the next change-check, after the widgets and
                # constraints have run, so binding is never read as an edit.
                self._cell_applied_sig = None
        else:
            self._cell_bound_structure = None
            self._cell_applied_sig = None

    def cell_fields_signature(self) -> Tuple[object, ...]:
        return (
            round(self.cell_a, 6),
            round(self.cell_b, 6),
            round(self.cell_c, 6),
            round(self.cell_alpha, 6),
            round(self.cell_beta, 6),
            round(self.cell_gamma, 6),
        )

    def apply_cell_constraints(self) -> None:
        """Clamp the cell fields, and enforce the aspect lock.

        The lock drives b and c from a rather than the other way round, which is
        why their inputs are greyed while it is on -- the same shape the builder's
        cubic/tetragonal linking already has.
        """
        self.cell_a = clamp_min(self.cell_a, MIN_CELL_LENGTH)
        self.cell_b = clamp_min(self.cell_b, MIN_CELL_LENGTH)
        self.cell_c = clamp_min(self.cell_c, MIN_CELL_LENGTH)
        for name in ("cell_alpha", "cell_beta", "cell_gamma"):
            value = float(getattr(self, name))
            setattr(self, name, min(max(value, MIN_CELL_ANGLE), MAX_CELL_ANGLE))
        if self.cell_lock_aspect:
            ratio_b, ratio_c = self._cell_aspect_ratio
            self.cell_b = clamp_min(self.cell_a * ratio_b, MIN_CELL_LENGTH)
            self.cell_c = clamp_min(self.cell_a * ratio_c, MIN_CELL_LENGTH)
        self.cell_tile_x = min(max(int(self.cell_tile_x), 1), 12)
        self.cell_tile_y = min(max(int(self.cell_tile_y), 1), 12)
        self.cell_tile_z = min(max(int(self.cell_tile_z), 1), 12)

    def capture_cell_aspect_ratio(self) -> None:
        """Freeze the current b:a and c:a, so locking preserves the shape on screen."""
        if self.cell_a > 0.0:
            self._cell_aspect_ratio = (
                self.cell_b / self.cell_a,
                self.cell_c / self.cell_a,
            )

    def apply_cell_edits_if_changed(self) -> None:
        """Strain the focused loaded structure to match the cell fields.

        The counterpart of ``regenerate_focus_from_builder_if_changed``, and the
        one place the two models differ in what they invalidate. A regeneration
        rebuilds the atom list, so everything indexed by site has to go; a strain
        moves the atoms it already has, keeping their count, order and labels, so
        saved spin configurations stay valid and only the energies -- which depend
        on distance through the exchange couplings -- go stale.
        """
        focus = self.focus
        if focus is None or not self.cell_editing_available():
            return
        if self._cell_bound_structure is not focus:
            return
        signature = self.cell_fields_signature()
        if self._cell_applied_sig is None:
            self._cell_applied_sig = signature  # baseline, no strain
            return
        if signature == self._cell_applied_sig:
            return
        try:
            new_lattice = lattice_from_parameters(
                focus.lattice,
                self.cell_a,
                self.cell_b,
                self.cell_c,
                self.cell_alpha,
                self.cell_beta,
                self.cell_gamma,
            )
            strain_structure(focus, new_lattice)
        except (ValueError, np.linalg.LinAlgError) as exc:
            # Leave the fields where the user put them and say why nothing moved.
            # An angle triple passes through impossible combinations on the way to
            # a possible one, so snapping them back mid-edit would fight the user.
            self.cell_message = str(exc)
            self._cell_applied_sig = signature
            return
        self.cell_message = ""
        self._cell_applied_sig = signature
        self.invalidate_after_geometry_change(focus)

    def invalidate_after_geometry_change(self, structure: ChemicalStructure) -> None:
        """What a change to a structure's *geometry* costs, as opposed to its atoms.

        Straining a cell, or switching it between periodic and cluster, moves the
        distances the exchange couplings are built from, so J and every energy
        derived from it are wrong. The atom list is untouched, though, so anything
        indexed by site is still meaningful -- which is why the saved
        configurations are re-energized here rather than thrown away, the one place
        this differs from a builder regeneration.
        """
        if self.magnetic_result_structure is structure:
            self.clear_solver_results()
        if self.interactive_updates_live():
            self.prepare_spin_baseline(structure)
            self.re_energize_saved_configurations(structure)
            self.spin_energies_stale = False
        else:
            self.spin_energies_stale = True

    def re_energize_saved_configurations(self, structure: ChemicalStructure) -> None:
        """Recompute saved configurations' energies against the current J matrix.

        Their moments survive a geometry change; the energies recorded beside them
        do not, and the Active Structure tree prints those verbatim. A
        configuration whose length no longer matches the magnetic sublattice
        belongs to a different cell and is left alone rather than guessed at.
        """
        indices = np.asarray(self.magnetic_site_indices, dtype=int)
        if indices.size == 0 or self.magnetic_j_matrix.size == 0:
            return
        for index, config in enumerate(structure.spin_configurations):
            moments = np.asarray(config.magnetic_moments, dtype=np.float64)
            if moments.ndim != 2 or moments.shape[0] != structure.atom_count:
                continue
            try:
                energy = compute_config_energy(self.magnetic_j_matrix, moments[indices])
            except (ValueError, IndexError):
                continue
            structure.spin_configurations[index] = replace(config, energy=float(energy))

    def tile_focus(self) -> None:
        """Replicate the focused loaded structure into a supercell.

        Not a live edit like the strain above: this changes the atom count, so
        every site-indexed result has to be dropped, and that is not something to
        do on each keystroke in a spin box. It is a button.
        """
        focus = self.focus
        if focus is None or not self.cell_editing_available():
            return
        counts = (self.cell_tile_x, self.cell_tile_y, self.cell_tile_z)
        if all(count == 1 for count in counts):
            self.cell_message = "Tiling by 1x1x1 would change nothing."
            return
        before = focus.atom_count
        try:
            tile_structure(focus, counts)
        except (ValueError, np.linalg.LinAlgError) as exc:
            self.cell_message = str(exc)
            return
        # Atom indices no longer mean what they meant, so anything keyed by them
        # goes -- the opposite of the strain path above.
        focus.spin_configurations.clear()
        self.oxidation_overrides.drop_atom_scope()
        self.oxidation_override_generation += 1
        self.active_saved_spin_index = -1
        if self.magnetic_result_structure is focus:
            self.clear_solver_results()
        self.load_cell_parameters_into_editor(focus)
        self._cell_applied_sig = self.cell_fields_signature()
        # Through the same gate as a builder edit: clearing _baseline_structure
        # instead would force the exchange rebuild past the paused-updates check,
        # on exactly the operation that makes that rebuild expensive.
        if self.interactive_updates_live():
            self.prepare_spin_baseline(focus)
            self.spin_energies_stale = False
        else:
            self.spin_energies_stale = True
        self.cell_message = (
            f"Tiled {counts[0]}x{counts[1]}x{counts[2]}: "
            f"{before} -> {focus.atom_count} atoms."
        )

    # ------------------------------------------------------------------
    # Per-frame memoization
    # ------------------------------------------------------------------
    def _cached(self, slot: object, key: object, build: Any) -> Any:
        """``build()``, reused while ``key`` is unchanged.

        ``slot`` separates independent memos; a caller that asks about more than one
        structure per frame puts the structure's identity in it, so the two answers
        coexist instead of evicting each other. ``key`` is what decides staleness.
        """
        cache = self._render_cache
        entry = cache.get(slot)
        if entry is not None and entry[0] == key:
            cache.move_to_end(slot)
            return entry[1]
        value = build()
        cache[slot] = (key, value)
        cache.move_to_end(slot)
        while len(cache) > RENDER_CACHE_SLOT_LIMIT:
            cache.popitem(last=False)
        return value

    @staticmethod
    def _structure_signature(structure: ChemicalStructure) -> Tuple[object, ...]:
        """Content key for a structure, plus the identity of its provenance.

        The builder mutates the focused structure in place, so identity alone says
        nothing about whether it changed. Hashing the geometry costs microseconds
        and cannot miss an edit; the generation-parameters identity is folded in
        because a regeneration always installs a fresh parameters object.
        """
        return (
            id(structure),
            id(getattr(structure, "generation_parameters", None)),
            _array_signature(structure.lattice, structure.cartesian_coords),
            tuple(structure.atomic_labels),
        )

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
        job = self.remote_job_for_placeholder(structure)
        if job is not None:
            # The row stood for a running job; without it there is nowhere for
            # the result to go, so the job is stopped along with it.
            self.cancel_remote_job(job)
            self._remote_placeholders = {
                key: item
                for key, item in self._remote_placeholders.items()
                if item is not structure
            }
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
        self.magnetic_pair_couplings = []
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

    def predicted_oxidation_assignment(self) -> OxidationStateAssignment | None:
        """The model's assignment, before any hand edits.

        One assignment, not a choice between many: the lowest-energy one the
        enumeration produced. Everything that acts on oxidation states goes through
        ``selected_oxidation_assignment`` instead; this is only for saying what the
        model, on its own, thought.
        """
        if not self.magnetic_oxidation_assignments:
            self.selected_oxidation_assignment_index = 0
            self.selected_spin_config_index = 0
            return None
        self.selected_oxidation_assignment_index = 0
        return self.magnetic_oxidation_assignments[0]

    def selected_oxidation_assignment(self) -> OxidationStateAssignment | None:
        """The assignment everything downstream should use: model plus hand edits.

        Every consumer of oxidation states -- render radii, the ion descriptors the
        exchange matrix is built from, the hover tooltip, the export -- already goes
        through here, so applying the overrides at this one point is what puts an
        edit into all of them at once.

        Memoized because it is read many times a frame and rebuilding it walks every
        overridden site. The override dicts are mutated in place, so identity says
        nothing about whether they changed; ``oxidation_override_generation`` is what
        the memo watches.
        """
        assignment = self.predicted_oxidation_assignment()
        structure = self.magnetic_analysis_structure
        if assignment is None or structure is None or self.oxidation_overrides.is_empty():
            self.oxidation_solve_unbalanced = False
            return assignment
        return self._cached(
            "effective_oxidation_assignment",
            (
                id(assignment),
                int(self.oxidation_override_generation),
                id(structure),
                structure.atom_count,
            ),
            lambda: self._constrained_oxidation_assignment(assignment, structure),
        )

    def _constrained_oxidation_assignment(
        self, predicted: OxidationStateAssignment, structure: ChemicalStructure
    ) -> OxidationStateAssignment:
        """Re-solve the cell with the hand-set states held fixed.

        An edit is a constraint on the solve, not a patch on its answer: the
        edited atoms keep their states and the model re-balances the rest, so
        the cell stays at its target net charge and the energy shown is the
        model's energy for the cell as edited. Only when no charge-balanced
        completion exists are the states applied verbatim, and the panel says so.
        """
        constraints = self.resolved_oxidation_overrides()
        solved = self.solve_oxidation_assignments(
            structure,
            constraints=constraints,
            prefer_like=predicted.site_oxidation_states,
        )
        if solved:
            self.oxidation_solve_unbalanced = False
            return solved[0]
        self.oxidation_solve_unbalanced = True
        return assignment_with_overrides(
            predicted,
            structure,
            self.oxidation_overrides,
            self.structure_supercell_repeats(structure),
        )

    def solve_oxidation_assignments(
        self,
        structure: ChemicalStructure,
        *,
        constraints: Dict[int, int] | None = None,
        prefer_like: np.ndarray | None = None,
    ) -> List[OxidationStateAssignment]:
        """The model's lowest-energy site assignment for ``structure``, or [].

        One assignment: anything past the head of the ranking is a guess between
        distributions the energy model cannot actually tell apart, and it is now
        an explicit edit -- a constraint here -- rather than a row in a list.
        """
        ranked = enumerate_site_group_assignments(
            structure.element_symbols(),
            charge=int(self.magnetic_net_charge),
            max_mixing=2,
            constraints=constraints or None,
            top_k=1,
        )
        if not ranked:
            return []
        return expand_group_assignments_to_site_assignments(
            ranked, structure, max_assignments=1, prefer_like=prefer_like
        )

    # ------------------------------------------------------------------
    # Manual oxidation-state edits
    # ------------------------------------------------------------------
    def structure_supercell_repeats(
        self, structure: ChemicalStructure | None
    ) -> Tuple[int, int, int]:
        """Primitive cells ``structure`` spans per axis; (1, 1, 1) when unknown.

        Read off the builder provenance rather than off the panel's spin boxes, so
        it describes the structure in hand rather than whatever the builder is
        currently pointed at. A structure with no provenance -- anything loaded from
        a file -- is its own cell, which is the reading that makes an edit on a CIF
        propagate to the copies of a later tiling.
        """
        params = getattr(structure, "generation_parameters", None)
        if params is None:
            return (1, 1, 1)
        factor = max(1, formula_unit_factor(getattr(params, "formula_mode", "perovskite")))
        return tuple(  # type: ignore[return-value]
            max(1, (int(count) + 1) // factor)
            for count in (params.n_oct_x, params.n_oct_y, params.n_oct_z)
        )

    def oxidation_edits_propagate(self) -> bool:
        """Whether an edit made right now names a site of the repeating motif.

        True on a unit cell, where every site *is* part of the motif and singling
        one out is not a thing the structure can express; false on a supercell,
        where singling one out is the entire point.
        """
        return self.structure_supercell_repeats(self.magnetic_analysis_structure) == (
            1,
            1,
            1,
        )

    def site_oxidation_state(self, atom_index: int) -> int | None:
        """The oxidation state atom ``atom_index`` currently carries, or None."""
        assignment = self.selected_oxidation_assignment()
        if assignment is None:
            return None
        states = assignment.site_oxidation_states
        if not 0 <= int(atom_index) < len(states):
            return None
        return int(states[int(atom_index)])

    def resolved_oxidation_overrides(self) -> Dict[int, int]:
        """Atom index -> hand-set charge for the analysed structure.

        Memoized: resolving the propagating edits walks every atom against the
        reference cell, and the panel asks once a frame for the atom under the
        cursor. ``oxidation_override_generation`` is what invalidates it -- the
        override dicts are mutated in place, so their identity says nothing.
        """
        structure = self.magnetic_analysis_structure
        if structure is None or self.oxidation_overrides.is_empty():
            return {}
        return self._cached(
            "resolved_oxidation_overrides",
            (
                id(structure),
                structure.atom_count,
                int(self.oxidation_override_generation),
            ),
            lambda: resolve_oxidation_overrides(
                structure,
                self.oxidation_overrides,
                self.structure_supercell_repeats(structure),
            ),
        )

    def site_oxidation_is_edited(self, atom_index: int) -> bool:
        """Whether ``atom_index`` is carrying a hand-set state rather than the model's."""
        return int(atom_index) in self.resolved_oxidation_overrides()

    def set_site_oxidation_state(self, atom_index: int, charge: int) -> None:
        """Set one atom's oxidation state by hand and re-derive what depends on it."""
        structure = self.magnetic_analysis_structure
        if structure is None or not 0 <= int(atom_index) < structure.atom_count:
            return
        self._apply_oxidation_edit(
            lambda: self.oxidation_overrides.set(
                structure,
                int(atom_index),
                min(
                    max(int(charge), MIN_EDITABLE_OXIDATION_STATE),
                    MAX_EDITABLE_OXIDATION_STATE,
                ),
                propagate=self.oxidation_edits_propagate(),
            )
        )

    def revert_site_oxidation_state(self, atom_index: int) -> None:
        """Hand ``atom_index`` back to the model."""
        structure = self.magnetic_analysis_structure
        if structure is None:
            return
        self._apply_oxidation_edit(
            lambda: self.oxidation_overrides.revert(
                structure, int(atom_index), propagate=self.oxidation_edits_propagate()
            )
        )

    def clear_oxidation_overrides(self) -> None:
        """Drop every hand-set state, in both scopes."""
        if self.oxidation_overrides.is_empty():
            return
        self._apply_oxidation_edit(self.oxidation_overrides.clear)

    def _apply_oxidation_edit(self, record: Any) -> None:
        """Record one edit and re-derive from it, reporting instead of dying.

        This is a text box wired to a rebuild of the exchange matrix and the whole
        spin landscape, over ion descriptors and Shannon tables that do not have an
        entry for every charge anyone can type. Something in that chain refusing a
        value is a normal outcome, and the panel says so; taking the app down with
        it would cost the session's structure, solved configurations and all.
        """
        self.oxidation_edit_message = ""
        try:
            record()
            self.after_oxidation_edit()
        except Exception as exc:  # keep the UI alive on any edit failure
            self.oxidation_edit_message = f"Could not apply that state: {exc}"

    def after_oxidation_edit(self) -> None:
        """Re-derive the exchange matrix and the landscape an edit invalidated.

        An oxidation state sets a site's d-shell, which sets its couplings, so an
        edit moves J and every energy computed against it -- exactly what a builder
        edit does, and it goes through the same gate: rebuilt now while the view can
        afford it, marked stale for the "Refresh energies" button when it cannot.
        That gate is a frame-rate one, so on a cell large enough that rebuilding
        costs the frame rate, editing stops rebuilding of its own accord and the
        panel keeps showing the last energies it had.
        """
        self.oxidation_override_generation += 1
        # Solved configurations were solved against the old J. The landscape's
        # *configurations* survive -- they are re-energized below -- but a cached
        # solve is an answer to a question that has changed.
        self.magnetic_solution_cache = {}
        focus = self.focus
        if focus is None:
            return
        if self.interactive_updates_live():
            self.prepare_spin_baseline(focus)
            self.spin_energies_stale = False
        else:
            self.spin_energies_stale = True

    def selected_spin_config(self) -> Any | None:
        configs = self.displayed_spin_configs()
        if not configs:
            self.selected_spin_config_index = 0
            return None
        self.selected_spin_config_index = min(
            max(self.selected_spin_config_index, 0),
            len(configs) - 1,
        )
        config = configs[self.selected_spin_config_index]
        # Recorded on the way out, so an edit later in the frame knows what to look
        # for once the landscape has been re-sorted.
        self._selected_spin_key = canonical_moment_key(config.all_moments)
        return config

    def restore_selected_spin_config(self) -> bool:
        """Point the selection back at the arrangement it was on, if it survived.

        Returns False when that arrangement is gone -- a replication change gives the
        configurations a different length, so there is nothing to hold on to.
        """
        key = self._selected_spin_key
        if key is None:
            return False
        for index, config in enumerate(self.spin_display_configs):
            if canonical_moment_key(config.all_moments) == key:
                self.selected_spin_config_index = index
                return True
        return False

    @staticmethod
    def _label_from_match(match: PatternMatch | None) -> str:
        if match is None or match.concentration > MAX_MATCH_DEFECT_CONCENTRATION:
            return "Other"
        return match.pattern.label

    @classmethod
    def _description_from_match(cls, match: PatternMatch | None) -> str:
        label = cls._label_from_match(match)
        if match is None or label == "Other" or match.is_exact:
            return label
        return f"{label}  {match.concentration * 100:.1f}% defects"

    def spin_classification_labels(self) -> List[str]:
        """Plot category per displayed config."""
        return [self._label_from_match(m) for m in self.displayed_pattern_matches()]

    def spin_classification_descriptions(self) -> List[str]:
        """Category plus defect concentration per displayed config, for the list."""
        return [self._description_from_match(m) for m in self.displayed_pattern_matches()]

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
        def resolve() -> PerovskiteSiteIndexing | None:
            params = getattr(structure, "generation_parameters", None)
            build = self.generated_build_for_structure(structure)
            if params is not None and build is not None:
                return site_indexing_from_generation_parameters(params, build)
            return recovered_site_indexing_from_magnetic_sites(structure)

        return self._cached(
            ("site_indexing", id(structure)),
            self._structure_signature(structure),
            resolve,
        )

    def b_grid_for_structure(self, structure: ChemicalStructure) -> np.ndarray | None:
        """Grid cell -> atom index for ``structure``, with -1 for vacated B sites.

        Derived from the site indexing rather than from ``build.b_site_indices``:
        the latter indexes the *ideal* build, which only coincides with structure
        positions when nothing has been removed.
        """
        indexing = self.resolved_site_indexing(structure)
        if (
            indexing is None
            or indexing.grid_to_site is None
            or indexing.b_grid_shape is None
        ):
            return None
        return np.asarray(indexing.grid_to_site, dtype=int).reshape(indexing.b_grid_shape)

    def compute_reference_configs(
        self,
        structure: ChemicalStructure,
        assignment: OxidationStateAssignment,
    ) -> list[tuple[str, SpinConfig]]:
        """The canonical orderings (G, C(a..c), F, A(a..c)) and their single-point energies.

        Orientations of any grid axis shorter than two cells are dropped by
        ``canonical_reference_patterns``, so a slab or a single-cell grid simply gets
        the subset it can actually distinguish.

        Any orderings entered by hand ride along on the same footing, which is what
        keeps them alive across re-energizations: like the canonical ones they are
        stored as a pattern and rescored, not as a frozen set of moments.
        """
        if self.magnetic_j_matrix.size == 0 or not self.magnetic_site_indices:
            return []
        site_indexing = self.resolved_site_indexing(structure)
        if site_indexing is None or site_indexing.b_site_indices.size == 0:
            return []
        try:
            configs = named_reference_spin_configs(
                structure,
                assignment,
                self.magnetic_j_matrix,
                self.magnetic_site_indices,
                site_indexing,
            )
        except Exception:
            return []
        if not self.custom_spin_patterns:
            return configs
        named = {label for label, _config in configs}
        try:
            custom = named_reference_spin_configs(
                structure,
                assignment,
                self.magnetic_j_matrix,
                self.magnetic_site_indices,
                site_indexing,
                patterns=[
                    label
                    for label in self.custom_spin_patterns
                    if label not in named
                ],
            )
        except Exception:
            return configs
        return configs + custom

    def interactive_updates_live(self, framerate: float | None = None) -> bool:
        """Whether a builder edit should re-energize the landscape right now.

        The re-energization is the expensive thing in the frame, so it pays for
        itself out of the frame rate and has to stop when the frame rate is what it
        is costing. Below ``AUTO_SPIN_UPDATE_MIN_FPS`` it pauses and edits go back
        to marking the energies stale; it resumes only once the view is comfortably
        clear again, at ``AUTO_SPIN_UPDATE_RESUME_FPS``.

        The two thresholds are what keep it from flapping. With one, pausing frees
        exactly the time that pushed the rate under it, the next frame clears the
        bar, and the landscape would rebuild every other frame -- worse than either
        state. ImGui's ``framerate`` is already smoothed over ~60 frames, so this
        responds over about a second rather than to one slow frame.
        """
        if not self.update_spin_energies_interactively:
            return False
        if framerate is None:
            framerate = current_framerate()
        # A framerate of zero is ImGui before it has measured anything; treat the
        # first frames as fast rather than pausing on no evidence.
        if framerate <= 0.0:
            return self._interactive_updates_live
        threshold = (
            AUTO_SPIN_UPDATE_MIN_FPS
            if self._interactive_updates_live
            else AUTO_SPIN_UPDATE_RESUME_FPS
        )
        self._interactive_updates_live = framerate >= threshold
        return self._interactive_updates_live

    def add_custom_spin_pattern(
        self,
        miller: Sequence[int],
        signs: str,
    ) -> bool:
        """Add a hand-entered ordering to the landscape. True when it took.

        The pattern is the same object the canonical orderings are: a plane family
        plus a sign string repeated across successive planes. Validation is against
        what the structure can actually express -- a period the cell has too few
        planes to resolve would silently fold back onto a shorter one and be scored
        as an ordering it is not.
        """
        signs = "".join(str(signs).split())
        if not signs or set(signs) - {"+", "-"}:
            self.custom_pattern_message = "The pattern must be non-empty '+' and '-'."
            return False
        if len(signs) > MAX_CUSTOM_PATTERN_PERIOD:
            self.custom_pattern_message = (
                f"Patterns are limited to {MAX_CUSTOM_PATTERN_PERIOD} planes."
            )
            return False
        try:
            pattern = PlanePattern(tuple(int(value) for value in miller), signs)
        except ValueError as exc:
            self.custom_pattern_message = str(exc)
            return False
        sublattice = self.magnetic_sublattice()
        if sublattice is None or sublattice.size == 0:
            self.custom_pattern_message = (
                "No magnetic sublattice to place an ordering on."
            )
            return False
        # (000) puts every site on one plane -- it is the family F lives on -- so it
        # holds exactly one sign however long the string is. Counting it as a single
        # plane refuses a longer pattern here for the same reason as anywhere else,
        # rather than silently folding it back onto F.
        available = (
            plane_count(sublattice.lattice_coords, pattern.miller)
            if any(pattern.miller)
            else 1
        )
        if available < pattern.period:
            self.custom_pattern_message = (
                f"{format_miller(pattern.miller)} spans {available} plane"
                f"{'' if available == 1 else 's'} here, too few for a "
                f"{pattern.period}-plane pattern."
            )
            return False

        label = pattern.plane_label
        if label in self.custom_spin_patterns:
            self.custom_pattern_message = f"{label} is already listed."
            return False
        # A canonical ordering re-entered by hand is the same ordering, and adding it
        # would put a second, identically-scored copy in the list. Named so, because
        # "already there" is more use when it says what it is already there as.
        for canonical in CANONICAL_PLANE_PATTERNS:
            if canonical.plane_label == label:
                self.custom_pattern_message = (
                    f"{label} is {canonical.label}, already in the landscape."
                    if canonical.name
                    else f"{label} is already in the landscape."
                )
                return False
        self.custom_spin_patterns.append(label)
        self.refresh_custom_spin_patterns()
        self.custom_pattern_message = self.select_custom_spin_pattern(label)
        return True

    def remove_custom_spin_pattern(self, label: str) -> None:
        if label not in self.custom_spin_patterns:
            return
        self.custom_spin_patterns.remove(label)
        self.refresh_custom_spin_patterns()
        self.custom_pattern_message = f"Removed {label}."

    def refresh_custom_spin_patterns(self) -> None:
        """Rescore the references so the custom orderings appear, or stop appearing.

        Cheap next to a solve: the references are one matrix product each, and the
        exchange matrix is untouched -- an ordering is a way of reading J, not a
        change to it.
        """
        structure = self.magnetic_analysis_structure
        assignment = self.selected_oxidation_assignment()
        if structure is None or assignment is None:
            return
        # Unit moments, as everywhere else that scores a reference: the spin
        # magnitude is already inside J, and handing over the formal moments scores
        # every custom ordering |mu|^2 too large -- 25x on an Fe(3+) cell.
        self.reference_configs = self.compute_reference_configs(
            structure, self.build_unit_moment_assignment(assignment)
        )
        self.refresh_landscape_energies()

    def select_custom_spin_pattern(self, label: str) -> str:
        """Land on the ordering ``label`` names, and say what became of it.

        A pattern the cell cannot tell apart from one already listed produces the
        very same moments and is folded into it by the deduplication, so nothing
        appears -- ``(123) +-`` and ``C(b)`` are one ordering on a 3x3x3 cubic cell.
        Reporting which ordering it turned out to be is the useful answer; letting
        the list look unchanged is not.
        """
        added = next(
            (config for name, config in self.reference_configs if name == label), None
        )
        if added is None:
            return f"{label} does not place on this structure."
        key = canonical_moment_key(added.all_moments)
        twin = next(
            (
                name
                for name, config in self.reference_configs
                if name != label and canonical_moment_key(config.all_moments) == key
            ),
            None,
        )
        for index, config in enumerate(self.displayed_spin_configs()):
            if canonical_moment_key(config.all_moments) == key:
                self.selected_spin_config_index = index
                break
        return (
            f"Added {label} -- the same ordering as {twin} on this cell."
            if twin is not None
            else f"Added {label}."
        )

    def magnetic_sublattice(self) -> MagneticSublattice | None:
        """The magnetic sublattice of the analysed structure, in solver site order.

        The plane patterns are defined against the sublattice's own lattice, and the
        configurations are indexed by ``magnetic_site_indices``, so the two have to be
        put in the same order before a configuration can be compared to a pattern.
        Returns ``None`` when the magnetic sites cannot be placed on a lattice.
        """
        structure = self.magnetic_analysis_structure
        if structure is None or not self.magnetic_site_indices:
            return None
        indexing = self.resolved_site_indexing(structure)
        sublattice = magnetic_sublattice_for(indexing)
        if sublattice is None or sublattice.size == 0:
            return None
        position_of = {int(site): row for row, site in enumerate(sublattice.site_indices)}
        rows = [position_of.get(int(site), -1) for site in self.magnetic_site_indices]
        if any(row < 0 for row in rows):
            # A magnetic site the sublattice does not place; patterns cannot describe
            # this configuration, so say so rather than silently dropping the site.
            return None
        return replace(
            sublattice,
            lattice_coords=sublattice.lattice_coords[rows],
            site_indices=sublattice.site_indices[rows],
        )

    def miller_plane_overlay(
        self, config: SpinConfig | None
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, float, float]] | None] | None:
        """``(cell-frame miller, offsets, per-sheet colours)`` for the plane overlay.

        Follows the selected configuration's matched pattern: the sheets sit on the
        planes the magnetic sites actually occupy and take the spin colour the
        pattern's sign string gives them, which is what makes an ordering legible as
        stacked sheets.
        """
        if config is None:
            return None
        return self._cached(
            "miller_plane_overlay",
            (self._landscape_generation, self.selected_spin_config_index, id(config)),
            lambda: self._build_miller_plane_overlay(config),
        )

    def _build_miller_plane_overlay(self, config: SpinConfig):
        structure = self.magnetic_analysis_structure
        match = self.match_for_config(config)
        sublattice = self.magnetic_sublattice()
        if match is None or sublattice is None or structure is None:
            return None
        indices = plane_indices(sublattice.lattice_coords, match.pattern.miller)
        if indices.size == 0:
            return None

        miller = sublattice.miller_in_cell(match.pattern.miller)
        # The offset has to be measured from the sites, not taken to be the plane
        # index. The magnetic sublattice does not generally share the cell's origin --
        # in a perovskite the B sites sit half a primitive cell off the corner, which
        # put every sheet exactly between two layers of atoms instead of through them.
        try:
            projection = structure.fractional_coords[sublattice.site_indices] @ miller
        except (IndexError, np.linalg.LinAlgError):
            return None
        distinct = np.unique(indices)
        offsets = np.array(
            [float(projection[indices == value].mean()) for value in distinct],
            dtype=np.float64,
        )
        signs = signs_from_ordinals(
            distinct.astype(int), match.pattern.signs, phase=match.phase
        )
        if match.flipped:
            signs = -signs
        colors = [SPIN_UP_COLOR if value > 0 else SPIN_DOWN_COLOR for value in signs]
        return miller, offsets, colors

    def effective_render_periodic_images(self) -> bool:
        """Whether to draw the closing boundary layer.

        The user's setting, and nothing else. Picking used to switch this off,
        because the boundary layer draws a corner site up to eight times and each
        copy answers to a click as if it were its own site. Ringing *every* copy
        of whatever is under the cursor says the same thing without taking the
        images away: the copies visibly move together, so they read as one site.
        """
        return bool(self.render_periodic_images)

    def structure_view_mode(self) -> str:
        """Which decoration the 3D view runs: "exchange" or "plain".

        The exchange view runs only while the exchange plot was the feature
        touched last *and* an atom's couplings are actually selected. Everything
        else is the plain structure, with the atom selection ringed over it.
        """
        if self.structure_view_focus == "exchange":
            return "exchange" if exchange_selection_site(self) >= 0 else "plain"
        return "plain"


    def spin_defect_site_indices(self, config: SpinConfig | None) -> list[int]:
        """Structure indices of the magnetic sites that disagree with the ideal."""
        if config is None:
            return []
        return self._cached(
            "spin_defect_sites",
            (self._landscape_generation, self.selected_spin_config_index, id(config)),
            lambda: self._build_spin_defect_site_indices(config),
        )

    def _build_spin_defect_site_indices(self, config: SpinConfig) -> list[int]:
        match = self.match_for_config(config)
        sublattice = self.magnetic_sublattice()
        if match is None or sublattice is None or match.is_exact:
            return []
        return [
            int(sublattice.site_indices[row])
            for row in np.flatnonzero(match.mismatched)
            if row < sublattice.site_indices.size
        ]

    def custom_plane_patterns(self) -> tuple[PlanePattern, ...]:
        """The hand-entered orderings as patterns, in the order they were added."""
        parsed = (parse_plane_label(label) for label in self.custom_spin_patterns)
        return tuple(pattern for pattern in parsed if pattern is not None)

    def match_pattern_candidates(
        self, lattice_coords: np.ndarray
    ) -> tuple[PlanePattern, ...]:
        """The orderings a configuration is scored against, custom ones included.

        A hand-entered ordering is a reference like any other once it is in the
        landscape, so it has to be in the set the classifier chooses from as well --
        otherwise the configuration the user just added is reported as the nearest
        canonical ordering, or as "Other", and the plot disagrees with the list it
        came from.

        Canonical patterns go first, so ``patterns_for_sites`` keeps the classical
        name when a custom ordering turns out to be one the cell cannot tell from a
        canonical one. That is the same rule ``select_custom_spin_pattern`` reports
        under, and the two would otherwise name the same state differently.
        """
        return patterns_for_sites(
            lattice_coords, CANONICAL_PLANE_PATTERNS + self.custom_plane_patterns()
        )

    def displayed_pattern_matches(self) -> List[PatternMatch | None]:
        """One match per displayed configuration, computed in a single pass.

        Every displayed configuration is scored against every pattern on every frame
        -- the list, the plot legend and the 3D badge all ask. Doing that one
        configuration at a time through a small LRU thrashed it completely on a large
        cell (25 rows competing for 16 slots, so every row missed every frame, at
        ~0.7 ms a row). One list, one cache entry, and the per-pattern plane indices
        shared across the whole landscape instead of rebuilt per row.
        """
        configs = self.displayed_spin_configs()
        sublattice = self.magnetic_sublattice()
        if sublattice is None or not configs:
            return [None] * len(configs)

        def compute() -> List[PatternMatch | None]:
            patterns = self.match_pattern_candidates(sublattice.lattice_coords)
            index = build_plane_index(sublattice.lattice_coords, patterns)
            return [
                best_matching_pattern(
                    np.asarray(config.all_moments, dtype=np.float64).reshape(-1),
                    sublattice.lattice_coords,
                    patterns,
                    plane_index=index,
                )
                for config in configs
            ]

        return self._cached(
            "displayed_pattern_matches",
            (
                self._landscape_generation,
                len(configs),
                sublattice.lattice_coords.tobytes(),
                tuple(self.custom_spin_patterns),
            ),
            compute,
        )

    def match_for_config(self, config: SpinConfig) -> PatternMatch | None:
        """The ideal ordering ``config`` is nearest to, and how far off it is.

        Displayed configurations are answered from the batch above; anything else --
        a configuration being saved, say -- is scored on its own.
        """
        configs = self.displayed_spin_configs()
        for position, candidate in enumerate(configs):
            if candidate is config:
                matches = self.displayed_pattern_matches()
                if position < len(matches):
                    return matches[position]
                break
        sublattice = self.magnetic_sublattice()
        if sublattice is None:
            return None
        return best_matching_pattern(
            np.asarray(config.all_moments, dtype=np.float64).reshape(-1),
            sublattice.lattice_coords,
            self.match_pattern_candidates(sublattice.lattice_coords),
        )

    def label_for_config(self, config: SpinConfig) -> str:
        """Plot/legend category for ``config``: a pattern name, or "Other".

        The bare category, with no defect concentration attached -- this is what the
        scatter colours and the legend key off, so it has to stay a small fixed set.
        A configuration further than ``MAX_MATCH_DEFECT_CONCENTRATION`` from every
        pattern is "Other": past that, "nearest" stops meaning anything.
        """
        return self._label_from_match(self.match_for_config(config))

    def described_config(self, config: SpinConfig) -> str:
        """``label_for_config`` plus the defect concentration when it is not exact."""
        return self._description_from_match(self.match_for_config(config))

    def refresh_landscape_energies(self, *, preserve_selection: bool = True) -> None:
        """Re-evaluate every retained configuration against the current J matrix.

        A single point per configuration -- no optimization. Configurations whose
        length no longer matches the magnetic-site count belong to a different cell
        (a replication change) and are dropped.

        ``preserve_selection`` keeps the selection on the same spin arrangement
        across the re-sort. A deliberate solve passes False, so a fresh solve still
        presents its ground state.
        """
        n_mag = len(self.magnetic_site_indices)
        if n_mag == 0 or self.magnetic_j_matrix.size == 0:
            self.spin_landscape = []
            self.spin_display_configs = []
            self._landscape_generation += 1
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

        reference_keys = {canonical_moment_key(c.all_moments) for c in references}
        merged = annotate_degeneracy(sort_and_rank(references + retained))

        def take(configs: List[SpinConfig], cap: int) -> List[SpinConfig]:
            """``cap`` configurations, with the references claiming their slots first."""
            refs = [
                config
                for config in configs
                if canonical_moment_key(config.all_moments) in reference_keys
            ]
            others = [
                config
                for config in configs
                if canonical_moment_key(config.all_moments) not in reference_keys
            ]
            return sort_and_rank(refs + others[: max(cap, len(refs)) - len(refs)])

        # The pool is kept deeper than the plot so that turning "Plot degenerate
        # configs" back on restores what collapsing hid, and it is bounded so that
        # re-energizing it on every builder edit stays cheap.
        self.spin_landscape = take(merged, SPIN_LANDSCAPE_POOL_LIMIT)
        displayed = (
            merged
            if self.plot_degenerate_configs
            else collapse_degenerate_configs(merged, reference_keys)
        )
        self.spin_display_configs = take(displayed, int(self.spin_plot_max_configs))
        self._landscape_generation += 1
        if preserve_selection and not self.restore_selected_spin_config():
            self.selected_spin_config_index = 0

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
            assignments = self.solve_oxidation_assignments(structure)
            if not assignments:
                self.reset_spin_landscape(NO_ASSIGNMENT_MESSAGE)
                return

            # The landscape is a result attached to this structure, so the output pane
            # and the save button treat it exactly like solver output.
            self.magnetic_analysis_structure = structure
            self.magnetic_result_structure = structure
            self.magnetic_result_structure_name = structure.name
            self.magnetic_oxidation_assignments = assignments
            self.selected_oxidation_assignment_index = 0
            assignment = self.selected_oxidation_assignment()
            if assignment is None:
                self.reset_spin_landscape(NO_ASSIGNMENT_MESSAGE)
                return
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
        self.spin_display_configs = []
        self._landscape_generation += 1
        self.magnetic_solution_cache = {}
        # An atom index names an atom of one structure only. The propagating edits
        # are keyed geometrically and survive -- they simply do not resolve onto a
        # structure that is not a supercell of the cell they were authored on.
        self.oxidation_overrides.drop_atom_scope()
        self.oxidation_override_generation += 1
        self.prepare_spin_baseline(focus)

    def refresh_spin_energies(self) -> None:
        """Recompute the baseline a builder edit left stale. Idempotent and cheap
        to call when nothing is stale, so the UI can call it unconditionally."""
        focus = self.focus
        if not self.spin_energies_stale or focus is None:
            return
        self.prepare_spin_baseline(focus)
        self.re_energize_saved_configurations(focus)
        self.spin_energies_stale = False

    def reset_spin_landscape(self, status: str = "") -> None:
        """Empty the landscape and everything derived from it.

        The assignments and J matrix go too: leaving the previous structure's values
        behind would show analysis for a structure that is no longer active. Leaves
        ``_baseline_structure`` alone -- callers that want the baseline recomputed
        clear it themselves.
        """
        self.spin_landscape = []
        self.spin_display_configs = []
        self._landscape_generation += 1
        self.reference_configs = []
        self.magnetic_oxidation_assignments = []
        self.selected_oxidation_assignment_index = 0
        self.selected_spin_config_index = 0
        self.magnetic_site_indices = []
        self.magnetic_j_matrix = np.zeros((0, 0), dtype=np.float64)
        self.magnetic_pair_couplings = []
        self.baseline_status = status

    def displayed_spin_configs(self) -> List[SpinConfig]:
        """Configurations shown in the plot, the results list, and the 3D view.

        Solver output is merged into the landscape, so this is the one accessor
        everything reads -- which is what makes reference points as interactive as
        solved ones.
        """
        if self.focus is None or self.magnetic_analysis_structure is not self.focus:
            return []
        return self.spin_display_configs

    def merge_solver_states_into_landscape(self, all_states: list[SpinConfig]) -> None:
        """Fold a completed solve into the persistent landscape pool.

        The selection is not carried over: a solve is an explicit request, and it
        should land on the ground state it just found.
        """
        self.spin_landscape = list(all_states)[:SPIN_LANDSCAPE_POOL_LIMIT]
        self.refresh_landscape_energies(preserve_selection=False)

    def save_selected_spin_configuration(self) -> None:
        config = self.selected_spin_config()
        structure = self.magnetic_result_structure
        if config is None or structure is None:
            self.spin_save_message = "Run Magnetic Structure and select a configuration first."
            return

        moments = self.expand_spin_moments_to_structure(config.all_moments, structure)

        # The solver runs on unit spins (magnitude is inside J), so the formal
        # per-site moments ride along on the saved configuration for export to
        # scale by -- an Fe(3+) site writes +-5.0 instead of +-1.0.
        assignment = self.selected_oxidation_assignment()
        magnitudes = (
            np.asarray(assignment.magnetic_moments, dtype=np.float64)
            if assignment is not None
            and len(assignment.magnetic_moments) == structure.atom_count
            else None
        )

        # Same exact label the plot uses, so a saved config is named A(c) only when it
        # really is A(c) rather than merely closest to it.
        classification = self.label_for_config(config)
        match = self.match_for_config(config)
        defect_fraction = 0.0 if match is None else float(match.concentration)

        # Read collinearity off the configuration itself: the landscape can outlive
        # the solve that produced it, so the solver's flag may no longer describe it.
        collinear = np.asarray(config.all_moments, dtype=np.float64).ndim == 1

        structure.spin_configurations.append(
            SavedSpinConfiguration(
                magnetic_moments=np.array(moments, dtype=np.float64, copy=True),
                energy=float(config.energy),
                magnetization=float(config.magnetization),
                classification=classification,
                defect_concentration=defect_fraction,
                collinear=collinear,
                site_moment_magnitudes=(
                    None if magnitudes is None else np.array(magnitudes, copy=True)
                ),
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

    def _export_via_browser(
        self,
        structures: Sequence[ChemicalStructure],
        description: str,
    ) -> None:
        """Export through a browser download instead of to a folder.

        The web build has no filesystem the user can reach, so the same export runs
        into a temporary directory and the bytes are handed to the page.
        """
        try:
            filename, payload, mime_type = export_bundle_bytes(structures)
        except Exception as exc:
            self.export_message = f"Export failed: {exc}"
            return
        if not _download_via_browser(filename, payload, mime_type):
            self.export_message = (
                "This page cannot start downloads. Reload it to pick up the current "
                "version of the app, then try again."
            )
            return
        self.export_message = f"Downloaded {description} as '{filename}'."

    def export_active_structure(self) -> None:
        structure = self.focus
        if IS_PYODIDE:
            if structure is None:
                self.export_message = "No active structure to export."
                return
            self._export_via_browser([structure], f"'{structure.name}'")
            return
        target = self.resolved_export_directory()
        if target is None:
            return
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
        if IS_PYODIDE:
            count = len(self.structures)
            self._export_via_browser(
                list(self.structures), f"{count} structure(s)"
            )
            return
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

    def turn_structure_view(self, degrees: float) -> None:
        """Turn the 3D view about the selected screen axis, from the +/- buttons.

        A swing towards a lattice axis still in flight is turned with it, so the press
        takes effect immediately and still ends up where the alignment was heading.
        """
        axis = SCREEN_TURN_AXES[self.screen_turn_axis_index % 3]
        self.structure_rotation = rotation_after_screen_turn(
            self.structure_rotation, axis, degrees
        )
        if self.structure_rotation_target is not None:
            self.structure_rotation_target = rotation_after_screen_turn(
                self.structure_rotation_target, axis, degrees
            )

    def unit_cell_count(self) -> int:
        """Perovskite cells the analysis structure spans, or 0 when it is not a grid.

        One B site is one cell, so the B-site grid's shape is the cell count -- which
        is what makes a per-cell moment comparable between a 2x2x2 and a 5x5x5 of the
        same material. A loaded structure whose magnetic sites do not form a grid has
        no cell to divide by and reports 0.
        """
        structure = self.magnetic_analysis_structure
        if structure is None:
            return 0
        indexing = self.resolved_site_indexing(structure)
        if indexing is None or indexing.b_grid_shape is None:
            return 0
        return int(np.prod(np.asarray(indexing.b_grid_shape, dtype=int)))

    def magnetization_basis(self) -> Tuple[np.ndarray | None, int]:
        """(formal |mu| per atom, cell count) for reporting a configuration's moment.

        The solver works in unit +-1 moments -- the size of a spin is inside J, not on
        the site -- so its own ``magnetization`` counts *sites*, not Bohr magnetons,
        and it grows with the supercell. Reporting a physical moment needs the formal
        high-spin magnitudes back off the oxidation assignment, the same source
        ``displayed_site_moment_magnitudes`` and export use, and a cell count to divide
        by. Read once per frame and passed down, rather than per listed row.
        """
        structure = self.magnetic_analysis_structure
        assignment = self.selected_oxidation_assignment()
        if structure is None or assignment is None:
            return None, 0
        magnitudes = np.abs(np.asarray(assignment.magnetic_moments, dtype=np.float64))
        if len(magnitudes) != structure.atom_count:
            return None, 0
        return magnitudes, self.unit_cell_count()

    def config_magnetization(
        self,
        config: Any,
        basis: Tuple[np.ndarray | None, int] | None = None,
    ) -> Tuple[float, str]:
        """``config``'s net moment and the unit it is in.

        ``mu_B/cell`` when both the high-spin magnitudes and a cell count are
        available, ``mu_B`` for the whole structure when only the magnitudes are, and
        the solver's own site count with an empty unit when neither is -- so the
        number on screen always says what it is rather than quietly changing meaning.
        """
        magnitudes, cells = self.magnetization_basis() if basis is None else basis
        structure = self.magnetic_analysis_structure
        if magnitudes is None or structure is None:
            return float(config.magnetization), ""
        vectors = self.expand_spin_moments_to_structure(config.all_moments, structure)
        signs = np.sign(np.asarray(vectors, dtype=np.float64)[:, 2])
        total = float(np.dot(signs, magnitudes))
        if cells <= 0:
            return total, "μB"
        return total / cells, "μB/cell"

    def displayed_site_moment_magnitudes(
        self,
        structure: ChemicalStructure,
    ) -> np.ndarray | None:
        """Formal high-spin moments, in mu_B, for the configuration on screen.

        The moments the solver hands back are unit-magnitude *directions* -- the size of
        a spin is already baked into the exchange couplings -- so how big each moment
        really is has to come from the oxidation-state assignment, exactly as export does
        in ``export_utils``. A saved configuration carries the assignment it was solved
        under; anything else is read against the assignment currently selected. None when
        no assignment is available, which leaves the spheres at their element radii.
        """
        saved = self.displayed_saved_spin_configuration()
        if saved is not None and saved.site_moment_magnitudes is not None:
            magnitudes = np.asarray(saved.site_moment_magnitudes, dtype=np.float64)
            if len(magnitudes) == structure.atom_count:
                return magnitudes
        assignment = self.selected_oxidation_assignment()
        if assignment is None:
            return None
        magnitudes = np.asarray(assignment.magnetic_moments, dtype=np.float64)
        if len(magnitudes) != structure.atom_count:
            return None
        return magnitudes

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
        source_b_grid = self.b_grid_for_structure(source_structure)
        target_b_grid = self.b_grid_for_structure(target_structure)
        if source_b_grid is None or target_b_grid is None:
            return None
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
            self.magnetic_pair_couplings = []
            self.magnetic_spin_status = "No structure is available for exchange-coupling analysis."
            return False

        try:
            descriptors = structure_ion_descriptors(structure, assignment)
            magnetic_sites = sorted(descriptors)
            bridges = build_bridges(structure, descriptors) if descriptors else []
            if not descriptors or not bridges:
                self.magnetic_j_matrix = np.zeros((0, 0), dtype=np.float64)
                self.magnetic_site_indices = []
                self.magnetic_pair_couplings = []
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
            # Kept in the model's own convention (J > 0 AFM), unlike the matrix
            # above: this is a readout, not solver input.
            self.magnetic_pair_couplings = pair_couplings(
                bridges, params, structure.cartesian_coords
            )
            self._exchange_generation += 1
        except Exception as exc:
            self.magnetic_j_matrix = np.zeros((0, 0), dtype=np.float64)
            self.magnetic_site_indices = []
            self.magnetic_pair_couplings = []
            self.magnetic_spin_status = f"Exchange-coupling assignment failed: {exc}"
            return False
        return True

    def run_selected_oxidation_assignment(self, *, force: bool = False) -> None:
        assignment = self.selected_oxidation_assignment()
        if assignment is None:
            self.magnetic_spin_status = "No oxidation states have been assigned."
            return

        cache_key = self.selected_oxidation_assignment_index
        if not force and cache_key in self.magnetic_solution_cache:
            # Rebuild J for the current states and restore the configurations solved
            # against them. The cache is emptied by anything that moves the states,
            # so what is in it always belongs to the assignment in hand.
            if not self.build_exchange_couplings_for_assignment(assignment):
                return
            structure = self.magnetic_analysis_structure
            if structure is not None:
                self.reference_configs = self.compute_reference_configs(
                    structure, self.build_unit_moment_assignment(assignment)
                )
            _, cached_states = self.magnetic_solution_cache[cache_key]
            self.merge_solver_states_into_landscape(cached_states)
            self.spin_energies_stale = False
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
                f"Spin solve failed for the current oxidation states: {exc}"
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
        self.spin_energies_stale = False
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
            assignments = self.solve_oxidation_assignments(structure)
            if not assignments:
                self.magnetic_oxidation_status = NO_ASSIGNMENT_MESSAGE
                self.magnetic_spin_status = "Spin solve skipped because no site-resolved assignments were produced."
                return

            self.magnetic_oxidation_assignments = assignments
            self.selected_oxidation_assignment_index = 0
            self.magnetic_oxidation_status = ""
            # Everything was re-enumerated from the current geometry just now.
            self.spin_energies_stale = False
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
        self.perovskite_supercell_x = max(1, self.perovskite_supercell_x)
        self.perovskite_supercell_y = max(1, self.perovskite_supercell_y)
        self.perovskite_supercell_z = max(1, self.perovskite_supercell_z)
        self.lattice_a = clamp_min(self.lattice_a, 2.0)
        self.lattice_b = clamp_min(self.lattice_b, 2.0)
        self.lattice_c = clamp_min(self.lattice_c, 2.0)
        self.ensure_high_entropy_rows()
        self.ensure_defect_entries()

        if self.perovskite_type == 0:
            self.lattice_b = self.lattice_a
            self.lattice_c = self.lattice_a
        elif self.perovskite_type == 1:
            self.lattice_b = self.lattice_a

        self.tilt_angle_x = min(max(self.tilt_angle_x, -45.0), 45.0)
        self.tilt_angle_y = min(max(self.tilt_angle_y, -45.0), 45.0)
        self.tilt_angle_z = min(max(self.tilt_angle_z, -45.0), 45.0)
        self.conform_stack_to_active_domain()

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

    def default_supercell_for_formula(self) -> tuple[int, int, int]:
        """Opening supercell for the current formula mode.

        The ordered modes already double the grid through their unit factor, so
        one primitive cell of those is what two of a plain perovskite is. Two of
        them is therefore a 4x4x4 octahedron grid, against 3x3x3 for the plain
        and high-entropy modes.
        """
        if self.formula_key() in ("double", "quadruple", "dq"):
            return (2, 2, 2)
        return (3, 3, 3)

    def apply_default_supercell_for_formula(self) -> None:
        (
            self.perovskite_supercell_x,
            self.perovskite_supercell_y,
            self.perovskite_supercell_z,
        ) = self.default_supercell_for_formula()

    def apply_default_composition_for_formula(self) -> None:
        if self.formula_key() != "dq":
            return
        self.a_site_element = "Ca"
        # Mg is fixed at +2, which pins the A' sublattice and leaves the oxidation
        # enumeration to the B sites: 674 charge-balanced distributions against the
        # 78,312 a mixed-valence Cu A' site produces.
        self.a2_site_element = "Mg"
        self.b_site_element = "Fe"
        self.b2_site_element = "Re"
        self.x_site_element = "O"

    def apply_defaults_for_formula(self) -> None:
        self.apply_default_supercell_for_formula()
        self.apply_default_composition_for_formula()

    def _formula_edit_signature(self) -> Tuple[object, ...]:
        """The builder state a formula change would discard.

        ``builder_fields_signature`` minus ``render_periodic_images``, which is
        a view option rather than an edit to the structure.
        """
        signature = self.builder_fields_signature()
        return (signature[0],) + signature[2:]

    def builder_has_edits(self) -> bool:
        """Whether the active structure carries work a formula change loses.

        Anything the user put there by hand: defects, saved spin
        configurations, or builder fields that differ from what this formula
        opens with. A structure still sitting at its defaults has nothing to
        lose, so changing its formula asks nothing.
        """
        focus = self.focus
        if focus is not None and getattr(focus, "spin_configurations", None):
            return True
        if self.defect_entries:
            return True
        reference = AppState()
        reference.formula_mode = int(self.formula_mode)
        reference.apply_defaults_for_formula()
        reference.apply_perovskite_constraints()
        return self._formula_edit_signature() != reference._formula_edit_signature()

    def apply_formula_change(self, mode: int) -> None:
        """Switch formula and open the default structure of the new type.

        The whole builder goes back to defaults, not just the composition: the
        answer to "this will lose all edits" has to actually lose them, or the
        cell that comes back is a mix of two formulas.
        """
        self.reset_builder_to_defaults()
        self.formula_mode = min(max(int(mode), 0), len(FORMULA_MODES) - 1)
        self.apply_defaults_for_formula()
        self.apply_perovskite_constraints()
        self._last_formula_mode = self.formula_mode
        self.pending_formula_mode = -1

    def create_structure_with_formula(self, mode: int) -> None:
        """Add a new structure of ``mode``, leaving the edited one alone."""
        self.create_new_structure()
        self.formula_mode = min(max(int(mode), 0), len(FORMULA_MODES) - 1)
        self.apply_defaults_for_formula()
        self.apply_perovskite_constraints()
        self._last_formula_mode = self.formula_mode
        self.pending_formula_mode = -1

    def formula_unit_factor(self) -> int:
        return formula_unit_factor(self.formula_key())

    def effective_n_oct(self, supercell: int) -> int:
        """Octahedron count along an axis for a supercell of ``supercell`` cells.

        ``supercell`` counts primitive cells, so 1 is the primitive cell itself.
        The ordered formula modes need an even grid, so their unit factor scales
        it up.
        """
        return max(1, int(supercell)) * self.formula_unit_factor() - 1

    def effective_oct_counts(self) -> tuple[int, int, int]:
        """``n_oct_*`` (octahedra minus one) of the whole stack."""
        if len(self.builder_domains) < 2:
            return (
                self.effective_n_oct(self.perovskite_supercell_x),
                self.effective_n_oct(self.perovskite_supercell_y),
                self.effective_n_oct(self.perovskite_supercell_z),
            )
        self.store_active_domain()
        counts = stack_oct_counts(self.builder_domains, self.stacking_axis)
        return (counts[0] - 1, counts[1] - 1, counts[2] - 1)

    def builder_cell_repeats(self) -> Tuple[int, int, int]:
        """How many cube edges long the built cell is along each axis.

        One more than the octahedron-grid count: the cell built from an n x n x n
        supercell of a simple perovskite is n cube edges on a side, and the
        doubled formulas double that.
        """
        return tuple(int(count) + 1 for count in self.effective_oct_counts())  # type: ignore[return-value]

    def builder_cell_lengths(self) -> Tuple[float, float, float]:
        """The lattice constants of the cell being built, supercell included."""
        repeats = self.builder_cell_repeats()
        return (
            float(self.lattice_a) * repeats[0],
            float(self.lattice_b) * repeats[1],
            float(self.lattice_c) * repeats[2],
        )

    def reference_domain(self) -> DomainSpec:
        """Domain 0: the one the cell origin and in-plane lattice are read from."""
        self.store_active_domain()
        return self.builder_domains[0]

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
        """Half edges of domain 0's octahedron: the build's centre-to-vertex distances.

        With one domain these are the buffer's lattice constants. With a stack
        they are domain 0's, which is where the cell origin and the in-plane
        spacing come from; the other domains' spacing along the stacking axis
        enters through ``builder_half_length_kwargs``.
        """
        self.apply_perovskite_constraints()
        lattice = self.reference_domain().lattice
        return (0.5 * lattice[0], 0.5 * lattice[1], 0.5 * lattice[2])

    def builder_half_length_kwargs(self) -> Dict[str, Any]:
        """``center_to_vertex_distance_*`` for ``build_perovskite``: per layer when stacked."""
        half_a, half_b, half_c = self.half_edge_lengths()
        if len(self.builder_domains) < 2:
            return dict(
                center_to_vertex_distance_x=half_a,
                center_to_vertex_distance_y=half_b,
                center_to_vertex_distance_z=half_c,
            )
        half_x, half_y, half_z = stack_half_lengths(self.builder_domains, self.stacking_axis)
        return dict(
            center_to_vertex_distance_x=half_x,
            center_to_vertex_distance_y=half_y,
            center_to_vertex_distance_z=half_z,
        )

    def stored_domains(self) -> List[DomainSpec]:
        """The domain list to record on generation parameters: empty for one domain."""
        if len(self.builder_domains) < 2:
            return []
        return [DomainSpec(**domain.to_dict()) for domain in self.builder_domains]

    def builder_cell_origin(self) -> np.ndarray:
        half_a, half_b, half_c = self.half_edge_lengths()
        return np.asarray(self.perovskite_center, dtype=np.float64) - np.array(
            [half_a, half_b, half_c],
            dtype=np.float64,
        )

    def builder_supercell_lattice(self) -> np.ndarray:
        self.apply_perovskite_constraints()
        if len(self.builder_domains) >= 2:
            return stack_lattice(self.builder_domains, self.stacking_axis)
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
        periodic,
    ) -> list[str]:
        if len(self.builder_domains) >= 2:
            self.store_active_domain()
            return domain_atomic_labels_for_build(
                build,
                periodic=periodic,
                domains=self.builder_domains,
                stacking_axis=self.stacking_axis,
            )
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

    def generated_perovskite_with_periodicity(self, periodic) -> PerovskiteBuild:
        periodic = periodic_axes(periodic)
        half_kwargs = self.builder_half_length_kwargs()
        effective_x, effective_y, effective_z = self.effective_oct_counts()
        # The Controls panel asks for this more than once per frame; the key is the
        # full argument list, so any builder edit rebuilds. Slotted per periodicity
        # so a caller that wants both does not evict the other every frame.
        key = (
            tuple(np.asarray(self.perovskite_center, dtype=np.float64).ravel().tolist()),
            effective_x,
            effective_y,
            effective_z,
            tuple(
                tuple(np.asarray(value, dtype=np.float64).ravel().tolist())
                for value in half_kwargs.values()
            ),
            int(self.stacking_axis),
            self.perovskite_tilt_system,
            self.tilt_angle_x,
            self.tilt_angle_y,
            self.tilt_angle_z,
        )
        return self._cached(
            f"perovskite_build_{periodic}",
            key,
            lambda: build_perovskite(
                center=self.perovskite_center,
                n_oct_x=effective_x,
                n_oct_y=effective_y,
                n_oct_z=effective_z,
                **half_kwargs,
                tilt_system=GLAZER_TILT_SYSTEMS[self.perovskite_tilt_system],
                tilt_angle_x_deg=self.tilt_angle_x,
                tilt_angle_y_deg=self.tilt_angle_y,
                tilt_angle_z_deg=self.tilt_angle_z,
                periodic=periodic,
            ),
        )

    def generated_chemical_structure(self) -> ChemicalStructure:
        return self.generated_chemical_structure_with_periodicity(self.treat_as_periodic)

    def generated_chemical_structure_with_periodicity(
        self,
        periodic,
    ) -> ChemicalStructure:
        periodic = periodic_axes(periodic)
        build = self.generated_perovskite_with_periodicity(periodic)
        lattice = self.builder_supercell_lattice()
        cell_origin = self.builder_cell_origin()
        if self.stack_problems():
            raise ValueError("; ".join(self.stack_problems()))
        # The top-level composition fields describe domain 0; in a stack the
        # buffer may hold another domain, so read them off the spec itself.
        reference = self.reference_domain()
        if reference.formula_mode == "high_entropy":
            a_symbol = normalized_distribution(
                reference.high_entropy_a_sites, site_name="A"
            )[0][0]
            b_symbol = normalized_distribution(
                reference.high_entropy_b_sites, site_name="B"
            )[0][0]
            x_symbol = normalized_distribution(
                reference.high_entropy_x_sites, site_name="X"
            )[0][0]
        elif len(self.builder_domains) >= 2:
            a_symbol = self._normalize_element_symbol(reference.a_site_element)
            b_symbol = self._normalize_element_symbol(reference.b_site_element)
            x_symbol = self._normalize_element_symbol(reference.x_site_element)
        elif self.formula_key() == "high_entropy":
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

        builder_defects = self.builder_defects()
        cartesian_coords, atomic_labels, site_roles, resolution = apply_defects(
            build,
            self.atomic_labels_for_build(build, periodic=periodic),
            periodic=periodic,
            stored_periodic=periodic,
            defects=builder_defects,
            cell_origin=cell_origin,
        )
        self.defect_message = "; ".join(resolution.warnings)
        structure = ChemicalStructure.with_zero_magnetic_moments(
            name="Builder preview",
            lattice=lattice,
            cartesian_coords=cartesian_coords,
            atomic_labels=atomic_labels,
            is_periodic=periodic,
        )

        half_a, half_b, half_c = self.half_edge_lengths()
        effective_x, effective_y, effective_z = self.effective_oct_counts()
        stacked = len(self.builder_domains) >= 2
        formula_key = reference.formula_mode if stacked else self.formula_key()
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
            domains=self.stored_domains(),
            stacking_axis=int(self.stacking_axis),
            a_site_element=a_symbol,
            b_site_element=b_symbol,
            x_site_element=x_symbol,
            formula_mode=formula_key,
            a2_site_element=(
                self._normalize_element_symbol(reference.a2_site_element)
                if formula_key in ("quadruple", "dq")
                else reference.a2_site_element.strip()
            ),
            b2_site_element=(
                self._normalize_element_symbol(reference.b2_site_element)
                if formula_key in ("double", "dq")
                else reference.b2_site_element.strip()
            ),
            high_entropy_a_sites=(
                normalized_distribution(reference.high_entropy_a_sites, site_name="A")
                if formula_key == "high_entropy"
                else list(reference.high_entropy_a_sites)
            ),
            high_entropy_b_sites=(
                normalized_distribution(reference.high_entropy_b_sites, site_name="B")
                if formula_key == "high_entropy"
                else list(reference.high_entropy_b_sites)
            ),
            high_entropy_x_sites=(
                normalized_distribution(reference.high_entropy_x_sites, site_name="X")
                if formula_key == "high_entropy"
                else list(reference.high_entropy_x_sites)
            ),
            spin_pattern="None",
            spin_moment_magnitude=0.0,
            x_vacancy_fraction=0.0,
            x_removed_count=0,
            removed_x_site_indices=np.zeros(0, dtype=np.int64),
            defects=builder_defects,
            site_roles=site_roles,
            permutation=np.arange(len(site_roles), dtype=np.int64),
            cell_origin=cell_origin,
            source="perovskite_builder",
        )
        return structure

    def generated_build_for_structure(
        self,
        structure: ChemicalStructure,
    ) -> PerovskiteBuild | None:
        def rebuild() -> PerovskiteBuild | None:
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

        # The no-provenance branch also depends on the live builder fields, so they
        # join the key -- they are already a cheap tuple.
        key = (self._structure_signature(structure), self.builder_fields_signature())
        return self._cached(("generated_build", id(structure)), key, rebuild)

    def current_structure(self) -> ChemicalStructure | None:
        """The active structure. Builder edits are applied to it in place."""
        return self.focus

    def rendered_structure(self) -> ChemicalStructure | None:
        """The structure the 3D view draws: the focus, or a non-periodic rebuild of it.

        Called more than once per frame (the view and its title share it), and the
        rebuild is a full structure generation, so it is memoized on the focus.
        """
        focus = self.focus
        if (
            # Strict: this branch *rebuilds from the parameters*, so for a relaxed
            # structure it would draw the geometry the relaxation started from and
            # show nothing of the result. Such a structure falls through and is
            # drawn as it actually is -- without image expansion, which is a
            # builder-only feature until tiling learns to work from coordinates.
            self.focus_has_generated_provenance()
            and self.effective_render_periodic_images()
            and focus is not None
            and focus.is_periodic
        ):
            return self._cached(
                "rendered_structure",
                (self._structure_signature(focus), focus.name),
                lambda: generated_structure_from_parameters(
                    focus.generation_parameters,
                    name=focus.name,
                    periodic=False,
                ),
            )
        return focus

    def focus_is_loaded(self) -> bool:
        """True when the focus is a structure with no generation parameters."""
        return (
            self.focus is not None
            and getattr(self.focus, "generation_parameters", None) is None
        )

    def sync_active_structure(self) -> None:
        self.sync_builder_binding()
        self.sync_cell_binding()
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

    # ------------------------------------------------------------------
    # Remote compute
    # ------------------------------------------------------------------
    def remote_client_if_any(self) -> Any:
        """The client, or None if nothing has ever been submitted or connected."""
        return self._remote_client

    def remote_client(self) -> Any:
        """The client, built on first use and kept in sync with the URL fields."""
        if self._remote_client is None:
            self._remote_client = RemoteClient(self.remote_url, self.remote_token)
        else:
            self._remote_client.url = self.remote_url
            self._remote_client.token = self.remote_token
        return self._remote_client

    def remote_params(self) -> Dict[str, Any]:
        return {
            "calculation": REMOTE_CALCULATIONS[self.remote_calculation_index],
            "optimizer": REMOTE_OPTIMIZERS[self.remote_optimizer_index],
            "fmax": float(self.remote_fmax),
            "steps": int(self.remote_steps),
        }

    def connect_remote(self) -> None:
        self.remote_message = ""
        self.remote_client().check_health()

    def submit_remote_job(self) -> None:
        """Send the active structure off to be relaxed.

        Validation happens here rather than on the server so an impossible
        combination (relaxing the cell of a cluster, say) is refused instantly
        instead of after a round trip and a queue slot.
        """
        structure = self.focus
        if structure is None:
            self.remote_message = "No active structure to submit."
            return
        params = self.remote_params()
        try:
            job = self.remote_client().submit(structure, params=params)
        except remote_protocol.ProtocolError as exc:
            self.remote_message = str(exc)
            return
        if params["calculation"] == "single-point":
            # Nothing new arrives from a single point -- the geometry is the
            # one already in the list -- so it gets no row; its energy goes to
            # the energy pane, which is where it is worth looking.
            self.two_d_plot_index = TWO_D_PLOT_LIVE_ENERGY
            self.remote_message = f"Submitted single point on {structure.name}."
            return
        # The job gets its row in the structure list now rather than when it
        # lands: a copy of the submitted structure, named for the result, right
        # under its source. The result replaces it in place.
        placeholder = copy.deepcopy(structure)
        placeholder.spin_configurations = []
        placeholder.name = self.unique_structure_name(f"{structure.name} relaxed")
        self.structures.insert(self.index_of(structure) + 1, placeholder)
        self._remote_placeholders[job.key] = placeholder
        # The trace is the thing worth watching while it runs, and the 2D pane is
        # where plots live, so submitting brings it up rather than leaving the user
        # to find it.
        self.two_d_plot_index = TWO_D_PLOT_RELAXATION
        self.remote_message = f"Submitted {structure.name}."

    def remote_job_for_placeholder(self, structure: ChemicalStructure) -> Any:
        """The live job whose result will replace ``structure``, or None."""
        client = self.remote_client_if_any()
        if client is None:
            return None
        for key, placeholder in self._remote_placeholders.items():
            if placeholder is structure:
                for job in client.jobs:
                    if job.key == key:
                        return job
        return None

    def remote_placeholder_status(self, structure: ChemicalStructure) -> str | None:
        """What the structure list says beside a placeholder row."""
        job = self.remote_job_for_placeholder(structure)
        if job is None:
            return None
        if job.status == remote_protocol.STATUS_QUEUED:
            return "cancelling..." if job.cancelling else "queued"
        if job.cancelling:
            return f"cancelling after step {job.step}..."
        return f"relaxing, step {job.step}" if job.energy is not None else "starting..."

    def cancel_remote_job(self, job: Any) -> None:
        client = self.remote_client_if_any()
        if client is not None:
            client.cancel(job)

    def _take_remote_placeholder(self, job: Any) -> ChemicalStructure | None:
        """The placeholder ``job`` was given, if it is still in the list."""
        placeholder = self._remote_placeholders.pop(job.key, None)
        if placeholder is None or self.index_of(placeholder) < 0:
            return None
        return placeholder

    def _replace_remote_placeholder(
        self, placeholder: ChemicalStructure | None, structure: ChemicalStructure
    ) -> None:
        """Put ``structure`` where ``placeholder`` sat (or at the end without one)."""
        if placeholder is None:
            structure.name = self.unique_structure_name(structure.name)
            self.structures.append(structure)
            return
        index = self.index_of(placeholder)
        structure.name = placeholder.name
        self.structures[index] = structure
        if self.focus is placeholder:
            self.set_focus(structure)
            self.sync_active_structure()

    def collect_remote_job(self, job: Any) -> None:
        """Take delivery of one finished job.

        A cancelled or failed job leaves the structure list alone -- there is
        nothing to add -- and says what happened in the panel instead.
        """
        if self.collect_live_energy(job):
            return
        if (job.params or {}).get("calculation") == "single-point":
            self.collect_single_point(job)
            return
        placeholder = self._take_remote_placeholder(job)
        if job.structure is None:
            # Nothing arrived, so the row that was holding its place goes too.
            if placeholder is not None:
                self.remove_structure(placeholder)
            if job.status == remote_protocol.STATUS_CANCELLED:
                self.remote_message = f"{job.label}: cancelled."
            else:
                self.remote_message = f"{job.label}: {job.error or 'failed'}."
            return

        structure = job.structure
        cancelled = job.status == remote_protocol.STATUS_CANCELLED
        self._replace_remote_placeholder(placeholder, structure)

        # Measured against the exact structure that was submitted, which the job
        # still holds -- so this is an exact statement about the relaxation, not
        # an estimate recovered from the geometry afterwards.
        drift = geometry_drift(job.template, structure)
        if drift is not None:
            self.relaxation_drift[id(structure)] = drift

        moments = remote_protocol.moments_from_result(job.result or {})
        if moments is not None and moments.size == structure.atom_count:
            self.chgnet_moments[id(structure)] = moments

        if self.remote_focus_on_arrival:
            self.set_focus(structure)
            self.sync_active_structure()
        energy = (job.result or {}).get("energy")
        steps = (job.result or {}).get("steps", 0)
        if cancelled:
            self.remote_message = (
                f"{structure.name}: stopped after {steps} steps"
                + (f", E = {float(energy):.6f} eV" if energy is not None else "")
                + " (not converged)."
            )
        else:
            self.remote_message = (
                f"{structure.name}: {float(energy):.6f} eV"
                if energy is not None
                else f"{structure.name} arrived."
            )

    def collect_single_point(self, job: Any) -> None:
        """A single point asked for by hand: one point on the energy pane.

        The structure it was computed on is already in the list, so nothing is
        added there; the |m| diagnostics attach to that structure, and the
        energy joins the pane beside the live samples.
        """
        source = job.template
        energy = (job.result or {}).get("energy") if job.status == remote_protocol.STATUS_DONE else None
        if energy is None:
            self.remote_message = (
                f"{job.label}: cancelled."
                if job.status == remote_protocol.STATUS_CANCELLED
                else f"{job.label}: {job.error or 'failed'}."
            )
            return
        moments = remote_protocol.moments_from_result(job.result or {})
        if moments is not None and source is not None and moments.size == source.atom_count:
            self.chgnet_moments[id(source)] = moments
        atom_count = source.atom_count if source is not None else 0
        self.record_energy_sample(float(energy), atom_count, label=job.label, live=False)
        per_atom = f" ({float(energy) / atom_count:.4f} eV/atom)" if atom_count else ""
        self.remote_message = f"{job.label}: {float(energy):.6f} eV{per_atom}"

    # -- live single-point energy ------------------------------------------
    @staticmethod
    def live_energy_signature(structure: ChemicalStructure) -> Tuple[object, ...]:
        """What the energy depends on: geometry, species and periodicity.

        Content only -- not identity. The builder installs a fresh parameters
        object on every regeneration, and a regeneration that changes nothing
        the potential sees should not cost a calculation.
        """
        return (
            _array_signature(structure.lattice, structure.cartesian_coords),
            tuple(structure.atomic_labels),
            tuple(periodic_axes(structure.periodic_axes if structure.periodic_axes is not None else structure.is_periodic)),
        )

    def set_live_energy_enabled(self, enabled: bool) -> None:
        self.live_energy_enabled = bool(enabled)
        if enabled:
            self.live_energy_message = ""
            # The plot is the point of switching this on.
            self.two_d_plot_index = TWO_D_PLOT_LIVE_ENERGY

    def record_energy_sample(
        self,
        energy: float,
        atom_count: int,
        *,
        label: str = "",
        live: bool = False,
        stamp: float | None = None,
    ) -> None:
        """Add one structure's energy to the pane, keeping the last N."""
        self.live_energy_points.append(
            EnergySample(
                energy=float(energy),
                atom_count=int(atom_count),
                label=label,
                live=bool(live),
                stamp=time.time() if stamp is None else float(stamp),
            )
        )
        del self.live_energy_points[:-LIVE_ENERGY_MAX_POINTS]

    def advance_live_energy(self, now: float | None = None) -> None:
        """One frame of the live energy: send the next request if it is time.

        Three gates, in order. Nothing is sent while a request is out, so the
        points arrive at whatever rate the server answers. Nothing is sent
        before the interval has passed since the last send, so the target rate
        is a ceiling. And an unchanged structure is not recomputed -- or
        re-plotted: the pane is one point per structure, so a structure that
        has not changed adds nothing to it.
        """
        if not self.live_energy_enabled or self.focus is None:
            return
        if self._live_energy_job is not None:
            return
        now = time.time() if now is None else float(now)
        if now - self._live_energy_sent_at < LIVE_ENERGY_INTERVAL_SECONDS:
            return
        focus = self.focus
        signature = self.live_energy_signature(focus)
        if signature == self._live_energy_signature and self.live_energy_last is not None:
            return
        try:
            job = self.remote_client().submit(
                focus, params={"calculation": "single-point"}, label="live energy"
            )
        except remote_protocol.ProtocolError as exc:
            self.live_energy_message = str(exc)
            self._live_energy_sent_at = now
            return
        self._live_energy_job = job
        self._live_energy_pending_signature = signature
        self._live_energy_sent_at = now

    def collect_live_energy(self, job: Any) -> bool:
        """Take delivery of a live-energy job; True if ``job`` was one.

        The job is forgotten from the client's list afterwards: two a second
        would bury the relaxations the list exists to show.
        """
        if job is not self._live_energy_job:
            return False
        self._live_energy_job = None
        energy = (job.result or {}).get("energy") if job.status == remote_protocol.STATUS_DONE else None
        if energy is None:
            self.live_energy_message = (
                "cancelled"
                if job.status == remote_protocol.STATUS_CANCELLED
                else (job.error or "failed")
            )
        else:
            self.live_energy_message = ""
            self.live_energy_last = float(energy)
            self._live_energy_signature = self._live_energy_pending_signature
            # Only while the toggle is still on: a sample that lands after it
            # was switched off is an answer nobody is waiting for.
            if self.live_energy_enabled:
                self.record_energy_sample(
                    float(energy),
                    job.template.atom_count,
                    label=job.template.name,
                    live=True,
                )
        client = self.remote_client_if_any()
        if client is not None:
            client.forget(job)
        return True

    def selected_remote_job(self) -> Any:
        """The job the trace plot draws, which follows the work unless pinned.

        Lives here rather than in the panel because two panels ask now: the job
        list in Calculate, and the 2D pane in the Structure View.
        """
        client = self.remote_client_if_any()
        if client is None or not client.jobs:
            return None
        for job in client.jobs:
            if job.key == self.remote_selected_job_key:
                return job
        live = client.live_jobs
        return live[0] if live else client.jobs[-1]

    def relaxation_drift_for(self, structure: ChemicalStructure | None) -> Any:
        """How far ``structure`` moved during the relaxation that produced it."""
        if structure is None:
            return None
        return self.relaxation_drift.get(id(structure))

    def chgnet_moments_for(self, structure: ChemicalStructure | None) -> np.ndarray | None:
        """CHGNet's |m| diagnostics for ``structure``, if it came from a relaxation."""
        if structure is None:
            return None
        return self.chgnet_moments.get(id(structure))

    def run_selected_calculation(self) -> None:
        structure = self.focus
        if structure is None:
            return
        self.run_magnetic_structure_calculation(structure=structure)

APP_STATE = AppState()


def axis_length_control(
    label: str,
    value: float,
    enabled: bool,
    linked_note: str = "",
    *,
    minimum: float = 2.0,
) -> float:
    if not enabled:
        imgui.begin_disabled()

    _, value = imgui.input_float(f"{label} (A)", value, 0.1, 1.0, "%.3f")

    if not enabled:
        imgui.end_disabled()
        if linked_note:
            imgui.same_line()
            imgui.text_disabled(linked_note)

    return clamp_min(value, minimum)


def tilt_angle_control(label: str, value: float, enabled: bool) -> float:
    """One tilt angle. Disabled axes are greyed, not hidden, so the row stays put.

    The slider is sized against the space actually left after its label, rather
    than taking ImGui's default width: the Controls panel is narrow enough by
    default that a default-width slider plus "Tilt a (deg)" runs off the edge.
    """
    if not enabled:
        imgui.begin_disabled()

    label_width = imgui.calc_text_size(label).x
    available = imgui.get_content_region_avail().x
    imgui.push_item_width(
        max(70.0, available - label_width - 2.0 * imgui.get_style().item_spacing.x)
    )
    _, value = imgui.slider_float(label, value, -45.0, 45.0, "%.1f deg")
    imgui.pop_item_width()

    if not enabled:
        imgui.end_disabled()

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


def element_box_note(element: str) -> Tuple[str, tuple]:
    """What to say beside a substitution's element box, and in what colour.

    Three states, none of them an error. An empty box is how a site is emptied,
    so it is labelled rather than warned about. A symbol no element table knows
    is marked ``(?)`` and left alone -- the builder has always accepted one, and
    a placeholder species is a legitimate thing to be building.
    """
    text = element.strip()
    if not text:
        return "(vacancy)", VACANCY_RENDER_COLOR
    try:
        known = is_valid_symbol(normalize_element_symbol(text))
    except ValueError:
        known = False
    return ("", (0.0, 0.0, 0.0, 0.0)) if known else ("(?)", UNKNOWN_ELEMENT_COLOR)


@dataclass
class SummaryRow:
    """One line of the structure summary, and how to draw it.

    ``note`` is a dim trailing remark on the same line -- what the *ideal*
    lattice would hold, where a defect has changed the tally.
    """

    text: str
    note: str = ""
    error: bool = False


def periodicity_note(periodic) -> str:
    """``"periodic"``, ``"cluster"``, or the periodic axes when they differ."""
    axes = periodic_axes(periodic)
    if all(axes):
        return "periodic"
    if not any(axes):
        return "cluster"
    return "periodic along " + "".join(
        name for name, flag in zip(DOMAIN_AXIS_NAMES, axes) if flag
    )


def element_tally(symbols: Sequence[str]) -> str:
    """``"Fe: 7, Zn: 1"`` -- what a set of sites is made of."""
    counts: Dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    return ", ".join(f"{symbol}: {count}" for symbol, count in sorted(counts.items()))


def builder_summary_rows(state: AppState) -> List[SummaryRow]:
    """The active structure at a glance: formula, cell, composition, tilts.

    Returned as rows rather than drawn in place so the 3D view can float them
    over a corner of the plot, where what they describe actually is. Reporting
    what the structure *contains* rather than what the ideal lattice would hold
    is the whole point of having it next to the picture -- defects are applied
    after the build, so the two differ exactly when you are working on them.

    The structure's name is not among them: it titles the box, so that collapsing
    the box leaves the name behind.
    """
    formula = FORMULA_MODES[int(state.formula_mode) % len(FORMULA_MODES)]
    rows: List[SummaryRow] = []
    if state.domain_count() >= 2:
        rows.append(
            SummaryRow(
                f"{state.domain_count()} domains stacked along "
                f"{DOMAIN_AXIS_NAMES[int(state.stacking_axis)]}",
                note=periodicity_note(state.treat_as_periodic),
            )
        )
        for index, domain in enumerate(state.builder_domains):
            rows.append(
                SummaryRow(
                    f"  {state.domain_label(index)}: "
                    f"{FORMULA_MODES[formula_index_from_key(domain.formula_mode)]}, "
                    f"{domain.oct_counts()[int(state.stacking_axis)]} layers"
                    + (" (interface from previous)" if domain.interface_from_previous else "")
                )
            )
    else:
        rows.append(
            SummaryRow(
                f"Formula: {formula}",
                note=periodicity_note(state.treat_as_periodic),
            )
        )
    cell_a, cell_b, cell_c = state.builder_cell_lengths()
    rows += [
        SummaryRow(
            f"a = {cell_a:.3f} A, b = {cell_b:.3f} A, c = {cell_c:.3f} A",
            note=(
                "cube edge "
                + ", ".join(
                    f"{value:.3f}"
                    for value in (state.lattice_a, state.lattice_b, state.lattice_c)
                )
                + " A"
            ),
        ),
    ]

    active_build = state.generated_perovskite()
    a_count = len(active_build.a_sites)
    b_count = len(active_build.b_sites)
    try:
        labels = state.atomic_labels_for_build(
            active_build, periodic=state.treat_as_periodic
        )
    except ValueError as exc:
        rows.append(SummaryRow(f"A sites: {a_count}"))
        rows.append(SummaryRow(f"B sites: {b_count}"))
        rows.append(SummaryRow(f"X sites: {len(active_build.x_sites)}"))
        rows.append(SummaryRow(str(exc), error=True))
        return rows

    ideal_by_role = {
        "A": labels[:a_count],
        "B": labels[a_count : a_count + b_count],
        "X": labels[a_count + b_count :],
    }
    focus = state.focus
    focus_params = getattr(focus, "generation_parameters", None)
    actual_by_role: Dict[str, List[str]] | None = None
    if focus is not None and focus_params is not None and focus_params.defects:
        actual_by_role = {"A": [], "B": [], "X": [], "H": []}
        for role, symbol in zip(focus_params.site_roles, focus.atomic_labels):
            actual_by_role.setdefault(role, []).append(symbol)

    for role, role_labels in ideal_by_role.items():
        ideal = element_tally(role_labels)
        if actual_by_role is None:
            rows.append(SummaryRow(f"{role} sites ({ideal})"))
            continue
        actual = element_tally(actual_by_role.get(role, []))
        rows.append(
            SummaryRow(
                f"{role} sites ({actual or 'none'})",
                note=f"ideal: {ideal}" if actual != ideal else "",
            )
        )
    if actual_by_role and actual_by_role.get("H"):
        rows.append(SummaryRow(f"Interstitial ({element_tally(actual_by_role['H'])})"))

    rows.append(
        SummaryRow(f"Tilt system: {GLAZER_TILT_SYSTEMS[state.perovskite_tilt_system]}")
    )
    rows.append(
        SummaryRow(
            "Tilt angles: "
            f"a = {state.tilt_angle_x:.1f} deg, "
            f"b = {state.tilt_angle_y:.1f} deg, "
            f"c = {state.tilt_angle_z:.1f} deg"
        )
    )
    return rows


def loaded_summary_rows(state: AppState) -> List[SummaryRow]:
    """The same readout for a structure with no builder provenance.

    It cannot report what ``builder_summary_rows`` reports -- there is no formula,
    no tilt system, and no A/B/X split to tally -- so it reports what a loaded
    structure does know: its cell, and what it is made of. The cell parameters
    are the ones the Cell editor drives, which is the point of showing them here
    rather than only in the panel.
    """
    focus = state.focus
    if focus is None:
        return []
    rows: List[SummaryRow] = [
        SummaryRow(
            f"Loaded: {focus.atom_count} atoms",
            note=periodicity_note(
                focus.periodic_axes if focus.periodic_axes is not None else focus.is_periodic
            ),
        )
    ]
    try:
        a, b, c, alpha, beta, gamma = cell_parameters(focus.lattice)
    except ValueError as exc:
        rows.append(SummaryRow(str(exc), error=True))
        return rows
    rows.append(SummaryRow(f"a = {a:.3f} A, b = {b:.3f} A, c = {c:.3f} A"))
    rows.append(
        SummaryRow(
            f"alpha = {alpha:.2f} deg, beta = {beta:.2f} deg, gamma = {gamma:.2f} deg"
        )
    )
    rows.append(SummaryRow(f"Composition: {element_tally(focus.element_symbols())}"))
    return rows


def summary_overlay_width(rows: Sequence[SummaryRow]) -> float:
    """How wide to hold the summary box, in pixels.

    Wide enough for the rows it has, and never narrower than a tilt line with
    every angle at full deflection. Left to size itself, the box would twitch
    every time an angle crossed from ``0.0`` to ``-12.5`` -- a readout that moves
    while you drag the slider you are reading is worse than one a few pixels
    wider than it needs to be.
    """
    spacing = imgui.get_style().item_spacing.x
    widest = imgui.calc_text_size(SUMMARY_WIDEST_TILT_ROW).x
    for row in rows:
        width = imgui.calc_text_size(row.text).x
        if row.note:
            width += spacing + imgui.calc_text_size(row.note).x
        widest = max(widest, width)
    return widest + 2.0 * imgui.get_style().window_padding.x


def draw_structure_summary_overlay(state: AppState, rect_min, rect_max) -> None:
    """Float the summary over the 3D view, titled with the structure's name.

    A real window rather than text on the plot's draw list: it gets a background
    to stay readable over the structure, it can be dragged out of the way, and it
    can be rolled up to its title bar -- which is why the name is the title and
    not a row, so that collapsing it leaves the name behind.

    Placed in the top-right corner of the plot the first time it appears and left
    alone after that, so a drag sticks. The id is fixed with ``###`` so renaming
    the structure retitles the box instead of replacing it with a fresh one back
    in the corner, having forgotten where it was put and whether it was open.

    Not dockable. It is a readout that belongs over the picture, and letting it
    dock would flash the whole docking overlay across the app every time it is
    nudged, offering to file it somewhere it makes no sense.
    """
    # A loaded structure has its own row set rather than none: the box is the
    # readout for whatever is on screen, and "nothing to say" was never true of
    # it -- only "nothing the builder can say".
    try:
        if state.is_builder_active():
            rows = builder_summary_rows(state)
        elif state.focus_is_loaded():
            rows = loaded_summary_rows(state)
        else:
            return
    except ValueError:
        return
    if not rows:
        return

    focus = state.focus
    title = (focus.name if focus is not None else "") or "Unnamed structure"
    imgui.set_next_window_pos(
        imgui.ImVec2(rect_max.x, rect_min.y),
        imgui.Cond_.first_use_ever,
        imgui.ImVec2(1.0, 0.0),
    )
    # Fixed width, free height: the height only changes when a row appears or
    # goes, which is a real change worth showing.
    imgui.set_next_window_size(
        imgui.ImVec2(summary_overlay_width(rows), 0.0), imgui.Cond_.always
    )
    imgui.set_next_window_bg_alpha(SUMMARY_OVERLAY_BG_ALPHA)
    flags = (
        imgui.WindowFlags_.no_resize.value
        | imgui.WindowFlags_.no_scrollbar.value
        | imgui.WindowFlags_.no_focus_on_appearing.value
        | imgui.WindowFlags_.no_nav.value
        | imgui.WindowFlags_.no_saved_settings.value
        | imgui.WindowFlags_.no_docking.value
    )
    if imgui.begin(f"{title}###structure_summary", None, flags):
        for row in rows:
            if row.error:
                imgui.push_style_color(imgui.Col_.text, (0.95, 0.35, 0.35, 1.0))
                imgui.text_wrapped(row.text)
                imgui.pop_style_color()
                continue
            imgui.text(row.text)
            if row.note:
                imgui.same_line()
                imgui.text_disabled(row.note)
    imgui.end()


FORMULA_CHANGE_POPUP = "Change formula?##formula_change_guard"


def formula_change_dialog(state: AppState) -> None:
    """Ask before a formula change discards the edits on the active structure.

    Modal, because the answer decides what the builder is even editing: the
    same structure rebuilt from the new formula's defaults, or a new one
    alongside the edited one.
    """
    if state.pending_formula_mode < 0:
        return
    if not imgui.is_popup_open(FORMULA_CHANGE_POPUP):
        imgui.open_popup(FORMULA_CHANGE_POPUP)
    opened, _ = imgui.begin_popup_modal(
        FORMULA_CHANGE_POPUP, None, imgui.WindowFlags_.always_auto_resize.value
    )
    if not opened:
        return
    requested = int(state.pending_formula_mode)
    name = "this structure" if state.focus is None else state.focus.name
    imgui.text_wrapped(
        f'You are changing the formula of "{name}". This will lose all edits '
        "unless you create a new structure."
    )
    imgui.spacing()
    imgui.text_disabled(
        f"{FORMULA_MODES[state.formula_mode]}  ->  {FORMULA_MODES[requested]}"
    )
    imgui.spacing()
    if imgui.button("Proceed", size=(140, 0)):
        state.apply_formula_change(requested)
        imgui.close_current_popup()
    imgui.same_line()
    if imgui.button("New Structure", size=(140, 0)):
        state.create_structure_with_formula(requested)
        imgui.close_current_popup()
    imgui.same_line()
    if imgui.button("Cancel", size=(90, 0)):
        state.pending_formula_mode = -1
        imgui.close_current_popup()
    imgui.end_popup()


def cell_length_control(
    label: str,
    value: float,
    enabled: bool,
    linked_note: str = "",
) -> float:
    """One cell edge. Like ``axis_length_control`` but with the cell's own floor.

    The builder's 2 A minimum is a statement about perovskites; a loaded file can
    legitimately carry a much shorter axis, so this clamps at ``MIN_CELL_LENGTH``.
    """
    if not enabled:
        imgui.begin_disabled()
    _, value = imgui.input_float(f"{label} (A)", value, 0.05, 0.5, "%.4f")
    if not enabled:
        imgui.end_disabled()
        if linked_note:
            imgui.same_line()
            imgui.text_disabled(linked_note)
    return clamp_min(value, MIN_CELL_LENGTH)


def cell_angle_control(label: str, value: float) -> float:
    _, value = imgui.input_float(f"{label} (deg)", value, 0.5, 5.0, "%.3f")
    return min(max(value, MIN_CELL_ANGLE), MAX_CELL_ANGLE)


def cell_controls(state: "AppState") -> None:
    """The cell editor: the loaded structure's counterpart to the Lattice section.

    Every edit here is a strain -- the cell moves and the fractional coordinates
    stay put -- so whatever distortion or relaxation the file carried survives it.
    That is the whole reason this is not just the builder's lattice fields pointed
    at a different structure: those *regenerate*, and regenerating a loaded
    structure would throw away the only thing it has.
    """
    imgui.text_wrapped(
        "Edits strain the cell and carry the atoms with it at fixed fractional "
        "coordinates, so the structure's distortions are preserved."
    )
    imgui.spacing()

    imgui.text("Cell lengths")
    locked = state.cell_lock_aspect
    state.cell_a = cell_length_control("a", state.cell_a, enabled=True)
    state.cell_b = cell_length_control(
        "b", state.cell_b, enabled=not locked, linked_note="locked to a"
    )
    state.cell_c = cell_length_control(
        "c", state.cell_c, enabled=not locked, linked_note="locked to a"
    )
    lock_changed, state.cell_lock_aspect = imgui.checkbox(
        "Lock aspect ratio", state.cell_lock_aspect
    )
    if lock_changed and state.cell_lock_aspect:
        # Freeze the shape that is on screen at the moment the box is ticked,
        # rather than whatever it was when the structure was loaded.
        state.capture_cell_aspect_ratio()
    if imgui.is_item_hovered():
        imgui.set_tooltip("Scale a, b and c together, keeping the cell's shape.")

    imgui.spacing()
    imgui.text("Cell angles")
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "alpha spans b and c, beta spans c and a, gamma spans a and b.\n"
            "The cell is rebuilt from these six parameters in its own orientation,\n"
            "so an edit that changes nothing leaves the structure exactly where it is."
        )
    state.cell_alpha = cell_angle_control("alpha", state.cell_alpha)
    state.cell_beta = cell_angle_control("beta", state.cell_beta)
    state.cell_gamma = cell_angle_control("gamma", state.cell_gamma)

    imgui.spacing()
    imgui.text("Tile into a supercell")
    imgui.push_item_width(110)
    _, state.cell_tile_x = imgui.input_int("Repeat a", state.cell_tile_x)
    _, state.cell_tile_y = imgui.input_int("Repeat b", state.cell_tile_y)
    _, state.cell_tile_z = imgui.input_int("Repeat c", state.cell_tile_z)
    imgui.pop_item_width()
    state.apply_cell_constraints()
    # A button rather than a live field: tiling changes the atom count, which
    # invalidates every saved spin configuration, and that is not something to do
    # on the way past 2 while typing 12.
    if imgui.button("Tile##cell_tile"):
        state.tile_focus()
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Replicates the cell. Saved spin configurations are dropped, because\n"
            "their per-atom moments no longer match the atom list."
        )

    if state.cell_message:
        imgui.spacing()
        imgui.text_wrapped(state.cell_message)


def periodic_axes_controls(axes: List[bool]) -> bool:
    """One checkbox: periodic in every direction, or a finite cluster.

    Edited in place as the per-axis triple the rest of the app speaks, so the
    model keeps the ability to be periodic along some axes only -- the widget
    just does not offer it.
    """
    changed, value = imgui.checkbox("Periodic##periodic", all(bool(flag) for flag in axes))
    if imgui.is_item_hovered():
        imgui.set_tooltip("Periodic in every direction, or a finite cluster.")
    if changed:
        axes[:] = [bool(value)] * 3
    return changed


def domain_controls(state: "AppState") -> None:
    """The domain row: an add button, the stacking axis, and the active domain's toggles.

    A structure is a stack of domains along one lattice direction; the builder
    fields below edit the domain chosen in the Structure panel's dropdown.
    Adding a domain grows the stack by one block matched in plane to the block
    it sits on.
    """
    imgui.text("Domains")
    if imgui.button("+ Add domain##add_domain"):
        state.add_domain()
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Grow the structure by one block along the stacking axis. The new "
            "domain must match the in-plane grid of the block it sits on."
        )
    if state.domain_count() >= 2:
        imgui.same_line()
        if imgui.button("Remove##remove_domain"):
            state.remove_domain(state.active_domain_index)

    imgui.same_line()
    imgui.push_item_width(60)
    axis_changed, axis = imgui.combo(
        "Stacking axis##stacking_axis", int(state.stacking_axis), ["a", "b", "c"]
    )
    imgui.pop_item_width()
    if axis_changed:
        state.set_stacking_axis(axis)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "The lattice direction domains are stacked along; the plane normal "
            "to it is where two domains meet."
        )

    if state.domain_count() >= 2:
        domain = state.active_domain()
        axis_name = ("a", "b", "c")[int(state.stacking_axis)]
        for boundary in state.active_domain_boundaries():
            if not boundary.editable:
                imgui.begin_disabled()
            changed, owns = imgui.checkbox(
                f"Owns interface with {boundary.neighbour_label} ({boundary.side})"
                f"##interface_{state.active_domain_index}_{boundary.side}",
                boundary.owned,
            )
            if not boundary.editable:
                imgui.end_disabled()
            if changed:
                state.set_interface_ownership(boundary, owns)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    f"The AO plane where this domain meets {boundary.neighbour_label} "
                    f"{boundary.side} it along {axis_name} (its A sites and the bridging "
                    "X atoms) takes this domain's composition when checked, and the "
                    "neighbour's when not. Exactly one of the two domains owns each "
                    "interface, so checking it here unchecks it there."
                    + (
                        ""
                        if boundary.editable
                        else "\nThe stacking axis is finite here, so there is no interface."
                    )
                )
        layers = domain.oct_counts()[int(state.stacking_axis)]
        imgui.text_disabled(
            f"Editing {state.domain_label(state.active_domain_index)}: "
            f"{layers} octahedral layer{'s' if layers != 1 else ''} along {axis_name}. "
            "Switch domains in the Structure panel."
        )
    if state.domain_message:
        imgui.push_style_color(imgui.Col_.text, (0.95, 0.35, 0.35, 1.0))
        imgui.text_wrapped(state.domain_message)
        imgui.pop_style_color()


def gui_controls() -> None:
    state = APP_STATE
    _drain_browser_uploads(state)
    _drain_remote_jobs(state)
    state.advance_live_energy()
    state.sync_builder_binding()
    state.sync_cell_binding()
    state.apply_perovskite_constraints()
    state.apply_cell_constraints()
    if imgui.button("New structure##builder"):
        state.create_new_structure()
    imgui.same_line()
    structure_load_controls(state)
    imgui.separator()

    imgui.spacing()
    if imgui.collapsing_header(
        "Perovskite builder##builder_panel", imgui.TreeNodeFlags_.default_open.value
    ):
        # Loading a file no longer decouples the whole panel. It decouples the
        # parts that speak about a perovskite the builder generated -- site roles,
        # octahedral tilts, grid-addressed defects -- and leaves the parts that are
        # true of any structure alone. Each capture is taken once per frame,
        # because editing inside a section can change the answer mid-frame and the
        # begin/end_disabled pair must agree.
        composition_disabled = not state.composition_editing_available()
        cell_editing = state.cell_editing_available()

        if not composition_disabled:
            domain_controls(state)
            imgui.spacing()

        imgui.text("Formula")
        if composition_disabled:
            imgui.same_line()
            imgui.text_disabled("(unavailable)")
            if imgui.is_item_hovered():
                imgui.set_tooltip(state.unavailable_reason("composition"))
            imgui.begin_disabled()
        imgui.push_item_width(250)
        formula_changed, state.formula_mode = imgui.combo(
            "##formula_mode",
            state.formula_mode,
            FORMULA_MODES,
        )
        imgui.pop_item_width()
        if formula_changed and state.formula_mode != state._last_formula_mode:
            requested = int(state.formula_mode)
            # Put the combo back before anything reads the builder. The rest of
            # the fields still belong to the old formula, and "has edits?" is
            # only meaningful against that formula's defaults -- compared
            # against the requested one, the defaults it opens with (a smaller
            # supercell, its own composition) would read as edits that nobody
            # made.
            state.formula_mode = state._last_formula_mode
            if state.domain_count() >= 2:
                # In a stack a formula is a property of one domain, so it is
                # changed in place -- matched to the neighbouring domains' grid
                # -- rather than by resetting the builder.
                state.set_active_domain_formula(requested)
            elif state.builder_has_edits():
                # Ask first: switching formula rebuilds from defaults, and the
                # edits it would drop are not something to discover afterwards.
                state.pending_formula_mode = requested
            else:
                state.apply_formula_change(requested)
        state.apply_perovskite_constraints()
        if composition_disabled:
            imgui.end_disabled()

        imgui.spacing()
        # Periodicity is a plain property of any structure, so it is edited
        # directly on the focus rather than through a regeneration.
        if cell_editing and state.focus is not None:
            focus_axes = list(
                state.focus.periodic_axes
                if state.focus.periodic_axes is not None
                else periodic_axes(state.focus.is_periodic)
            )
            changed = periodic_axes_controls(focus_axes)
            if changed:
                # Written straight onto the structure -- there is no regeneration
                # to route it through -- so the invalidation a builder edit gets
                # for free has to be asked for. ChemicalStructure.neighbors
                # branches on this, so the exchange couplings really do change.
                state.focus.set_periodic_axes(focus_axes)
                state.invalidate_after_geometry_change(state.focus)
        else:
            axes = list(state.periodic_axes_flags)
            if periodic_axes_controls(axes):
                state.periodic_axes_flags = tuple(axes)

        imgui.spacing()
        if imgui.collapsing_header(
            "Atoms##builder_atoms_panel", imgui.TreeNodeFlags_.default_open.value
        ):
            if composition_disabled:
                imgui.text_wrapped(state.unavailable_reason("composition"))
                imgui.spacing()
                imgui.begin_disabled()
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

            if not composition_disabled:
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
                        f"{symbol}: {count}"
                        for symbol, count in sorted(preview_counts.items())
                    )
                except ValueError as exc:
                    imgui.push_style_color(imgui.Col_.text, (0.95, 0.35, 0.35, 1.0))
                    imgui.text_wrapped(str(exc))
                    imgui.pop_style_color()
            if composition_disabled:
                imgui.end_disabled()

        imgui.spacing()
        if cell_editing:
            if imgui.collapsing_header(
                "Cell##loaded_cell_panel", imgui.TreeNodeFlags_.default_open.value
            ):
                cell_controls(state)
        elif imgui.collapsing_header("Lattice##builder_lattice_panel"):
            stacked = state.domain_count() >= 2
            axis_names = ("a", "b", "c")
            plane = in_plane_axes(state.stacking_axis) if stacked else ()
            if stacked:
                imgui.text_wrapped(
                    "In-plane size and spacing are shared by every domain of the "
                    "stack; editing them here moves the whole stack. Only the "
                    f"extent along {axis_names[int(state.stacking_axis)]} is this "
                    "domain's own."
                )
                imgui.spacing()
            supercell = [
                state.perovskite_supercell_x,
                state.perovskite_supercell_y,
                state.perovskite_supercell_z,
            ]
            for axis in range(3):
                _, supercell[axis] = imgui.input_int(
                    f"Supercell {axis_names[axis]}", supercell[axis], 1, 10
                )
                if stacked and axis in plane:
                    imgui.same_line()
                    imgui.text_disabled("(shared)")
            (
                state.perovskite_supercell_x,
                state.perovskite_supercell_y,
                state.perovskite_supercell_z,
            ) = (max(1, value) for value in supercell)
            state.apply_perovskite_constraints()
            imgui.spacing()
            imgui.text("Lattice constants")
            imgui.same_line()
            imgui.text_disabled("(whole cell)")
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "The lengths of the cell being built, supercell included.\n"
                    "Editing one sets the cube edge it divides into."
                )
            # Shown and edited as the lengths of the cell actually built rather
            # than as one octahedron's cube edge: a 3x3x3 supercell of a 4 A
            # perovskite is a 12 A cell, and that is the number the picture, the
            # export and the summary all agree on. The edge is what is stored.
            repeats = state.builder_cell_repeats()
            state.lattice_a = axis_length_control(
                "a",
                state.lattice_a * repeats[0],
                enabled=True,
                linked_note="shared" if (stacked and 0 in plane) else "",
                minimum=2.0 * repeats[0],
            ) / repeats[0]
            state.lattice_b = axis_length_control(
                "b",
                state.lattice_b * repeats[1],
                enabled=state.perovskite_type == 2,
                linked_note=(
                    "linked to a"
                    if state.perovskite_type in (0, 1)
                    else ("shared" if (stacked and 1 in plane) else "")
                ),
                minimum=2.0 * repeats[1],
            ) / repeats[1]
            state.lattice_c = axis_length_control(
                "c",
                state.lattice_c * repeats[2],
                enabled=state.perovskite_type in (1, 2),
                linked_note=(
                    "linked to a"
                    if state.perovskite_type == 0
                    else ("shared" if (stacked and 2 in plane) else "")
                ),
                minimum=2.0 * repeats[2],
            ) / repeats[2]

            imgui.spacing()
            imgui.text("Lattice type")
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
            if state.domain_count() >= 2:
                imgui.text_disabled("Shared by every domain of the stack.")
                imgui.spacing()
            tilt_controls_enabled = state.tilt_editing_available()
            if not tilt_controls_enabled:
                # Two different reasons to be off, and they are not interchangeable:
                # one is fixed by growing the supercell, the other by building the
                # structure instead of loading it.
                imgui.text_wrapped(
                    state.unavailable_reason("tilt")
                    or "Tilt systems need a supercell of at least 2 along every axis."
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

    # Outside the disabled scopes, or its own buttons would be greyed out with
    # everything else. A modal blocks the rest of the UI while it is up, so the
    # section it belongs to cannot be collapsed out from under it.
    formula_change_dialog(state)

    # Builder edits regenerate the active structure; cell edits strain it. Only
    # one of the two ever applies to a given structure -- see cell_editing_available.
    state.regenerate_focus_from_builder_if_changed()
    state.apply_cell_edits_if_changed()
    state.sync_active_structure()


def gui_rendering() -> None:
    """The Rendering panel, tabbed beside Calculate in the right-hand dock.

    Moved out of the Controls panel so the builder and the Selection panel
    have the whole left side.
    """
    state = APP_STATE
    _, state.show_unit_cell = imgui.checkbox("Draw unit cell", state.show_unit_cell)
    _, state.show_domain_cell = imgui.checkbox(
        "Draw active domain cell", state.show_domain_cell
    )
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Outline the block of the domain selected in the Structure panel, in "
            "yellow. Only a structure with two or more domains has one."
        )
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


def gui_calculate() -> None:
    """The calculation setup panel.

    Docked beside Magnetism rather than under the builder, so the whole
    solve-and-inspect loop lives on the right and the left stays the builder.
    """
    state = APP_STATE
    state.magnetic_solver_trials = max(0, state.magnetic_solver_trials)
    state.magnetic_solver_steps = max(0, state.magnetic_solver_steps)
    state.magnetic_solver_learning_rate = max(0.0, state.magnetic_solver_learning_rate)
    state.magnetic_solver_energy_tolerance = max(
        0.0, state.magnetic_solver_energy_tolerance
    )
    state.magnetic_solver_patience = max(0, state.magnetic_solver_patience)
    state.magnetic_solver_max_flip_order = max(0, state.magnetic_solver_max_flip_order)

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
        changed, state.magnetic_net_charge = imgui.input_int(
            "Net cell charge",
            state.magnetic_net_charge,
            1,
            1,
        )
        if changed:
            # This feeds the oxidation-state enumeration, so the assignment
            # list itself is invalid -- not just the spin solutions derived
            # from it. Drop both and force a fresh baseline.
            state.magnetic_oxidation_assignments = []
            state.selected_oxidation_assignment_index = 0
            state._baseline_structure = None
            state.magnetic_oxidation_status = (
                "Net charge changed. Re-run Magnetic Structure to re-enumerate "
                "oxidation states."
            )
        solver_settings_changed = solver_settings_changed or changed
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Total charge the oxidation-state enumeration must balance to.\n"
                "Leave at 0 unless you are modelling a deliberately charged cell:\n"
                "an oxygen vacancy is already compensated by reducing cations."
            )
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
        degeneracy_changed, state.plot_degenerate_configs = imgui.checkbox(
            "Plot degenerate configs", state.plot_degenerate_configs
        )
        plot_cap_changed = plot_cap_changed or degeneracy_changed
        imgui.text_disabled("Set max flip configs to 0 or less to represent no limit.")
        imgui.text_disabled(
            "Plotted configurations are kept across structure edits; reference "
            "orderings are always kept. Their energies follow the edits only while "
            "'Update spin energies interactively' is on -- otherwise they hold "
            "until refreshed from the results panel or a solve."
        )
        imgui.text_disabled(
            "Plot degenerate configs off: one configuration per distinct energy, "
            "so the cap reaches further up the landscape."
        )
        if plot_cap_changed:
            state.refresh_landscape_energies()
        if solver_settings_changed:
            state.magnetic_solution_cache = {}
            state.selected_spin_config_index = 0
            if state.magnetic_oxidation_assignments:
                state.magnetic_spin_status = (
                    "Solver settings changed. Re-run Magnetic Structure or "
                    "solve the current oxidation states again to refresh results."
                )

    imgui.spacing()
    if imgui.button("Run Magnetic Structure", size=(180, 0)):
        state.run_selected_calculation()

    imgui.spacing()
    gui_remote_compute(state)


def gui_remote_compute(state: "AppState") -> None:
    """CHGNet on another machine: connection, settings, and the job list.

    All of it in one collapsing header rather than split across Calculate and
    Magnetism. The results panel is about the spin pipeline and returns
    early whenever its results do not match the focus, which is exactly when a
    submitted job is most worth watching -- so the jobs live here, next to the
    button that creates them.
    """
    if not imgui.collapsing_header("Remote compute (CHGNet)##remote"):
        return

    client = state.remote_client_if_any()

    imgui.text_disabled(
        "Runs on a quick-mag server. Point this at 127.0.0.1 and put the tunnel "
        "(ssh -N -L 8765:127.0.0.1:8765 HOST) in front of it -- the browser build "
        "can only reach loopback."
    )
    imgui.spacing()

    # -- connection --------------------------------------------------------
    style = imgui.get_style()
    connect_label = "Connect"
    connect_width = imgui.calc_text_size(connect_label).x + style.frame_padding.x * 2.0
    field_width = max(
        120.0, imgui.get_content_region_avail().x - connect_width - style.item_spacing.x
    )
    imgui.push_item_width(field_width)
    _, state.remote_url = imgui.input_text("##remote_url", state.remote_url)
    imgui.pop_item_width()
    imgui.same_line()
    if imgui.button(connect_label):
        state.connect_remote()

    imgui.push_item_width(field_width)
    _, state.remote_token = imgui.input_text("Token##remote_token", state.remote_token)
    imgui.pop_item_width()
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Printed by 'quick-mag serve' at startup.\n"
            "It is what stops another browser tab from queueing work on your port."
        )

    if client is not None and client.health_error:
        imgui.push_style_color(imgui.Col_.text, (0.95, 0.35, 0.35, 1.0))
        imgui.text_wrapped(client.health_error)
        imgui.pop_style_color()
    elif client is not None and client.health:
        health = client.health
        device = health.get("cuda_device") or health.get("device") or "unknown device"
        queued = health.get("queue_depth", 0)
        imgui.push_style_color(imgui.Col_.text, REMOTE_STATUS_COLORS[remote_protocol.STATUS_DONE])
        imgui.text_wrapped(f"Connected - {device}" + (f", {queued} queued" if queued else ""))
        imgui.pop_style_color()
    else:
        imgui.text_disabled("Not connected. Press Connect to check the server.")

    imgui.spacing()
    imgui.separator()

    # -- what to run -------------------------------------------------------
    imgui.push_item_width(200.0)
    _, state.remote_calculation_index = imgui.combo(
        "Calculation##remote_calculation",
        state.remote_calculation_index,
        [REMOTE_CALCULATION_LABELS.get(key, key) for key in REMOTE_CALCULATIONS],
    )
    calculation = REMOTE_CALCULATIONS[state.remote_calculation_index]
    if imgui.is_item_hovered():
        imgui.set_tooltip(REMOTE_CALCULATION_HINTS.get(calculation, ""))

    single_point = calculation == "single-point"
    if single_point:
        imgui.begin_disabled()
    _, state.remote_optimizer_index = imgui.combo(
        "Optimizer##remote_optimizer", state.remote_optimizer_index, REMOTE_OPTIMIZERS
    )
    _, state.remote_fmax = imgui.input_float(
        "fmax (eV/A)##remote_fmax", state.remote_fmax, 0.001, 0.01, "%.4f"
    )
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Force convergence threshold."
        )
    _, state.remote_steps = imgui.input_int(
        "Max steps##remote_steps", state.remote_steps, 10, 50
    )
    if single_point:
        imgui.end_disabled()
    imgui.pop_item_width()
    state.remote_fmax = max(1e-6, float(state.remote_fmax))
    state.remote_steps = max(1, int(state.remote_steps))

    _, state.remote_focus_on_arrival = imgui.checkbox(
        "Focus the result when it arrives", state.remote_focus_on_arrival
    )

    imgui.spacing()
    target = state.focus
    if target is None:
        imgui.begin_disabled()
    if imgui.button("Calculate##remote_calculate", size=(200, 0)):
        state.submit_remote_job()
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Run the calculation chosen above on the active structure. A single\n"
            "point adds its energy to the \"CHGNet energy\" pane; an optimization\n"
            "adds a row to the structure list that the result fills in."
        )
    if target is None:
        imgui.end_disabled()
    if target is not None:
        imgui.same_line()
        imgui.text_disabled(f"{target.name} ({target.atom_count} atoms)")

    if state.remote_message:
        imgui.text_wrapped(state.remote_message)

    live_changed, live = imgui.checkbox(
        "Live CHGNet energy##live_energy", state.live_energy_enabled
    )
    if live_changed:
        state.set_live_energy_enabled(live)
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Keep a single-point energy of the active structure coming, at most\n"
            f"every {LIVE_ENERGY_INTERVAL_SECONDS * 1000:.0f} ms and one request at a time, "
            "plotted under\n\"CHGNet energy\" in the 2D pane as one point per structure. "
            "An unchanged\nstructure is not recomputed."
        )
    if state.live_energy_enabled:
        imgui.same_line()
        if state.live_energy_last is not None:
            focus = state.focus
            per_atom = (
                f" ({state.live_energy_last / focus.atom_count:.4f} eV/atom)"
                if focus is not None and focus.atom_count
                else ""
            )
            imgui.text_disabled(f"E = {state.live_energy_last:.4f} eV{per_atom}")
        elif state.live_energy_message:
            imgui.text_disabled(state.live_energy_message)
        else:
            imgui.text_disabled("waiting for the first point")

    # CHGNet reports unsigned |m| magnitudes. They are worth seeing -- they say
    # which sites the potential thinks carry moment at all -- but they are not a
    # spin configuration, so they are shown here and never written into the
    # structure's own moments.
    moments = state.chgnet_moments_for(target)
    if moments is not None and moments.size:
        imgui.text_disabled(
            f"CHGNet |m|: mean {float(moments.mean()):.3f}, "
            f"max {float(moments.max()):.3f} mu_B (diagnostic, unsigned)"
        )

    gui_relaxation_drift(state, target)

    # -- jobs --------------------------------------------------------------
    if client is None or not client.jobs:
        return

    imgui.spacing()
    imgui.separator()
    imgui.text("Jobs")
    imgui.same_line()
    if imgui.small_button("Clear finished##remote_clear"):
        client.clear_finished()

    for job in list(client.jobs):
        gui_remote_job_row(state, client, job)

    imgui.text_disabled(
        "The energy trace is in the 2D pane of the Structure View, under "
        "\"Relaxation\"."
    )


def gui_relaxation_drift(state: "AppState", structure: ChemicalStructure | None) -> None:
    """What the relaxation did to the geometry, for a structure that came from one.

    This is the honest replacement for the builder fields, which a relaxation
    silently invalidates: rather than showing parameters that no longer generate
    this structure, say how far it has moved away from them.
    """
    drift = state.relaxation_drift_for(structure)
    if drift is None:
        return

    imgui.spacing()
    if not imgui.tree_node_ex("Relaxation drift##relaxation_drift"):
        imgui.same_line()
        imgui.text_disabled(drift.headline())
        return

    imgui.text_disabled("Measured against the structure that was submitted.")
    imgui.text(f"Cell a b c:  {drift.cell_summary()}")
    imgui.text(f"Volume:      {(drift.volume_ratio - 1.0) * 100.0:+.3f}%")
    if drift.comparable:
        imgui.text(f"Atoms:       {drift.atom_summary()}")
    if not drift.cell_is_orthogonal:
        # The builder only ever emits diagonal cells, so this is the point of no
        # return for any future attempt to fit parameters back onto the geometry.
        imgui.push_style_color(imgui.Col_.text, (0.95, 0.75, 0.35, 1.0))
        imgui.text_wrapped(
            "The relaxed cell is no longer orthogonal, so no choice of builder "
            "parameters can reproduce it."
        )
        imgui.pop_style_color()
    imgui.text_disabled(
        "The builder panel is disabled for this structure: editing it there would "
        "rebuild the ideal geometry. Its cell is editable under Cell."
    )
    imgui.tree_pop()


def gui_remote_job_row(state: "AppState", client: Any, job: Any) -> None:
    """One line per job: what it is, where it got to, and how to stop it."""
    color = REMOTE_STATUS_COLORS.get(job.status, (0.8, 0.8, 0.8, 1.0))
    imgui.push_style_color(imgui.Col_.text, color)
    # The selectable spans the full row width, so without allow_overlap it takes
    # the click meant for the Cancel/x buttons drawn on top of it.
    clicked, _ = imgui.selectable(
        f"{job.label}##remote_job_{job.key}",
        job.key == state.remote_selected_job_key,
        imgui.SelectableFlags_.allow_overlap,
    )
    imgui.pop_style_color()
    if clicked:
        state.remote_selected_job_key = job.key

    imgui.same_line()
    imgui.text_disabled(f"{job.status_line()}  ({job.elapsed():.0f}s)")

    if job.is_live:
        imgui.same_line()
        if job.cancelling:
            imgui.begin_disabled()
            imgui.small_button(f"Cancelling##remote_cancel_{job.key}")
            imgui.end_disabled()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "The server stops a relaxation between optimizer steps, so this "
                    "takes as long as the step in progress. The geometry reached so "
                    "far is kept."
                )
        else:
            if imgui.small_button(f"Cancel##remote_cancel_{job.key}"):
                client.cancel(job)
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Stop after the current optimizer step. The partially relaxed "
                    "structure is kept."
                )
    else:
        imgui.same_line()
        if imgui.small_button(f"x##remote_forget_{job.key}"):
            client.forget(job)
            if state.remote_selected_job_key == job.key:
                state.remote_selected_job_key = ""


def relaxation_trace_status(state: "AppState") -> str:
    """What the Relaxation pane says when it has no curve to draw."""
    client = state.remote_client_if_any()
    if client is None or not client.jobs:
        return (
            "No relaxations yet. Submit one from Calculate > Remote compute and "
            "its energy trace appears here as it runs."
        )
    job = state.selected_remote_job()
    if job is None:
        return "Select a job in Calculate > Remote compute."
    if job.status == remote_protocol.STATUS_QUEUED:
        return f"{job.label}: queued."
    if not job.is_live and not job.trajectory():
        return f"{job.label}: {job.status_line()}"
    return f"{job.label}: waiting for the first optimizer steps..."


def plot_relaxation_trace(state: "AppState") -> "PlotRect | None":
    """2D ImPlot pane: the optimizer's energy against step, live.

    Costs nothing to collect -- the server is already accumulating exactly this
    array to answer each status poll -- so a relaxation is watchable while it
    runs rather than only reportable once it lands.

    Plotted as an energy *drop* from the starting geometry rather than as absolute
    eV. The absolute number is dominated by a constant of tens or hundreds of eV
    that says nothing about the relaxation, and on that scale the part worth
    seeing is a flat line.
    """
    job = state.selected_remote_job()
    energies = job.trajectory() if job is not None else []

    if len(energies) < 2:
        imgui.text_disabled(relaxation_trace_status(state))
    else:
        imgui.text_disabled(f"{job.label}: {job.status_line()}")

    if not implot.begin_plot("Relaxation##relaxation_trace", size=(-1, -1)):
        return None
    implot.setup_axes("Optimizer step", "E - E0 (eV)")
    setup_two_d_legend()

    if len(energies) >= 2:
        values = np.asarray(energies, dtype=np.float64)
        delta = values - values[0]
        steps = np.arange(len(values), dtype=np.float64)

        # Refit only when the curve's shape changes, then leave the axes free to
        # pan and zoom -- the same contract the spin landscape offers.
        axis_key = (job.key, len(values), round(float(delta.min()), 9))
        if axis_key != state._relaxation_plot_axis_key:
            state._relaxation_plot_axis_key = axis_key
            y_lo, y_hi = padded_two_d_limits(float(delta.min()), 0.0)
            implot.setup_axis_limits(
                implot.ImAxis_.x1, -0.5, len(values) - 0.5, implot.Cond_.always
            )
            implot.setup_axis_limits(implot.ImAxis_.y1, y_lo, y_hi, implot.Cond_.always)

        color = imgui.ImVec4(*RELAXATION_TRACE_COLOR)
        spec = implot.Spec()
        spec.line_color = color
        spec.marker = implot.Marker_.circle
        spec.marker_size = 2.5
        spec.marker_fill_color = color
        spec.marker_line_color = color
        implot.plot_line(job.label, steps, delta, spec)

        # The latest point, ringed: on a live job this is the frontier, and it is
        # what the eye is looking for when the curve is still growing.
        latest = implot.Spec()
        latest.marker = implot.Marker_.circle
        latest.marker_size = 5.0
        latest.marker_fill_color = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
        latest.marker_line_color = color
        implot.plot_scatter(
            "##relaxation_latest",
            np.array([steps[-1]], dtype=np.float64),
            np.array([delta[-1]], dtype=np.float64),
            latest,
        )

    rect = current_plot_rect()
    implot.end_plot()
    return rect


def plot_live_energy(state: "AppState") -> "PlotRect | None":
    """2D ImPlot pane: CHGNet single-point energies, one point per structure.

    The bottom axis counts structures -- the last LIVE_ENERGY_MAX_POINTS that
    an energy came back for, oldest on the left -- rather than time, so the
    pane is a record of where the structure has been rather than a scrolling
    trace. Energy per atom on the vertical: single points asked for by hand
    share the pane with the live samples, and cells of different sizes only
    compare that way. Live samples and hand-run single points are two series,
    so the ones you asked for stand out.
    """
    points = state.live_energy_points
    if not points:
        if state.live_energy_enabled:
            imgui.text_disabled("Waiting for the first energy from the server...")
        else:
            imgui.text_disabled(
                "No energies yet. Run a single point from Calculate > Remote compute, "
                "or switch on \"Live CHGNet energy\" there."
            )
    else:
        latest = points[-1]
        pending = "  (computing...)" if state._live_energy_job is not None else ""
        source = f" [{latest.label}]" if latest.label else ""
        imgui.text_disabled(
            f"Latest: {latest.per_atom:.4f} eV/atom, {latest.energy:.4f} eV total"
            f"{source}{pending}"
        )
    if state.live_energy_enabled and state.live_energy_message:
        imgui.text_disabled(f"Live energy: {state.live_energy_message}")

    if not implot.begin_plot("CHGNet energy##live_energy", size=(-1, -1)):
        return None
    implot.setup_axes(f"Structure (last {LIVE_ENERGY_MAX_POINTS})", "E (eV/atom)")
    setup_two_d_legend()

    if points:
        indices = np.arange(1, len(points) + 1, dtype=np.float64)
        energies = np.array([sample.per_atom for sample in points], dtype=np.float64)
        # Always refit: the pane is live by definition.
        implot.setup_axis_limits(
            implot.ImAxis_.x1, 0.0, float(max(len(points), 2)) + 1.0, implot.Cond_.always
        )
        y_lo, y_hi = padded_two_d_limits(float(energies.min()), float(energies.max()))
        if y_hi - y_lo < 1e-4:
            y_lo, y_hi = y_lo - 1e-3, y_hi + 1e-3
        implot.setup_axis_limits(implot.ImAxis_.y1, y_lo, y_hi, implot.Cond_.always)

        color = imgui.ImVec4(*LIVE_ENERGY_COLOR)
        line = implot.Spec()
        line.line_color = color
        line.marker = implot.Marker_.none
        implot.plot_line("##live_energy_path", indices, energies, line)

        live_mask = np.array([sample.live for sample in points], dtype=bool)
        if live_mask.any():
            live_spec = implot.Spec()
            live_spec.marker = implot.Marker_.circle
            live_spec.marker_size = 3.0
            live_spec.marker_fill_color = color
            live_spec.marker_line_color = color
            implot.plot_scatter("Live", indices[live_mask], energies[live_mask], live_spec)
        if (~live_mask).any():
            manual = imgui.ImVec4(1.0, 0.85, 0.30, 1.0)
            manual_spec = implot.Spec()
            manual_spec.marker = implot.Marker_.square
            manual_spec.marker_size = 4.5
            manual_spec.marker_fill_color = manual
            manual_spec.marker_line_color = manual
            implot.plot_scatter(
                "Single point", indices[~live_mask], energies[~live_mask], manual_spec
            )

        latest = implot.Spec()
        latest.marker = implot.Marker_.circle
        latest.marker_size = 7.0
        latest.marker_fill_color = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
        latest.marker_line_color = imgui.ImVec4(1.0, 1.0, 1.0, 0.9)
        implot.plot_scatter(
            "##live_energy_latest",
            np.array([indices[-1]], dtype=np.float64),
            np.array([energies[-1]], dtype=np.float64),
            latest,
        )

        # Hovering a point names the structure it came from.
        if implot.is_plot_hovered():
            mouse = implot.get_plot_mouse_pos()
            nearest = int(round(mouse.x)) - 1
            if 0 <= nearest < len(points) and abs(mouse.x - (nearest + 1)) < 0.5:
                sample = points[nearest]
                kind = "live" if sample.live else "single point"
                imgui.set_tooltip(
                    f"#{nearest + 1}  {sample.label or '?'} ({kind})\n"
                    f"{sample.per_atom:.5f} eV/atom, {sample.energy:.4f} eV, "
                    f"{sample.atom_count} atoms"
                )

    rect = current_plot_rect()
    implot.end_plot()
    return rect


def spin_result_view_options(state: "AppState") -> None:
    """The view toggles at the top of the results panel.

    Two columns: what to draw about the spins on the left, which site roles to draw
    at all on the right. Drawn before any early return, so they stay reachable when
    there is nothing to show yet.
    """
    if imgui.begin_table("##spin_view_options", 2, imgui.TableFlags_.none.value):
        imgui.table_next_row()

        imgui.table_set_column_index(0)
        _, state.color_atoms_by_spin = imgui.checkbox(
            "Color atoms by spin", state.color_atoms_by_spin
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Draw magnetic atoms in the spin-up/spin-down colors instead of\n"
                "their element colors, and size each sphere by its moment: a\n"
                f"moment of {SPIN_RADIUS_REFERENCE_MOMENT:g} draws at the full element radius, so a\n"
                "spin-1 ion is a fifth the size of a high-spin Fe3+. Sites with\n"
                "no moment (O, La) drop out of the view entirely."
            )
        _, state.show_miller_planes = imgui.checkbox(
            "Draw ordering planes", state.show_miller_planes
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Sheets on the lattice planes the selected ordering alternates\n"
                "across, tinted with the spin each plane carries."
            )
        _, state.show_spin_defect_rings = imgui.checkbox(
            "Ring deviating sites", state.show_spin_defect_rings
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Circle every magnetic site whose spin disagrees with the ideal\n"
                "ordering the configuration was matched to."
            )

        imgui.table_set_column_index(1)
        imgui.text_disabled("Sites drawn")
        _, state.show_a_sites = imgui.checkbox("A", state.show_a_sites)
        imgui.same_line()
        _, state.show_b_sites = imgui.checkbox("B", state.show_b_sites)
        imgui.same_line()
        _, state.show_x_sites = imgui.checkbox("X", state.show_x_sites)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Which site roles the 3D view draws. A structure loaded from a file\n"
                "records no roles, so all of its atoms are drawn regardless."
            )
        # The octahedra are the other half of "what is in the cell", so they sit with
        # the site toggles rather than off in the Rendering section.
        # Strict: the cages come from the build's own ideal octahedra, so on a
        # relaxed structure they would sit where the atoms no longer are.
        octahedra_available = state.focus_has_generated_provenance()
        if not octahedra_available:
            imgui.begin_disabled()
        _, state.show_octahedra = imgui.checkbox("Octahedra", state.show_octahedra)
        if not octahedra_available:
            imgui.end_disabled()
            if imgui.is_item_hovered():
                imgui.set_tooltip(
                    "Octahedra are drawn from the builder's lattice, which a\n"
                    "structure loaded from a file does not carry."
                )

        imgui.spacing()
        interactive_changed, state.update_spin_energies_interactively = imgui.checkbox(
            "Live energies", state.update_spin_energies_interactively
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Re-energize the spin landscape on every builder edit.\n"
                "That rebuilds the oxidation assignments and the exchange matrix,\n"
                "which costs tens to hundreds of milliseconds on a large cell -- so\n"
                f"it pauses itself below {AUTO_SPIN_UPDATE_MIN_FPS:g} fps and resumes\n"
                f"once the view is back above {AUTO_SPIN_UPDATE_RESUME_FPS:g}, and\n"
                "edits made while it is paused just mark the energies stale."
            )
        if interactive_changed and state.update_spin_energies_interactively:
            state.refresh_spin_energies()
        # Beside the checkbox, in space that is reserved whether or not there is
        # anything to say. The readout appears exactly when the frame rate is under
        # strain, which is the worst moment for it to reflow the panel it is
        # reporting on -- so the gap is held open with a blank of the same width.
        imgui.same_line()
        paused = (
            state.update_spin_energies_interactively
            and not state.interactive_updates_live()
        )
        imgui.text_disabled(
            f"{current_framerate():.0f} fps" if paused else PAUSED_FPS_PLACEHOLDER
        )
        if paused and imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Paused: below {AUTO_SPIN_UPDATE_MIN_FPS:g} fps.\n"
                f"Resumes above {AUTO_SPIN_UPDATE_RESUME_FPS:g}."
            )
        imgui.end_table()

    if state.spin_energies_stale:
        imgui.push_style_color(imgui.Col_.text, (0.95, 0.75, 0.35, 1.0))
        imgui.text_wrapped(
            "Energies are stale: the structure has been edited since they were last "
            "updated."
        )
        imgui.pop_style_color()
        if imgui.button("Refresh energies", size=(160, 0)):
            state.refresh_spin_energies()
    imgui.separator()


def gui_custom_spin_pattern(state: "AppState") -> None:
    """Enter an ordering by hand: a plane family and a sign string across it.

    The same two things that define every canonical ordering -- `G` is "flip on
    successive (111) planes", `A(c)` is the same on (001) -- so an ordering entered
    here is scored, classified, drawn and saved exactly as the built-in ones are,
    rather than being a second kind of configuration the rest of the UI has to know
    about.
    """
    if not imgui.tree_node_ex("Custom ordering"):
        return

    imgui.text_disabled(
        "A plane family and the signs to repeat across it. (111) +- is G."
    )
    imgui.push_item_width(150.0)
    _, state.custom_pattern_miller = imgui.input_int3(
        "Plane (hkl)##custom_pattern_miller", state.custom_pattern_miller
    )
    changed, signs = imgui.input_text(
        "Signs##custom_pattern_signs", state.custom_pattern_signs
    )
    if changed:
        # Filtered as it is typed rather than rejected on submit, so the box can
        # never hold something that is not a pattern.
        state.custom_pattern_signs = "".join(
            character for character in signs if character in "+-"
        )[:MAX_CUSTOM_PATTERN_PERIOD]
    imgui.pop_item_width()

    sublattice = state.magnetic_sublattice()
    can_add = bool(state.custom_pattern_signs) and sublattice is not None
    if not can_add:
        imgui.begin_disabled()
    if imgui.button("Add ordering", size=(150, 0)):
        state.add_custom_spin_pattern(
            state.custom_pattern_miller, state.custom_pattern_signs
        )
    if not can_add:
        imgui.end_disabled()

    if sublattice is not None and any(state.custom_pattern_miller):
        available = plane_count(
            sublattice.lattice_coords, tuple(state.custom_pattern_miller)
        )
        imgui.same_line()
        imgui.text_disabled(f"{available} planes available")

    if state.custom_pattern_message:
        imgui.text_wrapped(state.custom_pattern_message)

    for label in list(state.custom_spin_patterns):
        if imgui.small_button(f"x##remove_custom_{label}"):
            state.remove_custom_spin_pattern(label)
        imgui.same_line()
        imgui.text(label)

    imgui.tree_pop()


def atoms_panel_compensation_line(state: AppState) -> Tuple[str, tuple | None] | None:
    """The charge tally shown under the atom list, or None when there is nothing to say.

    ``(message, colour)``; colour None means the balanced, dimmed reading. Only a
    built structure has a stoichiometric reference to hold the elements against.
    """
    focus = state.focus
    if focus is None or state.atom_edit_mode() != "builder" or not state.builder_defects():
        return None
    try:
        reference_labels = state.atomic_labels_for_build(
            state.generated_perovskite(), periodic=state.treat_as_periodic
        )
    except ValueError:
        return None
    deficit, message = compensation_hint(reference_labels, focus.atomic_labels)
    return message, (None if deficit == 0 else (0.95, 0.75, 0.35, 1.0))


def atoms_panel_oxidation_summary(state: AppState) -> None:
    """What the whole cell adds up to, and what has been set by hand.

    The model's assignment is taken rather than chosen -- always the
    lowest-energy one it ranked -- so the per-atom boxes below are the place to
    disagree with it, and this line is what the disagreement costs.
    """
    assignment = state.selected_oxidation_assignment()
    if assignment is None or state.magnetic_analysis_structure is not state.focus:
        imgui.text_disabled(state.baseline_status or "No oxidation-state assignment yet.")
        return
    imgui.text_wrapped(format_oxidation_distribution(assignment.distributions))
    edit_count = len(state.oxidation_overrides)
    predicted = state.predicted_oxidation_assignment()
    if edit_count == 0:
        if predicted is not None:
            imgui.text_disabled(f"Lowest-energy assignment, E={predicted.total_energy:.3f}")
    else:
        edits = f"{edit_count} state{'' if edit_count == 1 else 's'} set by hand"
        if state.oxidation_solve_unbalanced:
            imgui.text_disabled(f"{edits}; no balanced completion, applied as set")
        elif predicted is not None:
            delta = assignment.total_energy - predicted.total_energy
            imgui.text_disabled(
                f"{edits}; re-solved around them, E={assignment.total_energy:.3f} "
                f"({delta:+.3f} vs model)"
            )
        else:
            imgui.text_disabled(edits)
        imgui.same_line()
        if imgui.small_button("Revert all##oxidation_overrides"):
            state.clear_oxidation_overrides()
    net_charge = int(np.sum(np.asarray(assignment.site_oxidation_states, dtype=int)))
    if net_charge == int(state.magnetic_net_charge):
        imgui.text_disabled(f"Net charge: {net_charge:+d}")
    else:
        imgui.push_style_color(imgui.Col_.text, OXIDATION_UNBALANCED_COLOR)
        imgui.text(
            f"Net charge: {net_charge:+d} (cell is set to {int(state.magnetic_net_charge):+d})"
        )
        imgui.pop_style_color()


def atoms_panel_slab_controls(state: AppState) -> None:
    """The selection slab: its lattice direction, thickness and position.

    The direction is three integers in the *lattice* basis -- ``[1 0 0]`` is
    along a, ``[1 1 0]`` toward the edge between a and b, ``[1 1 1]`` toward the
    far vertex -- so the same numbers mean the same thing in a triclinic cell,
    where "the z axis" would not. The slab is also draggable in the 3D view; the
    slider here is the same offset, for when a precise position is wanted.
    There is no on/off switch: touching a control brings the slab up, and
    "Reset selection", down by the list the slab fills, puts it away.
    """
    slab = state.selection_slab
    imgui.text("Slab")
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Narrow the cell to a layer through it: the atoms inside stay lit\n"
            "and clickable, the rest fade to context. Touching any of these\n"
            "controls brings the slab up; Reset selection puts it away.\n"
            "Drag the slab in the 3D view, or slide it here."
        )
    imgui.same_line()
    imgui.text_disabled("[u v w]")
    imgui.same_line()
    imgui.push_item_width(120.0)
    direction_changed, direction = imgui.input_int3(
        "##slab_direction", list(slab.direction_tuple())
    )
    imgui.pop_item_width()
    if direction_changed:
        slab.direction = (int(direction[0]), int(direction[1]), int(direction[2]))
        state.slab_enabled = True
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "Normal of the slab as a lattice direction u*a + v*b + w*c:\n"
            "[1 0 0] along a, [1 1 0] toward an edge, [1 1 1] toward a vertex."
        )
    imgui.same_line()
    imgui.text_disabled("thickness")
    imgui.same_line()
    imgui.push_item_width(80.0)
    thickness_changed, thickness = imgui.drag_float(
        "##slab_thickness", float(slab.thickness), 0.05, 0.0, 50.0, "%.2f A"
    )
    imgui.pop_item_width()
    if thickness_changed:
        slab.thickness = max(0.0, float(thickness))
        state.slab_enabled = True

    imgui.push_item_width(max(120.0, imgui.get_content_region_avail().x))
    offset_changed, offset = imgui.slider_float(
        "##slab_offset", float(slab.offset), 0.0, 1.0, "position %.3f"
    )
    imgui.pop_item_width()
    if offset_changed:
        slab.offset = float(min(1.0, max(0.0, offset)))
        state.slab_enabled = True
    if imgui.is_item_hovered():
        imgui.set_tooltip("Where the slab sits along its normal: 0 is one face of the cell, 1 the other.")
    if not any(slab.direction_tuple()):
        imgui.push_style_color(imgui.Col_.text, UNKNOWN_ELEMENT_COLOR)
        imgui.text("[0 0 0] is not a direction")
        imgui.pop_style_color()


def atoms_panel_row(state: AppState, row: AtomRow, *, selected: bool, editable: bool) -> None:
    """One row of the atom table: number, element box, oxidation state, actions.

    The element box *is* the edit: type a symbol to substitute, empty it to
    vacate. The buttons say the same things a click away -- ``x`` empties the
    site, ``+H`` hangs a proton on an oxygen, ``Restore`` fills a vacancy back
    in -- because "delete this atom" is a thing people reach for a button to do.
    """
    imgui.table_next_row()
    imgui.table_set_column_index(0)
    label = f"{row.index + 1}" if not row.vacant else "-"
    clicked, _ = imgui.selectable(
        f"{label}##row",
        selected,
        imgui.SelectableFlags_.span_all_columns | imgui.SelectableFlags_.allow_overlap,
    )
    if imgui.is_item_hovered():
        state.atom_hover_ref = row.ref
    if clicked:
        state.toggle_atom_selection(row.ref, additive=imgui.get_io().key_shift)

    imgui.table_set_column_index(1)
    if row.vacant:
        imgui.push_style_color(imgui.Col_.text, VACANCY_RENDER_COLOR)
        imgui.text(f"vacancy ({row.ideal_element})")
        imgui.pop_style_color()
    elif editable and not (row.role == "H" and state.atom_edit_mode() == "builder"):
        text = state.atom_edit_text if state.atom_edit_ref == row.ref else row.element
        imgui.push_item_width(48.0)
        _, text = imgui.input_text(
            "##element",
            text,
            imgui.InputTextFlags_.chars_no_blank | imgui.InputTextFlags_.auto_select_all,
        )
        imgui.pop_item_width()
        if imgui.is_item_active():
            state.atom_edit_ref = row.ref
            state.atom_edit_text = text
        if imgui.is_item_deactivated_after_edit():
            state.atom_edit_ref = None
            state.set_atom_element(row, text)
        elif imgui.is_item_deactivated() and state.atom_edit_ref == row.ref:
            state.atom_edit_ref = None
        if row.edited:
            imgui.same_line()
            imgui.push_style_color(imgui.Col_.text, OXIDATION_EDITED_COLOR)
            imgui.text(f"(was {row.ideal_element})" if row.ideal_element else "*")
            imgui.pop_style_color()
        else:
            note, color = element_box_note(row.element)
            if note:
                imgui.same_line()
                imgui.push_style_color(imgui.Col_.text, color)
                imgui.text(note)
                imgui.pop_style_color()
    else:
        imgui.text(row.element)
        if row.role == "H" and row.host_key is not None:
            imgui.same_line()
            imgui.text_disabled(f"on {site_key_display(row.host_key)}")

    imgui.table_set_column_index(2)
    if not row.vacant and state.magnetic_analysis_structure is state.focus:
        charge = state.site_oxidation_state(row.index)
        if charge is not None:
            edited = state.site_oxidation_is_edited(row.index)
            staged = (
                state.oxidation_edit_value
                if state.oxidation_edit_site == row.index
                else int(charge)
            )
            if edited:
                imgui.push_style_color(imgui.Col_.text, OXIDATION_EDITED_COLOR)
            imgui.push_item_width(52.0)
            # No ``enter_returns_true`` here: ImGui asserts on it in
            # ``InputScalar``. ``is_item_deactivated_after_edit`` fires once, on
            # Enter or on leaving the field, which is when the exchange matrix
            # should be rebuilt -- not on every digit typed on the way.
            _, value = imgui.input_int("##oxidation", staged, 0, 0)
            imgui.pop_item_width()
            if edited:
                imgui.pop_style_color()
            if imgui.is_item_active():
                state.oxidation_edit_site = row.index
                state.oxidation_edit_value = int(value)
            if imgui.is_item_deactivated_after_edit():
                state.oxidation_edit_site = -1
                state.set_site_oxidation_state(row.index, int(value))
            if edited:
                imgui.same_line()
                if imgui.small_button("Revert##oxidation_revert"):
                    state.revert_site_oxidation_state(row.index)
        else:
            imgui.text_disabled("-")
    else:
        imgui.text_disabled("-")

    imgui.table_set_column_index(3)
    if not editable or not state.atom_count_edits_available():
        return
    if row.vacant:
        if imgui.small_button("Restore##restore"):
            state.set_atom_element(row, row.ideal_element)
        return
    if imgui.small_button("x##vacate"):
        state.set_atom_element(row, "")
    if imgui.is_item_hovered():
        imgui.set_tooltip("Remove this atom, leaving a vacancy.")
    if row.element == "O":
        imgui.same_line()
        if imgui.small_button("+H##proton"):
            state.add_proton_to_atom(row)
        if imgui.is_item_hovered():
            imgui.set_tooltip("Attach a proton to this oxygen.")


# How many rows the picked list shows before it starts scrolling. It is the
# working set -- a handful of atoms held together for an edit -- so it stays
# short enough that the active list underneath is still worth reading.
SELECTED_LIST_MAX_ROWS = 6


def atoms_panel_table(
    state: AppState,
    table_id: str,
    listed: List[AtomRow],
    height: float,
    editable: bool,
    *,
    picked: bool,
) -> None:
    """One of the Selection panel's two atom lists, drawn into ``height``.

    Both lists are the same table of the same rows; ``picked`` only says
    whether the rows in it are the hand picks, which are drawn highlighted.
    """
    flags = (
        imgui.TableFlags_.scroll_y
        | imgui.TableFlags_.row_bg
        | imgui.TableFlags_.borders_inner_v
        | imgui.TableFlags_.sizing_fixed_fit
    )
    if not imgui.begin_table(table_id, 4, flags, imgui.ImVec2(0.0, max(0.0, height))):
        return
    imgui.table_setup_scroll_freeze(0, 1)
    imgui.table_setup_column("#", imgui.TableColumnFlags_.width_fixed, 44.0)
    imgui.table_setup_column("Element", imgui.TableColumnFlags_.width_stretch)
    imgui.table_setup_column("Ox. state", imgui.TableColumnFlags_.width_fixed, 120.0)
    imgui.table_setup_column("", imgui.TableColumnFlags_.width_fixed, 74.0)
    imgui.table_headers_row()
    # A ListClipper builds only the rows on screen, so a thousand-atom cell
    # costs the visible dozen rather than the cell. Row heights are uniform,
    # which is what the clipper needs.
    clipper = imgui.ListClipper()
    clipper.begin(len(listed))
    while clipper.step():
        for position in range(clipper.display_start, clipper.display_end):
            row = listed[position]
            imgui.push_id(str(row.ref))
            atoms_panel_row(state, row, selected=picked, editable=editable)
            imgui.pop_id()
    imgui.end_table()


def gui_atoms() -> None:
    """The Selection panel: every atom of the focus, or just the selection.

    One kind of row serves both the oxidation states and the defects, because
    the two were always about the same object: an atom, what it is, and what
    charge it carries. With nothing selected the list is the whole cell;
    switching the slab on narrows it to what the slab holds, which is also what
    the 3D view will let you click. Clicking an atom -- in the 3D view or in the
    list -- picks it out into a second, shorter list above, which keeps its
    atoms while the slab moves on, so a set can be gathered up and edited as a
    group. Picking changes no more than that: the cell stays on screen whole. Editing a row's element substitutes the
    atom -- or, emptied, vacates it -- and an oxygen's ``+H`` hangs a proton on
    it. On a built structure these become defect entries and the cell
    regenerates; on a loaded one the atoms are edited in place.
    """
    state = APP_STATE
    focus = state.focus
    if focus is None:
        imgui.text_wrapped("No structure is active.")
        return
    state.ensure_defect_entries()
    state.prune_atom_selection()
    editable = state.atom_edit_mode() != "none"

    atoms_panel_slab_controls(state)
    imgui.separator()
    atoms_panel_oxidation_summary(state)

    rows = state.atom_table()
    picked = state.picked_rows()
    picked_refs = {row.ref for row in picked}
    # The two lists are disjoint: an atom the slab holds moves up into the
    # picked list the moment it is clicked, rather than being listed twice.
    active = (
        [row for row in state.active_rows() if row.ref not in picked_refs]
        if state.selection_active()
        else rows
    )
    if state.selection_active():
        total = len(picked) + len(active)
        imgui.text(
            f"Selection: {total} of {len(rows)} site{'' if len(rows) == 1 else 's'}"
        )
    else:
        imgui.text(f"All {len(rows)} site{'' if len(rows) == 1 else 's'}")
        imgui.same_line()
        imgui.text_disabled("(click atoms in the 3D view, or switch the slab on, to select)")

    for message, color in (
        (state.atom_edit_message, UNKNOWN_ELEMENT_COLOR),
        (state.oxidation_edit_message, OXIDATION_UNBALANCED_COLOR),
        (state.defect_message, (0.95, 0.35, 0.35, 1.0)),
    ):
        if message:
            imgui.push_style_color(imgui.Col_.text, color)
            imgui.text_wrapped(message)
            imgui.pop_style_color()
    if state.focus_is_relaxed_from_builder():
        imgui.text_disabled(
            "Relaxed structure: atoms can be relabelled, but not removed or added, "
            "so the site numbering the builder parameters describe stays valid."
        )

    footer = atoms_panel_compensation_line(state)
    style = imgui.get_style()
    reserved = 0.0
    if footer is not None:
        reserved = (
            imgui.calc_text_size(
                footer[0], wrap_width=max(1.0, imgui.get_content_region_avail().x)
            ).y
            + style.item_spacing.y * 2.0
        )
    available = max(
        imgui.get_frame_height_with_spacing() * 3.0,
        imgui.get_content_region_avail().y - reserved,
    )
    # The picked list is sized to what it holds, up to a few rows, and never
    # takes more than half of what is left: it is the short working list, and
    # the active one under it is the one being read from. Whatever headings the
    # split costs come off the top, so the lists still end where one did.
    row_height = imgui.get_frame_height_with_spacing()
    # The lower list is headed only when the slab is filling it: that is when it
    # needs telling apart from the picks above, and where the switch that puts
    # the slab away belongs. With the slab off there is nothing below the picks
    # to head -- the whole cell is the picks' surroundings, not a second list.
    headings = (row_height if picked else 0.0) + (
        row_height if state.slab_enabled else 0.0
    )
    available = max(row_height * 2.0, available - headings)
    picked_height = 0.0
    if picked:
        wanted = row_height * (min(len(picked), SELECTED_LIST_MAX_ROWS) + 1.6)
        picked_height = min(wanted, available if not active else available * 0.5)

        imgui.text(f"Selected: {len(picked)}")
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Atoms clicked by hand. They stay here while the slab moves,\n"
                "so a set can be gathered up and edited together. Click a row\n"
                "again -- here or in the 3D view -- to let it go."
            )
        imgui.same_line()
        if imgui.small_button("Clear picks##atom_picks"):
            state.clear_picked_atoms()
        if imgui.is_item_hovered():
            imgui.set_tooltip("Drop the hand picks; the slab stays where it is.")
        atoms_panel_table(
            state, "##picked_atoms_table", picked, picked_height, editable, picked=True
        )
    if state.slab_enabled:
        imgui.text_disabled(f"Active: {len(active)}")
        # The switch for the slab sits on the list the slab fills, which is the
        # thing it changes; the picks above it are left alone.
        imgui.same_line()
        if imgui.small_button("Reset selection##slab_reset"):
            state.reset_slab_selection()
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Put the slab away: the whole cell is drawn, and clickable,\n"
                "again. The atoms picked out above stay picked."
            )
    # Nothing but hand picks: they are already listed above, and an empty table
    # under them would only be a header saying so.
    if active or not picked:
        atoms_panel_table(
            state,
            "##atoms_table",
            active,
            available - picked_height,
            editable,
            picked=False,
        )

    if footer is not None:
        message, color = footer
        imgui.push_style_color(
            imgui.Col_.text,
            color if color is not None else style.color_(imgui.Col_.text_disabled),
        )
        imgui.text_wrapped(message)
        imgui.pop_style_color()

    # A builder edit made here regenerates the focus the same way one made in
    # the Controls panel does.
    state.regenerate_focus_from_builder_if_changed()
    state.sync_active_structure()


def gui_calculation_output() -> None:
    state = APP_STATE

    imgui.text("Spin Visualization")
    imgui.separator()
    spin_result_view_options(state)

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

    imgui.text(f"Structure: {state.magnetic_result_structure_name}")

    selected_assignment = state.selected_oxidation_assignment()
    result_structure = state.magnetic_result_structure
    if selected_assignment is not None and result_structure is not None:
        imgui.text(
            f"Exchange matrix: {state.magnetic_j_matrix.shape[0]} x "
            f"{state.magnetic_j_matrix.shape[1]}"
            if state.magnetic_j_matrix.size
            else "Exchange matrix unavailable"
        )

    # Solving is what the oxidation states are *for*, so the button sits with them
    # rather than across the rule from them.
    if imgui.button("Solve current oxidation states", size=(220, 0)):
        state.run_selected_oxidation_assignment(force=True)

    imgui.spacing()
    imgui.separator()

    # --- Spin solver section ---
    all_states = state.displayed_spin_configs()

    selected_config = None
    selected_moments = None
    if selected_assignment is None:
        imgui.text_wrapped("No oxidation states have been assigned.")
    elif not all_states:
        imgui.text_wrapped(state.magnetic_spin_status)
    else:
        selected_config = state.selected_spin_config()
        if selected_config is not None and result_structure is not None:
            selected_moments = state.expand_spin_moments_to_structure(
                selected_config.all_moments, result_structure
            )

        imgui.spacing()
        # Read once and passed into every row: the moments and the cell count are the
        # same for the whole landscape, and looking them up per row would repeat an
        # assignment lookup and a grid resolve for each line on screen.
        basis = state.magnetization_basis()

        # The list first, then what is selected in it, then what you can do with the
        # selection. Reading order follows the act: pick a configuration, see what it
        # is, save it.
        list_height = max(120.0, imgui.get_content_region_avail().y * 0.4)
        imgui.text(f"Spin configurations ({len(all_states)})")
        if imgui.begin_child("##spin_config_list", (0.0, list_height), True):
            state.selected_spin_config_index = min(
                max(state.selected_spin_config_index, 0),
                max(len(all_states) - 1, 0),
            )
            config_labels = state.spin_classification_descriptions()
            # Clipped for the same reason as the site list: the plotted-configuration
            # cap can be raised well past what fits on screen.
            clipper = imgui.ListClipper()
            clipper.begin(len(all_states))
            while clipper.step():
                for index in range(clipper.display_start, clipper.display_end):
                    config = all_states[index]
                    ordering = (
                        config_labels[index] if index < len(config_labels) else "Other"
                    )
                    degeneracy = f" x{config.degeneracy}" if config.degeneracy > 1 else ""
                    moment, _ = state.config_magnetization(config, basis)
                    label = (
                        f"{index + 1:>3}. E={config.energy:.6f}  "
                        f"M={moment:.3f}  {ordering}{degeneracy}"
                        f"##spin_config_{index}"
                    )
                    clicked, _ = imgui.selectable(
                        label, state.selected_spin_config_index == index
                    )
                    if clicked:
                        state.selected_spin_config_index = index
        imgui.end_child()

        if selected_config is None:
            imgui.text_wrapped("No spin configurations were returned.")
        else:
            imgui.text(f"Energy: {selected_config.energy:.6f}")
            moment, unit = state.config_magnetization(selected_config, basis)
            imgui.text(f"Magnetization: {moment:.3f} {unit}".rstrip())
            imgui.text(f"Ordering: {state.described_config(selected_config)}")
            selected_match = state.match_for_config(selected_config)
            if selected_match is not None and not selected_match.is_exact:
                imgui.text_disabled(
                    f"{selected_match.defect_count} of "
                    f"{int(np.count_nonzero(np.asarray(selected_config.all_moments)))} "
                    f"magnetic sites disagree with ideal "
                    f"{selected_match.pattern.plane_label}"
                )
            if selected_config.degeneracy > 1:
                imgui.text(
                    f"Degeneracy: {selected_config.degeneracy} configurations "
                    "share this energy"
                )

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
        gui_custom_spin_pattern(state)

    # The per-site oxidation states and moments used to be listed here, one row per
    # atom. The rows are gone: they made you match a row against the structure by
    # eye. Per-site oxidation and moment now come off the atom itself -- hover it in
    # the 3D view (see ``site_hover_tooltip``), and click it to edit its state in the
    # Oxidation states section above. The assignment's distribution and model energy
    # live there too.


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


def structure_plot_flags(*, show_legend: bool) -> int:
    """ImPlot3D flags for the 3D structure view.

    The view box is a cube centred on the structure and re-applied every frame, which
    keeps the cell centred but also overrides ImPlot3D's own pan and zoom (both work by
    moving the axis limits). Both are therefore disabled here: panning is gone by
    design, and zoom is reimplemented as a scale on the computed box so that it
    composes with the centring instead of fighting it.

    Rotation is ours too, for a different reason: ImPlot3D's is a turntable pinned to the
    plot's c axis, which is degenerate in exactly the views the a/b/c buttons aim at. See
    ``rotation_after_drag``. What is given up is ImPlot3D's double-right-click-a-face
    alignment, which is what those buttons now do, and its reset, which
    ``apply_structure_rotation`` keeps on the same double right click.

    The hovered-coordinate readout in the corner goes with ``no_mouse_text``: it reports
    a point on the box's back faces rather than anything under the cursor, so on a
    structure it is noise.
    """
    flags = (
        implot3d.Flags_.equal.value
        | implot3d.Flags_.no_pan.value
        | implot3d.Flags_.no_zoom.value
        | implot3d.Flags_.no_rotate.value
        | implot3d.Flags_.no_mouse_text.value
    )
    if not show_legend:
        flags |= implot3d.Flags_.no_legend.value
    return flags


def zoom_after_wheel(current: float, wheel: float) -> float:
    """Zoom factor after ``wheel`` notches of scrolling, clamped to the usable range."""
    if abs(wheel) < 1e-6:
        return current
    low, high = STRUCTURE_ZOOM_RANGE
    return float(np.clip(current * (1.15**wheel), low, high))


def apply_structure_zoom(state: "AppState", plot_rect_min, plot_rect_max) -> None:
    """Scroll over the 3D plot zooms by scaling the padding around the structure."""
    if not imgui.is_mouse_hovering_rect(plot_rect_min, plot_rect_max):
        return
    state.structure_zoom = zoom_after_wheel(
        state.structure_zoom, float(imgui.get_io().mouse_wheel)
    )


def apply_structure_rotation(state: "AppState", plot_rect_min, plot_rect_max) -> None:
    """Right-drag over the 3D plot orbits the structure; double right-click resets it.

    The drag has to be latched: once it starts inside the plot it keeps going wherever
    the cursor wanders, and a right-press that started elsewhere must not grab the view.
    """
    right = imgui.MouseButton_.right
    hovering = imgui.is_mouse_hovering_rect(plot_rect_min, plot_rect_max)
    if hovering and imgui.is_mouse_double_clicked(right):
        state.structure_rotation_target = DEFAULT_STRUCTURE_ROTATION
        state._structure_drag_active = False
        return
    if not imgui.is_mouse_down(right):
        state._structure_drag_active = False
        return
    if hovering and imgui.is_mouse_clicked(right):
        state._structure_drag_active = True
    if not state._structure_drag_active:
        return

    delta = imgui.get_io().mouse_delta
    if abs(delta.x) < 1e-6 and abs(delta.y) < 1e-6:
        return
    # Taking hold of the view abandons a swing still in flight, rather than letting the
    # two fight over the same quaternion.
    state.structure_rotation_target = None
    state.structure_rotation = rotation_after_drag(
        state.structure_rotation, float(delta.x), float(delta.y)
    )


def advance_structure_alignment(state: "AppState") -> None:
    """Ease the view towards the pose an a/b/c button (or the reset) asked for."""
    target = state.structure_rotation_target
    if target is None:
        return
    state.structure_rotation, arrived = rotation_after_alignment_step(
        state.structure_rotation, target, float(imgui.get_io().delta_time)
    )
    if arrived:
        state.structure_rotation_target = None


def spin_plot_categories(state: "AppState") -> List[str]:
    """Legend order for the energy scatter: canonical, then hand-entered, then Other.

    Custom orderings go after the canonical ones and before "Other" so the legend
    reads as "the known orderings, then the ones you added". A custom label that
    happens to be canonical in plane notation is not repeated -- the classifier
    reports those under the classical name.
    """
    categories = list(SPIN_PLOT_CATEGORIES[:-1])
    seen = set(categories)
    for label in state.custom_spin_patterns:
        if label not in seen:
            seen.add(label)
            categories.append(label)
    categories.append("Other")
    return categories


def spin_plot_category(label: str, categories: Sequence[str]) -> str:
    """Scatter-plot category for a label; anything unlisted renders as Other."""
    return label if label in categories else "Other"


def spin_class_color(
    state: "AppState", category: str
) -> Tuple[float, float, float, float]:
    """Colour for one energy-scatter category.

    Canonical orderings keep their fixed colours. A hand-entered one takes the next
    colour from a separate palette, chosen by its position in the user's own list so
    that it does not move when another ordering is added above it -- and drawn from
    hues the canonical set leaves free, so a custom ordering never reads as an A or
    a C at a glance.
    """
    fixed = SPIN_CLASS_COLORS.get(category)
    if fixed is not None:
        return fixed
    try:
        rank = state.custom_spin_patterns.index(category)
    except ValueError:
        return SPIN_CLASS_COLORS["Other"]
    return CUSTOM_SPIN_CLASS_COLORS[rank % len(CUSTOM_SPIN_CLASS_COLORS)]


#: Top-left and bottom-right of a 2D plot's data area, in screen pixels.
PlotRect = Tuple[Tuple[float, float], Tuple[float, float]]


def current_plot_rect() -> PlotRect:
    """The open plot's data area as ((x0, y0), (x1, y1)). Valid only inside a plot."""
    pos, size = implot.get_plot_pos(), implot.get_plot_size()
    return ((pos.x, pos.y), (pos.x + size.x, pos.y + size.y))


def padded_two_d_limits(
    low: float, high: float, bottom: float = TWO_D_BOTTOM_HEADROOM
) -> Tuple[float, float]:
    """Y limits for a 2D plot, with the top left clear for the corner dropdowns.

    The extra headroom above is not symmetry-breaking for its own sake: the plot
    pickers float inside the plot's top corners, and without it the tallest bar or
    the highest-energy point sits underneath them. ``bottom`` is the matching margin
    below, which the exchange plot widens because all of its bars stand on one line.
    """
    span = high - low
    if span <= 1e-12:
        return low - 0.5, high + 0.5
    return low - span * bottom, high + span * TWO_D_TOP_HEADROOM


def setup_two_d_legend() -> None:
    """Put the legend in a strip above the axes rather than over the data.

    Inside the plot widget but outside the plot area, so it never covers a point or
    a bar and never collides with the dropdowns floating in the plot's corners.
    """
    implot.setup_legend(
        implot.Location_.north,
        implot.LegendFlags_.outside | implot.LegendFlags_.horizontal,
    )


def exchange_pair_label(element_a: str, element_b: str) -> str:
    """Legend/category name for a coupled pair of metals, order-independent."""
    first, second = sorted((element_a, element_b))
    return f"{first} - {second}"


@lru_cache(maxsize=256)
def exchange_pair_color(category: str, rank: int) -> Tuple[float, float, float, float]:
    """Colour for one element-pair category in the exchange plot.

    The base is the mean of the two elements' 3D-view colours, so a bar and the
    atoms it couples read as the same chemistry. Blending alone is not enough to
    separate categories -- Fe-Co and Fe-Ni land almost on top of each other -- so a
    lightness offset cycling over three steps is applied by the category's rank in
    the (sorted, hence stable) category list. Three steps because a structure with
    more than three distinct metal pairings is already past what colour can carry.
    """
    elements = [part.strip() for part in category.split("-")]
    colors = [
        ELEMENT_RENDER_COLORS.get(element, DEFAULT_ELEMENT_RENDER_COLOR)
        for element in elements
    ] or [DEFAULT_ELEMENT_RENDER_COLOR]
    base = [sum(channel) / len(colors) for channel in zip(*colors)]
    # Toward black or toward white, leaving the first category at the true blend.
    shift = (0.0, -0.28, 0.30)[rank % 3]
    shifted = [
        value + shift * (1.0 - value) if shift > 0 else value * (1.0 + shift)
        for value in base[:3]
    ]
    return (
        min(max(shifted[0], 0.0), 1.0),
        min(max(shifted[1], 0.0), 1.0),
        min(max(shifted[2], 0.0), 1.0),
        1.0,
    )


def exchange_site_label(state: "AppState", atom_index: int) -> str:
    """An atom as ``Fe12`` -- element symbol plus its index in the analysed structure.

    The index, not the oxidation state, because this names the *individual* atom: it
    labels the bars, where two sites of the same ion have to be told apart.
    """
    structure = state.magnetic_analysis_structure
    if structure is None or not 0 <= atom_index < structure.atom_count:
        return f"#{atom_index}"
    return f"{structure.element_symbols()[atom_index]}{atom_index}"


def exchange_ion_label(state: "AppState", atom_index: int) -> str:
    """An atom as ``Fe(3+)`` -- the ion, for the 3D hover.

    What identifies an atom under the cursor is the one already under the cursor;
    its index says nothing you cannot see. Its oxidation state is what sets its
    d-shell, and so its couplings. Falls back to the bar label where no assignment
    is available to read a charge from.
    """
    structure = state.magnetic_analysis_structure
    if structure is None or not 0 <= atom_index < structure.atom_count:
        return f"#{atom_index}"
    oxidation_states = structure_site_oxidation_states(state, structure)
    if oxidation_states is None or atom_index >= len(oxidation_states):
        return exchange_site_label(state, atom_index)
    charge = int(oxidation_states[atom_index])
    symbol = structure.element_symbols()[atom_index]
    return f"{symbol}({abs(charge)}{'+' if charge >= 0 else '-'})"


def exchange_site_moments(state: "AppState") -> np.ndarray | None:
    """Unit moment per atom for the selected configuration, or None.

    Normalised because only the relative orientation matters here: whether a
    coupling is satisfied is the sign of ``J * (m_i . m_j)``, and the magnitude is
    already inside J.
    """
    structure = state.magnetic_analysis_structure
    config = state.selected_spin_config()
    if structure is None or config is None:
        return None
    vectors = state.expand_spin_moments_to_structure(config.all_moments, structure)
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.shape != (structure.atom_count, 3):
        return None
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-12)


def exchange_pair_frustration(
    pair: PairCoupling, moments: np.ndarray | None
) -> float:
    """``J * (m_i . m_j)``: the pair's energy contribution in the model convention.

    Positive is frustrated -- the coupling wants the opposite of what the selected
    configuration does with it. Zero when either site carries no moment, or when no
    configuration is selected. This is the only part of the coupling plot that
    depends on the configuration: J itself does not.
    """
    if moments is None:
        return 0.0
    if not (0 <= pair.site_i < len(moments) and 0 <= pair.site_j < len(moments)):
        return 0.0
    return float(pair.j_eff * float(moments[pair.site_i] @ moments[pair.site_j]))


def exchange_bar_sort_key(pair: PairCoupling) -> Tuple[float, int, int]:
    """Strongest coupling first, ties broken on the atoms themselves.

    The magnitude is quantised before it is compared. Symmetry-equivalent couplings
    are physically identical but not bitwise so -- summing a pair's bridges in a
    different order moves the last two bits, and a cubic cell's six neighbours come
    out as 0.052394788093148625 against 0.0523947880931486. Comparing those raw
    makes the tie-break unreachable and the bars come out in an order that looks
    random and shifts under edits that changed nothing. The step is a billionth of a
    meV: far below any difference worth drawing, far above the noise.
    """
    return (
        -round(abs(pair.j_eff) * 1000.0, EXCHANGE_TIE_DECIMALS),
        pair.site_i,
        pair.site_j,
    )


def exchange_pairs_for_site(
    pairs: Sequence[PairCoupling], site: int
) -> List[PairCoupling]:
    """The couplings that touch ``site``; all of them when nothing is selected."""
    if site < 0:
        return list(pairs)
    return [pair for pair in pairs if site in (pair.site_i, pair.site_j)]


def visible_pair_couplings(state: "AppState") -> Tuple[List[PairCoupling], int]:
    """The couplings to draw, and how many there were before the cap.

    Filtered to a single atom when one is selected, which is what makes clicking an
    atom in the 3D view a useful question to ask.

    The order is fixed at the moment of selection rather than recomputed each frame:
    it is the ``exchange_bar_sort_key`` order of the pairs as they stood then, and
    it only changes when the selection moves or the couplings are rebuilt. Bars
    therefore keep their positions while a structure is being inspected instead of
    swapping places under the cursor.
    """
    selected = state.selected_site_index
    pairs = exchange_pairs_for_site(state.magnetic_pair_couplings, selected)

    order_key = (selected, state._exchange_generation)
    if order_key != state._exchange_bar_order_key:
        state._exchange_bar_order_key = order_key
        state._exchange_bar_order = tuple(
            (pair.site_i, pair.site_j)
            for pair in sorted(pairs, key=exchange_bar_sort_key)
        )
    positions = {key: rank for rank, key in enumerate(state._exchange_bar_order)}
    # A pair the frozen order has never seen sorts to the end, in its own definite
    # order, rather than colliding at rank 0.
    ordered = sorted(
        pairs,
        key=lambda pair: (
            positions.get((pair.site_i, pair.site_j), len(positions)),
            exchange_bar_sort_key(pair),
        ),
    )
    return ordered[:EXCHANGE_PLOT_MAX_BARS], len(ordered)


def plot_exchange_couplings(state: "AppState") -> "PlotRect | None":
    """2D ImPlot pane: one bar per coupled pair of magnetic sites, in meV.

    Plotted in the model's own sign convention (J > 0 antiferromagnetic), which is
    how ``magnetic_pair_couplings`` stores it -- unlike ``magnetic_j_matrix``, which
    is negated for the solver. Bars are grouped into one ImPlot series per element
    pairing, so the series name doubles as the legend entry and the colouring falls
    out of the grouping.
    """
    pairs, total = visible_pair_couplings(state)
    selected = state.selected_site_index
    # Refreshed by the hover block below; cleared first so the 3D ring dies the
    # frame after the cursor leaves the bars.
    state.exchange_hover_site = -1

    if not state.magnetic_pair_couplings:
        imgui.text_disabled(
            state.magnetic_spin_status
            or state.baseline_status
            or "Build or select a structure to see its exchange couplings."
        )
    elif selected >= 0:
        imgui.text_disabled(
            f"Couplings for {exchange_site_label(state, selected)} "
            f"({len(pairs)} of {len(state.magnetic_pair_couplings)})"
        )
        # The way back out, on the same line as the label that says where you are.
        # Outside the plot rather than floating in a corner of it: this leaves the
        # atom's couplings for all of them, which is a step out of the view, not a
        # control over what the view is drawing.
        imgui.same_line()
        back_width = imgui.calc_text_size("All couplings").x + (
            2.0 * imgui.get_style().frame_padding.x
        )
        imgui.set_cursor_pos_x(
            max(
                imgui.get_cursor_pos_x(),
                imgui.get_cursor_pos_x()
                + imgui.get_content_region_avail().x
                - back_width,
            )
        )
        if imgui.small_button("All couplings##clear_exchange_site"):
            state.selected_site_index = -1
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "Show every coupling again.\n"
                "Clicking the atom itself in the 3D view does the same."
            )
    if total > len(pairs):
        imgui.text_disabled(
            f"Showing the {len(pairs)} strongest of {total} couplings."
        )

    # No mouse readout: x is a bar position, not a quantity, so the corner would
    # report a meaningless number for half of it. The tooltip says what a bar is.
    if not implot.begin_plot(
        "Exchange couplings##exchange_couplings",
        size=(-1, -1),
        flags=implot.Flags_.no_mouse_text.value,
    ):
        return None

    # One label for the whole axis rather than a name per bar: the per-pair names
    # are long enough that even a handful of them crowd each other, and a screenful
    # of them is a smear. Which pair a bar is is a question about one bar, which is
    # what the hover tooltip answers. The axis is categorical, so its ticks and grid
    # lines say nothing either.
    implot.setup_axis(
        implot.ImAxis_.x1,
        "Magnetic Pairs",
        implot.AxisFlags_.no_grid_lines.value
        | implot.AxisFlags_.no_tick_labels.value
        | implot.AxisFlags_.no_tick_marks.value,
    )
    implot.setup_axis(implot.ImAxis_.y1, "J (meV, + = AFM)")
    setup_two_d_legend()

    if pairs:
        positions = np.arange(len(pairs), dtype=np.float64)
        values = np.array([pair.j_eff * 1000.0 for pair in pairs], dtype=np.float64)
        # The three channels behind each total, drawn beside it once the view has
        # narrowed to one atom's couplings. Not in the unfiltered view: that one
        # carries up to EXCHANGE_PLOT_MAX_BARS pairs, and four times that many bars
        # is a smear no reader gets a decomposition out of.
        grouped = selected >= 0
        component_values = (
            np.array(
                [
                    [pair.j_se * 1000.0, pair.j_de * 1000.0, pair.j_oe * 1000.0]
                    for pair in pairs
                ],
                dtype=np.float64,
            )
            if grouped
            else np.zeros((len(pairs), 3), dtype=np.float64)
        )
        categories = [
            exchange_pair_label(pair.metal_i, pair.metal_j) for pair in pairs
        ]
        categories_arr = np.array(categories, dtype=object)
        # Sorted so a category keeps its colour when the filter changes which pairs
        # are on screen, as long as the same pairings are present.
        ordered_categories = sorted(set(categories))

        # Refit when the content changes, then leave the axes free to pan and zoom --
        # the same rule the energy scatter follows, keyed on content rather than list
        # identity because the list is rebuilt every frame.
        # Zero is always in range: which side of it a bar falls on is the whole
        # point. The bounds otherwise follow the data, so an all-AFM structure does
        # not spend half the pane on an empty FM half.
        # Over the component bars too where they are drawn: a superexchange bar
        # taller than its total (an OE channel pulling the total back down) is
        # exactly the case the decomposition exists to show, and it must not be
        # clipped.
        drawn = (
            np.concatenate((values, component_values.reshape(-1)))
            if grouped
            else values
        )
        low, high = min(0.0, float(drawn.min())), max(0.0, float(drawn.max()))
        axis_key = (len(pairs), selected, round(low, 9), round(high, 9))
        if axis_key != state._exchange_plot_axis_key:
            state._exchange_plot_axis_key = axis_key
            implot.setup_axis_limits(
                implot.ImAxis_.x1, -0.7, len(pairs) - 0.3, implot.Cond_.always
            )
            y_lo, y_hi = padded_two_d_limits(
                low, high, bottom=EXCHANGE_BOTTOM_HEADROOM
            )
            implot.setup_axis_limits(implot.ImAxis_.y1, y_lo, y_hi, implot.Cond_.always)

        # The total keeps the element blend and its per-pairing legend entry in
        # both views; grouped, it is only narrower and shifted to the left of its
        # own components.
        total_offset = EXCHANGE_COMPONENT_OFFSETS[0] if grouped else 0.0
        total_size = EXCHANGE_COMPONENT_BAR_SIZE if grouped else 0.65
        for rank, category in enumerate(ordered_categories):
            mask = categories_arr == category
            color = imgui.ImVec4(*exchange_pair_color(category, rank))
            spec = implot.Spec()
            spec.fill_color = color
            spec.line_color = color
            implot.plot_bars(
                category,
                np.ascontiguousarray(positions[mask] + total_offset, dtype=np.float64),
                np.ascontiguousarray(values[mask], dtype=np.float64),
                total_size,
                spec,
            )

        if grouped:
            for channel, (name, color) in enumerate(EXCHANGE_COMPONENT_SERIES):
                spec = implot.Spec()
                spec.fill_color = imgui.ImVec4(*color)
                spec.line_color = imgui.ImVec4(*color)
                # Plotted even where the channel is identically zero across every
                # pair. A flat green line under a tall total says "this coupling is
                # pure superexchange", which is the reading; a missing series would
                # leave the legend entry unexplained and the groups uneven.
                implot.plot_bars(
                    name,
                    np.ascontiguousarray(
                        positions + EXCHANGE_COMPONENT_OFFSETS[channel + 1],
                        dtype=np.float64,
                    ),
                    np.ascontiguousarray(
                        component_values[:, channel], dtype=np.float64
                    ),
                    EXCHANGE_COMPONENT_BAR_SIZE,
                    spec,
                )

        # Which couplings the selected configuration disagrees with. J does not
        # depend on the configuration -- the bars are the same whichever is picked --
        # but whether each one is satisfied does, and that is what changes here when
        # the configuration dropdown moves.
        moments = exchange_site_moments(state)
        frustration = np.array(
            [exchange_pair_frustration(pair, moments) for pair in pairs],
            dtype=np.float64,
        )
        frustrated = frustration > 0.0
        if frustrated.any():
            # A ring at the bar's tip rather than a recolour: the colour already
            # carries the element pairing, and this has to compose with it.
            spec = implot.Spec()
            spec.marker = implot.Marker_.circle
            spec.marker_size = 4.5
            spec.marker_fill_color = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
            spec.marker_line_color = imgui.ImVec4(*EXCHANGE_FRUSTRATED_COLOR)
            spec.line_color = imgui.ImVec4(0.0, 0.0, 0.0, 0.0)
            implot.plot_scatter(
                "Frustrated",
                np.ascontiguousarray(
                    positions[frustrated] + total_offset, dtype=np.float64
                ),
                np.ascontiguousarray(values[frustrated], dtype=np.float64),
                spec,
            )

        # Zero line last, so it sits over the bars and the FM/AFM split is readable
        # even where a bar crosses it.
        zero_spec = implot.Spec()
        zero_spec.line_color = imgui.ImVec4(0.75, 0.75, 0.75, 0.85)
        zero_spec.flags = int(implot.ItemFlags_.no_legend) | int(
            implot.InfLinesFlags_.horizontal
        )
        implot.plot_inf_lines(
            "##exchange_zero", np.array([0.0], dtype=np.float64), zero_spec
        )

        hovered_index = exchange_hovered_bar(
            positions,
            values,
            half_width=(
                EXCHANGE_GROUP_HALF_WIDTH if grouped else EXCHANGE_BAR_HALF_WIDTH
            ),
            extra=component_values if grouped else None,
        )
        if hovered_index >= 0:
            pair = pairs[hovered_index]
            imgui.set_tooltip(
                exchange_pair_tooltip(state, pair, frustration[hovered_index])
            )
            # A bar stands for a pair, so a click on it has to mean one of the
            # two atoms. In the filtered view it means the *other* one --
            # clicking a bar walks the coupling away from the selected atom,
            # never bouncing back to "all couplings" -- and in the unfiltered
            # view it means the lower-indexed one, named the same way every
            # time so a bar is predictable.
            if selected >= 0 and selected in (pair.site_i, pair.site_j):
                target = pair.site_j if pair.site_i == selected else pair.site_i
            else:
                target = min(pair.site_i, pair.site_j)
            # The 3D view rings this atom while the bar is hovered -- the same
            # white ring hovering it there draws (read back next frame).
            state.exchange_hover_site = target
            if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                state.selected_site_index = (
                    -1 if state.selected_site_index == target else target
                )
                # Choosing a coupling claims the 3D view for the exchange plot.
                state.structure_view_focus = "exchange"

    rect = current_plot_rect()
    implot.end_plot()
    return rect


def exchange_hovered_bar(
    positions: np.ndarray,
    values: np.ndarray,
    half_width: float = EXCHANGE_BAR_HALF_WIDTH,
    extra: np.ndarray | None = None,
) -> int:
    """Index of the pair under the cursor, or -1. Call inside an open plot.

    A rectangle test in plot coordinates rather than the energy scatter's nearest-
    point-in-pixels: a bar is an extended shape, and hovering anywhere on it --
    including a tall bar's far end -- should name it.

    ``extra`` carries the other bars a pair is drawn with, one row per pair, and
    widens the test to cover the tallest of them. With ``half_width`` at the whole
    group's width this makes the group a single target: every bar in it, and the
    air between them, stands for the same pair, so they answer with one tooltip
    rather than flickering between four.
    """
    if not implot.is_plot_hovered():
        return -1
    mouse = implot.get_plot_mouse_pos()
    for index in range(len(positions)):
        if abs(mouse.x - positions[index]) > half_width:
            continue
        spanned = [0.0, float(values[index])]
        if extra is not None:
            spanned.extend(float(value) for value in extra[index])
        if min(spanned) <= mouse.y <= max(spanned):
            return index
    return -1


def exchange_pair_tooltip(
    state: "AppState",
    pair: PairCoupling,
    frustration: float = 0.0,
) -> str:
    """One pair's coupling with the geometry behind it, for the hover tooltip.

    Shared by the bar chart and the 3D exchange paths, so hovering a bond in the
    structure and hovering its bar say the same thing.
    """
    j_mev = pair.j_eff * 1000.0
    sense = "AFM" if j_mev > 0 else "FM"
    ligands = ", ".join(sorted(set(pair.ligands)))
    angles = (
        f"{sum(pair.angles_deg) / len(pair.angles_deg):.1f} deg"
        if pair.angles_deg
        else "n/a"
    )
    bridges = f"{pair.bridge_count} bridge{'s' if pair.bridge_count != 1 else ''}"
    # The selected atom leads, matching the bar tick labels.
    first, second = pair.site_i, pair.site_j
    if state.selected_site_index == second:
        first, second = second, first
    lines = [
        f"{exchange_site_label(state, first)} - "
        f"{exchange_site_label(state, second)}",
        f"J = {j_mev:+.3f} meV ({sense})",
        # The three channels the bars decompose J into, so the numbers behind them
        # can be read off exactly rather than compared by eye.
        f"  SE {pair.j_se * 1000.0:+.3f}   DE {pair.j_de * 1000.0:+.3f}"
        f"   OE {pair.j_oe * 1000.0:+.3f}",
        f"d = {pair.distance:.3f} A",
        f"{bridges} via {ligands}, mean angle {angles}",
    ]
    if frustration != 0.0:
        lines.append(
            "Frustrated in this configuration"
            if frustration > 0.0
            else "Satisfied in this configuration"
        )
    return "\n".join(lines)


def plot_spin_energy_scatter(state: "AppState") -> "PlotRect | None":
    """2D ImPlot pane: ΔE-from-ground-state vs rank, colored by exact reference match.

    The points persist across builder edits -- only their energies are recomputed --
    so this tracks the structure instead of resetting. Clicking the nearest point
    selects that spin configuration, mirroring a click in the spin-results list.

    Returns the plot's data-area rect so the pane can float its dropdowns over it,
    or None when the plot did not open.
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
        return None

    # Model energies are fit to DFT+U relative energies in eV (docs/examples-solver.md);
    # they rank orderings rather than reproduce them quantitatively.
    implot.setup_axes("Rank", "ΔE (eV)")
    setup_two_d_legend()

    if configs:
        labels = state.spin_classification_labels()
        ranks = np.arange(1, len(configs) + 1, dtype=np.float64)
        e0 = float(configs[0].energy)
        delta_e = np.array(
            [float(config.energy) - e0 for config in configs], dtype=np.float64
        )
        plot_categories = spin_plot_categories(state)
        categories = [
            spin_plot_category(
                labels[index] if index < len(labels) else "Other", plot_categories
            )
            for index in range(len(configs))
        ]
        categories_arr = np.array(categories, dtype=object)

        # Refit when the landscape's shape changes -- a new solve, or energies that
        # moved after a builder edit -- then leave the axes free to pan/zoom. Keyed on
        # content rather than list identity because the list is rebuilt on every
        # re-energization, which would otherwise refit on every frame.
        e_max = float(delta_e.max())
        # delta_e is measured from the ground state, so the floor is 0. The extra
        # room at the top is for the pickers floating in the plot's corners.
        y_lo, y_hi = padded_two_d_limits(0.0, e_max)
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

        for category in plot_categories:
            mask = categories_arr == category
            # Only show a classification in the legend if it actually occurs.
            if int(mask.sum()) == 0:
                continue
            color = imgui.ImVec4(*spin_class_color(state, category))
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

    # Measured before end_plot: the rect is only defined while the plot is open.
    rect = current_plot_rect()
    implot.end_plot()
    return rect


def split_pane_heights(
    available: float,
    two_d_fraction: float,
) -> Tuple[float, float]:
    """Heights for the 3D and 2D halves of the Structure View, in pixels.

    The fraction is what the splitter stores, but the minimums are what actually
    decides on a short pane: a plot squeezed under them is not a smaller plot, it
    is an unreadable one. When even the two minimums do not fit, they are shared in
    proportion so both stay on screen rather than one being pushed out entirely.
    """
    usable = max(available - PANE_SPLITTER_THICKNESS, 0.0)
    floor_total = MIN_PLOT3D_HEIGHT + MIN_TWO_D_HEIGHT
    if usable <= floor_total:
        share = usable / floor_total if floor_total > 0.0 else 0.0
        # Never zero: ImPlot reads a zero height as "size yourself to the window",
        # so a pane momentarily too small to measure -- the first frame after
        # launch reports a negative height -- would give a full-size plot rather
        # than a squeezed one.
        return (
            max(MIN_PLOT3D_HEIGHT * share, 1.0),
            max(MIN_TWO_D_HEIGHT * share, 1.0),
        )
    two_d = min(
        max(usable * two_d_fraction, MIN_TWO_D_HEIGHT), usable - MIN_PLOT3D_HEIGHT
    )
    return usable - two_d, two_d


def draw_pane_splitter(
    splitter_id: str,
    two_d_fraction: float,
    available: float,
) -> float:
    """A draggable rule between the 3D and 2D plots. Returns the new fraction.

    An invisible button rather than a styled widget: it has to sit where a
    separator did, be grabbable without looking like a control, and report a drag.
    The line is drawn onto the window's draw list so it still reads as the rule it
    replaces, brightening while it is being used.
    """
    imgui.spacing()
    top = imgui.get_cursor_screen_pos()
    width = imgui.get_content_region_avail().x
    imgui.invisible_button(splitter_id, imgui.ImVec2(width, PANE_SPLITTER_THICKNESS))
    hovered, active = imgui.is_item_hovered(), imgui.is_item_active()
    if hovered or active:
        imgui.set_mouse_cursor(imgui.MouseCursor_.resize_ns)
    if active:
        # Applied to the fraction rather than to a stored pixel height, so the
        # split survives the pane being resized around it.
        usable = max(available - PANE_SPLITTER_THICKNESS, 1.0)
        two_d_fraction -= imgui.get_io().mouse_delta.y / usable

    colour = int(
        imgui.Col_.separator_active
        if active
        else (imgui.Col_.separator_hovered if hovered else imgui.Col_.separator)
    )
    middle = top.y + PANE_SPLITTER_THICKNESS * 0.5
    packed = imgui.get_color_u32(colour)
    draw_list = imgui.get_window_draw_list()
    draw_list.add_line(
        imgui.ImVec2(top.x, middle),
        imgui.ImVec2(top.x + width, middle),
        packed,
        2.0 if (hovered or active) else 1.0,
    )
    # A short grip across the middle, so the rule reads as something to take hold
    # of rather than as the separator it replaces. Without it the band is eight
    # invisible pixels that nothing invites you to try.
    grip = min(PANE_SPLITTER_GRIP_WIDTH, width)
    centre = top.x + width * 0.5
    for offset in (-2.0, 2.0):
        draw_list.add_line(
            imgui.ImVec2(centre - grip * 0.5, middle + offset),
            imgui.ImVec2(centre + grip * 0.5, middle + offset),
            packed,
            1.0,
        )
    # Clamped here rather than in split_pane_heights so a drag that runs off the
    # end of the pane does not bank fraction it would have to be dragged back
    # through before anything moves.
    return min(max(two_d_fraction, 0.05), 0.95)


def draw_plot_corner_combo(
    window_id: str,
    corner: Tuple[float, float],
    pivot: Tuple[float, float],
    draw: Any,
) -> None:
    """Float one control over a plot corner.

    A borderless window rather than widgets submitted inside ``begin_plot``: it gets
    a background so it stays readable over the data, and it is outside the plot's
    item scope so it cannot be mistaken for a plot item. Repositioned every frame
    (``Cond_.always``) because it belongs to the plot, not to the user -- unlike the
    3D view's summary box, which is placed once and then left where it is dragged.
    """
    imgui.set_next_window_pos(
        imgui.ImVec2(*corner), imgui.Cond_.always, imgui.ImVec2(*pivot)
    )
    imgui.set_next_window_bg_alpha(SUMMARY_OVERLAY_BG_ALPHA)
    flags = (
        imgui.WindowFlags_.no_title_bar.value
        | imgui.WindowFlags_.no_resize.value
        | imgui.WindowFlags_.no_move.value
        | imgui.WindowFlags_.no_scrollbar.value
        | imgui.WindowFlags_.no_collapse.value
        | imgui.WindowFlags_.no_focus_on_appearing.value
        | imgui.WindowFlags_.no_nav.value
        | imgui.WindowFlags_.no_saved_settings.value
        | imgui.WindowFlags_.no_docking.value
        | imgui.WindowFlags_.always_auto_resize.value
    )
    if imgui.begin(window_id, None, flags):
        draw()
    imgui.end()


def draw_two_d_plot_overlays(state: "AppState", rect: PlotRect) -> None:
    """The plot picker, in the plot's top-left corner.

    Only the plot picker floats here. The configuration is chosen from the spin
    results list in the Magnetism panel, which shows more about each one
    than a combo ever could and does not have to sit over the data to do it.
    """
    (x0, y0), _bottom_right = rect
    inset = TWO_D_OVERLAY_INSET

    def draw_plot_picker() -> None:
        imgui.push_item_width(170.0)
        _, state.two_d_plot_index = imgui.combo(
            "##two_d_plot_kind", state.two_d_plot_index, TWO_D_PLOT_NAMES
        )
        imgui.pop_item_width()

    draw_plot_corner_combo(
        "##two_d_plot_picker", (x0 + inset, y0 + inset), (0.0, 0.0), draw_plot_picker
    )


def gui_two_d_pane(state: "AppState") -> None:
    """The 2D pane: one of three plots, with its pickers floated over the corners."""
    if state.two_d_plot_index == TWO_D_PLOT_RELAXATION:
        rect = plot_relaxation_trace(state)
    elif state.two_d_plot_index == TWO_D_PLOT_LIVE_ENERGY:
        rect = plot_live_energy(state)
    elif state.two_d_plot_index == 1:
        rect = plot_exchange_couplings(state)
    else:
        rect = plot_spin_energy_scatter(state)
    # After the plot is closed: beginning a window while a plot is open would nest it
    # inside the plot's item scope. Same reason the 3D summary overlay waits.
    if rect is not None:
        draw_two_d_plot_overlays(state, rect)


def vacancy_marker_mask(
    marker_cartesian: np.ndarray,
    lattice: np.ndarray,
    site_cartesian: Sequence[np.ndarray],
    *,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Which vacancy markers stand for any of the given vacated sites.

    Matched by fractional position modulo the cell: the render may draw the
    closing boundary layer, where a vacated corner site has several markers,
    and every one of them stands for the same row.
    """
    markers = np.asarray(marker_cartesian, dtype=np.float64).reshape(-1, 3)
    sites = np.asarray(list(site_cartesian), dtype=np.float64).reshape(-1, 3)
    if not len(markers) or not len(sites):
        return np.zeros(len(markers), dtype=bool)
    try:
        frame = np.asarray(lattice, dtype=np.float64).T
        marker_fractional = np.linalg.solve(frame, markers.T).T
        site_fractional = np.linalg.solve(frame, sites.T).T
    except np.linalg.LinAlgError:
        return np.zeros(len(markers), dtype=bool)
    delta = (marker_fractional[:, None, :] - site_fractional[None, :, :] + 0.5) % 1.0 - 0.5
    return np.all(np.abs(delta) < tolerance, axis=2).any(axis=1)


def vacancy_row_for_marker(
    state: "AppState", marker_cartesian: np.ndarray, lattice: np.ndarray
) -> "AtomRow | None":
    """The vacancy row a ghost marker stands for, or None."""
    vacant = [row for row in state.atom_table() if row.vacant]
    if not vacant:
        return None
    mask = vacancy_marker_mask(
        np.asarray([row.cartesian for row in vacant]),
        lattice,
        [np.asarray(marker_cartesian, dtype=np.float64)],
    )
    hits = np.flatnonzero(mask)
    return vacant[int(hits[0])] if len(hits) else None


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

    _title, coords, axis_label, use_cartesian = structure_plot_view()
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
    flags = structure_plot_flags(show_legend=state.show_legend)

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
            spin_ordering = state.described_config(selected_config)
    rendered_b_grid = state.b_grid_for_structure(structure)
    if (
        state.show_spin_classifications
        and rendered_b_grid is not None
        and selected_spin_moments is not None
    ):
        # Walks every nearest-neighbour bond in the grid, so it is only worth doing
        # when the badge below is actually going to print the result.
        alignment_counts = spin_alignment_edge_counts(
            structure.cartesian_coords, rendered_b_grid, selected_spin_moments
        )
    imgui.text("Look Down:")
    for axis_index, axis_name in enumerate("abc"):
        imgui.same_line()
        if imgui.small_button(f"{axis_name}##align_view_{axis_name}"):
            state.pending_view_axis = axis_index
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Look down the {axis_name} axis.\nClick again for the opposite face."
            )
    # Right-aligned, so the controls read left to right and the hint stays out of their
    # way. On a pane too narrow to hold both it simply trails the buttons.
    hint = "Right Click to Rotate"
    imgui.same_line()
    slack = imgui.get_content_region_avail().x - imgui.calc_text_size(hint).x
    if slack > 0.0:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + slack)
    imgui.text(hint)

    # Second row: which screen axis the buttons above step about. Radio buttons rather
    # than checkboxes because exactly one is always chosen -- clicking the selected one
    # again is a no-op, so there is no way to end up with no axis at all.
    imgui.text("Turn about:")
    for axis_index, axis_name in enumerate("xyz"):
        imgui.same_line()
        _, state.screen_turn_axis_index = imgui.radio_button(
            f"{axis_name}##screen_turn_{axis_name}",
            state.screen_turn_axis_index,
            axis_index,
        )
        if imgui.is_item_hovered():
            imgui.set_tooltip(SCREEN_TURN_AXIS_TOOLTIPS[axis_index])

    # Read after the radios, so a press this frame names the axis it just chose.
    step = STRUCTURE_TURN_STEP_DEGREES
    turn_axis_name = "xyz"[state.screen_turn_axis_index % 3]
    for label, degrees in ((f"-{step:g}°", -step), (f"+{step:g}°", step)):
        imgui.same_line()
        if imgui.small_button(f"{label}##turn_view_{degrees:+g}"):
            state.turn_structure_view(degrees)
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                f"Turn the view {step:g}° about the screen {turn_axis_name} axis."
            )

    # What the builder edits and the plots describe, named right over the
    # picture of it. Right-aligned under the rotate hint, and last on the row:
    # aligning it to the right edge leaves no room after it, so anything drawn
    # on the same line behind it lands off the pane.
    focus_name = "-" if state.focus is None else state.focus.name
    indicator = f"Active structure: {focus_name}"
    imgui.same_line()
    slack = imgui.get_content_region_avail().x - imgui.calc_text_size(indicator).x
    if slack > 0.0:
        imgui.set_cursor_pos_x(imgui.get_cursor_pos_x() + slack)
    imgui.text(indicator)

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
    # Opt-in, via the checkbox at the top of Magnetism. Left off, the atoms
    # keep their element colours and the sign computation is skipped entirely.
    displayed_spin_signs = (
        spin_signs_from_moments(displayed_spin_moments)
        if state.color_atoms_by_spin
        else None
    )
    # Sizing by moment rides along with the colouring: the same switch, so the spheres
    # never say one thing with their colour and another with their size. ``atom_radii``
    # stays the unscaled base -- vacancies below are sized from it, and a hole should
    # keep the size of the atom it replaces however the magnetic sites are drawn.
    render_radii = atom_radii
    formal_magnitudes = (
        state.displayed_site_moment_magnitudes(structure)
        if state.color_atoms_by_spin
        else None
    )
    if formal_magnitudes is not None:
        # The moment actually on a site is its direction times its formal magnitude, so
        # its size is the product: the high-spin value where this configuration polarises
        # the site, and zero where it leaves it alone.
        polarised = spin_moment_magnitudes(displayed_spin_moments, structure.atom_count)
        render_radii = spin_scaled_render_radii(
            atom_radii,
            formal_magnitudes
            if polarised is None
            else formal_magnitudes * (polarised > 0.0),
        )
    imgui.separator()

    # Vacancies have no atom to draw, so their markers come from the ideal build
    # -- or, for a loaded structure, from the record of what was removed.
    vacancy_coords, vacancy_labels = state.vacancy_markers(structure)
    vacancy_cartesian = np.asarray(vacancy_coords, dtype=np.float64).reshape(-1, 3)
    vacancy_radii = vacancy_render_radii(
        vacancy_labels,
        structure,
        atom_radii,
        render_with_ionic_radius=state.render_with_ionic_radius,
    )
    if len(vacancy_coords) and not use_cartesian:
        vacancy_coords = np.linalg.solve(
            structure.lattice.T, vacancy_coords.T
        ).T

    plot_coords = coords
    plot_axis_extents = sphere_axis_extents(
        render_radii, structure.lattice, use_cartesian
    )
    if len(vacancy_coords):
        plot_coords = np.vstack((plot_coords, vacancy_coords))
        plot_axis_extents = np.vstack(
            (
                plot_axis_extents,
                sphere_axis_extents(vacancy_radii, structure.lattice, use_cartesian),
            )
        )
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
        padding_scale=STRUCTURE_PLOT_PADDING / max(state.structure_zoom, 1e-3),
        axis_extents=plot_axis_extents,
    )
    # Shared by the corner triad and the a/b/c alignment buttons, so both talk about the
    # same directions -- in Cartesian mode those are the real lattice vectors, which the
    # box's own "a"/"b"/"c" labels are not.
    axis_directions = plot_axis_directions(structure.lattice, use_cartesian, plot_limits)
    plot_id = "##structure_view"

    # Reserve the lower portion of the pane for the 2D plot. The share is the
    # user's, dragged on the splitter between them; the minimums below keep either
    # plot from being squeezed to nothing when the pane itself is short.
    available = imgui.get_content_region_avail()
    plot3d_height, _two_d_height = split_pane_heights(
        available.y, state.two_d_pane_fraction
    )

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
        # Re-applied every frame, not just when the structure changes: the limits are a
        # cube centred on the structure, so this keeps the cell centred no matter what.
        implot3d.setup_axes_limits(*plot_limits, implot3d.Cond_.always)

        # The lattice directions only exist once the box is known, so a button press is
        # turned into a rotation target here rather than where it was clicked.
        if state.pending_view_axis is not None:
            # Judged against a swing still in flight when there is one, so a double press
            # flips to the far face rather than re-deciding from a halfway pose.
            settled = (
                state.structure_rotation
                if state.structure_rotation_target is None
                else state.structure_rotation_target
            )
            state.structure_rotation_target = view_rotation_for_axis(
                axis_directions,
                state.pending_view_axis,
                view_axis_alignment_sign(
                    settled, axis_directions, state.pending_view_axis
                ),
            )
            state.pending_view_axis = None
        advance_structure_alignment(state)
        # Pushed every frame, ahead of the first plotting command, because this is a
        # Setup call. Cond_.always also tells ImPlot3D the rotation is spoken for.
        implot3d.setup_box_rotation(
            implot3d.Quat(*state.structure_rotation), False, implot3d.Cond_.always
        )

        rect_pos, rect_size = implot3d.get_plot_rect_pos(), implot3d.get_plot_rect_size()
        rect_min = imgui.ImVec2(rect_pos.x, rect_pos.y)
        rect_max = imgui.ImVec2(rect_pos.x + rect_size.x, rect_pos.y + rect_size.y)
        apply_structure_zoom(state, rect_min, rect_max)
        # Reads this frame's drag, lands on the next one's box -- the same one-frame lag
        # the zoom above has always had, and just as invisible.
        apply_structure_rotation(state, rect_min, rect_max)

        if state.show_unit_cell:
            plot_unit_cell(structure.lattice, use_cartesian=axis_label == "A")
        if (
            state.show_domain_cell
            and state.domain_highlight_active
            and state.focus is not None
        ):
            domain_corners = domain_cell_vertices(
                state.focus,
                state.domain_index_for_structure(state.focus),
                use_cartesian=axis_label == "A",
            )
            if domain_corners is not None:
                plot_box_wireframe(
                    domain_corners,
                    label="##domain_cell",
                    color=DOMAIN_CELL_LINE_COLOR,
                    weight=DOMAIN_CELL_LINE_WEIGHT,
                )

        if state.show_miller_planes:
            overlay = state.miller_plane_overlay(state.selected_spin_config())
            if overlay is not None:
                overlay_miller, overlay_offsets, overlay_colors = overlay
                plot_miller_planes(
                    structure.lattice,
                    overlay_miller,
                    overlay_offsets,
                    use_cartesian=axis_label == "A",
                    colors=overlay_colors,
                )

        # Exactly one special view at a time: "exchange" or "plain", decided in
        # one place so the decorations can never mix.
        view_mode = state.structure_view_mode()

        # The atom selection, as the render sees it. Rows are refs; the picture is
        # render indices; the map between them is the focus's own atom index.
        focus = state.focus
        render_to_focus = (
            state._cached(
                ("render_to_focus", id(structure), id(focus)),
                (
                    state._structure_signature(structure),
                    state._structure_signature(focus),
                ),
                lambda: render_to_focus_indices(structure, focus),
            )
            if focus is not None
            else np.full(structure.atom_count, -1, dtype=np.int64)
        )
        selected_rows = state.active_rows()
        selected_focus_indices = {row.index for row in selected_rows if row.index >= 0}
        selected_render = (
            {
                int(render_index)
                for render_index in np.flatnonzero(
                    np.isin(render_to_focus, list(selected_focus_indices))
                )
            }
            if selected_focus_indices
            else set()
        )
        # Vacated sites in the selection, matched to their markers by position.
        selected_vacancy_mask = np.zeros(len(vacancy_cartesian), dtype=bool)
        if len(vacancy_cartesian) and any(row.vacant for row in selected_rows):
            selected_vacancy_mask = vacancy_marker_mask(
                vacancy_cartesian,
                structure.lattice,
                [row.cartesian for row in selected_rows if row.vacant],
            )
        # The hand picks alone, which get a ring of their own further down.
        # Every periodic image of a picked site is in here, the way the hover
        # rings take them all: a click on any image is the same click.
        picked_rows = state.picked_rows()
        picked_focus_indices = {row.index for row in picked_rows if row.index >= 0}
        picked_render = (
            {
                int(render_index)
                for render_index in np.flatnonzero(
                    np.isin(render_to_focus, list(picked_focus_indices))
                )
            }
            if picked_focus_indices
            else set()
        )
        picked_vacancy_mask = np.zeros(len(vacancy_cartesian), dtype=bool)
        if len(vacancy_cartesian) and any(row.vacant for row in picked_rows):
            picked_vacancy_mask = vacancy_marker_mask(
                vacancy_cartesian,
                structure.lattice,
                [row.cartesian for row in picked_rows if row.vacant],
            )
        # The slab is a filter, and the fade is what it looks like: with it on,
        # the atoms it holds (and any hand picks, which follow the picture
        # rather than the slab) stay at full strength and stay clickable, and
        # everything else drops to faint context that cannot be clicked. A hand
        # pick does not fade anything -- picking an atom out of the cell should
        # leave the cell it was picked out of on screen.
        filter_to_slab = state.slab_enabled

        slab_faces: List[np.ndarray] = []
        if state.slab_enabled and focus is not None:
            slab_faces = plot_selection_slab(
                focus.lattice,
                state.selection_slab,
                use_cartesian=use_cartesian,
                hovered=state._slab_drag is not None,
            )

        if (
            state.show_spin_classifications
            and rendered_build is not None
            and selected_spin_moments is not None
        ):
            plot_classification_lattice(coords, rendered_b_grid, selected_spin_moments)

        # The superexchange network of the selected atom. Built before anything is
        # drawn because it decides two things: which atoms stay opaque, and what the
        # white M-L-M paths further down run along. -1 in every mode but
        # "exchange" -- the signal to draw normally.
        exchange_site = (
            exchange_selection_site(state) if view_mode == "exchange" else -1
        )
        exchange_paths = exchange_render_paths(
            state, structure, coords, exchange_site, use_cartesian=use_cartesian
        )
        exchange_prominent = None
        exchange_prominent_ends: set[int] | None = None
        if exchange_site >= 0:
            exchange_prominent = exchange_prominent_render_atoms(
                coords, exchange_paths
            )
            # A neighbour across the boundary is drawn wrapped to the far face,
            # where no path endpoint reaches it; light it anyway, or the chart
            # would show a bar for an atom that faded into the cell.
            exchange_prominent_ends = exchange_unreached_partner_atoms(
                state, structure, exchange_site, exchange_prominent
            )
            exchange_prominent |= exchange_prominent_ends

        if (
            state.show_octahedra
            and rendered_build is not None
            and getattr(structure, "generation_parameters", None) is not None
        ):
            # Split the cages the same way the atoms are split, or the ones in
            # front of a focused plane sit over it at full strength and hide the
            # very layer they are meant to frame. Both halves go under the one
            # label, which ImPlot3D folds into a single legend entry.
            if exchange_prominent is not None:
                # The cages would sit over the exchange paths and hide exactly the
                # bonds being asked about, so they all fade rather than being split:
                # the network, not the framework, is what is being looked at.
                halves = [(None, None, FADED_OCTAHEDRON_ALPHA, False)]
            elif filter_to_slab:
                # The cages would sit over the slab's atoms and hide them, so
                # they all fade: the atoms, not the framework, are what is
                # being looked at.
                halves = [(None, None, FADED_OCTAHEDRON_ALPHA, False)]
            else:
                halves = [(None, None, OCTAHEDRON_ALPHA, True)]
            for keep_cells, drop_cells, alpha, with_edges in halves:
                triangle_vertices = octahedron_triangles_for_generated_structure(
                    structure,
                    rendered_build,
                    keep_cells=keep_cells,
                    drop_cells=drop_cells,
                )
                if triangle_vertices.size == 0:
                    continue
                # Thin white edges over the translucent fill: the cage reads as a
                # cage from its edges, and white keeps them legible against every
                # element colour without competing with any of them.
                #
                # The faded half drops its lines entirely. fill_alpha does not
                # reach the edges, and a cage is mostly edges -- fading one
                # without dropping its lines leaves a bright wireframe exactly
                # where the fade was supposed to be.
                spec_tri = implot3d.Spec(
                    fill_color=OCTAHEDRON_FILL_COLOR,
                    line_color=OCTAHEDRON_EDGE_COLOR,
                    line_weight=OCTAHEDRON_EDGE_WEIGHT,
                    marker=implot3d.Marker_.none,
                    fill_alpha=alpha,
                    flags=(
                        0 if with_edges else implot3d.TriangleFlags_.no_lines.value
                    ),
                )
                implot3d.plot_triangle(
                    "Octahedra",
                    np.ascontiguousarray(triangle_vertices[:, 0], dtype=np.float64),
                    np.ascontiguousarray(triangle_vertices[:, 1], dtype=np.float64),
                    np.ascontiguousarray(triangle_vertices[:, 2], dtype=np.float64),
                    spec=spec_tri,
                )

        sphere_detail = sphere_detail_for(structure.atom_count)
        drawn_atoms = visible_role_indices(
            structure,
            show_a=state.show_a_sites,
            show_b=state.show_b_sites,
            show_x=state.show_x_sites,
        )
        grouped_indices: Dict[
            tuple[str, tuple[float, float, float, float], bool], List[int]
        ] = {}
        for atom_index, element in enumerate(structure.atomic_labels):
            if atom_index not in drawn_atoms:
                continue
            if render_radii[atom_index] <= 0.0:
                # Sized down to nothing by the spin scaling. Dropped here rather than
                # left to the mesh builder, which would still register the group and
                # leave an entry in the legend with no spheres behind it.
                continue
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
            if filter_to_slab:
                prominent = atom_index in selected_render
            elif exchange_prominent is not None:
                prominent = atom_index in exchange_prominent
            else:
                prominent = True
            grouped_indices.setdefault((label, color, prominent), []).append(atom_index)

        # Faded first so the prominent atoms draw over them rather than
        # through them. Both halves are submitted under the same label, which
        # ImPlot3D folds into one legend entry -- hiding an element still hides
        # all of it, whether or not the plane happens to contain some of it.
        for (label, color, prominent), element_indices in sorted(
            grouped_indices.items(), key=lambda item: item[0][2]
        ):
            element_coords = ensure_xyz_array(coords[element_indices])
            element_radii = np.asarray(render_radii[element_indices], dtype=np.float64)
            if element_coords.shape[0] == 0:
                continue
            mesh = build_sphere_mesh(
                element_coords,
                element_radii,
                structure.lattice,
                use_cartesian=use_cartesian,
                detail=sphere_detail,
            )
            if filter_to_slab:
                # The slab's atoms at full strength; everything else stays as
                # faint translucent context rather than vanishing.
                fill_alpha = 1.0 if prominent else FADED_ATOM_ALPHA
            elif exchange_prominent is not None:
                fill_alpha = (
                    PROMINENT_ATOM_ALPHA if prominent else FADED_ATOM_ALPHA
                )
            else:
                fill_alpha = 0.92
            spec = implot3d.Spec(
                fill_color=color,
                line_color=color,
                fill_alpha=fill_alpha,
                flags=implot3d.MeshFlags_.no_lines.value,
            )
            implot3d.plot_mesh(label, mesh, spec=spec)

        # Whether the cursor is already on an atom. The exchange paths run between
        # atoms and pass right under them, so both would claim the same pixel; the
        # atom wins, because it is the one a click would act on and a tooltip that
        # described something else would be describing the wrong thing.
        atom_hovered = False

        # Pick atoms by clicking them. Left click is free for this: the view
        # orbits on the right button.
        #
        # What a click means depends on which 2D plot is up:
        #
        #   coupling plot -- the magnetic sites are selectable, and once one is
        #     selected only the atoms it actually couples to are, so a click can
        #     only ever walk along a bond that is drawn on screen. Clicking the
        #     selected atom again clears it.
        #   otherwise -- every atom, and every vacated site, is a pick target for
        #     the Selection panel: a click toggles the atom in the selection, and
        #     Shift-click only ever adds.
        selectable = state.two_d_plot_index == 1
        if not selectable:
            # The slab filters what can be picked: with it on, only the atoms
            # it holds -- and the hand picks, which stay clickable so they can
            # be let go of -- are targets, which is what the fade is showing.
            # Off, every atom is a target again.
            candidates = (
                sorted(selected_render)
                if filter_to_slab
                else list(range(len(coords)))
            )
        elif exchange_site >= 0:
            # The wrapped neighbours are lit, so they have to be clickable
            # too -- an atom that is bright but inert reads as a bug.
            candidates = sorted(
                set(exchange_pick_candidates(coords, exchange_paths))
                | (exchange_prominent_ends or set())
            )
        else:
            candidates = magnetic_pick_candidates(state, structure)

        # Vacated sites ride along as pick targets after the atoms, so a
        # vacancy can be selected and restored from its row.
        vacancy_targets = (
            [
                len(coords) + int(marker)
                for marker in range(len(vacancy_coords))
                if not filter_to_slab or selected_vacancy_mask[marker]
            ]
            if not selectable and len(vacancy_coords)
            else []
        )
        pick_display = (
            np.vstack((coords, vacancy_coords)) if vacancy_targets else coords
        )
        pick_candidates = list(candidates) + vacancy_targets

        mouse_over_plot = imgui.is_mouse_hovering_rect(rect_min, rect_max)
        hovered = -1
        if pick_candidates and mouse_over_plot:
            mouse = imgui.get_mouse_pos()
            depths = view_space_depth(
                pick_display[pick_candidates], plot_limits, state.structure_rotation
            )
            # Projecting every atom costs a pybind call each; cached on the view
            # so that moving the cursor over a still structure reprojects nothing.
            pixels = state._cached(
                ("pick_pixels", id(coords)),
                (
                    view_projection_key(
                        plot_limits,
                        state.structure_rotation,
                        state.structure_zoom,
                        rect_min,
                        rect_max,
                    ),
                    # The candidate set itself, not just its size: a selection
                    # of the same size but different atoms projects differently.
                    hash(tuple(pick_candidates)),
                ),
                lambda: candidate_pixels(pick_display, pick_candidates),
            )
            hovered = nearest_picked_atom(
                pixels, pick_candidates, depths, (mouse.x, mouse.y)
            )

        analysis = state.magnetic_analysis_structure
        hovered_vacancy = hovered - len(coords) if hovered >= len(coords) else -1
        hovered_focus = (
            int(render_to_focus[hovered]) if 0 <= hovered < len(coords) else -1
        )
        site = (
            -1
            if hovered < 0 or hovered_vacancy >= 0 or analysis is None
            else source_site_for_render_index(structure, analysis, hovered)
        )
        additive = imgui.get_io().key_shift
        if hovered_vacancy >= 0:
            # A ghost marker: the row of the vacated site it stands for.
            atom_hovered = True
            vacancy_extents = sphere_axis_extents(
                vacancy_radii, structure.lattice, use_cartesian
            )
            draw_site_highlight_rings(
                vacancy_coords,
                vacancy_extents,
                [hovered_vacancy],
                color=PICK_HOVER_COLOR,
            )
            row = vacancy_row_for_marker(
                state, vacancy_cartesian[hovered_vacancy], structure.lattice
            )
            if row is not None:
                imgui.set_tooltip(f"Vacancy ({row.ideal_element} site)")
                if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                    state.toggle_atom_selection(row.ref, additive=additive)
        elif hovered >= 0 and (site >= 0 or hovered_focus >= 0):
            atom_hovered = True
            pick_extents = sphere_axis_extents(
                render_radii, structure.lattice, use_cartesian
            )
            # Every image of the site, for the same reason the selection rings
            # them all: a click on any of them does the same thing.
            images = (
                highlighted_render_indices(structure, analysis, site)
                if site >= 0 and analysis is not None
                else [int(index) for index in np.flatnonzero(render_to_focus == hovered_focus)]
            )
            draw_site_highlight_rings(coords, pick_extents, images, color=PICK_HOVER_COLOR)
            if selectable and site >= 0:
                count = len(
                    exchange_pairs_for_site(state.magnetic_pair_couplings, site)
                )
                imgui.set_tooltip(
                    f"{exchange_ion_label(state, site)} - {count} coupling"
                    f"{'' if count == 1 else 's'}"
                )
                if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                    state.selected_site_index = (
                        -1 if state.selected_site_index == site else site
                    )
                    # Choosing a coupling claims the 3D view for the exchange plot.
                    state.structure_view_focus = "exchange"
            else:
                row = state.atom_row_for_index(hovered_focus)
                if site >= 0 and analysis is not None:
                    imgui.set_tooltip(site_hover_tooltip(state, analysis, site))
                elif row is not None:
                    imgui.set_tooltip(f"{row.element} #{row.index + 1}")
                if imgui.is_mouse_clicked(imgui.MouseButton_.left) and row is not None:
                    state.toggle_atom_selection(row.ref, additive=additive)

        # The slab is dragged along its own normal: press on either face and
        # pull. Atoms win the hover, so a click on an atom inside the slab
        # still picks the atom.
        if slab_faces and focus is not None:
            mouse = imgui.get_mouse_pos()
            over_slab = (
                mouse_over_plot
                and not atom_hovered
                and slab_faces_hovered(slab_faces, (mouse.x, mouse.y))
            )
            if state._slab_drag is None:
                if over_slab:
                    imgui.set_tooltip("Drag to slide the slab along its normal")
                    if imgui.is_mouse_clicked(imgui.MouseButton_.left):
                        state._slab_drag = (
                            float(state.selection_slab.offset),
                            float(mouse.x),
                            float(mouse.y),
                        )
            elif imgui.is_mouse_down(imgui.MouseButton_.left):
                start_offset, start_x, start_y = state._slab_drag
                scale = slab_pixels_per_angstrom(
                    focus.lattice, state.selection_slab, use_cartesian=use_cartesian
                )
                if scale is not None:
                    state.selection_slab.offset = slab_offset_after_drag(
                        focus.lattice,
                        state.selection_slab,
                        start_offset,
                        (mouse.x - start_x, mouse.y - start_y),
                        scale,
                    )
            else:
                state._slab_drag = None
        else:
            state._slab_drag = None

        if state.slab_enabled and focus is not None:
            draw_slab_arrow(
                focus.lattice, state.selection_slab, use_cartesian=use_cartesian
            )

        # The row under the cursor in the Atoms panel, in the same white the
        # cursor draws on an atom here -- hovering a row and hovering the atom
        # are the same gesture on the same object, so they get the same mark.
        # Read and cleared: the panel is drawn after the 3D view, so this is
        # last frame's row, and consuming it bounds how long a ring can outlive
        # the cursor to one frame.
        hover_ref, state.atom_hover_ref = state.atom_hover_ref, None
        if hover_ref is not None:
            hover_row = state.atom_row_for_ref(hover_ref)
            if hover_row is not None and hover_row.index >= 0:
                draw_site_highlight_rings(
                    coords,
                    sphere_axis_extents(render_radii, structure.lattice, use_cartesian),
                    [int(index) for index in np.flatnonzero(render_to_focus == hover_row.index)],
                    color=PICK_HOVER_COLOR,
                )
            elif hover_row is not None and len(vacancy_cartesian):
                mask = vacancy_marker_mask(
                    vacancy_cartesian, structure.lattice, [hover_row.cartesian]
                )
                draw_site_highlight_rings(
                    vacancy_coords,
                    sphere_axis_extents(vacancy_radii, structure.lattice, use_cartesian),
                    [int(index) for index in np.flatnonzero(mask)],
                    color=PICK_HOVER_COLOR,
                )

        # Ring every atom picked by hand. Always: in every view, faded or not,
        # slab up or down. The picks are a set being gathered up for an edit,
        # and a set that is only visible in a list is one that is easy to lose
        # track of -- the ring is what ties the row to the atom it names.
        if picked_render:
            draw_site_highlight_rings(
                coords,
                sphere_axis_extents(render_radii, structure.lattice, use_cartesian),
                sorted(picked_render),
                scale=PICKED_ATOM_RING_SCALE,
                color=PICKED_ATOM_RING_COLOR,
            )
        if picked_vacancy_mask.any():
            draw_site_highlight_rings(
                vacancy_coords,
                sphere_axis_extents(vacancy_radii, structure.lattice, use_cartesian),
                [int(marker) for marker in np.flatnonzero(picked_vacancy_mask)],
                scale=PICKED_ATOM_RING_SCALE,
                color=PICKED_ATOM_RING_COLOR,
            )

        # Ring the site whose oxidation state is being edited. Screen-space, so
        # it faces the viewer at any rotation, and drawn after the meshes so
        # nothing occludes it.
        analysis_structure = state.magnetic_analysis_structure
        if analysis_structure is not None and state.selected_site_index >= 0:
            draw_site_highlight_rings(
                coords,
                sphere_axis_extents(render_radii, structure.lattice, use_cartesian),
                highlighted_render_indices(
                    structure, analysis_structure, state.selected_site_index
                ),
            )

        # The atom under the cursor in a per-atom plot, in the same white the
        # cursor draws on an atom here. Read and cleared, rather than read: the
        # plot is drawn *after* the 3D view, so what arrives here is last
        # frame's atom, and consuming it bounds how long a ring can outlive the
        # cursor to one frame.
        hovered_row, state.oxidation_hover_site = state.oxidation_hover_site, -1
        if (
            analysis_structure is not None
            and hovered_row >= 0
            and hovered_row != state.selected_site_index
        ):
            draw_site_highlight_rings(
                coords,
                sphere_axis_extents(render_radii, structure.lattice, use_cartesian),
                highlighted_render_indices(structure, analysis_structure, hovered_row),
                color=PICK_HOVER_COLOR,
            )

        # The superexchange network of the selected atom, drawn as it physically
        # runs: metal to bridging ligand to metal, one polyline per bridge. Strength
        # is carried by opacity alone, so a strong coupling and a weak one differ in
        # the one channel the eye reads as "more", and a pair bridged twice shows
        # both of its paths.
        if exchange_site >= 0:
            # The bar chart's hover reaches back into the picture: the atom a
            # click on the hovered bar would select gets the same white ring
            # hovering it here draws.
            if state.exchange_hover_site >= 0 and analysis_structure is not None:
                draw_site_highlight_rings(
                    coords,
                    sphere_axis_extents(
                        render_radii, structure.lattice, use_cartesian
                    ),
                    highlighted_render_indices(
                        structure, analysis_structure, state.exchange_hover_site
                    ),
                    color=PICK_HOVER_COLOR,
                )
            paths = exchange_paths
            hovered_path = -1
            if paths and not atom_hovered and imgui.is_mouse_hovering_rect(
                rect_min, rect_max
            ):
                mouse = imgui.get_mouse_pos()
                hovered_path = nearest_exchange_path(
                    [candidate_pixels(path, range(len(path))) for _pair, path in paths],
                    (mouse.x, mouse.y),
                )
            strongest = max(abs(pair.j_eff) for pair, _path in paths) if paths else 0.0
            moments = exchange_site_moments(state)
            for index, (pair, path) in enumerate(paths):
                hovered = index == hovered_path
                # Yellow where the selected configuration fights the coupling, the
                # same yellow the frustrated bars are ringed in, so the two views
                # agree on which bonds are the unhappy ones.
                frustrated = exchange_pair_frustration(pair, moments) > 0.0
                color = list(
                    EXCHANGE_FRUSTRATED_COLOR if frustrated else EXCHANGE_PATH_COLOR
                )
                color[3] = (
                    1.0 if hovered else exchange_path_alpha(pair.j_eff, strongest)
                )
                # Two labels, one per colour, so the legend can say what the yellow
                # means. Every path of a kind goes under the same label, which
                # ImPlot3D folds into one entry -- hiding a kind hides all of it
                # rather than leaving the rest of its bonds behind.
                implot3d.plot_line(
                    "Frustrated path" if frustrated else "Exchange path",
                    np.ascontiguousarray(path[:, 0], dtype=np.float64),
                    np.ascontiguousarray(path[:, 1], dtype=np.float64),
                    np.ascontiguousarray(path[:, 2], dtype=np.float64),
                    spec=implot3d.Spec(
                        line_color=tuple(color),
                        line_weight=(
                            EXCHANGE_PATH_HOVER_WEIGHT
                            if hovered
                            else EXCHANGE_PATH_WEIGHT
                        ),
                        marker=implot3d.Marker_.none,
                    ),
                )
            if hovered_path >= 0:
                pair = paths[hovered_path][0]
                imgui.set_tooltip(
                    exchange_pair_tooltip(
                        state, pair, exchange_pair_frustration(pair, moments)
                    )
                )
                # Ring both atoms the hovered coupling joins. A path that
                # crosses the cell boundary runs off one edge and arrives at
                # the far face, so the two ends are nowhere near each other on
                # screen -- ringing both is what says they are one bond.
                if analysis_structure is not None:
                    hover_extents = sphere_axis_extents(
                        render_radii, structure.lattice, use_cartesian
                    )
                    for end in (pair.site_i, pair.site_j):
                        draw_site_highlight_rings(
                            coords,
                            hover_extents,
                            highlighted_render_indices(
                                structure, analysis_structure, end
                            ),
                            color=PICK_HOVER_COLOR,
                        )

        # Sites whose spin disagrees with the matched ideal ordering -- the per-site
        # picture behind the defect concentration in the results panel.
        if state.show_spin_defect_rings and analysis_structure is not None:
            defect_extents = sphere_axis_extents(
                render_radii, structure.lattice, use_cartesian
            )
            for site in state.spin_defect_site_indices(state.selected_spin_config()):
                draw_site_highlight_rings(
                    coords,
                    defect_extents,
                    highlighted_render_indices(structure, analysis_structure, site),
                    color=SPIN_DEFECT_RING_COLOR,
                )

        if len(vacancy_coords):
            # A hole fades with whatever it sits among: off the focused plane,
            # or off the selected atom's exchange network, it is context like
            # any other site. Split into a bright half and a faded one rather
            # than given one alpha, since a plane generally cuts only some of
            # the vacancies -- a fully opaque hole floating in the faded cell
            # was reading as if it were in the plane.
            if view_mode == "exchange":
                bright_mask = np.zeros(len(vacancy_coords), dtype=bool)
            elif filter_to_slab:
                bright_mask = selected_vacancy_mask
            else:
                bright_mask = np.ones(len(vacancy_coords), dtype=bool)
            halves = ((bright_mask, 0.92), (~bright_mask, FADED_ATOM_ALPHA))
            # Drawn last so a vacancy is never hidden behind a neighbouring atom.
            for mask, alpha in halves:
                if not mask.any():
                    continue
                vacancy_mesh = build_sphere_mesh(
                    ensure_xyz_array(np.asarray(vacancy_coords)[mask]),
                    np.asarray(vacancy_radii)[mask],
                    structure.lattice,
                    use_cartesian=use_cartesian,
                    detail=sphere_detail,
                )
                implot3d.plot_mesh(
                    "Vacancy",
                    vacancy_mesh,
                    spec=implot3d.Spec(
                        fill_color=VACANCY_RENDER_COLOR,
                        line_color=VACANCY_RENDER_COLOR,
                        fill_alpha=alpha,
                        flags=implot3d.MeshFlags_.no_lines.value,
                    ),
                )

        # Last, so nothing draws over it.
        draw_axis_orientation_widget(axis_directions, plot_limits)

        implot3d.end_plot()

        # After end_plot: the overlay is its own window, and beginning one while
        # a plot is open would nest it inside the plot's item scope.
        draw_structure_summary_overlay(state, rect_min, rect_max)

    state.two_d_pane_fraction = draw_pane_splitter(
        "##structure_pane_splitter", state.two_d_pane_fraction, available.y
    )
    gui_two_d_pane(state)


def structure_load_controls(state: "AppState") -> None:
    """Loading, at the top of the builder panel beside New structure.

    One browse button and nothing else: a path is picked, never typed. On the
    web the button opens the browser's file picker instead, and dragging a file
    onto the 3D view still works.
    """
    if imgui.button("Load structure..."):
        if IS_PYODIDE:
            _open_browser_file_picker()
        else:
            try:
                selection = pfd.open_file(
                    "Select geometry file",
                    "",
                    ["Geometry files", "*.cif *.vasp *.poscar POSCAR* CONTCAR*", "All files", "*"],
                ).result()
            except Exception as exc:
                state.load_error = f"File dialog failed: {exc}"
                selection = []
            if selection:
                state.load_geometry(Path(selection[0]))
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            ".cif or VASP/POSCAR. Dragging a file onto the 3D view works too."
        )

    if state.load_error:
        imgui.push_style_color(imgui.Col_.text, (0.95, 0.35, 0.35, 1.0))
        imgui.text_wrapped(state.load_error)
        imgui.pop_style_color()
    elif state.status_message:
        imgui.text_wrapped(state.status_message)

    if state.geometry is not None:
        geometry = state.geometry
        # Collapsed by default: the list below is the working surface, and the
        # file's own accounting is reference material.
        if imgui.tree_node_ex("Loaded file details"):
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
            imgui.tree_pop()


def structure_export_controls(state: "AppState") -> None:
    """Export, at the very bottom of the Active Structure panel."""
    if IS_PYODIDE:
        # No filesystem to write to or browse, so the export comes back through
        # the browser instead. The folder row below would do nothing here.
        imgui.text_disabled("Exports arrive as a browser download; several files as a zip.")
    else:
        imgui.text_disabled("Output folder")
        # Widths come from the available width rather than a fixed 220px, so the
        # trailing button survives a narrow dock.
        style = imgui.get_style()
        browse_label = "Browse..."
        browse_width = (
            imgui.calc_text_size(browse_label).x + style.frame_padding.x * 2.0
        )
        field_width = max(
            80.0,
            imgui.get_content_region_avail().x - browse_width - style.item_spacing.x,
        )
        imgui.push_item_width(field_width)
        _, state.export_directory = imgui.input_text(
            "##export_directory",
            state.export_directory,
        )
        imgui.pop_item_width()
        imgui.same_line()
        if imgui.button(browse_label):
            try:
                selection = pfd.select_folder("Select export folder").result()
            except Exception as exc:
                state.export_message = f"Folder dialog failed: {exc}"
                selection = ""
            if selection:
                state.export_directory = selection

    if imgui.button("Export active structure", size=(180, 0)):
        state.export_active_structure()
    imgui.same_line()
    if imgui.button("Export all structures", size=(180, 0)):
        state.export_all_structures()
    if state.export_message:
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
    _structure_placeholder_note(state, structure)
    _structure_domain_dropdown(state, structure, reg_id)


def _structure_placeholder_note(state: "AppState", structure: ChemicalStructure) -> None:
    """``(relaxing, step N)`` beside a row that is waiting for its result."""
    status = state.remote_placeholder_status(structure)
    if status is None:
        return
    imgui.same_line()
    imgui.text_disabled(f"({status})")
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "This row holds the place of a running calculation and shows the "
            "structure that was submitted. The result replaces it when it "
            "arrives; deleting the row cancels the job."
        )


def _structure_domain_dropdown(
    state: "AppState", structure: ChemicalStructure, reg_id: int
) -> None:
    """A combo of the structure's domains, under its row, for a stacked structure.

    Picking one focuses the structure and points the builder at that domain.
    A single-domain structure has nothing to choose, so it gets no row.
    """
    params = getattr(structure, "generation_parameters", None)
    if params is None or not params.is_multi_domain():
        return
    axis_name = DOMAIN_AXIS_NAMES[int(params.stacking_axis)]
    labels = []
    for index, domain in enumerate(params.domain_specs()):
        layers = domain.oct_counts()[int(params.stacking_axis)]
        composition = domain_composition(domain)
        labels.append(f"Domain {index + 1}: {composition}, {layers} layers along {axis_name}")
    current = min(max(state.domain_index_for_structure(structure), 0), len(labels) - 1)
    imgui.indent()
    imgui.push_item_width(-1.0)
    changed, chosen = imgui.combo(f"##domains_of_struct{reg_id}", current, labels)
    imgui.pop_item_width()
    if imgui.is_item_hovered():
        imgui.set_tooltip(
            "The domain the builder edits. Picking one outlines its block in yellow "
            "in the view; clicking the structure's own row clears the outline."
        )
    imgui.unindent()
    if changed:
        state.choose_structure_domain(structure, chosen)


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
    _structure_placeholder_note(state, structure)
    _structure_domain_dropdown(state, structure, reg_id)
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

    # Right-clicking a row renames or deletes it; the context menu says so, so
    # the panel does not spend a line of text on it. New structure and loading
    # live at the top of the builder panel now, so this panel is just the list
    # with the export block under it.
    # The list lives in a child sized to everything above the export block, so
    # the export controls sit flush at the bottom of the panel and the list
    # gets all the room in between (scrolling on its own when it outgrows it).
    style = imgui.get_style()
    footer = imgui.get_frame_height_with_spacing()  # export buttons row
    if IS_PYODIDE:
        footer += imgui.get_text_line_height_with_spacing()  # download note
    else:
        footer += imgui.get_text_line_height_with_spacing()  # "Output folder"
        footer += imgui.get_frame_height_with_spacing()  # folder row
    if state.export_message:
        footer += (
            imgui.calc_text_size(
                state.export_message,
                wrap_width=max(1.0, imgui.get_content_region_avail().x),
            ).y
            + style.item_spacing.y
        )
    footer += style.item_spacing.y * 2.0 + 1.0  # the separator and its gaps

    imgui.begin_child("##structure_rows", imgui.ImVec2(0.0, -footer))
    registry: list = []
    for structure in list(state.structures):
        _render_structure_row(state, structure, registry)

    _rename_structure_popup(state)

    # Apply the deferred context-menu deletion after the list render.
    if state._pending_structure_delete is not None:
        state.remove_structure(state._pending_structure_delete)
        state._pending_structure_delete = None
    imgui.end_child()

    imgui.separator()
    structure_export_controls(state)


def create_docking_splits() -> List[hello_imgui.DockingSplit]:
    split_left = hello_imgui.DockingSplit()
    split_left.initial_dock = "MainDockSpace"
    split_left.new_dock = "ControlsSpace"
    split_left.direction = imgui.Dir.left
    split_left.ratio = 0.24

    # The Selection panel takes the lower part of the left side: the builder
    # above says what the cell is, the list below says what every atom in it is.
    split_atoms = hello_imgui.DockingSplit()
    split_atoms.initial_dock = "ControlsSpace"
    split_atoms.new_dock = "SelectionSpace"
    split_atoms.direction = imgui.Dir.down
    split_atoms.ratio = 0.42

    # The right-hand dock carries both calculation panels now, so it opens wider.
    split_right = hello_imgui.DockingSplit()
    split_right.initial_dock = "MainDockSpace"
    split_right.new_dock = "CalculationOutputSpace"
    split_right.direction = imgui.Dir.right
    split_right.ratio = 0.40

    split_active = hello_imgui.DockingSplit()
    split_active.initial_dock = "CalculationOutputSpace"
    split_active.new_dock = "ActiveStructureSpace"
    split_active.direction = imgui.Dir.up
    split_active.ratio = 0.32
    return [split_left, split_atoms, split_right, split_active]


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
    calculation_output.label = "Magnetism"
    calculation_output.dock_space_name = "CalculationOutputSpace"
    calculation_output.gui_function = gui_calculation_output

    # Tabbed with Magnetism: setup and results share the right-hand dock.
    calculate = hello_imgui.DockableWindow()
    calculate.label = "Calculate"
    calculate.dock_space_name = "CalculationOutputSpace"
    calculate.gui_function = gui_calculate

    # A third tab in the same dock: the view options left the Controls panel so
    # the builder has the whole left side.
    rendering = hello_imgui.DockableWindow()
    rendering.label = "Rendering"
    rendering.dock_space_name = "CalculationOutputSpace"
    rendering.gui_function = gui_rendering

    active = hello_imgui.DockableWindow()
    active.label = "Active Structure"
    active.dock_space_name = "ActiveStructureSpace"
    active.gui_function = gui_active_structure

    atoms = hello_imgui.DockableWindow()
    atoms.label = "Selection"
    atoms.dock_space_name = "SelectionSpace"
    atoms.gui_function = gui_atoms

    # Controls leads so a builder edit is applied before the panels that read it;
    # Atoms follows it for the same reason, and precedes the 3D view so an edit
    # made in a row is drawn the same frame.
    # Magnetism precedes Calculate so the results tab is the one on top.
    windows = [controls, atoms, structure, calculation_output, calculate, rendering, active]
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
