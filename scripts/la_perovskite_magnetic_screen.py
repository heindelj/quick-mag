#!/usr/bin/env python3
"""Screen La-based single and double perovskites: build -> CHGNet relax -> spin solve -> VASP inputs.

For every B-site element (and every B/B' pair) this
  1. builds the perovskite with ``quick_mag.build_cli`` on a 4x2x2 pseudocubic
     B-site grid,
  2. perturbs the cell and the atoms off the symmetric seed, then relaxes cell +
     positions with CHGNet,
  3. predicts oxidation states, exchange couplings and the low-energy collinear
     spin configurations, and
  4. writes the relaxed geometry (CIF + POSCAR), one MAGMOM line per magnetic
     ordering, a summary CSV, and a ready-to-run VASP directory per ordering.

Why 4x2x2 and not 2x2x2 or 3x3x3
--------------------------------
The canonical orderings are sign strings across planes of the B-site lattice, so
the grid has to be commensurate with them:

  * F, A(a/b/c), C(a/b/c) and G have period 2 -- they need an even number of B
    planes along every axis they alternate on. An odd grid (3x3x3) wraps the
    pattern onto itself and the "A-type" you score is really A-type with a plane
    of frustrated bonds at the periodic boundary.
  * E-type is an up-up-down-down modulation, period 4, so it needs *four* planes
    along its axis. ``canonical_reference_patterns((2,2,2))`` and ``((3,3,3))``
    both drop E entirely for that reason.

4x2x2 is the smallest grid that carries all of F, A, C, G and E(a) exactly. It is
also 16 magnetic sites, which is exactly the solver's exact-enumeration limit, so
the ground state is found by exhaustive Ising enumeration rather than an
optimizer. Doubles keep their rocksalt B/B' ordering on this grid.

Why the seed is perturbed
-------------------------
A builder seed is a stationary point of the potential: every force and every
stress component vanishes by symmetry, so a gradient optimizer has no direction
to move in and "converges" on the symmetric structure no matter how tight --fmax
is. That is not a minimum -- for a Jahn-Teller ion like Mn(3+) it is a saddle,
and the real minimum sits well below it with the octahedra distorted.

So every seed gets a random symmetric strain on the cell (--strain, fractional)
and a Gaussian rattle on the atoms (--rattle, Angstrom) before CHGNet sees it.
That breaks the symmetry, leaves a real gradient, and lets the relaxation fall
into a distorted basin.

One perturbation samples one basin. --n-restarts N relaxes N independently
perturbed copies and keeps the lowest-energy one; the per-restart energies are
printed and their spread goes in the CSV, which is how you tell whether the
basin is unique or the search is still missing something.

Atom order
----------
Every file written for one structure -- CIF, POSCAR, magmom lines, INCAR MAGMOM --
shares a single atom order, grouped into one block per element by
``quick_mag.vasp_io.grouped_by_species``. Atom order is the one thing a MAGMOM has
to agree with, so it is fixed once and everything is written from it.

Usage
-----
    python la_perovskite_magnetic_screen.py -o screen_out
    python la_perovskite_magnetic_screen.py -o screen_out --n-restarts 3
    python la_perovskite_magnetic_screen.py -o screen_out --potcar-dir ~/vasp/potpaw_PBE
    python la_perovskite_magnetic_screen.py --only-singles --fmax 0.02
    python la_perovskite_magnetic_screen.py --b-elements Fe,Mn --dry-run

Requires the CHGNet extra: ``pip install -e ".[chgnet]"``.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import time
import traceback
import zlib
from dataclasses import replace
from pathlib import Path

import numpy as np

from quick_mag.build_cli import build_structures, configure_build_parser
from quick_mag.classify_spin_structure import classify_structure
from quick_mag.export_utils import export_structure, format_magmom_line, sanitize_filename
from quick_mag.ion_descriptors import structure_ion_descriptors
from quick_mag.magnetic_cli import _resolve_site_indexing
from quick_mag.magnetic_moments import (
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
from quick_mag.spin_solver import (
    NO_NONZERO_MAGNETIC_MOMENTS_MESSAGE,
    solve_for_assignment,
)
from quick_mag.structure import ChemicalStructure, SavedSpinConfiguration
from quick_mag.vasp_io import grouped_by_species, write_poscar

# ---------------------------------------------------------------------------
# What to screen
# ---------------------------------------------------------------------------

DEFAULT_B_ELEMENTS = ("Cr", "Mn", "Fe", "Co", "Ni")
A_SITE = "La"
X_SITE = "O"

# The B-site grid. 4 along a is what makes E(a) representable; see the module
# docstring. Change it and re-check REFERENCE_PATTERNS below.
N_CELLS = (4, 2, 2)

# Cubic seed lattice parameter per pseudocubic cell (A). CHGNet relaxes the cell,
# so this only has to be a sane starting point.
SEED_A = 3.90

# The orderings scored for every structure, in report order. Each is scored
# independently of the solver, so they are reported whether or not the solver
# lands on them. A, C and E single out one axis of the B grid, which is not
# equivalent to the other two on a 4x2x2 cell, hence the explicit orientations.
# E only has an (a) orientation here: b and c have only two planes.
REFERENCE_PATTERNS = (
    "F",
    "A(a)",
    "A(b)",
    "A(c)",
    "C(a)",
    "C(b)",
    "C(c)",
    "G",
    "E(a)",
)


# ---------------------------------------------------------------------------
# VASP settings
#
# Ported from quick_magnetic_configs/quick_mag_v0.1/vasp_inputs.py so this script
# has no pymatgen dependency. The INCAR keys, the U values and the k-point
# density rule are that project's; only the plumbing is new.
# ---------------------------------------------------------------------------

#: Elements treated as +U centres (LDAUL = 2). From that project's METALS set.
METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Al", "Ga", "In", "Sn", "Pb", "Bi", "Sb", "Tl", "Ge",
}

#: Effective Hubbard U (eV). ``None`` means "a +U centre with no tabulated value",
#: which falls back to DEFAULT_U; an element missing from the table gets 0.
U_VALUES_METALS = {
    "V": 3.25, "Cr": 3.7, "Mn": 3.9, "Fe": 5.3, "Co": 3.32, "Ni": 6.2,
    "Mo": 4.38, "W": 6.2,
    "Sc": None, "Ti": None, "Cu": None, "Zn": None, "Y": None, "Zr": None,
    "Nb": None, "Tc": None, "Ru": None, "Rh": None, "Pd": None, "Ag": None,
    "Cd": None, "La": None, "Hf": None, "Ta": None, "Re": None, "Os": None,
    "Ir": None, "Pt": None, "Au": None, "Hg": None, "Al": None, "Ga": None,
    "In": None, "Sn": None, "Pb": None, "Bi": None, "Sb": None, "Tl": None,
    "Ge": None,
}
DEFAULT_U = 5.0

#: Everything except LDAU* (per-species, built below) and MAGMOM (per-atom).
#: No NSW/IBRION, so VASP's defaults make this a static run -- the geometry is
#: already CHGNet-relaxed and only the spin state varies between directories.
BASE_INCAR = {
    "ALGO": "Normal",
    "EDIFF": 1e-6,
    "ENCUT": 520,
    "ISMEAR": -5,
    "ISPIN": 2,
    "ISYM": 0,
    "LCHARG": True,
    "LDAU": True,
    "LDAUTYPE": 2,
    "LMAXMIX": 4,
    "LNONCOLLINEAR": False,
    "LORBIT": 11,
    "LWAVE": False,
    "NCORE": 8,
    "NELM": 150,
    "PREC": "Accurate",
}


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    return configure_build_parser(argparse.ArgumentParser(add_help=False))


def build_single(b_element: str) -> ChemicalStructure:
    """One LaBO3 cell on the N_CELLS B grid."""
    argv = [
        "--formula", "perovskite",
        "--a-site", A_SITE,
        "--b-site", b_element,
        "--x-site", X_SITE,
        "--a", str(SEED_A),
        "--n-cells-x", str(N_CELLS[0]),
        "--n-cells-y", str(N_CELLS[1]),
        "--n-cells-z", str(N_CELLS[2]),
        "--name", f"{A_SITE}{b_element}{X_SITE}3_{''.join(map(str, N_CELLS))}",
    ]
    return build_structures(_build_parser().parse_args(argv))[0]


def build_double(b_element: str, b2_element: str) -> ChemicalStructure:
    """One La2BB'O6 cell with rocksalt B/B' ordering on the N_CELLS B grid."""
    argv = [
        "--formula", "double",
        "--a-site", A_SITE,
        "--b-site", b_element,
        "--b2-site", b2_element,
        "--x-site", X_SITE,
        "--a", str(SEED_A),
        "--n-cells-x", str(N_CELLS[0]),
        "--n-cells-y", str(N_CELLS[1]),
        "--n-cells-z", str(N_CELLS[2]),
        "--name", f"{A_SITE}2{b_element}{b2_element}{X_SITE}6_{''.join(map(str, N_CELLS))}",
    ]
    return build_structures(_build_parser().parse_args(argv))[0]


def enumerate_systems(b_elements, want_singles: bool, want_doubles: bool):
    """``(kind, name, builder)`` for every system to run, singles first."""
    jobs = []
    if want_singles:
        for element in b_elements:
            jobs.append(("single", element, lambda e=element: build_single(e)))
    if want_doubles:
        # Unordered pairs: LaFeMn and LaMnFe are the same rocksalt cell.
        for first, second in itertools.combinations(b_elements, 2):
            jobs.append(
                (
                    "double",
                    f"{first}/{second}",
                    lambda a=first, b=second: build_double(a, b),
                )
            )
    return jobs


# ---------------------------------------------------------------------------
# Breaking the seed's symmetry
# ---------------------------------------------------------------------------


def perturb_structure(
    structure: ChemicalStructure,
    *,
    rattle: float,
    strain: float,
    rng: np.random.Generator,
) -> ChemicalStructure:
    """A copy of ``structure`` with a strained cell and rattled atoms.

    ``strain`` is the standard deviation of a random *symmetric* strain tensor
    applied as ``lattice @ (I + eps)``; symmetric so the cell is deformed rather
    than rotated, which would change nothing physical. The atoms ride along with
    the cell affinely (their fractional coordinates are preserved), then take an
    independent Gaussian displacement of standard deviation ``rattle`` Angstrom.

    Atom order and count are untouched, so ``generation_parameters`` still
    describes the topology the B-site indexing reads.
    ``geometry_matches_generation`` is cleared because the geometry is no longer
    what those parameters rebuild.
    """
    lattice = np.asarray(structure.lattice, dtype=np.float64)
    fractional = np.asarray(structure.fractional_coords, dtype=np.float64)

    if strain > 0.0:
        eps = rng.normal(0.0, float(strain), size=(3, 3))
        eps = 0.5 * (eps + eps.T)
        lattice = lattice @ (np.eye(3) + eps)

    coords = fractional @ lattice
    if rattle > 0.0:
        coords = coords + rng.normal(0.0, float(rattle), size=coords.shape)

    return ChemicalStructure.with_zero_magnetic_moments(
        name=structure.name,
        lattice=lattice,
        cartesian_coords=coords,
        atomic_labels=list(structure.atomic_labels),
        is_periodic=structure.is_periodic,
        periodic_axes=getattr(structure, "periodic_axes", None),
        generation_parameters=structure.generation_parameters,
        geometry_matches_generation=False,
    )


def octahedral_distortion(structure: ChemicalStructure, b_elements) -> float:
    """Mean spread (max - min, Angstrom) of the six B-X distances per octahedron.

    Zero on a symmetric seed and non-zero once the octahedra distort, so it is
    the cheapest direct read on whether the relaxation actually left the
    high-symmetry structure -- a Jahn-Teller Mn(3+) octahedron should come out
    around 0.2-0.4 A. Distances use the minimum-image convention.
    """
    symbols = list(structure.element_symbols())
    b_sites = [i for i, s in enumerate(symbols) if s in b_elements]
    x_sites = [i for i, s in enumerate(symbols) if s == X_SITE]
    if not b_sites or len(x_sites) < 6:
        return float("nan")

    lattice = np.asarray(structure.lattice, dtype=np.float64)
    fractional = np.asarray(structure.fractional_coords, dtype=np.float64)
    x_fractional = fractional[x_sites]

    spreads = []
    for site in b_sites:
        delta = x_fractional - fractional[site]
        delta -= np.round(delta)  # minimum image
        distances = np.sort(np.linalg.norm(delta @ lattice, axis=1))[:6]
        spreads.append(float(distances[-1] - distances[0]))
    return float(np.mean(spreads)) if spreads else float("nan")


# ---------------------------------------------------------------------------
# Magnetism
# ---------------------------------------------------------------------------


def label_tag(label: str) -> str:
    """A filename-safe tag for an ordering: ``A(a)`` -> ``Aa``, ``GS[C]`` -> ``GS``."""
    if label.startswith("GS"):
        return "GS"
    return sanitize_filename(label.replace("(", "").replace(")", ""))


def _scatter(compact: np.ndarray, magnetic_site_indices, n_atoms: int) -> np.ndarray:
    """Compact per-magnetic-site moments -> full (N, 3) array along z."""
    full = np.zeros((n_atoms, 3), dtype=np.float64)
    values = np.asarray(compact, dtype=np.float64).reshape(-1)
    for row, site in enumerate(magnetic_site_indices):
        if row < values.size and 0 <= int(site) < n_atoms:
            full[int(site), 2] = float(values[row])
    return full


def _classify(structure, full_moments, site_indexing) -> str:
    """F/A/C/G/E/Other for a full (N, 3) moment array, or "" if unclassifiable."""
    if site_indexing is None:
        return ""
    try:
        probe = ChemicalStructure.with_zero_magnetic_moments(
            name=structure.name,
            lattice=np.array(structure.lattice, dtype=np.float64, copy=True),
            cartesian_coords=np.array(
                structure.cartesian_coords, dtype=np.float64, copy=True
            ),
            atomic_labels=list(structure.atomic_labels),
            is_periodic=structure.is_periodic,
        )
        probe.magnetic_moments[:] = full_moments
        return classify_structure(probe, site_indexing=site_indexing).label
    except Exception:
        return ""


def solve_magnetism(structure: ChemicalStructure, args) -> dict:
    """Oxidation states -> exchange -> reference orderings + solved ground state.

    Returns a dict with the assignment, the per-site formal moments, and a list of
    ``(label, energy, full_moments)`` entries: every pattern in
    ``REFERENCE_PATTERNS`` that the grid supports, followed by the solver's ground
    state. Raises ``RuntimeError`` when the structure has no solvable magnetism.
    """
    labels = structure.element_symbols()
    ranked = enumerate_oxidation_states_by_energy(
        labels, charge=args.charge, max_mixing=args.max_mixing, top_k=1
    )
    if not ranked:
        raise RuntimeError("no charge-balanced oxidation-state assignment found")

    assignments = expand_distribution_to_site_assignments([ranked[0][0]], structure)
    if not assignments:
        raise RuntimeError("oxidation distribution could not be placed on sites")
    assignment = assignments[0]

    descriptors = structure_ion_descriptors(structure, assignment)
    if not descriptors:
        raise RuntimeError("no transition-metal magnetic sites in this assignment")
    magnetic_site_indices = sorted(descriptors)

    bridges = build_bridges(structure, descriptors)
    if not bridges:
        raise RuntimeError("no M-L-M superexchange bridges found")

    site_index = {site: i for i, site in enumerate(magnetic_site_indices)}
    j_matrix = to_solver_couplings(
        build_Jeff_matrix(bridges, site_index, default_params())
    )
    site_indexing = _resolve_site_indexing(structure, magnetic_site_indices)
    if site_indexing is None:
        raise RuntimeError(
            "magnetic sublattice is not a perovskite B grid; reference orderings "
            "are not defined"
        )

    # The polarization J already carries spin magnitude, so the solver and the
    # reference scoring both run on unit (+-1) spins. Physical magnitudes come
    # back at export time from assignment.magnetic_moments.
    unit_assignment = replace(
        assignment,
        magnetic_moments=(np.abs(assignment.magnetic_moments) > 1e-8).astype(float),
    )

    reference = named_reference_spin_configs(
        structure,
        unit_assignment,
        j_matrix,
        magnetic_site_indices,
        site_indexing,
        patterns=REFERENCE_PATTERNS,
    )

    n_mag = len(magnetic_site_indices)
    method = "exact" if n_mag <= args.exact_max_sites else "optimizer"
    try:
        _base, all_states = solve_for_assignment(
            unit_assignment,
            j_matrix,
            magnetic_site_indices=magnetic_site_indices,
            method=method,
            collinear=True,
            n_trials=args.n_trials,
            n_steps=args.n_steps,
        )
    except ValueError as exc:
        if str(exc) == NO_NONZERO_MAGNETIC_MOMENTS_MESSAGE:
            raise RuntimeError("no unpaired electrons on any site") from exc
        raise

    n_atoms = structure.atom_count
    entries = [
        (
            str(name),
            float(config.energy),
            _scatter(config.all_moments, magnetic_site_indices, n_atoms),
        )
        for name, config in reference
    ]

    # The overall ground state across the solved search and the reference set.
    candidates = list(all_states) + [config for _n, config in reference]
    ground = min(candidates, key=lambda c: c.energy)
    ground_moments = _scatter(ground.all_moments, magnetic_site_indices, n_atoms)
    ground_label = _classify(structure, ground_moments, site_indexing)
    entries.append((f"GS[{ground_label or 'other'}]", float(ground.energy), ground_moments))

    return {
        "assignment": assignment,
        "distribution": format_oxidation_distribution(assignment.distributions),
        "magnitudes": np.asarray(assignment.magnetic_moments, dtype=np.float64),
        "entries": entries,
        "ground_label": ground_label,
        "ground_energy": float(ground.energy),
        "method": method,
        "n_magnetic_sites": n_mag,
        "site_indexing": site_indexing,
    }


# ---------------------------------------------------------------------------
# VASP inputs
# ---------------------------------------------------------------------------


def species_order(structure: ChemicalStructure) -> list[str]:
    """Element symbols in POSCAR block order (the structure must be grouped)."""
    order: list[str] = []
    for symbol in structure.element_symbols():
        if symbol not in order:
            order.append(symbol)
    return order


def ldau_arrays(elements) -> dict:
    """``LDAUL``/``LDAUU``/``LDAUJ``, one entry per POSCAR species block.

    A +U centre gets ``LDAUL = 2`` and its tabulated U; everything else gets
    ``LDAUL = -1`` and zero. An element listed in the table with no value is still
    a +U centre and falls back to :data:`DEFAULT_U`.
    """
    ldaul, ldauu, ldauj = [], [], []
    for element in elements:
        if element in METALS:
            u_value = U_VALUES_METALS.get(element)
            ldaul.append(2)
            ldauu.append(float(DEFAULT_U if u_value is None else u_value))
            ldauj.append(0.0)
        else:
            ldaul.append(-1)
            ldauu.append(0.0)
            ldauj.append(0.0)
    return {"LDAUL": ldaul, "LDAUU": ldauu, "LDAUJ": ldauj}


def kpoint_grid(
    structure: ChemicalStructure, density: float = 1000.0, *, isotropic: bool = True
) -> tuple[int, int, int]:
    """A Gamma-centred grid at roughly ``density`` k-points per reciprocal atom.

    The per-axis grid is ``n_i = (target / (b1 b2 b3))^(1/3) * b_i`` with
    ``target = density / n_atoms``, which puts more divisions along the short real
    axes. Any 2*pi in the reciprocal lengths cancels out of that expression, so the
    convention does not matter.

    ``isotropic`` then averages the three (rounding up) into a cubic grid, which is
    what the original script does. Turn it off to keep the per-axis grid, which is
    the cheaper and more even choice on an elongated cell like 4x2x2.
    """
    reciprocal = np.linalg.inv(np.asarray(structure.lattice, dtype=np.float64)).T
    lengths = np.linalg.norm(reciprocal, axis=1)
    target = float(density) / max(1, structure.atom_count)
    scale = (target / float(np.prod(lengths))) ** (1.0 / 3.0)
    grid = [max(1, int(round(scale * length))) for length in lengths]
    if isotropic:
        average = int(math.ceil(sum(grid) / 3.0))
        return (average, average, average)
    return (grid[0], grid[1], grid[2])


def format_incar(incar: dict) -> str:
    """VASP-friendly INCAR text: booleans as ``.TRUE.``, sequences space-joined."""
    lines = []
    for key, value in sorted(incar.items()):
        if isinstance(value, bool):
            rendered = ".TRUE." if value else ".FALSE."
        elif isinstance(value, (list, tuple, np.ndarray)):
            rendered = " ".join(
                str(int(item)) if float(item) == int(item) else f"{float(item):g}"
                for item in value
            )
        elif isinstance(value, float) and value == int(value):
            rendered = str(int(value))
        else:
            rendered = str(value)
        lines.append(f"{key} = {rendered}")
    return "\n".join(lines) + "\n"


def format_kpoints(grid) -> str:
    """A Gamma-centred automatic KPOINTS file."""
    return (
        "Automatic Gamma-centred mesh\n"
        "0\n"
        "Gamma\n"
        f"{grid[0]} {grid[1]} {grid[2]}\n"
        "0 0 0\n"
    )


def write_potcar(elements, potcar_dir: Path, path: Path) -> None:
    """Concatenate ``<potcar_dir>/<element>/POTCAR`` in POSCAR species order."""
    sources = [Path(potcar_dir) / element / "POTCAR" for element in elements]
    missing = [str(source) for source in sources if not source.exists()]
    if missing:
        raise FileNotFoundError(f"POTCAR not found: {', '.join(missing)}")
    with path.open("wb") as destination:
        for source in sources:
            destination.write(source.read_bytes())


def write_vasp_config(
    structure: ChemicalStructure,
    magmoms: np.ndarray,
    out_dir: Path,
    *,
    args,
    metadata: dict,
) -> tuple[int, int, int]:
    """Write INCAR, POSCAR, KPOINTS and (optionally) POTCAR for one spin state.

    ``structure`` must already be grouped by species, and ``magmoms`` must be one
    signed scalar per atom in that same order -- the POSCAR is written from the
    structure and MAGMOM straight from the array, so the two cannot drift apart.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    elements = species_order(structure)

    grid = kpoint_grid(
        structure, args.kpoints_density, isotropic=not args.anisotropic_kpoints
    )

    incar = dict(BASE_INCAR)
    incar.update(ldau_arrays(elements))
    if incar.get("ISMEAR") != -5:
        incar.setdefault("SIGMA", 0.05)
    incar.update(args.incar_overrides)
    incar["MAGMOM"] = [float(value) for value in np.asarray(magmoms).reshape(-1)]

    if incar.get("ISMEAR") == -5 and min(grid) < 2:
        # The tetrahedron method wants a genuinely 3D mesh; a grid with a
        # single division along an axis is the usual cause of VASP's
        # "Fatal error detecting k-mesh" here.
        print(
            f"    WARNING: ISMEAR=-5 with a {grid[0]}x{grid[1]}x{grid[2]} k-mesh. "
            "Raise --kpoints-density, drop --anisotropic-kpoints, or set "
            "--incar ISMEAR=0 --incar SIGMA=0.05."
        )

    write_poscar(structure, out_dir / "POSCAR", comment=metadata["comment"])
    (out_dir / "INCAR").write_text(format_incar(incar))
    (out_dir / "KPOINTS").write_text(format_kpoints(grid))
    if args.potcar_dir is not None:
        write_potcar(elements, Path(args.potcar_dir).expanduser(), out_dir / "POTCAR")

    (out_dir / "config.json").write_text(
        json.dumps({**metadata, "kpoint_grid": list(grid)}, indent=2, sort_keys=True)
        + "\n"
    )
    return grid


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(
    relaxed: ChemicalStructure,
    magnetism: dict,
    args,
    chgnet_note: str,
) -> dict:
    """Write the geometry, magmom files, per-ordering VASP directories and manifest.

    Everything is written from one species-grouped copy of the structure, so line
    *i* of any magmom file is atom *i* of the CIF, the POSCAR and every VASP POSCAR
    underneath.
    """
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(relaxed.name)
    magnitudes = magnetism["magnitudes"]

    # Stage every ordering onto the structure, then group by species once: the
    # reorder carries the moments and the magnitudes with it.
    staged = ChemicalStructure.with_zero_magnetic_moments(
        name=relaxed.name,
        lattice=np.array(relaxed.lattice, dtype=np.float64, copy=True),
        cartesian_coords=np.array(relaxed.cartesian_coords, dtype=np.float64, copy=True),
        atomic_labels=list(relaxed.atomic_labels),
        is_periodic=relaxed.is_periodic,
    )
    for label, energy, moments in magnetism["entries"]:
        staged.spin_configurations.append(
            SavedSpinConfiguration(
                magnetic_moments=moments,
                energy=float(energy),
                classification=label,
                collinear=True,
                site_moment_magnitudes=np.array(magnitudes, copy=True),
            )
        )
    ordered = grouped_by_species(staged)

    # <stem>.cif, <stem>.vasp and <stem>_spins.txt, all in that one atom order.
    export_structure(ordered, out_dir, formats=("cif", "vasp"))

    # One magmom line per ordering, scaled to the formal per-site moments.
    lines = {
        config.classification: format_magmom_line(
            config.magnetic_moments, True, config.site_moment_magnitudes
        )
        for config in ordered.spin_configurations
    }

    incar_dir = out_dir / f"{stem}_MAGMOM"
    incar_dir.mkdir(parents=True, exist_ok=True)
    for config in ordered.spin_configurations:
        label = config.classification
        (incar_dir / f"MAGMOM_{label_tag(label)}.txt").write_text(
            f"# {relaxed.name}  ordering={label}  "
            f"quick-mag model energy={config.energy:.6f}\n"
            f"# atom order matches {stem}.vasp and {stem}.cif\n"
            f"MAGMOM = {lines[label]}\n"
        )

    references = [c for c in ordered.spin_configurations if not c.classification.startswith("GS")]
    e_ref = min(config.energy for config in references)

    # A VASP run directory per ordering, all under one folder, one subfolder per
    # perovskite.
    vasp_root = Path(args.vasp_dir).expanduser() if args.vasp_dir else out_dir / "vasp"
    grids = {}
    if not args.no_vasp:
        system_dir = vasp_root / stem
        for config in ordered.spin_configurations:
            label = config.classification
            magmoms = [float(value) for value in lines[label].split()]
            grids[label] = write_vasp_config(
                ordered,
                np.asarray(magmoms),
                system_dir / label_tag(label),
                args=args,
                metadata={
                    "comment": f"{relaxed.name}  {label}  CHGNet-relaxed",
                    "system": relaxed.name,
                    "ordering": label,
                    "quick_mag_model_energy": float(config.energy),
                    "dE_vs_best_reference": float(config.energy - e_ref),
                    "is_predicted_ground_state": label.startswith("GS"),
                    "oxidation_states": magnetism["distribution"],
                    "chgnet": chgnet_note,
                    "magmom": magmoms,
                    "atom_labels": list(ordered.element_symbols()),
                },
            )

    manifest = [
        f"# {relaxed.name}",
        f"# CHGNet: {chgnet_note}",
        f"# oxidation states: {magnetism['distribution']}",
        f"# spin solve: {magnetism['n_magnetic_sites']} magnetic sites, "
        f"method={magnetism['method']}",
        f"# predicted ground state: {magnetism['ground_label'] or 'other'}",
        "#",
        f"# line -> ordering in {stem}_spins.txt "
        f"(atom order = {stem}.cif = {stem}.vasp = every VASP POSCAR below)",
        f"# {'line':>4}  {'ordering':>10}  {'energy':>14}  {'dE_vs_best_ref':>16}",
    ]
    for index, config in enumerate(ordered.spin_configurations, start=1):
        label = config.classification
        note = ""
        if label.startswith("GS"):
            # Is the solver's ground state literally one of the references, or an
            # ordering the canonical set does not contain? Compared up to a global
            # spin flip, which is the same state.
            same = [
                other.classification
                for other in references
                if np.allclose(config.magnetic_moments, other.magnetic_moments)
                or np.allclose(config.magnetic_moments, -other.magnetic_moments)
            ]
            note = (
                f"   # same state as {same[0]}"
                if same
                else "   # not one of the canonical orderings"
            )
        manifest.append(
            f"  {index:>4}  {label:>10}  {config.energy:>14.6f}"
            f"  {config.energy - e_ref:>16.6f}{note}"
        )
    if grids:
        manifest.append("#")
        manifest.append(f"# VASP inputs: {vasp_root / stem}")
        manifest.append(f"# k-point grid: {grids[next(iter(grids))]}")
    (out_dir / f"{stem}_orderings.txt").write_text("\n".join(manifest) + "\n")

    return {
        "stem": stem,
        "configs": list(ordered.spin_configurations),
        "e_ref": e_ref,
        "vasp_dir": (vasp_root / stem) if grids else None,
        "kpoint_grid": grids.get(next(iter(grids))) if grids else None,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def relax_best(seed: ChemicalStructure, args, calculator, b_elements):
    """Relax ``--n-restarts`` perturbed copies of ``seed``; keep the lowest.

    Each restart gets its own perturbation from a seeded generator, so a rerun
    with the same ``--seed`` reproduces the same geometries exactly.
    """
    from quick_mag.chgnet_runner import run_chgnet_calculation

    calculation = "atoms" if args.fix_cell else "cell+atoms"
    energies = []
    best = None

    for restart in range(max(1, args.n_restarts)):
        # Seeded per (system, restart) so one system's geometry does not depend
        # on how many other systems ran before it. crc32, not hash(): Python
        # randomises string hashing per process, which would quietly make --seed
        # reproduce nothing.
        rng = np.random.default_rng(
            [zlib.crc32(seed.name.encode()), int(args.seed), restart]
        )
        start = perturb_structure(seed, rattle=args.rattle, strain=args.strain, rng=rng)
        result = run_chgnet_calculation(
            start,
            calculation,
            optimizer=args.optimizer,
            fmax=args.fmax,
            steps=args.steps,
            verbose=args.verbose,
            calculator=calculator,
        )
        energies.append(float(result.energy))
        distortion = octahedral_distortion(result.final_structure, b_elements)
        flag = "" if result.converged else "  NOT CONVERGED"
        print(
            f"    restart {restart + 1}/{max(1, args.n_restarts)}: "
            f"E={result.energy:.6f} eV  steps={result.steps}  "
            f"fmax={result.max_force:.5f}  B-X spread={distortion:.3f} A{flag}"
        )
        if best is None or result.energy < best[0].energy:
            best = (result, restart, distortion)

    return best[0], best[1], best[2], energies


def run_one(kind: str, label: str, builder, args, calculator) -> dict:
    """Build -> perturb -> relax -> solve -> write for one system. Returns a CSV row."""
    started = time.time()
    seed = builder()
    b_elements = set(seed.element_symbols()) - {A_SITE, X_SITE}
    print(f"\n{'=' * 78}\n{seed.name}  ({kind}, {seed.atom_count} atoms)\n{'=' * 78}")
    print(
        f"  Perturbing the seed: rattle={args.rattle} A, strain={args.strain}, "
        f"{max(1, args.n_restarts)} restart(s)."
    )

    if args.seeds_dir:
        seeds = Path(args.seeds_dir).expanduser()
        seeds.mkdir(parents=True, exist_ok=True)
        export_structure(seed, seeds)

    result, best_restart, distortion, energies = relax_best(
        seed, args, calculator, b_elements
    )
    spread = (max(energies) - min(energies)) if len(energies) > 1 else 0.0
    relaxed = result.final_structure
    chgnet_note = (
        f"E={result.energy:.6f} eV  ({result.energy_per_atom:.6f} eV/atom)  "
        f"steps={result.steps}  fmax={result.max_force:.5f} eV/A  "
        f"converged={result.converged}  restart={best_restart + 1}/{len(energies)}  "
        f"B-X spread={distortion:.3f} A"
    )
    print(f"  CHGNet best: {chgnet_note}")
    if len(energies) > 1:
        print(f"  Restart energy spread: {spread:.6f} eV")
    if not result.converged:
        print("  WARNING: CHGNet did not reach fmax; geometry is not a minimum.")
    if not np.isnan(distortion) and distortion < 0.01:
        print(
            "  WARNING: octahedra came out essentially undistorted -- the relaxation "
            "may have fallen back onto the symmetric structure. Try a larger "
            "--rattle/--strain or more --n-restarts."
        )

    magnetism = solve_magnetism(relaxed, args)
    written = write_outputs(relaxed, magnetism, args, chgnet_note)

    print("\n  Reference orderings (dE vs lowest reference):")
    for config in written["configs"]:
        print(
            f"    {config.classification:>10}  {config.energy:>14.6f}"
            f"  {config.energy - written['e_ref']:>12.6f}"
        )
    print(f"  Predicted ground state: {magnetism['ground_label'] or 'other'}")
    if written["vasp_dir"] is not None:
        print(
            f"  VASP inputs: {written['vasp_dir']}  "
            f"(k-points {written['kpoint_grid']})"
        )
    print(
        f"  Wrote {written['stem']}.* to {args.output_dir}  "
        f"({time.time() - started:.1f} s)"
    )

    row = {
        "name": relaxed.name,
        "kind": kind,
        "b_sites": label,
        "n_atoms": relaxed.atom_count,
        "n_magnetic_sites": magnetism["n_magnetic_sites"],
        "solve_method": magnetism["method"],
        "chgnet_energy_eV": f"{result.energy:.6f}",
        "chgnet_energy_eV_per_atom": f"{result.energy_per_atom:.6f}",
        "chgnet_steps": result.steps,
        "chgnet_max_force_eV_A": f"{result.max_force:.6f}",
        "chgnet_converged": result.converged,
        "n_restarts": len(energies),
        "best_restart": best_restart + 1,
        "restart_spread_eV": f"{spread:.6f}",
        "bx_bond_spread_A": f"{distortion:.4f}",
        "oxidation_states": magnetism["distribution"],
        "ground_state": magnetism["ground_label"] or "other",
        "ground_state_energy": f"{magnetism['ground_energy']:.6f}",
        "kpoint_grid": (
            "x".join(str(n) for n in written["kpoint_grid"])
            if written["kpoint_grid"]
            else ""
        ),
        "vasp_dir": str(written["vasp_dir"]) if written["vasp_dir"] else "",
        "status": "ok",
        "note": "",
    }
    by_label = {config.classification: config.energy for config in written["configs"]}
    for pattern in REFERENCE_PATTERNS:
        energy = by_label.get(pattern)
        row[f"E_{pattern}"] = "" if energy is None else f"{energy:.6f}"
        row[f"dE_{pattern}"] = (
            "" if energy is None else f"{energy - written['e_ref']:.6f}"
        )
    return row


CSV_FIELDS = (
    [
        "name", "kind", "b_sites", "n_atoms", "n_magnetic_sites", "solve_method",
        "chgnet_energy_eV", "chgnet_energy_eV_per_atom", "chgnet_steps",
        "chgnet_max_force_eV_A", "chgnet_converged",
        "n_restarts", "best_restart", "restart_spread_eV", "bx_bond_spread_A",
        "oxidation_states", "ground_state", "ground_state_energy",
    ]
    + [f"{prefix}_{pattern}" for pattern in REFERENCE_PATTERNS for prefix in ("E", "dE")]
    + ["kpoint_grid", "vasp_dir", "status", "note"]
)


def parse_incar_overrides(values) -> dict:
    """``["ENCUT=600", "NCORE=4"]`` -> ``{"ENCUT": 600, "NCORE": 4}``.

    Values are read as int, then float, then a bare ``.TRUE.``/``.FALSE.``, then
    left as text -- so ``ALGO=Fast`` and ``LWAVE=.TRUE.`` both work.
    """
    overrides: dict = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError(f"--incar expects KEY=VALUE, got {item!r}")
        key, _, raw = item.partition("=")
        text = raw.strip()
        upper = text.upper()
        if upper in (".TRUE.", "TRUE", ".FALSE.", "FALSE"):
            parsed = upper.startswith(".T") or upper == "TRUE"
        else:
            try:
                parsed = int(text)
            except ValueError:
                try:
                    parsed = float(text)
                except ValueError:
                    parsed = text
        overrides[key.strip().upper()] = parsed
    return overrides


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "-o", "--output-dir", default="la_perovskite_screen",
        help="Directory for relaxed structures, magmom files and the CSV.",
    )
    parser.add_argument(
        "--seeds-dir", default=None,
        help="Also write the unrelaxed builder seeds here (default: not kept).",
    )
    parser.add_argument(
        "--b-elements", default=",".join(DEFAULT_B_ELEMENTS),
        help=f"Comma-separated B-site elements (default {','.join(DEFAULT_B_ELEMENTS)}).",
    )
    parser.add_argument("--only-singles", action="store_true", help="Skip the doubles.")
    parser.add_argument("--only-doubles", action="store_true", help="Skip the singles.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List the systems and their sizes, then stop (no CHGNet, no solve).",
    )

    perturb = parser.add_argument_group(
        "symmetry breaking (the seed is a stationary point; see the module docstring)"
    )
    perturb.add_argument(
        "--rattle", type=float, default=0.08,
        help="Gaussian per-atom displacement in Angstrom (default 0.08). "
             "0 disables the rattle.",
    )
    perturb.add_argument(
        "--strain", type=float, default=0.02,
        help="Std. dev. of the random symmetric cell strain, as a fraction "
             "(default 0.02 = 2%%). 0 disables it.",
    )
    perturb.add_argument(
        "--n-restarts", type=int, default=1,
        help="Independently perturbed relaxations per system; the lowest-energy "
             "one is kept (default 1). Use 3 or more to check whether the basin "
             "is unique -- the spread lands in the CSV.",
    )
    perturb.add_argument(
        "--seed", type=int, default=0,
        help="Base RNG seed for the perturbations (default 0). Same seed, same "
             "geometries.",
    )

    chgnet = parser.add_argument_group("CHGNet relaxation")
    chgnet.add_argument("--fmax", type=float, default=0.005,
                        help="Force convergence, eV/A (default 0.005; loose values "
                             "stop before symmetry breaks).")
    chgnet.add_argument("--steps", type=int, default=500, help="Max optimizer steps.")
    chgnet.add_argument("--optimizer", default="LBFGS", choices=["LBFGS", "FIRE", "BFGS"])
    chgnet.add_argument("--fix-cell", action="store_true",
                        help="Relax positions only, holding the lattice fixed.")
    chgnet.add_argument("--device", default=None, help="Torch device, e.g. cpu / cuda / mps.")
    chgnet.add_argument("--verbose", action="store_true", help="Per-step optimizer output.")

    solve = parser.add_argument_group("spin solve")
    solve.add_argument("--charge", type=int, default=0)
    solve.add_argument("--max-mixing", type=int, default=2)
    solve.add_argument("--exact-max-sites", type=int, default=16,
                       help="Exact Ising enumeration at or below this many magnetic "
                            "sites, optimizer above (default 16).")
    solve.add_argument("--n-trials", type=int, default=30)
    solve.add_argument("--n-steps", type=int, default=250)

    vasp = parser.add_argument_group("VASP inputs")
    vasp.add_argument("--no-vasp", action="store_true",
                      help="Skip VASP input generation entirely.")
    vasp.add_argument("--vasp-dir", default=None,
                      help="Root for the VASP directories (default <output-dir>/vasp). "
                           "Each perovskite gets a subfolder, and each ordering a "
                           "subfolder under that.")
    vasp.add_argument("--potcar-dir", default=None,
                      help="POTCAR root holding <Element>/POTCAR. Without it no "
                           "POTCAR is written and the runs will not start.")
    vasp.add_argument("--kpoints-density", type=float, default=1000.0,
                      help="Target k-points per reciprocal atom (default 1000).")
    vasp.add_argument("--anisotropic-kpoints", action="store_true",
                      help="Keep the per-axis k-point grid instead of averaging it "
                           "into a cubic one. Cheaper and more even on a 4x2x2 cell.")
    vasp.add_argument("--incar", action="append", metavar="KEY=VALUE", default=[],
                      help="Override or add an INCAR tag; repeatable "
                           "(e.g. --incar ENCUT=600 --incar NCORE=4).")

    args = parser.parse_args(argv)
    if args.only_singles and args.only_doubles:
        parser.error("--only-singles and --only-doubles are mutually exclusive.")
    try:
        args.incar_overrides = parse_incar_overrides(args.incar)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    b_elements = [e.strip() for e in args.b_elements.split(",") if e.strip()]
    jobs = enumerate_systems(
        b_elements,
        want_singles=not args.only_doubles,
        want_doubles=not args.only_singles,
    )

    print(f"B-site grid: {N_CELLS[0]}x{N_CELLS[1]}x{N_CELLS[2]}  seed a = {SEED_A} A")
    print(
        f"Perturbation: rattle {args.rattle} A, strain {args.strain}, "
        f"{max(1, args.n_restarts)} restart(s), RNG seed {args.seed}"
    )
    print(f"Reference orderings: {', '.join(REFERENCE_PATTERNS)}")
    print(
        f"{len(jobs)} system(s): "
        f"{sum(1 for k, _l, _b in jobs if k == 'single')} single, "
        f"{sum(1 for k, _l, _b in jobs if k == 'double')} double"
    )
    if not args.no_vasp and args.potcar_dir is None:
        print(
            "Note: no --potcar-dir given, so no POTCAR is written. The VASP "
            "directories are complete otherwise; add POTCARs before submitting."
        )

    if args.dry_run:
        for kind, label, builder in jobs:
            structure = builder()
            print(f"  {kind:>6}  {structure.name:<28}  {structure.atom_count:>4} atoms")
        return 0

    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    # One model load for the whole screen.
    from quick_mag.chgnet_runner import load_calculator

    calculator = load_calculator(args.device)

    rows = []
    failures = 0
    for kind, label, builder in jobs:
        try:
            rows.append(run_one(kind, label, builder, args, calculator))
        except Exception as exc:  # keep the screen going; record what broke
            failures += 1
            traceback.print_exc()
            print(f"  FAILED: {label} ({exc})")
            rows.append(
                {
                    "name": f"{A_SITE}-{label}",
                    "kind": kind,
                    "b_sites": label,
                    "status": "failed",
                    "note": f"{type(exc).__name__}: {exc}",
                }
            )

    csv_path = out_dir / "summary.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"\n{'=' * 78}")
    print(f"Wrote {len(rows) - failures}/{len(rows)} systems to {out_dir}")
    print(f"Summary: {csv_path}")
    if not args.no_vasp:
        root = Path(args.vasp_dir).expanduser() if args.vasp_dir else out_dir / "vasp"
        print(f"VASP inputs: {root}")
    if failures:
        print(f"{failures} system(s) failed; see the 'status'/'note' columns.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
