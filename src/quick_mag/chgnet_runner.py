"""CHGNet single-point and relaxation calculations on ``ChemicalStructure``.

Thin ASE layer over CHGNet's own ``CHGNetCalculator``. Nothing here imports
pymatgen: structures cross into ASE via :func:`to_ase_atoms` and come back via
:func:`from_ase_atoms`, and CHGNet's calculator takes an ``ase.Atoms`` directly.

``ase`` and ``chgnet`` are optional dependencies (``pip install -e '.[chgnet]'``),
so this module must only ever be imported lazily — the rest of the package stays
numpy/scipy-only for the Pyodide build.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.filters import FrechetCellFilter
from ase.optimize.bfgs import BFGS
from ase.optimize.fire import FIRE
from ase.optimize.lbfgs import LBFGS

from quick_mag.structure import ChemicalStructure

class CalculationCancelled(RuntimeError):
    """Raised out of the optimizer's observer when a caller asks it to stop.

    ASE exposes no abort hook, so an observer attached at ``interval=1`` is the
    only place a relaxation can be interrupted between steps with the calculator
    in a consistent state. The server turns this into a cancelled job; a local
    caller that passes no ``should_stop`` never sees it.

    ``partial`` carries the state of the relaxation at the moment it stopped --
    a :class:`CHGNetResult` with ``converged=False`` -- when the stop landed
    between steps, so a relaxation can be cut short on purpose and still hand
    back the geometry it had reached. It is None when nothing was computed.
    """

    def __init__(self, message: str, partial: "CHGNetResult | None" = None) -> None:
        super().__init__(message)
        self.partial = partial


# The calculation modes exposed on the command line.
CALCULATIONS = ("single-point", "atoms", "cell", "cell+atoms")
OPTIMIZERS = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}

# A non-periodic structure still needs a lattice for CHGNet's graph converter, so
# it is dropped into a large vacuum box before the calculation.
NON_PERIODIC_BOX_LENGTH = 100.0
NON_PERIODIC_BOX_PADDING = 10.0


@dataclass
class CHGNetResult:
    """Everything one CHGNet calculation produced.

    ``magnetic_moments`` holds CHGNet's predicted *unsigned* magnitudes (|m|, in
    Bohr magnetons). They are diagnostics only and are deliberately never written
    into ``ChemicalStructure.magnetic_moments``, which carries signed spins from
    the solver.
    """

    calculation: str
    initial_structure: ChemicalStructure
    final_structure: ChemicalStructure
    energy: float
    forces: np.ndarray
    stress: np.ndarray
    magnetic_moments: np.ndarray
    trajectory_energies: List[float]
    steps: int
    converged: bool
    #: True when the caller stopped the relaxation before the optimizer did.
    cancelled: bool = False

    @property
    def energy_per_atom(self) -> float:
        count = self.final_structure.atom_count
        return float(self.energy) / count if count else 0.0

    @property
    def max_force(self) -> float:
        forces = np.asarray(self.forces, dtype=np.float64)
        if forces.size == 0:
            return 0.0
        return float(np.linalg.norm(forces, axis=1).max())


def _boxed_lattice(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vacuum box (lattice, shifted coords) enclosing a non-periodic structure."""
    coords = np.asarray(coords, dtype=np.float64)
    if coords.size == 0:
        return np.diag(np.full(3, NON_PERIODIC_BOX_LENGTH)), coords.copy()
    mins = coords.min(axis=0)
    spans = coords.max(axis=0) - mins
    lengths = np.maximum(
        np.full(3, NON_PERIODIC_BOX_LENGTH, dtype=np.float64),
        spans + 2.0 * NON_PERIODIC_BOX_PADDING,
    )
    return np.diag(lengths), coords - mins + NON_PERIODIC_BOX_PADDING


def _periodic_axes_of(structure: ChemicalStructure) -> tuple[bool, bool, bool]:
    axes = getattr(structure, "periodic_axes", None)
    if axes is None:
        flag = bool(structure.is_periodic)
        return flag, flag, flag
    return tuple(bool(flag) for flag in axes)  # type: ignore[return-value]


def _slab_lattice(lattice: np.ndarray, axes) -> np.ndarray:
    """``lattice`` with each finite axis lengthened by the vacuum padding."""
    rows = np.asarray(lattice, dtype=np.float64).copy()
    for axis, periodic in enumerate(axes):
        if periodic:
            continue
        length = float(np.linalg.norm(rows[axis]))
        if length > 0.0:
            rows[axis] *= (length + 2.0 * NON_PERIODIC_BOX_PADDING) / length
    return rows


def to_ase_atoms(structure: ChemicalStructure) -> Atoms:
    """Convert a ``ChemicalStructure`` to ``ase.Atoms``.

    Element symbols come from ``element_symbols()`` so oxidation-state suffixes
    (``"Fe3+"``) never reach ASE. Non-periodic structures are boxed into a large
    cell — CHGNet builds its graph from a lattice — but keep ``pbc=True`` so the
    graph converter and the stress tensor stay well defined.
    """
    return _to_ase_atoms_with_offset(structure)[0]


def _to_ase_atoms_with_offset(
    structure: ChemicalStructure,
) -> tuple[Atoms, np.ndarray]:
    """``to_ase_atoms`` plus the translation applied when boxing a cluster.

    Subtracting the offset from the ASE positions puts a relaxed non-periodic
    structure back where the input structure lived.
    """
    coords = np.asarray(structure.cartesian_coords, dtype=np.float64)
    axes = _periodic_axes_of(structure)
    if all(axes):
        lattice = np.asarray(structure.lattice, dtype=np.float64)
        offset = np.zeros(3, dtype=np.float64)
    elif any(axes):
        # A slab: periodic in some directions, finite in the rest. The finite
        # axes get vacuum padding so the periodic images CHGNet builds along
        # them do not touch, and the coordinates stay where they are.
        lattice = _slab_lattice(structure.lattice, axes)
        offset = np.zeros(3, dtype=np.float64)
    else:
        lattice, boxed = _boxed_lattice(coords)
        offset = (boxed - coords).mean(axis=0) if coords.size else np.zeros(3)
        coords = boxed
    atoms = Atoms(
        symbols=structure.element_symbols(),
        positions=coords,
        cell=lattice,
        pbc=True,
    )
    return atoms, offset


def from_ase_atoms(
    atoms: Atoms,
    *,
    template: ChemicalStructure,
    offset: Optional[np.ndarray] = None,
) -> ChemicalStructure:
    """Rebuild a ``ChemicalStructure`` from ``atoms``, following ``template``.

    ASE preserves atom order, so the template's original ``atomic_labels`` (with
    any oxidation suffixes), periodicity, and builder ``generation_parameters``
    are carried through the calculation. The *topological* half of the provenance
    stays valid because ``site_indexing_from_generation_parameters`` depends only
    on atom order and the B-site grid shape, neither of which a relaxation
    changes. The geometric half does not, so ``geometry_matches_generation`` is
    cleared: rebuilding from these parameters would throw the relaxation away.

    A non-periodic template keeps its own lattice: the vacuum box the calculation
    ran in is an artifact, so the coordinates are shifted back by ``offset``.
    """
    positions = np.asarray(atoms.get_positions(), dtype=np.float64)
    axes = _periodic_axes_of(template)
    if all(axes):
        lattice = np.asarray(atoms.get_cell(), dtype=np.float64)
    elif any(axes):
        # The padded finite axes carry no information; keep the template's.
        lattice = np.asarray(atoms.get_cell(), dtype=np.float64)
        template_lattice = np.asarray(template.lattice, dtype=np.float64)
        for axis, periodic in enumerate(axes):
            if not periodic:
                lattice[axis] = template_lattice[axis]
    else:
        lattice = np.asarray(template.lattice, dtype=np.float64)
        if offset is not None:
            positions = positions - np.asarray(offset, dtype=np.float64)
    return ChemicalStructure.with_zero_magnetic_moments(
        name=template.name,
        lattice=lattice,
        cartesian_coords=positions,
        atomic_labels=list(template.atomic_labels),
        is_periodic=template.is_periodic,
        periodic_axes=getattr(template, "periodic_axes", None),
        generation_parameters=template.generation_parameters,
        # The parameters still describe this structure's topology, which is what
        # site indexing reads, but they no longer rebuild its geometry: that is
        # the whole point of having relaxed it.
        geometry_matches_generation=False,
    )


@lru_cache(maxsize=4)
def load_calculator(device: Optional[str] = None):
    """Load and cache CHGNet's ASE calculator (one model load per device)."""
    from chgnet.model.dynamics import CHGNetCalculator

    return CHGNetCalculator(use_device=device)


@contextlib.contextmanager
def _quiet_numerical_warnings():
    """Silence the per-step numerical chatter of a CHGNet cell relaxation.

    ``FrechetCellFilter`` takes a matrix logarithm of the cell every step, and
    SciPy reports its (~1e-13) estimated error as a warning each time; CHGNet
    likewise warns when it reads a scalar off a grad-tracking tensor. Both are
    noise that would otherwise bury the results, so they are matched by message
    and dropped rather than being blanket-suppressed.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="logm result may be inaccurate")
        warnings.filterwarnings(
            "ignore", message=".*requires_grad=True to a scalar.*"
        )
        yield


def _optimization_target(atoms: Atoms, calculation: str):
    """Wrap ``atoms`` in whatever ASE object relaxes the requested variables."""
    if calculation == "atoms":
        return atoms
    if calculation == "cell":
        # Freeze every position, then let the cell filter move the lattice.
        atoms.set_constraint(FixAtoms(indices=list(range(len(atoms)))))
        return FrechetCellFilter(atoms)
    if calculation == "cell+atoms":
        return FrechetCellFilter(atoms)
    raise ValueError(f"Unsupported optimization '{calculation}'.")


def run_chgnet_calculation(
    structure: ChemicalStructure,
    calculation: str = "cell+atoms",
    *,
    optimizer: str = "LBFGS",
    fmax: float = 0.005,
    steps: int = 500,
    verbose: bool = False,
    calculator=None,
    device: Optional[str] = None,
    progress: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> CHGNetResult:
    """Run one CHGNet calculation and return the relaxed structure and properties.

    ``progress`` is called once per optimizer step with the running step count,
    energy, max force, and the energy trace so far -- enough for a caller to show
    a live convergence plot without waiting for the result. ``should_stop`` is
    polled at the same points; returning True raises :class:`CalculationCancelled`
    out of the optimizer. Both default to None, which restores the original
    behaviour exactly.
    """
    if calculation not in CALCULATIONS:
        raise ValueError(
            f"Unknown calculation '{calculation}'. Choose from {list(CALCULATIONS)}."
        )
    if optimizer not in OPTIMIZERS:
        raise ValueError(
            f"Unknown optimizer '{optimizer}'. Choose from {list(OPTIMIZERS)}."
        )
    if calculation in ("cell", "cell+atoms") and not structure.is_periodic:
        raise ValueError(
            f"Cannot relax the cell of the non-periodic structure "
            f"'{structure.name}'; it is placed in an arbitrary vacuum box. "
            "Use --fix-cell (atoms only) or --sp."
        )

    atoms, offset = _to_ase_atoms_with_offset(structure)
    atoms.calc = calculator if calculator is not None else load_calculator(device)

    trajectory_energies: List[float] = []
    converged = True
    cancelled = False
    started_at = time.perf_counter()
    last_step_at = started_at

    def _emit_progress() -> None:
        """Report the step just recorded. Free: the calculator computes energy,
        forces, stress and magmoms in one pass, so reading forces back here does
        not trigger a second evaluation.

        The timing fields are the diagnostic for a relaxation that has slowed
        down: ``step_seconds`` is the wall time of the step just finished and
        ``wall_seconds`` the time since the calculation began, both measured on
        the compute host, so a change in pace shows up on the server side or on
        the transport side but not ambiguously in between."""
        nonlocal last_step_at
        now = time.perf_counter()
        step_seconds = now - last_step_at
        last_step_at = now
        if progress is None:
            return
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
        progress(
            {
                "step": max(len(trajectory_energies) - 1, 0),
                "energy": trajectory_energies[-1] if trajectory_energies else None,
                "max_force": (
                    float(np.linalg.norm(forces, axis=1).max()) if forces.size else 0.0
                ),
                "trajectory_energies": list(trajectory_energies),
                "step_seconds": step_seconds,
                "wall_seconds": now - started_at,
            }
        )

    def _build_result() -> CHGNetResult:
        energy = float(atoms.get_potential_energy())
        if not trajectory_energies or not np.isclose(trajectory_energies[-1], energy):
            trajectory_energies.append(energy)
        return CHGNetResult(
            calculation=calculation,
            initial_structure=structure,
            final_structure=from_ase_atoms(atoms, template=structure, offset=offset),
            energy=energy,
            forces=np.asarray(atoms.get_forces(), dtype=np.float64),
            stress=np.asarray(atoms.get_stress(), dtype=np.float64),
            magnetic_moments=np.asarray(atoms.get_magnetic_moments(), dtype=np.float64),
            trajectory_energies=trajectory_energies,
            steps=max(len(trajectory_energies) - 1, 0),
            converged=converged,
            cancelled=cancelled,
        )

    with _quiet_numerical_warnings():
        if calculation == "single-point":
            trajectory_energies.append(float(atoms.get_potential_energy()))
            _emit_progress()
        else:
            target = _optimization_target(atoms, calculation)

            def _record_energy() -> None:
                if should_stop is not None and should_stop():
                    raise CalculationCancelled(
                        f"Cancelled after {max(len(trajectory_energies) - 1, 0)} steps."
                    )
                trajectory_energies.append(float(atoms.get_potential_energy()))
                _emit_progress()

            stream = sys.stdout if verbose else io.StringIO()
            with contextlib.redirect_stdout(stream):
                dynamics = OPTIMIZERS[optimizer](target)
                try:
                    _record_energy()
                    dynamics.attach(_record_energy, interval=1)
                    converged = bool(dynamics.run(fmax=fmax, steps=steps))
                except CalculationCancelled as exc:
                    # Stopped between steps: the calculator holds a consistent
                    # energy and forces for the current geometry, so the
                    # partially relaxed structure goes back with the
                    # cancellation rather than being thrown away.
                    converged = False
                    cancelled = True
                    partial = None
                    if trajectory_energies:
                        partial = _build_result()
                    raise CalculationCancelled(str(exc), partial) from None

        return _build_result()


__all__ = [
    "CALCULATIONS",
    "CHGNetResult",
    "CalculationCancelled",
    "OPTIMIZERS",
    "from_ase_atoms",
    "load_calculator",
    "run_chgnet_calculation",
    "to_ase_atoms",
]
