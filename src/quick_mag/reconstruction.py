"""Fit the closest idealized builder structure to a relaxed one.

A relaxation keeps a structure's *topology* -- the atom order, the site roles
and the octahedral grid its ``generation_parameters`` describe -- and throws
away its *geometry*. The builder can no longer regenerate it, but the question
"what ideal perovskite is this closest to?" is well posed exactly because the
topology survived: every relaxed atom is still atom ``n`` of a known canonical
build, so the parameters can be fitted by regenerating candidate ideal
structures and comparing them atom by atom.

What is fitted:

* the lattice constants, read straight off the relaxed cell -- the builder only
  emits diagonal cells, so the cell lengths per octahedron *are* the answer;
* the Glazer tilt system, by trying every one of the 23 in every axis
  orientation (71 distinct spellings; ``a+b-b-`` turned to put its in-phase
  axis along c is ``a-a-c+``);
* the tilt angles of each system, by minimizing the RMSD over its independent
  angles.

What is not: shear (a non-orthogonal relaxed cell is compared by its lengths and
the residual shows up in the RMSD), per-domain spacing along the stacking axis
(all domains share the relaxed cell's overall scale), and defects, which are
carried over unchanged.

The fit is exposed both as a one-shot function and as a :class:`ReconstructionJob`
that evaluates one tilt system per :meth:`~ReconstructionJob.step`, so the UI --
which has no threads in the browser -- can spread the work over frames.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from quick_mag.generation import generated_structure_from_parameters
from quick_mag.perovskite_builder import (
    active_glazer_parameter_axes,
    canonicalize_glazer_tilt_angles_deg,
    glazer_tilt_orientations,
)
from quick_mag.structure import (
    GLAZER_TILT_SYSTEMS,
    ChemicalStructure,
    PerovskiteGenerationParameters,
)

# Tilt angles beyond this are not perovskites any more; the fit stays inside it.
MAX_TILT_DEG = 30.0
# Starting angles for every free parameter. Zero is a stationary point of the
# RMSD for several systems (the tilt enters the oxygen positions through a
# sine), so a second start away from it is needed to find a tilted minimum.
ANGLE_STARTS_DEG: Tuple[float, ...] = (0.0, 8.0)


class ReconstructionCancelled(RuntimeError):
    """Stopped between tilt systems because the caller asked."""


@dataclass(frozen=True)
class TiltCandidate:
    tilt_system: str
    tilt_angles_deg: Tuple[float, float, float]
    rmsd: float


@dataclass
class IdealReconstruction:
    """The ideal builder structure closest to a relaxed one, and how close it is."""

    #: Name of the relaxed structure the fit was made against.
    source_name: str
    #: Builder parameters of the fit; regenerating them gives ``ideal``.
    params: PerovskiteGenerationParameters
    #: The fitted ideal structure, in the relaxed structure's atom order.
    ideal: ChemicalStructure
    #: Per-atom distance from the relaxed position to the ideal one, in Angstrom.
    distances: np.ndarray
    #: Every tilt system tried, best first.
    candidates: List[TiltCandidate] = field(default_factory=list)

    @property
    def atom_count(self) -> int:
        return int(len(self.distances))

    @property
    def rmsd(self) -> float:
        if not len(self.distances):
            return 0.0
        return float(np.sqrt(np.mean(self.distances**2)))

    @property
    def max_distance(self) -> float:
        return float(self.distances.max()) if len(self.distances) else 0.0

    @property
    def max_distance_index(self) -> int:
        return int(self.distances.argmax()) if len(self.distances) else -1

    @property
    def tilt_system(self) -> str:
        return str(self.params.tilt_system)

    @property
    def tilt_angles_deg(self) -> Tuple[float, float, float]:
        return (
            float(self.params.tilt_angle_x_deg),
            float(self.params.tilt_angle_y_deg),
            float(self.params.tilt_angle_z_deg),
        )

    @property
    def lattice_constants(self) -> Tuple[float, float, float]:
        """Single-octahedron edge lengths of domain 0."""
        return (
            2.0 * float(self.params.center_to_vertex_distance_x),
            2.0 * float(self.params.center_to_vertex_distance_y),
            2.0 * float(self.params.center_to_vertex_distance_z),
        )

    def headline(self) -> str:
        a, b, c = self.lattice_constants
        angles = ", ".join(f"{value:.1f}" for value in self.tilt_angles_deg)
        return (
            f"{self.tilt_system} ({angles} deg), a b c = {a:.3f} {b:.3f} {c:.3f} A, "
            f"RMSD {self.rmsd:.4f} A"
        )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def _wrapped_deltas(
    reference: ChemicalStructure, ideal: ChemicalStructure
) -> np.ndarray:
    """``reference - ideal`` per atom, minimum-imaged along the periodic axes.

    Wrapped under the *reference* lattice, since that is the cell the relaxed
    atoms actually live in. The mean is then removed: a rigid translation of the
    whole structure is not a deviation from the ideal.
    """
    deltas = np.asarray(reference.cartesian_coords, dtype=np.float64) - np.asarray(
        ideal.cartesian_coords, dtype=np.float64
    )
    lattice = np.asarray(reference.lattice, dtype=np.float64)
    axes = reference.periodic_axes
    if axes is None:
        flag = bool(reference.is_periodic)
        axes = (flag, flag, flag)
    if any(axes) and abs(float(np.linalg.det(lattice))) > 1e-9:
        fractional = np.linalg.solve(lattice.T, deltas.T).T
        for axis, periodic in enumerate(axes):
            if periodic:
                fractional[:, axis] -= np.round(fractional[:, axis])
        deltas = fractional @ lattice
    if len(deltas):
        deltas = deltas - deltas.mean(axis=0)
    return deltas


def _distances(reference: ChemicalStructure, ideal: ChemicalStructure) -> np.ndarray:
    return np.linalg.norm(_wrapped_deltas(reference, ideal), axis=1)


def _rmsd(reference: ChemicalStructure, ideal: ChemicalStructure) -> float:
    distances = _distances(reference, ideal)
    return float(np.sqrt(np.mean(distances**2))) if len(distances) else 0.0


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------


def base_parameters(structure: ChemicalStructure) -> PerovskiteGenerationParameters:
    """The relaxed structure's parameters with the cell fitted and tilts cleared.

    The relaxed cell's lengths, divided by the octahedron count per axis, give
    the edge length per octahedron directly. A stack keeps its domains' relative
    spacing and scales them all by the same factor.
    """
    params = structure.generation_parameters
    if params is None:
        raise ValueError("only a structure with builder provenance can be reconstructed.")
    lengths = np.linalg.norm(np.asarray(structure.lattice, dtype=np.float64), axis=1)
    ideal_lengths = np.linalg.norm(params.supercell_lattice(), axis=1)
    scale = np.where(ideal_lengths > 0.0, lengths / np.where(ideal_lengths > 0.0, ideal_lengths, 1.0), 1.0)

    domains = [
        replace(
            domain,
            lattice=tuple(float(edge * factor) for edge, factor in zip(domain.lattice, scale)),
        )
        for domain in params.domains
    ]
    return replace(
        params,
        center_to_vertex_distance_x=float(params.center_to_vertex_distance_x * scale[0]),
        center_to_vertex_distance_y=float(params.center_to_vertex_distance_y * scale[1]),
        center_to_vertex_distance_z=float(params.center_to_vertex_distance_z * scale[2]),
        cell_origin=np.asarray(params.cell_origin, dtype=np.float64)
        * np.asarray(scale, dtype=np.float64),
        center=np.asarray(params.center, dtype=np.float64) * np.asarray(scale, dtype=np.float64),
        domains=domains,
        tilt_system="a0a0a0",
        tilt_angle_x_deg=0.0,
        tilt_angle_y_deg=0.0,
        tilt_angle_z_deg=0.0,
    )


def _with_tilt(
    params: PerovskiteGenerationParameters,
    tilt_system: str,
    free_angles: Sequence[float],
) -> PerovskiteGenerationParameters:
    """``params`` with ``tilt_system`` and its free angles set, canonicalized."""
    selectors = active_glazer_parameter_axes(tilt_system)
    angles = [0.0, 0.0, 0.0]
    cursor = 0
    for axis, free in enumerate(selectors):
        if free:
            angles[axis] = float(free_angles[cursor])
            cursor += 1
    x, y, z = canonicalize_glazer_tilt_angles_deg(tilt_system, *angles)
    return replace(
        params,
        tilt_system=tilt_system,
        tilt_angle_x_deg=x,
        tilt_angle_y_deg=y,
        tilt_angle_z_deg=z,
    )


def _regenerate(params: PerovskiteGenerationParameters, name: str) -> ChemicalStructure:
    return generated_structure_from_parameters(params, name=name, periodic=params.periodic_axes)


def _fit_tilt_system(
    structure: ChemicalStructure,
    base: PerovskiteGenerationParameters,
    tilt_system: str,
) -> TiltCandidate:
    """Best angles of one tilt system, by RMSD."""
    from scipy.optimize import minimize

    free_count = sum(active_glazer_parameter_axes(tilt_system))

    def objective(angles: np.ndarray) -> float:
        if np.any(np.abs(angles) > MAX_TILT_DEG):
            return 1e3 + float(np.abs(angles).max())
        candidate = _regenerate(_with_tilt(base, tilt_system, angles), "fit")
        if candidate.atom_count != structure.atom_count:
            return 1e6
        return _rmsd(structure, candidate)

    if free_count == 0:
        params = _with_tilt(base, tilt_system, [])
        return TiltCandidate(
            tilt_system,
            (params.tilt_angle_x_deg, params.tilt_angle_y_deg, params.tilt_angle_z_deg),
            objective(np.zeros(0)),
        )

    best_angles = np.zeros(free_count)
    best_value = float("inf")
    for start in ANGLE_STARTS_DEG:
        result = minimize(
            objective,
            np.full(free_count, start, dtype=np.float64),
            method="Nelder-Mead",
            options={"xatol": 0.05, "fatol": 1e-5, "maxfev": 60 * free_count},
        )
        if float(result.fun) < best_value:
            best_value = float(result.fun)
            best_angles = np.asarray(result.x, dtype=np.float64)
    params = _with_tilt(base, tilt_system, best_angles)
    return TiltCandidate(
        tilt_system,
        (params.tilt_angle_x_deg, params.tilt_angle_y_deg, params.tilt_angle_z_deg),
        best_value,
    )


# A candidate within this relative margin of the best RMSD is "as good": among
# those, the one with the fewest free angles wins. Otherwise noise always hands
# the fit to the most general system, since an extra angle never fits worse.
PARSIMONY_MARGIN = 0.02


def free_angle_count(tilt_system: str) -> int:
    return int(sum(active_glazer_parameter_axes(tilt_system)))


def rank_candidates(candidates: Sequence[TiltCandidate]) -> List[TiltCandidate]:
    """Best first: lowest RMSD, ties within :data:`PARSIMONY_MARGIN` broken by simplicity."""
    if not candidates:
        return []
    by_rmsd = sorted(candidates, key=lambda candidate: candidate.rmsd)
    threshold = by_rmsd[0].rmsd * (1.0 + PARSIMONY_MARGIN) + 1e-6
    close = [candidate for candidate in by_rmsd if candidate.rmsd <= threshold]
    rest = [candidate for candidate in by_rmsd if candidate.rmsd > threshold]
    close.sort(key=lambda candidate: (free_angle_count(candidate.tilt_system), candidate.rmsd))
    return close + rest


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


class ReconstructionJob:
    """An incremental fit: one tilt system per :meth:`step`.

    Tilt systems are tried in the order of :data:`GLAZER_TILT_SYSTEMS`, each in
    all of its axis orientations. A grid with a single octahedron along any axis
    cannot alternate tilts, so only the untilted system is tried for it.
    """

    def __init__(
        self,
        structure: ChemicalStructure,
        *,
        tilt_systems: Optional[Sequence[str]] = None,
    ) -> None:
        self.structure = structure
        self.base = base_parameters(structure)
        self.tilt_systems = list(
            tilt_systems
            if tilt_systems is not None
            else glazer_tilt_orientations(GLAZER_TILT_SYSTEMS)
        )
        grid = self.base.grid_shape()
        if min(grid) < 2:
            self.tilt_systems = ["a0a0a0"]
        self.candidates: List[TiltCandidate] = []
        self.error: str = ""
        self.result: Optional[IdealReconstruction] = None

    @property
    def total(self) -> int:
        return len(self.tilt_systems)

    @property
    def completed(self) -> int:
        return len(self.candidates)

    @property
    def done(self) -> bool:
        return self.result is not None or bool(self.error)

    def status_line(self) -> str:
        if self.error:
            return f"Reconstruction failed: {self.error}"
        if self.result is not None:
            return f"Reconstructed: {self.result.headline()}"
        return f"Reconstructing {self.structure.name}: {self.completed}/{self.total} tilt systems"

    def step(self) -> bool:
        """Fit the next tilt system; True once the job has finished."""
        if self.done:
            return True
        try:
            if self.completed < self.total:
                tilt_system = self.tilt_systems[self.completed]
                self.candidates.append(_fit_tilt_system(self.structure, self.base, tilt_system))
            if self.completed >= self.total:
                self._finish()
        except (ValueError, RuntimeError) as exc:
            self.error = str(exc)
        return self.done

    def run(
        self,
        *,
        progress: Optional[Callable[[Dict[str, Any]], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> IdealReconstruction:
        """Run to completion, reporting after each tilt system.

        ``progress`` receives ``{"step", "total", "phase"}``; ``should_stop`` is
        checked between tilt systems and raises :class:`ReconstructionCancelled`.
        """
        while not self.step():
            if progress is not None:
                progress({"step": self.completed, "total": self.total, "phase": "reconstruct"})
            if should_stop is not None and should_stop():
                raise ReconstructionCancelled("Reconstruction cancelled.")
        if self.result is None:
            raise ValueError(self.error or "reconstruction produced no result.")
        return self.result

    def progress_line(self) -> str:
        return f"{self.completed}/{self.total} tilt systems"

    def _finish(self) -> None:
        ranked = rank_candidates(self.candidates)
        best = ranked[0]
        params = replace(
            self.base,
            tilt_system=best.tilt_system,
            tilt_angle_x_deg=best.tilt_angles_deg[0],
            tilt_angle_y_deg=best.tilt_angles_deg[1],
            tilt_angle_z_deg=best.tilt_angles_deg[2],
        )
        ideal = _regenerate(params, f"{self.structure.name}_ideal")
        # Same atom order, so the ideal can carry the relaxed structure's moments
        # and be dropped into every per-site feature as a stand-in.
        ideal.magnetic_moments = np.asarray(
            self.structure.magnetic_moments, dtype=np.float64
        ).copy()
        # Deliberately not translated onto the relaxed atoms: the ideal stays
        # exactly what its parameters regenerate, so the builder can drive it.
        self.result = IdealReconstruction(
            source_name=self.structure.name,
            params=ideal.generation_parameters,
            ideal=ideal,
            distances=_distances(self.structure, ideal),
            candidates=ranked,
        )


def reconstruct_ideal(
    structure: ChemicalStructure,
    *,
    tilt_systems: Optional[Sequence[str]] = None,
) -> IdealReconstruction:
    """Fit the closest ideal structure to ``structure`` in one call."""
    return ReconstructionJob(structure, tilt_systems=tilt_systems).run()


# ---------------------------------------------------------------------------
# Wire format (see ``quick_mag.remote.protocol``)
# ---------------------------------------------------------------------------


def reconstruction_to_payload(reconstruction: IdealReconstruction) -> Dict[str, Any]:
    """JSON-safe result: the fitted parameters and the per-atom error.

    The ideal structure itself does not travel; the client regenerates it from
    the parameters, which is one build.
    """
    from quick_mag.structure import generation_parameters_to_json

    return {
        "kind": "reconstruct",
        "source_name": reconstruction.source_name,
        "generation_parameters": generation_parameters_to_json(reconstruction.params),
        "distances": [float(value) for value in reconstruction.distances],
        "rmsd": float(reconstruction.rmsd),
        "candidates": [
            {
                "tilt_system": candidate.tilt_system,
                "tilt_angles_deg": [float(v) for v in candidate.tilt_angles_deg],
                "rmsd": float(candidate.rmsd),
            }
            for candidate in reconstruction.candidates
        ],
    }


def reconstruction_from_payload(
    structure: ChemicalStructure, payload: Dict[str, Any]
) -> IdealReconstruction:
    """Rebuild a fit for ``structure`` from :func:`reconstruction_to_payload` output.

    Regenerates the ideal from the fitted parameters and recomputes the per-atom
    distances against ``structure``, so the numbers shown are exactly those of
    the structure in hand rather than whatever the server compared against.
    """
    from quick_mag.structure import generation_parameters_from_json

    params = generation_parameters_from_json(dict(payload["generation_parameters"]))
    ideal = _regenerate(params, f"{structure.name}_ideal")
    if ideal.atom_count != structure.atom_count:
        raise ValueError(
            f"The fitted parameters build {ideal.atom_count} atoms but the structure "
            f"has {structure.atom_count}."
        )
    ideal.magnetic_moments = np.asarray(structure.magnetic_moments, dtype=np.float64).copy()
    candidates = [
        TiltCandidate(
            str(item["tilt_system"]),
            tuple(float(v) for v in item["tilt_angles_deg"]),  # type: ignore[arg-type]
            float(item["rmsd"]),
        )
        for item in payload.get("candidates", [])
    ]
    return IdealReconstruction(
        source_name=str(payload.get("source_name") or structure.name),
        params=ideal.generation_parameters,
        ideal=ideal,
        distances=_distances(structure, ideal),
        candidates=candidates,
    )


__all__ = [
    "IdealReconstruction",
    "ReconstructionCancelled",
    "reconstruction_from_payload",
    "reconstruction_to_payload",
    "rank_candidates",
    "ReconstructionJob",
    "TiltCandidate",
    "base_parameters",
    "reconstruct_ideal",
]
