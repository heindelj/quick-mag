"""The wire format shared by the remote-calculation client and server.

Everything crossing the network is plain JSON built from lists and floats, so this
module is importable in the numpy/scipy-only Pyodide build. It must never import
ase, chgnet, torch, or :mod:`quick_mag.chgnet_runner`.

The payload is deliberately smaller than a ``ChemicalStructure``. Provenance --
``generation_parameters``, saved spin configurations, the signed moments from the
solver -- never travels, because the client still holds the structure it submitted
and rebuilds the relaxed one against that template (:func:`structure_from_result`)
exactly the way ``chgnet_runner.from_ase_atoms`` does locally. The server therefore
only has to send numbers back.

``protocol`` is checked on both ends so a browser tab running a cached ``index.html``
from before a format change fails with a sentence rather than a ``KeyError``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from quick_mag.structure import ChemicalStructure

# Bump on any incompatible change to the shapes below.
PROTOCOL_VERSION = 1

# The kinds of job a server can be asked to run. "chgnet" relaxes a structure;
# "reconstruct" fits the closest ideal builder structure to a relaxed one (see
# ``quick_mag.reconstruction``), which needs the builder provenance the chgnet
# payload leaves out -- so that kind carries ``generation_parameters`` along.
JOB_KINDS = ("chgnet", "reconstruct")

# Job lifecycle. QUEUED and RUNNING are live; the rest are terminal.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"
LIVE_STATUSES = (STATUS_QUEUED, STATUS_RUNNING)
TERMINAL_STATUSES = (STATUS_DONE, STATUS_ERROR, STATUS_CANCELLED)

# Mirrors chgnet_runner.CALCULATIONS / OPTIMIZERS, duplicated rather than imported:
# this module has to stay importable where chgnet_runner is not.
CALCULATIONS = ("single-point", "atoms", "cell", "cell+atoms")
OPTIMIZERS = ("FIRE", "BFGS", "LBFGS")

DEFAULT_PARAMS: Dict[str, Any] = {
    "calculation": "cell+atoms",
    "optimizer": "LBFGS",
    "fmax": 0.005,
    "steps": 500,
}


class ProtocolError(ValueError):
    """A payload that does not conform to this protocol.

    Raised on both ends: the server turns it into a 400 with the message intact,
    and the client surfaces it in the UI, so the text is written to be read by a
    person rather than logged.
    """


# --------------------------------------------------------------------------
# JSON hygiene
# --------------------------------------------------------------------------

def _finite_or_none(value: float) -> Optional[float]:
    """``float(value)``, or None when it is NaN/inf.

    ``json.dumps`` emits bare ``NaN`` and ``Infinity`` by default, which are not
    JSON and which ``JSON.parse`` rejects outright -- so a single diverged force
    component would break the browser client rather than reporting a failed
    relaxation. Non-finite numbers become null instead, and :func:`dumps` runs
    with ``allow_nan=False`` so anything missed fails loudly on the server rather
    than silently in the page.
    """
    number = float(value)
    return number if math.isfinite(number) else None


def _array_to_json(array: Any) -> List:
    """Nested lists from an array-like, with non-finite entries as null."""
    values = np.asarray(array, dtype=np.float64)
    if not np.isfinite(values).all():
        return [
            _finite_or_none(item) if values.ndim == 1 else _array_to_json(item)
            for item in values
        ]
    return values.tolist()


def dumps(payload: Dict[str, Any]) -> str:
    """Serialize a payload, refusing to emit non-JSON numeric literals."""
    return json.dumps(payload, allow_nan=False)


def loads(text: str) -> Dict[str, Any]:
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise ProtocolError(f"Response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProtocolError("Expected a JSON object at the top level.")
    return payload


# --------------------------------------------------------------------------
# Field readers -- every one of these produces a sentence, not a traceback
# --------------------------------------------------------------------------

def _require(payload: Dict[str, Any], key: str) -> Any:
    if key not in payload:
        raise ProtocolError(f"Missing required field '{key}'.")
    return payload[key]


def _as_float_array(value: Any, *, field: str, shape: Optional[tuple] = None) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"Field '{field}' is not numeric: {exc}") from exc
    if shape is not None and array.shape != shape:
        raise ProtocolError(
            f"Field '{field}' has shape {array.shape}, expected {shape}."
        )
    if not np.isfinite(array).all():
        raise ProtocolError(f"Field '{field}' contains non-finite values.")
    return array


def _optional_float_array(value: Any, *, field: str) -> Optional[np.ndarray]:
    """Arrays the server may legitimately omit (a stress-free cluster, say)."""
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    return array


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------

def structure_to_payload(
    structure: ChemicalStructure, *, include_provenance: bool = False
) -> Dict[str, Any]:
    """The minimum the remote side needs to run a calculation.

    ``include_provenance`` adds the builder ``generation_parameters``, which a
    reconstruction needs and a relaxation does not.
    """
    payload = {
        "name": structure.name,
        "lattice": _array_to_json(structure.lattice),
        "cartesian_coords": _array_to_json(structure.cartesian_coords),
        "atomic_labels": list(structure.atomic_labels),
        "is_periodic": bool(structure.is_periodic),
        "periodic_axes": [bool(flag) for flag in _structure_periodic_axes(structure)],
    }
    params = getattr(structure, "generation_parameters", None)
    if include_provenance and params is not None:
        from quick_mag.structure import generation_parameters_to_json

        payload["generation_parameters"] = generation_parameters_to_json(params)
    return payload


def _structure_periodic_axes(structure: ChemicalStructure):
    axes = getattr(structure, "periodic_axes", None)
    if axes is None:
        flag = bool(structure.is_periodic)
        return (flag, flag, flag)
    return tuple(bool(flag) for flag in axes)


def structure_from_payload(payload: Dict[str, Any]) -> ChemicalStructure:
    """Rebuild a structure on the compute host.

    Moments come back zeroed: the submitted signed spins are the client's business
    and CHGNet neither reads nor returns them.
    """
    if not isinstance(payload, dict):
        raise ProtocolError("Field 'structure' must be a JSON object.")

    labels = _require(payload, "atomic_labels")
    if not isinstance(labels, list) or not all(isinstance(item, str) for item in labels):
        raise ProtocolError("Field 'structure.atomic_labels' must be a list of strings.")

    coords = _as_float_array(
        _require(payload, "cartesian_coords"), field="structure.cartesian_coords"
    )
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ProtocolError(
            "Field 'structure.cartesian_coords' must be an (N, 3) array, "
            f"got shape {coords.shape}."
        )
    if len(labels) != len(coords):
        raise ProtocolError(
            f"'structure' has {len(labels)} labels but {len(coords)} coordinates."
        )

    lattice = _as_float_array(
        _require(payload, "lattice"), field="structure.lattice", shape=(3, 3)
    )
    generation_parameters = None
    if isinstance(payload.get("generation_parameters"), dict):
        from quick_mag.structure import generation_parameters_from_json

        try:
            generation_parameters = generation_parameters_from_json(
                payload["generation_parameters"]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"structure.generation_parameters is malformed: {exc}") from exc
    return ChemicalStructure.with_zero_magnetic_moments(
        name=str(payload.get("name") or "structure"),
        lattice=lattice,
        cartesian_coords=coords,
        atomic_labels=labels,
        is_periodic=bool(payload.get("is_periodic", True)),
        periodic_axes=(
            tuple(bool(flag) for flag in payload["periodic_axes"])
            if isinstance(payload.get("periodic_axes"), (list, tuple))
            and len(payload["periodic_axes"]) == 3
            else None
        ),
        generation_parameters=generation_parameters,
        # Provenance travels only for a reconstruction, whose whole premise is
        # that the geometry no longer matches it.
        geometry_matches_generation=generation_parameters is None,
    )


# --------------------------------------------------------------------------
# Requests
# --------------------------------------------------------------------------

def normalize_reconstruct_params(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Parameters of a reconstruction: an optional list of tilt systems to try."""
    merged = dict(params or {})
    tilt_systems = merged.get("tilt_systems")
    if tilt_systems is not None:
        if not isinstance(tilt_systems, (list, tuple)) or not all(
            isinstance(item, str) for item in tilt_systems
        ):
            raise ProtocolError("'tilt_systems' must be a list of Glazer strings.")
        tilt_systems = [str(item) for item in tilt_systems]
    return {"calculation": "reconstruct", "tilt_systems": tilt_systems}


def normalize_params(
    params: Optional[Dict[str, Any]], *, kind: str = "chgnet"
) -> Dict[str, Any]:
    """Fill in defaults and reject values the runner would reject anyway.

    Checking here rather than in the runner means a typo comes back as a 400 with
    an explanation instead of costing a queue slot and a model load first.
    """
    if kind == "reconstruct":
        return normalize_reconstruct_params(params)
    merged = dict(DEFAULT_PARAMS)
    merged.update(params or {})

    calculation = str(merged["calculation"])
    if calculation not in CALCULATIONS:
        raise ProtocolError(
            f"Unknown calculation '{calculation}'. Choose from {list(CALCULATIONS)}."
        )
    optimizer = str(merged["optimizer"])
    if optimizer not in OPTIMIZERS:
        raise ProtocolError(
            f"Unknown optimizer '{optimizer}'. Choose from {list(OPTIMIZERS)}."
        )
    try:
        fmax = float(merged["fmax"])
        steps = int(merged["steps"])
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"'fmax' and 'steps' must be numbers: {exc}") from exc
    if not math.isfinite(fmax) or fmax <= 0.0:
        raise ProtocolError("'fmax' must be a positive number.")
    if steps <= 0:
        raise ProtocolError("'steps' must be a positive integer.")

    return {
        "calculation": calculation,
        "optimizer": optimizer,
        "fmax": fmax,
        "steps": steps,
    }


def check_structure_against_params(
    structure: ChemicalStructure, params: Dict[str, Any]
) -> None:
    """Reject combinations the runner could never carry out.

    Called on both ends. On the client it is what stops an impossible request
    from costing a round trip, a queue slot and a model load before anything says
    so; on the server it is what stops a client that skipped the check.
    """
    if params.get("calculation") == "reconstruct":
        if getattr(structure, "generation_parameters", None) is None:
            raise ProtocolError(
                f"'{structure.name}' has no builder provenance, so there is nothing "
                "to reconstruct it against. Only structures built by the builder "
                "(and relaxed from it) can be reconstructed."
            )
        return
    if params["calculation"] in ("cell", "cell+atoms") and not structure.is_periodic:
        raise ProtocolError(
            f"Cannot relax the cell of the non-periodic structure "
            f"'{structure.name}': it sits in an arbitrary vacuum box. Use the "
            "'atoms' or 'single-point' calculation instead."
        )


def build_job_request(
    structure: ChemicalStructure,
    *,
    kind: str = "chgnet",
    params: Optional[Dict[str, Any]] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """The body of ``POST /jobs``."""
    if kind not in JOB_KINDS:
        raise ProtocolError(f"Unknown job kind '{kind}'. Choose from {list(JOB_KINDS)}.")
    resolved = normalize_params(params, kind=kind)
    check_structure_against_params(structure, resolved)
    return {
        "protocol": PROTOCOL_VERSION,
        "kind": kind,
        "label": label or structure.name,
        "structure": structure_to_payload(
            structure, include_provenance=(kind == "reconstruct")
        ),
        "params": resolved,
    }


def parse_job_request(
    payload: Dict[str, Any], *, max_atoms: Optional[int] = None
) -> Dict[str, Any]:
    """Validate a submitted job on the server.

    Returns ``{"kind", "label", "structure", "params"}`` with ``structure`` already
    a :class:`ChemicalStructure`.
    """
    check_protocol(payload)

    kind = str(payload.get("kind", "chgnet"))
    if kind not in JOB_KINDS:
        raise ProtocolError(
            f"This server cannot run job kind '{kind}'. It supports {list(JOB_KINDS)}."
        )

    structure = structure_from_payload(_require(payload, "structure"))
    if max_atoms is not None and structure.atom_count > max_atoms:
        raise ProtocolError(
            f"Structure has {structure.atom_count} atoms; this server accepts at "
            f"most {max_atoms}. Raise the limit with --max-atoms."
        )

    params = normalize_params(payload.get("params"), kind=kind)
    check_structure_against_params(structure, params)

    return {
        "kind": kind,
        "label": str(payload.get("label") or structure.name),
        "structure": structure,
        "params": params,
    }


def check_protocol(payload: Dict[str, Any]) -> None:
    """Reject a payload written against a different version of this file.

    The likely source is a browser tab still running a cached ``index.html``, so
    the message says what to do about it.
    """
    version = payload.get("protocol")
    if version is None:
        raise ProtocolError(
            "Payload has no 'protocol' field. It was produced by a different "
            "version of quick-mag; reload the page to pick up the current build."
        )
    if int(version) != PROTOCOL_VERSION:
        raise ProtocolError(
            f"Protocol mismatch: payload speaks version {version}, this build "
            f"speaks {PROTOCOL_VERSION}. Update whichever side is older "
            "(a hard reload usually fixes the browser)."
        )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------

def result_to_payload(result: Any) -> Dict[str, Any]:
    """Serialize a ``chgnet_runner.CHGNetResult``.

    Typed as ``Any`` on purpose: the annotation would be the only reason this
    module needed to import ``chgnet_runner``, which it is not allowed to do.
    """
    final = result.final_structure
    return {
        "calculation": result.calculation,
        "energy": _finite_or_none(result.energy),
        "energy_per_atom": _finite_or_none(result.energy_per_atom),
        "max_force": _finite_or_none(result.max_force),
        "forces": _array_to_json(result.forces),
        "stress": _array_to_json(result.stress),
        # CHGNet's unsigned |m| magnitudes: a diagnostic, never a spin configuration.
        "magnetic_moments": _array_to_json(result.magnetic_moments),
        "trajectory_energies": [
            _finite_or_none(value) for value in result.trajectory_energies
        ],
        "final_lattice": _array_to_json(final.lattice),
        "final_coords": _array_to_json(final.cartesian_coords),
        "steps": int(result.steps),
        "converged": bool(result.converged),
    }


def structure_from_result(
    template: ChemicalStructure,
    result: Dict[str, Any],
    *,
    name: Optional[str] = None,
) -> ChemicalStructure:
    """Rebuild the relaxed structure client-side, following ``template``.

    The mirror of ``chgnet_runner.from_ase_atoms``: atom order is preserved by the
    calculation, so the template's oxidation-suffixed labels, periodicity, and
    ``generation_parameters`` all carry through -- which is what keeps site
    indexing (and therefore the spin machinery) valid on the returned structure.

    A non-periodic template keeps its own lattice; the vacuum box the calculation
    ran in is an artifact of the remote side and the coordinates already came back
    shifted out of it.
    """
    coords = _as_float_array(_require(result, "final_coords"), field="final_coords")
    if coords.shape != (template.atom_count, 3):
        raise ProtocolError(
            f"Result has {coords.shape[0]} coordinates but the submitted structure "
            f"has {template.atom_count} atoms. The server answered about a "
            "different structure."
        )
    if template.is_periodic:
        lattice = _as_float_array(
            _require(result, "final_lattice"), field="final_lattice", shape=(3, 3)
        )
        # A slab's finite axes were padded with vacuum for the calculation and
        # carry nothing worth keeping; the template's rows are the real ones.
        template_lattice = np.asarray(template.lattice, dtype=np.float64)
        for axis, periodic in enumerate(_structure_periodic_axes(template)):
            if not periodic:
                lattice[axis] = template_lattice[axis]
    else:
        lattice = np.asarray(template.lattice, dtype=np.float64)

    return ChemicalStructure.with_zero_magnetic_moments(
        name=name or f"{template.name}_chgnet",
        lattice=lattice,
        cartesian_coords=coords,
        atomic_labels=list(template.atomic_labels),
        is_periodic=template.is_periodic,
        periodic_axes=getattr(template, "periodic_axes", None),
        generation_parameters=template.generation_parameters,
        # Topology yes, geometry no -- see ChemicalStructure for what that means.
        geometry_matches_generation=False,
    )


def moments_from_result(result: Dict[str, Any]) -> Optional[np.ndarray]:
    """CHGNet's per-site |m| magnitudes, or None when the server sent none."""
    values = _optional_float_array(result.get("magnetic_moments"), field="magnetic_moments")
    if values is None or values.size == 0:
        return None
    return np.abs(values.reshape(-1))


def trajectory_from(payload: Dict[str, Any]) -> List[float]:
    """Energy trace from either a progress block or a finished result."""
    raw: Iterable = payload.get("trajectory_energies") or []
    return [float(value) for value in raw if value is not None]


__all__ = [
    "CALCULATIONS",
    "DEFAULT_PARAMS",
    "JOB_KINDS",
    "LIVE_STATUSES",
    "OPTIMIZERS",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "STATUS_CANCELLED",
    "STATUS_DONE",
    "STATUS_ERROR",
    "STATUS_QUEUED",
    "STATUS_RUNNING",
    "TERMINAL_STATUSES",
    "build_job_request",
    "check_protocol",
    "check_structure_against_params",
    "dumps",
    "loads",
    "moments_from_result",
    "normalize_params",
    "parse_job_request",
    "result_to_payload",
    "structure_from_payload",
    "structure_from_result",
    "structure_to_payload",
    "trajectory_from",
]
