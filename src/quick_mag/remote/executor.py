"""Where a submitted job actually runs.

The HTTP server owns queueing, status and cancellation; an executor owns nothing
but the calculation. That seam exists so the same server can drive a job that runs
in-process on a workstation and one that goes through a batch scheduler on a
cluster: :class:`InlineExecutor` is the former, and a scheduler-backed executor
implements the same three methods without the server, the client, the protocol, or
the UI changing at all.

``run`` is deliberately synchronous. The server always calls it from a worker
thread, so an executor that has to wait on something (a queue, a job id, a file
appearing on a shared filesystem) simply blocks and reports progress as it goes.

This module only ever runs on the compute host and is excluded from the browser
manifest, so unlike the rest of ``quick_mag.remote`` it may import CHGNet -- though
it still does so lazily, so ``--help`` works on a machine without it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from quick_mag.remote import protocol
from quick_mag.structure import ChemicalStructure

ProgressCallback = Callable[[Dict[str, Any]], None]
StopCheck = Callable[[], bool]

MISSING_DEPENDENCY_MESSAGE = (
    "This server was started without the CHGNet dependencies.\n"
    "Install them on the compute host with:  pip install -e '.[chgnet]'"
)


class JobCancelled(RuntimeError):
    """The job stopped because someone asked it to, not because it failed.

    ``result`` is the payload of whatever the job had reached when it stopped
    (a partially relaxed structure, say), or None when there is nothing worth
    handing back. The server stores it on the cancelled job so a relaxation can
    be cut short deliberately and still deliver its geometry.
    """

    def __init__(self, message: str, result: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.result = result


class JobExecutor:
    """The contract a server needs from whatever runs the science."""

    #: Short identifier reported by ``GET /health``.
    name = "abstract"

    def describe(self) -> Dict[str, Any]:
        """Executor facts for ``/health``: device, model state, backend."""
        raise NotImplementedError

    def prepare(self) -> None:
        """Do any expensive one-time setup before the first job arrives.

        Called once at startup and allowed to fail loudly: a server that cannot
        load its model should say so while someone is still watching the terminal,
        not on the first submission.
        """

    def run(
        self,
        kind: str,
        structure: ChemicalStructure,
        params: Dict[str, Any],
        *,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCheck] = None,
    ) -> Dict[str, Any]:
        """Run one job and return its result payload (see ``protocol``)."""
        raise NotImplementedError


class InlineExecutor(JobExecutor):
    """Runs the calculation in this process, on this machine.

    Right for a workstation or a node you have to yourself. One job at a time --
    the server's single worker thread guarantees that -- because CHGNet wants the
    device serially anyway.
    """

    name = "inline"

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device
        self._calculator = None

    # -- introspection -----------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "executor": self.name,
            "device": self.device or "auto",
            "model_loaded": self._calculator is not None,
        }
        # Reported rather than required: a CPU-only host is a perfectly good
        # server, it is just worth knowing which one you are talking to before
        # submitting a 500-atom cell.
        try:
            import torch  # type: ignore

            info["torch"] = str(torch.__version__)
            info["cuda_available"] = bool(torch.cuda.is_available())
            if torch.cuda.is_available():
                info["cuda_device"] = torch.cuda.get_device_name(0)
        except Exception:
            info["torch"] = None
            info["cuda_available"] = False
        return info

    def prepare(self) -> None:
        """Load CHGNet now so the first job does not pay for it.

        The load itself is inside the guard, not just the import of the wrapper:
        ``quick_mag.chgnet_runner`` imports cleanly on a machine with ase but no
        chgnet, and the missing package only announces itself when the calculator
        is actually built -- which is where the unhelpful bare ModuleNotFoundError
        would otherwise escape to the client.
        """
        try:
            from quick_mag.chgnet_runner import load_calculator

            self._calculator = load_calculator(self.device)
        except ImportError as exc:
            raise ImportError(f"{MISSING_DEPENDENCY_MESSAGE}\n(import error: {exc})") from exc

    # -- the work ----------------------------------------------------------

    def run(
        self,
        kind: str,
        structure: ChemicalStructure,
        params: Dict[str, Any],
        *,
        progress: Optional[ProgressCallback] = None,
        should_stop: Optional[StopCheck] = None,
    ) -> Dict[str, Any]:
        if kind == "reconstruct":
            return self._run_reconstruction(structure, params, progress, should_stop)
        if kind != "chgnet":
            raise ValueError(f"InlineExecutor cannot run job kind '{kind}'.")
        try:
            from quick_mag.chgnet_runner import (
                CalculationCancelled,
                run_chgnet_calculation,
            )
        except ImportError as exc:
            raise ImportError(f"{MISSING_DEPENDENCY_MESSAGE}\n(import error: {exc})") from exc

        if self._calculator is None:
            self.prepare()

        try:
            result = run_chgnet_calculation(
                structure,
                params["calculation"],
                optimizer=params["optimizer"],
                fmax=params["fmax"],
                steps=params["steps"],
                calculator=self._calculator,
                progress=progress,
                should_stop=should_stop,
            )
        except CalculationCancelled as exc:
            # Translated at the boundary so the server never has to import
            # chgnet_runner just to recognize a cancellation. The partial
            # result, when there is one, rides along.
            partial = getattr(exc, "partial", None)
            payload = protocol.result_to_payload(partial) if partial is not None else None
            raise JobCancelled(str(exc), payload) from exc

        return protocol.result_to_payload(result)

    @staticmethod
    def _run_reconstruction(
        structure: ChemicalStructure,
        params: Dict[str, Any],
        progress: Optional[ProgressCallback],
        should_stop: Optional[StopCheck],
    ) -> Dict[str, Any]:
        """Fit the closest ideal structure. Needs numpy and scipy, nothing else."""
        from quick_mag.reconstruction import (
            ReconstructionCancelled,
            ReconstructionJob,
            reconstruction_to_payload,
        )

        job = ReconstructionJob(structure, tilt_systems=params.get("tilt_systems"))
        try:
            result = job.run(progress=progress, should_stop=should_stop)
        except ReconstructionCancelled as exc:
            raise JobCancelled(str(exc)) from exc
        return reconstruction_to_payload(result)


__all__ = ["InlineExecutor", "JobCancelled", "JobExecutor", "MISSING_DEPENDENCY_MESSAGE"]
