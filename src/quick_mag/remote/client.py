"""Talking to a quick-mag calculation server without blocking the frame loop.

The UI is an immediate-mode loop: whatever happens here must return in
microseconds and be picked up on a later frame. That is the same constraint the
browser file-upload path already solves -- an async producer fills a queue, and
:func:`_drain_browser_uploads` empties it once a frame -- so this module uses the
same shape, with one queue and one ``poll()`` call per frame regardless of which
machine the work is running on.

Two transports sit behind one interface:

* :class:`ThreadTransport` -- a daemon thread per request over ``urllib``. Used by
  the desktop app.
* :class:`BrowserTransport` -- hands the request to ``window.quickMagRemote`` and
  reads results back off a JS array. Used by the Pyodide build, where ``urllib``
  cannot open a socket at all.

Both are stdlib-only and this module is staged into the browser, so it must not
import ase, chgnet, or :mod:`quick_mag.chgnet_runner`.
"""

from __future__ import annotations

import itertools
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from quick_mag.remote import protocol
from quick_mag.structure import ChemicalStructure

DEFAULT_URL = "http://127.0.0.1:8765"

# How often a live job is re-polled. Emphatically not once a frame: at 60 fps that
# would be 60 requests a second down an SSH tunnel to watch a relaxation that
# emits a step every few hundred milliseconds.
POLL_INTERVAL_SECONDS = 1.0

# After a failed poll the interval doubles up to this ceiling, so a dropped VPN
# does not turn into a tight retry loop.
MAX_POLL_INTERVAL_SECONDS = 10.0

# Consecutive transport failures before the UI is told the server is unreachable.
# One is not enough: a tunnel hiccups.
FAILURES_BEFORE_DISCONNECTED = 3

REQUEST_TIMEOUT_SECONDS = 20.0


@dataclass
class Response:
    """One completed request, whichever transport carried it."""

    request_id: str
    ok: bool
    status: int = 0
    text: str = ""
    error: str = ""


class Transport:
    """Fire-and-forget request delivery with results collected later."""

    def request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[str],
    ) -> None:
        raise NotImplementedError

    def drain(self) -> List[Response]:
        """Responses that arrived since the last call. Never blocks."""
        raise NotImplementedError


class ThreadTransport(Transport):
    """Desktop transport: one daemon thread per request, results into a list.

    A thread per request rather than a pool because the client issues at most a
    couple of requests a second by construction, and a pool would add a shutdown
    problem to a process that currently has none.
    """

    def __init__(self, timeout: float = REQUEST_TIMEOUT_SECONDS) -> None:
        self.timeout = timeout
        self._lock = threading.Lock()
        self._done: List[Response] = []

    def request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[str],
    ) -> None:
        thread = threading.Thread(
            target=self._run,
            args=(request_id, method, url, dict(headers), body),
            name=f"quick-mag-remote-{request_id}",
            daemon=True,
        )
        thread.start()

    def _run(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[str],
    ) -> None:
        payload = body.encode("utf-8") if body is not None else None
        request = urllib.request.Request(url, data=payload, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
                result = Response(request_id, True, response.status, text)
        except urllib.error.HTTPError as exc:
            # A 4xx still carries a JSON body explaining what was wrong with the
            # request, which is far more useful than the status line alone.
            text = ""
            try:
                text = exc.read().decode("utf-8")
            except Exception:  # noqa: BLE001
                pass
            result = Response(request_id, False, exc.code, text, _http_error_message(exc.code, text))
        except Exception as exc:  # noqa: BLE001 - urllib raises a wide zoo
            result = Response(request_id, False, 0, "", f"{type(exc).__name__}: {exc}")
        with self._lock:
            self._done.append(result)

    def drain(self) -> List[Response]:
        with self._lock:
            responses, self._done = self._done, []
        return responses


def is_browser_runtime() -> bool:
    """True only inside a real Pyodide build.

    Emphatically not ``import js``: that is a probe for a module name, and
    ``js`` is also an ordinary PyPI package, so on a desktop with one installed
    the probe succeeds and the browser transport gets chosen for a process that
    has no page. Pyodide runs on Emscripten and nothing else does, so the
    platform is the question actually being asked.
    """
    return sys.platform == "emscripten"


class BrowserTransport(Transport):
    """Pyodide transport: the page owns the fetch, Python owns the queue.

    Mirrors the upload and download bridges already in ``web/index.html``. The
    page pushes JSON *strings* onto ``window.quickMagRemoteResults`` rather than
    objects, so nothing has to survive a JS-to-Python object conversion.
    """

    RESULT_QUEUE = "quickMagRemoteResults"

    def __init__(self) -> None:
        if not is_browser_runtime():
            raise RuntimeError(
                "BrowserTransport needs a Pyodide runtime; this is "
                f"{sys.platform}. Use ThreadTransport instead."
            )
        import js  # type: ignore

        if not hasattr(js, "window"):
            raise RuntimeError(
                "The 'js' module here is not Pyodide's - it exposes no window. "
                "Something else on sys.path is shadowing it."
            )
        # Failures raised on the way *into* JavaScript are recorded here rather
        # than on the page's queue: if the bridge is what broke, the queue is
        # exactly the thing that cannot be trusted to carry the news.
        self._local_failures: List[Response] = []

    def _bridge(self):
        import js  # type: ignore

        bridge = getattr(js.window, "quickMagRemote", None)
        if bridge is None:
            raise RuntimeError(
                "This page has no remote-compute bridge. It is running a cached "
                "index.html from before the feature existed - reload with a hard "
                "refresh."
            )
        return bridge

    def request(
        self,
        request_id: str,
        method: str,
        url: str,
        headers: Dict[str, str],
        body: Optional[str],
    ) -> None:
        try:
            self._bridge().request(request_id, method, url, json.dumps(headers), body)
        except Exception as exc:  # noqa: BLE001
            self._local_failures.append(
                Response(request_id, False, 0, "", f"{type(exc).__name__}: {exc}")
            )

    def drain(self) -> List[Response]:
        import js  # type: ignore

        responses, self._local_failures = self._local_failures, []
        queue = getattr(js.window, self.RESULT_QUEUE, None)
        if not queue or not len(queue):
            return responses
        raw = [str(queue[index]) for index in range(len(queue))]
        queue.length = 0  # clear the JS side so nothing is delivered twice
        for item in raw:
            try:
                envelope = json.loads(item)
            except ValueError:
                continue
            responses.append(
                Response(
                    request_id=str(envelope.get("id", "")),
                    ok=bool(envelope.get("ok")),
                    status=int(envelope.get("status") or 0),
                    text=str(envelope.get("text") or ""),
                    error=str(envelope.get("error") or ""),
                )
            )
        return responses


def _http_error_message(status: int, text: str) -> str:
    """Turn a status code and body into something worth showing a person."""
    detail = ""
    try:
        payload = json.loads(text) if text else {}
        detail = str(payload.get("error") or "")
    except ValueError:
        detail = text.strip()[:200]
    if status == 401:
        return detail or "The server rejected the token."
    if status == 404:
        return detail or "The server has no record of that job."
    if status == 400:
        return detail or "The server rejected the request."
    return detail or f"HTTP {status}."


def default_transport() -> Transport:
    """Whichever transport this build can actually use.

    Falls back rather than propagating: a desktop process must never end up
    holding a transport that needs a page, and a browser build with a broken
    bridge is better off reporting failed requests than refusing to start.
    """
    if not is_browser_runtime():
        return ThreadTransport()
    try:
        return BrowserTransport()
    except Exception:  # noqa: BLE001
        return ThreadTransport()


@dataclass
class RemoteJob:
    """A submitted calculation, as the client tracks it.

    ``template`` is the structure that was submitted. Holding on to it is what
    lets the relaxed structure keep its provenance: the result carries only
    numbers, and everything else is recovered from here.
    """

    key: str
    label: str
    template: ChemicalStructure
    params: Dict[str, Any]
    id: Optional[str] = None
    status: str = protocol.STATUS_QUEUED
    submitted_at: float = field(default_factory=time.time)
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    structure: Optional[ChemicalStructure] = None
    #: Wall-clock stamp at the moment this job reached a terminal state. Set once,
    #: so the elapsed readout stops counting instead of ticking up forever beside
    #: a job that finished ten minutes ago.
    finished_at: Optional[float] = None
    collected: bool = False
    next_poll_at: float = 0.0
    poll_interval: float = POLL_INTERVAL_SECONDS
    consecutive_failures: int = 0
    _inflight: bool = False
    _cancel_sent: bool = False

    @property
    def is_live(self) -> bool:
        return self.status in protocol.LIVE_STATUSES

    @property
    def step(self) -> int:
        return int(self.progress.get("step") or 0)

    @property
    def energy(self) -> Optional[float]:
        value = self.progress.get("energy")
        return float(value) if value is not None else None

    @property
    def max_force(self) -> Optional[float]:
        value = self.progress.get("max_force")
        return float(value) if value is not None else None

    def trajectory(self) -> List[float]:
        if self.result is not None:
            return protocol.trajectory_from(self.result)
        return protocol.trajectory_from(self.progress)

    def elapsed(self) -> float:
        """How long this job took, or has taken so far. Frozen once it finishes."""
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.submitted_at

    def status_line(self) -> str:
        """One line for the job list. The UI never has to compose this itself."""
        if self.status == protocol.STATUS_QUEUED:
            return "queued"
        if self.status == protocol.STATUS_RUNNING:
            energy = self.energy
            if energy is None:
                return "running"
            force = self.max_force
            tail = f", |F|max {force:.3f}" if force is not None else ""
            return f"step {self.step}, E = {energy:.4f} eV{tail}"
        if self.status == protocol.STATUS_DONE:
            energy = (self.result or {}).get("energy")
            steps = (self.result or {}).get("steps", 0)
            converged = (self.result or {}).get("converged", True)
            state = "converged" if converged else "NOT converged"
            if energy is None:
                return f"done ({state})"
            return f"E = {float(energy):.6f} eV, {steps} steps, {state}"
        if self.status == protocol.STATUS_CANCELLED:
            return "cancelled"
        return self.error or "failed"


class RemoteClient:
    """Submits jobs and tracks them, one non-blocking ``poll()`` per frame."""

    def __init__(
        self,
        url: str = DEFAULT_URL,
        token: str = "",
        transport: Optional[Transport] = None,
        poll_interval: float = POLL_INTERVAL_SECONDS,
    ) -> None:
        self.url = url
        self.token = token
        self.poll_interval = poll_interval
        self.transport = transport if transport is not None else default_transport()
        self.jobs: List[RemoteJob] = []
        self.health: Optional[Dict[str, Any]] = None
        self.health_error: Optional[str] = None
        self.message: str = ""
        self.connected: bool = False
        self._by_key: Dict[str, RemoteJob] = {}
        self._inflight: Dict[str, Any] = {}
        self._ids = itertools.count(1)
        self._local_failures: List[Response] = []

    # -- helpers -----------------------------------------------------------

    def _endpoint(self, path: str) -> str:
        return f"{self.url.rstrip('/')}{path}"

    def _headers(self, *, json_body: bool = False) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if json_body:
            headers["Content-Type"] = "application/json"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _send(self, intent: str, method: str, path: str, body: Optional[str] = None, **extra: Any) -> str:
        """Dispatch a request. Never raises.

        Every caller is reached from an imgui callback, and an exception there
        unwinds through the C++ frame loop and takes the window down. A transport
        that cannot even accept the request is turned into a failed response
        instead, so it arrives at the panel as a message like any other failure.
        """
        request_id = f"r{next(self._ids)}"
        self._inflight[request_id] = {"intent": intent, **extra}
        try:
            self.transport.request(
                request_id,
                method,
                self._endpoint(path),
                self._headers(json_body=body is not None),
                body,
            )
        except Exception as exc:  # noqa: BLE001
            self._local_failures.append(
                Response(request_id, False, 0, "", f"{type(exc).__name__}: {exc}")
            )
        return request_id

    # -- public actions ----------------------------------------------------

    def check_health(self) -> None:
        """Ask the server what it is. Resolves on a later ``poll()``."""
        self.health_error = None
        self.message = "Connecting..."
        self._send("health", "GET", "/health")

    def submit(
        self,
        structure: ChemicalStructure,
        *,
        params: Optional[Dict[str, Any]] = None,
        label: Optional[str] = None,
    ) -> RemoteJob:
        """Queue a calculation. Raises :class:`protocol.ProtocolError` on bad params."""
        request = protocol.build_job_request(structure, params=params, label=label)
        job = RemoteJob(
            key=f"local-{next(self._ids)}",
            label=request["label"],
            template=structure,
            params=request["params"],
            poll_interval=self.poll_interval,
        )
        job._inflight = True
        self.jobs.append(job)
        self._by_key[job.key] = job
        self._send("submit", "POST", "/jobs", protocol.dumps(request), key=job.key)
        return job

    def cancel(self, job: RemoteJob) -> None:
        if job.id is None or not job.is_live or job._cancel_sent:
            return
        job._cancel_sent = True
        self._send("cancel", "DELETE", f"/jobs/{job.id}", key=job.key)

    def forget(self, job: RemoteJob) -> None:
        """Drop a finished job from the list. Live jobs keep running remotely."""
        self.jobs = [item for item in self.jobs if item.key != job.key]
        self._by_key.pop(job.key, None)

    def clear_finished(self) -> None:
        for job in [item for item in self.jobs if not item.is_live]:
            self.forget(job)

    @property
    def live_jobs(self) -> List[RemoteJob]:
        return [job for job in self.jobs if job.is_live]

    # -- the once-a-frame call --------------------------------------------

    def poll(self) -> List[RemoteJob]:
        """Collect responses and re-poll live jobs.

        Returns the jobs that reached a terminal state on this call, so the caller
        can act on each result exactly once.
        """
        finished: List[RemoteJob] = []
        delivered, self._local_failures = self._local_failures, []
        try:
            delivered.extend(self.transport.drain())
        except Exception as exc:  # noqa: BLE001
            # Same rule as _send: a broken transport must not unwind the frame.
            self.connected = False
            self.message = f"Transport failure: {exc}"
        for response in delivered:
            job = self._handle(response)
            if job is not None and not job.is_live and not job.collected:
                job.collected = True
                finished.append(job)

        now = time.time()
        for job in self.jobs:
            if not job.is_live or job.id is None or job._inflight:
                continue
            if now >= job.next_poll_at:
                job._inflight = True
                self._send("status", "GET", f"/jobs/{job.id}", key=job.key)
        return finished

    # -- response handling -------------------------------------------------

    def _handle(self, response: Response) -> Optional[RemoteJob]:
        pending = self._inflight.pop(response.request_id, None)
        if pending is None:
            return None  # a response to a request from a previous server/session
        intent = pending["intent"]
        job = self._by_key.get(pending.get("key", ""))
        if job is not None:
            job._inflight = False

        if intent == "health":
            self._handle_health(response)
            return None
        if job is None:
            return None
        if intent == "submit":
            return self._handle_submit(job, response)
        if intent == "status":
            return self._handle_status(job, response)
        if intent == "cancel":
            # A cancel is only a request; the next status poll reports what
            # actually happened, so a failure here is not worth surfacing.
            if response.ok:
                self._apply_payload(job, protocol.loads(response.text))
            return job
        return None

    def _handle_health(self, response: Response) -> None:
        if not response.ok:
            self.connected = False
            self.health = None
            self.health_error = response.error or "Could not reach the server."
            self.message = self.health_error
            return
        try:
            payload = protocol.loads(response.text)
            protocol.check_protocol(payload)
        except protocol.ProtocolError as exc:
            self.connected = False
            self.health = None
            self.health_error = str(exc)
            self.message = str(exc)
            return
        self.health = payload
        self.health_error = None
        self.connected = True
        device = payload.get("cuda_device") or payload.get("device") or "unknown device"
        self.message = f"Connected: {device}"

    def _handle_submit(self, job: RemoteJob, response: Response) -> Optional[RemoteJob]:
        if not response.ok:
            job.status = protocol.STATUS_ERROR
            job.error = response.error or "Submission failed."
            job.finished_at = time.time()
            self.message = f"{job.label}: {job.error}"
            return job
        try:
            payload = protocol.loads(response.text)
        except protocol.ProtocolError as exc:
            job.status = protocol.STATUS_ERROR
            job.error = str(exc)
            job.finished_at = time.time()
            return job
        job.id = str(payload.get("id") or "")
        self.connected = True
        self._apply_payload(job, payload)
        job.next_poll_at = time.time() + min(job.poll_interval, 0.25)
        return job if not job.is_live else None

    def _handle_status(self, job: RemoteJob, response: Response) -> Optional[RemoteJob]:
        if not response.ok:
            job.consecutive_failures += 1
            # Back off rather than give up: the usual cause is a tunnel that
            # dropped on a VPN reconnect, and the job is still running remotely.
            job.poll_interval = min(job.poll_interval * 2.0, MAX_POLL_INTERVAL_SECONDS)
            job.next_poll_at = time.time() + job.poll_interval
            if job.consecutive_failures >= FAILURES_BEFORE_DISCONNECTED:
                self.connected = False
                self.message = (
                    f"Lost contact with the server ({response.error}). "
                    "Still retrying; the job may still be running."
                )
            if response.status == 404:
                # The server restarted and has never heard of this job. Retrying
                # would poll forever.
                job.status = protocol.STATUS_ERROR
                job.error = "The server no longer has this job (did it restart?)."
                job.finished_at = time.time()
                return job
            return None

        job.consecutive_failures = 0
        job.poll_interval = self.poll_interval
        job.next_poll_at = time.time() + job.poll_interval
        self.connected = True
        try:
            self._apply_payload(job, protocol.loads(response.text))
        except protocol.ProtocolError as exc:
            job.status = protocol.STATUS_ERROR
            job.error = str(exc)
            job.finished_at = time.time()
        return job

    def _apply_payload(self, job: RemoteJob, payload: Dict[str, Any]) -> None:
        job.status = str(payload.get("status") or job.status)
        if not job.is_live and job.finished_at is None:
            job.finished_at = time.time()
        job.progress = dict(payload.get("progress") or {})
        job.error = payload.get("error") or job.error
        result = payload.get("result")
        if result is not None and job.result is None:
            job.result = result
            try:
                job.structure = protocol.structure_from_result(job.template, result)
            except protocol.ProtocolError as exc:
                job.status = protocol.STATUS_ERROR
                job.error = str(exc)


__all__ = [
    "BrowserTransport",
    "DEFAULT_URL",
    "RemoteClient",
    "RemoteJob",
    "Response",
    "ThreadTransport",
    "Transport",
    "default_transport",
    "is_browser_runtime",
]
