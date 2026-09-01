"""The HTTP service that runs quick-mag calculations for a remote UI.

Deliberately built on :mod:`http.server` rather than a framework: the whole
surface is six routes, and a server that installs with nothing but the package's
own dependencies is far easier to stand up on a machine whose network policy you
do not control.

Shape of it: submissions land in a queue, one worker thread drains it, and job
records live in memory behind a lock. One worker on purpose -- CHGNet wants the
device serially, so a second would only interleave badly.

Security posture. The server binds ``127.0.0.1`` by default and expects to be
reached through an SSH tunnel (``ssh -N -L 8765:127.0.0.1:8765 host``), which
makes SSH the authentication and leaves nothing listening on the network. The
bearer token is the second layer, and it is not optional-by-accident: once a page
has been granted local-network permission, *any* site the browser visits can
reach this port, so a token is what stops an unrelated tab from queueing work.

This module runs only on the compute host and is excluded from the browser
manifest; it may import CHGNet (lazily, via the executor).
"""

from __future__ import annotations

import argparse
import hmac
import os
import queue
import secrets
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional

from quick_mag.remote import protocol
from quick_mag.remote.executor import InlineExecutor, JobCancelled, JobExecutor
from quick_mag.structure import ChemicalStructure

DEFAULT_PORT = 8765
DEFAULT_HOST = "127.0.0.1"

# Terminal jobs kept for collection. The client polls by id, so a handful of
# completed jobs have to survive long enough to be read; beyond that they are
# just a slow leak in a process meant to run for weeks.
RETAINED_TERMINAL_JOBS = 64

# Read from the environment first so a service unit can supply the token without
# putting it on a command line that shows up in `ps`.
TOKEN_ENV_VAR = "QUICK_MAG_TOKEN"

# Request bodies larger than this are refused unread: a structure is tens of KB.
MAX_BODY_BYTES = 32 * 1024 * 1024


@dataclass
class Job:
    """One submitted calculation and everything known about it so far."""

    id: str
    kind: str
    label: str
    structure: ChemicalStructure
    params: Dict[str, Any]
    status: str = protocol.STATUS_QUEUED
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    progress: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cancel_requested: bool = False

    def summary(self) -> Dict[str, Any]:
        """Everything but the result -- what a job list needs."""
        return {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "atom_count": self.structure.atom_count,
            "calculation": self.params.get("calculation"),
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed": self._elapsed(),
            "progress": self.progress,
            "error": self.error,
        }

    def detail(self) -> Dict[str, Any]:
        payload = self.summary()
        payload["protocol"] = protocol.PROTOCOL_VERSION
        if self.result is not None:
            payload["result"] = self.result
        return payload

    def _elapsed(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else time.time()
        return end - self.started_at


class JobStore:
    """Job records plus the worker thread that drains them.

    Every mutation goes through ``_lock``; the worker writes and the request
    threads read, and progress updates arrive often enough that a torn read would
    otherwise be a real possibility rather than a theoretical one.
    """

    def __init__(self, executor: JobExecutor, *, max_atoms: Optional[int] = None) -> None:
        self.executor = executor
        self.max_atoms = max_atoms
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._counter = 0
        # Distinguishes ids issued by this process from ones a client remembers
        # across a server restart, so a stale poll 404s instead of silently
        # matching an unrelated new job.
        self._instance = secrets.token_hex(3)
        self._worker = threading.Thread(target=self._work, name="quick-mag-worker", daemon=True)
        self._worker.start()

    # -- submission --------------------------------------------------------

    def submit(self, request: Dict[str, Any]) -> Job:
        with self._lock:
            self._counter += 1
            job_id = f"{self._instance}-{self._counter:03d}"
            job = Job(
                id=job_id,
                kind=request["kind"],
                label=request["label"],
                structure=request["structure"],
                params=request["params"],
            )
            self._jobs[job_id] = job
            self._order.append(job_id)
        self._queue.put(job_id)
        return job

    # -- reads -------------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def summaries(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._jobs[job_id].summary() for job_id in self._order]

    def queue_depth(self) -> int:
        with self._lock:
            return sum(
                1 for job in self._jobs.values() if job.status == protocol.STATUS_QUEUED
            )

    def running_job_id(self) -> Optional[str]:
        with self._lock:
            for job in self._jobs.values():
                if job.status == protocol.STATUS_RUNNING:
                    return job.id
        return None

    # -- cancellation ------------------------------------------------------

    def cancel(self, job_id: str) -> Optional[Job]:
        """Flag a job for cancellation; the worker does the actual stopping.

        A queued job is finished here and now -- nothing has started, so there is
        nothing to unwind. A running one only gets a flag, which the optimizer's
        per-step observer notices on its next step.
        """
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if job.status in protocol.TERMINAL_STATUSES:
                return job
            job.cancel_requested = True
            if job.status == protocol.STATUS_QUEUED:
                job.status = protocol.STATUS_CANCELLED
                job.finished_at = time.time()
                job.error = "Cancelled before it started."
            return job

    # -- the worker --------------------------------------------------------

    def _work(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.get(job_id)
            if job is None:
                continue
            with self._lock:
                if job.status != protocol.STATUS_QUEUED:
                    continue  # cancelled while it waited
                job.status = protocol.STATUS_RUNNING
                job.started_at = time.time()
            self._run_job(job)

    def _run_job(self, job: Job) -> None:
        def on_progress(update: Dict[str, Any]) -> None:
            with self._lock:
                job.progress = update

        def should_stop() -> bool:
            with self._lock:
                return job.cancel_requested

        try:
            result = self.executor.run(
                job.kind,
                job.structure,
                job.params,
                progress=on_progress,
                should_stop=should_stop,
            )
        except JobCancelled as exc:
            self._finish(job, protocol.STATUS_CANCELLED, error=str(exc))
        except Exception as exc:  # noqa: BLE001 - one bad job must not kill the worker
            # The traceback goes to the server's own log; the client gets the
            # exception's message, which is the part a person can act on.
            traceback.print_exc()
            self._finish(job, protocol.STATUS_ERROR, error=f"{type(exc).__name__}: {exc}")
        else:
            self._finish(job, protocol.STATUS_DONE, result=result)
        self._prune()

    def _finish(
        self,
        job: Job,
        status: str,
        *,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            job.status = status
            job.finished_at = time.time()
            job.result = result
            job.error = error

    def _prune(self) -> None:
        """Drop the oldest terminal jobs once too many have piled up."""
        with self._lock:
            terminal = [
                job_id
                for job_id in self._order
                if self._jobs[job_id].status in protocol.TERMINAL_STATUSES
            ]
            excess = len(terminal) - RETAINED_TERMINAL_JOBS
            for job_id in terminal[:excess] if excess > 0 else []:
                self._jobs.pop(job_id, None)
                self._order.remove(job_id)


class _Handler(BaseHTTPRequestHandler):
    """Routing, CORS and auth. All state lives on ``self.server``."""

    protocol_version = "HTTP/1.1"
    server_version = "quick-mag"
    sys_version = ""

    # -- plumbing ----------------------------------------------------------

    @property
    def store(self) -> JobStore:
        return self.server.store  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        if getattr(self.server, "quiet", False):  # type: ignore[attr-defined]
            return
        sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {fmt % args}\n")

    def _allowed_origin(self) -> Optional[str]:
        allowed = self.server.allow_origins  # type: ignore[attr-defined]
        origin = self.headers.get("Origin")
        if "*" in allowed:
            return origin or "*"
        if origin and origin in allowed:
            return origin
        return None

    def _send(self, status: int, payload: Optional[Dict[str, Any]] = None) -> None:
        body = protocol.dumps(payload).encode("utf-8") if payload is not None else b""
        self.send_response(status)
        if payload is not None:
            self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        origin = self._allowed_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the UI navigated away mid-poll; nothing to do about it

    def _error(self, status: int, message: str) -> None:
        self._send(status, {"error": message, "protocol": protocol.PROTOCOL_VERSION})

    def _authorized(self) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        if not token:
            return True
        header = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not header.startswith(prefix):
            return False
        # Constant-time: the token is short and an attacker can retry freely.
        return hmac.compare_digest(header[len(prefix):], token)

    def _read_body(self) -> Dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise protocol.ProtocolError("Malformed Content-Length header.")
        if length <= 0:
            raise protocol.ProtocolError("Request body is empty.")
        if length > MAX_BODY_BYTES:
            raise protocol.ProtocolError(
                f"Request body is {length} bytes; the limit is {MAX_BODY_BYTES}."
            )
        return protocol.loads(self.rfile.read(length).decode("utf-8"))

    # -- routes ------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's naming
        """CORS preflight.

        Note what is *not* here: no ``Access-Control-Allow-Private-Network``.
        Chrome's Private Network Access scheme, which needed that header, was
        replaced by Local Network Access, where a user permission prompt gates
        the request and the target device opts into nothing.
        """
        self.send_response(204)
        origin = self._allowed_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"

        if path == "/":
            # A person who pastes the URL into a browser deserves a sentence.
            self._send(200, {"service": "quick-mag", "health": "/health", "jobs": "/jobs"})
            return
        if path == "/health":
            self._send(200, self._health())
            return
        if not self._authorized():
            self._error(401, "Missing or invalid bearer token.")
            return
        if path == "/jobs":
            self._send(200, {"jobs": self.store.summaries()})
            return
        if path.startswith("/jobs/"):
            job = self.store.get(path[len("/jobs/"):])
            if job is None:
                self._error(404, "No such job. The server may have restarted.")
                return
            self._send(200, job.detail())
            return
        self._error(404, f"No route for GET {path}.")

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/jobs":
            self._error(404, f"No route for POST {path}.")
            return
        if not self._authorized():
            self._error(401, "Missing or invalid bearer token.")
            return
        try:
            payload = self._read_body()
            request = protocol.parse_job_request(payload, max_atoms=self.store.max_atoms)
        except protocol.ProtocolError as exc:
            self._error(400, str(exc))
            return

        job = self.store.submit(request)
        self.log_message(
            "queued %s  %s  %s  %d atoms",
            job.id, job.label, job.params["calculation"], job.structure.atom_count,
        )
        self._send(202, job.detail())

    def do_DELETE(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if not path.startswith("/jobs/"):
            self._error(404, f"No route for DELETE {path}.")
            return
        if not self._authorized():
            self._error(401, "Missing or invalid bearer token.")
            return
        job = self.store.cancel(path[len("/jobs/"):])
        if job is None:
            self._error(404, "No such job.")
            return
        self._send(200, job.detail())

    def _health(self) -> Dict[str, Any]:
        """Unauthenticated on purpose: it exposes no job data, and it is what the
        UI's Connect button uses to tell 'wrong URL' from 'wrong token'."""
        payload = {
            "ok": True,
            "service": "quick-mag",
            "protocol": protocol.PROTOCOL_VERSION,
            "kinds": list(protocol.JOB_KINDS),
            "requires_token": bool(self.server.token),  # type: ignore[attr-defined]
            "queue_depth": self.store.queue_depth(),
            "running": self.store.running_job_id(),
            "max_atoms": self.store.max_atoms,
        }
        payload.update(self.store.executor.describe())
        return payload


class QuickMagServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address,
        *,
        store: JobStore,
        token: Optional[str],
        allow_origins: List[str],
        quiet: bool = False,
    ) -> None:
        super().__init__(address, _Handler)
        self.store = store
        self.token = token
        self.allow_origins = allow_origins
        self.quiet = quiet


def create_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    token: Optional[str] = None,
    allow_origins: Optional[List[str]] = None,
    device: Optional[str] = None,
    max_atoms: Optional[int] = None,
    executor: Optional[JobExecutor] = None,
    quiet: bool = False,
) -> QuickMagServer:
    store = JobStore(executor or InlineExecutor(device=device), max_atoms=max_atoms)
    return QuickMagServer(
        (host, port),
        store=store,
        token=token,
        allow_origins=list(allow_origins or ["*"]),
        quiet=quiet,
    )


SERVE_DESCRIPTION = (
    "Run the calculation server that a remote quick-mag UI submits jobs to. "
    "Binds 127.0.0.1 by default: reach it from another machine with an SSH "
    "tunnel (ssh -N -L 8765:127.0.0.1:8765 HOST) rather than by opening a port."
)


def configure_serve_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--host", default=DEFAULT_HOST,
        help=f"Interface to bind (default {DEFAULT_HOST}; use 0.0.0.0 only behind "
        "a firewall you trust).",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"Port to listen on (default {DEFAULT_PORT}).",
    )
    parser.add_argument(
        "--token", default=None,
        help=f"Bearer token clients must present. Defaults to ${TOKEN_ENV_VAR} if "
        "set, otherwise one is generated and printed at startup.",
    )
    parser.add_argument(
        "--no-token", action="store_true",
        help="Disable token auth. Only sensible on a machine nothing else can reach.",
    )
    parser.add_argument(
        "--allow-origin", action="append", default=None,
        help="Browser origin allowed to call this server; repeatable. Defaults to "
        "any origin, which is safe in combination with a token.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device for CHGNet (e.g. cuda, cpu). Default: let CHGNet choose.",
    )
    parser.add_argument(
        "--max-atoms", type=int, default=None,
        help="Reject structures larger than this many atoms.",
    )
    parser.add_argument(
        "--no-preload", action="store_true",
        help="Skip loading the CHGNet model at startup (the first job loads it).",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-request logging.",
    )
    return parser


def run_serve(args: argparse.Namespace) -> int:
    token = None if args.no_token else (args.token or os.environ.get(TOKEN_ENV_VAR) or secrets.token_urlsafe(16))
    server = create_server(
        host=args.host,
        port=args.port,
        token=token,
        allow_origins=args.allow_origin,
        device=args.device,
        max_atoms=args.max_atoms,
        quiet=args.quiet,
    )

    print(f"quick-mag server listening on http://{args.host}:{args.port}")
    if token:
        print(f"  token: {token}")
        print("  (paste this into the UI's Remote compute panel)")
    else:
        print("  token: disabled (--no-token)")

    if not args.no_preload:
        print("  loading CHGNet...", flush=True)
        try:
            server.store.executor.prepare()
        except Exception as exc:  # noqa: BLE001
            print(f"  could not load CHGNet: {exc}", file=sys.stderr)
            print("  the server is up; jobs will fail until this is fixed.", file=sys.stderr)
        else:
            print("  ready.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down.")
    finally:
        server.server_close()
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=SERVE_DESCRIPTION)
    configure_serve_parser(parser)
    return run_serve(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
