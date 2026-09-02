"""Client and server against each other over a real socket.

The executor is a stub rather than CHGNet: everything between the submit button
and the relaxed structure is exercised here, and none of it should depend on
having torch installed. The one thing a stub cannot cover is the CHGNet call
itself, which ``test_chgnet_interop`` already owns.
"""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quick_mag.remote import protocol  # noqa: E402
from quick_mag.remote.client import RemoteClient, ThreadTransport  # noqa: E402
from quick_mag.remote.executor import JobCancelled, JobExecutor  # noqa: E402
from quick_mag.remote.server import create_server  # noqa: E402
from quick_mag.structure import (  # noqa: E402
    ChemicalStructure,
    generate_random_test_perovskite,
)

TIMEOUT = 10.0


class StubExecutor(JobExecutor):
    """Stands in for CHGNet: reports a few steps, then shifts every atom by 0.1."""

    name = "stub"

    def __init__(self, steps: int = 3, block_until_cancelled: bool = False) -> None:
        self.steps = steps
        self.block_until_cancelled = block_until_cancelled
        self.prepared = False
        self.seen: List[str] = []

    def describe(self) -> Dict[str, Any]:
        return {"executor": self.name, "device": "stub", "model_loaded": self.prepared}

    def prepare(self) -> None:
        self.prepared = True

    def run(self, kind, structure, params, *, progress=None, should_stop=None):
        self.seen.append(structure.name)
        energies: List[float] = []
        step = 0
        while True:
            if should_stop is not None and should_stop():
                raise JobCancelled(f"Cancelled after {step} steps.")
            if not self.block_until_cancelled and step > self.steps:
                break
            energies.append(-100.0 - step)
            if progress is not None:
                progress(
                    {
                        "step": step,
                        "energy": energies[-1],
                        "max_force": 0.5 / (step + 1),
                        "trajectory_energies": list(energies),
                    }
                )
            step += 1
            if self.block_until_cancelled:
                time.sleep(0.01)

        coords = np.asarray(structure.cartesian_coords) + 0.1
        return {
            "calculation": params["calculation"],
            "energy": energies[-1],
            "energy_per_atom": energies[-1] / max(structure.atom_count, 1),
            "max_force": 0.004,
            "forces": np.zeros_like(coords).tolist(),
            "stress": [0.0] * 6,
            "magnetic_moments": [3.7] * structure.atom_count,
            "trajectory_energies": energies,
            "final_lattice": np.asarray(structure.lattice).tolist(),
            "final_coords": coords.tolist(),
            "steps": len(energies) - 1,
            "converged": True,
        }


class ServerFixture:
    """A server on an ephemeral port, plus a client pointed at it."""

    def __init__(self, executor: JobExecutor, token: Optional[str] = "s3cret", **kwargs):
        self.executor = executor
        self.server = create_server(
            host="127.0.0.1", port=0, token=token, executor=executor, quiet=True, **kwargs
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address[:2]
        self.url = f"http://{host}:{port}"

    def client(self, token: str = "s3cret") -> RemoteClient:
        return RemoteClient(
            self.url, token, transport=ThreadTransport(timeout=TIMEOUT), poll_interval=0.05
        )

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def pump(client: RemoteClient, until, timeout: float = TIMEOUT) -> List[Any]:
    """Drive ``poll()`` the way a frame loop would until ``until()`` is true."""
    finished: List[Any] = []
    deadline = time.time() + timeout
    while time.time() < deadline:
        finished.extend(client.poll())
        if until():
            return finished
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for the client to settle.")


def sample_structure() -> ChemicalStructure:
    return ChemicalStructure.with_zero_magnetic_moments(
        name="LaMnO3_test",
        lattice=np.diag([4.0, 4.0, 4.0]),
        cartesian_coords=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]]),
        atomic_labels=["La3+", "Mn3+"],
    )


class TestHealth(unittest.TestCase):
    def setUp(self):
        self.fixture = ServerFixture(StubExecutor())
        self.addCleanup(self.fixture.close)

    def test_health_reports_the_executor_and_needs_no_token(self):
        client = RemoteClient(self.fixture.url, "", transport=ThreadTransport(TIMEOUT))
        client.check_health()
        pump(client, lambda: client.health is not None or client.health_error)

        self.assertIsNone(client.health_error)
        self.assertTrue(client.connected)
        self.assertEqual(client.health["executor"], "stub")
        self.assertTrue(client.health["requires_token"])
        self.assertEqual(client.health["protocol"], protocol.PROTOCOL_VERSION)

    def test_a_wrong_url_reports_rather_than_raises(self):
        client = RemoteClient(
            "http://127.0.0.1:1", "", transport=ThreadTransport(timeout=1.0)
        )
        client.check_health()
        pump(client, lambda: client.health_error is not None)
        self.assertFalse(client.connected)


class TestJobLifecycle(unittest.TestCase):
    def setUp(self):
        self.executor = StubExecutor()
        self.fixture = ServerFixture(self.executor)
        self.addCleanup(self.fixture.close)

    def test_a_job_runs_and_the_relaxed_structure_comes_back(self):
        client = self.fixture.client()
        structure = sample_structure()
        job = client.submit(structure, params={"calculation": "cell+atoms"})

        finished = pump(client, lambda: not job.is_live and job.result is not None)

        self.assertEqual(job.status, protocol.STATUS_DONE)
        self.assertIn(job, finished)
        self.assertIsNotNone(job.structure)
        np.testing.assert_allclose(
            job.structure.cartesian_coords, structure.cartesian_coords + 0.1
        )
        self.assertEqual(job.structure.atomic_labels, structure.atomic_labels)
        self.assertEqual(job.structure.name, "LaMnO3_test_chgnet")
        self.assertTrue(job.trajectory())

    def test_a_finished_job_is_handed_over_exactly_once(self):
        # The UI adds a structure for each job the poll reports, so a job
        # reported twice would silently duplicate it in the structure list.
        client = self.fixture.client()
        job = client.submit(sample_structure())
        pump(client, lambda: not job.is_live)
        for _ in range(5):
            self.assertEqual(client.poll(), [])

    def test_provenance_survives_the_round_trip(self):
        original, _ = generate_random_test_perovskite(np.random.default_rng(11))
        client = self.fixture.client()
        job = client.submit(original, params={"calculation": "atoms"})
        pump(client, lambda: not job.is_live)

        self.assertIs(
            job.structure.generation_parameters, original.generation_parameters
        )

    def test_progress_arrives_while_the_job_runs(self):
        client = self.fixture.client()
        job = client.submit(sample_structure())
        pump(client, lambda: not job.is_live)
        # The stub finishes fast, so assert on the final trace rather than
        # racing to observe a mid-flight step.
        self.assertGreaterEqual(len(job.trajectory()), 2)
        self.assertIn("E =", job.status_line())

    def test_the_job_list_survives_a_client_that_asks_for_it(self):
        client = self.fixture.client()
        job = client.submit(sample_structure())
        pump(client, lambda: not job.is_live)
        self.assertEqual(len(client.jobs), 1)
        client.clear_finished()
        self.assertEqual(client.jobs, [])


class TestFailureModes(unittest.TestCase):
    def test_a_bad_token_is_reported_in_words(self):
        fixture = ServerFixture(StubExecutor())
        self.addCleanup(fixture.close)
        client = fixture.client(token="wrong")
        job = client.submit(sample_structure())
        pump(client, lambda: not job.is_live)

        self.assertEqual(job.status, protocol.STATUS_ERROR)
        self.assertIn("token", (job.error or "").lower())

    def test_an_oversized_structure_is_refused_at_submission(self):
        fixture = ServerFixture(StubExecutor(), max_atoms=1)
        self.addCleanup(fixture.close)
        client = fixture.client()
        job = client.submit(sample_structure())
        pump(client, lambda: not job.is_live)

        self.assertEqual(job.status, protocol.STATUS_ERROR)
        self.assertIn("at most 1", job.error)

    def test_an_executor_failure_becomes_a_readable_job_error(self):
        class Broken(StubExecutor):
            def run(self, *args, **kwargs):
                raise RuntimeError("the model is not on this machine")

        fixture = ServerFixture(Broken())
        self.addCleanup(fixture.close)
        client = fixture.client()
        job = client.submit(sample_structure())
        pump(client, lambda: not job.is_live)

        self.assertEqual(job.status, protocol.STATUS_ERROR)
        self.assertIn("not on this machine", job.error)

    def test_a_running_job_can_be_cancelled(self):
        fixture = ServerFixture(StubExecutor(block_until_cancelled=True))
        self.addCleanup(fixture.close)
        client = fixture.client()
        job = client.submit(sample_structure())

        pump(client, lambda: job.status == protocol.STATUS_RUNNING)
        client.cancel(job)
        pump(client, lambda: not job.is_live)

        self.assertEqual(job.status, protocol.STATUS_CANCELLED)
        self.assertIsNone(job.structure)


if __name__ == "__main__":
    unittest.main()


HAS_ASE = __import__("importlib.util", fromlist=["util"]).find_spec("ase") is not None


@unittest.skipUnless(HAS_ASE, "ase is not installed (pip install -e '.[chgnet]')")
class TestTheWholeStack(unittest.TestCase):
    """Submit button to relaxed structure, with only the model itself faked.

    The InlineExecutor is the real one and it drives the real ASE optimizer; the
    only substitution is the calculator, because CHGNet's weights would mean a
    torch install for a test whose subject is the plumbing. Everything between --
    the executor, the queue, HTTP, JSON, the client's polling, and the rebuild of
    the structure against its template -- is exactly what runs in production.
    """

    def setUp(self):
        from quick_mag.remote.executor import InlineExecutor
        from test_chgnet_interop import _EMTWithMoments, _perturbed_copper

        executor = InlineExecutor()
        # Pre-set so prepare() never reaches for CHGNet.
        executor._calculator = _EMTWithMoments.build()
        self.structure = _perturbed_copper()
        self.fixture = ServerFixture(executor)
        self.addCleanup(self.fixture.close)

    def test_a_relaxation_makes_the_round_trip(self):
        client = self.fixture.client()
        job = client.submit(
            self.structure, params={"calculation": "atoms", "fmax": 0.05, "steps": 50}
        )
        pump(client, lambda: not job.is_live, timeout=60.0)

        self.assertEqual(job.status, protocol.STATUS_DONE, job.error)
        relaxed = job.structure
        self.assertIsNotNone(relaxed)
        self.assertEqual(relaxed.atom_count, self.structure.atom_count)
        # It moved, and it moved towards something: the trace has to fall.
        self.assertFalse(
            np.allclose(relaxed.cartesian_coords, self.structure.cartesian_coords)
        )
        trace = job.trajectory()
        self.assertGreater(len(trace), 1)
        self.assertLess(trace[-1], trace[0])
        self.assertTrue(job.result["converged"])
        # Atoms-only: the lattice must come back untouched.
        np.testing.assert_allclose(relaxed.lattice, self.structure.lattice)

    def test_chgnet_magnitudes_survive_as_unsigned_diagnostics(self):
        client = self.fixture.client()
        job = client.submit(self.structure, params={"calculation": "single-point"})
        pump(client, lambda: not job.is_live, timeout=60.0)

        moments = protocol.moments_from_result(job.result)
        self.assertEqual(moments.shape, (self.structure.atom_count,))
        self.assertTrue((moments >= 0).all())
        # And they are nowhere near the structure's own (signed, solver-owned) moments.
        np.testing.assert_allclose(job.structure.magnetic_moments, 0.0)


class TestReconstructionJobs(unittest.TestCase):
    """The "reconstruct" job kind: the fit runs on the server, not in the UI."""

    def setUp(self):
        from quick_mag.remote.executor import InlineExecutor

        # The real executor: reconstruction needs numpy and scipy only, and its
        # branch comes before the CHGNet import, so no torch is needed here.
        self.fixture = ServerFixture(InlineExecutor())
        self.addCleanup(self.fixture.close)

    @staticmethod
    def relaxed_structure():
        import copy

        from quick_mag.domains import DomainSpec
        from quick_mag.generation import stacked_structure_from_domains

        truth = stacked_structure_from_domains(
            [DomainSpec(n_cells=(2, 2, 2), lattice=(4.0, 4.0, 4.0))],
            name="truth",
            periodic=True,
            tilt_system="a0a0c-",
            tilt_angles_deg=(0.0, 0.0, 6.0),
        )
        relaxed = copy.deepcopy(truth)
        relaxed.name = "relaxed"
        relaxed.cartesian_coords = truth.cartesian_coords + np.array([0.2, 0.0, 0.1])
        relaxed.geometry_matches_generation = False
        return relaxed

    def test_provenance_travels_only_for_a_reconstruction(self):
        structure = self.relaxed_structure()
        chgnet = protocol.build_job_request(structure, params={"calculation": "atoms"})
        self.assertNotIn("generation_parameters", chgnet["structure"])
        fit = protocol.build_job_request(
            structure, kind="reconstruct", params={"tilt_systems": ["a0a0a0", "a0a0c-"]}
        )
        self.assertIn("generation_parameters", fit["structure"])
        parsed = protocol.parse_job_request(fit)
        self.assertIsNotNone(parsed["structure"].generation_parameters)
        self.assertFalse(parsed["structure"].geometry_matches_generation)
        self.assertEqual(parsed["params"]["tilt_systems"], ["a0a0a0", "a0a0c-"])

    def test_a_loaded_structure_cannot_be_reconstructed(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.build_job_request(sample_structure(), kind="reconstruct")

    def test_the_fit_makes_the_round_trip(self):
        from quick_mag.reconstruction import reconstruction_from_payload

        client = self.fixture.client()
        structure = self.relaxed_structure()
        job = client.submit(
            structure,
            kind="reconstruct",
            params={"tilt_systems": ["a0a0a0", "a0a0c-", "a0a0c+"]},
        )
        finished = pump(client, lambda: not job.is_live, timeout=60.0)
        self.assertIn(job, finished)
        self.assertEqual(job.status, protocol.STATUS_DONE, job.error)
        self.assertIsNone(job.structure)  # nothing to rebuild for a fit
        self.assertIn("a0a0c-", job.status_line())

        reconstruction = reconstruction_from_payload(structure, job.result)
        self.assertEqual(reconstruction.tilt_system, "a0a0c-")
        self.assertAlmostEqual(reconstruction.tilt_angles_deg[2], 6.0, delta=0.3)
        self.assertLess(reconstruction.rmsd, 1e-3)
        self.assertEqual(reconstruction.atom_count, structure.atom_count)
        self.assertEqual(reconstruction.ideal.atomic_labels, structure.atomic_labels)
        self.assertTrue(reconstruction.ideal.geometry_matches_generation)

    def test_a_running_fit_can_be_cancelled(self):
        client = self.fixture.client()
        job = client.submit(self.relaxed_structure(), kind="reconstruct")
        pump(client, lambda: job.status == protocol.STATUS_RUNNING, timeout=30.0)
        client.cancel(job)
        pump(client, lambda: not job.is_live, timeout=60.0)
        self.assertEqual(job.status, protocol.STATUS_CANCELLED)
