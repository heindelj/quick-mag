"""The wire format: what survives a round trip, and what is rejected."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quick_mag.remote import protocol  # noqa: E402
from quick_mag.structure import (  # noqa: E402
    ChemicalStructure,
    generate_random_test_perovskite,
)


def simple_structure(periodic: bool = True) -> ChemicalStructure:
    return ChemicalStructure.with_zero_magnetic_moments(
        name="FeO_test",
        lattice=np.diag([4.0, 4.2, 4.4]),
        cartesian_coords=np.array([[0.0, 0.0, 0.0], [2.0, 2.1, 2.2]]),
        atomic_labels=["Fe3+", "O2-"],
        is_periodic=periodic,
    )


class TestStructureRoundTrip(unittest.TestCase):
    def test_a_loaded_structure_survives_the_wire(self):
        original = simple_structure()
        payload = json.loads(protocol.dumps(protocol.structure_to_payload(original)))
        restored = protocol.structure_from_payload(payload)

        np.testing.assert_allclose(restored.lattice, original.lattice)
        np.testing.assert_allclose(
            restored.cartesian_coords, original.cartesian_coords
        )
        self.assertEqual(restored.atomic_labels, original.atomic_labels)
        self.assertTrue(restored.is_periodic)

    def test_a_generated_perovskite_survives_the_wire(self):
        original, _ = generate_random_test_perovskite(np.random.default_rng(7))
        restored = protocol.structure_from_payload(
            json.loads(protocol.dumps(protocol.structure_to_payload(original)))
        )
        self.assertEqual(restored.atom_count, original.atom_count)
        np.testing.assert_allclose(
            restored.cartesian_coords, original.cartesian_coords
        )

    def test_provenance_never_travels_but_is_recovered_from_the_template(self):
        # The whole reason the payload can stay small: generation_parameters is
        # what site indexing (and therefore every spin feature) depends on, and
        # it comes back from the client's own copy rather than from the server.
        original, _ = generate_random_test_perovskite(np.random.default_rng(3))
        self.assertIsNotNone(original.generation_parameters)
        self.assertNotIn(
            "generation_parameters", protocol.structure_to_payload(original)
        )

        result = {
            "final_lattice": original.lattice.tolist(),
            "final_coords": original.cartesian_coords.tolist(),
        }
        relaxed = protocol.structure_from_result(original, result)
        self.assertIs(relaxed.generation_parameters, original.generation_parameters)
        self.assertEqual(relaxed.atomic_labels, original.atomic_labels)
        self.assertEqual(relaxed.name, f"{original.name}_chgnet")

    def test_a_non_periodic_template_keeps_its_own_lattice(self):
        # The remote side boxes a cluster into a vacuum cell; that box is an
        # artifact of the calculation and must not come back as the structure's
        # lattice.
        template = simple_structure(periodic=False)
        result = {
            "final_lattice": np.diag([100.0, 100.0, 100.0]).tolist(),
            "final_coords": template.cartesian_coords.tolist(),
        }
        relaxed = protocol.structure_from_result(template, result)
        np.testing.assert_allclose(relaxed.lattice, template.lattice)


class TestRequestValidation(unittest.TestCase):
    def test_defaults_are_filled_in(self):
        params = protocol.normalize_params(None)
        self.assertEqual(params["calculation"], "cell+atoms")
        self.assertEqual(params["optimizer"], "LBFGS")

    def test_an_unknown_optimizer_is_refused(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.normalize_params({"optimizer": "NEWTON"})

    def test_a_non_positive_fmax_is_refused(self):
        with self.assertRaises(protocol.ProtocolError):
            protocol.normalize_params({"fmax": 0.0})

    def test_a_cell_relaxation_of_a_cluster_is_refused_before_it_is_queued(self):
        request = protocol.build_job_request(
            simple_structure(periodic=False), params={"calculation": "atoms"}
        )
        request["params"]["calculation"] = "cell+atoms"
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.parse_job_request(request)
        self.assertIn("non-periodic", str(caught.exception))

    def test_max_atoms_is_enforced(self):
        request = protocol.build_job_request(simple_structure())
        with self.assertRaises(protocol.ProtocolError):
            protocol.parse_job_request(request, max_atoms=1)

    def test_a_stale_client_gets_a_sentence_not_a_keyerror(self):
        request = protocol.build_job_request(simple_structure())
        request["protocol"] = protocol.PROTOCOL_VERSION + 1
        with self.assertRaises(protocol.ProtocolError) as caught:
            protocol.parse_job_request(request)
        self.assertIn("Protocol mismatch", str(caught.exception))

    def test_a_mismatched_result_is_refused(self):
        template = simple_structure()
        with self.assertRaises(protocol.ProtocolError):
            protocol.structure_from_result(
                template, {"final_coords": [[0.0, 0.0, 0.0]], "final_lattice": template.lattice.tolist()}
            )


class TestJsonHygiene(unittest.TestCase):
    def test_non_finite_numbers_become_null_rather_than_invalid_json(self):
        # json.dumps emits bare NaN by default, which JSON.parse rejects: one
        # diverged force would otherwise break the browser client outright.
        payload = {"forces": protocol._array_to_json(np.array([[1.0, np.nan, 3.0]]))}
        text = protocol.dumps(payload)
        self.assertIn("null", text)
        self.assertEqual(json.loads(text)["forces"], [[1.0, None, 3.0]])

    def test_dumps_refuses_to_emit_nan(self):
        with self.assertRaises(ValueError):
            protocol.dumps({"energy": float("nan")})


if __name__ == "__main__":
    unittest.main()


class TestBrowserSafety(unittest.TestCase):
    """The modules staged into Pyodide must stay importable there.

    The browser build ships numpy and scipy and nothing else. A stray import of
    ase, chgnet or torch anywhere in the staged set takes the whole web app down
    at startup with an ImportError and no other clue, so it is worth an assertion
    rather than a comment.
    """

    STAGED_MODULES = ("__init__.py", "protocol.py", "client.py")
    HOST_ONLY_MODULES = ("server.py", "executor.py")
    FORBIDDEN = ("ase", "chgnet", "torch", "chgnet_runner")

    def _source(self, name: str) -> str:
        return (SRC / "quick_mag" / "remote" / name).read_text()

    def _imported_names(self, name: str):
        """Every module named by an import in ``name``, at any nesting level.

        Parsed rather than grepped: the modules explain in prose which imports
        they are avoiding, and a line-based check flags its own documentation.
        """
        import ast

        names = set()
        for node in ast.walk(ast.parse(self._source(name))):
            if isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module)
        return names

    def test_the_staged_modules_never_reach_for_the_compute_stack(self):
        for name in self.STAGED_MODULES:
            for imported in self._imported_names(name):
                root = imported.split(".")[0]
                for banned in self.FORBIDDEN:
                    self.assertNotEqual(
                        root, banned,
                        f"{name} imports {imported}; it is staged into the browser.",
                    )
                self.assertNotIn(
                    "chgnet_runner", imported,
                    f"{name} imports {imported}; it is staged into the browser.",
                )

    def test_the_manifest_stages_the_client_and_excludes_the_server(self):
        import json

        manifest = json.loads((SRC.parent / "web" / "manifest.json").read_text())
        destinations = {entry["dest"] for entry in manifest["files"]}
        for name in self.STAGED_MODULES:
            self.assertIn(f"/app/quick_mag/remote/{name}", destinations)
        for name in self.HOST_ONLY_MODULES:
            self.assertNotIn(f"/app/quick_mag/remote/{name}", destinations)

    def test_the_ui_module_only_imports_the_browser_safe_half(self):
        source = (SRC / "quick_mag" / "quick_mag_ui.py").read_text()
        self.assertNotIn("remote.server", source)
        self.assertNotIn("remote.executor", source)


class TestTransportSelection(unittest.TestCase):
    """Which transport a desktop process ends up holding.

    This is a regression test with a specific history: the choice used to be made
    by ``import js`` succeeding, and ``js`` is also an ordinary PyPI package. On a
    desktop environment that happened to have one installed, the browser transport
    was selected for a process with no page, and the first Connect took the whole
    window down with ``module 'js' has no attribute 'window'``.
    """

    def setUp(self):
        from quick_mag.remote import client

        self.client_module = client
        self.addCleanup(sys.modules.pop, "js", None)

    def _install_impostor_js(self) -> None:
        """A module named ``js`` that is not Pyodide's."""
        import types

        sys.modules["js"] = types.ModuleType("js")

    def test_a_bare_import_of_js_does_not_make_this_a_browser(self):
        self._install_impostor_js()
        self.assertFalse(self.client_module.is_browser_runtime())

    def test_the_desktop_gets_the_thread_transport_even_with_a_js_module_present(self):
        self._install_impostor_js()
        transport = self.client_module.default_transport()
        self.assertIsInstance(transport, self.client_module.ThreadTransport)

    def test_the_browser_transport_refuses_to_be_built_off_the_browser(self):
        self._install_impostor_js()
        with self.assertRaises(RuntimeError):
            self.client_module.BrowserTransport()

    def test_a_transport_that_cannot_accept_a_request_reports_instead_of_raising(self):
        # Anything reached from an imgui callback that raises unwinds through the
        # C++ frame loop and closes the window, so the client swallows it and
        # turns it into a failed response.
        class Hostile(self.client_module.Transport):
            def request(self, *args, **kwargs):
                raise RuntimeError("no bridge here")

            def drain(self):
                return []

        client = self.client_module.RemoteClient(transport=Hostile())
        client.check_health()  # must not raise
        client.poll()
        self.assertFalse(client.connected)
        self.assertIn("no bridge here", client.health_error or "")
