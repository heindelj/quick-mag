"""The remote-compute panel: it draws, and a finished job lands correctly.

Same harness as ``test_ui_frames``: a real ImGui context, real frames, no window
and no GPU. What that buys here is the class of bug the state-level tests cannot
see -- a widget ImGui refuses -- across every state the panel can be in, since
most of them (a running job, a failed one, a live energy trace) are otherwise
only reachable with a server on the other end.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("imgui_bundle")

from imgui_bundle import imgui, implot, implot3d  # noqa: E402

import quick_mag.quick_mag_ui as quick_mag_ui  # noqa: E402
from quick_mag.quick_mag_ui import AppState, gui_calculate, gui_remote_compute  # noqa: E402
from quick_mag.remote import protocol  # noqa: E402
from quick_mag.remote.client import RemoteClient, RemoteJob, Transport  # noqa: E402


class SilentTransport(Transport):
    """Accepts requests and never answers, so no test touches the network."""

    def __init__(self):
        self.sent = []

    def request(self, request_id, method, url, headers, body):
        self.sent.append((method, url))

    def drain(self):
        return []


@pytest.fixture
def frames():
    context = imgui.create_context()
    plot_context = implot.create_context()
    plot3d_context = implot3d.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1600.0, 1000.0)
    io.delta_time = 1.0 / 60.0
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures

    def draw(render) -> None:
        imgui.new_frame()
        imgui.begin("test")
        try:
            # The panel lives behind a collapsing header, and a closed header
            # draws none of the widgets this test exists to exercise.
            imgui.set_next_item_open(True, imgui.Cond_.always)
            render()
        finally:
            imgui.end()
            imgui.end_frame()

    try:
        yield draw
    finally:
        implot3d.destroy_context(plot3d_context)
        implot.destroy_context(plot_context)
        imgui.destroy_context(context)


@pytest.fixture
def state():
    app = AppState()
    previous = quick_mag_ui.APP_STATE
    quick_mag_ui.APP_STATE = app
    try:
        yield app
    finally:
        quick_mag_ui.APP_STATE = previous


def attach_client(app: AppState) -> RemoteClient:
    client = RemoteClient(app.remote_url, app.remote_token, transport=SilentTransport())
    app._remote_client = client
    return client


def job_with(app: AppState, status: str, **kwargs) -> RemoteJob:
    job = RemoteJob(
        key=f"k{status}",
        label=f"{status}-job",
        template=app.focus,
        params=app.remote_params(),
        id=f"srv-{status}",
        status=status,
        **kwargs,
    )
    return job


def fake_result(app: AppState, shift: float = 0.1) -> dict:
    structure = app.focus
    coords = np.asarray(structure.cartesian_coords) + shift
    return {
        "energy": -412.5,
        "energy_per_atom": -412.5 / structure.atom_count,
        "forces": np.zeros_like(coords).tolist(),
        "stress": [0.0] * 6,
        "magnetic_moments": [3.5] * structure.atom_count,
        "trajectory_energies": [-410.0, -411.4, -412.5],
        "final_lattice": np.asarray(structure.lattice).tolist(),
        "final_coords": coords.tolist(),
        "steps": 2,
        "converged": True,
    }


class TestPanelDraws:
    def test_it_draws_before_anything_is_connected(self, frames, state):
        frames(lambda: gui_remote_compute(state))

    def test_it_draws_inside_the_calculate_panel(self, frames, state):
        frames(gui_calculate)

    def test_it_draws_when_connected(self, frames, state):
        client = attach_client(state)
        client.health = {
            "protocol": protocol.PROTOCOL_VERSION,
            "device": "cuda",
            "cuda_device": "NVIDIA A100",
            "queue_depth": 2,
        }
        client.connected = True
        frames(lambda: gui_remote_compute(state))

    def test_it_draws_a_connection_failure(self, frames, state):
        client = attach_client(state)
        client.health_error = "Could not reach the server."
        frames(lambda: gui_remote_compute(state))

    def test_it_draws_every_job_status(self, frames, state):
        client = attach_client(state)
        client.jobs = [
            job_with(state, protocol.STATUS_QUEUED),
            job_with(
                state,
                protocol.STATUS_RUNNING,
                progress={
                    "step": 12,
                    "energy": -411.2,
                    "max_force": 0.08,
                    "trajectory_energies": [-410.0, -410.9, -411.2],
                },
            ),
            job_with(state, protocol.STATUS_DONE, result=fake_result(state)),
            job_with(state, protocol.STATUS_ERROR, error="the model is missing"),
            job_with(state, protocol.STATUS_CANCELLED),
        ]
        for job in client.jobs:
            client._by_key[job.key] = job
        # Once with the default selection, once with each job explicitly chosen,
        # since the energy trace is drawn for whichever one is selected.
        frames(lambda: gui_remote_compute(state))
        for job in client.jobs:
            state.remote_selected_job_key = job.key
            frames(lambda: gui_remote_compute(state))

    def test_it_draws_the_single_point_mode_with_the_optimizer_disabled(self, frames, state):
        state.remote_calculation_index = quick_mag_ui.REMOTE_CALCULATIONS.index(
            "single-point"
        )
        frames(lambda: gui_remote_compute(state))

    def test_it_draws_with_chgnet_diagnostics_on_the_focused_structure(self, frames, state):
        state.chgnet_moments[id(state.focus)] = np.full(state.focus.atom_count, 3.2)
        frames(lambda: gui_remote_compute(state))


class TestCollection:
    def test_a_finished_job_joins_the_structure_list_and_takes_focus(self, state):
        before = len(state.structures)
        template = state.focus
        job = job_with(state, protocol.STATUS_DONE, result=fake_result(state))
        job.structure = protocol.structure_from_result(template, job.result)

        state.collect_remote_job(job)

        assert len(state.structures) == before + 1
        arrived = state.structures[-1]
        assert state.focus is arrived
        np.testing.assert_allclose(
            arrived.cartesian_coords, np.asarray(template.cartesian_coords) + 0.1
        )

    def test_chgnet_magnitudes_are_kept_apart_from_the_signed_moments(self, state):
        # CHGNet returns unsigned |m|. Writing those into the structure's own
        # moments would look like a spin configuration to everything downstream.
        template = state.focus
        job = job_with(state, protocol.STATUS_DONE, result=fake_result(state))
        job.structure = protocol.structure_from_result(template, job.result)
        state.collect_remote_job(job)

        arrived = state.structures[-1]
        np.testing.assert_allclose(arrived.magnetic_moments, 0.0)
        diagnostics = state.chgnet_moments_for(arrived)
        assert diagnostics is not None
        np.testing.assert_allclose(diagnostics, 3.5)

    def test_a_name_collision_is_resolved_rather_than_shadowing(self, state):
        template = state.focus
        for _ in range(2):
            job = job_with(state, protocol.STATUS_DONE, result=fake_result(state))
            job.structure = protocol.structure_from_result(template, job.result)
            state.collect_remote_job(job)
        names = [item.name for item in state.structures]
        assert len(names) == len(set(names))

    def test_a_failed_job_adds_nothing_and_says_why(self, state):
        before = len(state.structures)
        job = job_with(state, protocol.STATUS_ERROR, error="CUDA out of memory")
        state.collect_remote_job(job)
        assert len(state.structures) == before
        assert "CUDA out of memory" in state.remote_message

    def test_focus_stays_put_when_asked_to(self, state):
        state.remote_focus_on_arrival = False
        template = state.focus
        job = job_with(state, protocol.STATUS_DONE, result=fake_result(state))
        job.structure = protocol.structure_from_result(template, job.result)
        state.collect_remote_job(job)
        assert state.focus is template


class TestSubmission:
    def test_an_impossible_combination_is_refused_without_a_round_trip(self, state):
        client = attach_client(state)
        state.focus.is_periodic = False
        state.remote_calculation_index = quick_mag_ui.REMOTE_CALCULATIONS.index(
            "cell+atoms"
        )
        state.submit_remote_job()

        assert client.transport.sent == []
        assert "non-periodic" in state.remote_message

    def test_a_valid_submission_reaches_the_transport(self, state):
        client = attach_client(state)
        state.submit_remote_job()
        assert client.transport.sent == [("POST", f"{state.remote_url}/jobs")]
        assert len(client.jobs) == 1
