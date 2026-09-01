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


def relaxed_from(app: AppState, scale: float = 0.97) -> "object":
    """A finished job whose result shrank the cell and nudged every atom."""
    template = app.focus
    result = fake_result(app, shift=0.05)
    result["final_lattice"] = (np.asarray(template.lattice) * scale).tolist()
    job = job_with(app, protocol.STATUS_DONE, result=result)
    job.structure = protocol.structure_from_result(template, result)
    return job


class TestRelaxedStructureSurvivesTheFrameLoop:
    """The bug this class exists for.

    A relaxed structure keeps its ``generation_parameters`` so site indexing
    still works. Everything in the builder half of the UI used to read that as
    "the builder owns this structure", which meant the 3D view rebuilt the ideal
    geometry from those parameters and drew it instead of the relaxation, and the
    builder fields showed the pre-relaxation numbers. The symptom was a finished
    optimization whose lattice constants were exactly the defaults.
    """

    def test_the_relaxed_lattice_is_not_the_template_lattice(self, state):
        template_lattice = np.array(state.focus.lattice)
        job = relaxed_from(state)
        state.collect_remote_job(job)
        arrived = state.structures[-1]
        assert not np.allclose(arrived.lattice, template_lattice)

    def test_many_frames_do_not_regenerate_over_it(self, frames, state):
        job = relaxed_from(state)
        state.collect_remote_job(job)
        arrived = state.focus
        lattice = np.array(arrived.lattice)
        coords = np.array(arrived.cartesian_coords)

        # The builder binds, baselines and checks for edits across frames, so one
        # frame proves nothing; the overwrite used to land on a later one.
        for _ in range(6):
            frames(quick_mag_ui.gui_controls)

        np.testing.assert_allclose(state.focus.lattice, lattice)
        np.testing.assert_allclose(state.focus.cartesian_coords, coords)

    def test_the_3d_view_draws_the_relaxed_geometry(self, state):
        # rendered_structure() rebuilds from the generation parameters when the
        # builder owns the focus. For a relaxed structure that rebuild *is* the
        # bug: it returns the geometry the relaxation started from.
        job = relaxed_from(state)
        state.collect_remote_job(job)
        rendered = state.rendered_structure()
        assert rendered is state.focus
        np.testing.assert_allclose(rendered.lattice, state.focus.lattice)

    def test_the_builder_lets_go_and_the_cell_editor_takes_over(self, state):
        assert state.focus_has_generated_provenance()
        assert not state.cell_editing_available()

        state.collect_remote_job(relaxed_from(state))

        assert state.focus_has_generation_parameters()   # topology kept
        assert not state.focus_has_generated_provenance()  # geometry not
        assert state.focus_is_relaxed_from_builder()
        assert state.cell_editing_available()
        assert not state.is_builder_active()
        assert not state.tilt_editing_available()
        assert not state.defect_editing_available()
        assert "relaxed" in state.unavailable_reason("tilt")

    def test_the_cell_editor_shows_the_relaxed_lattice_constants(self, frames, state):
        job = relaxed_from(state)
        state.collect_remote_job(job)
        frames(quick_mag_ui.gui_controls)
        expected = np.linalg.norm(np.asarray(state.focus.lattice), axis=1)
        assert state.cell_a == pytest.approx(expected[0], abs=1e-6)
        assert state.cell_b == pytest.approx(expected[1], abs=1e-6)
        assert state.cell_c == pytest.approx(expected[2], abs=1e-6)

    def test_the_drift_is_recorded_against_the_submitted_structure(self, state):
        template = state.focus
        job = relaxed_from(state)
        state.collect_remote_job(job)
        drift = state.relaxation_drift_for(state.structures[-1])

        assert drift is not None
        assert drift.cell_changed
        assert drift.atoms_moved
        assert drift.volume_ratio == pytest.approx(0.97**3, rel=1e-9)
        np.testing.assert_allclose(
            drift.lengths_before, np.linalg.norm(np.asarray(template.lattice), axis=1)
        )

    def test_the_drift_panel_draws(self, frames, state):
        state.collect_remote_job(relaxed_from(state))
        attach_client(state)
        frames(lambda: gui_remote_compute(state))


class TestElapsedTimer:
    def test_it_stops_when_the_job_stops(self, state):
        import time

        job = job_with(state, protocol.STATUS_RUNNING)
        job.submitted_at = time.time() - 5.0
        running = job.elapsed()
        assert running >= 5.0

        # Terminal status arrives; from here the readout must hold still.
        job._inflight = False
        client = attach_client(state)
        client._by_key[job.key] = job
        client.jobs = [job]
        client._apply_payload(job, {"status": protocol.STATUS_DONE})

        settled = job.elapsed()
        time.sleep(0.05)
        assert job.elapsed() == settled

    def test_a_rejected_submission_also_stops_the_clock(self, state):
        import time

        from quick_mag.remote.client import Response

        client = attach_client(state)
        state.submit_remote_job()
        job = client.jobs[0]
        request_id = next(iter(client._inflight))
        client._handle(Response(request_id, False, 401, "", "The server rejected the token."))

        assert job.finished_at is not None
        settled = job.elapsed()
        time.sleep(0.05)
        assert job.elapsed() == settled


class TestRelaxationPane:
    """The trace moved out of Calculate and into the 2D pane."""

    def test_submitting_brings_the_pane_up(self, state):
        attach_client(state)
        state.submit_remote_job()
        assert state.two_d_plot_index == quick_mag_ui.TWO_D_PLOT_RELAXATION

    def test_the_pane_draws_with_nothing_to_show(self, frames, state):
        state.two_d_plot_index = quick_mag_ui.TWO_D_PLOT_RELAXATION
        frames(lambda: quick_mag_ui.gui_two_d_pane(state))

    def test_the_pane_draws_for_every_job_status(self, frames, state):
        client = attach_client(state)
        client.jobs = [
            job_with(state, protocol.STATUS_QUEUED),
            job_with(
                state,
                protocol.STATUS_RUNNING,
                progress={
                    "step": 3,
                    "energy": -411.4,
                    "max_force": 0.09,
                    "trajectory_energies": [-410.0, -410.9, -411.2, -411.4],
                },
            ),
            job_with(state, protocol.STATUS_DONE, result=fake_result(state)),
            job_with(state, protocol.STATUS_ERROR, error="CUDA out of memory"),
        ]
        for job in client.jobs:
            client._by_key[job.key] = job
        state.two_d_plot_index = quick_mag_ui.TWO_D_PLOT_RELAXATION
        for job in client.jobs:
            state.remote_selected_job_key = job.key
            frames(lambda: quick_mag_ui.gui_two_d_pane(state))

    def test_the_selection_follows_the_live_job_until_one_is_pinned(self, state):
        client = attach_client(state)
        done = job_with(state, protocol.STATUS_DONE, result=fake_result(state))
        running = job_with(state, protocol.STATUS_RUNNING)
        client.jobs = [done, running]
        for job in client.jobs:
            client._by_key[job.key] = job

        assert state.selected_remote_job() is running
        state.remote_selected_job_key = done.key
        assert state.selected_remote_job() is done
