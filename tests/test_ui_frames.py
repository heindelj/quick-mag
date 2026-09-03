"""Render the panels for real frames and fail if ImGui refuses anything.

Every other test here calls ``AppState`` methods and never draws. That leaves a
whole class of bug untested: an ImGui assertion. Those are C++ ``IM_ASSERT``s
surfaced as ``RuntimeError``, they fire only when the offending widget is
actually submitted, and -- because the assertions differ between ImGui releases
-- a call can be fine on the machine it was written on and take the app down on
the next one. That is exactly how ``input_int`` with ``enter_returns_true``
shipped: the flag belongs to ``InputText`` alone, and ``InputScalar`` asserts on
it in the versions that check.

ImGui needs no window and no GPU to run a frame -- a context, a display size and
a backend flag saying the renderer handles its own font texture are enough -- so
this costs milliseconds and catches anything the panels submit that ImGui will
not accept.

These are smoke tests. They assert that drawing does not raise; what the panels
say is pinned by the state-level tests next door.
"""

from __future__ import annotations

import pytest

pytest.importorskip("imgui_bundle")

from imgui_bundle import imgui, implot, implot3d  # noqa: E402

from quick_mag.quick_mag_ui import (  # noqa: E402
    AppState,
    gui_atoms,
    gui_calculation_output,
)
import quick_mag.quick_mag_ui as quick_mag_ui  # noqa: E402


@pytest.fixture
def frames():
    """A live ImGui context, and a ``draw(fn)`` that runs one frame through it."""
    context = imgui.create_context()
    plot_context = implot.create_context()
    plot3d_context = implot3d.create_context()
    io = imgui.get_io()
    io.display_size = imgui.ImVec2(1600.0, 1000.0)
    io.delta_time = 1.0 / 60.0
    # Without this ImGui asserts that the font atlas was never built, which is
    # the renderer backend's job and there is no renderer here.
    io.backend_flags |= imgui.BackendFlags_.renderer_has_textures

    def draw(render) -> None:
        imgui.new_frame()
        imgui.begin("test")
        try:
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
def solved_state():
    """An ``AppState`` with results, pointed at its own module-level singleton.

    ``gui_calculation_output`` reads ``APP_STATE`` rather than taking an argument,
    so the panel under test has to be looking at the state under test.
    """
    state = AppState()
    state.sync_builder_binding()
    state.regenerate_focus_from_builder_if_changed()
    state.run_magnetic_structure_calculation(structure=state.focus)
    previous = quick_mag_ui.APP_STATE
    quick_mag_ui.APP_STATE = state
    try:
        yield state
    finally:
        quick_mag_ui.APP_STATE = previous


def test_the_results_panel_draws(frames, solved_state):
    frames(gui_calculation_output)


def test_the_atoms_panel_draws_with_every_atom_listed(frames, solved_state):
    frames(gui_atoms)


def test_the_atoms_panel_draws_a_selection_and_the_slab(frames, solved_state):
    # The slab controls, a selection narrowing the list, and the 3D view with
    # the slab faces, arrow and selection rings all submit.
    solved_state.slab_enabled = True
    solved_state.selection_slab.direction = (1, 1, 0)
    solved_state.selection_slab.thickness = 1.5
    rows = solved_state.atom_table()
    solved_state.toggle_atom_selection(rows[0].ref)
    frames(gui_atoms)
    assert solved_state.selected_rows()
    frames(quick_mag_ui.gui_structure_view)


def test_the_atoms_panel_draws_with_hand_set_states_on_it(frames, solved_state):
    # Edited rows take a style colour and a revert button, and the header swaps
    # the model energy for the edit count and a button. None of that is
    # exercised by the unedited pass above.
    structure = solved_state.magnetic_analysis_structure
    iron = structure.element_symbols().index("Fe")
    solved_state.set_site_oxidation_state(iron, 4)
    solved_state.selected_site_index = iron
    frames(gui_atoms)

    # ...and with the net charge left unbalanced, which is its own coloured row.
    solved_state.set_site_oxidation_state(iron, 2)
    frames(gui_atoms)


def test_the_atoms_panel_draws_vacancies_substitutions_and_protons(frames, solved_state):
    rows = solved_state.atom_table()
    oxygen = next(row for row in rows if row.element == "O")
    iron = next(row for row in rows if row.element == "Fe")
    lanthanum = next(row for row in rows if row.element == "La")
    solved_state.set_atom_element(iron, "Co")
    solved_state.set_atom_element(lanthanum, "")
    solved_state.add_proton_to_atom(oxygen)
    solved_state.regenerate_focus_from_builder_if_changed()
    assert "Co" in solved_state.focus.atomic_labels
    assert "H" in solved_state.focus.atomic_labels
    vacant = [row for row in solved_state.atom_table() if row.vacant]
    assert len(vacant) == 1
    solved_state.toggle_atom_selection(vacant[0].ref)
    frames(gui_atoms)
    frames(quick_mag_ui.gui_structure_view)


def test_the_atoms_panel_draws_for_a_loaded_structure(frames):
    import numpy as np

    from quick_mag.structure import ChemicalStructure

    state = AppState()
    structure = ChemicalStructure(
        name="loaded",
        lattice=np.eye(3) * 4.0,
        cartesian_coords=np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [2.0, 2.0, 0.0]]),
        atomic_labels=["Sr", "Ti", "O"],
        magnetic_moments=np.zeros((3, 3)),
    )
    state.structures.append(structure)
    state.set_focus(structure)
    state.sync_active_structure()
    rows = state.atom_table()
    state.set_atom_element(rows[2], "")
    assert structure.atom_count == 2
    previous = quick_mag_ui.APP_STATE
    quick_mag_ui.APP_STATE = state
    try:
        frames(gui_atoms)
        state.slab_enabled = True
        frames(quick_mag_ui.gui_structure_view)
    finally:
        quick_mag_ui.APP_STATE = previous


def test_the_panel_draws_before_anything_has_been_run(frames):
    state = AppState()
    previous = quick_mag_ui.APP_STATE
    quick_mag_ui.APP_STATE = state
    try:
        frames(gui_calculation_output)
    finally:
        quick_mag_ui.APP_STATE = previous


def test_the_builder_panel_draws_with_a_stack_of_domains(frames):
    """The domain row, per-axis periodicity and the interface toggle all submit."""
    state = AppState()
    state.sync_builder_binding()
    state.regenerate_focus_from_builder_if_changed()
    assert state.add_domain(), state.domain_message
    state.periodic_axes_flags = (True, True, False)
    previous = quick_mag_ui.APP_STATE
    quick_mag_ui.APP_STATE = state
    try:
        frames(quick_mag_ui.gui_controls)
        frames(quick_mag_ui.gui_controls)
        assert state.focus.generation_parameters.is_multi_domain()
        assert state.focus.periodic_axes == (True, True, False)
        frames(quick_mag_ui.gui_structure_view)
    finally:
        quick_mag_ui.APP_STATE = previous
