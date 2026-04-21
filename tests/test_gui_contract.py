"""Non-visual tests for the GUI public API."""

from __future__ import annotations

from synchro_jump.gui.app import SynchroJumpApp


class _FakeButton:
    """Minimal button stub exposing one Tk-like configure/state contract."""

    def __init__(self) -> None:
        self.state = None

    def configure(self, *, state) -> None:
        """Store the requested widget state."""

        self.state = state


def test_gui_exposes_runtime_actions() -> None:
    """The GUI class should expose runtime build and solve actions."""

    assert callable(getattr(SynchroJumpApp, "build_ocp"))
    assert callable(getattr(SynchroJumpApp, "solve_ocp"))
    assert callable(getattr(SynchroJumpApp, "toggle_animation"))
    assert callable(getattr(SynchroJumpApp, "reset_animation"))
    assert "Rigide unilateral" in SynchroJumpApp.contact_model_by_label
    assert "Compliant unilateral" in SynchroJumpApp.contact_model_by_label
    assert SynchroJumpApp.default_solve_iterations > 0
    assert SynchroJumpApp.max_solve_iterations >= SynchroJumpApp.default_solve_iterations
    assert SynchroJumpApp.animation_delay_ms > 0


def test_gui_button_states_follow_ocp_build_state() -> None:
    """The solve button should stay disabled until one successful OCP build."""

    app = object.__new__(SynchroJumpApp)
    app.build_button = _FakeButton()
    app.solve_button = _FakeButton()
    app.ocp_is_built = False

    app._update_ocp_button_states()
    assert app.build_button.state == "normal"
    assert app.solve_button.state == "disabled"

    app._set_ocp_built_state(True)
    assert app.build_button.state == "disabled"
    assert app.solve_button.state == "normal"
