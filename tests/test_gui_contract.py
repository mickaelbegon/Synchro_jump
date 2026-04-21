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


class _FakeLabel:
    """Minimal label stub exposing one Tk-like mapping contract."""

    def __init__(self) -> None:
        self.mapped = False
        self.pack_kwargs = None

    def winfo_ismapped(self) -> bool:
        """Return whether the widget is currently shown."""

        return self.mapped

    def pack(self, **kwargs) -> None:
        """Store one pack call and mark the widget as visible."""

        self.pack_kwargs = kwargs
        self.mapped = True

    def pack_forget(self) -> None:
        """Hide the widget."""

        self.mapped = False


class _FakeStringVar:
    """Minimal string-variable stub for GUI contract tests."""

    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        """Store the requested string value."""

        self.value = value


class _FakeRoot:
    """Minimal root stub exposing one `update_idletasks` method."""

    def __init__(self) -> None:
        self.updated = False

    def update_idletasks(self) -> None:
        """Record that idle tasks were requested."""

        self.updated = True


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


def test_gui_busy_indicator_can_be_shown_and_hidden() -> None:
    """The busy icon should appear during work and disappear afterwards."""

    app = object.__new__(SynchroJumpApp)
    app.busy_var = _FakeStringVar()
    app.busy_label = _FakeLabel()
    app.animation_scale = object()
    app.root = _FakeRoot()

    app._show_busy_indicator("Construction de l'OCP en cours")
    assert app.busy_var.value.startswith("⌛ ")
    assert app.busy_label.mapped is True
    assert app.root.updated is True

    app._hide_busy_indicator()
    assert app.busy_var.value == ""
    assert app.busy_label.mapped is False
