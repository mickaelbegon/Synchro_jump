"""Non-visual tests for the GUI public API."""

from __future__ import annotations

import numpy as np

import synchro_jump.gui.app as app_module
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
    assert "Sans plateforme" in SynchroJumpApp.contact_model_by_label
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


def test_runtime_q_trajectory_supports_joint_only_solutions() -> None:
    """The GUI should animate runtime solutions even when only `q_joints` are present."""

    app = object.__new__(SynchroJumpApp)
    app.runtime_solution = type(
        "_RuntimeSolution",
        (),
        {"state_trajectories": {"q_joints": np.array([[0.0, 1.0], [2.0, 3.0]])}},
    )()

    q_trajectory = app._runtime_q_trajectory()

    assert q_trajectory.shape == (2, 2)


def test_runtime_joint_angle_and_torque_helpers_keep_last_two_dofs() -> None:
    """The GUI should expose knee/hip-like series from the last two generalized coordinates."""

    app = object.__new__(SynchroJumpApp)
    app.runtime_solution = type(
        "_RuntimeSolution",
        (),
        {
            "time": np.array([0.0, 0.5, 1.0]),
            "state_trajectories": {
                "q_joints": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.1, 0.2, 0.3],
                        [0.2, 0.3, 0.4],
                        [0.4, 0.5, 0.6],
                        [0.7, 0.8, 0.9],
                    ]
                )
            },
            "control_trajectories": {
                "tau_joints": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0],
                        [2.0, 2.0, 2.0],
                        [30.0, 31.0, 32.0],
                        [40.0, 41.0, 42.0],
                    ]
                )
            },
        },
    )()

    angle_trajectories = app._runtime_joint_angle_trajectory_deg()
    control_time, torque_trajectories = app._runtime_joint_torque_trajectory_nm()

    assert angle_trajectories["Genou"].shape == (3,)
    assert angle_trajectories["Hanche"].shape == (3,)
    assert control_time.shape == (3,)
    assert np.allclose(torque_trajectories["Genou"], [30.0, 31.0, 32.0])
    assert np.allclose(torque_trajectories["Hanche"], [40.0, 41.0, 42.0])


def test_runtime_com_helpers_return_planar_trajectory_and_velocity() -> None:
    """The GUI should reconstruct a planar CoM path and its finite-difference velocity."""

    app = object.__new__(SynchroJumpApp)
    app.runtime_solution = type(
        "_RuntimeSolution",
        (),
        {
            "time": np.array([0.0, 0.5, 1.0]),
            "state_trajectories": {
                "q_joints": np.array(
                    [
                        [0.0, 0.05, 0.1],
                        [-0.2, -0.1, 0.0],
                        [0.2, 0.1, 0.0],
                    ]
                )
            },
        },
    )()
    app.base_settings = type("_Settings", (), {"athlete_height_m": 1.6})()
    app.current_mass_kg = lambda: 50.0
    app.current_contact_model_key = lambda: "no_platform"

    morphology = app.current_morphology()
    com_trajectory = app._runtime_com_planar_trajectory(morphology)
    com_velocity_trajectory = app._runtime_com_planar_velocity_trajectory(morphology)

    assert com_trajectory.shape == (2, 3)
    assert com_velocity_trajectory.shape == (2, 3)
    assert np.all(np.isfinite(com_trajectory))
    assert np.all(np.isfinite(com_velocity_trajectory))


def test_animation_frame_change_uses_dynamic_only_refresh() -> None:
    """Moving the animation slider should avoid a full GUI refresh."""

    app = object.__new__(SynchroJumpApp)
    recorded_calls = []
    app.refresh = lambda **kwargs: recorded_calls.append(kwargs)

    app._on_animation_frame_change("2")

    assert recorded_calls == [{"dynamic_only": True}]


def test_avatar_status_line_reports_unavailable_reason(monkeypatch) -> None:
    """The GUI should expose a readable raster-avatar diagnostic line."""

    app = object.__new__(SynchroJumpApp)
    monkeypatch.setattr(
        app_module,
        "avatar_rendering_diagnostics",
        lambda: (False, "Pillow n'est pas installe, retour au stick figure."),
    )

    assert app._avatar_status_line().startswith("indisponible, Pillow")


def test_draw_raster_avatar_forwards_alpha_to_segment_renderer(monkeypatch) -> None:
    """The raster avatar helper should forward one shared alpha to all sprites."""

    app = object.__new__(SynchroJumpApp)
    recorded_alpha = []
    recorded_flip = []

    monkeypatch.setattr(app_module, "avatar_rendering_diagnostics", lambda: (True, "ok"))

    def _fake_draw_segment_image(*_args, **kwargs):
        recorded_alpha.append(kwargs["alpha"])
        recorded_flip.append(kwargs["flip_horizontal"])
        return True

    monkeypatch.setattr(app_module, "draw_segment_image", _fake_draw_segment_image)
    app.avatar_flip_horizontal = True

    ok = app._draw_raster_avatar(
        object(),
        {
            "foot": (0.0, 0.0),
            "knee": (0.1, 0.6),
            "hip": (0.2, 1.0),
            "head": (0.25, 1.4),
        },
        alpha=0.25,
    )

    assert ok is True
    assert recorded_alpha == [0.25, 0.25, 0.25]
    assert recorded_flip == [True, True, True]


def test_segment_com_positions_delegate_to_the_display_model() -> None:
    """The GUI helper should expose the segmental CoMs of the displayed model."""

    app = object.__new__(SynchroJumpApp)
    app.current_model_definition = lambda: type(
        "_ModelDefinition",
        (),
        {
            "segment_center_of_mass_positions": lambda self, _q: {
                "leg_foot": (0.0, 0.3),
                "thigh": (0.1, 0.8),
                "trunk": (0.2, 1.2),
            }
        },
    )()

    segment_coms = app._segment_com_positions((0.0, 0.0, 0.0))

    assert segment_coms["leg_foot"] == (0.0, 0.3)
    assert segment_coms["thigh"] == (0.1, 0.8)
    assert segment_coms["trunk"] == (0.2, 1.2)


def test_joint_torque_limits_follow_the_current_mass() -> None:
    """The displayed torque limits should match the OCP mass-dependent bounds."""

    app = object.__new__(SynchroJumpApp)
    app.current_mass_kg = lambda: 50.0

    torque_limits = app._joint_torque_limits_nm()

    assert torque_limits == {"Genou": 750.0, "Hanche": 1000.0}
