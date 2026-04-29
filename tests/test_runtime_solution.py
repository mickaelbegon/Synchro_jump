"""Tests for runtime OCP solve summaries."""

from __future__ import annotations

from pathlib import Path
import importlib.util

import numpy as np
import pytest

from synchro_jump.optimization.problem import CONTACT_MODEL_COMPLIANT_UNILATERAL, VerticalJumpOcpSettings
from synchro_jump.optimization.runtime_solution import (
    _add_platform_force_to_numeric_generalized_force,
    _configure_ipopt_solver,
    _print_terminal_objective_breakdown,
    ensure_local_hsl_library,
    solve_ocp_runtime_summary,
    summarize_solved_ocp,
)


class _FakeSolution:
    """Minimal solution-like object for summary extraction tests."""

    def __init__(self) -> None:
        self.cost = -1.25
        self.status = 1
        self.real_time_to_optimize = 0.42
        self._time = np.array([[0.0], [0.3], [0.6]])
        self._states = {
            "q_roots": np.array([[0.0, 0.1, 0.2], [0.0, 0.2, 0.5], [0.0, 0.0, 0.1]]),
            "q_joints": np.array([[-1.7, -1.4, -1.0], [1.7, 1.4, 1.1]]),
            "qdot_roots": np.array([[0.0, 0.1, 0.2], [0.0, 0.4, 1.2], [0.0, 0.1, 0.1]]),
            "qdot_joints": np.array([[0.0, 0.3, 0.5], [0.0, -0.1, -0.2]]),
            "platform_position": np.array([[0.0, 0.03, 0.08]]),
            "platform_velocity": np.array([[0.0, 0.15, 0.20]]),
        }
        self._controls = {
            "tau_joints": np.array([[10.0, 12.0, np.nan], [6.0, 5.0, np.nan]]),
        }

    def decision_states(self, *, to_merge):
        """Return the stored state trajectories."""

        _ = to_merge
        return self._states

    def stepwise_controls(self, *, to_merge):
        """Return the stored control trajectories."""

        _ = to_merge
        return self._controls

    def decision_time(self, *, to_merge):
        """Return the stored decision times."""

        _ = to_merge
        return self._time


def test_summarize_solved_ocp_extracts_runtime_metrics(tmp_path: Path) -> None:
    """A solved OCP summary should expose trajectory and jump metrics."""

    model_path = tmp_path / "jumper.bioMod"
    model_path.write_text("version 4\n", encoding="utf-8")

    def fake_com_evaluator(_model_path: str | Path, state_trajectories: dict[str, np.ndarray]):
        assert "platform_position" in state_trajectories
        return np.array([0.92, 0.99, 1.08]), np.array([0.0, 0.6, 1.4])

    def fake_contact_evaluator(
        _model_path: str | Path,
        state_trajectories: dict[str, np.ndarray],
        control_trajectories: dict[str, np.ndarray],
        **kwargs,
    ):
        assert "q_roots" in state_trajectories
        assert "tau_joints" in control_trajectories
        assert kwargs["peak_force_newtons"] == 1100.0
        return (
            np.array([1100.0, 1100.0, 1050.0]),
            np.array([450.0, 180.0, 0.0]),
            np.array([0.2, 0.1, -0.3]),
        )

    summary = summarize_solved_ocp(
        _FakeSolution(),
        model_path=model_path,
        requested_iterations=5,
        n_phases=1,
        merge_nodes_token="nodes",
        peak_force_newtons=1100.0,
        platform_mass_kg=80.0,
        total_duration_s=2.0,
        com_evaluator=fake_com_evaluator,
        contact_force_evaluator=fake_contact_evaluator,
        external_force_ap_evaluator=lambda *_args, **_kwargs: np.array([10.0, 15.0, 20.0]),
    )

    assert summary.success
    assert summary.requested_iterations == 5
    assert summary.n_phases == 1
    assert summary.solver_status == 1
    assert summary.objective_value == -1.25
    assert summary.solve_time_s == 0.42
    assert summary.final_time_s == 0.6
    assert summary.takeoff_com_height_m == 1.08
    assert summary.takeoff_com_vertical_velocity_m_s == 1.4
    assert summary.predicted_apex_height_m > summary.takeoff_com_height_m
    assert summary.final_contact_force_n == 0.0
    assert summary.takeoff_condition_satisfied is True
    assert summary.contact_model == "rigid_unilateral"
    assert summary.state_trajectories["q_roots"].shape == (3, 3)
    assert summary.control_trajectories["tau_joints"].shape == (2, 3)
    assert summary.com_height_trajectory_m.shape == (3,)
    assert np.allclose(summary.contact_force_trajectory_n, [450.0, 180.0, 0.0])
    assert np.allclose(summary.external_force_ap_trajectory_n, [10.0, 15.0, 20.0])
    assert np.allclose(summary.platform_force_trajectory_n, [1100.0, 1100.0, 1050.0])


def test_summarize_solved_ocp_passes_selected_contact_model_to_evaluator(tmp_path: Path) -> None:
    """The runtime summary should forward the selected contact model."""

    model_path = tmp_path / "jumper.bioMod"
    model_path.write_text("version 4\n", encoding="utf-8")

    def fake_com_evaluator(_model_path: str | Path, _state_trajectories: dict[str, np.ndarray]):
        return np.array([0.9, 1.0]), np.array([0.0, 1.0])

    def fake_contact_evaluator(
        _model_path: str | Path,
        _state_trajectories: dict[str, np.ndarray],
        _control_trajectories: dict[str, np.ndarray],
        **kwargs,
    ):
        assert kwargs["contact_model"] == CONTACT_MODEL_COMPLIANT_UNILATERAL
        return np.array([1100.0, 1050.0]), np.array([120.0, 0.0]), np.array([0.0, 0.0])

    summary = summarize_solved_ocp(
        _FakeSolution(),
        model_path=model_path,
        requested_iterations=5,
        n_phases=1,
        merge_nodes_token="nodes",
        peak_force_newtons=1100.0,
        platform_mass_kg=80.0,
        contact_model=CONTACT_MODEL_COMPLIANT_UNILATERAL,
        total_duration_s=2.0,
        com_evaluator=fake_com_evaluator,
        contact_force_evaluator=fake_contact_evaluator,
    )

    assert summary.contact_model == CONTACT_MODEL_COMPLIANT_UNILATERAL


def test_solve_ocp_runtime_summary_reports_missing_optional_dependency(tmp_path: Path) -> None:
    """The solve wrapper should fail gracefully without `bioptim`."""

    if importlib.util.find_spec("bioptim") is not None:
        pytest.skip("`bioptim` is installed in this environment")

    summary = solve_ocp_runtime_summary(
        VerticalJumpOcpSettings(athlete_mass_kg=50.0),
        1100.0,
        model_output_dir=tmp_path,
        maximum_iterations=3,
    )

    assert not summary.success
    assert "Dependance optionnelle manquante" in summary.message
    assert summary.requested_iterations == 3


def test_configure_ipopt_solver_enables_iteration_and_timing_logs() -> None:
    """The IPOPT helper should request verbose iteration and timing output."""

    class _FakeSolver:
        def __init__(self) -> None:
            self.maximum_iterations = None
            self.print_level = None
            self.linear_solver = None
            self.options = {}

        def set_maximum_iterations(self, value: int) -> None:
            self.maximum_iterations = value

        def set_print_level(self, value: int) -> None:
            self.print_level = value

        def set_linear_solver(self, value: str) -> None:
            self.linear_solver = value

        def set_option_unsafe(self, value, name: str) -> None:
            self.options[name] = value

    solver = _FakeSolver()

    _configure_ipopt_solver(
        solver,
        maximum_iterations=1000,
        print_level=5,
        linear_solver="ma57",
        hsl_library_path="/tmp/libhsl.dylib",
    )

    assert solver.maximum_iterations == 1000
    assert solver.print_level == 5
    assert solver.linear_solver == "ma57"
    assert solver.options["hsllib"] == "/tmp/libhsl.dylib"
    assert solver.options["print_timing_statistics"] == "yes"
    assert solver.options["print_frequency_iter"] == 1
    assert solver.options["print_frequency_time"] == 0


def test_print_terminal_objective_breakdown_reports_split_jump_terms(capsys) -> None:
    """The terminal breakdown should dissociate CoM height and ballistic gain."""

    summary = summarize_solved_ocp(
        _FakeSolution(),
        model_path=Path("dummy.bioMod"),
        requested_iterations=5,
        n_phases=1,
        merge_nodes_token="nodes",
        peak_force_newtons=1100.0,
        platform_mass_kg=80.0,
        total_duration_s=2.0,
        com_evaluator=lambda *_args, **_kwargs: (np.array([0.9, 1.0, 1.08]), np.array([0.0, 0.6, 1.4])),
        contact_force_evaluator=lambda *_args, **_kwargs: (
            np.array([1100.0, 1100.0, 1050.0]),
            np.array([450.0, 180.0, 0.0]),
            np.array([0.2, 0.1, -0.3]),
        ),
    )

    _print_terminal_objective_breakdown(summary)
    captured = capsys.readouterr().out

    assert "-z_CoM(T)" in captured
    assert "-max(vz_CoM(T), 0)^2 / (2g)" in captured


def test_ensure_local_hsl_library_copies_one_candidate(tmp_path: Path) -> None:
    """The HSL helper should copy one discovered library into the project-local folder."""

    source_library = tmp_path / "source" / "libhsl.dylib"
    source_library.parent.mkdir(parents=True)
    source_library.write_bytes(b"fake hsl")

    copied_library = ensure_local_hsl_library(
        local_dir=tmp_path / "local",
        candidate_paths=(source_library,),
    )

    assert copied_library == tmp_path / "local" / "libhsl.dylib"
    assert copied_library.read_bytes() == b"fake hsl"


def test_platform_force_is_added_to_vertical_root_translation() -> None:
    """The platform actuation should feed the vertical root generalized force."""

    generalized_force = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    updated_force = _add_platform_force_to_numeric_generalized_force(generalized_force, 1100.0)

    assert np.allclose(updated_force, [1.0, 1102.0, 3.0, 4.0, 5.0])


def test_summarize_solved_ocp_can_use_no_platform_equivalent_contact(tmp_path: Path) -> None:
    """The runtime summary should accept an equivalent contact reconstructed from CoM acceleration."""

    model_path = tmp_path / "jumper.bioMod"
    model_path.write_text("version 4\n", encoding="utf-8")

    def fake_com_evaluator(_model_path: str | Path, _state_trajectories: dict[str, np.ndarray]):
        return np.array([0.9, 1.0, 1.1]), np.array([0.0, 0.4, 0.8])

    def fake_contact_evaluator(
        _model_path: str | Path,
        _state_trajectories: dict[str, np.ndarray],
        _control_trajectories: dict[str, np.ndarray],
        **kwargs,
    ):
        assert kwargs["contact_model"] == "no_platform"
        assert kwargs["athlete_mass_kg"] == 50.0
        return np.zeros(3), np.array([0.0, 25.0, 50.0]), np.array([0.0, 0.5, 1.0])

    summary = summarize_solved_ocp(
        _FakeSolution(),
        model_path=model_path,
        requested_iterations=5,
        n_phases=1,
        merge_nodes_token="nodes",
        peak_force_newtons=1100.0,
        platform_mass_kg=80.0,
        athlete_mass_kg=50.0,
        contact_model="no_platform",
        total_duration_s=2.0,
        com_evaluator=fake_com_evaluator,
        contact_force_evaluator=fake_contact_evaluator,
    )

    assert summary.contact_model == "no_platform"
    assert np.allclose(summary.contact_force_trajectory_n, [0.0, 25.0, 50.0])
    assert np.allclose(summary.platform_force_trajectory_n, [0.0, 0.0, 0.0])
