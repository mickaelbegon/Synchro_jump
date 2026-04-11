"""Tests for runtime OCP solve summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synchro_jump.optimization.problem import CONTACT_MODEL_COMPLIANT_UNILATERAL, VerticalJumpOcpSettings
from synchro_jump.optimization.runtime_solution import solve_ocp_runtime_summary, summarize_solved_ocp


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

    summary = solve_ocp_runtime_summary(
        VerticalJumpOcpSettings(athlete_mass_kg=50.0),
        1100.0,
        model_output_dir=tmp_path,
        maximum_iterations=3,
    )

    assert not summary.success
    assert "Dependance optionnelle manquante" in summary.message
    assert summary.requested_iterations == 3
