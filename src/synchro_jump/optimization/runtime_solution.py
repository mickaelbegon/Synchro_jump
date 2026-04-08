"""Runtime helpers to solve and summarize the `bioptim` OCP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from synchro_jump.optimization.bioptim_ocp import VerticalJumpBioptimOcpBuilder
from synchro_jump.optimization.problem import VerticalJumpOcpSettings


@dataclass(frozen=True)
class OcpSolveSummary:
    """Describe one runtime OCP solve attempt."""

    success: bool
    message: str
    model_path: Path
    state_names: tuple[str, ...] = ()
    control_names: tuple[str, ...] = ()
    n_phases: int = 0
    requested_iterations: int = 0
    solver_status: int | None = None
    objective_value: float | None = None
    solve_time_s: float | None = None
    final_time_s: float | None = None
    takeoff_com_height_m: float | None = None
    takeoff_com_vertical_velocity_m_s: float | None = None
    predicted_apex_height_m: float | None = None
    time: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    state_trajectories: dict[str, np.ndarray] = field(default_factory=dict)
    control_trajectories: dict[str, np.ndarray] = field(default_factory=dict)
    com_height_trajectory_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    com_vertical_velocity_trajectory_m_s: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))


def _merge_split_states(state_trajectories: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Return one full generalized vector assembled from split root and joint arrays."""

    split_roots = state_trajectories.get(f"{prefix}_roots")
    split_joints = state_trajectories.get(f"{prefix}_joints")
    if split_roots is not None and split_joints is not None:
        return np.vstack((split_roots, split_joints))

    merged = state_trajectories.get(prefix)
    if merged is None:
        raise KeyError(f"Missing {prefix} trajectories in solution data")
    return merged


def _solution_scalar(value: Any) -> float | None:
    """Convert one scalar-like runtime value to a Python float."""

    if value is None:
        return None
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return None
    return float(array.reshape((-1,))[0])


def evaluate_com_trajectory(
    model_path: str | Path,
    state_trajectories: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate CoM height and vertical velocity along one state trajectory."""

    from casadi import DM
    from bioptim import BiorbdModel

    model = BiorbdModel(str(Path(model_path)))
    q_trajectory = _merge_split_states(state_trajectories, "q")
    qdot_trajectory = _merge_split_states(state_trajectories, "qdot")

    com_fun = model.center_of_mass()
    com_velocity_fun = model.center_of_mass_velocity()
    heights = np.zeros(q_trajectory.shape[1], dtype=float)
    vertical_velocities = np.zeros(q_trajectory.shape[1], dtype=float)

    for node_index in range(q_trajectory.shape[1]):
        q_column = DM(q_trajectory[:, node_index].reshape((-1, 1)))
        qdot_column = DM(qdot_trajectory[:, node_index].reshape((-1, 1)))
        com = np.asarray(com_fun(q_column, DM()), dtype=float).reshape((-1,))
        com_velocity = np.asarray(com_velocity_fun(q_column, qdot_column, DM()), dtype=float).reshape((-1,))
        heights[node_index] = com[2]
        vertical_velocities[node_index] = com_velocity[2]

    return heights, vertical_velocities


def summarize_solved_ocp(
    solution: Any,
    *,
    model_path: str | Path,
    requested_iterations: int,
    n_phases: int,
    merge_nodes_token: Any,
    gravity: float = 9.81,
    com_evaluator: Callable[[str | Path, dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]] = evaluate_com_trajectory,
) -> OcpSolveSummary:
    """Convert one runtime solution object into one GUI-friendly summary."""

    state_trajectories = {
        key: np.asarray(value, dtype=float)
        for key, value in solution.decision_states(to_merge=merge_nodes_token).items()
    }
    control_trajectories = {
        key: np.asarray(value, dtype=float)
        for key, value in solution.stepwise_controls(to_merge=merge_nodes_token).items()
    }
    time = np.asarray(solution.decision_time(to_merge=merge_nodes_token), dtype=float).reshape((-1,))
    com_height_trajectory_m, com_vertical_velocity_trajectory_m_s = com_evaluator(model_path, state_trajectories)

    final_height = float(com_height_trajectory_m[-1]) if com_height_trajectory_m.size else None
    final_vertical_velocity = (
        float(com_vertical_velocity_trajectory_m_s[-1]) if com_vertical_velocity_trajectory_m_s.size else None
    )
    predicted_apex_height = None
    if final_height is not None and final_vertical_velocity is not None:
        predicted_apex_height = final_height + max(final_vertical_velocity, 0.0) ** 2 / (2.0 * gravity)

    objective_value = _solution_scalar(getattr(solution, "cost", None))
    solver_status = getattr(solution, "status", None)
    message = (
        f"Resolution OCP terminee avec statut solveur {solver_status} "
        f"apres au plus {requested_iterations} iterations."
    )

    return OcpSolveSummary(
        success=True,
        message=message,
        model_path=Path(model_path),
        state_names=tuple(state_trajectories.keys()),
        control_names=tuple(control_trajectories.keys()),
        n_phases=n_phases,
        requested_iterations=requested_iterations,
        solver_status=int(solver_status) if solver_status is not None else None,
        objective_value=objective_value,
        solve_time_s=_solution_scalar(getattr(solution, "real_time_to_optimize", None)),
        final_time_s=float(time[-1]) if time.size else None,
        takeoff_com_height_m=final_height,
        takeoff_com_vertical_velocity_m_s=final_vertical_velocity,
        predicted_apex_height_m=predicted_apex_height,
        time=time,
        state_trajectories=state_trajectories,
        control_trajectories=control_trajectories,
        com_height_trajectory_m=com_height_trajectory_m,
        com_vertical_velocity_trajectory_m_s=com_vertical_velocity_trajectory_m_s,
    )


def solve_ocp_runtime_summary(
    settings: VerticalJumpOcpSettings,
    peak_force_newtons: float,
    *,
    model_output_dir: str | Path = "generated",
    maximum_iterations: int = 5,
    print_level: int = 0,
) -> OcpSolveSummary:
    """Build and solve the runtime OCP, then summarize the result."""

    builder = VerticalJumpBioptimOcpBuilder(settings=settings)
    model_path = builder.export_model(model_output_dir)

    try:
        from bioptim import SolutionMerge, Solver
    except ModuleNotFoundError as exc:
        dependency_name = getattr(exc, "name", None) or str(exc)
        return OcpSolveSummary(
            success=False,
            message=f"Dependance optionnelle manquante pour resoudre l'OCP: {dependency_name}",
            model_path=model_path,
            requested_iterations=maximum_iterations,
        )

    try:
        ocp = builder.build_ocp(peak_force_newtons=peak_force_newtons, model_path=model_path)
        solver = Solver.IPOPT()
        solver.set_maximum_iterations(maximum_iterations)
        solver.set_print_level(print_level)
        solution = ocp.solve(solver)
        return summarize_solved_ocp(
            solution,
            model_path=model_path,
            requested_iterations=maximum_iterations,
            n_phases=ocp.n_phases,
            merge_nodes_token=SolutionMerge.NODES,
        )
    except ModuleNotFoundError as exc:
        dependency_name = getattr(exc, "name", None) or str(exc)
        return OcpSolveSummary(
            success=False,
            message=f"Dependance optionnelle manquante pour resoudre l'OCP: {dependency_name}",
            model_path=model_path,
            requested_iterations=maximum_iterations,
        )
    except Exception as exc:  # pragma: no cover - runtime safety net
        return OcpSolveSummary(
            success=False,
            message=f"Echec de resolution de l'OCP: {exc}",
            model_path=model_path,
            requested_iterations=maximum_iterations,
        )
