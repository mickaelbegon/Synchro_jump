"""Runtime helpers to solve and summarize the `bioptim` OCP."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import shutil
from typing import Any, Callable

import numpy as np

from synchro_jump.optimization.bioptim_ocp import (
    VerticalJumpBioptimOcpBuilder,
    _model_contact_axis,
    _model_contact_names,
)
from synchro_jump.optimization.contact import PlatformInteractionModel
from synchro_jump.optimization.explicit_platform import (
    platform_actuation_force,
    predicted_apex_height_expression_numeric,
    solve_coupled_platform_dynamics_numeric,
)
from synchro_jump.optimization.problem import (
    CONTACT_MODEL_COMPLIANT_UNILATERAL,
    CONTACT_MODEL_NO_PLATFORM,
    CONTACT_MODEL_RIGID_UNILATERAL,
    VerticalJumpOcpSettings,
)
from synchro_jump.optimization.solution_cache import (
    load_cached_solution_summary,
    save_cached_solution_summary,
)


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
    final_contact_force_n: float | None = None
    takeoff_condition_satisfied: bool | None = None
    contact_model: str = ""
    time: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    state_trajectories: dict[str, np.ndarray] = field(default_factory=dict)
    control_trajectories: dict[str, np.ndarray] = field(default_factory=dict)
    com_height_trajectory_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    com_vertical_velocity_trajectory_m_s: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    contact_force_trajectory_n: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    external_force_ap_trajectory_n: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    platform_force_trajectory_n: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    platform_acceleration_trajectory_m_s2: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    from_cache: bool = False


def _add_platform_force_to_numeric_generalized_force(
    generalized_force: np.ndarray,
    platform_force_newtons: float,
) -> np.ndarray:
    """Inject the platform actuation into the vertical root translation DoF."""

    updated_force = np.asarray(generalized_force, dtype=float).copy()
    if updated_force.shape[0] > 1:
        updated_force[1] += float(platform_force_newtons)
    return updated_force


def _default_hsl_library_candidates() -> tuple[Path, ...]:
    """Return the known local HSL library candidates ordered by preference."""

    home = Path.home()
    env_root = home / "miniconda3" / "envs"
    candidates = [
        Path("generated/solver_libs/libhsl.dylib"),
        home / "Documents" / "GIT" / "ThirdParty-HSL" / "lib" / "lib" / "libhsl.dylib",
    ]
    candidates.extend(sorted(env_root.glob("*/lib/libhsl.dylib")))
    candidates.extend(sorted(env_root.glob("*/lib/libcoinhsl.dylib")))

    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return tuple(unique_candidates)


def ensure_local_hsl_library(
    *,
    local_dir: str | Path = "generated/solver_libs",
    preferred_path: str | Path | None = None,
    candidate_paths: tuple[Path, ...] | None = None,
) -> Path | None:
    """Ensure one local copy of `libhsl.dylib` is available for IPOPT/MA57."""

    local_dir_path = Path(local_dir)
    local_target = local_dir_path / "libhsl.dylib"
    if local_target.exists():
        return local_target

    candidates = []
    if preferred_path is not None:
        candidates.append(Path(preferred_path))
    if candidate_paths is not None:
        candidates.extend(candidate_paths)
    else:
        candidates.extend(_default_hsl_library_candidates())

    for candidate in candidates:
        if not candidate.exists():
            continue
        source_path = candidate.resolve()
        local_dir_path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, local_target)
        return local_target

    return None


def _configure_ipopt_solver(
    solver: Any,
    *,
    maximum_iterations: int,
    print_level: int,
    linear_solver: str,
    hsl_library_path: str | Path | None = None,
) -> None:
    """Configure IPOPT to print each iteration and detailed timing statistics."""

    solver.set_maximum_iterations(maximum_iterations)
    solver.set_print_level(print_level)
    solver.set_linear_solver(linear_solver)
    if hsl_library_path is not None:
        solver.set_option_unsafe(str(Path(hsl_library_path)), "hsllib")
    solver.set_option_unsafe("yes", "print_timing_statistics")
    solver.set_option_unsafe(1, "print_frequency_iter")
    solver.set_option_unsafe(0, "print_frequency_time")


def _merge_split_states(state_trajectories: dict[str, np.ndarray], prefix: str) -> np.ndarray:
    """Return one full generalized vector assembled from split root and joint arrays."""

    split_roots = state_trajectories.get(f"{prefix}_roots")
    split_joints = state_trajectories.get(f"{prefix}_joints")
    if split_roots is not None and split_joints is not None:
        return np.vstack((split_roots, split_joints))
    if split_joints is not None:
        return split_joints
    if split_roots is not None:
        return split_roots

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


def _print_terminal_objective_breakdown(summary: OcpSolveSummary) -> None:
    """Print a readable decomposition of the jump objective in the terminal."""

    if summary.takeoff_com_height_m is None or summary.takeoff_com_vertical_velocity_m_s is None:
        return

    positive_vertical_velocity = max(float(summary.takeoff_com_vertical_velocity_m_s), 0.0)
    ballistic_gain = positive_vertical_velocity**2 / (2.0 * 9.81)
    print("\n[SynchroJump] Decomposition numerique de l'objectif de saut")
    print(f"  -z_CoM(T) = {-float(summary.takeoff_com_height_m):.6f}")
    print(f"  -max(vz_CoM(T), 0)^2 / (2g) = {-ballistic_gain:.6f}")
    print(f"  z_CoM(T) = {float(summary.takeoff_com_height_m):.6f} m")
    print(f"  max(vz_CoM(T), 0) = {positive_vertical_velocity:.6f} m/s")
    print(f"  gain balistique predit = {ballistic_gain:.6f} m")


def _contact_index_from_name(contact_names: tuple[str, ...], contact_name: str) -> int:
    """Resolve one rigid-contact index from one exported contact name."""

    names = list(contact_names)
    if contact_name in names:
        return names.index(contact_name)

    axis_matches = [index for index, name in enumerate(names) if name.startswith(f"{contact_name}_")]
    if len(axis_matches) == 1:
        return axis_matches[0]

    raise ValueError(f"Unknown contact name: {contact_name}")


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


def evaluate_external_force_ap_trajectory(
    model_path: str | Path,
    state_trajectories: dict[str, np.ndarray],
    *,
    time: np.ndarray,
    athlete_mass_kg: float,
) -> np.ndarray:
    """Estimate the anteroposterior external force from CoM horizontal acceleration."""

    from casadi import DM
    from bioptim import BiorbdModel

    if time.size == 0:
        return np.array([], dtype=float)

    model = BiorbdModel(str(Path(model_path)))
    q_trajectory = _merge_split_states(state_trajectories, "q")
    com_fun = model.center_of_mass()
    com_x_trajectory = np.zeros(q_trajectory.shape[1], dtype=float)

    for node_index in range(q_trajectory.shape[1]):
        q_column = DM(q_trajectory[:, node_index].reshape((-1, 1)))
        com = np.asarray(com_fun(q_column, DM()), dtype=float).reshape((-1,))
        com_x_trajectory[node_index] = com[0]

    if time.size < 2 or np.any(np.diff(time) <= 0.0):
        return np.zeros_like(time, dtype=float)

    com_x_velocity = np.gradient(com_x_trajectory, time)
    com_x_acceleration = np.gradient(com_x_velocity, time)
    return athlete_mass_kg * com_x_acceleration


def evaluate_no_platform_equivalent_contact_force_trajectory(
    model_path: str | Path,
    state_trajectories: dict[str, np.ndarray],
    control_trajectories: dict[str, np.ndarray],
    *,
    time: np.ndarray,
    athlete_mass_kg: float,
    gravity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return one equivalent external vertical force from CoM acceleration in no-platform mode."""

    from casadi import DM
    from bioptim import BiorbdModel

    model = BiorbdModel(str(Path(model_path)))
    q_trajectory = _merge_split_states(state_trajectories, "q")
    qdot_trajectory = _merge_split_states(state_trajectories, "qdot")
    tau_trajectory = np.asarray(control_trajectories["tau_joints"], dtype=float)
    mass_matrix_fun = model.mass_matrix()
    nonlinear_effects_fun = model.non_linear_effects()
    com_acceleration_fun = model.center_of_mass_acceleration()

    equivalent_contact_force = np.zeros(time.shape[0], dtype=float)
    com_acceleration_trajectory = np.zeros(time.shape[0], dtype=float)
    platform_force_trajectory = np.zeros(time.shape[0], dtype=float)

    for node_index in range(time.shape[0]):
        q_column = DM(q_trajectory[:, node_index].reshape((-1, 1)))
        qdot_column = DM(qdot_trajectory[:, node_index].reshape((-1, 1)))
        tau_column = np.asarray(tau_trajectory[:, min(node_index, tau_trajectory.shape[1] - 1)], dtype=float).reshape((-1, 1))
        qddot = np.linalg.solve(
            np.asarray(mass_matrix_fun(q_column, DM()), dtype=float),
            tau_column - np.asarray(nonlinear_effects_fun(q_column, qdot_column, DM()), dtype=float),
        )
        com_acceleration = np.asarray(
            com_acceleration_fun(q_column, qdot_column, DM(qddot), DM()),
            dtype=float,
        ).reshape((-1,))
        com_acceleration_trajectory[node_index] = com_acceleration[2]
        equivalent_contact_force[node_index] = athlete_mass_kg * (com_acceleration[2] + gravity)

    return platform_force_trajectory, equivalent_contact_force, com_acceleration_trajectory


def _evaluate_compliant_contact_force_trajectory(
    state_trajectories: dict[str, np.ndarray],
    *,
    time: np.ndarray,
    peak_force_newtons: float,
    platform_mass_kg: float,
    contact_stiffness_n_per_m: float,
    contact_damping_n_s_per_m: float,
    total_duration_s: float,
    taper_duration_s: float,
    gravity: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the compliant contact interaction along one trajectory."""

    q_trajectory = _merge_split_states(state_trajectories, "q")
    qdot_trajectory = _merge_split_states(state_trajectories, "qdot")
    platform_position_trajectory = np.asarray(state_trajectories["platform_position"], dtype=float).reshape((-1,))
    platform_velocity_trajectory = np.asarray(state_trajectories["platform_velocity"], dtype=float).reshape((-1,))
    interaction = PlatformInteractionModel(
        platform_mass_kg=platform_mass_kg,
        gravity=gravity,
        contact_stiffness_n_per_m=contact_stiffness_n_per_m,
        contact_damping_n_s_per_m=contact_damping_n_s_per_m,
    )

    platform_force_trajectory = np.zeros(time.shape[0], dtype=float)
    contact_force_trajectory = np.zeros(time.shape[0], dtype=float)
    platform_acceleration_trajectory = np.zeros(time.shape[0], dtype=float)

    for node_index, current_time in enumerate(time):
        platform_force = platform_actuation_force(
            float(current_time),
            peak_force_newtons=peak_force_newtons,
            total_duration_s=total_duration_s,
            taper_duration_s=taper_duration_s,
        )
        contact_force = interaction.compliant_contact_force(
            platform_position_m=float(platform_position_trajectory[node_index]),
            platform_velocity_m_s=float(platform_velocity_trajectory[node_index]),
            foot_position_m=float(q_trajectory[1, node_index]),
            foot_velocity_m_s=float(qdot_trajectory[1, node_index]),
        )
        platform_force_trajectory[node_index] = platform_force
        contact_force_trajectory[node_index] = contact_force
        platform_acceleration_trajectory[node_index] = (
            platform_force - platform_mass_kg * gravity - contact_force
        ) / platform_mass_kg

    return platform_force_trajectory, contact_force_trajectory, platform_acceleration_trajectory


def _evaluate_rigid_contact_force_trajectory(
    model_path: str | Path,
    state_trajectories: dict[str, np.ndarray],
    control_trajectories: dict[str, np.ndarray],
    *,
    time: np.ndarray,
    peak_force_newtons: float,
    platform_mass_kg: float,
    total_duration_s: float,
    taper_duration_s: float,
    gravity: float,
    contact_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the rigid coupled contact interaction along one trajectory."""

    from casadi import DM
    from bioptim import BiorbdModel

    model = BiorbdModel(str(Path(model_path)))
    q_trajectory = _merge_split_states(state_trajectories, "q")
    qdot_trajectory = _merge_split_states(state_trajectories, "qdot")
    tau_joints_trajectory = np.asarray(control_trajectories["tau_joints"], dtype=float)

    contact_index = _contact_index_from_name(_model_contact_names(model), contact_name)
    contact_axis = _model_contact_axis(model, contact_index)
    contact_acceleration = model.rigid_contact_acceleration(contact_index, contact_axis)
    mass_matrix_fun = model.mass_matrix()
    nonlinear_effects_fun = model.non_linear_effects()

    platform_force_trajectory = np.zeros(time.shape[0], dtype=float)
    contact_force_trajectory = np.zeros(time.shape[0], dtype=float)
    platform_acceleration_trajectory = np.zeros(time.shape[0], dtype=float)

    for node_index, current_time in enumerate(time):
        q_column = DM(q_trajectory[:, node_index].reshape((-1, 1)))
        qdot_column = DM(qdot_trajectory[:, node_index].reshape((-1, 1)))
        zero_qddot = DM.zeros(model.nb_q, 1)

        contact_bias = float(
            np.asarray(contact_acceleration(q_column, qdot_column, zero_qddot, DM()), dtype=float).reshape((-1,))[0]
        )
        contact_jacobian = np.zeros((1, model.nb_q), dtype=float)
        for dof_index in range(model.nb_q):
            basis_qddot = DM.zeros(model.nb_q, 1)
            basis_qddot[dof_index] = 1.0
            basis_acceleration = float(
                np.asarray(contact_acceleration(q_column, qdot_column, basis_qddot, DM()), dtype=float)
                .reshape((-1,))[0]
            )
            contact_jacobian[0, dof_index] = basis_acceleration - contact_bias

        tau_joints = np.asarray(control_trajectories["tau_joints"][:, min(node_index, tau_joints_trajectory.shape[1] - 1)])
        tau_joints = np.nan_to_num(tau_joints.astype(float), nan=0.0)
        platform_force = platform_actuation_force(
            float(current_time),
            peak_force_newtons=peak_force_newtons,
            total_duration_s=total_duration_s,
            taper_duration_s=taper_duration_s,
        )
        tau = np.concatenate((np.zeros(model.nb_root, dtype=float), tau_joints))
        tau = _add_platform_force_to_numeric_generalized_force(tau, platform_force)
        coupled_solution = solve_coupled_platform_dynamics_numeric(
            mass_matrix=np.asarray(mass_matrix_fun(q_column, DM()), dtype=float),
            nonlinear_effects=np.asarray(nonlinear_effects_fun(q_column, qdot_column, DM()), dtype=float),
            tau=tau,
            contact_jacobian=contact_jacobian,
            contact_bias=contact_bias,
            platform_force_newtons=platform_force,
            platform_mass_kg=platform_mass_kg,
            gravity=gravity,
        )
        platform_force_trajectory[node_index] = platform_force
        contact_force_trajectory[node_index] = coupled_solution.contact_force
        platform_acceleration_trajectory[node_index] = coupled_solution.platform_acceleration

    return platform_force_trajectory, contact_force_trajectory, platform_acceleration_trajectory


def evaluate_contact_force_trajectory(
    model_path: str | Path,
    state_trajectories: dict[str, np.ndarray],
    control_trajectories: dict[str, np.ndarray],
    *,
    time: np.ndarray,
    peak_force_newtons: float,
    platform_mass_kg: float,
    athlete_mass_kg: float = 50.0,
    contact_model: str = CONTACT_MODEL_RIGID_UNILATERAL,
    contact_stiffness_n_per_m: float = 30000.0,
    contact_damping_n_s_per_m: float = 1500.0,
    total_duration_s: float = 2.0,
    taper_duration_s: float = 0.3,
    gravity: float = 9.81,
    contact_name: str = "platform_contact",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the selected contact interaction along one trajectory."""

    if contact_model == CONTACT_MODEL_RIGID_UNILATERAL:
        return _evaluate_rigid_contact_force_trajectory(
            model_path,
            state_trajectories,
            control_trajectories,
            time=time,
            peak_force_newtons=peak_force_newtons,
            platform_mass_kg=platform_mass_kg,
            total_duration_s=total_duration_s,
            taper_duration_s=taper_duration_s,
            gravity=gravity,
            contact_name=contact_name,
        )

    if contact_model == CONTACT_MODEL_COMPLIANT_UNILATERAL:
        return _evaluate_compliant_contact_force_trajectory(
            state_trajectories,
            time=time,
            peak_force_newtons=peak_force_newtons,
            platform_mass_kg=platform_mass_kg,
            contact_stiffness_n_per_m=contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=contact_damping_n_s_per_m,
            total_duration_s=total_duration_s,
            taper_duration_s=taper_duration_s,
            gravity=gravity,
        )

    if contact_model == CONTACT_MODEL_NO_PLATFORM:
        return evaluate_no_platform_equivalent_contact_force_trajectory(
            model_path,
            state_trajectories,
            control_trajectories,
            time=time,
            athlete_mass_kg=athlete_mass_kg,
            gravity=gravity,
        )

    raise ValueError(f"Unsupported contact model: {contact_model}")


def summarize_solved_ocp(
    solution: Any,
    *,
    model_path: str | Path,
    requested_iterations: int,
    n_phases: int,
    merge_nodes_token: Any,
    peak_force_newtons: float,
    platform_mass_kg: float,
    contact_model: str = CONTACT_MODEL_RIGID_UNILATERAL,
    athlete_mass_kg: float = 50.0,
    contact_stiffness_n_per_m: float = 30000.0,
    contact_damping_n_s_per_m: float = 1500.0,
    total_duration_s: float = 2.0,
    taper_duration_s: float = 0.3,
    gravity: float = 9.81,
    contact_name: str = "platform_contact",
    com_evaluator: Callable[[str | Path, dict[str, np.ndarray]], tuple[np.ndarray, np.ndarray]] = evaluate_com_trajectory,
    contact_force_evaluator: Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    external_force_ap_evaluator: Callable[..., np.ndarray] | None = None,
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
    contact_force_evaluator = contact_force_evaluator or evaluate_contact_force_trajectory
    platform_force_trajectory_n, contact_force_trajectory_n, platform_acceleration_trajectory_m_s2 = (
        contact_force_evaluator(
            model_path,
            state_trajectories,
            control_trajectories,
            time=time,
            peak_force_newtons=peak_force_newtons,
            platform_mass_kg=platform_mass_kg,
            contact_model=contact_model,
            athlete_mass_kg=athlete_mass_kg,
            contact_stiffness_n_per_m=contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=contact_damping_n_s_per_m,
            total_duration_s=total_duration_s,
            taper_duration_s=taper_duration_s,
            gravity=gravity,
            contact_name=contact_name,
        )
    )
    if external_force_ap_evaluator is None:
        external_force_ap_trajectory_n = np.zeros_like(contact_force_trajectory_n)
    else:
        external_force_ap_trajectory_n = np.asarray(
            external_force_ap_evaluator(
                model_path,
                state_trajectories,
                time=time,
                athlete_mass_kg=athlete_mass_kg,
            ),
            dtype=float,
        ).reshape((-1,))

    final_height = float(com_height_trajectory_m[-1]) if com_height_trajectory_m.size else None
    final_vertical_velocity = (
        float(com_vertical_velocity_trajectory_m_s[-1]) if com_vertical_velocity_trajectory_m_s.size else None
    )
    predicted_apex_height = None
    if final_height is not None and final_vertical_velocity is not None:
        predicted_apex_height = predicted_apex_height_expression_numeric(
            final_height,
            final_vertical_velocity,
            gravity=gravity,
        )

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
        final_contact_force_n=float(contact_force_trajectory_n[-1]) if contact_force_trajectory_n.size else None,
        takeoff_condition_satisfied=(
            bool(abs(float(contact_force_trajectory_n[-1])) <= 1.0) if contact_force_trajectory_n.size else None
        ),
        contact_model=contact_model,
        time=time,
        state_trajectories=state_trajectories,
        control_trajectories=control_trajectories,
        com_height_trajectory_m=com_height_trajectory_m,
        com_vertical_velocity_trajectory_m_s=com_vertical_velocity_trajectory_m_s,
        contact_force_trajectory_n=contact_force_trajectory_n,
        external_force_ap_trajectory_n=external_force_ap_trajectory_n,
        platform_force_trajectory_n=platform_force_trajectory_n,
        platform_acceleration_trajectory_m_s2=platform_acceleration_trajectory_m_s2,
    )


def solve_ocp_runtime_summary(
    settings: VerticalJumpOcpSettings,
    peak_force_newtons: float,
    *,
    model_output_dir: str | Path = "generated",
    cache_dir: str | Path = "generated/optimal_solution_cache",
    use_cache: bool = False,
    maximum_iterations: int = 1000,
    print_level: int = 5,
) -> OcpSolveSummary:
    """Build and solve the runtime OCP, then summarize the result."""

    builder = VerticalJumpBioptimOcpBuilder(settings=settings)
    if use_cache:
        cached_summary = load_cached_solution_summary(
            cache_dir,
            settings,
            peak_force_newtons,
            maximum_iterations,
        )
        if cached_summary is not None:
            return replace(
                cached_summary,
                from_cache=True,
                message=f"{cached_summary.message} (chargee depuis le cache)",
            )
    try:
        model_path = builder.export_model(model_output_dir)
    except (ImportError, ModuleNotFoundError) as exc:
        dependency_name = getattr(exc, "name", None) or str(exc)
        return OcpSolveSummary(
            success=False,
            message=f"Dependance optionnelle manquante pour resoudre l'OCP: {dependency_name}",
            model_path=Path(model_output_dir) / "vertical_jumper_3segments.bioMod",
            requested_iterations=maximum_iterations,
            contact_model=settings.contact_model,
        )

    try:
        from bioptim import CostType, SolutionMerge, Solver
    except ModuleNotFoundError as exc:
        dependency_name = getattr(exc, "name", None) or str(exc)
        return OcpSolveSummary(
            success=False,
            message=f"Dependance optionnelle manquante pour resoudre l'OCP: {dependency_name}",
            model_path=model_path,
            requested_iterations=maximum_iterations,
            contact_model=settings.contact_model,
        )

    try:
        ocp = builder.build_ocp(peak_force_newtons=peak_force_newtons, model_path=model_path)
        solver = Solver.IPOPT()
        hsl_library_path = None
        if settings.ipopt_linear_solver.lower() == "ma57":
            hsl_library_path = ensure_local_hsl_library(preferred_path=settings.ipopt_hsl_library_path)
            if hsl_library_path is None:
                return OcpSolveSummary(
                    success=False,
                    message=(
                        "Impossible de configurer IPOPT avec ma57: aucune librairie HSL "
                        "(`libhsl.dylib`) n'a ete trouvee localement ou dans les environnements Conda."
                    ),
                    model_path=model_path,
                    requested_iterations=maximum_iterations,
                    contact_model=settings.contact_model,
                )
        _configure_ipopt_solver(
            solver,
            maximum_iterations=maximum_iterations,
            print_level=print_level,
            linear_solver=settings.ipopt_linear_solver,
            hsl_library_path=hsl_library_path,
        )
        solution = ocp.solve(solver)
        print("\n[SynchroJump] Detail des termes de la fonction objectif")
        solution.print_cost(CostType.OBJECTIVES)
        summary = summarize_solved_ocp(
            solution,
            model_path=model_path,
            requested_iterations=maximum_iterations,
            n_phases=ocp.n_phases,
            merge_nodes_token=SolutionMerge.NODES,
            peak_force_newtons=peak_force_newtons,
            platform_mass_kg=settings.platform_mass_kg,
            contact_model=settings.contact_model,
            athlete_mass_kg=settings.athlete_mass_kg,
            contact_stiffness_n_per_m=settings.contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=settings.contact_damping_n_s_per_m,
            total_duration_s=settings.final_time_upper_bound_s,
            external_force_ap_evaluator=evaluate_external_force_ap_trajectory,
        )
        save_cached_solution_summary(
            cache_dir,
            settings,
            peak_force_newtons,
            maximum_iterations,
            summary,
        )
        _print_terminal_objective_breakdown(summary)
        return summary
    except ModuleNotFoundError as exc:
        dependency_name = getattr(exc, "name", None) or str(exc)
        return OcpSolveSummary(
            success=False,
            message=f"Dependance optionnelle manquante pour resoudre l'OCP: {dependency_name}",
            model_path=model_path,
            requested_iterations=maximum_iterations,
            contact_model=settings.contact_model,
        )
    except RuntimeError as exc:
        return OcpSolveSummary(
            success=False,
            message=str(exc),
            model_path=model_path,
            requested_iterations=maximum_iterations,
            contact_model=settings.contact_model,
        )
    except Exception as exc:  # pragma: no cover - runtime safety net
        return OcpSolveSummary(
            success=False,
            message=f"Echec de resolution de l'OCP: {exc}",
            model_path=model_path,
            requested_iterations=maximum_iterations,
            contact_model=settings.contact_model,
        )
