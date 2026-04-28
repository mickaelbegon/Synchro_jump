"""Structured initial guesses for the explicit-platform jump OCP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OcpInitialGuess:
    """State and control trajectories used to warm-start the OCP."""

    q: np.ndarray
    qdot: np.ndarray
    platform_position: np.ndarray
    platform_velocity: np.ndarray
    tau: np.ndarray


def complete_extension_configuration(
    initial_q: tuple[float, ...] | list[float] | np.ndarray,
    *,
    final_platform_height_m: float = 1.3,
) -> np.ndarray:
    """Return one fully extended generalized configuration."""

    q_final = np.asarray(initial_q, dtype=float).reshape((-1,)).copy()
    if q_final.shape[0] not in (3, 5):
        raise ValueError("The reduced jumper initial guess expects 3 or 5 generalized coordinates")

    if q_final.shape[0] == 5:
        q_final[1] = final_platform_height_m
        q_final[2] = 0.0
        q_final[3] = 0.0
        q_final[4] = 0.0
    else:
        q_final[0] = 0.0
        q_final[1] = 0.0
        q_final[2] = 0.0
    return q_final


def _finite_difference(values: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Return one first-order finite-difference derivative along the node axis."""

    if values.shape[1] != time.shape[0]:
        raise ValueError("values and time must share the same number of nodes")
    if time.shape[0] < 2:
        return np.zeros_like(values)
    return np.gradient(values, time, axis=1, edge_order=1)


def _inverse_dynamics_trajectory(model: Any, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray) -> np.ndarray:
    """Return the generalized torques corresponding to one state trajectory."""

    inverse_dynamics = model.inverse_dynamics()
    parameters = np.zeros((0, 1))
    external_forces = np.zeros((0, 1))
    tau = np.zeros((q.shape[0], q.shape[1] - 1), dtype=float)

    for node_index in range(q.shape[1] - 1):
        q_column = q[:, node_index].reshape((-1, 1))
        qdot_column = qdot[:, node_index].reshape((-1, 1))
        qddot_column = qddot[:, node_index].reshape((-1, 1))
        tau[:, node_index] = np.asarray(
            inverse_dynamics(q_column, qdot_column, qddot_column, external_forces, parameters),
            dtype=float,
        ).reshape((-1,))

    return tau


def _static_equilibrium_torque(model: Any, q: np.ndarray) -> np.ndarray:
    """Return the generalized torque required to hold one posture statically."""

    inverse_dynamics = model.inverse_dynamics()
    parameters = np.zeros((0, 1))
    external_forces = np.zeros((0, 1))
    zero_qdot = np.zeros_like(q)
    zero_qddot = np.zeros_like(q)
    return np.asarray(
        inverse_dynamics(q, zero_qdot, zero_qddot, external_forces, parameters),
        dtype=float,
    ).reshape((-1,))


def static_equilibrium_torque(model: Any, q: np.ndarray) -> np.ndarray:
    """Return the generalized torque required to hold one posture statically."""

    q_column = np.asarray(q, dtype=float).reshape((-1, 1))
    return _static_equilibrium_torque(model, q_column)


def build_linear_inverse_dynamics_initial_guess(
    model: Any,
    initial_q: tuple[float, ...] | list[float] | np.ndarray,
    *,
    duration_s: float,
    n_shooting: int,
    final_platform_height_m: float = 1.3,
) -> OcpInitialGuess:
    """Build one warm start from linear kinematics and inverse dynamics."""

    if duration_s <= 0.0:
        raise ValueError("duration_s must be strictly positive")
    if n_shooting <= 0:
        raise ValueError("n_shooting must be strictly positive")

    q_start = np.asarray(initial_q, dtype=float).reshape((-1,))
    q_end = complete_extension_configuration(q_start, final_platform_height_m=final_platform_height_m)

    state_time = np.linspace(0.0, duration_s, n_shooting + 1)
    q = np.vstack(
        [
            np.linspace(q_start[dof_index], q_end[dof_index], n_shooting + 1)
            for dof_index in range(q_start.shape[0])
        ]
    )
    qdot = _finite_difference(q, state_time)
    qddot = _finite_difference(qdot, state_time)

    if q_start.shape[0] == 5:
        platform_position = np.linspace(0.0, final_platform_height_m, n_shooting + 1, dtype=float).reshape((1, -1))
        platform_velocity = _finite_difference(platform_position, state_time)
    else:
        platform_position = np.zeros((0, n_shooting + 1), dtype=float)
        platform_velocity = np.zeros((0, n_shooting + 1), dtype=float)
    tau = _inverse_dynamics_trajectory(model, q, qdot, qddot)
    if tau.shape[1] > 0:
        tau[:, 0] = _static_equilibrium_torque(model, q[:, 0].reshape((-1, 1)))

    return OcpInitialGuess(
        q=q,
        qdot=qdot,
        platform_position=platform_position,
        platform_velocity=platform_velocity,
        tau=tau,
    )
