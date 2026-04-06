"""Helpers for the explicit vertical platform dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from synchro_jump.optimization.force_profile import PlatformForceProfile


@dataclass(frozen=True)
class CoupledPlatformSolution:
    """Solution of the coupled jumper-platform vertical linear system."""

    qddot: np.ndarray
    contact_force: float
    platform_acceleration: float


def platform_actuation_force(
    time: float,
    peak_force_newtons: float,
    total_duration_s: float = 2.0,
    taper_duration_s: float = 0.3,
) -> float:
    """Return the platform actuation force at one instant."""

    profile = PlatformForceProfile(
        peak_force_newtons=peak_force_newtons,
        total_duration=total_duration_s,
        taper_duration=taper_duration_s,
    )
    return profile.force_at(time)


def solve_coupled_platform_dynamics_numeric(
    mass_matrix: np.ndarray,
    nonlinear_effects: np.ndarray,
    tau: np.ndarray,
    contact_jacobian: np.ndarray,
    contact_bias: float,
    platform_force_newtons: float,
    platform_mass_kg: float,
    gravity: float,
) -> CoupledPlatformSolution:
    """Solve the coupled planar jumper-platform vertical system.

    The unknowns are ``qddot``, the contact force ``lambda`` applied upward on the
    athlete, and the platform acceleration.
    """

    mass_matrix = np.asarray(mass_matrix, dtype=float)
    nonlinear_effects = np.asarray(nonlinear_effects, dtype=float).reshape((-1, 1))
    tau = np.asarray(tau, dtype=float).reshape((-1, 1))
    contact_jacobian = np.asarray(contact_jacobian, dtype=float).reshape((1, -1))

    nq = mass_matrix.shape[0]
    augmented_matrix = np.block(
        [
            [mass_matrix, -contact_jacobian.T, np.zeros((nq, 1))],
            [contact_jacobian, np.zeros((1, 1)), -np.ones((1, 1))],
            [np.zeros((1, nq)), np.ones((1, 1)), platform_mass_kg * np.ones((1, 1))],
        ]
    )
    rhs = np.vstack(
        [
            tau - nonlinear_effects,
            [[-float(contact_bias)]],
            [[platform_force_newtons - platform_mass_kg * gravity]],
        ]
    )
    solution = np.linalg.solve(augmented_matrix, rhs)
    return CoupledPlatformSolution(
        qddot=solution[:nq, 0],
        contact_force=float(solution[nq, 0]),
        platform_acceleration=float(solution[nq + 1, 0]),
    )


def predicted_apex_height_expression_numeric(
    center_of_mass_height: float,
    vertical_velocity: float,
    gravity: float = 9.81,
) -> float:
    """Return the predicted apex height from one take-off state."""

    positive_velocity = max(vertical_velocity, 0.0)
    return center_of_mass_height + positive_velocity**2 / (2.0 * gravity)
