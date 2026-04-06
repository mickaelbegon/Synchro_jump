"""Tests for the explicit platform-dynamics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from synchro_jump.optimization.explicit_platform import (
    platform_actuation_force,
    predicted_apex_height_expression_numeric,
    solve_coupled_platform_dynamics_numeric,
)


def test_platform_actuation_force_matches_requested_profile() -> None:
    """The explicit-platform helper matches the requested 2 s force profile."""

    assert platform_actuation_force(0.0, 1200.0) == pytest.approx(1200.0)
    assert platform_actuation_force(1.85, 1200.0) == pytest.approx(900.0)
    assert platform_actuation_force(2.0, 1200.0) == pytest.approx(600.0)


def test_coupled_platform_numeric_solver_solves_one_dof_system() -> None:
    """The reduced linear solve returns the expected acceleration and contact force."""

    solution = solve_coupled_platform_dynamics_numeric(
        mass_matrix=np.array([[2.0]]),
        nonlinear_effects=np.array([0.0]),
        tau=np.array([0.0]),
        contact_jacobian=np.array([[1.0]]),
        contact_bias=0.0,
        platform_force_newtons=100.0,
        platform_mass_kg=10.0,
        gravity=0.0,
    )

    assert solution.qddot[0] == pytest.approx(100.0 / 12.0)
    assert solution.contact_force == pytest.approx(100.0 / 6.0)
    assert solution.platform_acceleration == pytest.approx(25.0 / 3.0)


def test_predicted_apex_height_numeric_ignores_negative_takeoff_velocity() -> None:
    """Descending take-off speed does not increase the predicted apex height."""

    apex = predicted_apex_height_expression_numeric(0.9, -1.0)

    assert apex == pytest.approx(0.9)
