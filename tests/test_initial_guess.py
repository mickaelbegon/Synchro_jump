"""Tests for the structured OCP initial guess."""

from __future__ import annotations

import numpy as np

from synchro_jump.optimization.initial_guess import (
    build_linear_inverse_dynamics_initial_guess,
    complete_extension_configuration,
    static_equilibrium_torque,
)


class _FakeInverseDynamicsModel:
    """Small fake model exposing one deterministic inverse dynamics."""

    def inverse_dynamics(self):
        """Return one callable inverse-dynamics surrogate."""

        def _call(q, qdot, qddot, _external_forces, _parameters):
            return q + 2.0 * qdot + 3.0 * qddot

        return _call


def test_complete_extension_configuration_sets_vertical_lift_and_extension() -> None:
    """The final configuration should reach full extension at 1.3 m."""

    q_final = complete_extension_configuration((0.0, 0.0, 0.2, -1.0, 0.8), final_platform_height_m=1.3)

    assert np.allclose(q_final, [0.0, 1.3, 0.0, 0.0, 0.0])


def test_build_linear_inverse_dynamics_initial_guess_returns_consistent_shapes() -> None:
    """The warm start should provide state nodes and control nodes with matching sizes."""

    guess = build_linear_inverse_dynamics_initial_guess(
        _FakeInverseDynamicsModel(),
        (0.0, 0.0, 0.0, -1.7453292519943295, 1.7453292519943295),
        duration_s=1.0,
        n_shooting=4,
    )

    assert guess.q.shape == (5, 5)
    assert guess.qdot.shape == (5, 5)
    assert guess.platform_position.shape == (1, 5)
    assert guess.platform_velocity.shape == (1, 5)
    assert guess.tau.shape == (5, 4)
    assert np.isclose(guess.platform_position[0, 0], 0.0)
    assert np.isclose(guess.platform_position[0, -1], 1.3)
    assert np.isclose(guess.q[3, -1], 0.0)
    assert np.isclose(guess.q[4, -1], 0.0)


def test_build_linear_inverse_dynamics_initial_guess_starts_with_static_equilibrium_tau() -> None:
    """The first control node should match a static inverse-dynamics balance."""

    initial_q = (0.0, 0.0, 0.0, -1.7453292519943295, 1.7453292519943295)
    guess = build_linear_inverse_dynamics_initial_guess(
        _FakeInverseDynamicsModel(),
        initial_q,
        duration_s=1.0,
        n_shooting=4,
    )

    assert np.allclose(guess.tau[:, 0], np.asarray(initial_q, dtype=float))


def test_static_equilibrium_torque_uses_zero_velocity_and_acceleration() -> None:
    """The static helper should query inverse dynamics with qdot=qddot=0."""

    q = np.array([0.1, -0.2, 0.3])
    tau = static_equilibrium_torque(_FakeInverseDynamicsModel(), q)

    assert np.allclose(tau, q)
