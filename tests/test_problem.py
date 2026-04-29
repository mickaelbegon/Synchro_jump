"""Tests for the vertical jump OCP settings."""

from __future__ import annotations

import pytest

from synchro_jump.optimization.problem import (
    CONTACT_MODEL_COMPLIANT_UNILATERAL,
    CONTACT_MODEL_NO_PLATFORM,
    CONTACT_MODEL_RIGID_UNILATERAL,
    VerticalJumpOcpSettings,
    discrete_contact_models,
    discrete_force_slider_values,
    discrete_mass_slider_values,
    snap_to_discrete_value,
)


def test_force_slider_values_match_requested_grid() -> None:
    """The platform-force slider exposes one integer-valued grid."""

    values = discrete_force_slider_values()
    assert len(values) == 21
    assert values[0] == 900
    assert values[-1] == 1300
    assert all(isinstance(value, int) for value in values)
    assert all(next_value - value == 20 for value, next_value in zip(values, values[1:]))


def test_mass_slider_values_match_requested_grid() -> None:
    """The athlete-mass slider exposes one integer-valued grid."""

    values = discrete_mass_slider_values()
    assert len(values) == 18
    assert values[0] == 40
    assert values[-1] == 57
    assert all(isinstance(value, int) for value in values)
    assert all(next_value - value == 1 for value, next_value in zip(values, values[1:]))


def test_contact_model_values_match_supported_modes() -> None:
    """The project should expose both rigid and compliant unilateral contact."""

    assert discrete_contact_models() == (
        CONTACT_MODEL_RIGID_UNILATERAL,
        CONTACT_MODEL_COMPLIANT_UNILATERAL,
        CONTACT_MODEL_NO_PLATFORM,
    )


def test_settings_reject_mass_outside_slider_range() -> None:
    """The OCP settings should reject masses outside the slider bounds."""

    with pytest.raises(ValueError, match="slider range"):
        VerticalJumpOcpSettings(athlete_mass_kg=60.0)


def test_snap_to_discrete_value_returns_the_nearest_grid_point() -> None:
    """Slider values should be snapped to the closest admissible point."""

    values = discrete_force_slider_values()
    snapped = snap_to_discrete_value(1112.0, values)

    assert snapped in values


def test_settings_reject_negative_contact_stiffness() -> None:
    """The compliant contact parameters should stay physically meaningful."""

    with pytest.raises(ValueError, match="contact_stiffness"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, contact_stiffness_n_per_m=-1.0)


def test_settings_reject_unknown_contact_model() -> None:
    """The OCP settings should reject unsupported contact choices."""

    with pytest.raises(ValueError, match="contact_model"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, contact_model="unsupported")


def test_settings_reject_non_boolean_use_sx() -> None:
    """The OCP settings should keep one explicit boolean symbolic-type choice."""

    with pytest.raises(ValueError, match="use_sx"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, use_sx="yes")


def test_settings_reject_non_boolean_expand_dynamics() -> None:
    """The OCP settings should keep one explicit boolean expansion choice."""

    with pytest.raises(ValueError, match="expand_dynamics"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, expand_dynamics="yes")


def test_settings_reject_empty_ipopt_linear_solver() -> None:
    """The IPOPT linear-solver selection should stay explicit."""

    with pytest.raises(ValueError, match="ipopt_linear_solver"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, ipopt_linear_solver="")


def test_default_joint_torque_bounds_match_requested_limit() -> None:
    """The default articulated torque bounds should now allow +/-1000 Nm."""

    settings = VerticalJumpOcpSettings(athlete_mass_kg=50.0)

    assert settings.tau_min_nm == -1000.0
    assert settings.tau_max_nm == 1000.0


def test_settings_reject_non_positive_angular_momentum_bound() -> None:
    """The angular-momentum bound should remain strictly positive."""

    with pytest.raises(ValueError, match="angular_momentum_bound_n_s"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, angular_momentum_bound_n_s=0.0)


def test_settings_reject_torque_regularization_tail_outside_valid_range() -> None:
    """The excluded torque-regularization tail should stay within the shooting horizon."""

    with pytest.raises(ValueError, match="torque_regularization_excluded_tail_nodes"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, torque_regularization_excluded_tail_nodes=-1)

    with pytest.raises(ValueError, match="torque_regularization_excluded_tail_nodes"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, torque_regularization_excluded_tail_nodes=100)

    with pytest.raises(ValueError, match="torque_regularization_tail_weight"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, torque_regularization_tail_weight=-0.1)

    with pytest.raises(ValueError, match="torque_regularization_tail_weight"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, torque_regularization_tail_weight=1.1)


def test_settings_reject_negative_final_mayer_weights() -> None:
    """The terminal height and vertical-speed weights should remain non-negative."""

    with pytest.raises(ValueError, match="final_com_height_mayer_weight"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, final_com_height_mayer_weight=-1.0)

    with pytest.raises(ValueError, match="final_vertical_velocity_mayer_weight"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, final_vertical_velocity_mayer_weight=-1.0)
