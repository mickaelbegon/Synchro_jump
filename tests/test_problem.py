"""Tests for the vertical jump OCP settings."""

from __future__ import annotations

import pytest

from synchro_jump.optimization.problem import (
    CONTACT_MODEL_COMPLIANT_UNILATERAL,
    CONTACT_MODEL_RIGID_UNILATERAL,
    VerticalJumpOcpSettings,
    discrete_contact_models,
    discrete_force_slider_values,
    discrete_mass_slider_values,
)


def test_force_slider_values_match_requested_grid() -> None:
    """The platform-force slider increments by 50 N between 900 N and 1300 N."""

    assert discrete_force_slider_values() == (900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300)


def test_mass_slider_values_match_requested_grid() -> None:
    """The athlete-mass slider exposes the four requested values."""

    assert discrete_mass_slider_values() == (40, 45, 50, 55)


def test_contact_model_values_match_supported_modes() -> None:
    """The project should expose both rigid and compliant unilateral contact."""

    assert discrete_contact_models() == (
        CONTACT_MODEL_RIGID_UNILATERAL,
        CONTACT_MODEL_COMPLIANT_UNILATERAL,
    )


def test_settings_reject_mass_outside_slider_grid() -> None:
    """The OCP settings stay aligned with the GUI slider values."""

    with pytest.raises(ValueError, match="slider value"):
        VerticalJumpOcpSettings(athlete_mass_kg=52.0)


def test_settings_reject_negative_contact_stiffness() -> None:
    """The compliant contact parameters should stay physically meaningful."""

    with pytest.raises(ValueError, match="contact_stiffness"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, contact_stiffness_n_per_m=-1.0)


def test_settings_reject_unknown_contact_model() -> None:
    """The OCP settings should reject unsupported contact choices."""

    with pytest.raises(ValueError, match="contact_model"):
        VerticalJumpOcpSettings(athlete_mass_kg=50.0, contact_model="unsupported")
