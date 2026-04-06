"""Tests for the surrogate jump metrics."""

from __future__ import annotations

import pytest

from synchro_jump.optimization.surrogate import (
    estimate_apex_from_contact_profile,
    estimate_takeoff_velocity_from_contact_profile,
)


def test_takeoff_velocity_is_zero_without_enough_support_force() -> None:
    """Body-weight support alone does not create a positive take-off speed."""

    velocity = estimate_takeoff_velocity_from_contact_profile(
        contact_force_profile_newtons=[490.5, 490.5, 490.5],
        athlete_mass_kg=50.0,
        total_duration_s=1.0,
    )

    assert velocity == pytest.approx(0.0)


def test_takeoff_velocity_grows_with_positive_net_impulse() -> None:
    """A profile stronger than body weight yields a positive take-off speed."""

    velocity = estimate_takeoff_velocity_from_contact_profile(
        contact_force_profile_newtons=[800.0, 800.0, 800.0],
        athlete_mass_kg=50.0,
        total_duration_s=1.0,
    )

    assert velocity > 0.0


def test_apex_from_contact_profile_exceeds_takeoff_height_when_velocity_positive() -> None:
    """Positive take-off velocity increases the apex estimate."""

    apex = estimate_apex_from_contact_profile(
        contact_force_profile_newtons=[800.0, 800.0, 800.0],
        athlete_mass_kg=50.0,
        takeoff_center_of_mass_height_m=0.9,
        total_duration_s=1.0,
    )

    assert apex > 0.9
