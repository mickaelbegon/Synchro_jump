"""Tests for the rigid platform-contact model."""

from __future__ import annotations

import pytest

from synchro_jump.optimization.contact import PlatformInteractionModel


def test_contact_force_matches_platform_force_balance() -> None:
    """The contact force follows directly from the platform equation of motion."""

    interaction = PlatformInteractionModel(platform_mass_kg=80.0, gravity=10.0)

    contact_force = interaction.contact_force(
        platform_actuation_force_newtons=1200.0,
        platform_vertical_acceleration=2.0,
    )

    assert contact_force == pytest.approx(240.0)


def test_liftoff_residual_is_zero_when_contact_force_vanishes() -> None:
    """The lift-off residual is zero at the instant of take-off."""

    interaction = PlatformInteractionModel(platform_mass_kg=80.0, gravity=9.81)
    acceleration = 1000.0 / 80.0 - 9.81

    assert interaction.liftoff_residual(1000.0, acceleration) == pytest.approx(0.0)
