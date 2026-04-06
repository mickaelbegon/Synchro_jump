"""Tests for ballistic height estimation."""

from __future__ import annotations

import math

import pytest

from synchro_jump.optimization.estimator import estimate_jump_apex_height


def test_estimate_jump_apex_height_adds_ballistic_gain() -> None:
    """A positive take-off velocity increases the apex height."""

    apex = estimate_jump_apex_height(center_of_mass_height=1.0, vertical_velocity=2.0)

    assert apex == pytest.approx(1.0 + 4.0 / (2.0 * 9.81))


def test_estimate_jump_apex_height_clamps_negative_velocity_gain() -> None:
    """A descending velocity does not add ballistic height."""

    apex = estimate_jump_apex_height(center_of_mass_height=0.9, vertical_velocity=-0.5)

    assert apex == pytest.approx(0.9)


def test_estimate_jump_apex_height_rejects_non_positive_gravity() -> None:
    """Gravity must stay strictly positive."""

    with pytest.raises(ValueError, match="strictly positive"):
        estimate_jump_apex_height(center_of_mass_height=1.0, vertical_velocity=1.0, gravity=0.0)
