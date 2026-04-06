"""Tests for the vertical jump OCP settings."""

from __future__ import annotations

import pytest

from synchro_jump.optimization.problem import (
    VerticalJumpOcpSettings,
    discrete_force_slider_values,
    discrete_mass_slider_values,
)


def test_force_slider_values_match_requested_grid() -> None:
    """The platform-force slider increments by 50 N between 900 N and 1300 N."""

    assert discrete_force_slider_values() == (900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300)


def test_mass_slider_values_match_requested_grid() -> None:
    """The athlete-mass slider exposes the four requested values."""

    assert discrete_mass_slider_values() == (40, 45, 50, 55)


def test_settings_reject_mass_outside_slider_grid() -> None:
    """The OCP settings stay aligned with the GUI slider values."""

    with pytest.raises(ValueError, match="slider value"):
        VerticalJumpOcpSettings(athlete_mass_kg=52.0)
