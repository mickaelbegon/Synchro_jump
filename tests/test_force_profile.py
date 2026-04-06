"""Tests for the platform force profile."""

from __future__ import annotations

import pytest

from synchro_jump.optimization.force_profile import PlatformForceProfile


def test_force_profile_stays_constant_before_taper() -> None:
    """The force remains at the peak value before the final ramp."""

    profile = PlatformForceProfile(peak_force_newtons=1100.0)

    assert profile.force_at(0.0) == 1100.0
    assert profile.force_at(1.6) == 1100.0


def test_force_profile_halves_over_final_taper() -> None:
    """The force reaches half its peak value at the end of the profile."""

    profile = PlatformForceProfile(peak_force_newtons=1200.0)

    assert profile.force_at(1.85) == pytest.approx(900.0)
    assert profile.force_at(2.0) == pytest.approx(600.0)
    assert profile.force_at(3.0) == pytest.approx(600.0)
