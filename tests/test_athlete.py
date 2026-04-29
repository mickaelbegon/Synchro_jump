"""Tests for the reduced athlete anthropometry."""

from __future__ import annotations

import pytest

from synchro_jump.modeling.athlete import AthleteMorphology


def test_segment_lengths_sum_to_body_height() -> None:
    """The reduced segment lengths reconstruct the standing height."""

    morphology = AthleteMorphology(height_m=1.60, mass_kg=50.0)
    lengths = morphology.segment_lengths

    assert lengths.leg_foot + lengths.thigh + lengths.trunk == pytest.approx(1.60)


def test_default_initial_flexion_matches_problem_statement() -> None:
    """The initial generalized knee/hip amplitudes default to 90 degrees."""

    morphology = AthleteMorphology()

    assert morphology.initial_joint_flexion_deg == pytest.approx(90.0)
