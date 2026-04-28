"""Tests for the reduced jumper model definition."""

from __future__ import annotations

import math

import pytest

from synchro_jump.modeling import AthleteMorphology, PlanarJumperModelDefinition


def test_model_definition_exposes_requested_dofs_and_torques() -> None:
    """The reduced jumper keeps a planar floating base and two actuated joints."""

    model_definition = PlanarJumperModelDefinition(AthleteMorphology(height_m=1.60, mass_kg=50.0))

    assert model_definition.q_size == 5
    assert model_definition.tau_size == 2


def test_biomod_text_contains_segments_and_contact() -> None:
    """The exported model text includes the kinematic chain and platform contact."""

    biomod_text = PlanarJumperModelDefinition().to_biomod_text()

    assert "segment\tleg_foot" in biomod_text
    assert "segment\tthigh" in biomod_text
    assert "segment\ttrunk" in biomod_text
    assert "contact\tplatform_contact" in biomod_text


def test_initial_configuration_aligns_center_of_mass_over_ankle() -> None:
    """The initial posture should place the CoM above the ankle contact."""

    model_definition = PlanarJumperModelDefinition(AthleteMorphology(height_m=1.60, mass_kg=50.0))

    q_init = model_definition.initial_joint_configuration_rad
    center_of_mass_x, _ = model_definition.center_of_mass_position(q_init)

    assert center_of_mass_x == pytest.approx(q_init[0], abs=1e-8)


def test_initial_alignment_keeps_requested_knee_and_hip_flexion() -> None:
    """The CoM alignment should primarily adjust the ankle-equivalent rotation."""

    model_definition = PlanarJumperModelDefinition(AthleteMorphology(height_m=1.60, mass_kg=50.0))

    q_crouched = model_definition.crouched_joint_configuration_rad
    q_aligned = model_definition.initial_joint_configuration_rad

    assert q_aligned[2] != pytest.approx(q_crouched[2])
    assert q_aligned[3] == pytest.approx(-math.radians(100.0))
    assert q_aligned[4] == pytest.approx(math.radians(100.0))


def test_no_platform_model_definition_exposes_three_dofs_and_no_contact() -> None:
    """The simplified no-platform model should keep only the three rotational DoFs."""

    model_definition = PlanarJumperModelDefinition(
        AthleteMorphology(height_m=1.60, mass_kg=50.0),
        floating_base=False,
        include_platform_contact=False,
    )

    biomod_text = model_definition.to_biomod_text()

    assert model_definition.q_size == 3
    assert model_definition.tau_size == 3
    assert "translations\txz" not in biomod_text
    assert "contact\tplatform_contact" not in biomod_text
