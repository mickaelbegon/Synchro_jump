"""Tests for the reduced jumper model definition."""

from __future__ import annotations

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
