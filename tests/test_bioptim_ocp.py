"""Tests for the OCP blueprint layer."""

from __future__ import annotations

import pytest

from synchro_jump.optimization.bioptim_ocp import VerticalJumpBioptimOcpBuilder
from synchro_jump.optimization.problem import VerticalJumpOcpSettings


def test_blueprint_uses_requested_predicted_height_objective() -> None:
    """The first OCP version targets the predicted apex height."""

    builder = VerticalJumpBioptimOcpBuilder()
    blueprint = builder.blueprint(peak_force_newtons=1100.0)

    assert blueprint.objective_name == "MINIMIZE_PREDICTED_COM_HEIGHT"


def test_blueprint_contact_target_ends_with_zero_or_positive_force() -> None:
    """The surrogate contact-force target remains physically non-negative."""

    builder = VerticalJumpBioptimOcpBuilder()
    target = builder.blueprint(peak_force_newtons=900.0).contact_force_target()

    assert len(target) == builder.settings.n_shooting
    assert min(target) >= 0.0


def test_builder_rejects_non_slider_force_values() -> None:
    """The OCP builder stays synchronized with the GUI force slider."""

    builder = VerticalJumpBioptimOcpBuilder(VerticalJumpOcpSettings())

    with pytest.raises(ValueError, match="slider value"):
        builder.blueprint(peak_force_newtons=975.0)
