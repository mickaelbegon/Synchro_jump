"""Tests for the OCP blueprint layer."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from synchro_jump.optimization.bioptim_ocp import (
    VerticalJumpBioptimOcpBuilder,
    _import_bioptim_build_api,
)
from synchro_jump.optimization.problem import (
    CONTACT_MODEL_COMPLIANT_UNILATERAL,
    CONTACT_MODEL_RIGID_UNILATERAL,
    VerticalJumpOcpSettings,
)


def _fake_bioptim_module(**extra_attributes):
    """Return one fake `bioptim` module exposing the minimum build API."""

    module = SimpleNamespace(
        __version__="test",
        BiorbdModel=object,
        BoundsList=object,
        ConstraintFcn=object,
        ConstraintList=object,
        ControlType=object,
        InitialGuessList=object,
        InterpolationType=object,
        Node=object,
        ObjectiveFcn=object,
        ObjectiveList=object,
        OdeSolver=object,
        OptimalControlProgram=object,
        PhaseDynamics=object,
    )
    for key, value in extra_attributes.items():
        setattr(module, key, value)
    return module


def test_blueprint_uses_requested_predicted_height_objective() -> None:
    """The first OCP version targets the predicted apex height."""

    builder = VerticalJumpBioptimOcpBuilder()
    blueprint = builder.blueprint(peak_force_newtons=1100.0)

    assert blueprint.objective_name == "CUSTOM_PREDICTED_COM_HEIGHT"


def test_blueprint_uses_explicit_platform_dynamics_name() -> None:
    """The current OCP scaffold exposes the explicit platform dynamics."""

    builder = VerticalJumpBioptimOcpBuilder()
    blueprint = builder.blueprint(peak_force_newtons=1100.0)

    assert blueprint.dynamics_name == "TORQUE_DRIVEN_WITH_EXPLICIT_PLATFORM_RIGID_CONTACT"
    assert blueprint.contact_model_name == "RIGID_UNILATERAL"


def test_blueprint_can_switch_to_compliant_contact() -> None:
    """The builder should expose the compliant unilateral contact option."""

    builder = VerticalJumpBioptimOcpBuilder(
        VerticalJumpOcpSettings(
            athlete_mass_kg=50.0,
            contact_model=CONTACT_MODEL_COMPLIANT_UNILATERAL,
        )
    )
    blueprint = builder.blueprint(peak_force_newtons=1100.0)

    assert builder.settings.contact_model == CONTACT_MODEL_COMPLIANT_UNILATERAL
    assert blueprint.dynamics_name == "TORQUE_DRIVEN_WITH_EXPLICIT_PLATFORM_COMPLIANT_CONTACT"
    assert blueprint.contact_model_name == "COMPLIANT_UNILATERAL"


def test_blueprint_contact_target_ends_with_zero_or_positive_force() -> None:
    """The surrogate contact-force target remains physically non-negative."""

    builder = VerticalJumpBioptimOcpBuilder()
    target = builder.blueprint(peak_force_newtons=900.0).contact_force_target()

    assert len(target) == builder.settings.n_shooting
    assert min(target) >= 0.0


def test_builder_snaps_force_to_the_discrete_slider_grid() -> None:
    """The OCP builder should accept arbitrary force values and snap them to the grid."""

    builder = VerticalJumpBioptimOcpBuilder(VerticalJumpOcpSettings())
    blueprint = builder.blueprint(peak_force_newtons=975.0)

    assert blueprint.peak_force_newtons in builder.settings.force_slider_values_newtons


def test_import_bioptim_build_api_accepts_legacy_dynamics(monkeypatch) -> None:
    """The compatibility layer should still recognize the historical API."""

    monkeypatch.setitem(sys.modules, "bioptim", _fake_bioptim_module(Dynamics=object))

    api = _import_bioptim_build_api()

    assert api["api_kind"] == "legacy"
    assert api["Dynamics"] is object


def test_import_bioptim_build_api_accepts_modern_dynamics_options(monkeypatch) -> None:
    """The compatibility layer should accept the `bioptim>=3.4` API."""

    monkeypatch.setitem(
        sys.modules,
        "bioptim",
        _fake_bioptim_module(
            DynamicsOptions=object,
            ConfigureVariables=object,
            StateDynamics=object,
        ),
    )

    api = _import_bioptim_build_api()

    assert api["api_kind"] == "modern"
    assert api["DynamicsOptions"] is object


def test_build_ocp_smoke_when_bioptim_is_available(tmp_path: Path) -> None:
    """The explicit-platform OCP can be instantiated end-to-end."""

    pytest.importorskip("bioptim")
    pytest.importorskip("biorbd_casadi")
    from casadi import SX

    builder = VerticalJumpBioptimOcpBuilder(VerticalJumpOcpSettings(athlete_mass_kg=50.0))
    model_path = builder.export_model(tmp_path)

    ocp = builder.build_ocp(peak_force_newtons=1100.0, model_path=model_path)

    assert ocp.n_phases == 1
    assert ocp.cx is SX
    assert "tau_joints" in ocp.nlp[0].controls.keys()
