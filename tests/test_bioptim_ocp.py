"""Tests for the OCP blueprint layer."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from synchro_jump.optimization.bioptim_ocp import (
    VerticalJumpBioptimOcpBuilder,
    _import_bioptim_build_api,
    _shooting_weight_with_excluded_tail,
    _vertical_com_velocity,
)
from synchro_jump.optimization.problem import (
    CONTACT_MODEL_COMPLIANT_UNILATERAL,
    CONTACT_MODEL_NO_PLATFORM,
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
        ObjectiveWeight=object,
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


def test_blueprint_uses_piecewise_constant_controls() -> None:
    """The OCP should expose piecewise-constant torque controls."""

    builder = VerticalJumpBioptimOcpBuilder()
    blueprint = builder.blueprint(peak_force_newtons=1100.0)

    assert blueprint.control_type == "CONSTANT"


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


def test_blueprint_can_switch_to_no_platform_mode() -> None:
    """The builder should expose the simplified no-platform mode."""

    builder = VerticalJumpBioptimOcpBuilder(
        VerticalJumpOcpSettings(
            athlete_mass_kg=50.0,
            contact_model=CONTACT_MODEL_NO_PLATFORM,
        )
    )
    blueprint = builder.blueprint(peak_force_newtons=1100.0)

    assert builder.settings.contact_model == CONTACT_MODEL_NO_PLATFORM
    assert blueprint.dynamics_name == "TORQUE_DRIVEN_NO_PLATFORM"
    assert blueprint.contact_model_name == "NO_PLATFORM"
    assert set(blueprint.contact_force_target()) == {0.0}


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


def test_build_ocp_applies_time_constraint_bounds_when_bioptim_is_available(tmp_path: Path) -> None:
    """The free-final-time bounds should be converted into bounded phase dt."""

    pytest.importorskip("bioptim")
    pytest.importorskip("biorbd_casadi")

    settings = VerticalJumpOcpSettings(
        athlete_mass_kg=50.0,
        contact_model=CONTACT_MODEL_NO_PLATFORM,
    )
    builder = VerticalJumpBioptimOcpBuilder(settings)
    model_path = builder.export_model(tmp_path)

    ocp = builder.build_ocp(peak_force_newtons=1100.0, model_path=model_path)

    expected_dt_min = settings.final_time_lower_bound_s / settings.n_shooting
    expected_dt_max = settings.final_time_upper_bound_s / settings.n_shooting

    assert ocp.dt_parameter_bounds.min[0, 0] == pytest.approx(expected_dt_min)
    assert ocp.dt_parameter_bounds.max[0, 0] == pytest.approx(expected_dt_max)


def test_build_ocp_locks_distal_rotation_torque_in_no_platform_mode(tmp_path: Path) -> None:
    """The simplified no-platform mode should keep the distal rotation passive."""

    pytest.importorskip("bioptim")
    pytest.importorskip("biorbd_casadi")

    settings = VerticalJumpOcpSettings(
        athlete_mass_kg=50.0,
        contact_model=CONTACT_MODEL_NO_PLATFORM,
    )
    builder = VerticalJumpBioptimOcpBuilder(settings)
    model_path = builder.export_model(tmp_path)

    ocp = builder.build_ocp(peak_force_newtons=1100.0, model_path=model_path)

    tau_bounds = ocp.nlp[0].u_bounds["tau_joints"]
    tau_init = ocp.nlp[0].u_init["tau_joints"]

    assert tau_bounds.min[0, 0] == pytest.approx(0.0)
    assert tau_bounds.max[0, 0] == pytest.approx(0.0)
    assert tau_bounds.min[1, 0] == pytest.approx(-750.0)
    assert tau_bounds.max[1, 0] == pytest.approx(750.0)
    assert tau_bounds.min[2, 0] == pytest.approx(-1000.0)
    assert tau_bounds.max[2, 0] == pytest.approx(1000.0)
    assert np.allclose(tau_init.init[0, :], 0.0)


def test_build_ocp_reduces_knee_torque_bounds_with_platform_mode(tmp_path: Path) -> None:
    """The knee and hip torque bounds should scale with the athlete mass."""

    pytest.importorskip("bioptim")
    pytest.importorskip("biorbd_casadi")

    settings = VerticalJumpOcpSettings(athlete_mass_kg=50.0)
    builder = VerticalJumpBioptimOcpBuilder(settings)
    model_path = builder.export_model(tmp_path)

    ocp = builder.build_ocp(peak_force_newtons=1100.0, model_path=model_path)

    tau_bounds = ocp.nlp[0].u_bounds["tau_joints"]

    assert tau_bounds.min[0, 0] == pytest.approx(-750.0)
    assert tau_bounds.max[0, 0] == pytest.approx(750.0)
    assert tau_bounds.min[1, 0] == pytest.approx(-1000.0)
    assert tau_bounds.max[1, 0] == pytest.approx(1000.0)


def test_shooting_weight_with_excluded_tail_softens_the_last_three_nodes() -> None:
    """The torque regularization selector should keep a small penalty on the last nodes."""

    weights = _shooting_weight_with_excluded_tail(10, 3, 0.2)

    assert np.allclose(weights[:7], 1.0)
    assert np.allclose(weights[7:], 0.2)


def test_aligned_initial_configuration_nulls_com_x_when_bioptim_is_available(tmp_path: Path) -> None:
    """The exported-model Jacobian alignment should bring the CoM horizontal position to zero."""

    pytest.importorskip("bioptim")
    pytest.importorskip("biorbd_casadi")
    from casadi import DM
    from bioptim import BiorbdModel
    settings = VerticalJumpOcpSettings(
        athlete_mass_kg=50.0,
        contact_model=CONTACT_MODEL_NO_PLATFORM,
    )
    builder = VerticalJumpBioptimOcpBuilder(settings)
    model_path = builder.export_model(tmp_path)

    aligned_q = np.asarray(builder.aligned_initial_joint_configuration_rad(model_path=model_path), dtype=float)
    com = np.asarray(
        BiorbdModel(str(model_path)).center_of_mass()(DM(aligned_q.reshape((-1, 1))), DM()),
        dtype=float,
    ).reshape((-1,))

    assert com[0] == pytest.approx(0.0, abs=1e-8)


def test_aligned_initial_configuration_keeps_knee_and_hip_flexion_when_bioptim_is_available(tmp_path: Path) -> None:
    """The exported-model alignment should only adjust the ankle-equivalent angle."""

    pytest.importorskip("bioptim")
    pytest.importorskip("biorbd_casadi")

    settings = VerticalJumpOcpSettings(
        athlete_mass_kg=50.0,
        contact_model=CONTACT_MODEL_NO_PLATFORM,
    )
    builder = VerticalJumpBioptimOcpBuilder(settings)
    model_path = builder.export_model(tmp_path)

    aligned_q = np.asarray(builder.aligned_initial_joint_configuration_rad(model_path=model_path), dtype=float)

    assert aligned_q[1] == pytest.approx(-np.deg2rad(100.0))
    assert aligned_q[2] == pytest.approx(np.deg2rad(100.0))




def test_vertical_com_velocity_returns_the_vertical_component() -> None:
    """The custom CoM-velocity helper should expose the vertical component only."""

    controller = SimpleNamespace(
        model=SimpleNamespace(
            center_of_mass_velocity=lambda: (
                lambda _q, _qdot, _params: np.array([[1.2], [-0.4], [3.6]])
            )
        ),
        parameters=SimpleNamespace(cx=np.array([])),
        cx=np.array([]),
        states={
            "q": SimpleNamespace(cx=np.zeros((3, 1))),
            "qdot": SimpleNamespace(cx=np.zeros((3, 1))),
        },
        controls={
            "tau": SimpleNamespace(cx=np.zeros((3, 1))),
        },
    )

    assert float(_vertical_com_velocity(controller)) == pytest.approx(3.6)
