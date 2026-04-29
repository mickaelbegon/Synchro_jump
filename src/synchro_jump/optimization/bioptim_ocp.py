"""`bioptim` OCP builder for the reduced planar vertical jump."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path

import numpy as np

from synchro_jump.modeling import AthleteMorphology, PlanarJumperModelDefinition
from synchro_jump.optimization.contact import PlatformInteractionModel
from synchro_jump.optimization.force_profile import PlatformForceProfile
from synchro_jump.optimization.initial_guess import (
    build_linear_inverse_dynamics_initial_guess,
    static_equilibrium_torque,
)
from synchro_jump.optimization.problem import (
    CONTACT_MODEL_COMPLIANT_UNILATERAL,
    CONTACT_MODEL_NO_PLATFORM,
    CONTACT_MODEL_RIGID_UNILATERAL,
    VerticalJumpOcpSettings,
    snap_to_discrete_value,
)


@dataclass(frozen=True)
class VerticalJumpOcpBlueprint:
    """High-level OCP description independent of optional runtime dependencies."""

    settings: VerticalJumpOcpSettings
    peak_force_newtons: float
    objective_name: str = "CUSTOM_PREDICTED_COM_HEIGHT"
    dynamics_name: str = "TORQUE_DRIVEN_WITH_EXPLICIT_PLATFORM"
    contact_name: str = "platform_contact"
    contact_model_name: str = ""
    ode_solver_name: str = "RK4"
    control_type: str = "CONSTANT"

    def contact_force_target(self, final_time_guess: float | None = None) -> tuple[float, ...]:
        """Return one surrogate contact-force target used by the GUI."""

        duration = final_time_guess or self.settings.final_time_upper_bound_s
        if self.settings.contact_model == CONTACT_MODEL_NO_PLATFORM:
            return tuple(0.0 for _ in range(self.settings.n_shooting))
        profile = PlatformForceProfile(
            peak_force_newtons=self.peak_force_newtons,
            total_duration=duration,
        )
        interaction = PlatformInteractionModel(
            platform_mass_kg=self.settings.platform_mass_kg,
            gravity=9.81,
            contact_stiffness_n_per_m=self.settings.contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=self.settings.contact_damping_n_s_per_m,
        )
        initial_q = _model_definition_from_settings(self.settings).initial_joint_configuration_rad
        targets = []
        for node_index in range(self.settings.n_shooting):
            time = duration * node_index / max(self.settings.n_shooting - 1, 1)
            surrogate_contact = _contact_force_target_from_interaction(
                interaction,
                self.settings,
                q_root_z_guess=initial_q[1],
                qdot_root_z_guess=0.0,
                platform_force=profile.force_at(time),
            )
            targets.append(max(surrogate_contact, 0.0))
        return tuple(targets)


def _uses_platform_states(contact_model: str) -> bool:
    """Return whether the selected mode requires explicit platform states."""

    return contact_model in (CONTACT_MODEL_RIGID_UNILATERAL, CONTACT_MODEL_COMPLIANT_UNILATERAL)


def _floating_base_for_contact_model(contact_model: str) -> bool:
    """Return whether the selected mode uses the floating-base jumper model."""

    return contact_model != CONTACT_MODEL_NO_PLATFORM


def _model_definition_from_settings(settings: VerticalJumpOcpSettings) -> PlanarJumperModelDefinition:
    """Return the jumper model definition matching the selected contact mode."""

    return PlanarJumperModelDefinition(
        morphology=AthleteMorphology(
            height_m=settings.athlete_height_m,
            mass_kg=settings.athlete_mass_kg,
        ),
        floating_base=_floating_base_for_contact_model(settings.contact_model),
        include_platform_contact=settings.contact_model != CONTACT_MODEL_NO_PLATFORM,
    )


def _knee_control_index(settings: VerticalJumpOcpSettings) -> int:
    """Return the index of the knee torque within `tau_joints`."""

    return 1 if settings.contact_model == CONTACT_MODEL_NO_PLATFORM else 0


def _hip_control_index(settings: VerticalJumpOcpSettings) -> int:
    """Return the index of the hip torque within `tau_joints`."""

    return 2 if settings.contact_model == CONTACT_MODEL_NO_PLATFORM else 1


def _import_bioptim_build_api():
    """Return the `bioptim` runtime symbols required by the current OCP builder."""

    import bioptim

    common_names = (
        "BiorbdModel",
        "BoundsList",
        "ConstraintFcn",
        "ConstraintList",
        "ControlType",
        "InitialGuessList",
        "InterpolationType",
        "Node",
        "ObjectiveFcn",
        "ObjectiveList",
        "ObjectiveWeight",
        "OdeSolver",
        "OptimalControlProgram",
        "PhaseDynamics",
    )
    missing_names = [name for name in common_names if not hasattr(bioptim, name)]
    if missing_names:
        version = getattr(bioptim, "__version__", "unknown")
        raise RuntimeError(
            "Version de bioptim non supportee pour cet OCP explicite: "
            f"{version}. Symboles manquants: {', '.join(missing_names)}. "
            "Installe une version compatible de l'environnement Conda du projet."
        )

    api = {name: getattr(bioptim, name) for name in common_names}
    if hasattr(bioptim, "Dynamics"):
        api["api_kind"] = "legacy"
        api["Dynamics"] = bioptim.Dynamics
        return api

    if hasattr(bioptim, "DynamicsOptions"):
        modern_names = ("DynamicsOptions", "ConfigureVariables", "StateDynamics")
        missing_modern_names = [name for name in modern_names if not hasattr(bioptim, name)]
        if missing_modern_names:
            version = getattr(bioptim, "__version__", "unknown")
            raise RuntimeError(
                "Version de bioptim non supportee pour cet OCP explicite: "
                f"{version}. Symboles manquants: {', '.join(missing_modern_names)}."
            )
        api["api_kind"] = "modern"
        api["DynamicsOptions"] = bioptim.DynamicsOptions
        api["ConfigureVariables"] = bioptim.ConfigureVariables
        api["StateDynamics"] = bioptim.StateDynamics
        return api

    version = getattr(bioptim, "__version__", "unknown")
    raise RuntimeError(
        "Version de bioptim non supportee pour cet OCP explicite: "
        f"{version}. Symboles manquants: Dynamics ou DynamicsOptions."
    )


def _symbolic_positive_part(value):
    """Return the positive part of one CasADi or numeric value."""

    from casadi import if_else

    return if_else(value > 0, value, 0)


def _constant_bounds_with_fixed_start(lower_bounds, upper_bounds, start_values):
    """Build three-column bounds with a fixed initial node."""

    import numpy as np

    lower = np.asarray(lower_bounds, dtype=float).reshape((-1, 1))
    upper = np.asarray(upper_bounds, dtype=float).reshape((-1, 1))
    start = np.asarray(start_values, dtype=float).reshape((-1, 1))
    return np.hstack((start, lower, lower)), np.hstack((start, upper, upper))


def _shooting_weight_with_excluded_tail(
    n_shooting: int,
    excluded_tail_nodes: int,
    tail_weight: float = 0.0,
):
    """Return one per-shooting-node weight vector with a softened tail."""

    if n_shooting <= 0:
        raise ValueError("n_shooting must be strictly positive")
    if excluded_tail_nodes < 0:
        raise ValueError("excluded_tail_nodes must stay non-negative")
    if excluded_tail_nodes >= n_shooting:
        raise ValueError("excluded_tail_nodes must stay below n_shooting")
    if not (0.0 <= tail_weight <= 1.0):
        raise ValueError("tail_weight must stay within [0, 1]")

    weights = np.ones((n_shooting,), dtype=float)
    if excluded_tail_nodes:
        weights[-excluded_tail_nodes:] = tail_weight
    return weights


def _static_control_target_for_first_interval(
    model,
    initial_q,
    settings: VerticalJumpOcpSettings,
) -> np.ndarray:
    """Return the first control target enforcing static equilibrium at the initial posture."""

    full_static_torque = static_equilibrium_torque(model, np.asarray(initial_q, dtype=float))
    target = np.asarray(full_static_torque[model.nb_root :], dtype=float).reshape((-1, 1))
    if settings.contact_model == CONTACT_MODEL_NO_PLATFORM and target.shape[0] > 0:
        target[0, 0] = 0.0
    return target


def _align_configuration_to_zero_com_x(
    model,
    initial_q,
    *,
    tolerance: float = 1e-10,
    max_iterations: int = 25,
) -> np.ndarray:
    """Align one configuration so that the model CoM horizontal position reaches zero."""

    from casadi import DM, Function, SX, jacobian

    q_values = np.asarray(initial_q, dtype=float).reshape((-1, 1)).copy()
    q_symbol = SX.sym("q_align", model.nb_q, 1)
    parameters = SX.zeros(0, 1)
    com_x_expression = model.center_of_mass()(q_symbol, parameters)[0]
    com_x_jacobian_expression = jacobian(com_x_expression, q_symbol)
    com_x_and_jacobian = Function(
        "com_x_and_jacobian",
        [q_symbol],
        [com_x_expression, com_x_jacobian_expression],
    )

    for _ in range(max_iterations):
        com_x_value, jacobian_value = com_x_and_jacobian(DM(q_values))
        horizontal_error = float(np.asarray(com_x_value, dtype=float).reshape((-1,))[0])
        if abs(horizontal_error) <= tolerance:
            break

        jacobian_row = np.asarray(jacobian_value, dtype=float).reshape((1, -1))
        jacobian_norm_sq = float((jacobian_row @ jacobian_row.T).reshape((-1,))[0])
        if jacobian_norm_sq <= 1e-12:
            break

        pseudo_inverse = jacobian_row.T / jacobian_norm_sq
        q_values -= pseudo_inverse * horizontal_error

    return q_values.reshape((-1,))


def _contact_index_from_name(model, contact_name: str) -> int:
    """Resolve a rigid-contact index without relying on optional helpers."""

    contact_names = list(_model_contact_names(model))
    if contact_name in contact_names:
        return contact_names.index(contact_name)

    axis_matches = [index for index, name in enumerate(contact_names) if name.startswith(f"{contact_name}_")]
    if len(axis_matches) == 1:
        return axis_matches[0]

    raise ValueError(f"Unknown contact name: {contact_name}")


def _model_dof_names(model) -> tuple[str, ...]:
    """Return the model DoF names across supported `bioptim` APIs."""

    if hasattr(model, "name_dofs"):
        return tuple(model.name_dofs)
    if hasattr(model, "name_dof"):
        return tuple(model.name_dof)
    raise AttributeError("The provided model does not expose degree-of-freedom names")


def _model_contact_names(model) -> tuple[str, ...]:
    """Return the rigid-contact names across supported `bioptim` APIs."""

    if hasattr(model, "contact_names"):
        return tuple(model.contact_names)
    if hasattr(model, "rigid_contact_names"):
        return tuple(model.rigid_contact_names)
    raise AttributeError("The provided model does not expose rigid-contact names")


def _model_contact_axis(model, contact_index: int) -> int:
    """Return the unique vertical rigid-contact axis across supported APIs."""

    if hasattr(model, "rigid_contact_index"):
        return int(model.rigid_contact_index(contact_index)[0])
    if hasattr(model, "rigid_contact_axes_index"):
        return int(model.rigid_contact_axes_index(contact_index)[0])
    raise AttributeError("The provided model does not expose rigid-contact axes")


def _symbolic_platform_force(time, peak_force_newtons: float, total_duration_s: float, taper_duration_s: float):
    """Return a CasADi-compatible piecewise-linear actuation profile."""

    from casadi import if_else

    ramp_start = total_duration_s - taper_duration_s
    final_force = 0.5 * peak_force_newtons
    ramp_value = peak_force_newtons * (1.0 - 0.5 * (time - ramp_start) / taper_duration_s)
    return if_else(
        time <= 0,
        peak_force_newtons,
        if_else(
            time <= ramp_start,
            peak_force_newtons,
            if_else(time >= total_duration_s, final_force, ramp_value),
        ),
    )


def _split_q_vectors(nlp, states, controls):
    """Extract the split generalized coordinates, velocities, and controls."""

    from bioptim import DynamicsFunctions
    from casadi import vertcat

    q_roots = (
        DynamicsFunctions.get(nlp.states["q_roots"], states)
        if "q_roots" in nlp.states
        else nlp.cx.zeros(nlp.model.nb_root, 1)
    )
    q_joints = (
        DynamicsFunctions.get(nlp.states["q_joints"], states)
        if "q_joints" in nlp.states
        else DynamicsFunctions.get(nlp.states["q"], states)
    )
    qdot_roots = (
        DynamicsFunctions.get(nlp.states["qdot_roots"], states)
        if "qdot_roots" in nlp.states
        else nlp.cx.zeros(nlp.model.nb_root, 1)
    )
    qdot_joints = (
        DynamicsFunctions.get(nlp.states["qdot_joints"], states)
        if "qdot_joints" in nlp.states
        else DynamicsFunctions.get(nlp.states["qdot"], states)
    )
    tau_joints = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)
    platform_position = DynamicsFunctions.get(nlp.states["platform_position"], states)
    platform_velocity = DynamicsFunctions.get(nlp.states["platform_velocity"], states)
    q = vertcat(q_roots, q_joints)
    qdot = vertcat(qdot_roots, qdot_joints)
    return q, qdot, tau_joints, platform_position, platform_velocity


def _controller_q_qdot_tau(controller):
    """Return full generalized coordinates, velocities, and torques from one controller."""

    from casadi import vertcat

    states = controller.states
    controls = controller.controls
    cx = controller.cx
    model = controller.model

    if "q" in states:
        q = states["q"].cx
    else:
        q_roots = states["q_roots"].cx if "q_roots" in states else cx.zeros(model.nb_root, 1)
        q = vertcat(q_roots, states["q_joints"].cx)

    if "qdot" in states:
        qdot = states["qdot"].cx
    else:
        qdot_roots = states["qdot_roots"].cx if "qdot_roots" in states else cx.zeros(model.nb_root, 1)
        qdot = vertcat(qdot_roots, states["qdot_joints"].cx)

    if "tau" in controls:
        tau = controls["tau"].cx
    else:
        tau = vertcat(cx.zeros(model.nb_root, 1), controls["tau_joints"].cx)

    return q, qdot, tau


def _symbolic_compliant_contact_force(
    q,
    qdot,
    *,
    platform_position,
    platform_velocity,
    contact_stiffness_n_per_m: float,
    contact_damping_n_s_per_m: float,
):
    """Return the compliant contact force between the platform and the foot."""

    foot_position = q[1]
    foot_velocity = qdot[1]
    compression = _symbolic_positive_part(platform_position - foot_position)
    closing_speed = _symbolic_positive_part(platform_velocity - foot_velocity)
    return _symbolic_positive_part(
        contact_stiffness_n_per_m * compression + contact_damping_n_s_per_m * closing_speed
    )


def _add_platform_force_to_generalized_force(generalized_force, platform_force_newtons):
    """Inject the platform actuation into the vertical root translation DoF."""

    if generalized_force.shape[0] > 1:
        generalized_force[1] = generalized_force[1] + platform_force_newtons
    return generalized_force


def _coupled_platform_dynamics_symbolic(
    model,
    q,
    qdot,
    tau_joints,
    parameters,
    *,
    contact_index: int,
    platform_force_newtons,
    platform_mass_kg: float,
    gravity: float,
    cx_type,
):
    """Solve the rigid coupled jumper-platform dynamics symbolically."""

    from casadi import horzcat, jacobian, solve, substitute, vertcat

    nq = q.shape[0]
    tau = vertcat(cx_type.zeros(model.nb_root, 1), tau_joints)
    tau = _add_platform_force_to_generalized_force(tau, platform_force_newtons)
    zero_qddot = cx_type.zeros(nq, 1)
    qddot_symbol = cx_type.sym("qddot_contact", nq, 1)
    contact_axis = _model_contact_axis(model, contact_index)
    contact_acceleration = model.rigid_contact_acceleration(contact_index, contact_axis)
    contact_acceleration_expression = contact_acceleration(q, qdot, qddot_symbol, parameters)
    contact_jacobian = substitute(jacobian(contact_acceleration_expression, qddot_symbol), qddot_symbol, zero_qddot)
    contact_bias = contact_acceleration(q, qdot, zero_qddot, parameters)
    mass_matrix = model.mass_matrix()(q, parameters)
    nonlinear_effects = model.non_linear_effects()(q, qdot, parameters)

    augmented_matrix = vertcat(
        horzcat(mass_matrix, -contact_jacobian.T, cx_type.zeros(nq, 1)),
        horzcat(contact_jacobian, cx_type.zeros(1, 1), -cx_type.ones(1, 1)),
        horzcat(cx_type.zeros(1, nq), cx_type.ones(1, 1), platform_mass_kg * cx_type.ones(1, 1)),
    )
    rhs = vertcat(
        tau - nonlinear_effects,
        -contact_bias,
        platform_force_newtons - platform_mass_kg * gravity,
    )
    solution = solve(augmented_matrix, rhs)
    qddot = solution[:nq]
    contact_force = solution[nq]
    platform_acceleration = solution[nq + 1]
    return qddot, contact_force, platform_acceleration


def _predicted_apex_height(controller, gravity: float = 9.81):
    """Custom Mayer objective based on CoM height and vertical velocity."""

    q, qdot, _ = _controller_q_qdot_tau(controller)
    com = controller.model.center_of_mass()(q, controller.parameters.cx)
    vertical_velocity = _symbolic_positive_part(_vertical_com_velocity(controller))
    return -(com[2] + vertical_velocity**2 / (2.0 * gravity))


_predicted_apex_height.__name__ = "predicted_apex_height"


def _vertical_com_velocity(controller):
    """Return the vertical CoM velocity component."""

    q, qdot, _ = _controller_q_qdot_tau(controller)
    com_velocity = controller.model.center_of_mass_velocity()(q, qdot, controller.parameters.cx)
    if getattr(com_velocity, "shape", None) == (3, 1):
        return com_velocity[2, 0]
    return com_velocity[2]


_vertical_com_velocity.__name__ = "vertical_com_velocity"


def _final_com_anteroposterior_velocity_squared(controller):
    """Return the squared anteroposterior CoM velocity at the final node."""

    q, qdot, _ = _controller_q_qdot_tau(controller)
    com_velocity = controller.model.center_of_mass_velocity()(q, qdot, controller.parameters.cx)
    return com_velocity[0] ** 2


_final_com_anteroposterior_velocity_squared.__name__ = "final_com_anteroposterior_velocity_squared"


def _final_extension_error_squared(controller):
    """Return the squared distance to the fully extended rotational posture."""

    from casadi import sumsqr

    q, _, _ = _controller_q_qdot_tau(controller)
    rotational_q = q[2:] if q.shape[0] > 3 else q
    return sumsqr(rotational_q)


_final_extension_error_squared.__name__ = "final_extension_error_squared"


def _sagittal_angular_momentum(controller):
    """Return the sagittal-plane angular momentum component."""

    q, qdot, _ = _controller_q_qdot_tau(controller)
    return controller.model.angular_momentum()(q, qdot, controller.parameters.cx)[1]


_sagittal_angular_momentum.__name__ = "sagittal_angular_momentum"


def _no_platform_external_vertical_force(
    controller,
    *,
    athlete_mass_kg: float,
    gravity: float,
):
    """Return the equivalent external vertical force in the no-platform mode."""

    from casadi import solve

    q, qdot, tau = _controller_q_qdot_tau(controller)
    qddot = solve(
        controller.model.mass_matrix()(q, controller.parameters.cx),
        tau - controller.model.non_linear_effects()(q, qdot, controller.parameters.cx),
    )
    com_acceleration = controller.model.center_of_mass_acceleration()(q, qdot, qddot, controller.parameters.cx)
    return athlete_mass_kg * (com_acceleration[2] + gravity)


_no_platform_external_vertical_force.__name__ = "no_platform_external_vertical_force"


def _contact_force_penalty(
    controller,
    contact_model: str,
    contact_name: str,
    peak_force_newtons: float,
    total_duration_s: float,
    taper_duration_s: float,
    platform_mass_kg: float,
    athlete_mass_kg: float,
    gravity: float,
    contact_stiffness_n_per_m: float,
    contact_damping_n_s_per_m: float,
):
    """Return the selected contact force used in custom constraints."""

    if contact_model == CONTACT_MODEL_RIGID_UNILATERAL:
        q, qdot, tau = _controller_q_qdot_tau(controller)
        _, contact_force, _ = _coupled_platform_dynamics_symbolic(
            controller.model,
            q,
            qdot,
            tau[controller.model.nb_root :],
            controller.parameters.cx,
            contact_index=_contact_index_from_name(controller.model, contact_name),
            platform_force_newtons=_symbolic_platform_force(
                controller.time.cx,
                peak_force_newtons=peak_force_newtons,
                total_duration_s=total_duration_s,
                taper_duration_s=taper_duration_s,
            ),
            platform_mass_kg=platform_mass_kg,
            gravity=gravity,
            cx_type=controller.cx,
        )
        return contact_force

    if contact_model == CONTACT_MODEL_COMPLIANT_UNILATERAL:
        return _symbolic_compliant_contact_force(
            controller.q,
            controller.qdot,
            platform_position=controller.states["platform_position"].cx,
            platform_velocity=controller.states["platform_velocity"].cx,
            contact_stiffness_n_per_m=contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=contact_damping_n_s_per_m,
        )

    if contact_model == CONTACT_MODEL_NO_PLATFORM:
        return _no_platform_external_vertical_force(
            controller,
            athlete_mass_kg=athlete_mass_kg,
            gravity=gravity,
        )

    raise ValueError(f"Unsupported contact model: {contact_model}")


_contact_force_penalty.__name__ = "contact_force_penalty"


def _contact_force_target_from_interaction(
    interaction: PlatformInteractionModel,
    settings: VerticalJumpOcpSettings,
    q_root_z_guess: float,
    qdot_root_z_guess: float,
    platform_force: float,
    *,
    platform_position_guess: float = 0.0,
    platform_velocity_guess: float = 0.0,
) -> float:
    """Return one GUI surrogate contact force for the selected contact mode."""

    if settings.contact_model == CONTACT_MODEL_RIGID_UNILATERAL:
        surrogate_contact = interaction.contact_force(
            platform_actuation_force_newtons=platform_force,
            platform_vertical_acceleration=0.0,
        )
        return max(surrogate_contact, 0.0)

    if settings.contact_model == CONTACT_MODEL_COMPLIANT_UNILATERAL:
        return interaction.compliant_contact_force(
            platform_position_m=platform_position_guess,
            platform_velocity_m_s=platform_velocity_guess,
            foot_position_m=q_root_z_guess,
            foot_velocity_m_s=qdot_root_z_guess,
        )

    raise ValueError(f"Unsupported contact model: {settings.contact_model}")


def _contact_model_dynamics_name(contact_model: str) -> str:
    """Return a user-facing dynamic name for one contact model."""

    if contact_model == CONTACT_MODEL_RIGID_UNILATERAL:
        return "TORQUE_DRIVEN_WITH_EXPLICIT_PLATFORM_RIGID_CONTACT"
    if contact_model == CONTACT_MODEL_COMPLIANT_UNILATERAL:
        return "TORQUE_DRIVEN_WITH_EXPLICIT_PLATFORM_COMPLIANT_CONTACT"
    if contact_model == CONTACT_MODEL_NO_PLATFORM:
        return "TORQUE_DRIVEN_NO_PLATFORM"
    raise ValueError(f"Unsupported contact model: {contact_model}")


def _contact_model_label(contact_model: str) -> str:
    """Return a compact user-facing label for one contact model."""

    if contact_model == CONTACT_MODEL_RIGID_UNILATERAL:
        return "RIGID_UNILATERAL"
    if contact_model == CONTACT_MODEL_COMPLIANT_UNILATERAL:
        return "COMPLIANT_UNILATERAL"
    if contact_model == CONTACT_MODEL_NO_PLATFORM:
        return "NO_PLATFORM"
    raise ValueError(f"Unsupported contact model: {contact_model}")


def _configure_explicit_platform_dynamics(
    ocp,
    nlp,
    *,
    peak_force_newtons: float,
    total_duration_s: float,
    taper_duration_s: float,
    platform_mass_kg: float,
    gravity: float,
    contact_model: str,
    contact_name: str,
    contact_stiffness_n_per_m: float,
    contact_damping_n_s_per_m: float,
    numerical_data_timeseries=None,
    contact_type=(),
    **_,
):
    """Configure the explicit platform dynamics with split root/joint states."""

    from bioptim import ConfigureProblem

    _ = numerical_data_timeseries
    _ = contact_type

    name_dof = list(_model_dof_names(nlp.model))
    name_q_roots = name_dof[: nlp.model.nb_root]
    name_q_joints = name_dof[nlp.model.nb_root :]

    if name_q_roots:
        ConfigureProblem.configure_new_variable("q_roots", name_q_roots, ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_new_variable("q_joints", name_q_joints, ocp, nlp, as_states=True, as_controls=False)
    if name_q_roots:
        ConfigureProblem.configure_new_variable("qdot_roots", name_q_roots, ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_new_variable("qdot_joints", name_q_joints, ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_new_variable(
        "platform_position", ["z_platform"], ocp, nlp, as_states=True, as_controls=False
    )
    ConfigureProblem.configure_new_variable(
        "platform_velocity", ["zdot_platform"], ocp, nlp, as_states=True, as_controls=False
    )
    ConfigureProblem.configure_new_variable(
        "tau_joints", name_q_joints, ocp, nlp, as_states=False, as_controls=True
    )
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        _explicit_platform_dynamics,
        peak_force_newtons=peak_force_newtons,
        total_duration_s=total_duration_s,
        taper_duration_s=taper_duration_s,
        platform_mass_kg=platform_mass_kg,
        gravity=gravity,
        contact_model=contact_model,
        contact_name=contact_name,
        contact_stiffness_n_per_m=contact_stiffness_n_per_m,
        contact_damping_n_s_per_m=contact_damping_n_s_per_m,
    )


def _configure_explicit_platform_states(ConfigureVariables, ocp, nlp) -> None:
    """Configure the split explicit-platform state variables for `bioptim>=3.4`."""

    name_dof = list(_model_dof_names(nlp.model))
    name_q_roots = name_dof[: nlp.model.nb_root]
    name_q_joints = name_dof[nlp.model.nb_root :]
    if name_q_roots:
        ConfigureVariables.configure_new_variable("q_roots", name_q_roots, ocp, nlp, as_states=True, as_controls=False)
    ConfigureVariables.configure_new_variable("q_joints", name_q_joints, ocp, nlp, as_states=True, as_controls=False)
    if name_q_roots:
        ConfigureVariables.configure_new_variable(
            "qdot_roots",
            name_q_roots,
            ocp,
            nlp,
            as_states=True,
            as_controls=False,
        )
    ConfigureVariables.configure_new_variable("qdot_joints", name_q_joints, ocp, nlp, as_states=True, as_controls=False)
    ConfigureVariables.configure_new_variable(
        "platform_position",
        ["z_platform"],
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
    )
    ConfigureVariables.configure_new_variable(
        "platform_velocity",
        ["zdot_platform"],
        ocp,
        nlp,
        as_states=True,
        as_controls=False,
    )


def _configure_explicit_platform_controls(ConfigureVariables, ocp, nlp) -> None:
    """Configure the explicit-platform controls for `bioptim>=3.4`."""

    name_dof = list(_model_dof_names(nlp.model))
    name_q_joints = name_dof[nlp.model.nb_root :]
    ConfigureVariables.configure_new_variable(
        "tau_joints",
        name_q_joints,
        ocp,
        nlp,
        as_states=False,
        as_controls=True,
    )


def _configure_no_platform_dynamics(
    ocp,
    nlp,
    **_,
):
    """Configure the reduced torque-driven dynamics without explicit platform states."""

    from bioptim import ConfigureProblem

    name_dof = list(_model_dof_names(nlp.model))
    ConfigureProblem.configure_new_variable("q_joints", name_dof, ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_new_variable("qdot_joints", name_dof, ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_new_variable("tau_joints", name_dof, ocp, nlp, as_states=False, as_controls=True)
    ConfigureProblem.configure_dynamics_function(
        ocp,
        nlp,
        _no_platform_dynamics,
    )


def _configure_no_platform_states(ConfigureVariables, ocp, nlp) -> None:
    """Configure the reduced no-platform state variables for `bioptim>=3.4`."""

    name_dof = list(_model_dof_names(nlp.model))
    ConfigureVariables.configure_new_variable("q_joints", name_dof, ocp, nlp, as_states=True, as_controls=False)
    ConfigureVariables.configure_new_variable("qdot_joints", name_dof, ocp, nlp, as_states=True, as_controls=False)


def _configure_no_platform_controls(ConfigureVariables, ocp, nlp) -> None:
    """Configure the reduced no-platform controls for `bioptim>=3.4`."""

    name_dof = list(_model_dof_names(nlp.model))
    ConfigureVariables.configure_new_variable("tau_joints", name_dof, ocp, nlp, as_states=False, as_controls=True)


def _make_explicit_platform_model_class(BiorbdModel, StateDynamics, ConfigureVariables):
    """Return one `bioptim>=3.4` custom model exposing the explicit platform dynamics."""

    class ExplicitPlatformBiorbdModel(BiorbdModel, StateDynamics):
        """Custom `BiorbdModel` carrying the explicit moving-platform dynamics."""

        @property
        def state_configuration_functions(self):
            return [lambda ocp, nlp: _configure_explicit_platform_states(ConfigureVariables, ocp, nlp)]

        @property
        def control_configuration_functions(self):
            return [lambda ocp, nlp: _configure_explicit_platform_controls(ConfigureVariables, ocp, nlp)]

        @property
        def algebraic_configuration_functions(self):
            return []

        @property
        def extra_configuration_functions(self):
            return []

        @staticmethod
        def dynamics(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            **extra_parameters,
        ):
            return _explicit_platform_dynamics(
                time,
                states,
                controls,
                parameters,
                algebraic_states,
                numerical_timeseries,
                nlp,
                **extra_parameters,
            )

    return ExplicitPlatformBiorbdModel


def _make_no_platform_model_class(BiorbdModel, StateDynamics, ConfigureVariables):
    """Return one `bioptim>=3.4` custom model exposing the reduced no-platform dynamics."""

    class NoPlatformBiorbdModel(BiorbdModel, StateDynamics):
        """Custom `BiorbdModel` carrying the simplified torque-driven dynamics."""

        @property
        def state_configuration_functions(self):
            return [lambda ocp, nlp: _configure_no_platform_states(ConfigureVariables, ocp, nlp)]

        @property
        def control_configuration_functions(self):
            return [lambda ocp, nlp: _configure_no_platform_controls(ConfigureVariables, ocp, nlp)]

        @property
        def algebraic_configuration_functions(self):
            return []

        @property
        def extra_configuration_functions(self):
            return []

        @staticmethod
        def dynamics(
            time,
            states,
            controls,
            parameters,
            algebraic_states,
            numerical_timeseries,
            nlp,
            **extra_parameters,
        ):
            return _no_platform_dynamics(
                time,
                states,
                controls,
                parameters,
                algebraic_states,
                numerical_timeseries,
                nlp,
                **extra_parameters,
            )

    return NoPlatformBiorbdModel


def _instantiate_ocp(OptimalControlProgram, api_kind: str, bio_model, dynamics, n_shooting: int, phase_time: float, **kwargs):
    """Instantiate the OCP with the constructor layout matching the installed `bioptim`."""

    if api_kind == "legacy":
        return OptimalControlProgram(bio_model, dynamics, n_shooting, phase_time, **kwargs)
    if api_kind == "modern":
        return OptimalControlProgram(bio_model, n_shooting, phase_time, dynamics=dynamics, **kwargs)

    signature = inspect.signature(OptimalControlProgram.__init__)
    parameter_order = tuple(signature.parameters)
    raise RuntimeError(f"Unsupported OptimalControlProgram signature: {parameter_order}")


def _explicit_platform_dynamics(
    time,
    states,
    controls,
    parameters,
    algebraic_states,
    numerical_timeseries,
    nlp,
    *,
    peak_force_newtons: float,
    total_duration_s: float,
    taper_duration_s: float,
    platform_mass_kg: float,
    gravity: float,
    contact_model: str,
    contact_name: str,
    contact_stiffness_n_per_m: float,
    contact_damping_n_s_per_m: float,
):
    """Custom dynamics for the explicit moving platform."""

    from bioptim import DynamicsEvaluation, DynamicsFunctions
    from casadi import solve, vertcat

    _ = algebraic_states
    _ = numerical_timeseries

    q, qdot, tau_joints, platform_position, platform_velocity = _split_q_vectors(nlp, states, controls)
    if nlp.model.nb_root:
        dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    else:
        dq = nlp.model.reshape_qdot()(q, qdot, parameters)
    platform_force_newtons = _symbolic_platform_force(
        time,
        peak_force_newtons=peak_force_newtons,
        total_duration_s=total_duration_s,
        taper_duration_s=taper_duration_s,
    )

    if contact_model == CONTACT_MODEL_RIGID_UNILATERAL:
        qddot, _, platform_acceleration = _coupled_platform_dynamics_symbolic(
            nlp.model,
            q,
            qdot,
            tau_joints,
            parameters,
            contact_index=_contact_index_from_name(nlp.model, contact_name),
            platform_force_newtons=platform_force_newtons,
            platform_mass_kg=platform_mass_kg,
            gravity=gravity,
            cx_type=nlp.cx,
        )
    elif contact_model == CONTACT_MODEL_COMPLIANT_UNILATERAL:
        contact_force = _symbolic_compliant_contact_force(
            q,
            qdot,
            platform_position=platform_position,
            platform_velocity=platform_velocity,
            contact_stiffness_n_per_m=contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=contact_damping_n_s_per_m,
        )
        tau = vertcat(nlp.cx.zeros(nlp.model.nb_root, 1), tau_joints)
        tau = _add_platform_force_to_generalized_force(tau, platform_force_newtons)
        contact_generalized_force = nlp.cx.zeros(nlp.model.nb_q, 1)
        contact_generalized_force[1] = contact_force
        qddot = solve(
            nlp.model.mass_matrix()(q, parameters),
            tau + contact_generalized_force - nlp.model.non_linear_effects()(q, qdot, parameters),
        )
        platform_acceleration = (platform_force_newtons - platform_mass_kg * gravity - contact_force) / platform_mass_kg
    else:
        raise ValueError(f"Unsupported contact model: {contact_model}")

    dxdt = vertcat(
        dq[: nlp.model.nb_root],
        dq[nlp.model.nb_root :],
        qddot[: nlp.model.nb_root],
        qddot[nlp.model.nb_root :],
        platform_velocity,
        platform_acceleration,
    )
    return DynamicsEvaluation(dxdt=dxdt, defects=None)


def _no_platform_dynamics(
    time,
    states,
    controls,
    parameters,
    algebraic_states,
    numerical_timeseries,
    nlp,
    **_,
):
    """Simplified torque-driven dynamics without explicit platform states."""

    from bioptim import DynamicsEvaluation, DynamicsFunctions
    from casadi import solve, vertcat

    _ = time
    _ = algebraic_states
    _ = numerical_timeseries

    q_joints = DynamicsFunctions.get(nlp.states["q_joints"], states)
    qdot_joints = DynamicsFunctions.get(nlp.states["qdot_joints"], states)
    tau_joints = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)
    dq = nlp.model.reshape_qdot()(q_joints, qdot_joints, parameters)
    qddot = solve(
        nlp.model.mass_matrix()(q_joints, parameters),
        tau_joints - nlp.model.non_linear_effects()(q_joints, qdot_joints, parameters),
    )
    return DynamicsEvaluation(dxdt=vertcat(dq, qddot), defects=None)


class VerticalJumpBioptimOcpBuilder:
    """Build the reduced vertical-jump OCP when `bioptim` is available."""

    def __init__(self, settings: VerticalJumpOcpSettings | None = None) -> None:
        """Store the validated OCP settings."""

        self.settings = settings or VerticalJumpOcpSettings()

    def blueprint(self, peak_force_newtons: float) -> VerticalJumpOcpBlueprint:
        """Create a serializable OCP blueprint for one force-slider choice."""

        peak_force_newtons = snap_to_discrete_value(peak_force_newtons, self.settings.force_slider_values_newtons)
        return VerticalJumpOcpBlueprint(
            settings=self.settings,
            peak_force_newtons=peak_force_newtons,
            dynamics_name=_contact_model_dynamics_name(self.settings.contact_model),
            contact_model_name=_contact_model_label(self.settings.contact_model),
        )

    def export_model(self, output_dir: str | Path) -> Path:
        """Export the reduced jumper model to one `.bioMod` file."""

        model_definition = _model_definition_from_settings(self.settings)
        model_name = (
            "vertical_jumper_3segments_no_platform.bioMod"
            if self.settings.contact_model == CONTACT_MODEL_NO_PLATFORM
            else "vertical_jumper_3segments.bioMod"
        )
        output_path = Path(output_dir) / model_name
        return model_definition.write_biomod(output_path)

    def aligned_initial_joint_configuration_rad(
        self,
        *,
        model_path: str | Path | None = None,
        tolerance: float = 1e-10,
        max_iterations: int = 25,
    ) -> tuple[float, ...]:
        """Return one initial posture aligned on the true exported-model CoM."""

        from bioptim import BiorbdModel

        model_filepath = Path(model_path) if model_path is not None else self.export_model(Path.cwd() / "generated")
        model_definition = _model_definition_from_settings(self.settings)
        initial_q = model_definition.crouched_joint_configuration_rad
        aligned_q = _align_configuration_to_zero_com_x(
            BiorbdModel(str(model_filepath)),
            initial_q,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )
        return tuple(float(value) for value in aligned_q)

    def build_ocp(
        self,
        peak_force_newtons: float,
        *,
        model_path: str | Path | None = None,
        final_time_guess: float = 1.0,
    ):
        """Instantiate the explicit-platform `bioptim` OCP."""

        bioptim_api = _import_bioptim_build_api()
        BiorbdModel = bioptim_api["BiorbdModel"]
        BoundsList = bioptim_api["BoundsList"]
        ConstraintFcn = bioptim_api["ConstraintFcn"]
        ConstraintList = bioptim_api["ConstraintList"]
        ControlType = bioptim_api["ControlType"]
        InitialGuessList = bioptim_api["InitialGuessList"]
        InterpolationType = bioptim_api["InterpolationType"]
        Node = bioptim_api["Node"]
        ObjectiveFcn = bioptim_api["ObjectiveFcn"]
        ObjectiveList = bioptim_api["ObjectiveList"]
        ObjectiveWeight = bioptim_api["ObjectiveWeight"]
        OdeSolver = bioptim_api["OdeSolver"]
        OptimalControlProgram = bioptim_api["OptimalControlProgram"]
        PhaseDynamics = bioptim_api["PhaseDynamics"]
        api_kind = bioptim_api["api_kind"]

        uses_platform_states = _uses_platform_states(self.settings.contact_model)
        blueprint = self.blueprint(peak_force_newtons)
        model_filepath = Path(model_path) if model_path is not None else self.export_model(Path.cwd() / "generated")
        if api_kind == "legacy":
            bio_model = BiorbdModel(str(model_filepath))
        else:
            if uses_platform_states:
                ExplicitPlatformBiorbdModel = _make_explicit_platform_model_class(
                    BiorbdModel,
                    bioptim_api["StateDynamics"],
                    bioptim_api["ConfigureVariables"],
                )
                bio_model = ExplicitPlatformBiorbdModel(str(model_filepath))
            else:
                NoPlatformBiorbdModel = _make_no_platform_model_class(
                    BiorbdModel,
                    bioptim_api["StateDynamics"],
                    bioptim_api["ConfigureVariables"],
                )
                bio_model = NoPlatformBiorbdModel(str(model_filepath))

        objective_functions = ObjectiveList()
        torque_regularization_selector = _shooting_weight_with_excluded_tail(
            self.settings.n_shooting,
            self.settings.torque_regularization_excluded_tail_nodes,
            self.settings.torque_regularization_tail_weight,
        )
        torque_regularization_weight = ObjectiveWeight(
            1e-5 * torque_regularization_selector,
            interpolation=InterpolationType.EACH_FRAME,
        )
        torque_derivative_regularization_weight = ObjectiveWeight(
            1e-6 * torque_regularization_selector,
            interpolation=InterpolationType.EACH_FRAME,
        )
        objective_functions.add(
            _predicted_apex_height,
            custom_type=ObjectiveFcn.Mayer,
            node=Node.END,
            gravity=9.81,
        )
        objective_functions.add(
            _final_com_anteroposterior_velocity_squared,
            custom_type=ObjectiveFcn.Mayer,
            node=Node.END,
            weight=1e-2,
        )
        objective_functions.add(
            _final_extension_error_squared,
            custom_type=ObjectiveFcn.Mayer,
            node=Node.END,
            weight=self.settings.final_extension_mayer_weight,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau_joints",
            weight=torque_regularization_weight,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau_joints",
            derivative=True,
            weight=torque_derivative_regularization_weight,
        )

        initial_q = list(self.aligned_initial_joint_configuration_rad(model_path=model_filepath))
        initial_qdot = [0.0] * bio_model.nb_qdot
        initial_guess = build_linear_inverse_dynamics_initial_guess(
            bio_model,
            initial_q,
            duration_s=final_time_guess,
            n_shooting=self.settings.n_shooting,
            final_platform_height_m=1.3,
        )
        first_interval_static_control_target = _static_control_target_for_first_interval(
            bio_model,
            initial_q,
            self.settings,
        )

        constraints = ConstraintList()
        if uses_platform_states:
            for node in (Node.ALL_SHOOTING, Node.END):
                bounds = (0.0, 5000.0) if node == Node.ALL_SHOOTING else (0.0, 0.0)
                constraints.add(
                    _contact_force_penalty,
                    node=node,
                    min_bound=bounds[0],
                    max_bound=bounds[1],
                    contact_model=self.settings.contact_model,
                    contact_name=blueprint.contact_name,
                    peak_force_newtons=peak_force_newtons,
                    total_duration_s=self.settings.final_time_upper_bound_s,
                    taper_duration_s=0.3,
                    platform_mass_kg=self.settings.platform_mass_kg,
                    athlete_mass_kg=self.settings.athlete_mass_kg,
                    gravity=9.81,
                    contact_stiffness_n_per_m=self.settings.contact_stiffness_n_per_m,
                    contact_damping_n_s_per_m=self.settings.contact_damping_n_s_per_m,
                )
        else:
            constraints.add(
                _contact_force_penalty,
                node=Node.ALL_SHOOTING,
                min_bound=1e-6,
                max_bound=1e6,
                contact_model=self.settings.contact_model,
                contact_name=blueprint.contact_name,
                peak_force_newtons=peak_force_newtons,
                total_duration_s=self.settings.final_time_upper_bound_s,
                taper_duration_s=0.3,
                platform_mass_kg=self.settings.platform_mass_kg,
                athlete_mass_kg=self.settings.athlete_mass_kg,
                gravity=9.81,
                contact_stiffness_n_per_m=self.settings.contact_stiffness_n_per_m,
                contact_damping_n_s_per_m=self.settings.contact_damping_n_s_per_m,
            )
            constraints.add(
                _contact_force_penalty,
                node=Node.END,
                min_bound=0.0,
                max_bound=0.0,
                contact_model=self.settings.contact_model,
                contact_name=blueprint.contact_name,
                peak_force_newtons=peak_force_newtons,
                total_duration_s=self.settings.final_time_upper_bound_s,
                taper_duration_s=0.3,
                platform_mass_kg=self.settings.platform_mass_kg,
                athlete_mass_kg=self.settings.athlete_mass_kg,
                gravity=9.81,
                contact_stiffness_n_per_m=self.settings.contact_stiffness_n_per_m,
                contact_damping_n_s_per_m=self.settings.contact_damping_n_s_per_m,
            )
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT,
            node=Node.END,
            min_bound=self.settings.final_time_lower_bound_s,
            max_bound=self.settings.final_time_upper_bound_s,
            phase=0,
        )
        constraints.add(
            _sagittal_angular_momentum,
            node=Node.ALL_SHOOTING,
            min_bound=-self.settings.angular_momentum_bound_n_s,
            max_bound=self.settings.angular_momentum_bound_n_s,
        )
        for node in (Node.ALL_SHOOTING, Node.END):
            constraints.add(
                _vertical_com_velocity,
                node=node,
                min_bound=0.0,
                max_bound=1e6,
            )
        constraints.add(
            ConstraintFcn.TRACK_CONTROL,
            key="tau_joints",
            node=Node.START,
            target=first_interval_static_control_target,
        )

        dynamics_kwargs = dict(
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            expand_dynamics=self.settings.expand_dynamics,
            ode_solver=OdeSolver.RK4(n_integration_steps=self.settings.rk4_substeps),
        )
        explicit_platform_kwargs = dict(
            peak_force_newtons=peak_force_newtons,
            total_duration_s=self.settings.final_time_upper_bound_s,
            taper_duration_s=0.3,
            platform_mass_kg=self.settings.platform_mass_kg,
            gravity=9.81,
            contact_model=self.settings.contact_model,
            contact_name=blueprint.contact_name,
            contact_stiffness_n_per_m=self.settings.contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=self.settings.contact_damping_n_s_per_m,
        )
        if api_kind == "legacy":
            Dynamics = bioptim_api["Dynamics"]
            if uses_platform_states:
                dynamics = Dynamics(
                    _configure_explicit_platform_dynamics,
                    dynamic_function=_explicit_platform_dynamics,
                    **dynamics_kwargs,
                    **explicit_platform_kwargs,
                )
            else:
                dynamics = Dynamics(
                    _configure_no_platform_dynamics,
                    dynamic_function=_no_platform_dynamics,
                    **dynamics_kwargs,
                )
        else:
            DynamicsOptions = bioptim_api["DynamicsOptions"]
            dynamics = DynamicsOptions(
                **dynamics_kwargs,
                **(explicit_platform_kwargs if uses_platform_states else {}),
            )

        q_bounds = bio_model.bounds_from_ranges("q")
        qdot_bounds = bio_model.bounds_from_ranges("qdot")
        q_min = q_bounds.min[:, 0]
        q_max = q_bounds.max[:, 0]
        qdot_min = qdot_bounds.min[:, 0]
        qdot_max = qdot_bounds.max[:, 0]

        x_bounds = BoundsList()
        q_roots_bounds = _constant_bounds_with_fixed_start(
            q_min[: bio_model.nb_root],
            q_max[: bio_model.nb_root],
            initial_q[: bio_model.nb_root],
        )
        q_joints_bounds = _constant_bounds_with_fixed_start(
            q_min[bio_model.nb_root :],
            q_max[bio_model.nb_root :],
            initial_q[bio_model.nb_root :],
        )
        qdot_roots_bounds = _constant_bounds_with_fixed_start(
            qdot_min[: bio_model.nb_root],
            qdot_max[: bio_model.nb_root],
            initial_qdot[: bio_model.nb_root],
        )
        qdot_joints_bounds = _constant_bounds_with_fixed_start(
            qdot_min[bio_model.nb_root :],
            qdot_max[bio_model.nb_root :],
            initial_qdot[bio_model.nb_root :],
        )
        if uses_platform_states:
            platform_position_bounds = _constant_bounds_with_fixed_start([-0.2], [2.5], [0.0])
            platform_velocity_bounds = _constant_bounds_with_fixed_start([-10.0], [10.0], [0.0])

        if bio_model.nb_root:
            x_bounds.add(
                "q_roots",
                min_bound=q_roots_bounds[0],
                max_bound=q_roots_bounds[1],
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )
        x_bounds.add(
            "q_joints",
            min_bound=q_joints_bounds[0],
            max_bound=q_joints_bounds[1],
            interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        )
        if bio_model.nb_root:
            x_bounds.add(
                "qdot_roots",
                min_bound=qdot_roots_bounds[0],
                max_bound=qdot_roots_bounds[1],
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )
        x_bounds.add(
            "qdot_joints",
            min_bound=qdot_joints_bounds[0],
            max_bound=qdot_joints_bounds[1],
            interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
        )
        if uses_platform_states:
            x_bounds.add(
                "platform_position",
                min_bound=platform_position_bounds[0],
                max_bound=platform_position_bounds[1],
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )
            x_bounds.add(
                "platform_velocity",
                min_bound=platform_velocity_bounds[0],
                max_bound=platform_velocity_bounds[1],
                interpolation=InterpolationType.CONSTANT_WITH_FIRST_AND_LAST_DIFFERENT,
            )

        x_init = InitialGuessList()
        if bio_model.nb_root:
            x_init.add(
                "q_roots",
                initial_guess.q[: bio_model.nb_root, :],
                interpolation=InterpolationType.EACH_FRAME,
            )
        x_init.add(
            "q_joints",
            initial_guess.q[bio_model.nb_root :, :],
            interpolation=InterpolationType.EACH_FRAME,
        )
        if bio_model.nb_root:
            x_init.add(
                "qdot_roots",
                initial_guess.qdot[: bio_model.nb_root, :],
                interpolation=InterpolationType.EACH_FRAME,
            )
        x_init.add(
            "qdot_joints",
            initial_guess.qdot[bio_model.nb_root :, :],
            interpolation=InterpolationType.EACH_FRAME,
        )
        if uses_platform_states:
            x_init.add(
                "platform_position",
                initial_guess.platform_position,
                interpolation=InterpolationType.EACH_FRAME,
            )
            x_init.add(
                "platform_velocity",
                initial_guess.platform_velocity,
                interpolation=InterpolationType.EACH_FRAME,
            )

        n_joint_tau = bio_model.nb_q - bio_model.nb_root
        u_bounds = BoundsList()
        tau_min_bounds = [self.settings.tau_min_nm] * n_joint_tau
        tau_max_bounds = [self.settings.tau_max_nm] * n_joint_tau
        athlete_mass_kg = float(self.settings.athlete_mass_kg)
        knee_control_index = _knee_control_index(self.settings)
        hip_control_index = _hip_control_index(self.settings)
        if knee_control_index < len(tau_min_bounds):
            tau_min_bounds[knee_control_index] = -15.0 * athlete_mass_kg
            tau_max_bounds[knee_control_index] = 15.0 * athlete_mass_kg
        if hip_control_index < len(tau_min_bounds):
            tau_min_bounds[hip_control_index] = -20.0 * athlete_mass_kg
            tau_max_bounds[hip_control_index] = 20.0 * athlete_mass_kg
        if self.settings.contact_model == CONTACT_MODEL_NO_PLATFORM and tau_min_bounds:
            # In the reduced no-platform mode, the distal root-like rotation is kept passive.
            tau_min_bounds[0] = 0.0
            tau_max_bounds[0] = 0.0
        u_bounds["tau_joints"] = tau_min_bounds, tau_max_bounds

        u_init = InitialGuessList()
        tau_initial_guess = np.asarray(initial_guess.tau[bio_model.nb_root :, :], dtype=float).copy()
        if self.settings.contact_model == CONTACT_MODEL_NO_PLATFORM and tau_initial_guess.shape[0] > 0:
            tau_initial_guess[0, :] = 0.0
        u_init.add(
            "tau_joints",
            tau_initial_guess,
            interpolation=InterpolationType.EACH_FRAME,
        )

        return _instantiate_ocp(
            OptimalControlProgram,
            api_kind,
            bio_model,
            dynamics,
            self.settings.n_shooting,
            final_time_guess,
            x_bounds=x_bounds,
            x_init=x_init,
            u_bounds=u_bounds,
            u_init=u_init,
            objective_functions=objective_functions,
            constraints=constraints,
            control_type=ControlType.CONSTANT,
            n_threads=self.settings.n_threads,
            use_sx=self.settings.use_sx,
        )
