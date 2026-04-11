"""`bioptim` OCP builder for the reduced planar vertical jump."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synchro_jump.modeling import AthleteMorphology, PlanarJumperModelDefinition
from synchro_jump.optimization.contact import PlatformInteractionModel
from synchro_jump.optimization.force_profile import PlatformForceProfile
from synchro_jump.optimization.problem import VerticalJumpOcpSettings


@dataclass(frozen=True)
class VerticalJumpOcpBlueprint:
    """High-level OCP description independent of optional runtime dependencies."""

    settings: VerticalJumpOcpSettings
    peak_force_newtons: float
    objective_name: str = "CUSTOM_PREDICTED_COM_HEIGHT"
    dynamics_name: str = "TORQUE_DRIVEN_WITH_EXPLICIT_PLATFORM"
    contact_name: str = "platform_contact"
    ode_solver_name: str = "RK4"
    control_type: str = "CONSTANT"

    def contact_force_target(self, final_time_guess: float | None = None) -> tuple[float, ...]:
        """Return one surrogate contact-force target used by the GUI."""

        duration = final_time_guess or self.settings.final_time_upper_bound_s
        profile = PlatformForceProfile(
            peak_force_newtons=self.peak_force_newtons,
            total_duration=duration,
        )
        interaction = PlatformInteractionModel(platform_mass_kg=self.settings.platform_mass_kg)
        targets = []
        for node_index in range(self.settings.n_shooting):
            time = duration * node_index / max(self.settings.n_shooting - 1, 1)
            surrogate_contact = interaction.contact_force(
                platform_actuation_force_newtons=profile.force_at(time),
                platform_vertical_acceleration=0.0,
            )
            targets.append(max(surrogate_contact, 0.0))
        return tuple(targets)


def _import_bioptim_build_api():
    """Return the `bioptim` runtime symbols required by the current OCP builder."""

    import bioptim

    required_names = (
        "BiorbdModel",
        "BoundsList",
        "ConstraintFcn",
        "ConstraintList",
        "ControlType",
        "Dynamics",
        "InitialGuessList",
        "InterpolationType",
        "Node",
        "ObjectiveFcn",
        "ObjectiveList",
        "OdeSolver",
        "OptimalControlProgram",
        "PhaseDynamics",
    )
    missing_names = [name for name in required_names if not hasattr(bioptim, name)]
    if missing_names:
        version = getattr(bioptim, "__version__", "unknown")
        raise RuntimeError(
            "Version de bioptim non supportee pour cet OCP explicite: "
            f"{version}. Symboles manquants: {', '.join(missing_names)}. "
            "Utilise l'environnement Conda du projet avec bioptim=3.2.1."
        )

    return {name: getattr(bioptim, name) for name in required_names}


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

    q_roots = DynamicsFunctions.get(nlp.states["q_roots"], states)
    q_joints = DynamicsFunctions.get(nlp.states["q_joints"], states)
    qdot_roots = DynamicsFunctions.get(nlp.states["qdot_roots"], states)
    qdot_joints = DynamicsFunctions.get(nlp.states["qdot_joints"], states)
    tau_joints = DynamicsFunctions.get(nlp.controls["tau_joints"], controls)
    platform_position = DynamicsFunctions.get(nlp.states["platform_position"], states)
    platform_velocity = DynamicsFunctions.get(nlp.states["platform_velocity"], states)
    q = vertcat(q_roots, q_joints)
    qdot = vertcat(qdot_roots, qdot_joints)
    return q, qdot, tau_joints, platform_position, platform_velocity


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


def _predicted_apex_height(controller, gravity: float = 9.81):
    """Custom Mayer objective based on CoM height and vertical velocity."""

    com = controller.model.center_of_mass()(controller.q, controller.parameters.cx)
    com_velocity = controller.model.center_of_mass_velocity()(controller.q, controller.qdot, controller.parameters.cx)
    vertical_velocity = _symbolic_positive_part(com_velocity[2])
    return -(com[2] + vertical_velocity**2 / (2.0 * gravity))


_predicted_apex_height.__name__ = "predicted_apex_height"


def _contact_force_penalty(
    controller,
    contact_stiffness_n_per_m: float,
    contact_damping_n_s_per_m: float,
):
    """Return the compliant contact force used in custom constraints."""

    return _symbolic_compliant_contact_force(
        controller.q,
        controller.qdot,
        platform_position=controller.states["platform_position"].cx,
        platform_velocity=controller.states["platform_velocity"].cx,
        contact_stiffness_n_per_m=contact_stiffness_n_per_m,
        contact_damping_n_s_per_m=contact_damping_n_s_per_m,
    )


_contact_force_penalty.__name__ = "contact_force_penalty"


def _configure_explicit_platform_dynamics(
    ocp,
    nlp,
    *,
    peak_force_newtons: float,
    total_duration_s: float,
    taper_duration_s: float,
    platform_mass_kg: float,
    gravity: float,
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

    name_dof = list(nlp.model.name_dof)
    name_q_roots = name_dof[: nlp.model.nb_root]
    name_q_joints = name_dof[nlp.model.nb_root :]

    ConfigureProblem.configure_new_variable("q_roots", name_q_roots, ocp, nlp, as_states=True, as_controls=False)
    ConfigureProblem.configure_new_variable("q_joints", name_q_joints, ocp, nlp, as_states=True, as_controls=False)
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
        contact_stiffness_n_per_m=contact_stiffness_n_per_m,
        contact_damping_n_s_per_m=contact_damping_n_s_per_m,
    )


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
    contact_stiffness_n_per_m: float,
    contact_damping_n_s_per_m: float,
):
    """Custom dynamics for the explicit moving platform."""

    from bioptim import DynamicsEvaluation, DynamicsFunctions
    from casadi import solve, vertcat

    _ = algebraic_states
    _ = numerical_timeseries

    q, qdot, tau_joints, platform_position, platform_velocity = _split_q_vectors(nlp, states, controls)
    dq = DynamicsFunctions.compute_qdot(nlp, q, qdot)
    contact_force = _symbolic_compliant_contact_force(
        q,
        qdot,
        platform_position=platform_position,
        platform_velocity=platform_velocity,
        contact_stiffness_n_per_m=contact_stiffness_n_per_m,
        contact_damping_n_s_per_m=contact_damping_n_s_per_m,
    )
    tau = vertcat(nlp.cx.zeros(nlp.model.nb_root, 1), tau_joints)
    contact_generalized_force = nlp.cx.zeros(nlp.model.nb_q, 1)
    contact_generalized_force[1] = contact_force
    qddot = solve(
        nlp.model.mass_matrix()(q, parameters),
        tau + contact_generalized_force - nlp.model.non_linear_effects()(q, qdot, parameters),
    )
    platform_force_newtons = _symbolic_platform_force(
        time,
        peak_force_newtons=peak_force_newtons,
        total_duration_s=total_duration_s,
        taper_duration_s=taper_duration_s,
    )
    platform_acceleration = (platform_force_newtons - platform_mass_kg * gravity - contact_force) / platform_mass_kg
    dxdt = vertcat(
        dq[: nlp.model.nb_root],
        dq[nlp.model.nb_root :],
        qddot[: nlp.model.nb_root],
        qddot[nlp.model.nb_root :],
        platform_velocity,
        platform_acceleration,
    )
    return DynamicsEvaluation(dxdt=dxdt, defects=None)


class VerticalJumpBioptimOcpBuilder:
    """Build the reduced vertical-jump OCP when `bioptim` is available."""

    def __init__(self, settings: VerticalJumpOcpSettings | None = None) -> None:
        """Store the validated OCP settings."""

        self.settings = settings or VerticalJumpOcpSettings()

    def blueprint(self, peak_force_newtons: float) -> VerticalJumpOcpBlueprint:
        """Create a serializable OCP blueprint for one force-slider choice."""

        if peak_force_newtons not in self.settings.force_slider_values_newtons:
            raise ValueError("peak_force_newtons must match one slider value")
        return VerticalJumpOcpBlueprint(settings=self.settings, peak_force_newtons=peak_force_newtons)

    def export_model(self, output_dir: str | Path) -> Path:
        """Export the reduced jumper model to one `.bioMod` file."""

        model_definition = PlanarJumperModelDefinition(
            morphology=AthleteMorphology(
                height_m=self.settings.athlete_height_m,
                mass_kg=self.settings.athlete_mass_kg,
            )
        )
        output_path = Path(output_dir) / "vertical_jumper_3segments.bioMod"
        return model_definition.write_biomod(output_path)

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
        Dynamics = bioptim_api["Dynamics"]
        InitialGuessList = bioptim_api["InitialGuessList"]
        InterpolationType = bioptim_api["InterpolationType"]
        Node = bioptim_api["Node"]
        ObjectiveFcn = bioptim_api["ObjectiveFcn"]
        ObjectiveList = bioptim_api["ObjectiveList"]
        OdeSolver = bioptim_api["OdeSolver"]
        OptimalControlProgram = bioptim_api["OptimalControlProgram"]
        PhaseDynamics = bioptim_api["PhaseDynamics"]

        blueprint = self.blueprint(peak_force_newtons)
        model_filepath = Path(model_path) if model_path is not None else self.export_model(Path.cwd() / "generated")
        bio_model = BiorbdModel(str(model_filepath))

        objective_functions = ObjectiveList()
        objective_functions.add(
            _predicted_apex_height,
            custom_type=ObjectiveFcn.Mayer,
            node=Node.END,
            gravity=9.81,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau_joints",
            weight=1e-5,
        )

        constraints = ConstraintList()
        constraints.add(
            _contact_force_penalty,
            node=Node.ALL_SHOOTING,
            min_bound=0.0,
            max_bound=5000.0,
            contact_stiffness_n_per_m=self.settings.contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=self.settings.contact_damping_n_s_per_m,
        )
        constraints.add(
            _contact_force_penalty,
            node=Node.END,
            min_bound=0.0,
            max_bound=0.0,
            contact_stiffness_n_per_m=self.settings.contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=self.settings.contact_damping_n_s_per_m,
        )
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT,
            node=Node.END,
            minimum=self.settings.final_time_lower_bound_s,
            maximum=self.settings.final_time_upper_bound_s,
        )

        dynamics = Dynamics(
            _configure_explicit_platform_dynamics,
            dynamic_function=_explicit_platform_dynamics,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
            expand_dynamics=False,
            ode_solver=OdeSolver.RK4(n_integration_steps=self.settings.rk4_substeps),
            peak_force_newtons=peak_force_newtons,
            total_duration_s=self.settings.final_time_upper_bound_s,
            taper_duration_s=0.3,
            platform_mass_kg=self.settings.platform_mass_kg,
            gravity=9.81,
            contact_stiffness_n_per_m=self.settings.contact_stiffness_n_per_m,
            contact_damping_n_s_per_m=self.settings.contact_damping_n_s_per_m,
        )

        q_bounds = bio_model.bounds_from_ranges("q")
        qdot_bounds = bio_model.bounds_from_ranges("qdot")
        q_min = q_bounds.min[:, 0]
        q_max = q_bounds.max[:, 0]
        qdot_min = qdot_bounds.min[:, 0]
        qdot_max = qdot_bounds.max[:, 0]

        initial_q = list(
            PlanarJumperModelDefinition(
                morphology=AthleteMorphology(
                    height_m=self.settings.athlete_height_m,
                    mass_kg=self.settings.athlete_mass_kg,
                )
            ).initial_joint_configuration_rad
        )
        initial_qdot = [0.0] * bio_model.nb_qdot

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
        platform_position_bounds = _constant_bounds_with_fixed_start([-0.2], [2.5], [0.0])
        platform_velocity_bounds = _constant_bounds_with_fixed_start([-10.0], [10.0], [0.0])

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
        x_init["q_roots"] = initial_q[: bio_model.nb_root]
        x_init["q_joints"] = initial_q[bio_model.nb_root :]
        x_init["qdot_roots"] = initial_qdot[: bio_model.nb_root]
        x_init["qdot_joints"] = initial_qdot[bio_model.nb_root :]
        x_init["platform_position"] = [0.0]
        x_init["platform_velocity"] = [0.0]

        n_joint_tau = bio_model.nb_q - bio_model.nb_root
        u_bounds = BoundsList()
        u_bounds["tau_joints"] = [self.settings.tau_min_nm] * n_joint_tau, [self.settings.tau_max_nm] * n_joint_tau

        u_init = InitialGuessList()
        u_init["tau_joints"] = [0.0] * n_joint_tau

        return OptimalControlProgram(
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
        )
