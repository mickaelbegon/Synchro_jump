"""`bioptim` OCP builder for the reduced planar vertical jump."""

from __future__ import annotations

import math
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
    objective_name: str = "MINIMIZE_PREDICTED_COM_HEIGHT"
    dynamics_name: str = "TORQUE_DRIVEN_WITH_CONTACT_SURROGATE"
    contact_name: str = "platform_contact"
    ode_solver_name: str = "RK4"
    control_type: str = "CONSTANT"

    def contact_force_target(self, final_time_guess: float | None = None) -> tuple[float, ...]:
        """Return the target contact-force profile used in the first OCP version.

        This first implementation uses a surrogate target derived from the
        platform actuation profile and the platform static weight.
        """

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
        """Instantiate the first version of the `bioptim` OCP.

        The current implementation uses native rigid contact in `bioptim` and
        tracks a surrogate vertical contact-force profile derived from the
        platform actuation. The exact platform-state dynamics remain separated in
        pure-Python helpers and can be promoted to custom dynamics next.
        """

        import numpy as np
        from bioptim import (
            BiorbdModel,
            BoundsList,
            ConstraintFcn,
            ConstraintList,
            ControlType,
            DynamicsFcn,
            DynamicsList,
            InitialGuessList,
            Node,
            ObjectiveFcn,
            ObjectiveList,
            OdeSolver,
            OptimalControlProgram,
            PhaseDynamics,
        )

        blueprint = self.blueprint(peak_force_newtons)
        model_filepath = Path(model_path) if model_path is not None else self.export_model(Path.cwd() / "generated")
        bio_model = BiorbdModel(str(model_filepath))

        objective_functions = ObjectiveList()
        objective_functions.add(
            ObjectiveFcn.Mayer.MINIMIZE_PREDICTED_COM_HEIGHT,
            node=Node.END,
            weight=-1.0,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
            key="tau",
            weight=1e-5,
        )
        objective_functions.add(
            ObjectiveFcn.Lagrange.TRACK_CONTACT_FORCES,
            index=0,
            target=np.array([blueprint.contact_force_target(final_time_guess)]),
            weight=1e-4,
        )

        constraints = ConstraintList()
        constraints.add(
            ConstraintFcn.TIME_CONSTRAINT,
            node=Node.END,
            minimum=self.settings.final_time_lower_bound_s,
            maximum=self.settings.final_time_upper_bound_s,
        )
        constraints.add(
            ConstraintFcn.TRACK_CONTACT_FORCES,
            node=Node.END,
            index=0,
            target=np.array([[0.0]]),
        )

        dynamics = DynamicsList()
        dynamics.add(
            DynamicsFcn.TORQUE_DRIVEN,
            with_contact=True,
            phase_dynamics=PhaseDynamics.SHARED_DURING_THE_PHASE,
        )

        x_bounds = BoundsList()
        x_bounds["q"] = bio_model.bounds_from_ranges("q")
        x_bounds["qdot"] = bio_model.bounds_from_ranges("qdot")

        initial_q = list(
            PlanarJumperModelDefinition(
                morphology=AthleteMorphology(
                    height_m=self.settings.athlete_height_m,
                    mass_kg=self.settings.athlete_mass_kg,
                )
            ).initial_joint_configuration_rad
        )
        x_bounds["q"][:, 0] = initial_q
        x_bounds["qdot"][:, 0] = [0.0] * bio_model.nb_qdot

        x_init = InitialGuessList()
        x_init["q"] = initial_q
        x_init["qdot"] = [0.0] * bio_model.nb_qdot

        u_bounds = BoundsList()
        u_bounds["tau"] = [self.settings.tau_min_nm] * bio_model.nb_tau, [self.settings.tau_max_nm] * bio_model.nb_tau

        u_init = InitialGuessList()
        u_init["tau"] = [0.0] * bio_model.nb_tau

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
            ode_solver=OdeSolver.RK4(n_integration_steps=self.settings.rk4_substeps),
            control_type=ControlType.CONSTANT,
        )
