"""Problem settings for the planar vertical jump OCP."""

from __future__ import annotations

from dataclasses import dataclass, field

CONTACT_MODEL_RIGID_UNILATERAL = "rigid_unilateral"
CONTACT_MODEL_COMPLIANT_UNILATERAL = "compliant_unilateral"


def discrete_force_slider_values() -> tuple[int, ...]:
    """Return the admissible platform-force slider values."""

    return tuple(range(900, 1301, 50))


def discrete_mass_slider_values() -> tuple[int, ...]:
    """Return the admissible athlete-mass slider values."""

    return (40, 45, 50, 55)


def discrete_contact_models() -> tuple[str, ...]:
    """Return the supported athlete-platform contact models."""

    return (CONTACT_MODEL_RIGID_UNILATERAL, CONTACT_MODEL_COMPLIANT_UNILATERAL)


@dataclass(frozen=True)
class VerticalJumpOcpSettings:
    """Static settings for the first version of the jump OCP."""

    platform_mass_kg: float = 80.0
    athlete_height_m: float = 1.60
    athlete_mass_kg: float = 50.0
    contact_model: str = CONTACT_MODEL_RIGID_UNILATERAL
    contact_stiffness_n_per_m: float = 30000.0
    contact_damping_n_s_per_m: float = 1500.0
    tau_min_nm: float = -500.0
    tau_max_nm: float = 500.0
    final_time_upper_bound_s: float = 2.0
    final_time_lower_bound_s: float = 0.2
    n_shooting: int = 100
    rk4_substeps: int = 4
    n_threads: int = 6
    initial_joint_flexion_deg: float = 100.0
    force_slider_values_newtons: tuple[int, ...] = field(default_factory=discrete_force_slider_values)
    mass_slider_values_kg: tuple[int, ...] = field(default_factory=discrete_mass_slider_values)

    def __post_init__(self) -> None:
        """Validate the OCP settings."""

        if self.platform_mass_kg <= 0.0:
            raise ValueError("platform_mass_kg must be strictly positive")
        if self.athlete_height_m <= 0.0:
            raise ValueError("athlete_height_m must be strictly positive")
        if self.athlete_mass_kg not in self.mass_slider_values_kg:
            raise ValueError("athlete_mass_kg must match one slider value")
        if self.contact_model not in discrete_contact_models():
            raise ValueError("contact_model must match one supported contact mode")
        if self.contact_stiffness_n_per_m < 0.0:
            raise ValueError("contact_stiffness_n_per_m must stay non-negative")
        if self.contact_damping_n_s_per_m < 0.0:
            raise ValueError("contact_damping_n_s_per_m must stay non-negative")
        if self.tau_min_nm >= self.tau_max_nm:
            raise ValueError("tau_min_nm must stay below tau_max_nm")
        if self.final_time_lower_bound_s <= 0.0:
            raise ValueError("final_time_lower_bound_s must be strictly positive")
        if self.final_time_lower_bound_s >= self.final_time_upper_bound_s:
            raise ValueError("final_time_lower_bound_s must stay below final_time_upper_bound_s")
        if self.n_shooting <= 0:
            raise ValueError("n_shooting must be strictly positive")
        if self.rk4_substeps <= 0:
            raise ValueError("rk4_substeps must be strictly positive")
        if self.n_threads <= 0:
            raise ValueError("n_threads must be strictly positive")
        if not self.force_slider_values_newtons:
            raise ValueError("force_slider_values_newtons cannot be empty")
