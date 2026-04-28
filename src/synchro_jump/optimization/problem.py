"""Problem settings for the planar vertical jump OCP."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CONTACT_MODEL_RIGID_UNILATERAL = "rigid_unilateral"
CONTACT_MODEL_COMPLIANT_UNILATERAL = "compliant_unilateral"
CONTACT_MODEL_NO_PLATFORM = "no_platform"


def discrete_force_slider_values() -> tuple[int, ...]:
    """Return the admissible integer platform-force slider values."""

    return tuple(900 + 20 * index for index in range(21))


def discrete_mass_slider_values() -> tuple[int, ...]:
    """Return the admissible integer athlete-mass slider values."""

    return tuple(40 + index for index in range(18))


def discrete_contact_models() -> tuple[str, ...]:
    """Return the supported athlete-platform contact models."""

    return (
        CONTACT_MODEL_RIGID_UNILATERAL,
        CONTACT_MODEL_COMPLIANT_UNILATERAL,
        CONTACT_MODEL_NO_PLATFORM,
    )


def snap_to_discrete_value(value: float, admissible_values: tuple[float, ...]) -> float:
    """Return the closest admissible slider value."""

    return min(admissible_values, key=lambda candidate: abs(candidate - value))


def matches_discrete_value(value: float, admissible_values: tuple[float, ...], *, tolerance: float = 1e-9) -> bool:
    """Return whether one value belongs to the requested discrete grid."""

    return abs(snap_to_discrete_value(value, admissible_values) - value) <= tolerance


@dataclass(frozen=True)
class VerticalJumpOcpSettings:
    """Static settings for the first version of the jump OCP."""

    platform_mass_kg: float = 80.0
    athlete_height_m: float = 1.60
    athlete_mass_kg: float = 50.0
    contact_model: str = CONTACT_MODEL_RIGID_UNILATERAL
    contact_stiffness_n_per_m: float = 30000.0
    contact_damping_n_s_per_m: float = 1500.0
    tau_min_nm: float = -1000.0
    tau_max_nm: float = 1000.0
    angular_momentum_bound_n_s: float = 5.0
    torque_regularization_excluded_tail_nodes: int = 3
    final_time_upper_bound_s: float = 2.0
    final_time_lower_bound_s: float = 0.2
    n_shooting: int = 100
    rk4_substeps: int = 4
    n_threads: int = 6
    use_sx: bool = True
    expand_dynamics: bool = True
    ipopt_linear_solver: str = "ma57"
    ipopt_hsl_library_path: str | None = None
    initial_joint_flexion_deg: float = 100.0
    force_slider_values_newtons: tuple[float, ...] = field(default_factory=discrete_force_slider_values)
    mass_slider_values_kg: tuple[float, ...] = field(default_factory=discrete_mass_slider_values)

    def __post_init__(self) -> None:
        """Validate the OCP settings."""

        if self.platform_mass_kg <= 0.0:
            raise ValueError("platform_mass_kg must be strictly positive")
        if self.athlete_height_m <= 0.0:
            raise ValueError("athlete_height_m must be strictly positive")
        if not (self.mass_slider_values_kg[0] <= self.athlete_mass_kg <= self.mass_slider_values_kg[-1]):
            raise ValueError("athlete_mass_kg must stay within the slider range")
        if self.contact_model not in discrete_contact_models():
            raise ValueError("contact_model must match one supported contact mode")
        if self.contact_stiffness_n_per_m < 0.0:
            raise ValueError("contact_stiffness_n_per_m must stay non-negative")
        if self.contact_damping_n_s_per_m < 0.0:
            raise ValueError("contact_damping_n_s_per_m must stay non-negative")
        if self.tau_min_nm >= self.tau_max_nm:
            raise ValueError("tau_min_nm must stay below tau_max_nm")
        if self.angular_momentum_bound_n_s <= 0.0:
            raise ValueError("angular_momentum_bound_n_s must be strictly positive")
        if self.torque_regularization_excluded_tail_nodes < 0:
            raise ValueError("torque_regularization_excluded_tail_nodes must stay non-negative")
        if self.torque_regularization_excluded_tail_nodes >= self.n_shooting:
            raise ValueError("torque_regularization_excluded_tail_nodes must stay below n_shooting")
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
        if not isinstance(self.use_sx, bool):
            raise ValueError("use_sx must be a boolean")
        if not isinstance(self.expand_dynamics, bool):
            raise ValueError("expand_dynamics must be a boolean")
        if not self.ipopt_linear_solver:
            raise ValueError("ipopt_linear_solver cannot be empty")
        if self.ipopt_hsl_library_path is not None and not isinstance(self.ipopt_hsl_library_path, (str, Path)):
            raise ValueError("ipopt_hsl_library_path must be a path-like string when provided")
        if not self.force_slider_values_newtons:
            raise ValueError("force_slider_values_newtons cannot be empty")
