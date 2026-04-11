"""Contact-model helpers for the moving-platform jump."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformInteractionModel:
    """Interaction model between the athlete and the moving platform.

    Two complementary contact descriptions are provided:

    - a rigid-contact balance derived from the platform dynamics
    - a compliant contact law based on foot-platform compression

    The rigid platform dynamics are written as:

    ``m_p * y_ddot = F_platform - m_p * g - R_contact``

    where ``R_contact`` is the vertical force transmitted by the athlete onto the
    platform. The upward reaction received by the athlete has the same magnitude.
    """

    platform_mass_kg: float = 80.0
    gravity: float = 9.81
    contact_stiffness_n_per_m: float = 30000.0
    contact_damping_n_s_per_m: float = 1500.0

    def __post_init__(self) -> None:
        """Validate the interaction parameters."""

        if self.platform_mass_kg <= 0.0:
            raise ValueError("platform_mass_kg must be strictly positive")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be strictly positive")
        if self.contact_stiffness_n_per_m < 0.0:
            raise ValueError("contact_stiffness_n_per_m must stay non-negative")
        if self.contact_damping_n_s_per_m < 0.0:
            raise ValueError("contact_damping_n_s_per_m must stay non-negative")

    def compression(self, *, platform_position_m: float, foot_position_m: float) -> float:
        """Return the positive compression between the platform and the foot."""

        return max(platform_position_m - foot_position_m, 0.0)

    def closing_speed(
        self,
        *,
        platform_velocity_m_s: float,
        foot_velocity_m_s: float,
    ) -> float:
        """Return the positive closing speed between the platform and the foot."""

        return max(platform_velocity_m_s - foot_velocity_m_s, 0.0)

    def contact_force(
        self,
        platform_actuation_force_newtons: float,
        platform_vertical_acceleration: float,
    ) -> float:
        """Compute the instantaneous athlete-platform contact force."""

        return platform_actuation_force_newtons - self.platform_mass_kg * (
            self.gravity + platform_vertical_acceleration
        )

    def compliant_contact_force(
        self,
        *,
        platform_position_m: float,
        platform_velocity_m_s: float,
        foot_position_m: float,
        foot_velocity_m_s: float,
    ) -> float:
        """Return the compliant normal force transmitted by the platform."""

        compression = self.compression(
            platform_position_m=platform_position_m,
            foot_position_m=foot_position_m,
        )
        closing_speed = self.closing_speed(
            platform_velocity_m_s=platform_velocity_m_s,
            foot_velocity_m_s=foot_velocity_m_s,
        )
        return max(
            self.contact_stiffness_n_per_m * compression
            + self.contact_damping_n_s_per_m * closing_speed,
            0.0,
        )

    def liftoff_residual(
        self,
        platform_actuation_force_newtons: float,
        platform_vertical_acceleration: float,
    ) -> float:
        """Return the terminal contact-force residual used for lift-off."""

        return self.contact_force(
            platform_actuation_force_newtons=platform_actuation_force_newtons,
            platform_vertical_acceleration=platform_vertical_acceleration,
        )
