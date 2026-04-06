"""Contact-model helpers for the moving-platform jump."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformInteractionModel:
    """Rigid-contact interaction between the athlete and the moving platform.

    The platform dynamics are written as:

    ``m_p * y_ddot = F_platform - m_p * g - R_contact``

    where ``R_contact`` is the vertical force transmitted by the athlete onto the
    platform. The upward reaction received by the athlete has the same magnitude.
    """

    platform_mass_kg: float = 80.0
    gravity: float = 9.81

    def __post_init__(self) -> None:
        """Validate the interaction parameters."""

        if self.platform_mass_kg <= 0.0:
            raise ValueError("platform_mass_kg must be strictly positive")
        if self.gravity <= 0.0:
            raise ValueError("gravity must be strictly positive")

    def contact_force(
        self,
        platform_actuation_force_newtons: float,
        platform_vertical_acceleration: float,
    ) -> float:
        """Compute the instantaneous athlete-platform contact force."""

        return platform_actuation_force_newtons - self.platform_mass_kg * (
            self.gravity + platform_vertical_acceleration
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
