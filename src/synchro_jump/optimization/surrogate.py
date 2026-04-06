"""Surrogate jump metrics used for the interactive GUI."""

from __future__ import annotations

from collections.abc import Sequence

from synchro_jump.optimization.estimator import estimate_jump_apex_height


def estimate_takeoff_velocity_from_contact_profile(
    contact_force_profile_newtons: Sequence[float],
    athlete_mass_kg: float,
    total_duration_s: float,
    gravity: float = 9.81,
) -> float:
    """Estimate take-off velocity from a contact-force profile."""

    if athlete_mass_kg <= 0.0:
        raise ValueError("athlete_mass_kg must be strictly positive")
    if total_duration_s <= 0.0:
        raise ValueError("total_duration_s must be strictly positive")
    if gravity <= 0.0:
        raise ValueError("gravity must be strictly positive")
    if not contact_force_profile_newtons:
        raise ValueError("contact_force_profile_newtons cannot be empty")

    dt = total_duration_s / max(len(contact_force_profile_newtons) - 1, 1)
    velocity = 0.0
    for contact_force in contact_force_profile_newtons:
        velocity += (contact_force / athlete_mass_kg - gravity) * dt
    return max(velocity, 0.0)


def estimate_apex_from_contact_profile(
    contact_force_profile_newtons: Sequence[float],
    athlete_mass_kg: float,
    takeoff_center_of_mass_height_m: float,
    total_duration_s: float,
    gravity: float = 9.81,
) -> float:
    """Estimate apex height from the surrogate contact-force profile."""

    takeoff_velocity = estimate_takeoff_velocity_from_contact_profile(
        contact_force_profile_newtons=contact_force_profile_newtons,
        athlete_mass_kg=athlete_mass_kg,
        total_duration_s=total_duration_s,
        gravity=gravity,
    )
    return estimate_jump_apex_height(
        center_of_mass_height=takeoff_center_of_mass_height_m,
        vertical_velocity=takeoff_velocity,
        gravity=gravity,
    )
