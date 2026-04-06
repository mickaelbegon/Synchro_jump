"""Utilities to estimate jump performance from take-off conditions."""

from __future__ import annotations

import math


def estimate_jump_apex_height(
    center_of_mass_height: float,
    vertical_velocity: float,
    gravity: float = 9.81,
) -> float:
    """Estimate the apex height reached after take-off.

    Parameters
    ----------
    center_of_mass_height:
        Vertical center-of-mass position at take-off, in meters.
    vertical_velocity:
        Vertical center-of-mass velocity at take-off, in meters per second.
    gravity:
        Positive gravity magnitude, in meters per second squared.

    Returns
    -------
    float
        Estimated apex height, in meters.

    Raises
    ------
    ValueError
        If gravity is not strictly positive.
    """

    if gravity <= 0.0:
        raise ValueError("gravity must be strictly positive")

    ballistic_gain = max(vertical_velocity, 0.0) ** 2 / (2.0 * gravity)
    return center_of_mass_height + ballistic_gain
