"""Force-profile helpers for the moving platform actuation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PlatformForceProfile:
    """Piecewise-linear force profile used to actuate the platform.

    The force remains constant over the first part of the motion, then decreases
    linearly during the final ramp. The end value is half of the peak force.
    """

    peak_force_newtons: float
    total_duration: float = 2.0
    taper_duration: float = 0.3

    def __post_init__(self) -> None:
        """Validate the force-profile parameters."""

        if self.peak_force_newtons <= 0.0:
            raise ValueError("peak_force_newtons must be strictly positive")
        if self.total_duration <= 0.0:
            raise ValueError("total_duration must be strictly positive")
        if self.taper_duration <= 0.0:
            raise ValueError("taper_duration must be strictly positive")
        if self.taper_duration > self.total_duration:
            raise ValueError("taper_duration cannot exceed total_duration")

    @property
    def final_force_newtons(self) -> float:
        """Return the terminal force level."""

        return 0.5 * self.peak_force_newtons

    @property
    def ramp_start(self) -> float:
        """Return the time at which the decreasing ramp begins."""

        return self.total_duration - self.taper_duration

    def force_at(self, time: float) -> float:
        """Return the actuation force at one time instant."""

        if time <= 0.0:
            return self.peak_force_newtons
        if time >= self.total_duration:
            return self.final_force_newtons
        if time <= self.ramp_start:
            return self.peak_force_newtons

        normalized_time = (time - self.ramp_start) / self.taper_duration
        return self.peak_force_newtons * (1.0 - 0.5 * normalized_time)
