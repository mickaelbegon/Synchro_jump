"""Anthropometric helpers for the reduced planar jumper model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SegmentLengths:
    """Lengths of the reduced three-segment jumper model."""

    leg_foot: float
    thigh: float
    trunk: float


@dataclass(frozen=True)
class AthleteMorphology:
    """Anthropometry for the reduced planar jumper.

    The reduced model merges the shank and foot into one distal segment, keeps
    the thigh as the middle segment, and lumps the remaining body into a trunk
    segment attached at the hip.
    """

    height_m: float = 1.60
    mass_kg: float = 50.0

    def __post_init__(self) -> None:
        """Validate the anthropometric inputs."""

        if self.height_m <= 0.0:
            raise ValueError("height_m must be strictly positive")
        if self.mass_kg <= 0.0:
            raise ValueError("mass_kg must be strictly positive")

    @property
    def segment_lengths(self) -> SegmentLengths:
        """Return reduced segment lengths derived from standing height.

        The proportions are chosen for a compact planar jumper model:
        - leg_foot: 29% of height
        - thigh: 24.5% of height
        - trunk: remaining height
        """

        leg_foot = 0.29 * self.height_m
        thigh = 0.245 * self.height_m
        trunk = self.height_m - leg_foot - thigh
        return SegmentLengths(leg_foot=leg_foot, thigh=thigh, trunk=trunk)

    @property
    def total_mass_kg(self) -> float:
        """Return the athlete mass for convenience."""

        return self.mass_kg

    @property
    def initial_joint_flexion_deg(self) -> float:
        """Return the default initial hip and knee flexion angle."""

        return 100.0
