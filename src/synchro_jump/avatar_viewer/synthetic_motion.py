"""Synthetic q(t) and pose-sequence helpers for the 3D avatar prototype."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from synchro_jump.avatar_viewer.math3d import quaternion_from_axis_angle
from synchro_jump.avatar_viewer.retargeting import BiomechanicalPose


@dataclass(frozen=True)
class SyntheticQSeries:
    """Small synthetic motion bundle used by the demo viewer."""

    time_s: np.ndarray
    root_translation_xyz_m: np.ndarray
    root_pitch_rad: np.ndarray
    pelvis_pitch_rad: np.ndarray
    waist_pitch_rad: np.ndarray
    spine_lower_pitch_rad: np.ndarray
    spine_upper_pitch_rad: np.ndarray
    neck_pitch_rad: np.ndarray
    head_pitch_rad: np.ndarray


def generate_demo_q_series(
    *,
    duration_s: float = 4.0,
    frames_per_second: int = 60,
) -> SyntheticQSeries:
    """Return one light synthetic q(t) focused on the root and back chain."""

    sample_count = max(int(duration_s * frames_per_second), 2)
    time = np.linspace(0.0, duration_s, sample_count)
    phase = 2.0 * math.pi * time / duration_s

    root_vertical = 0.02 * (1.0 - np.cos(phase))
    root_ap = 0.03 * np.sin(phase)
    root_translation = np.column_stack((root_ap, np.zeros_like(time), root_vertical))

    root_pitch = np.deg2rad(6.0) * np.sin(phase)
    pelvis_pitch = np.deg2rad(4.0) * np.sin(phase)
    waist_pitch = np.deg2rad(8.0) * np.sin(phase)
    spine_lower_pitch = np.deg2rad(12.0) * np.sin(phase)
    spine_upper_pitch = np.deg2rad(10.0) * np.sin(phase)
    neck_pitch = np.deg2rad(4.0) * np.sin(phase)
    head_pitch = np.deg2rad(2.0) * np.sin(phase)

    return SyntheticQSeries(
        time_s=time,
        root_translation_xyz_m=root_translation,
        root_pitch_rad=root_pitch,
        pelvis_pitch_rad=pelvis_pitch,
        waist_pitch_rad=waist_pitch,
        spine_lower_pitch_rad=spine_lower_pitch,
        spine_upper_pitch_rad=spine_upper_pitch,
        neck_pitch_rad=neck_pitch,
        head_pitch_rad=head_pitch,
    )


def generate_demo_pose_sequence(
    *,
    duration_s: float = 4.0,
    frames_per_second: int = 60,
) -> tuple[SyntheticQSeries, list[BiomechanicalPose]]:
    """Return both synthetic q(t) arrays and the corresponding pose sequence."""

    q_series = generate_demo_q_series(duration_s=duration_s, frames_per_second=frames_per_second)
    poses: list[BiomechanicalPose] = []
    for frame_index, time_s in enumerate(q_series.time_s):
        poses.append(
            BiomechanicalPose(
                time_s=float(time_s),
                root_translation_xyz_m=tuple(float(value) for value in q_series.root_translation_xyz_m[frame_index]),
                local_rotations_xyzw_by_bone={
                    "root": quaternion_from_axis_angle((1.0, 0.0, 0.0), q_series.root_pitch_rad[frame_index]),
                    "pelvis": quaternion_from_axis_angle((1.0, 0.0, 0.0), q_series.pelvis_pitch_rad[frame_index]),
                    "waist": quaternion_from_axis_angle((1.0, 0.0, 0.0), q_series.waist_pitch_rad[frame_index]),
                    "spine_lower": quaternion_from_axis_angle(
                        (1.0, 0.0, 0.0), q_series.spine_lower_pitch_rad[frame_index]
                    ),
                    "spine_upper": quaternion_from_axis_angle(
                        (1.0, 0.0, 0.0), q_series.spine_upper_pitch_rad[frame_index]
                    ),
                    "neck": quaternion_from_axis_angle((1.0, 0.0, 0.0), q_series.neck_pitch_rad[frame_index]),
                    "head": quaternion_from_axis_angle((1.0, 0.0, 0.0), q_series.head_pitch_rad[frame_index]),
                },
            )
        )
    return q_series, poses
