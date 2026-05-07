"""Biomechanical-to-rig retargeting primitives."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from synchro_jump.avatar_viewer.mapping import RigBoneMapping
from synchro_jump.avatar_viewer.math3d import (
    apply_frame_correction,
    local_from_global,
    quaternion_identity,
    normalize_quaternion,
)


@dataclass(frozen=True)
class BiomechanicalPose:
    """Pose description sent to the retargeter.

    Either `global_rotations_xyzw_by_bone` or `local_rotations_xyzw_by_bone` can
    be provided. The retargeter converts global rotations into parent-relative
    local rotations before applying rig-specific axis corrections.
    """

    time_s: float
    root_translation_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    global_rotations_xyzw_by_bone: dict[str, np.ndarray] = field(default_factory=dict)
    local_rotations_xyzw_by_bone: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class RigPose:
    """Local rig pose ready to be applied bone-by-bone."""

    time_s: float
    root_translation_xyz_m: tuple[float, float, float]
    local_quaternions_xyzw_by_rig_bone: dict[str, np.ndarray]


class BiomechanicalRetargeter:
    """Convert biomechanical rotations into local rig quaternions."""

    def __init__(self, mapping: RigBoneMapping) -> None:
        self.mapping = mapping

    def _local_source_rotations(self, pose: BiomechanicalPose) -> dict[str, np.ndarray]:
        """Return local source rotations from either local or global input."""

        if pose.local_rotations_xyzw_by_bone:
            return {
                name: normalize_quaternion(quaternion)
                for name, quaternion in pose.local_rotations_xyzw_by_bone.items()
            }

        local_rotations: dict[str, np.ndarray] = {}
        for biomechanical_name, entry in self.mapping.entries_by_biomechanical_name.items():
            global_quaternion = pose.global_rotations_xyzw_by_bone.get(biomechanical_name)
            if global_quaternion is None:
                continue
            if entry.parent_biomechanical_name and entry.parent_biomechanical_name in pose.global_rotations_xyzw_by_bone:
                parent_global = pose.global_rotations_xyzw_by_bone[entry.parent_biomechanical_name]
                local_rotations[biomechanical_name] = local_from_global(parent_global, global_quaternion)
            else:
                local_rotations[biomechanical_name] = normalize_quaternion(global_quaternion)
        return local_rotations

    def retarget_pose(self, pose: BiomechanicalPose) -> RigPose:
        """Return one local rig pose for Panda3D joint control.

        Critical step:
        1. Convert optional global rotations into local parent-child rotations.
        2. Apply the per-bone axis correction between the biomechanical frame and
           the graphical rig frame.
        3. Fill non-driven bones with their configured default local rotation so
           that arms and legs can remain aesthetically fixed for now.
        """

        source_local_rotations = self._local_source_rotations(pose)
        rig_local_quaternions: dict[str, np.ndarray] = {}
        for biomechanical_name, entry in self.mapping.entries_by_biomechanical_name.items():
            source_local = source_local_rotations.get(biomechanical_name, entry.default_local_quaternion_numpy)
            if not entry.is_driven and biomechanical_name not in source_local_rotations:
                source_local = entry.default_local_quaternion_numpy
            corrected_local = apply_frame_correction(source_local, entry.axis_correction_numpy)
            rig_local_quaternions[entry.rig_name] = corrected_local

        axis_order = self.mapping.translation_axis_order_xyz
        source_translation = np.asarray(pose.root_translation_xyz_m, dtype=float).reshape((3,))
        reordered_translation = tuple(float(source_translation[index]) for index in axis_order)
        return RigPose(
            time_s=float(pose.time_s),
            root_translation_xyz_m=reordered_translation,
            local_quaternions_xyzw_by_rig_bone=rig_local_quaternions,
        )

    def identity_pose(self, *, time_s: float = 0.0) -> RigPose:
        """Return one neutral pose aligned with the mapping defaults."""

        return self.retarget_pose(
            BiomechanicalPose(
                time_s=time_s,
                local_rotations_xyzw_by_bone={
                    biomechanical_name: quaternion_identity()
                    for biomechanical_name, entry in self.mapping.entries_by_biomechanical_name.items()
                    if entry.is_driven
                },
            )
        )
