"""Tests for the pure-Python avatar retargeting layer."""

from __future__ import annotations

import numpy as np

from synchro_jump.avatar_viewer.mapping import default_cc_base_mapping
from synchro_jump.avatar_viewer.math3d import (
    local_from_global,
    quaternion_from_axis_angle,
    quaternion_identity,
)
from synchro_jump.avatar_viewer.retargeting import BiomechanicalPose, BiomechanicalRetargeter


def test_retargeter_converts_global_rotations_to_local_parent_child_rotations() -> None:
    """Global input rotations should be converted into parent-relative local quaternions."""

    mapping = default_cc_base_mapping()
    retargeter = BiomechanicalRetargeter(mapping)

    root_global = quaternion_from_axis_angle((1.0, 0.0, 0.0), 0.2)
    pelvis_global = quaternion_from_axis_angle((1.0, 0.0, 0.0), 0.5)
    pose = BiomechanicalPose(
        time_s=0.0,
        global_rotations_xyzw_by_bone={
            "root": root_global,
            "pelvis": pelvis_global,
        },
    )

    rig_pose = retargeter.retarget_pose(pose)
    expected_pelvis_local = local_from_global(root_global, pelvis_global)

    np.testing.assert_allclose(
        rig_pose.local_quaternions_xyzw_by_rig_bone["CC_Base_Pelvis"],
        expected_pelvis_local,
        atol=1e-9,
    )


def test_retargeter_keeps_default_identity_for_non_driven_limbs() -> None:
    """Non-driven limb bones should remain on their configured default pose."""

    mapping = default_cc_base_mapping()
    retargeter = BiomechanicalRetargeter(mapping)

    rig_pose = retargeter.identity_pose()

    np.testing.assert_allclose(
        rig_pose.local_quaternions_xyzw_by_rig_bone["CC_Base_L_Thigh"],
        quaternion_identity(),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        rig_pose.local_quaternions_xyzw_by_rig_bone["CC_Base_R_Upperarm"],
        quaternion_identity(),
        atol=1e-9,
    )
