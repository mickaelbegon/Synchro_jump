"""Panda3D rig loading and joint-control helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from synchro_jump.avatar_viewer.glb_inspector import GlbRigInspector
from synchro_jump.avatar_viewer.mapping import RigBoneMapping
from synchro_jump.avatar_viewer.retargeting import BiomechanicalPose, BiomechanicalRetargeter, RigPose


class AvatarViewerDependencyError(RuntimeError):
    """Raised when optional 3D viewer dependencies are missing."""


class AvatarAssetNotFoundError(FileNotFoundError):
    """Raised when the requested avatar file cannot be loaded."""


class AvatarRigMappingError(RuntimeError):
    """Raised when the graphical rig does not match the expected mapping."""


def _require_panda3d():
    """Import Panda3D lazily and raise one explicit installation hint."""

    try:
        from direct.actor.Actor import Actor  # type: ignore
        from panda3d.core import LQuaternionf  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise AvatarViewerDependencyError(
            "Panda3D dependencies are missing. Install `panda3d` and `panda3d-gltf` "
            "alongside `PySide6` to enable the 3D avatar viewer."
        ) from exc
    return Actor, LQuaternionf


@dataclass
class PandaRiggedAvatar:
    """Loaded Panda3D actor with controlled joints."""

    actor: object
    controlled_joint_nodes_by_rig_name: dict[str, object]
    root_joint_node: object

    def apply_rig_pose(self, rig_pose: RigPose) -> None:
        """Apply one local rig pose to the controlled Panda joints.

        Panda3D expects local rotations to be applied bone-by-bone on the joint
        controllers returned by `controlJoint()`. This keeps the mesh static and
        only updates the skeleton pose each frame.
        """

        _Actor, LQuaternionf = _require_panda3d()
        root_translation = rig_pose.root_translation_xyz_m
        self.root_joint_node.setPos(*root_translation)
        for rig_name, quaternion_xyzw in rig_pose.local_quaternions_xyzw_by_rig_bone.items():
            joint_node = self.controlled_joint_nodes_by_rig_name.get(rig_name)
            if joint_node is None:
                continue
            x, y, z, w = np.asarray(quaternion_xyzw, dtype=float).reshape((4,))
            joint_node.setQuat(LQuaternionf(float(w), float(x), float(y), float(z)))


class RiggedAvatarLoader:
    """Load one rigged avatar and expose controlled joints."""

    def __init__(self, asset_path: Path, mapping: RigBoneMapping) -> None:
        self.asset_path = Path(asset_path)
        self.mapping = mapping
        self.inspector = GlbRigInspector.from_glb(self.asset_path)

    def load(self, parent_node) -> PandaRiggedAvatar:
        """Load the avatar actor under one Panda3D parent node."""

        if not self.asset_path.exists():
            raise AvatarAssetNotFoundError(f"Avatar asset not found: {self.asset_path}")

        missing_mapping_entries = self.inspector.missing_mapping_entries(self.mapping)
        if missing_mapping_entries:
            joined = ", ".join(missing_mapping_entries)
            raise AvatarRigMappingError(
                "The avatar rig does not match the expected mapping. Missing entries: "
                f"{joined}"
            )

        Actor, _LQuaternionf = _require_panda3d()
        try:
            actor = Actor(str(self.asset_path))
        except Exception as exc:  # pragma: no cover - optional dependency
            raise AvatarAssetNotFoundError(
                "Unable to load the avatar asset. If the file is a .glb, "
                "make sure `panda3d-gltf` is installed and importable."
            ) from exc

        actor.reparentTo(parent_node)
        controlled_joints: dict[str, object] = {}
        for entry in self.mapping.entries_by_biomechanical_name.values():
            joint_node = actor.controlJoint(None, "modelRoot", entry.rig_name)
            if joint_node is None or joint_node.isEmpty():
                raise AvatarRigMappingError(
                    f"Panda3D could not control rig joint `{entry.rig_name}`. "
                    "Check the rig names or the loader backend."
                )
            controlled_joints[entry.rig_name] = joint_node

        root_joint_node = controlled_joints[self.mapping.root_rig_bone_name]
        return PandaRiggedAvatar(
            actor=actor,
            controlled_joint_nodes_by_rig_name=controlled_joints,
            root_joint_node=root_joint_node,
        )


class BiomechanicalRigController:
    """Bridge one biomechanical pose source to one Panda3D avatar."""

    def __init__(self, avatar: PandaRiggedAvatar, retargeter: BiomechanicalRetargeter) -> None:
        self.avatar = avatar
        self.retargeter = retargeter

    def apply_biomechanical_pose(self, pose: BiomechanicalPose) -> RigPose:
        """Retarget and apply one pose to the avatar rig."""

        rig_pose = self.retargeter.retarget_pose(pose)
        self.avatar.apply_rig_pose(rig_pose)
        return rig_pose
