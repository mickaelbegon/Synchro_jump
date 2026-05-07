"""Default rig mapping for the bundled CC Base artistic-skater avatar."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from synchro_jump.avatar_viewer.math3d import quaternion_from_euler_deg_xyz, quaternion_identity


@dataclass(frozen=True)
class RigBoneMappingEntry:
    """Mapping metadata between one biomechanical bone and one rig bone."""

    biomechanical_name: str
    rig_name: str
    parent_biomechanical_name: str | None
    axis_correction_quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    default_local_quaternion_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    is_driven: bool = True

    @property
    def axis_correction_numpy(self) -> np.ndarray:
        """Return the axis correction as one normalized numpy quaternion."""

        return np.asarray(self.axis_correction_quaternion_xyzw, dtype=float)

    @property
    def default_local_quaternion_numpy(self) -> np.ndarray:
        """Return the default local rotation as one numpy quaternion."""

        return np.asarray(self.default_local_quaternion_xyzw, dtype=float)


@dataclass(frozen=True)
class RigBoneMapping:
    """Collection of rig-bone mapping entries and scene-level conventions."""

    root_biomechanical_name: str
    root_rig_bone_name: str
    entries_by_biomechanical_name: dict[str, RigBoneMappingEntry] = field(default_factory=dict)
    spine_chain_biomechanical_names: tuple[str, ...] = ("pelvis", "waist", "spine_lower", "spine_upper", "neck", "head")
    translation_axis_order_xyz: tuple[int, int, int] = (0, 1, 2)

    def entry(self, biomechanical_name: str) -> RigBoneMappingEntry:
        """Return one mapping entry or raise a helpful error."""

        try:
            return self.entries_by_biomechanical_name[biomechanical_name]
        except KeyError as exc:
            raise KeyError(f"Unknown biomechanical bone in mapping: {biomechanical_name}") from exc

    def recognized_rig_names(self) -> tuple[str, ...]:
        """Return the rig-bone names referenced by the mapping."""

        return tuple(entry.rig_name for entry in self.entries_by_biomechanical_name.values())


def _entry(
    biomechanical_name: str,
    rig_name: str,
    parent_biomechanical_name: str | None,
    *,
    axis_correction_deg_xyz=(0.0, 0.0, 0.0),
    default_local_deg_xyz=(0.0, 0.0, 0.0),
    is_driven: bool = True,
) -> RigBoneMappingEntry:
    """Build one mapping entry with readable degree-based defaults."""

    return RigBoneMappingEntry(
        biomechanical_name=biomechanical_name,
        rig_name=rig_name,
        parent_biomechanical_name=parent_biomechanical_name,
        axis_correction_quaternion_xyzw=tuple(quaternion_from_euler_deg_xyz(axis_correction_deg_xyz)),
        default_local_quaternion_xyzw=tuple(quaternion_from_euler_deg_xyz(default_local_deg_xyz)),
        is_driven=is_driven,
    )


def default_cc_base_mapping() -> RigBoneMapping:
    """Return a conservative first-pass mapping for the bundled CC Base rig.

    The current prototype intentionally drives only the root/back chain by
    default. Limbs are recognized and kept available in the mapping, but they
    remain fixed at the rig rest pose until the biomechanical mapping is
    validated interactively.
    """

    entries = {
        "root": _entry("root", "CC_Base_Hip", None, is_driven=True),
        "pelvis": _entry("pelvis", "CC_Base_Pelvis", "root", is_driven=True),
        "waist": _entry("waist", "CC_Base_Waist", "pelvis", is_driven=True),
        "spine_lower": _entry("spine_lower", "CC_Base_Spine01", "waist", is_driven=True),
        "spine_upper": _entry("spine_upper", "CC_Base_Spine02", "spine_lower", is_driven=True),
        "neck": _entry("neck", "CC_Base_NeckTwist01", "spine_upper", is_driven=True),
        "head": _entry("head", "CC_Base_Head", "neck", is_driven=True),
        "left_thigh": _entry("left_thigh", "CC_Base_L_Thigh", "pelvis", is_driven=False),
        "left_shin": _entry("left_shin", "CC_Base_L_Calf", "left_thigh", is_driven=False),
        "left_foot": _entry("left_foot", "CC_Base_L_Foot", "left_shin", is_driven=False),
        "right_thigh": _entry("right_thigh", "CC_Base_R_Thigh", "pelvis", is_driven=False),
        "right_shin": _entry("right_shin", "CC_Base_R_Calf", "right_thigh", is_driven=False),
        "right_foot": _entry("right_foot", "CC_Base_R_Foot", "right_shin", is_driven=False),
        "left_clavicle": _entry("left_clavicle", "CC_Base_L_Clavicle", "spine_upper", is_driven=False),
        "left_upperarm": _entry("left_upperarm", "CC_Base_L_Upperarm", "left_clavicle", is_driven=False),
        "left_forearm": _entry("left_forearm", "CC_Base_L_Forearm", "left_upperarm", is_driven=False),
        "left_hand": _entry("left_hand", "CC_Base_L_Hand", "left_forearm", is_driven=False),
        "right_clavicle": _entry("right_clavicle", "CC_Base_R_Clavicle", "spine_upper", is_driven=False),
        "right_upperarm": _entry("right_upperarm", "CC_Base_R_Upperarm", "right_clavicle", is_driven=False),
        "right_forearm": _entry("right_forearm", "CC_Base_R_Forearm", "right_upperarm", is_driven=False),
        "right_hand": _entry("right_hand", "CC_Base_R_Hand", "right_forearm", is_driven=False),
    }
    return RigBoneMapping(
        root_biomechanical_name="root",
        root_rig_bone_name="CC_Base_Hip",
        entries_by_biomechanical_name=entries,
        spine_chain_biomechanical_names=("root", "pelvis", "waist", "spine_lower", "spine_upper", "neck", "head"),
        translation_axis_order_xyz=(0, 1, 2),
    )
