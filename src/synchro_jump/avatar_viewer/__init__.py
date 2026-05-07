"""3D avatar viewer helpers for biomechanical retargeting.

This module is intentionally optional: the pure-Python inspection and
retargeting utilities work without Panda3D or Qt, while the actual 3D viewer
loads those dependencies lazily.
"""

from synchro_jump.avatar_viewer.glb_inspector import GlbRigInspector, RigInspectionReport
from synchro_jump.avatar_viewer.mapping import RigBoneMapping, RigBoneMappingEntry, default_cc_base_mapping
from synchro_jump.avatar_viewer.retargeting import BiomechanicalPose, BiomechanicalRetargeter, RigPose
from synchro_jump.avatar_viewer.synthetic_motion import SyntheticQSeries, generate_demo_pose_sequence

__all__ = [
    "BiomechanicalPose",
    "BiomechanicalRetargeter",
    "GlbRigInspector",
    "RigBoneMapping",
    "RigBoneMappingEntry",
    "RigInspectionReport",
    "RigPose",
    "SyntheticQSeries",
    "default_cc_base_mapping",
    "generate_demo_pose_sequence",
]
