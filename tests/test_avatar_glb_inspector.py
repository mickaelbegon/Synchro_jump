"""Tests for the pure-Python GLB rig inspector."""

from __future__ import annotations

from pathlib import Path

from synchro_jump.avatar_viewer.glb_inspector import GlbRigInspector
from synchro_jump.avatar_viewer.mapping import default_cc_base_mapping


ASSET_PATH = Path("assets/avatar_3d/rigged_character.glb")


def test_glb_inspector_recovers_expected_cc_base_bones() -> None:
    """The bundled GLB should expose the expected CC Base back-chain names."""

    inspector = GlbRigInspector.from_glb(ASSET_PATH)
    report = inspector.build_report(default_cc_base_mapping())

    assert "CC_Base_Hip" in report.spine_bones
    assert "CC_Base_Spine01" in report.spine_bones
    assert "CC_Base_Spine02" in report.spine_bones
    assert "CC_Base_Head" in report.spine_bones
    assert not report.missing_mapping_entries


def test_glb_inspector_hierarchy_mentions_armature_root() -> None:
    """The hierarchy text should include the GLB armature root."""

    inspector = GlbRigInspector.from_glb(ASSET_PATH)
    hierarchy_text = "\n".join(inspector.hierarchy_lines())

    assert "Armature.001" in hierarchy_text
    assert "CC_Base_Hip" in hierarchy_text
