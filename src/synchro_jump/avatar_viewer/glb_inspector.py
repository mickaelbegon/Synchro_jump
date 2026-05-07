"""Pure-Python GLB inspection helpers used before the runtime viewer is available."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import struct

from synchro_jump.avatar_viewer.mapping import RigBoneMapping, default_cc_base_mapping


JSON_CHUNK_TYPE = 0x4E4F534A
GLB_MAGIC = b"glTF"


@dataclass(frozen=True)
class GlbNode:
    """One node extracted from the GLB JSON scene description."""

    index: int
    name: str
    children: tuple[int, ...]
    translation_xyz: tuple[float, float, float]
    rotation_xyzw: tuple[float, float, float, float]
    skin: int | None


@dataclass(frozen=True)
class RigInspectionReport:
    """Formatted rig inspection output for logs or GUI side panels."""

    hierarchy_lines: tuple[str, ...]
    spine_bones: tuple[str, ...]
    recognized_bones: tuple[str, ...]
    missing_mapping_entries: tuple[str, ...]

    def to_multiline_text(self) -> str:
        """Return one readable text report."""

        missing_lines = [f"- {name}" for name in self.missing_mapping_entries] if self.missing_mapping_entries else ["- none"]
        sections = [
            "Hierarchy:",
            *self.hierarchy_lines,
            "",
            "Back bones:",
            *(f"- {name}" for name in self.spine_bones),
            "",
            "Recognized mapping:",
            *(f"- {name}" for name in self.recognized_bones),
            "",
            "Missing mapping entries:",
            *missing_lines,
        ]
        return "\n".join(sections)


class GlbRigInspector:
    """Inspect one rigged GLB file without loading a full 3D engine."""

    def __init__(
        self,
        asset_path: Path,
        nodes: tuple[GlbNode, ...],
        scene_root_indices: tuple[int, ...],
        skin_joint_indices: tuple[int, ...],
    ) -> None:
        self.asset_path = Path(asset_path)
        self.nodes = nodes
        self.scene_root_indices = scene_root_indices
        self.skin_joint_indices = skin_joint_indices
        self._nodes_by_index = {node.index: node for node in nodes}
        self._nodes_by_name = {node.name: node for node in nodes}

    @classmethod
    def from_glb(cls, asset_path: Path) -> "GlbRigInspector":
        """Parse the GLB JSON chunk and build one inspector instance."""

        path = Path(asset_path)
        if not path.exists():
            raise FileNotFoundError(f"Avatar GLB not found: {path}")

        with path.open("rb") as file:
            magic, _version, _length = struct.unpack("<4sII", file.read(12))
            if magic != GLB_MAGIC:
                raise ValueError(f"Invalid GLB header for asset: {path}")
            chunk_length, chunk_type = struct.unpack("<II", file.read(8))
            if chunk_type != JSON_CHUNK_TYPE:
                raise ValueError(f"First GLB chunk is not JSON for asset: {path}")
            scene_description = json.loads(file.read(chunk_length).decode("utf-8"))

        nodes = []
        for index, node_data in enumerate(scene_description.get("nodes", [])):
            nodes.append(
                GlbNode(
                    index=index,
                    name=node_data.get("name", f"node_{index}"),
                    children=tuple(node_data.get("children", [])),
                    translation_xyz=tuple(node_data.get("translation", [0.0, 0.0, 0.0])),
                    rotation_xyzw=tuple(node_data.get("rotation", [0.0, 0.0, 0.0, 1.0])),
                    skin=node_data.get("skin"),
                )
            )

        scenes = scene_description.get("scenes", [])
        default_scene_index = scene_description.get("scene", 0)
        scene_root_indices = tuple(scenes[default_scene_index].get("nodes", [])) if scenes else tuple()
        skin_joint_indices = tuple(scene_description.get("skins", [{}])[0].get("joints", [])) if scene_description.get("skins") else tuple()
        return cls(path, tuple(nodes), scene_root_indices, skin_joint_indices)

    def node_names(self) -> tuple[str, ...]:
        """Return all node names in file order."""

        return tuple(node.name for node in self.nodes)

    def hierarchy_lines(self) -> tuple[str, ...]:
        """Return the full hierarchy as indented text lines."""

        lines: list[str] = []

        def walk(node_index: int, indent: int) -> None:
            node = self._nodes_by_index[node_index]
            lines.append(f"{'  ' * indent}- {node.name}")
            for child_index in node.children:
                walk(child_index, indent + 1)

        for root_index in self.scene_root_indices:
            walk(root_index, 0)
        return tuple(lines)

    def recognized_mapping_entries(self, mapping: RigBoneMapping | None = None) -> tuple[str, ...]:
        """Return the biomechanical names whose target rig bones are present."""

        selected_mapping = mapping or default_cc_base_mapping()
        recognized = []
        for biomechanical_name, entry in selected_mapping.entries_by_biomechanical_name.items():
            if entry.rig_name in self._nodes_by_name:
                recognized.append(f"{biomechanical_name} -> {entry.rig_name}")
        return tuple(recognized)

    def missing_mapping_entries(self, mapping: RigBoneMapping | None = None) -> tuple[str, ...]:
        """Return the mapping entries not found in the GLB skeleton."""

        selected_mapping = mapping or default_cc_base_mapping()
        missing = []
        for biomechanical_name, entry in selected_mapping.entries_by_biomechanical_name.items():
            if entry.rig_name not in self._nodes_by_name:
                missing.append(f"{biomechanical_name} -> {entry.rig_name}")
        return tuple(missing)

    def spine_bones(self, mapping: RigBoneMapping | None = None) -> tuple[str, ...]:
        """Return the recognized spine/back chain for the selected mapping."""

        selected_mapping = mapping or default_cc_base_mapping()
        bones = []
        for biomechanical_name in selected_mapping.spine_chain_biomechanical_names:
            entry = selected_mapping.entries_by_biomechanical_name.get(biomechanical_name)
            if entry and entry.rig_name in self._nodes_by_name:
                bones.append(entry.rig_name)
        return tuple(bones)

    def build_report(self, mapping: RigBoneMapping | None = None) -> RigInspectionReport:
        """Return one formatted rig-inspection report."""

        selected_mapping = mapping or default_cc_base_mapping()
        return RigInspectionReport(
            hierarchy_lines=self.hierarchy_lines(),
            spine_bones=self.spine_bones(selected_mapping),
            recognized_bones=self.recognized_mapping_entries(selected_mapping),
            missing_mapping_entries=self.missing_mapping_entries(selected_mapping),
        )
