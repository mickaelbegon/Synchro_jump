"""Launch the optional 3D avatar prototype."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def parse_args() -> argparse.Namespace:
    """Parse the small set of viewer CLI options."""

    parser = argparse.ArgumentParser(description="Launch the Synchro Jump 3D avatar prototype.")
    parser.add_argument(
        "--asset",
        type=Path,
        default=Path("assets/avatar_3d/rigged_character.glb"),
        help="Path to the rigged GLB asset to display.",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Print the rig inspection report without launching the GUI.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the rig inspector and optionally launch the 3D viewer."""

    from synchro_jump.avatar_viewer import GlbRigInspector, default_cc_base_mapping, generate_demo_pose_sequence
    from synchro_jump.avatar_viewer.rigged_avatar import AvatarViewerDependencyError
    from synchro_jump.avatar_viewer.viewer_3d import default_avatar_viewer_config, launch_avatar_viewer

    args = parse_args()
    asset_path = args.asset.resolve()
    inspector = GlbRigInspector.from_glb(asset_path)
    report = inspector.build_report(default_cc_base_mapping())
    print(report.to_multiline_text())

    if args.inspect_only:
        return 0

    try:
        _q_series, poses = generate_demo_pose_sequence()
        return launch_avatar_viewer(default_avatar_viewer_config(asset_path), poses)
    except AvatarViewerDependencyError as exc:
        print("")
        print("3D viewer dependencies are missing:")
        print(exc)
        print("")
        print("Suggested installation:")
        print("  python -m pip install -e .[avatar3d]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
