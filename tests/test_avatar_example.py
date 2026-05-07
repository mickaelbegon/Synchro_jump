"""Tests for the standalone 3D avatar example launcher."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

import pytest

from synchro_jump.avatar_viewer.rigged_avatar import AvatarViewerDependencyError


EXAMPLE_PATH = Path("examples/run_avatar_gui.py").resolve()


def test_avatar_example_inspect_only_prints_report(monkeypatch, capsys) -> None:
    """The example should support a pure inspection mode without 3D dependencies."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EXAMPLE_PATH),
            "--inspect-only",
            "--asset",
            str(Path("assets/avatar_3d/rigged_character.glb").resolve()),
        ],
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")

    assert exit_info.value.code == 0
    captured = capsys.readouterr()
    assert "Hierarchy:" in captured.out
    assert "Back bones:" in captured.out
    assert "CC_Base_Spine02" in captured.out


def test_avatar_example_reports_missing_viewer_dependencies(monkeypatch, capsys) -> None:
    """The example should fail gracefully when Qt/Panda3D are unavailable."""

    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(EXAMPLE_PATH),
            "--asset",
            str(Path("assets/avatar_3d/rigged_character.glb").resolve()),
        ],
    )

    import synchro_jump.avatar_viewer.viewer_3d as viewer_3d

    monkeypatch.setattr(
        viewer_3d,
        "launch_avatar_viewer",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AvatarViewerDependencyError("missing runtime")),
    )

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_path(str(EXAMPLE_PATH), run_name="__main__")

    assert exit_info.value.code == 1
    captured = capsys.readouterr()
    assert "3D viewer dependencies are missing:" in captured.out
    assert "missing runtime" in captured.out
