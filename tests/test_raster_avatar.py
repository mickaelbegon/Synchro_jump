"""Unit tests for the raster avatar diagnostics."""

from __future__ import annotations

from pathlib import Path

import synchro_jump.gui.raster_avatar as raster_avatar


def test_avatar_rendering_diagnostics_reports_missing_pillow(monkeypatch) -> None:
    """The diagnostics should explain when Pillow is unavailable."""

    raster_avatar.avatar_rendering_diagnostics.cache_clear()
    monkeypatch.setattr(raster_avatar, "pillow_available", lambda: False)

    available, message = raster_avatar.avatar_rendering_diagnostics()

    assert available is False
    assert "Pillow" in message
    raster_avatar.avatar_rendering_diagnostics.cache_clear()


def test_avatar_rendering_diagnostics_reports_missing_assets(monkeypatch, tmp_path: Path) -> None:
    """The diagnostics should explain when one raster asset is missing."""

    raster_avatar.avatar_rendering_diagnostics.cache_clear()
    monkeypatch.setattr(raster_avatar, "pillow_available", lambda: True)
    monkeypatch.setattr(raster_avatar, "_asset_dir", lambda: tmp_path)

    available, message = raster_avatar.avatar_rendering_diagnostics()

    assert available is False
    assert "assets raster manquants" in message
    raster_avatar.avatar_rendering_diagnostics.cache_clear()
