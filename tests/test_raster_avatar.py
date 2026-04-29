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


def test_leg_sprite_uses_metatarsal_support_anchor() -> None:
    """The distal leg sprite anchor should sit at the forefoot support point."""

    spec = raster_avatar.sprite_spec("leg_foot")

    assert spec.filename == "jambe_pied_extension.png"
    assert spec.distal_anchor_px[0] == 591.0
    assert spec.distal_anchor_px[1] == 1110.0
