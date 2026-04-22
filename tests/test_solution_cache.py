"""Tests for the persistent OCP solution cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from synchro_jump.optimization.problem import VerticalJumpOcpSettings
from synchro_jump.optimization.runtime_solution import OcpSolveSummary
from synchro_jump.optimization.solution_cache import (
    cache_file_path,
    load_cached_solution_summary,
    save_cached_solution_summary,
)


def test_solution_cache_round_trip(tmp_path: Path) -> None:
    """One stored summary should be retrievable from the cache."""

    settings = VerticalJumpOcpSettings(athlete_mass_kg=VerticalJumpOcpSettings().mass_slider_values_kg[0])
    summary = OcpSolveSummary(
        success=True,
        message="ok",
        model_path=tmp_path / "model.bioMod",
        requested_iterations=1000,
        contact_model=settings.contact_model,
        time=np.array([0.0, 1.0]),
    )

    cache_path = save_cached_solution_summary(tmp_path, settings, 1100.0, 1000, summary)
    cached_summary = load_cached_solution_summary(tmp_path, settings, 1100.0, 1000)

    assert cache_path == cache_file_path(tmp_path, settings, 1100.0, 1000)
    assert cached_summary is not None
    assert cached_summary.success is True
    assert cached_summary.message == "ok"
