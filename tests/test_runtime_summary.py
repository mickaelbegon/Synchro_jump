"""Tests for runtime OCP summaries."""

from __future__ import annotations

from pathlib import Path

from synchro_jump.optimization.bioptim_ocp import VerticalJumpBioptimOcpBuilder
from synchro_jump.optimization.problem import VerticalJumpOcpSettings
from synchro_jump.optimization.runtime_summary import build_ocp_runtime_summary


class _FakeVariablePool:
    """Minimal variable pool exposing one `keys()` method."""

    def __init__(self, keys: tuple[str, ...]) -> None:
        self._keys = keys

    def keys(self):
        """Return the configured variable names."""

        return self._keys


class _FakeNlp:
    """Minimal fake phase container."""

    def __init__(self) -> None:
        self.states = _FakeVariablePool(("q_roots", "q_joints"))
        self.controls = _FakeVariablePool(("tau_joints",))


class _FakeOcp:
    """Minimal fake OCP used to test the summary wrapper."""

    def __init__(self) -> None:
        self.nlp = [_FakeNlp()]
        self.n_phases = 1


def test_build_ocp_runtime_summary_reports_missing_optional_dependency(monkeypatch, tmp_path: Path) -> None:
    """Missing runtime dependencies should produce a friendly summary."""

    model_path = tmp_path / "jumper.bioMod"
    model_path.write_text("version 4\n", encoding="utf-8")

    monkeypatch.setattr(VerticalJumpBioptimOcpBuilder, "export_model", lambda self, _path: model_path)

    def fail_build(self, peak_force_newtons: float, *, model_path: Path, final_time_guess: float = 1.0):
        _ = peak_force_newtons
        _ = model_path
        _ = final_time_guess
        raise ModuleNotFoundError("No module named 'bioptim'")

    monkeypatch.setattr(VerticalJumpBioptimOcpBuilder, "build_ocp", fail_build)

    summary = build_ocp_runtime_summary(VerticalJumpOcpSettings(athlete_mass_kg=50.0), 1100.0)

    assert not summary.success
    assert "Dependance optionnelle manquante" in summary.message
    assert summary.model_path == model_path


def test_build_ocp_runtime_summary_exposes_state_and_control_names(monkeypatch, tmp_path: Path) -> None:
    """Successful runtime builds should summarize the OCP structure."""

    model_path = tmp_path / "jumper.bioMod"
    model_path.write_text("version 4\n", encoding="utf-8")

    monkeypatch.setattr(VerticalJumpBioptimOcpBuilder, "export_model", lambda self, _path: model_path)
    monkeypatch.setattr(VerticalJumpBioptimOcpBuilder, "build_ocp", lambda self, **kwargs: _FakeOcp())

    summary = build_ocp_runtime_summary(VerticalJumpOcpSettings(athlete_mass_kg=50.0), 1100.0)

    assert summary.success
    assert summary.n_phases == 1
    assert summary.state_names == ("q_roots", "q_joints")
    assert summary.control_names == ("tau_joints",)
