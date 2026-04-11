"""Runtime helpers to build and inspect the `bioptim` OCP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synchro_jump.optimization.bioptim_ocp import VerticalJumpBioptimOcpBuilder
from synchro_jump.optimization.problem import VerticalJumpOcpSettings


@dataclass(frozen=True)
class OcpRuntimeSummary:
    """Describe the runtime status of one OCP instantiation attempt."""

    success: bool
    message: str
    model_path: Path
    state_names: tuple[str, ...] = ()
    control_names: tuple[str, ...] = ()
    n_phases: int = 0


def build_ocp_runtime_summary(
    settings: VerticalJumpOcpSettings,
    peak_force_newtons: float,
    *,
    model_output_dir: str | Path = "generated",
) -> OcpRuntimeSummary:
    """Attempt to instantiate the runtime OCP and summarize the result."""

    builder = VerticalJumpBioptimOcpBuilder(settings=settings)
    model_path = builder.export_model(model_output_dir)

    try:
        ocp = builder.build_ocp(peak_force_newtons=peak_force_newtons, model_path=model_path)
    except ModuleNotFoundError as exc:
        dependency_name = getattr(exc, "name", None) or str(exc)
        return OcpRuntimeSummary(
            success=False,
            message=f"Dependance optionnelle manquante pour construire l'OCP: {dependency_name}",
            model_path=model_path,
        )
    except RuntimeError as exc:
        return OcpRuntimeSummary(
            success=False,
            message=str(exc),
            model_path=model_path,
        )
    except Exception as exc:  # pragma: no cover - runtime safety net
        return OcpRuntimeSummary(
            success=False,
            message=f"Echec de construction de l'OCP: {exc}",
            model_path=model_path,
        )

    return OcpRuntimeSummary(
        success=True,
        message="OCP bioptim construite avec succes.",
        model_path=model_path,
        state_names=tuple(ocp.nlp[0].states.keys()),
        control_names=tuple(ocp.nlp[0].controls.keys()),
        n_phases=ocp.n_phases,
    )
