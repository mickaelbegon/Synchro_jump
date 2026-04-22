"""Persistent cache for solved vertical-jump OCP summaries."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
from typing import TYPE_CHECKING

from synchro_jump.optimization.problem import VerticalJumpOcpSettings

if TYPE_CHECKING:
    from synchro_jump.optimization.runtime_solution import OcpSolveSummary


def cache_key_payload(
    settings: VerticalJumpOcpSettings,
    peak_force_newtons: float,
    maximum_iterations: int,
) -> dict[str, object]:
    """Return the serializable payload used to identify one cached solve."""

    payload = asdict(settings)
    payload["peak_force_newtons"] = float(peak_force_newtons)
    payload["maximum_iterations"] = int(maximum_iterations)
    return payload


def cache_key_digest(
    settings: VerticalJumpOcpSettings,
    peak_force_newtons: float,
    maximum_iterations: int,
) -> str:
    """Return the stable digest associated with one solve request."""

    serialized = json.dumps(
        cache_key_payload(settings, peak_force_newtons, maximum_iterations),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def cache_file_path(
    cache_dir: str | Path,
    settings: VerticalJumpOcpSettings,
    peak_force_newtons: float,
    maximum_iterations: int,
) -> Path:
    """Return the cache-file path associated with one solve request."""

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{cache_key_digest(settings, peak_force_newtons, maximum_iterations)}.pkl"


def load_cached_solution_summary(
    cache_dir: str | Path,
    settings: VerticalJumpOcpSettings,
    peak_force_newtons: float,
    maximum_iterations: int,
) -> "OcpSolveSummary | None":
    """Load one cached OCP solution summary when available."""

    cache_path = cache_file_path(cache_dir, settings, peak_force_newtons, maximum_iterations)
    if not cache_path.exists():
        return None

    with cache_path.open("rb") as cache_stream:
        cached_summary = pickle.load(cache_stream)
    if not hasattr(cached_summary, "success") or not hasattr(cached_summary, "state_trajectories"):
        return None
    return cached_summary


def save_cached_solution_summary(
    cache_dir: str | Path,
    settings: VerticalJumpOcpSettings,
    peak_force_newtons: float,
    maximum_iterations: int,
    summary: "OcpSolveSummary",
) -> Path:
    """Persist one solved OCP summary for later reuse."""

    cache_path = cache_file_path(cache_dir, settings, peak_force_newtons, maximum_iterations)
    with cache_path.open("wb") as cache_stream:
        pickle.dump(summary, cache_stream)
    return cache_path
