"""Synchro Jump package."""

from synchro_jump.modeling.athlete import AthleteMorphology
from synchro_jump.optimization.bioptim_ocp import VerticalJumpBioptimOcpBuilder, VerticalJumpOcpBlueprint
from synchro_jump.optimization.contact import PlatformInteractionModel
from synchro_jump.optimization.estimator import estimate_jump_apex_height
from synchro_jump.optimization.explicit_platform import (
    CoupledPlatformSolution,
    platform_actuation_force,
    predicted_apex_height_expression_numeric,
    solve_coupled_platform_dynamics_numeric,
)
from synchro_jump.optimization.force_profile import PlatformForceProfile
from synchro_jump.optimization.problem import VerticalJumpOcpSettings
from synchro_jump.optimization.runtime_solution import (
    OcpSolveSummary,
    evaluate_com_trajectory,
    solve_ocp_runtime_summary,
    summarize_solved_ocp,
)
from synchro_jump.optimization.runtime_summary import OcpRuntimeSummary, build_ocp_runtime_summary
from synchro_jump.optimization.surrogate import estimate_takeoff_velocity_from_contact_profile

__all__ = [
    "AthleteMorphology",
    "CoupledPlatformSolution",
    "OcpRuntimeSummary",
    "OcpSolveSummary",
    "PlatformForceProfile",
    "PlatformInteractionModel",
    "VerticalJumpBioptimOcpBuilder",
    "VerticalJumpOcpBlueprint",
    "VerticalJumpOcpSettings",
    "build_ocp_runtime_summary",
    "evaluate_com_trajectory",
    "estimate_jump_apex_height",
    "estimate_takeoff_velocity_from_contact_profile",
    "platform_actuation_force",
    "predicted_apex_height_expression_numeric",
    "solve_ocp_runtime_summary",
    "solve_coupled_platform_dynamics_numeric",
    "summarize_solved_ocp",
]
