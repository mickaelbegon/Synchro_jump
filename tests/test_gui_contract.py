"""Non-visual tests for the GUI public API."""

from __future__ import annotations

from synchro_jump.gui.app import SynchroJumpApp


def test_gui_exposes_runtime_actions() -> None:
    """The GUI class should expose runtime build and solve actions."""

    assert callable(getattr(SynchroJumpApp, "build_ocp"))
    assert callable(getattr(SynchroJumpApp, "solve_ocp"))
    assert SynchroJumpApp.default_solve_iterations > 0
    assert SynchroJumpApp.max_solve_iterations >= SynchroJumpApp.default_solve_iterations
