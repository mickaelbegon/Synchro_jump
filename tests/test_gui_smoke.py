"""GUI smoke tests."""

from __future__ import annotations

import os
import tkinter as tk

import pytest

pytest.importorskip("matplotlib")

if os.environ.get("ENABLE_TK_GUI_TESTS") != "1":
    pytest.skip("Tk GUI smoke tests are opt-in and require a live display session.", allow_module_level=True)

from synchro_jump.gui.app import SynchroJumpApp


def test_gui_builds_controls_and_status() -> None:
    """The GUI builds the requested sliders, axes, and summary text."""

    root = tk.Tk()
    root.withdraw()
    try:
        app = SynchroJumpApp(root)
        assert app.force_scale.cget("from") == pytest.approx(900.0)
        assert app.force_scale.cget("to") == pytest.approx(1300.0)
        assert app.mass_scale.cget("from") == pytest.approx(40.0)
        assert app.mass_scale.cget("to") == pytest.approx(55.0)
        assert "vitesse de decollage" in app.status_var.get()
        assert app.force_axis is not None
        assert app.pose_axis is not None
        assert app.kinematics_axis is not None
        assert callable(app.build_ocp)
        assert callable(app.solve_ocp)
    finally:
        root.destroy()
