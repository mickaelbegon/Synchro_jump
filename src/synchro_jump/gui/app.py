"""Minimal GUI entry point for Synchro Jump.

The richer plotting interface is implemented incrementally so the package can be
imported even when optional GUI dependencies are absent.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk


def launch_app() -> None:
    """Start a minimal GUI window."""

    root = tk.Tk()
    root.title("Synchro Jump")
    root.geometry("700x240")

    frame = ttk.Frame(root, padding=20)
    frame.pack(fill=tk.BOTH, expand=True)

    label = ttk.Label(
        frame,
        text="Synchro Jump GUI scaffold.\nThe OCP controls and figures are added in later features.",
        justify=tk.CENTER,
    )
    label.pack(expand=True)

    root.mainloop()
