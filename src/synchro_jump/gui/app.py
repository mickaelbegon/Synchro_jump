"""Interactive GUI with sliders and figures for the reduced jump model."""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from synchro_jump.modeling import AthleteMorphology, PlanarJumperModelDefinition
from synchro_jump.optimization import (
    PlatformForceProfile,
    VerticalJumpBioptimOcpBuilder,
    estimate_jump_apex_height,
    estimate_takeoff_velocity_from_contact_profile,
)


class SynchroJumpApp:
    """Tkinter application with sliders, figures, and a jump summary."""

    def __init__(self, root: tk.Tk) -> None:
        """Create the main window and its interactive controls."""

        self.root = root
        self.root.title("Synchro Jump")
        self.root.geometry("1320x820")

        self.base_settings = VerticalJumpBioptimOcpBuilder().settings
        self.force_var = tk.DoubleVar(value=1100.0)
        self.mass_var = tk.DoubleVar(value=float(self.base_settings.athlete_mass_kg))
        self.status_var = tk.StringVar()

        self.figure_widget = None
        self.force_axis = None
        self.pose_axis = None

        main_frame = ttk.Frame(root, padding=16)
        main_frame.pack(fill=tk.BOTH, expand=True)

        controls_frame = ttk.LabelFrame(main_frame, text="Parametres", padding=12)
        controls_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        figures_frame = ttk.Frame(main_frame)
        figures_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls(controls_frame)
        self._build_figures(figures_frame)
        self.refresh()

    def _build_controls(self, parent: ttk.Frame) -> None:
        """Create the sliders and action button."""

        ttk.Label(parent, text="Force plateforme (N)").pack(anchor=tk.W)
        self.force_scale = tk.Scale(
            parent,
            from_=self.base_settings.force_slider_values_newtons[0],
            to=self.base_settings.force_slider_values_newtons[-1],
            resolution=50,
            orient=tk.HORIZONTAL,
            variable=self.force_var,
            command=lambda _value: self.refresh(),
            length=260,
        )
        self.force_scale.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(parent, text="Masse sauteur (kg)").pack(anchor=tk.W)
        self.mass_scale = tk.Scale(
            parent,
            from_=self.base_settings.mass_slider_values_kg[0],
            to=self.base_settings.mass_slider_values_kg[-1],
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=self.mass_var,
            command=lambda _value: self.refresh(),
            length=260,
        )
        self.mass_scale.pack(fill=tk.X, pady=(0, 12))

        ttk.Button(parent, text="Exporter le modele", command=self.export_model).pack(fill=tk.X, pady=(8, 8))
        ttk.Label(parent, textvariable=self.status_var, wraplength=260, justify=tk.LEFT).pack(
            fill=tk.X, pady=(12, 0)
        )

    def _build_figures(self, parent: ttk.Frame) -> None:
        """Create the Matplotlib figure area when available."""

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
        except ModuleNotFoundError:
            ttk.Label(
                parent,
                text=(
                    "Matplotlib n'est pas installe dans cet environnement.\n"
                    "La GUI garde les sliders et l'export du modele."
                ),
                justify=tk.CENTER,
            ).pack(fill=tk.BOTH, expand=True)
            return

        figure = Figure(figsize=(9.2, 6.4), dpi=100)
        self.force_axis = figure.add_subplot(1, 2, 1)
        self.pose_axis = figure.add_subplot(1, 2, 2)

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.figure_widget = canvas

    def current_settings(self):
        """Return the OCP settings associated with the current sliders."""

        return self.base_settings.__class__(athlete_mass_kg=int(self.mass_var.get()))

    def current_morphology(self) -> AthleteMorphology:
        """Return the morphology associated with the current mass slider."""

        return AthleteMorphology(height_m=self.base_settings.athlete_height_m, mass_kg=float(self.mass_var.get()))

    def current_profile(self) -> PlatformForceProfile:
        """Return the current force profile selected by the slider."""

        return PlatformForceProfile(peak_force_newtons=float(self.force_var.get()))

    def current_contact_profile(self) -> tuple[float, ...]:
        """Return the surrogate contact-force profile."""

        builder = VerticalJumpBioptimOcpBuilder(settings=self.current_settings())
        return builder.blueprint(float(self.force_var.get())).contact_force_target()

    def refresh(self) -> None:
        """Refresh the figures and the textual summary."""

        contact_profile = self.current_contact_profile()
        morphology = self.current_morphology()
        takeoff_velocity = estimate_takeoff_velocity_from_contact_profile(
            contact_force_profile_newtons=contact_profile,
            athlete_mass_kg=morphology.mass_kg,
            total_duration_s=self.base_settings.final_time_upper_bound_s,
        )
        takeoff_height = 0.56 * morphology.height_m
        apex_height = estimate_jump_apex_height(takeoff_height, takeoff_velocity)

        self.status_var.set(
            "Resume surrogate:\n"
            f"- vitesse de decollage estimee: {takeoff_velocity:.2f} m/s\n"
            f"- hauteur apex estimee: {apex_height:.2f} m\n"
            f"- noeuds OCP: {self.base_settings.n_shooting}\n"
            f"- temps libre <= {self.base_settings.final_time_upper_bound_s:.1f} s"
        )

        if self.force_axis is None or self.pose_axis is None or self.figure_widget is None:
            return

        self._draw_force_figure(contact_profile)
        self._draw_pose_figure(morphology)
        self.figure_widget.draw_idle()

    def _draw_force_figure(self, contact_profile: tuple[float, ...]) -> None:
        """Draw the platform and contact-force profiles."""

        profile = self.current_profile()
        times = [
            profile.total_duration * index / max(self.base_settings.n_shooting - 1, 1)
            for index in range(self.base_settings.n_shooting)
        ]
        actuation = [profile.force_at(time) for time in times]

        self.force_axis.clear()
        self.force_axis.plot(times, actuation, color="#d1495b", linewidth=2.2, label="Force plateforme")
        self.force_axis.plot(times, contact_profile, color="#2b59c3", linewidth=2.2, label="Contact surrogate")
        self.force_axis.set_xlabel("Temps (s)")
        self.force_axis.set_ylabel("Force (N)")
        self.force_axis.set_title("Profils de force")
        self.force_axis.grid(alpha=0.25)
        self.force_axis.legend(loc="upper right")

    def _draw_pose_figure(self, morphology: AthleteMorphology) -> None:
        """Draw a sagittal stick figure for the initial posture."""

        points = self._pose_points(morphology)

        self.pose_axis.clear()
        self.pose_axis.plot(
            [points["foot"][0], points["knee"][0], points["hip"][0], points["head"][0]],
            [points["foot"][1], points["knee"][1], points["hip"][1], points["head"][1]],
            color="#333333",
            linewidth=3.0,
            marker="o",
        )
        self.pose_axis.plot([-0.35, 0.35], [0.0, 0.0], color="#8c5e34", linewidth=4.0)
        self.pose_axis.set_aspect("equal", adjustable="box")
        self.pose_axis.set_xlim(-0.7, 0.8)
        self.pose_axis.set_ylim(-0.05, morphology.height_m + 0.1)
        self.pose_axis.set_xlabel("A/P (m)")
        self.pose_axis.set_ylabel("Vertical (m)")
        self.pose_axis.set_title("Modele 3 segments")
        self.pose_axis.grid(alpha=0.2)

    def _pose_points(self, morphology: AthleteMorphology) -> dict[str, tuple[float, float]]:
        """Compute the displayed joint positions of the reduced model."""

        lengths = morphology.segment_lengths
        q_root_x, q_root_z, q_root_rot, q_knee, q_hip = PlanarJumperModelDefinition(
            morphology=morphology
        ).initial_joint_configuration_rad

        def advance(origin: tuple[float, float], angle: float, length: float) -> tuple[float, float]:
            return (origin[0] + length * math.cos(angle), origin[1] + length * math.sin(angle))

        leg_angle = math.pi / 2.0 + q_root_rot
        thigh_angle = leg_angle + q_knee
        trunk_angle = thigh_angle + q_hip

        foot = (q_root_x, q_root_z)
        knee = advance(foot, leg_angle, lengths.leg_foot)
        hip = advance(knee, thigh_angle, lengths.thigh)
        head = advance(hip, trunk_angle, lengths.trunk)
        return {"foot": foot, "knee": knee, "hip": hip, "head": head}

    def export_model(self) -> None:
        """Export the current `.bioMod` file and append the path to the status text."""

        builder = VerticalJumpBioptimOcpBuilder(settings=self.current_settings())
        model_path = builder.export_model(Path("generated"))
        self.status_var.set(f"{self.status_var.get()}\n\nModele exporte: {model_path}")


def launch_app() -> None:
    """Start the interactive GUI."""

    root = tk.Tk()
    SynchroJumpApp(root)
    root.mainloop()
