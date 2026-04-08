"""Interactive GUI with sliders and figures for the reduced jump model."""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from synchro_jump.modeling import AthleteMorphology, PlanarJumperModelDefinition
from synchro_jump.optimization import (
    OcpSolveSummary,
    PlatformForceProfile,
    VerticalJumpBioptimOcpBuilder,
    build_ocp_runtime_summary,
    estimate_jump_apex_height,
    estimate_takeoff_velocity_from_contact_profile,
    solve_ocp_runtime_summary,
)


class SynchroJumpApp:
    """Tkinter application with sliders, figures, and jump-runtime summaries."""

    quick_solve_iterations = 5

    def __init__(self, root: tk.Tk) -> None:
        """Create the main window and its interactive controls."""

        self.root = root
        self.root.title("Synchro Jump")
        self.root.geometry("1460x840")

        self.base_settings = VerticalJumpBioptimOcpBuilder().settings
        self.force_var = tk.DoubleVar(value=1100.0)
        self.mass_var = tk.DoubleVar(value=float(self.base_settings.athlete_mass_kg))
        self.status_var = tk.StringVar()
        self.export_status = ""
        self.build_status = ""
        self.solution_status = ""
        self.runtime_solution: OcpSolveSummary | None = None

        self.figure_widget = None
        self.force_axis = None
        self.pose_axis = None
        self.kinematics_axis = None

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
        """Create the sliders and runtime action buttons."""

        ttk.Label(parent, text="Force plateforme (N)").pack(anchor=tk.W)
        self.force_scale = tk.Scale(
            parent,
            from_=self.base_settings.force_slider_values_newtons[0],
            to=self.base_settings.force_slider_values_newtons[-1],
            resolution=50,
            orient=tk.HORIZONTAL,
            variable=self.force_var,
            command=self._on_parameter_change,
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
            command=self._on_parameter_change,
            length=260,
        )
        self.mass_scale.pack(fill=tk.X, pady=(0, 12))

        ttk.Button(parent, text="Construire l'OCP", command=self.build_ocp).pack(fill=tk.X, pady=(8, 8))
        ttk.Button(
            parent,
            text=f"Resoudre l'OCP ({self.quick_solve_iterations} iter.)",
            command=self.solve_ocp,
        ).pack(fill=tk.X, pady=(0, 8))
        ttk.Button(parent, text="Exporter le modele", command=self.export_model).pack(fill=tk.X, pady=(0, 8))
        ttk.Label(parent, textvariable=self.status_var, wraplength=300, justify=tk.LEFT).pack(
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
                    "La GUI garde les sliders, le build runtime et l'export du modele."
                ),
                justify=tk.CENTER,
            ).pack(fill=tk.BOTH, expand=True)
            return

        figure = Figure(figsize=(12.0, 6.4), dpi=100)
        self.force_axis = figure.add_subplot(1, 3, 1)
        self.pose_axis = figure.add_subplot(1, 3, 2)
        self.kinematics_axis = figure.add_subplot(1, 3, 3)

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.figure_widget = canvas

    def _on_parameter_change(self, _value: str) -> None:
        """Invalidate runtime results when one slider changes."""

        self._invalidate_runtime_results()
        self.refresh()

    def _invalidate_runtime_results(self) -> None:
        """Drop runtime build/solve summaries tied to the previous sliders."""

        self.build_status = ""
        self.solution_status = ""
        self.runtime_solution = None

    def current_settings(self):
        """Return the OCP settings associated with the current sliders."""

        return self.base_settings.__class__(athlete_mass_kg=int(self.mass_var.get()))

    def current_force_newtons(self) -> float:
        """Return the current platform-force slider value."""

        return float(self.force_var.get())

    def current_morphology(self) -> AthleteMorphology:
        """Return the morphology associated with the current mass slider."""

        return AthleteMorphology(height_m=self.base_settings.athlete_height_m, mass_kg=float(self.mass_var.get()))

    def current_profile(self) -> PlatformForceProfile:
        """Return the current force profile selected by the slider."""

        return PlatformForceProfile(peak_force_newtons=self.current_force_newtons())

    def current_contact_profile(self) -> tuple[float, ...]:
        """Return the surrogate contact-force profile."""

        builder = VerticalJumpBioptimOcpBuilder(settings=self.current_settings())
        return builder.blueprint(self.current_force_newtons()).contact_force_target()

    def refresh(self) -> None:
        """Refresh the figures and the textual summary."""

        contact_profile = self.current_contact_profile()
        morphology = self.current_morphology()
        blueprint = VerticalJumpBioptimOcpBuilder(settings=self.current_settings()).blueprint(self.current_force_newtons())
        takeoff_velocity = estimate_takeoff_velocity_from_contact_profile(
            contact_force_profile_newtons=contact_profile,
            athlete_mass_kg=morphology.mass_kg,
            total_duration_s=self.base_settings.final_time_upper_bound_s,
        )
        takeoff_height = 0.56 * morphology.height_m
        apex_height = estimate_jump_apex_height(takeoff_height, takeoff_velocity)

        status_lines = [
            "Resume surrogate:\n"
            f"- vitesse de decollage estimee: {takeoff_velocity:.2f} m/s\n"
            f"- hauteur apex estimee: {apex_height:.2f} m\n"
            f"- noeuds OCP: {self.base_settings.n_shooting}\n"
            f"- temps libre <= {self.base_settings.final_time_upper_bound_s:.1f} s",
            (
                "Configuration OCP:\n"
                f"- objectif: {blueprint.objective_name}\n"
                f"- dynamique: {blueprint.dynamics_name}\n"
                "- decollage impose: force contact finale = 0 N"
            ),
        ]
        if self.build_status:
            status_lines.append(self.build_status)
        if self.solution_status:
            status_lines.append(self.solution_status)
        if self.export_status:
            status_lines.append(self.export_status)
        self.status_var.set("\n\n".join(status_lines))

        if (
            self.force_axis is None
            or self.pose_axis is None
            or self.kinematics_axis is None
            or self.figure_widget is None
        ):
            return

        self._draw_force_figure(contact_profile)
        self._draw_pose_figure(morphology)
        self._draw_kinematics_figure()
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
        if self.runtime_solution is not None and self.runtime_solution.final_time_s is not None:
            self.force_axis.axvline(
                self.runtime_solution.final_time_s,
                color="#6c9a8b",
                linestyle="--",
                linewidth=1.6,
                label="Temps final runtime",
            )
        self.force_axis.set_xlabel("Temps (s)")
        self.force_axis.set_ylabel("Force (N)")
        self.force_axis.set_title("Profils de force")
        self.force_axis.grid(alpha=0.25)
        self.force_axis.legend(loc="upper right")

    def _draw_pose_figure(self, morphology: AthleteMorphology) -> None:
        """Draw the initial and optimized sagittal stick figures."""

        initial_points = self._pose_points_from_q(
            morphology,
            PlanarJumperModelDefinition(morphology=morphology).initial_joint_configuration_rad,
        )

        self.pose_axis.clear()
        self._draw_stick_figure(
            self.pose_axis,
            initial_points,
            color="#6b6b6b",
            label="Posture initiale",
            linestyle="--",
            alpha=0.85,
        )

        if self.runtime_solution is not None:
            state_trajectories = self.runtime_solution.state_trajectories
            q_roots = state_trajectories.get("q_roots")
            q_joints = state_trajectories.get("q_joints")
            if q_roots is not None and q_joints is not None:
                final_q = tuple([*q_roots[:, -1], *q_joints[:, -1]])
                final_points = self._pose_points_from_q(morphology, final_q)
                self._draw_stick_figure(
                    self.pose_axis,
                    final_points,
                    color="#c44536",
                    label="Posture fin OCP",
                    linestyle="-",
                    alpha=1.0,
                )

        self.pose_axis.plot([-0.35, 0.35], [0.0, 0.0], color="#8c5e34", linewidth=4.0)
        self.pose_axis.set_aspect("equal", adjustable="box")
        self.pose_axis.set_xlim(-0.7, 0.8)
        self.pose_axis.set_ylim(-0.05, morphology.height_m + 0.15)
        self.pose_axis.set_xlabel("A/P (m)")
        self.pose_axis.set_ylabel("Vertical (m)")
        self.pose_axis.set_title("Modele 3 segments")
        self.pose_axis.grid(alpha=0.2)
        self.pose_axis.legend(loc="upper right")

    def _draw_stick_figure(
        self,
        axis,
        points: dict[str, tuple[float, float]],
        *,
        color: str,
        label: str,
        linestyle: str,
        alpha: float,
    ) -> None:
        """Draw one sagittal stick figure on one Matplotlib axis."""

        axis.plot(
            [points["foot"][0], points["knee"][0], points["hip"][0], points["head"][0]],
            [points["foot"][1], points["knee"][1], points["hip"][1], points["head"][1]],
            color=color,
            linewidth=3.0,
            marker="o",
            linestyle=linestyle,
            alpha=alpha,
            label=label,
        )

    def _draw_kinematics_figure(self) -> None:
        """Draw runtime kinematics when one solved trajectory is available."""

        self.kinematics_axis.clear()
        if self.runtime_solution is None or self.runtime_solution.time.size == 0:
            self.kinematics_axis.text(
                0.5,
                0.5,
                "Aucune solution runtime.\nUtilise 'Resoudre l'OCP'.",
                ha="center",
                va="center",
                transform=self.kinematics_axis.transAxes,
            )
            self.kinematics_axis.set_title("Resultat runtime")
            self.kinematics_axis.set_xticks([])
            self.kinematics_axis.set_yticks([])
            return

        time = self.runtime_solution.time
        self.kinematics_axis.plot(
            time,
            self.runtime_solution.com_height_trajectory_m,
            color="#3a7d44",
            linewidth=2.2,
            label="Hauteur CoM",
        )

        platform_position = self.runtime_solution.state_trajectories.get("platform_position")
        if platform_position is not None:
            self.kinematics_axis.plot(
                time,
                platform_position.reshape((-1,)),
                color="#2b59c3",
                linewidth=2.0,
                label="Plateforme",
            )

        self.kinematics_axis.set_xlabel("Temps (s)")
        self.kinematics_axis.set_ylabel("Hauteur (m)")
        self.kinematics_axis.set_title("Cinematique runtime")
        self.kinematics_axis.grid(alpha=0.25)
        self.kinematics_axis.legend(loc="best")

    def _pose_points_from_q(
        self,
        morphology: AthleteMorphology,
        q_values: tuple[float, float, float, float, float],
    ) -> dict[str, tuple[float, float]]:
        """Compute the displayed joint positions for one generalized state."""

        lengths = morphology.segment_lengths
        q_root_x, q_root_z, q_root_rot, q_knee, q_hip = q_values

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

    def build_ocp(self) -> None:
        """Attempt to instantiate the `bioptim` OCP for the current sliders."""

        self.runtime_solution = None
        summary = build_ocp_runtime_summary(
            settings=self.current_settings(),
            peak_force_newtons=self.current_force_newtons(),
        )
        if summary.success:
            self.build_status = (
                "Construction runtime:\n"
                f"- succes: oui\n"
                f"- phases: {summary.n_phases}\n"
                f"- etats: {', '.join(summary.state_names)}\n"
                f"- controles: {', '.join(summary.control_names)}"
            )
        else:
            self.build_status = f"Construction runtime:\n- succes: non\n- message: {summary.message}"
        self.solution_status = ""
        self.refresh()

    def solve_ocp(self) -> None:
        """Solve one quick runtime OCP and expose the trajectories in the GUI."""

        self.solution_status = (
            "Resolution runtime:\n"
            f"- en cours...\n- iterations max: {self.quick_solve_iterations}"
        )
        self.refresh()
        self.root.update_idletasks()

        summary = solve_ocp_runtime_summary(
            settings=self.current_settings(),
            peak_force_newtons=self.current_force_newtons(),
            maximum_iterations=self.quick_solve_iterations,
        )
        if summary.success:
            self.runtime_solution = summary
            self.solution_status = (
                "Resolution runtime:\n"
                f"- succes: oui\n"
                f"- statut solveur: {summary.solver_status}\n"
                f"- temps final: {summary.final_time_s:.2f} s\n"
                f"- apex predit: {summary.predicted_apex_height_m:.2f} m\n"
                f"- temps solveur: {summary.solve_time_s:.2f} s"
            )
        else:
            self.runtime_solution = None
            self.solution_status = f"Resolution runtime:\n- succes: non\n- message: {summary.message}"
        self.refresh()

    def export_model(self) -> None:
        """Export the current `.bioMod` file and append the path to the status text."""

        builder = VerticalJumpBioptimOcpBuilder(settings=self.current_settings())
        model_path = builder.export_model(Path("generated"))
        self.export_status = f"Modele exporte: {model_path}"
        self.refresh()


def launch_app() -> None:
    """Start the interactive GUI."""

    root = tk.Tk()
    SynchroJumpApp(root)
    root.mainloop()
