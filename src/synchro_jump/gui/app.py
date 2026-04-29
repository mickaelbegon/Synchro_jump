"""Interactive GUI with sliders and figures for the reduced jump model."""

from __future__ import annotations

import math
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import numpy as np

from synchro_jump.gui.raster_avatar import avatar_rendering_diagnostics, draw_segment_image
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
from synchro_jump.optimization.problem import (
    CONTACT_MODEL_COMPLIANT_UNILATERAL,
    CONTACT_MODEL_NO_PLATFORM,
    CONTACT_MODEL_RIGID_UNILATERAL,
    snap_to_discrete_value,
)


class SynchroJumpApp:
    """Tkinter application with sliders, figures, and jump-runtime summaries."""

    default_solve_iterations = 1000
    max_solve_iterations = 1000
    animation_delay_ms = 80
    avatar_flip_horizontal = True
    contact_model_labels = {
        CONTACT_MODEL_RIGID_UNILATERAL: "Rigide unilateral",
        CONTACT_MODEL_COMPLIANT_UNILATERAL: "Compliant unilateral",
        CONTACT_MODEL_NO_PLATFORM: "Sans plateforme",
    }
    contact_model_by_label = {label: key for key, label in contact_model_labels.items()}

    def __init__(self, root: tk.Tk) -> None:
        """Create the main window and its interactive controls."""

        self.root = root
        self.root.title("Synchro Jump")
        self.root.geometry("1460x840")

        self.base_settings = VerticalJumpBioptimOcpBuilder().settings
        self.force_var = tk.DoubleVar(value=1100.0)
        self.mass_var = tk.DoubleVar(value=float(self.base_settings.athlete_mass_kg))
        self.contact_model_var = tk.StringVar(
            value=self.contact_model_labels[self.base_settings.contact_model]
        )
        self.use_cache_var = tk.BooleanVar(value=True)
        self.solve_iterations_var = tk.IntVar(value=self.default_solve_iterations)
        self.animation_frame_var = tk.IntVar(value=0)
        self.status_var = tk.StringVar()
        self.busy_var = tk.StringVar(value="")
        self.export_status = ""
        self.build_status = ""
        self.solution_status = ""
        self.ocp_is_built = False
        self.runtime_solution: OcpSolveSummary | None = None
        self.animation_playing = False
        self.animation_job: str | None = None
        self._cached_contact_profile_key = None
        self._cached_contact_profile: tuple[float, ...] | None = None
        self._cached_blueprint_key = None
        self._cached_blueprint = None
        self._cached_initial_q_key = None
        self._cached_initial_q: tuple[float, ...] | None = None
        self._cached_display_model_key = None
        self._cached_display_model = None
        self._cached_runtime_com_key = None
        self._cached_runtime_com_trajectory: np.ndarray | None = None
        self._cached_runtime_com_velocity_key = None
        self._cached_runtime_com_velocity_trajectory: np.ndarray | None = None

        self.build_button = None
        self.solve_button = None
        self.busy_label = None
        self.figure_widget = None
        self.summary_axis = None
        self.force_axis = None
        self.pose_axis = None
        self.kinematics_axis = None
        self.kinematics_velocity_axis = None
        self.joint_angle_axis = None
        self.joint_torque_axis = None
        self.play_pause_button = None
        self.play_pause_text = tk.StringVar(value="▶")

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
            resolution=1,
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
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.mass_var,
            command=self._on_parameter_change,
            length=260,
        )
        self.mass_scale.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(parent, text="Modele de contact").pack(anchor=tk.W)
        self.contact_model_combo = ttk.Combobox(
            parent,
            state="readonly",
            textvariable=self.contact_model_var,
            values=tuple(self.contact_model_labels.values()),
        )
        self.contact_model_combo.bind("<<ComboboxSelected>>", self._on_contact_model_change)
        self.contact_model_combo.pack(fill=tk.X, pady=(0, 12))

        ttk.Label(parent, text="Iterations solveur").pack(anchor=tk.W)
        self.solve_iterations_scale = tk.Scale(
            parent,
            from_=0,
            to=self.max_solve_iterations,
            resolution=5,
            orient=tk.HORIZONTAL,
            variable=self.solve_iterations_var,
            length=260,
        )
        self.solve_iterations_scale.pack(fill=tk.X, pady=(0, 12))
        ttk.Checkbutton(parent, text="Utiliser le cache optimal", variable=self.use_cache_var).pack(
            anchor=tk.W, pady=(0, 12)
        )

        self.build_button = ttk.Button(parent, text="Construire l'OCP", command=self.build_ocp)
        self.build_button.pack(fill=tk.X, pady=(8, 8))
        self.solve_button = ttk.Button(parent, text="Resoudre l'OCP", command=self.solve_ocp)
        self.solve_button.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(parent, text="Exporter le modele", command=self.export_model).pack(fill=tk.X, pady=(0, 8))

        self.busy_label = ttk.Label(parent, textvariable=self.busy_var, wraplength=300, justify=tk.LEFT)

        ttk.Label(parent, text="Animation trajectoire").pack(anchor=tk.W, pady=(12, 0))
        animation_frame = ttk.Frame(parent)
        animation_frame.pack(fill=tk.X, pady=(0, 8))
        self.play_pause_button = ttk.Button(
            animation_frame,
            textvariable=self.play_pause_text,
            command=self.toggle_animation,
            width=3,
        )
        self.play_pause_button.pack(side=tk.LEFT, padx=(0, 8))
        self.animation_scale = tk.Scale(
            animation_frame,
            from_=0,
            to=0,
            resolution=1,
            orient=tk.HORIZONTAL,
            variable=self.animation_frame_var,
            command=self._on_animation_frame_change,
            length=220,
        )
        self.animation_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(parent, textvariable=self.status_var, wraplength=300, justify=tk.LEFT).pack(
            fill=tk.X, pady=(12, 0)
        )
        self._update_ocp_button_states()

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

        figure = Figure(figsize=(13.6, 7.2), dpi=100)
        gridspec = figure.add_gridspec(3, 3, width_ratios=(0.95, 0.95, 1.15), height_ratios=(0.52, 1.0, 1.0))
        figure.subplots_adjust(left=0.06, right=0.98, top=0.95, bottom=0.08, wspace=0.34, hspace=0.42)
        self.summary_axis = figure.add_subplot(gridspec[0, 0:2])
        self.force_axis = figure.add_subplot(gridspec[1:, 0])
        self.pose_axis = figure.add_subplot(gridspec[1:, 1])
        self.kinematics_axis = figure.add_subplot(gridspec[0:2, 2])
        self.kinematics_velocity_axis = self.kinematics_axis.twinx()
        self.joint_angle_axis = figure.add_subplot(gridspec[2, 2])
        self.joint_torque_axis = self.joint_angle_axis.twinx()

        canvas = FigureCanvasTkAgg(figure, master=parent)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.figure_widget = canvas

    def _on_parameter_change(self, _value: str) -> None:
        """Invalidate runtime results when one slider changes."""

        self.force_var.set(self.current_force_newtons())
        self.mass_var.set(self.current_mass_kg())
        self._clear_display_cache()
        self._invalidate_runtime_results()
        self.refresh()

    def _on_animation_frame_change(self, _value: str) -> None:
        """Redraw the figures when the animation cursor changes."""

        self.refresh(dynamic_only=True)

    def _on_contact_model_change(self, _event=None) -> None:
        """Invalidate runtime results when the contact model changes."""

        self._clear_display_cache()
        self._invalidate_runtime_results()
        self.refresh()

    def _invalidate_runtime_results(self) -> None:
        """Drop runtime build/solve summaries tied to the previous sliders."""

        self._stop_animation()
        self._clear_runtime_cache()
        self.build_status = ""
        self.solution_status = ""
        self._set_ocp_built_state(False)
        self.runtime_solution = None
        self.animation_frame_var.set(0)
        if hasattr(self, "animation_scale"):
            self.animation_scale.configure(to=0)
        self._sync_animation_button_label()

    def _clear_display_cache(self) -> None:
        """Invalidate cached slider-dependent display data."""

        self._cached_contact_profile_key = None
        self._cached_contact_profile = None
        self._cached_blueprint_key = None
        self._cached_blueprint = None
        self._cached_initial_q_key = None
        self._cached_initial_q = None
        self._cached_display_model_key = None
        self._cached_display_model = None

    def _clear_runtime_cache(self) -> None:
        """Invalidate cached runtime trajectories used only for plotting."""

        self._cached_runtime_com_key = None
        self._cached_runtime_com_trajectory = None
        self._cached_runtime_com_velocity_key = None
        self._cached_runtime_com_velocity_trajectory = None

    def _set_ocp_built_state(self, is_built: bool) -> None:
        """Store the OCP build state and refresh the action buttons."""

        self.ocp_is_built = is_built
        self._update_ocp_button_states()

    def _update_ocp_button_states(self) -> None:
        """Enable or disable build/solve buttons from the current OCP state."""

        if self.build_button is not None:
            self.build_button.configure(state=tk.DISABLED if self.ocp_is_built else tk.NORMAL)
        if self.solve_button is not None:
            self.solve_button.configure(state=tk.NORMAL if self.ocp_is_built else tk.DISABLED)

    def _show_busy_indicator(self, message: str) -> None:
        """Display one compact busy icon and message while runtime work is ongoing."""

        self.busy_var.set(f"⌛ {message}")
        if self.busy_label is not None and not self.busy_label.winfo_ismapped():
            self.busy_label.pack(fill=tk.X, pady=(8, 8), before=self.animation_scale)
        self.root.update_idletasks()

    def _hide_busy_indicator(self) -> None:
        """Hide the busy indicator after one runtime action completes."""

        self.busy_var.set("")
        if self.busy_label is not None and self.busy_label.winfo_ismapped():
            self.busy_label.pack_forget()

    def _sync_animation_button_label(self) -> None:
        """Refresh the compact play/pause button label."""

        self.play_pause_text.set("❚❚" if self.animation_playing else "▶")

    def _log_runtime_request(self, action: str) -> None:
        """Print one compact runtime banner in the terminal."""

        print(
            "[SynchroJump]"
            f" {action}"
            f" | force={self.current_force_newtons():.0f} N"
            f" | masse={self.current_mass_kg():.0f} kg"
            f" | contact={self.current_contact_model_key()}"
        )

    def current_settings(self):
        """Return the OCP settings associated with the current sliders."""

        return self.base_settings.__class__(
            athlete_mass_kg=self.current_mass_kg(),
            contact_model=self.current_contact_model_key(),
        )

    def current_force_newtons(self) -> float:
        """Return the current platform-force slider value."""

        return snap_to_discrete_value(float(self.force_var.get()), self.base_settings.force_slider_values_newtons)

    def current_mass_kg(self) -> float:
        """Return the current athlete-mass slider value."""

        return snap_to_discrete_value(float(self.mass_var.get()), self.base_settings.mass_slider_values_kg)

    def current_solver_iterations(self) -> int:
        """Return the requested number of IPOPT iterations."""

        return int(self.solve_iterations_var.get())

    def current_contact_model_key(self) -> str:
        """Return the selected contact-model key."""

        return self.contact_model_by_label[self.contact_model_var.get()]

    def current_animation_frame(self) -> int:
        """Return the currently displayed animation frame index."""

        if self.runtime_solution is None or self.runtime_solution.time.size == 0:
            return 0
        return max(0, min(int(self.animation_frame_var.get()), self.runtime_solution.time.size - 1))

    def current_morphology(self) -> AthleteMorphology:
        """Return the morphology associated with the current mass slider."""

        return AthleteMorphology(height_m=self.base_settings.athlete_height_m, mass_kg=self.current_mass_kg())

    def current_model_definition(self) -> PlanarJumperModelDefinition:
        """Return the displayed jumper model matching the selected mode."""

        return PlanarJumperModelDefinition(
            morphology=self.current_morphology(),
            floating_base=self.current_contact_model_key() != CONTACT_MODEL_NO_PLATFORM,
            include_platform_contact=self.current_contact_model_key() != CONTACT_MODEL_NO_PLATFORM,
        )

    def current_profile(self) -> PlatformForceProfile:
        """Return the current force profile selected by the slider."""

        return PlatformForceProfile(peak_force_newtons=self.current_force_newtons())

    def _current_blueprint(self):
        """Return the cached OCP blueprint for the current sliders."""

        cache_key = (self.current_contact_model_key(), self.current_mass_kg(), self.current_force_newtons())
        if self._cached_blueprint_key != cache_key or self._cached_blueprint is None:
            self._cached_blueprint = VerticalJumpBioptimOcpBuilder(settings=self.current_settings()).blueprint(
                self.current_force_newtons()
            )
            self._cached_blueprint_key = cache_key
        return self._cached_blueprint

    def current_contact_profile(self) -> tuple[float, ...]:
        """Return the surrogate contact-force profile."""

        cache_key = (self.current_contact_model_key(), self.current_mass_kg(), self.current_force_newtons())
        cached_key = getattr(self, "_cached_contact_profile_key", None)
        cached_profile = getattr(self, "_cached_contact_profile", None)
        if cached_key == cache_key and cached_profile is not None:
            return cached_profile
        if self.current_contact_model_key() == CONTACT_MODEL_NO_PLATFORM:
            self._cached_contact_profile = tuple(0.0 for _ in range(self.base_settings.n_shooting))
        else:
            self._cached_contact_profile = self._current_blueprint().contact_force_target()
        self._cached_contact_profile_key = cache_key
        return self._cached_contact_profile

    def current_initial_q(self) -> tuple[float, ...]:
        """Return the cached initial posture aligned on the true exported model when available."""

        cache_key = (self.current_contact_model_key(), self.current_mass_kg())
        cached_key = getattr(self, "_cached_initial_q_key", None)
        cached_q = getattr(self, "_cached_initial_q", None)
        if cached_key == cache_key and cached_q is not None:
            return cached_q

        builder = VerticalJumpBioptimOcpBuilder(settings=self.current_settings())
        try:
            model_path = builder.export_model(Path("generated"))
            self._cached_initial_q = builder.aligned_initial_joint_configuration_rad(model_path=model_path)
        except Exception:
            self._cached_initial_q = self.current_model_definition().initial_joint_configuration_rad
        self._cached_initial_q_key = cache_key
        return self._cached_initial_q

    def current_display_biorbd_model(self):
        """Return one cached `BiorbdModel` matching the currently displayed sliders."""

        cache_key = (self.current_contact_model_key(), self.current_mass_kg())
        cached_key = getattr(self, "_cached_display_model_key", None)
        cached_model = getattr(self, "_cached_display_model", None)
        if cached_key == cache_key and cached_model is not None:
            return cached_model

        try:
            from bioptim import BiorbdModel

            builder = VerticalJumpBioptimOcpBuilder(settings=self.current_settings())
            model_path = builder.export_model(Path("generated"))
            cached_model = BiorbdModel(str(model_path))
        except Exception:
            cached_model = None

        self._cached_display_model_key = cache_key
        self._cached_display_model = cached_model
        return self._cached_display_model

    def _display_pose_and_com_from_q(
        self,
        morphology: AthleteMorphology,
        q_values: tuple[float, ...] | list[float],
    ) -> tuple[dict[str, tuple[float, float]], tuple[float, float], dict[str, tuple[float, float]]]:
        """Return pose markers, global CoM, and segmental CoMs for one displayed state."""

        biorbd_model = self.current_display_biorbd_model()
        if biorbd_model is not None:
            try:
                from casadi import DM

                q_array = np.asarray(q_values, dtype=float).reshape((-1, 1))
                q_dm = DM(q_array)
                parameters = DM()
                marker_points = {}
                for displayed_name, marker_name in (
                    ("foot", "foot_contact"),
                    ("knee", "knee"),
                    ("hip", "hip"),
                    ("head", "head"),
                ):
                    marker_index = biorbd_model.marker_index(marker_name)
                    marker = np.asarray(biorbd_model.marker(marker_index)(q_dm, parameters), dtype=float).reshape((-1,))
                    marker_points[displayed_name] = (float(marker[0]), float(marker[2]))

                com = np.asarray(biorbd_model.center_of_mass()(q_dm, parameters), dtype=float).reshape((-1,))
                global_com = (float(com[0]), float(com[2]))
                segment_coms = {
                    "leg_foot": (
                        marker_points["foot"][0] + 0.55 * (marker_points["knee"][0] - marker_points["foot"][0]),
                        marker_points["foot"][1] + 0.55 * (marker_points["knee"][1] - marker_points["foot"][1]),
                    ),
                    "thigh": (
                        marker_points["knee"][0] + 0.45 * (marker_points["hip"][0] - marker_points["knee"][0]),
                        marker_points["knee"][1] + 0.45 * (marker_points["hip"][1] - marker_points["knee"][1]),
                    ),
                    "trunk": (
                        marker_points["hip"][0] + 0.5 * (marker_points["head"][0] - marker_points["hip"][0]),
                        marker_points["hip"][1] + 0.5 * (marker_points["head"][1] - marker_points["hip"][1]),
                    ),
                }
                return marker_points, global_com, segment_coms
            except Exception:
                pass

        points = self._analytic_pose_points_from_q(morphology, tuple(q_values))
        model_definition = self.current_model_definition()
        global_com = model_definition.center_of_mass_position(tuple(q_values))
        segment_coms = model_definition.segment_center_of_mass_positions(tuple(q_values))
        return points, global_com, segment_coms

    def _runtime_q_trajectory(self) -> np.ndarray | None:
        """Return the full runtime generalized-coordinate trajectory when available."""

        if self.runtime_solution is None:
            return None

        q_roots = self.runtime_solution.state_trajectories.get("q_roots")
        q_joints = self.runtime_solution.state_trajectories.get("q_joints")
        if q_roots is not None and q_joints is not None:
            return np.vstack((q_roots, q_joints))
        if q_joints is not None:
            return q_joints
        if q_roots is not None:
            return q_roots
        return self.runtime_solution.state_trajectories.get("q")

    def _runtime_joint_angle_trajectory_deg(self) -> dict[str, np.ndarray] | None:
        """Return the knee and hip angle trajectories in degrees."""

        q_trajectory = self._runtime_q_trajectory()
        if q_trajectory is None or q_trajectory.shape[0] < 2:
            return None
        return {
            "Genou": np.degrees(q_trajectory[-2, :]),
            "Hanche": np.degrees(q_trajectory[-1, :]),
        }

    def _runtime_joint_torque_trajectory_nm(self) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
        """Return the knee and hip torque trajectories in Nm."""

        if self.runtime_solution is None:
            return None

        tau_trajectory = self.runtime_solution.control_trajectories.get("tau_joints")
        if tau_trajectory is None or tau_trajectory.shape[0] < 2:
            return None

        control_time = self.runtime_solution.time
        if tau_trajectory.shape[1] == max(control_time.size - 1, 0):
            control_time = control_time[:-1]
        elif tau_trajectory.shape[1] != control_time.size and tau_trajectory.shape[1] > 0:
            control_time = np.linspace(control_time[0], control_time[-1], tau_trajectory.shape[1])

        return control_time, {
            "Genou": tau_trajectory[-2, :],
            "Hanche": tau_trajectory[-1, :],
        }

    def _joint_torque_limits_nm(self) -> dict[str, float]:
        """Return the displayed knee and hip torque limits in Nm."""

        athlete_mass_kg = float(self.current_mass_kg())
        return {
            "Genou": 15.0 * athlete_mass_kg,
            "Hanche": 20.0 * athlete_mass_kg,
        }

    def _runtime_com_planar_trajectory(self, morphology: AthleteMorphology) -> np.ndarray | None:
        """Return the runtime CoM planar trajectory `(x, z)` from the displayed model."""

        q_trajectory = self._runtime_q_trajectory()
        if q_trajectory is None or q_trajectory.shape[1] == 0:
            return None

        cache_key = (
            id(self.runtime_solution),
            morphology.height_m,
            morphology.mass_kg,
            self.current_contact_model_key(),
        )
        cached_key = getattr(self, "_cached_runtime_com_key", None)
        cached_trajectory = getattr(self, "_cached_runtime_com_trajectory", None)
        if cached_key == cache_key and cached_trajectory is not None:
            return cached_trajectory

        com_trajectory = np.zeros((2, q_trajectory.shape[1]), dtype=float)
        for frame_index in range(q_trajectory.shape[1]):
            _, (com_x, com_z), _ = self._display_pose_and_com_from_q(
                morphology,
                tuple(q_trajectory[:, frame_index].tolist()),
            )
            com_trajectory[:, frame_index] = (com_x, com_z)
        self._cached_runtime_com_key = cache_key
        self._cached_runtime_com_trajectory = com_trajectory
        return self._cached_runtime_com_trajectory

    def _runtime_com_planar_velocity_trajectory(
        self,
        morphology: AthleteMorphology,
    ) -> np.ndarray | None:
        """Return the finite-difference planar CoM velocity `(vx, vz)` along the runtime solution."""

        if self.runtime_solution is None or self.runtime_solution.time.size < 2:
            return None

        cache_key = (
            id(self.runtime_solution),
            morphology.height_m,
            morphology.mass_kg,
            self.current_contact_model_key(),
        )
        cached_velocity_key = getattr(self, "_cached_runtime_com_velocity_key", None)
        cached_velocity = getattr(self, "_cached_runtime_com_velocity_trajectory", None)
        if cached_velocity_key == cache_key and cached_velocity is not None:
            return cached_velocity

        com_trajectory = self._runtime_com_planar_trajectory(morphology)
        if com_trajectory is None:
            return None

        time = np.asarray(self.runtime_solution.time, dtype=float)
        if np.any(np.diff(time) <= 0.0):
            return None

        velocities = np.zeros_like(com_trajectory)
        velocities[0, :] = np.gradient(com_trajectory[0, :], time)
        velocities[1, :] = np.gradient(com_trajectory[1, :], time)
        self._cached_runtime_com_velocity_key = cache_key
        self._cached_runtime_com_velocity_trajectory = velocities
        return self._cached_runtime_com_velocity_trajectory

    def _segment_com_positions(
        self,
        q_values: tuple[float, ...] | list[float],
    ) -> dict[str, tuple[float, float]]:
        """Return the displayed segmental CoM positions for one generalized state."""

        _, _, segment_coms = self._display_pose_and_com_from_q(self.current_morphology(), tuple(q_values))
        return segment_coms

    def refresh(self, *, dynamic_only: bool = False) -> None:
        """Refresh the figures and the textual summary."""

        morphology = self.current_morphology()
        if dynamic_only:
            if (
                self.pose_axis is None
                or self.kinematics_axis is None
                or self.kinematics_velocity_axis is None
                or self.joint_angle_axis is None
                or self.joint_torque_axis is None
                or self.figure_widget is None
            ):
                return
            self._draw_pose_figure(morphology)
            self._draw_kinematics_figure()
            self._draw_joint_figure()
            self.figure_widget.draw_idle()
            return

        contact_profile = self.current_contact_profile()
        blueprint = self._current_blueprint()
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
                f"- modele contact: {self.contact_model_var.get()}\n"
                f"- label contact: {blueprint.contact_model_name}\n"
                + (
                    (
                        f"- contact physique: k={self.base_settings.contact_stiffness_n_per_m:.0f} N/m, "
                        f"c={self.base_settings.contact_damping_n_s_per_m:.0f} N.s/m\n"
                        "- decollage impose: force contact finale = 0 N"
                    )
                    if self.current_contact_model_key() != CONTACT_MODEL_NO_PLATFORM
                    else "- mode simplifie: sans plateforme ni force de contact explicite\n"
                    "- force exterieure equivalente affichee: masse x (acceleration verticale du CoM + g)"
                )
            ),
            f"Avatar:\n- {self._avatar_status_line()}",
        ]
        if self.runtime_solution is not None and self.runtime_solution.time.size:
            frame_index = self.current_animation_frame()
            status_lines.append(
                "Animation:\n"
                f"- frame: {frame_index + 1}/{self.runtime_solution.time.size}\n"
                f"- temps courant: {self.runtime_solution.time[frame_index]:.2f} s\n"
                f"- lecture: {'oui' if self.animation_playing else 'non'}"
            )
        if self.build_status:
            status_lines.append(self.build_status)
        if self.solution_status:
            status_lines.append(self.solution_status)
        if self.export_status:
            status_lines.append(self.export_status)
        self.status_var.set("\n\n".join(status_lines))

        if (
            self.summary_axis is None
            or self.force_axis is None
            or self.pose_axis is None
            or self.kinematics_axis is None
            or self.kinematics_velocity_axis is None
            or self.joint_angle_axis is None
            or self.joint_torque_axis is None
            or self.figure_widget is None
        ):
            return

        self._draw_summary_table(blueprint=blueprint, surrogate_apex_height_m=apex_height)
        self._draw_force_figure(contact_profile)
        self._draw_pose_figure(morphology)
        self._draw_kinematics_figure()
        self._draw_joint_figure()
        self.figure_widget.draw_idle()

    def _draw_summary_table(self, *, blueprint, surrogate_apex_height_m: float) -> None:
        """Draw one compact table summarizing the current OCP conditions and jump height."""

        self.summary_axis.clear()
        self.summary_axis.axis("off")

        displayed_height = surrogate_apex_height_m
        height_source = "Surrogate"
        if self.runtime_solution is not None and self.runtime_solution.predicted_apex_height_m is not None:
            displayed_height = self.runtime_solution.predicted_apex_height_m
            height_source = "Runtime"

        table_rows = [
            ("Type d'OCP", blueprint.dynamics_name.replace("_", " ")),
            ("Modele contact", self.contact_model_var.get()),
            ("Force plateforme", f"{self.current_force_newtons():.0f} N"),
            ("Masse sauteur", f"{self.current_mass_kg():.0f} kg"),
            ("Hauteur du saut", f"{displayed_height:.3f} m ({height_source})"),
        ]

        table = self.summary_axis.table(
            cellText=[[label, value] for label, value in table_rows],
            colLabels=["Condition", "Valeur"],
            cellLoc="left",
            colLoc="left",
            loc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.15)
        for (row_index, col_index), cell in table.get_celld().items():
            cell.set_edgecolor("#d0d7de")
            if row_index == 0:
                cell.set_facecolor("#e9f1f7")
                cell.set_text_props(weight="bold")
            elif col_index == 0:
                cell.set_facecolor("#f7f7f7")

        self.summary_axis.set_title("Synthese du saut", fontsize=11, pad=6.0)

    def _draw_force_figure(self, contact_profile: tuple[float, ...]) -> None:
        """Draw the platform and contact-force profiles."""

        profile = self.current_profile()
        times = [
            profile.total_duration * index / max(self.base_settings.n_shooting - 1, 1)
            for index in range(self.base_settings.n_shooting)
        ]
        actuation = [profile.force_at(time) for time in times]

        self.force_axis.clear()
        if self.current_contact_model_key() != CONTACT_MODEL_NO_PLATFORM:
            self.force_axis.plot(times, actuation, color="#d1495b", linewidth=2.2, label="Force plateforme")
        self.force_axis.plot(times, contact_profile, color="#2b59c3", linewidth=2.2, label="Contact surrogate")
        if self.runtime_solution is not None:
            if self.runtime_solution.platform_force_trajectory_n.size:
                self.force_axis.plot(
                    self.runtime_solution.time,
                    self.runtime_solution.platform_force_trajectory_n,
                    color="#f08a24",
                    linewidth=1.8,
                    linestyle="--",
                    label="Force plateforme runtime",
                )
            if self.runtime_solution.contact_force_trajectory_n.size:
                self.force_axis.plot(
                    self.runtime_solution.time,
                    self.runtime_solution.contact_force_trajectory_n,
                    color="#0b6e4f",
                    linewidth=2.4,
                    label=(
                        "Force exterieure runtime (m*(a_com,z + g))"
                        if self.current_contact_model_key() == CONTACT_MODEL_NO_PLATFORM
                        else f"Contact runtime ({self.contact_model_var.get()})"
                    ),
                )
            if self.runtime_solution.final_time_s is not None:
                self.force_axis.axvline(
                    self.runtime_solution.final_time_s,
                    color="#6c9a8b",
                    linestyle="--",
                    linewidth=1.6,
                    label="Temps final runtime",
                )
            if (
                self.runtime_solution.final_time_s is not None
                and self.runtime_solution.final_contact_force_n is not None
            ):
                self.force_axis.scatter(
                    [self.runtime_solution.final_time_s],
                    [self.runtime_solution.final_contact_force_n],
                    color="#0b6e4f",
                    s=32,
                    zorder=5,
                )
        self.force_axis.set_xlabel("Temps (s)")
        self.force_axis.set_ylabel("Force (N)")
        self.force_axis.set_title("Profils de force")
        self.force_axis.grid(alpha=0.25)
        handles, labels = self.force_axis.get_legend_handles_labels()
        if handles:
            self.force_axis.legend(handles, labels, loc="upper left", fontsize=8.5, framealpha=0.9)

    def _draw_pose_figure(self, morphology: AthleteMorphology) -> None:
        """Draw the initial and optimized sagittal stick figures."""

        initial_q = self.current_initial_q()
        initial_points, initial_com, initial_segment_coms = self._display_pose_and_com_from_q(morphology, initial_q)
        avatar_available, _ = avatar_rendering_diagnostics()

        self.pose_axis.clear()
        if avatar_available:
            self._draw_raster_avatar(self.pose_axis, initial_points, alpha=0.22)
            self._draw_kinematic_chain_overlay(
                self.pose_axis,
                initial_points,
                color="#6b6b6b",
                label="Posture initiale",
                linestyle="--",
                alpha=0.65,
            )
        else:
            self._draw_stick_figure(
                self.pose_axis,
                initial_points,
                color="#6b6b6b",
                label="Posture initiale",
                linestyle="--",
                alpha=0.85,
            )
        self._draw_center_of_mass_markers(
            self.pose_axis,
            initial_points["foot"][0],
            initial_com,
            initial_segment_coms,
            alpha=0.32,
            show_labels=False,
        )

        q_trajectory = self._runtime_q_trajectory()
        if q_trajectory is not None and q_trajectory.shape[0] >= 3:
            com_trajectory = self._runtime_com_planar_trajectory(morphology)
            com_velocity_trajectory = self._runtime_com_planar_velocity_trajectory(morphology)
            if com_trajectory is not None:
                self.pose_axis.plot(
                    com_trajectory[0, :],
                    com_trajectory[1, :],
                    color="#577590",
                    linewidth=1.0,
                    linestyle=(0, (2, 2)),
                    alpha=0.9,
                    label="Trajectoire CoM",
                )
                if com_velocity_trajectory is not None:
                    highlighted_frames = sorted(
                        {
                            min(frame_target - 1, q_trajectory.shape[1] - 1)
                            for frame_target in (25, 50, 75, 100)
                            if q_trajectory.shape[1] > 0
                        }
                    )
                    arrow_scale = 0.06
                    for frame_index in highlighted_frames:
                        com_position = com_trajectory[:, frame_index]
                        com_velocity = com_velocity_trajectory[:, frame_index]
                        speed = float(np.linalg.norm(com_velocity))
                        self.pose_axis.annotate(
                            "",
                            xy=(
                                com_position[0] + arrow_scale * com_velocity[0],
                                com_position[1] + arrow_scale * com_velocity[1],
                            ),
                            xytext=(com_position[0], com_position[1]),
                            arrowprops=dict(arrowstyle="->", color="#577590", lw=0.8, alpha=0.8),
                        )
                        self.pose_axis.text(
                            com_position[0] + 0.01,
                            com_position[1] + 0.01,
                            f"{speed:.4f} m/s",
                            color="#577590",
                            fontsize=7,
                            alpha=0.8,
                        )
            animation_index = self.current_animation_frame()
            animated_q = tuple(q_trajectory[:, animation_index].tolist())
            animated_points, animated_com, animated_segment_coms = self._display_pose_and_com_from_q(
                morphology,
                animated_q,
            )
            if not self._draw_raster_avatar(self.pose_axis, animated_points, alpha=1.0):
                self._draw_stick_figure(
                    self.pose_axis,
                    animated_points,
                    color="#f94144",
                    label="Frame animee",
                    linestyle="-",
                    alpha=1.0,
                )
            else:
                self._draw_kinematic_chain_overlay(
                    self.pose_axis,
                    animated_points,
                    color="#1d3557",
                    label="Chaine cinematique",
                    linestyle="-",
                    alpha=0.9,
                )
            self._draw_center_of_mass_markers(
                self.pose_axis,
                animated_points["foot"][0],
                animated_com,
                animated_segment_coms,
                alpha=0.95,
                show_labels=True,
            )
            if self.runtime_solution is not None and self.runtime_solution.contact_force_trajectory_n.size:
                reaction_force_ap = 0.0
                if self.runtime_solution.external_force_ap_trajectory_n.size:
                    reaction_force_ap = float(self.runtime_solution.external_force_ap_trajectory_n[animation_index])
                reaction_force_vertical = float(self.runtime_solution.contact_force_trajectory_n[animation_index])
                max_reaction_force = max(
                    float(
                        np.max(
                            np.hypot(
                                np.asarray(self.runtime_solution.external_force_ap_trajectory_n, dtype=float)
                                if self.runtime_solution.external_force_ap_trajectory_n.size
                                else np.zeros_like(self.runtime_solution.contact_force_trajectory_n),
                                np.asarray(self.runtime_solution.contact_force_trajectory_n, dtype=float),
                            )
                        )
                    ),
                    1.0,
                )
                reaction_scale = 0.28 * morphology.height_m / max_reaction_force
                foot_point = animated_points["foot"]
                self.pose_axis.annotate(
                    "",
                    xy=(
                        foot_point[0] + reaction_scale * reaction_force_ap,
                        foot_point[1] + reaction_scale * reaction_force_vertical,
                    ),
                    xytext=foot_point,
                    arrowprops=dict(arrowstyle="->", color="#90be6d", lw=1.1, alpha=0.9),
                )
                self.pose_axis.text(
                    0.02,
                    0.08,
                    f"Fx={reaction_force_ap:.1f} N | Fz={reaction_force_vertical:.1f} N",
                    transform=self.pose_axis.transAxes,
                    color="#588157",
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#d8f3dc", alpha=0.78),
                )

        self.pose_axis.plot([-0.35, 0.35], [0.0, 0.0], color="#8c5e34", linewidth=4.0)
        self.pose_axis.set_aspect("equal", adjustable="box")
        self.pose_axis.set_xlim(-0.7, 0.8)
        self.pose_axis.set_ylim(-0.05, morphology.height_m + 0.15)
        self.pose_axis.set_xlabel("A/P (m)")
        self.pose_axis.set_ylabel("Vertical (m)")
        self.pose_axis.set_title("Modele 3 segments")
        self.pose_axis.grid(alpha=0.2)
        handles, labels = self.pose_axis.get_legend_handles_labels()
        if handles:
            self.pose_axis.legend(
                handles,
                labels,
                loc="upper right",
                fontsize=8.5,
                framealpha=0.9,
            )

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

    def _draw_kinematic_chain_overlay(
        self,
        axis,
        points: dict[str, tuple[float, float]],
        *,
        color: str,
        label: str,
        linestyle: str,
        alpha: float,
    ) -> None:
        """Draw a light line overlay of the kinematic chain on top of the avatar."""

        chain_x = [points["foot"][0], points["knee"][0], points["hip"][0], points["head"][0]]
        chain_y = [points["foot"][1], points["knee"][1], points["hip"][1], points["head"][1]]
        axis.plot(
            chain_x,
            chain_y,
            color=color,
            linewidth=1.8,
            linestyle=linestyle,
            alpha=alpha,
            zorder=6.6,
            label=label,
        )
        axis.scatter(
            chain_x,
            chain_y,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            s=22,
            alpha=alpha,
            zorder=6.7,
        )

    def _avatar_status_line(self) -> str:
        """Return one concise status line for raster avatar availability."""

        available, message = avatar_rendering_diagnostics()
        return message if available else f"indisponible, {message}"

    def _draw_raster_avatar(
        self,
        axis,
        points: dict[str, tuple[float, float]],
        *,
        alpha: float = 1.0,
    ) -> bool:
        """Draw the animated avatar with raster segments when the assets are available."""

        available, _ = avatar_rendering_diagnostics()
        if not available:
            return False

        ok = True
        ok &= draw_segment_image(
            axis,
            "leg_foot",
            distal_point=points["foot"],
            proximal_point=points["knee"],
            alpha=alpha,
            flip_horizontal=self.avatar_flip_horizontal,
            zorder=4.0,
        )
        ok &= draw_segment_image(
            axis,
            "thigh",
            distal_point=points["knee"],
            proximal_point=points["hip"],
            alpha=alpha,
            flip_horizontal=self.avatar_flip_horizontal,
            zorder=4.2,
        )
        ok &= draw_segment_image(
            axis,
            "trunk",
            distal_point=points["hip"],
            proximal_point=points["head"],
            alpha=alpha,
            flip_horizontal=self.avatar_flip_horizontal,
            zorder=4.4,
        )
        return ok

    def _draw_center_of_mass_markers(
        self,
        axis,
        support_x: float,
        center_of_mass: tuple[float, float],
        segment_centers: dict[str, tuple[float, float]],
        *,
        alpha: float,
        show_labels: bool,
    ) -> None:
        """Draw the global and segmental CoM markers on the sagittal pose figure."""

        projection_x = center_of_mass[0]
        ground_y = 0.0
        axis.plot(
            [projection_x, projection_x],
            [ground_y, center_of_mass[1]],
            color="#8d99ae",
            linewidth=1.0,
            linestyle=(0, (1, 2)),
            alpha=max(0.2, 0.75 * alpha),
            zorder=2.1,
            label="Projection CoM" if show_labels else None,
        )
        axis.scatter(
            [support_x],
            [ground_y],
            color="#8c5e34",
            edgecolors="white",
            linewidths=0.7,
            s=32,
            alpha=alpha,
            zorder=6.0,
            label="Bout du pied" if show_labels else None,
        )
        segment_style = {
            "leg_foot": ("CoM jambe/pied", "#f4a261", "s"),
            "thigh": ("CoM cuisse", "#e76f51", "^"),
            "trunk": ("CoM tronc", "#6d597a", "D"),
        }
        for name, (_label, color, marker) in segment_style.items():
            point = segment_centers[name]
            axis.scatter(
                [point[0]],
                [point[1]],
                color=color,
                edgecolors="white",
                linewidths=0.7,
                s=34,
                marker=marker,
                alpha=alpha,
                zorder=5.8,
            )

        axis.scatter(
            [center_of_mass[0]],
            [center_of_mass[1]],
            color="#1d3557",
            edgecolors="white",
            linewidths=0.8,
            s=54,
            marker="o",
            alpha=alpha,
            zorder=6.1,
            label="CoM global" if show_labels else None,
        )
        if show_labels:
            horizontal_offset = projection_x - support_x
            axis.text(
                0.02,
                0.98,
                "Projection CoM:\n"
                f"x_CoM={projection_x:+.3f} m\n"
                f"x_appui={support_x:+.3f} m\n"
                f"Δx={horizontal_offset:+.3f} m",
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                color="#1d3557",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#cbd5e1", alpha=0.85),
                zorder=6.2,
            )

    def _draw_kinematics_figure(self) -> None:
        """Draw runtime kinematics when one solved trajectory is available."""

        self.kinematics_axis.clear()
        self.kinematics_velocity_axis.clear()
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
            self.kinematics_velocity_axis.set_yticks([])
            return

        time = self.runtime_solution.time
        frame_index = self.current_animation_frame()
        self.kinematics_axis.plot(
            time,
            self.runtime_solution.com_height_trajectory_m,
            color="#3a7d44",
            linewidth=2.2,
            label="Position CoM",
        )
        self.kinematics_velocity_axis.plot(
            time,
            self.runtime_solution.com_vertical_velocity_trajectory_m_s,
            color="#577590",
            linewidth=1.8,
            linestyle="--",
            label="Vitesse verticale CoM",
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
        self.kinematics_axis.axvline(
            time[frame_index],
            color="#f94144",
            linewidth=1.8,
            linestyle="--",
            label="Frame courant",
        )

        self.kinematics_axis.set_xlabel("Temps (s)")
        self.kinematics_axis.set_ylabel("Position (m)")
        self.kinematics_velocity_axis.set_ylabel("Vitesse CoM (m/s)")
        self.kinematics_velocity_axis.yaxis.set_label_position("right")
        self.kinematics_velocity_axis.yaxis.tick_right()
        self.kinematics_axis.set_title("Cinematique runtime")
        self.kinematics_axis.grid(alpha=0.25)
        position_handles, position_labels = self.kinematics_axis.get_legend_handles_labels()
        velocity_handles, velocity_labels = self.kinematics_velocity_axis.get_legend_handles_labels()
        self.kinematics_axis.legend(
            position_handles + velocity_handles,
            position_labels + velocity_labels,
            loc="upper left",
            fontsize=9,
            framealpha=0.9,
        )

    def _draw_joint_figure(self) -> None:
        """Draw the knee/hip angles and torques underneath the runtime kinematics."""

        self.joint_angle_axis.clear()
        self.joint_torque_axis.clear()
        if self.runtime_solution is None or self.runtime_solution.time.size == 0:
            self.joint_angle_axis.text(
                0.5,
                0.5,
                "Angles et couples\nindisponibles sans solution.",
                ha="center",
                va="center",
                transform=self.joint_angle_axis.transAxes,
            )
            self.joint_angle_axis.set_title("Angles et couples")
            self.joint_angle_axis.set_xticks([])
            self.joint_angle_axis.set_yticks([])
            self.joint_torque_axis.set_yticks([])
            return

        time = self.runtime_solution.time
        frame_index = self.current_animation_frame()
        angle_trajectories = self._runtime_joint_angle_trajectory_deg()
        torque_bundle = self._runtime_joint_torque_trajectory_nm()

        if angle_trajectories is not None:
            self.joint_angle_axis.plot(time, angle_trajectories["Genou"], color="#33658a", linewidth=2.0, label="Genou (deg)")
            self.joint_angle_axis.plot(time, angle_trajectories["Hanche"], color="#86bbd8", linewidth=2.0, label="Hanche (deg)")
        if torque_bundle is not None:
            control_time, torque_trajectories = torque_bundle
            torque_limits = self._joint_torque_limits_nm()
            self.joint_torque_axis.step(
                control_time,
                torque_trajectories["Genou"],
                color="#f26419",
                linewidth=1.8,
                where="post",
                label="Tau genou (Nm)",
            )
            self.joint_torque_axis.step(
                control_time,
                torque_trajectories["Hanche"],
                color="#f6ae2d",
                linewidth=1.8,
                where="post",
                label="Tau hanche (Nm)",
            )
            self.joint_torque_axis.axhline(
                torque_limits["Genou"],
                color="#f26419",
                linewidth=1.0,
                linestyle=":",
                alpha=0.8,
            )
            self.joint_torque_axis.axhline(
                -torque_limits["Genou"],
                color="#f26419",
                linewidth=1.0,
                linestyle=":",
                alpha=0.8,
            )
            self.joint_torque_axis.axhline(
                torque_limits["Hanche"],
                color="#f6ae2d",
                linewidth=1.0,
                linestyle=":",
                alpha=0.8,
            )
            self.joint_torque_axis.axhline(
                -torque_limits["Hanche"],
                color="#f6ae2d",
                linewidth=1.0,
                linestyle=":",
                alpha=0.8,
            )
            self.joint_angle_axis.text(
                0.02,
                0.98,
                f"Limites couples:\nGenou ±{torque_limits['Genou']:.0f} Nm\nHanche ±{torque_limits['Hanche']:.0f} Nm",
                transform=self.joint_angle_axis.transAxes,
                ha="left",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#e5e7eb", alpha=0.85),
            )

        self.joint_angle_axis.axvline(
            time[frame_index],
            color="#f94144",
            linewidth=1.5,
            linestyle="--",
        )
        self.joint_angle_axis.set_xlabel("Temps (s)")
        self.joint_angle_axis.set_ylabel("Angle (deg)")
        self.joint_torque_axis.set_ylabel("Couple (Nm)")
        self.joint_angle_axis.set_title("Angles et couples articulaires", pad=4)
        self.joint_angle_axis.grid(alpha=0.25)

        angle_handles, angle_labels = self.joint_angle_axis.get_legend_handles_labels()
        torque_handles, torque_labels = self.joint_torque_axis.get_legend_handles_labels()
        self.joint_angle_axis.legend(
            angle_handles + torque_handles,
            angle_labels + torque_labels,
            loc="lower left",
            fontsize=9,
            framealpha=0.9,
        )

    def _analytic_pose_points_from_q(
        self,
        morphology: AthleteMorphology,
        q_values: tuple[float, float, float, float, float],
    ) -> dict[str, tuple[float, float]]:
        """Compute one analytic fallback pose when the true `BiorbdModel` is unavailable."""

        lengths = morphology.segment_lengths
        if len(q_values) == 5:
            q_root_x, q_root_z, q_root_rot, q_knee, q_hip = q_values
        elif len(q_values) == 3:
            q_root_x, q_root_z = 0.0, 0.0
            q_root_rot, q_knee, q_hip = q_values
        else:
            raise ValueError("The displayed jumper pose expects 3 or 5 generalized coordinates")

        def advance(origin: tuple[float, float], angle: float, length: float) -> tuple[float, float]:
            return (origin[0] + length * math.sin(angle), origin[1] + length * math.cos(angle))

        leg_angle = q_root_rot
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
        self._log_runtime_request("Construction OCP")
        self._show_busy_indicator("Construction de l'OCP en cours")
        try:
            summary = build_ocp_runtime_summary(
                settings=self.current_settings(),
                peak_force_newtons=self.current_force_newtons(),
            )
            print(f"[SynchroJump] Construction OCP terminee | succes={'oui' if summary.success else 'non'}")
            if summary.success:
                self.build_status = (
                    "Construction runtime:\n"
                    f"- succes: oui\n"
                    f"- phases: {summary.n_phases}\n"
                    f"- etats: {', '.join(summary.state_names)}\n"
                    f"- controles: {', '.join(summary.control_names)}"
                )
                self._set_ocp_built_state(True)
            else:
                self.build_status = f"Construction runtime:\n- succes: non\n- message: {summary.message}"
                self._set_ocp_built_state(False)
            self.solution_status = ""
            self.refresh()
        finally:
            self._hide_busy_indicator()

    def solve_ocp(self) -> None:
        """Solve one quick runtime OCP and expose the trajectories in the GUI."""

        if not self.ocp_is_built:
            self.solution_status = "Resolution runtime:\n- succes: non\n- message: construis d'abord l'OCP"
            self.refresh()
            return

        self._stop_animation()
        self.solution_status = (
            "Resolution runtime:\n"
            f"- en cours...\n- iterations max: {self.current_solver_iterations()}\n"
            "- journal IPOPT detaille: terminal"
        )
        self.refresh()
        self._log_runtime_request("Lancement resolution OCP")
        print(
            "[SynchroJump]"
            f" Options solveur | iterations_max={self.current_solver_iterations()}"
            f" | cache={'oui' if self.use_cache_var.get() else 'non'}"
        )
        self._show_busy_indicator("Resolution de l'OCP en cours")
        try:
            summary = solve_ocp_runtime_summary(
                settings=self.current_settings(),
                peak_force_newtons=self.current_force_newtons(),
                use_cache=bool(self.use_cache_var.get()),
                maximum_iterations=self.current_solver_iterations(),
            )
            print(
                "[SynchroJump]"
                f" Resolution OCP terminee | succes={'oui' if summary.success else 'non'}"
                f" | cache={'oui' if summary.from_cache else 'non'}"
            )
            if summary.success:
                self.runtime_solution = summary
                self.animation_frame_var.set(0)
                self.animation_scale.configure(to=max(summary.time.size - 1, 0))
                self._sync_animation_button_label()
                self.solution_status = (
                    "Resolution runtime:\n"
                    f"- succes: oui\n"
                    f"- cache: {'oui' if summary.from_cache else 'non'}\n"
                    f"- modele contact: {self.contact_model_var.get()}\n"
                    f"- iterations demandees: {summary.requested_iterations}\n"
                    f"- statut solveur: {summary.solver_status}\n"
                    f"- temps final: {summary.final_time_s:.2f} s\n"
                    f"- force contact finale: {summary.final_contact_force_n:.2f} N\n"
                    f"- decollage respecte: {'oui' if summary.takeoff_condition_satisfied else 'non'}\n"
                    f"- apex predit: {summary.predicted_apex_height_m:.2f} m\n"
                    f"- temps solveur: {summary.solve_time_s:.2f} s"
                )
            else:
                self.runtime_solution = None
                self.animation_scale.configure(to=0)
                self._sync_animation_button_label()
                self.solution_status = f"Resolution runtime:\n- succes: non\n- message: {summary.message}"
            self.refresh()
        finally:
            self._hide_busy_indicator()

    def toggle_animation(self) -> None:
        """Start or pause the runtime trajectory animation."""

        if self.runtime_solution is None or self.runtime_solution.time.size == 0:
            return

        if self.animation_playing:
            self._stop_animation()
            self.refresh(dynamic_only=True)
            return

        self.animation_playing = True
        self._sync_animation_button_label()
        self._schedule_animation_step()
        self.refresh(dynamic_only=True)

    def reset_animation(self) -> None:
        """Return the animation cursor to the first frame."""

        self._stop_animation()
        self.animation_frame_var.set(0)
        self.refresh(dynamic_only=True)

    def _schedule_animation_step(self) -> None:
        """Schedule the next animation frame if playback is active."""

        if not self.animation_playing:
            return
        self.animation_job = self.root.after(self.animation_delay_ms, self._advance_animation)

    def _advance_animation(self) -> None:
        """Advance the animation cursor by one frame."""

        if self.runtime_solution is None or self.runtime_solution.time.size == 0:
            self._stop_animation()
            return

        next_index = self.current_animation_frame() + 1
        if next_index >= self.runtime_solution.time.size:
            next_index = 0
        self.animation_frame_var.set(next_index)
        self.refresh(dynamic_only=True)
        self._schedule_animation_step()

    def _stop_animation(self) -> None:
        """Stop the playback loop if one is active."""

        self.animation_playing = False
        if self.animation_job is not None:
            self.root.after_cancel(self.animation_job)
            self.animation_job = None
        self._sync_animation_button_label()

    def export_model(self) -> None:
        """Export the current `.bioMod` file and append the path to the status text."""

        builder = VerticalJumpBioptimOcpBuilder(settings=self.current_settings())
        model_path = builder.export_model(Path("generated"))
        self.export_status = f"Modele exporte: {model_path}"
        self.refresh()


def launch_app() -> None:
    """Start the interactive GUI."""

    root = tk.Tk()
    app = SynchroJumpApp(root)
    print(f"[SynchroJump] Avatar raster | {app._avatar_status_line()}")
    root.mainloop()
