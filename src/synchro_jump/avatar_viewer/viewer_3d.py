"""Minimal Panda3D + Qt viewer for the rigged avatar prototype."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from synchro_jump.avatar_viewer.glb_inspector import GlbRigInspector
from synchro_jump.avatar_viewer.mapping import RigBoneMapping, default_cc_base_mapping
from synchro_jump.avatar_viewer.retargeting import BiomechanicalPose, BiomechanicalRetargeter
from synchro_jump.avatar_viewer.rigged_avatar import (
    AvatarViewerDependencyError,
    BiomechanicalRigController,
    RiggedAvatarLoader,
)


def _require_qt():
    """Import Qt lazily and raise one explicit installation hint."""

    try:
        from PySide6.QtCore import QTimer, Qt  # type: ignore
        from PySide6.QtWidgets import (  # type: ignore
            QApplication,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QPushButton,
            QPlainTextEdit,
            QSlider,
            QVBoxLayout,
            QWidget,
        )
    except Exception as exc:  # pragma: no cover - optional dependency
        raise AvatarViewerDependencyError(
            "Qt dependencies are missing. Install `PySide6`, `panda3d`, and "
            "`panda3d-gltf` to launch the 3D avatar prototype."
        ) from exc
    return {
        "QApplication": QApplication,
        "QHBoxLayout": QHBoxLayout,
        "QLabel": QLabel,
        "QMainWindow": QMainWindow,
        "QPlainTextEdit": QPlainTextEdit,
        "QPushButton": QPushButton,
        "QSlider": QSlider,
        "QTimer": QTimer,
        "QVBoxLayout": QVBoxLayout,
        "QWidget": QWidget,
        "Qt": Qt,
    }


def _require_panda3d():
    """Import Panda3D lazily and raise one explicit installation hint."""

    try:
        from direct.showbase.ShowBase import ShowBase  # type: ignore
        from panda3d.core import AmbientLight, DirectionalLight, WindowProperties, loadPrcFileData  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise AvatarViewerDependencyError(
            "Panda3D dependencies are missing. Install `panda3d` and `panda3d-gltf` "
            "alongside `PySide6` to enable the 3D avatar viewer."
        ) from exc
    return ShowBase, AmbientLight, DirectionalLight, WindowProperties, loadPrcFileData


@dataclass(frozen=True)
class AvatarViewerConfig:
    """Small configuration bundle for the 3D viewer."""

    asset_path: Path
    mapping: RigBoneMapping
    window_title: str = "Synchro Jump 3D Avatar"
    clear_color_rgba: tuple[float, float, float, float] = (0.92, 0.94, 0.98, 1.0)
    camera_position_xyz: tuple[float, float, float] = (2.3, -5.5, 1.8)
    camera_target_xyz: tuple[float, float, float] = (0.0, 0.0, 1.1)


class _EmbeddedPandaApp:
    """Own the Panda3D scene embedded in a Qt native widget."""

    def __init__(self, parent_window_id: int, width: int, height: int, config: AvatarViewerConfig) -> None:
        ShowBase, AmbientLight, DirectionalLight, WindowProperties, loadPrcFileData = _require_panda3d()

        loadPrcFileData("", "window-type none")
        loadPrcFileData("", "audio-library-name null")
        loadPrcFileData("", "sync-video false")
        self._show_base = ShowBase(windowType="none")
        properties = WindowProperties()
        properties.setParentWindow(parent_window_id)
        properties.setSize(max(width, 1), max(height, 1))
        self._show_base.openDefaultWindow(props=properties)
        self._show_base.disableMouse()
        self._show_base.win.setClearColor(config.clear_color_rgba)

        ambient = AmbientLight("ambient")
        ambient.setColor((0.55, 0.55, 0.6, 1.0))
        ambient_np = self._show_base.render.attachNewNode(ambient)
        self._show_base.render.setLight(ambient_np)

        key = DirectionalLight("key")
        key.setColor((0.9, 0.9, 0.95, 1.0))
        key_np = self._show_base.render.attachNewNode(key)
        key_np.setHpr(-25.0, -35.0, 0.0)
        self._show_base.render.setLight(key_np)

        self._show_base.camera.setPos(*config.camera_position_xyz)
        self._show_base.camera.lookAt(*config.camera_target_xyz)

    @property
    def render(self):
        """Return the Panda render root."""

        return self._show_base.render

    def resize(self, width: int, height: int) -> None:
        """Resize the embedded Panda window."""

        _ShowBase, _AmbientLight, _DirectionalLight, WindowProperties, _loadPrcFileData = _require_panda3d()
        props = WindowProperties()
        props.setSize(max(width, 1), max(height, 1))
        self._show_base.win.requestProperties(props)

    def step(self) -> None:
        """Advance one Panda frame from the Qt timer."""

        self._show_base.taskMgr.step()


def _build_avatar_viewer_widget_class():
    """Create the Qt widget lazily so imports stay optional."""

    qt = _require_qt()
    QWidget = qt["QWidget"]

    class Avatar3DViewer(QWidget):  # type: ignore[misc]
        """Native Qt widget embedding Panda3D in-place."""

        def __init__(self, config: AvatarViewerConfig, poses: list[BiomechanicalPose]) -> None:
            super().__init__()
            self._config = config
            self._poses = poses
            self._current_frame_index = 0
            self._embedded_app: _EmbeddedPandaApp | None = None
            self._controller: BiomechanicalRigController | None = None
            self.setMinimumSize(640, 480)
            self.setAttribute(qt["Qt"].WidgetAttribute.WA_NativeWindow, True)
            self.setAttribute(qt["Qt"].WidgetAttribute.WA_DontCreateNativeAncestors, True)

        def initialize_scene(self) -> None:
            """Create the Panda3D scene and load the rigged avatar once."""

            if self._embedded_app is not None:
                return
            self._embedded_app = _EmbeddedPandaApp(int(self.winId()), self.width(), self.height(), self._config)
            retargeter = BiomechanicalRetargeter(self._config.mapping)
            avatar = RiggedAvatarLoader(self._config.asset_path, self._config.mapping).load(self._embedded_app.render)
            self._controller = BiomechanicalRigController(avatar=avatar, retargeter=retargeter)
            self.set_frame(0)

        def set_frame(self, frame_index: int) -> None:
            """Apply one pose frame to the avatar."""

            if not self._poses:
                return
            self.initialize_scene()
            assert self._controller is not None
            bounded_index = max(0, min(frame_index, len(self._poses) - 1))
            self._current_frame_index = bounded_index
            self._controller.apply_biomechanical_pose(self._poses[bounded_index])

        def current_frame_index(self) -> int:
            """Return the currently displayed frame index."""

            return self._current_frame_index

        def step_render_loop(self) -> None:
            """Advance the Panda render loop by one frame."""

            self.initialize_scene()
            assert self._embedded_app is not None
            self._embedded_app.step()

        def resizeEvent(self, event) -> None:  # pragma: no cover - GUI integration
            super().resizeEvent(event)
            if self._embedded_app is not None:
                self._embedded_app.resize(self.width(), self.height())

        def showEvent(self, event) -> None:  # pragma: no cover - GUI integration
            super().showEvent(event)
            self.initialize_scene()

    return Avatar3DViewer


class AvatarViewerDemoWindow:
    """Qt main window bundling the 3D viewer and rig inspection text."""

    def __init__(self, config: AvatarViewerConfig, poses: list[BiomechanicalPose]) -> None:
        qt = _require_qt()
        QMainWindow = qt["QMainWindow"]
        QWidget = qt["QWidget"]
        QHBoxLayout = qt["QHBoxLayout"]
        QVBoxLayout = qt["QVBoxLayout"]
        QPlainTextEdit = qt["QPlainTextEdit"]
        QPushButton = qt["QPushButton"]
        QSlider = qt["QSlider"]
        QLabel = qt["QLabel"]
        Qt = qt["Qt"]
        QTimer = qt["QTimer"]

        Avatar3DViewer = _build_avatar_viewer_widget_class()

        self._window = QMainWindow()
        self._window.setWindowTitle(config.window_title)

        central = QWidget()
        layout = QHBoxLayout(central)

        self.viewer = Avatar3DViewer(config, poses)
        layout.addWidget(self.viewer, stretch=3)

        inspector = GlbRigInspector.from_glb(config.asset_path)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.addWidget(QLabel("Rig inspection"))
        self.report_box = QPlainTextEdit()
        self.report_box.setReadOnly(True)
        self.report_box.setPlainText(inspector.build_report(config.mapping).to_multiline_text())
        right_layout.addWidget(self.report_box, stretch=1)

        self.play_button = QPushButton("Play / Pause")
        right_layout.addWidget(self.play_button)

        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(len(poses) - 1, 0))
        right_layout.addWidget(self.frame_slider)
        layout.addWidget(right_panel, stretch=2)
        self._window.setCentralWidget(central)

        self._playing = False
        self._pose_count = len(poses)
        self._render_timer = QTimer()
        self._render_timer.setInterval(16)
        self._render_timer.timeout.connect(self._on_tick)
        self._render_timer.start()
        self.play_button.clicked.connect(self._toggle_play)
        self.frame_slider.valueChanged.connect(self.viewer.set_frame)

    def _toggle_play(self) -> None:
        """Toggle sequence playback."""

        self._playing = not self._playing

    def _on_tick(self) -> None:
        """Advance playback and step Panda's render loop."""

        self.viewer.step_render_loop()
        if self._playing and self._pose_count > 0:
            next_frame = (self.viewer.current_frame_index() + 1) % self._pose_count
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(next_frame)
            self.frame_slider.blockSignals(False)
            self.viewer.set_frame(next_frame)

    def show(self) -> None:
        """Show the main window."""

        self._window.show()


def launch_avatar_viewer(config: AvatarViewerConfig, poses: list[BiomechanicalPose]) -> int:
    """Launch the standalone Qt + Panda3D avatar viewer."""

    qt = _require_qt()
    QApplication = qt["QApplication"]
    application = QApplication.instance() or QApplication([])
    window = AvatarViewerDemoWindow(config=config, poses=poses)
    window.show()
    return application.exec()


def default_avatar_viewer_config(asset_path: Path) -> AvatarViewerConfig:
    """Return the default viewer configuration for the bundled CC Base rig."""

    return AvatarViewerConfig(asset_path=Path(asset_path), mapping=default_cc_base_mapping())
