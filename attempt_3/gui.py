#!/usr/bin/env python3
"""Tricho desktop GUI (PySide6).

Two tabs:
  * Camera   - live preview with the hair mask and contact points overlaid
  * Settings - edit the five tunables, then Save or Reset to Defaults

The detector runs in a background QThread so the UI stays responsive. Saving
writes config.json and hot-applies the new values to the running detector.
"""

import re
import sys
import threading
import time

import cv2
from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from detector import (
    CONFIG_PATH,
    DEFAULTS,
    ConfigError,
    Detector,
    draw_overlay,
    load_config,
    save_config,
    validate_config,
)

PREVIEW_MAX_WIDTH = 800
DWELL_FRAMES = 3


class DetectorWorker(QThread):
    frame_ready = Signal(object)   # BGR ndarray
    status_ready = Signal(bool)    # warn
    failed = Signal(str)

    def __init__(self, cfg, camera_index=0, dwell_frames=DWELL_FRAMES):
        super().__init__()
        self._cfg = dict(cfg)
        self._camera_index = camera_index
        self._dwell = dwell_frames
        self._stop = threading.Event()
        self._lock = threading.Lock()

    def set_config(self, cfg):
        with self._lock:
            self._cfg = dict(cfg)

    def _snapshot(self):
        with self._lock:
            return dict(self._cfg)

    def stop(self):
        self._stop.set()
        self.wait(3000)

    def run(self):
        try:
            detector = Detector()
        except Exception as e:                       # models missing/corrupt
            self.failed.emit(f"Could not load models: {e}")
            return

        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            detector.close()
            self.failed.emit(f"Could not open camera index {self._camera_index}")
            return

        next_t = time.monotonic()
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.failed.emit("Camera read failed")
                    break

                cfg = self._snapshot()
                ts = int(time.monotonic() * 1000)
                result = detector.process(frame, ts, cfg, self._dwell)

                vis = draw_overlay(frame, result)
                h, w = vis.shape[:2]
                if w > PREVIEW_MAX_WIDTH:
                    vis = cv2.resize(
                        vis, (PREVIEW_MAX_WIDTH, int(h * PREVIEW_MAX_WIDTH / w)))

                self.frame_ready.emit(vis)
                self.status_ready.emit(result.warn)

                next_t += cfg["frame_interval_ms"] / 1000.0
                sleep = next_t - time.monotonic()
                if sleep > 0:
                    self._stop.wait(sleep)
                else:
                    next_t = time.monotonic()
        finally:
            cap.release()
            detector.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tricho")
        self.resize(920, 720)

        cfg, warning = self._initial_config()
        self.cfg = cfg
        self._last_image = None

        tabs = QTabWidget()
        tabs.addTab(self._build_camera_tab(), "Camera")
        tabs.addTab(self._build_settings_tab(), "Settings")
        self.setCentralWidget(tabs)

        self._populate(cfg)
        if warning:
            self._set_status(warning, error=True)

        self.worker = DetectorWorker(cfg)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status_ready.connect(self._on_status)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _initial_config():
        try:
            return load_config(CONFIG_PATH), ""
        except ConfigError as e:
            return validate_config(dict(DEFAULTS)), f"{e}; using built-in defaults"

    def _build_camera_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        self.image_label = QLabel("Starting camera…")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 360)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Expanding)
        self.image_label.setStyleSheet("background:#111; color:#888;")

        self.state_label = QLabel("safe")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.state_label.setMinimumHeight(40)
        self.state_label.setStyleSheet(self._state_style(False))

        layout.addWidget(self.image_label, 1)
        layout.addWidget(self.state_label)
        return page

    def _build_settings_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        form = QFormLayout()

        self.points_edit = QLineEdit()
        self.points_edit.setPlaceholderText("4,8,12,16,20,3,7,11,15,19")
        form.addRow("Contact points (0–20):", self.points_edit)
        hint = QLabel("Fingertips: 4, 8, 12, 16, 20 · DIP joints: 3, 7, 11, 15, 19")
        hint.setStyleSheet("color:#777;")
        form.addRow("", hint)

        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.0, 3600.0)
        self.cooldown_spin.setSingleStep(0.5)
        self.cooldown_spin.setDecimals(1)
        self.cooldown_spin.setSuffix(" s")
        form.addRow("Cooldown:", self.cooldown_spin)

        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(10, 5000)
        self.interval_spin.setSingleStep(10)
        self.interval_spin.setSuffix(" ms")
        self.interval_spin.valueChanged.connect(self._update_fps)
        self.fps_label = QLabel()
        self.fps_label.setStyleSheet("color:#777;")
        form.addRow("Frame interval:", self.interval_spin)
        form.addRow("", self.fps_label)

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0, 500.0)
        self.radius_spin.setSingleStep(1.0)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setSuffix(" px")
        form.addRow("Contact radius:", self.radius_spin)

        self.title_edit = QLineEdit()
        form.addRow("Notification title:", self.title_edit)

        self.message_edit = QLineEdit()
        form.addRow("Notification message:", self.message_edit)

        outer.addLayout(form)
        outer.addStretch(1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        outer.addWidget(self.status_label)

        buttons = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self._save)
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.clicked.connect(self._reset)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.reset_button)
        outer.addLayout(buttons)

        return page

    # ------------------------------------------------------------ form helpers
    def _populate(self, cfg):
        self.points_edit.setText(",".join(str(p) for p in cfg["contact_points"]))
        self.cooldown_spin.setValue(float(cfg["cooldown_seconds"]))
        self.interval_spin.setValue(int(round(cfg["frame_interval_ms"])))
        self.radius_spin.setValue(float(cfg["radius_px"]))
        self.title_edit.setText(cfg["notification_title"])
        self.message_edit.setText(cfg["notification_message"])
        self._update_fps(self.interval_spin.value())

    def _collect(self):
        raw = self.points_edit.text().strip()
        try:
            points = [int(tok) for tok in re.split(r"[,\s]+", raw) if tok]
        except ValueError:
            raise ConfigError(
                "contact_points must be comma-separated integers in 0..20")
        return validate_config({
            "contact_points": points,
            "cooldown_seconds": self.cooldown_spin.value(),
            "frame_interval_ms": float(self.interval_spin.value()),
            "radius_px": self.radius_spin.value(),
            "notification_title": self.title_edit.text(),
            "notification_message": self.message_edit.text(),
        })

    def _update_fps(self, ms):
        self.fps_label.setText(f"≈ {1000.0 / ms:.1f} FPS")

    def _set_status(self, text, error=False):
        self.status_label.setStyleSheet(
            "color:#c62828;" if error else "color:#2e7d32;")
        self.status_label.setText(text)

    # ---------------------------------------------------------------- actions
    def _save(self):
        try:
            cfg = self._collect()
        except ConfigError as e:
            self._set_status(f"Not saved: {e}", error=True)
            return
        try:
            save_config(CONFIG_PATH, cfg)
        except OSError as e:
            self._set_status(f"Could not write {CONFIG_PATH}: {e}", error=True)
            return
        self.cfg = cfg
        self.worker.set_config(cfg)
        self._set_status("Saved.", error=False)

    def _reset(self):
        self._populate(validate_config(dict(DEFAULTS)))
        self._set_status("Defaults loaded — click Save to apply.", error=False)

    # --------------------------------------------------------------- callbacks
    def _on_frame(self, frame):
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._last_image = QImage(
            rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self._last_image is None:
            return
        pixmap = QPixmap.fromImage(self._last_image)
        self.image_label.setPixmap(pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _on_status(self, warn):
        self.state_label.setText(
            "PULLING — hands away from your hair" if warn else "safe")
        self.state_label.setStyleSheet(self._state_style(warn))

    def _on_failed(self, message):
        self.state_label.setText("error")
        self.state_label.setStyleSheet(self._state_style(True))
        self._set_status(message, error=True)

    @staticmethod
    def _state_style(warn):
        if warn:
            return ("background:#c62828; color:white; font-size:20px; "
                    "font-weight:bold; border-radius:6px; padding:6px;")
        return ("background:#2e7d32; color:white; font-size:20px; "
                "font-weight:bold; border-radius:6px; padding:6px;")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
