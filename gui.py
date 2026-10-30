#!/usr/bin/env python3
"""Tricho desktop GUI (PySide6).

Three tabs:
  * Camera   - live preview with the hair mask and contact points overlaid
  * Settings - edit the five tunables, then Save or Reset to Defaults
  * Activity - yearly heatmap of pulling episodes

The detector runs in a background QThread so the UI stays responsive. Saving
writes config.json and hot-applies the new values to the running detector.
"""

import datetime
import os
import re
import sys
import threading
import time

import cv2
from PySide6.QtCore import QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
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

from activity import counts_for_year, last_pull_time, load_activity, years_in
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
from heatmap import HeatmapWidget
from sound import SoundPlayer

PREVIEW_MAX_WIDTH = 800
DWELL_FRAMES = 3
FEEDBACK_URL = ("https://docs.google.com/forms/d/e/"
                "1FAIpQLSfDUIuLhHDg1paNdIf2CJqjNX1tQKJW5cqKV-TVLTISETO0Cw/"
                "viewform?usp=publish-editor")


class DetectorWorker(QThread):
    frame_ready = Signal(object)   # BGR ndarray
    status_ready = Signal(bool)    # warn
    pull_recorded = Signal()       # a new pulling episode was counted
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
                if result.episode_started:
                    self.pull_recorded.emit()

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
        self._preview_player = None
        self._last_pull = None
        self._has_pulls = False

        tabs = QTabWidget()
        tabs.addTab(self._build_camera_tab(), "Camera")
        tabs.addTab(self._build_settings_tab(), "Settings")
        tabs.addTab(self._build_activity_tab(), "Activity")
        self.setCentralWidget(tabs)
        self.tabs = tabs
        tabs.currentChanged.connect(self._on_tab_changed)

        self._populate(cfg)
        if warning:
            self._set_status(warning, error=True)
        self._refresh_activity()

        self.worker = DetectorWorker(cfg)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.status_ready.connect(self._on_status)
        self.worker.pull_recorded.connect(self._refresh_activity)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

        self._streak_timer = QTimer(self)
        self._streak_timer.setInterval(1000)
        self._streak_timer.timeout.connect(self._update_streak)
        self._streak_timer.start()

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

        sound_row = QHBoxLayout()
        self.sound_edit = QLineEdit()
        self.sound_edit.setPlaceholderText(
            "Path to a .wav, .mp3, .ogg or .flac file")
        sound_row.addWidget(self.sound_edit)
        self.sound_browse = QPushButton("Browse…")
        self.sound_browse.clicked.connect(self._browse_sound)
        sound_row.addWidget(self.sound_browse)
        self.sound_test = QPushButton("Test")
        self.sound_test.clicked.connect(self._test_sound)
        sound_row.addWidget(self.sound_test)
        form.addRow("Sound effect:", sound_row)

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

    def _build_activity_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        header = QHBoxLayout()
        header.addWidget(QLabel("Year:"))
        self.year_combo = QComboBox()
        self.year_combo.currentTextChanged.connect(self._refresh_activity)
        header.addWidget(self.year_combo)
        header.addStretch(1)
        self.total_label = QLabel("")
        header.addWidget(self.total_label)
        layout.addLayout(header)

        self.heatmap = HeatmapWidget()
        layout.addWidget(self.heatmap)

        self.contact_label = QLabel(
            f'<a href="{FEEDBACK_URL}" '
            'style="color:#2a6fb0; font-size:15px;">Contact us</a>')
        self.contact_label.setOpenExternalLinks(True)
        self.contact_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.contact_label)

        self.description_label = QLabel(
            "Each square is one day. A day with no pulling stays bright green; "
            "every pull removes 10 from the green channel, so darker squares "
            "mean more pulling that day.")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color:#666;")
        layout.addWidget(self.description_label)

        self.streak_label = QLabel("No pulls recorded yet.")
        self.streak_label.setStyleSheet(
            "font-size:15px; font-weight:bold; padding-top:4px;")
        layout.addWidget(self.streak_label)

        layout.addStretch(1)
        return page

    # ------------------------------------------------------------ form helpers
    def _populate(self, cfg):
        self.points_edit.setText(",".join(str(p) for p in cfg["contact_points"]))
        self.cooldown_spin.setValue(float(cfg["cooldown_seconds"]))
        self.interval_spin.setValue(int(round(cfg["frame_interval_ms"])))
        self.radius_spin.setValue(float(cfg["radius_px"]))
        self.title_edit.setText(cfg["notification_title"])
        self.message_edit.setText(cfg["notification_message"])
        self.sound_edit.setText(cfg["sound_path"])
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
            "sound_path": self.sound_edit.text().strip(),
        })

    def _update_fps(self, ms):
        self.fps_label.setText(f"≈ {1000.0 / ms:.1f} FPS")

    def _browse_sound(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose sound effect", self.sound_edit.text(),
            "Audio files (*.wav *.mp3 *.ogg *.flac);;All files (*)")
        if path:
            self.sound_edit.setText(path)

    def _test_sound(self):
        path = self.sound_edit.text().strip()
        if not path:
            self._set_status("No sound file selected.", error=True)
            return
        if not os.path.isfile(path):
            self._set_status(f"File not found: {path}", error=True)
            return
        if self._preview_player is None:
            self._preview_player = SoundPlayer()
        self._preview_player.play(path)
        self._set_status("Playing sound…", error=False)

    def _set_status(self, text, error=False):
        self.status_label.setStyleSheet(
            "color:#c62828;" if error else "color:#2e7d32;")
        self.status_label.setText(text)

    def _refresh_activity(self, *_args):
        data = load_activity()
        today = datetime.date.today().year
        years = sorted(set(years_in(data)) | {today})
        labels = [str(y) for y in years]
        current = self.year_combo.currentText()

        self.year_combo.blockSignals(True)
        self.year_combo.clear()
        self.year_combo.addItems(labels)
        self.year_combo.setCurrentText(
            current if current in labels else str(today))
        self.year_combo.blockSignals(False)

        self._last_pull = last_pull_time(data)
        self._has_pulls = any(
            counts_for_year(data, y) for y in years_in(data))
        self._update_streak()

        year = int(self.year_combo.currentText())
        counts = counts_for_year(data, year)
        self.heatmap.set_data(year, counts)
        self.total_label.setText(
            f"{sum(int(v) for v in counts.values())} pulls in {year}")

    def _on_tab_changed(self, index):
        if self.tabs.tabText(index) == "Activity":
            self._refresh_activity()

    @staticmethod
    def _plural(value, unit):
        return f"{value} {unit}" + ("" if value == 1 else "s")

    def _update_streak(self):
        if self._last_pull is None:
            self.streak_label.setText(
                "Last pull time unavailable." if self._has_pulls
                else "No pulls recorded yet.")
            return
        elapsed = int((datetime.datetime.now() - self._last_pull).total_seconds())
        elapsed = max(0, elapsed)
        days, rest = divmod(elapsed, 86400)
        hours, rest = divmod(rest, 3600)
        minutes, seconds = divmod(rest, 60)
        self.streak_label.setText(
            "I have not pulled my hair in "
            f"{self._plural(days, 'day')}, {self._plural(hours, 'hour')}, "
            f"{self._plural(minutes, 'minute')}, "
            f"{self._plural(seconds, 'second')}")

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
        self._set_status("Defaults loaded. Click Save to apply.", error=False)

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
            "PULLING: hands away from your hair" if warn else "safe")
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
