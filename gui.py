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
from PySide6.QtCore import QDate, QThread, QTimer, Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from activity import (all_pulls, counts_per_day, load_activity,
                      load_highscores, years_in)
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
from graph import VIEWS, PullGraph, build_graph_data
from heatmap import HeatmapWidget
from sound import SoundPlayer

PREVIEW_MAX_WIDTH = 800
DWELL_FRAMES = 3
FEEDBACK_URL = ("https://docs.google.com/forms/d/e/"
                "1FAIpQLSfDUIuLhHDg1paNdIf2CJqjNX1tQKJW5cqKV-TVLTISETO0Cw/"
                "viewform?usp=publish-editor")


def format_duration(seconds):
    seconds = max(0, int(seconds))
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours or days:
        parts.append(f"{hours}h")
    if minutes or hours or days:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def format_range(start_iso, end_iso):
    try:
        start = datetime.datetime.fromisoformat(start_iso)
        end = datetime.datetime.fromisoformat(end_iso)
    except (TypeError, ValueError):
        return f"{start_iso} - {end_iso}"

    if start.date() == end.date():
        return (f"{start.strftime('%d %b %Y, %H:%M')} - "
                f"{end.strftime('%H:%M')}")
    return (f"{start.strftime('%d %b %Y, %H:%M')} - "
            f"{end.strftime('%d %b %Y, %H:%M')}")


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
            detector = Detector(self._snapshot())
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
                detector.reconfigure(cfg)
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
        self._session_start = datetime.datetime.now()
        self._pulls = []

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
        self.worker.pull_recorded.connect(self._on_pull_recorded)
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
        page.setObjectName("settingsPage")
        page.setStyleSheet("#settingsPage { background:#f5f6f8; }")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Settings")
        title.setStyleSheet("font-size:22px; font-weight:800; color:#111827;")
        layout.addWidget(title)
        subtitle = QLabel("Tune detection, the calendar, and alerts")
        subtitle.setStyleSheet("font-size:12px; color:#6b7280;")
        layout.addWidget(subtitle)

        # --- Detection -----------------------------------------------------
        detection_card, detection_layout = self._card("Detection", "🎯")
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.points_edit = QLineEdit()
        self.points_edit.setPlaceholderText("4,8,12,16,20,3,7,11,15,19")
        form.addRow("Contact points (0–20):", self.points_edit)
        hint = QLabel("Fingertips: 4, 8, 12, 16, 20 · DIP joints: 3, 7, 11, 15, 19")
        hint.setStyleSheet("color:#6b7280;")
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
        self.fps_label.setStyleSheet("color:#6b7280;")
        form.addRow("Frame interval:", self.interval_spin)
        form.addRow("", self.fps_label)

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0, 500.0)
        self.radius_spin.setSingleStep(1.0)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setSuffix(" px")
        form.addRow("Contact radius:", self.radius_spin)

        detection_layout.addLayout(form)
        layout.addWidget(detection_card)

        # --- Calendar ------------------------------------------------------
        calendar_card, calendar_layout = self._card("Calendar", "🎨")
        calendar_form = QFormLayout()
        calendar_form.setContentsMargins(0, 0, 0, 0)
        calendar_form.setSpacing(10)
        calendar_form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.color_step_spin = QSpinBox()
        self.color_step_spin.setRange(1, 255)
        self.color_step_spin.setSingleStep(1)
        calendar_form.addRow("Heatmap step:", self.color_step_spin)
        color_hint = QLabel(
            "Green removed per pull. At 10, a day with 10 pulls is (0, 155, 0) "
            "and 26+ pulls turns the square black. Higher values fade faster.")
        color_hint.setWordWrap(True)
        color_hint.setStyleSheet("color:#6b7280;")
        calendar_form.addRow("", color_hint)

        calendar_layout.addLayout(calendar_form)
        layout.addWidget(calendar_card)

        # --- Alerts --------------------------------------------------------
        alerts_card, alerts_layout = self._card("Alerts", "🔔")
        alerts_form = QFormLayout()
        alerts_form.setContentsMargins(0, 0, 0, 0)
        alerts_form.setSpacing(10)
        alerts_form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        self.title_edit = QLineEdit()
        alerts_form.addRow("Notification title:", self.title_edit)

        self.message_edit = QLineEdit()
        alerts_form.addRow("Notification message:", self.message_edit)

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
        alerts_form.addRow("Sound effect:", sound_row)

        alerts_layout.addLayout(alerts_form)
        layout.addWidget(alerts_card)

        # --- Advanced ------------------------------------------------------
        advanced_card, advanced_layout = self._card("Advanced", "⚙️")
        advanced_note = QLabel(
            "Model-level tuning. Saving rebuilds the hand and face detectors "
            "when their confidence values change.")
        advanced_note.setWordWrap(True)
        advanced_note.setStyleSheet("color:#6b7280;")
        advanced_layout.addWidget(advanced_note)

        advanced_form = QFormLayout()
        advanced_form.setContentsMargins(0, 0, 0, 0)
        advanced_form.setSpacing(8)
        advanced_form.setLabelAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def add_advanced(label, widget, description):
            advanced_form.addRow(label, widget)
            note = QLabel(description)
            note.setWordWrap(True)
            note.setStyleSheet("color:#6b7280;")
            advanced_form.addRow("", note)

        self.hand_confirm_spin = QSpinBox()
        self.hand_confirm_spin.setRange(1, 30)
        self.hand_confirm_spin.setSingleStep(1)
        add_advanced(
            "Hand confirm frames:", self.hand_confirm_spin,
            "Consecutive frames a hand must be seen before its landmarks are "
            "used. Higher: fewer false hands, slower to react. Lower: faster, "
            "but more false triggers.")

        self.hair_threshold_spin = QDoubleSpinBox()
        self.hair_threshold_spin.setRange(0.01, 1.0)
        self.hair_threshold_spin.setSingleStep(0.01)
        self.hair_threshold_spin.setDecimals(2)
        add_advanced(
            "Hair threshold:", self.hair_threshold_spin,
            "Minimum hair probability per pixel. Higher: stricter, fewer "
            "false hair regions but a smaller mask. Lower: more hair, but "
            "more false positives.")

        self.face_conf_spin = QDoubleSpinBox()
        self.face_conf_spin.setRange(0.0, 1.0)
        self.face_conf_spin.setSingleStep(0.01)
        self.face_conf_spin.setDecimals(2)
        add_advanced(
            "Face confidence:", self.face_conf_spin,
            "Minimum score for the face gate. Higher: hair segmentation runs "
            "less often (may pause when you are partly out of frame). Lower: "
            "more permissive, but more spurious faces.")

        self.hand_detect_spin = QDoubleSpinBox()
        self.hand_detect_spin.setRange(0.0, 1.0)
        self.hand_detect_spin.setSingleStep(0.01)
        self.hand_detect_spin.setDecimals(2)
        add_advanced(
            "Hand detection confidence:", self.hand_detect_spin,
            "Confidence needed to initially detect a hand. Higher: fewer "
            "false hands, but may miss real ones. Lower: more sensitive, but "
            "more false positives.")

        self.hand_presence_spin = QDoubleSpinBox()
        self.hand_presence_spin.setRange(0.0, 1.0)
        self.hand_presence_spin.setSingleStep(0.01)
        self.hand_presence_spin.setDecimals(2)
        add_advanced(
            "Hand presence confidence:", self.hand_presence_spin,
            "Confidence needed to keep tracking a hand. Higher: drops weak "
            "tracks and re-detects. Lower: keeps tracking longer, but may "
            "hold onto a bad detection.")

        self.hand_tracking_spin = QDoubleSpinBox()
        self.hand_tracking_spin.setRange(0.0, 1.0)
        self.hand_tracking_spin.setSingleStep(0.01)
        self.hand_tracking_spin.setDecimals(2)
        add_advanced(
            "Hand tracking confidence:", self.hand_tracking_spin,
            "How closely a hand must match the previous frame to keep "
            "tracking. Higher: re-detects more often. Lower: tracks longer "
            "through movement.")

        advanced_layout.addLayout(advanced_form)
        layout.addWidget(advanced_card)

        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)

        # --- Sticky save / reset footer ------------------------------------
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)

        buttons = QHBoxLayout()
        self.save_button = QPushButton("Save")
        self.save_button.setStyleSheet(
            "QPushButton { background:#2e7d32; color:white; font-weight:600; "
            "padding:8px 24px; border:none; border-radius:8px; }"
            "QPushButton:hover { background:#276b2b; }")
        self.save_button.clicked.connect(self._save)
        self.reset_button = QPushButton("Reset to Defaults")
        self.reset_button.setStyleSheet(
            "QPushButton { background:#ffffff; color:#374151; font-weight:600; "
            "padding:8px 20px; border:1px solid #d1d5db; border-radius:8px; }"
            "QPushButton:hover { background:#f3f4f6; }")
        self.reset_button.clicked.connect(self._reset)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.reset_button)
        buttons.addStretch(1)

        footer = QWidget()
        footer.setObjectName("settingsFooter")
        footer.setStyleSheet(
            "#settingsFooter { background:#ffffff; "
            "border-top:1px solid #e4e7ec; }")
        footer_layout = QVBoxLayout(footer)
        footer_layout.setContentsMargins(16, 10, 16, 12)
        footer_layout.setSpacing(6)
        footer_layout.addWidget(self.status_label)
        footer_layout.addLayout(buttons)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)
        wrapper_layout.addWidget(scroll, 1)
        wrapper_layout.addWidget(footer)
        return wrapper

    @staticmethod
    def _card(title, icon=""):
        card = QFrame()
        card.setObjectName("card")
        card.setStyleSheet(
            "QFrame#card { background:#ffffff; border:1px solid #e4e7ec; "
            "border-radius:12px; }")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)
        heading = QLabel(f"{icon}  {title}")
        heading.setStyleSheet(
            "font-size:14px; font-weight:700; color:#1f2937;")
        layout.addWidget(heading)
        return card, layout

    def _build_activity_tab(self):
        page = QWidget()
        page.setObjectName("activityPage")
        page.setStyleSheet("#activityPage { background:#f5f6f8; }")
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        title = QLabel("Your activity")
        title.setStyleSheet("font-size:22px; font-weight:800; color:#111827;")
        layout.addWidget(title)
        subtitle = QLabel("Patterns, streaks, and records at a glance")
        subtitle.setStyleSheet("font-size:12px; color:#6b7280;")
        layout.addWidget(subtitle)

        calendar_card, calendar_layout = self._card("Pull calendar", "📅")
        calendar_header = QHBoxLayout()
        calendar_header.addWidget(QLabel("Year:"))
        self.year_combo = QComboBox()
        self.year_combo.currentTextChanged.connect(self._refresh_activity)
        calendar_header.addWidget(self.year_combo)
        calendar_header.addStretch(1)
        self.total_label = QLabel("")
        self.total_label.setStyleSheet(
            "font-size:13px; font-weight:600; color:#2e7d32;")
        calendar_header.addWidget(self.total_label)
        calendar_layout.addLayout(calendar_header)

        self.heatmap = HeatmapWidget()
        calendar_layout.addWidget(self.heatmap)

        self.contact_label = QLabel(
            f'<a href="{FEEDBACK_URL}" '
            'style="color:#2a6fb0; font-size:14px;">Contact us</a>')
        self.contact_label.setOpenExternalLinks(True)
        self.contact_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        calendar_layout.addWidget(self.contact_label)

        self.description_label = QLabel(
            "Each square is one day. A day with no pulling stays bright green; "
            "every pull removes 10 from the green channel, so darker squares "
            "mean more pulling that day.")
        self.description_label.setWordWrap(True)
        self.description_label.setStyleSheet("color:#6b7280;")
        calendar_layout.addWidget(self.description_label)
        layout.addWidget(calendar_card)

        streak_card, streak_layout = self._card("Current streak", "🔥")
        self.streak_label = QLabel("No pulls recorded yet.")
        self.streak_label.setWordWrap(True)
        self.streak_label.setStyleSheet(
            "font-size:20px; font-weight:700; color:#14532d;")
        streak_layout.addWidget(self.streak_label)
        layout.addWidget(streak_card)

        graph_card, graph_layout = self._card("Pulls over time", "📈")
        graph_controls = QHBoxLayout()
        graph_controls.addWidget(QLabel("View:"))
        self.graph_view = QComboBox()
        self.graph_view.addItems(list(VIEWS.keys()))
        self.graph_view.setCurrentText("Year · 1 month")
        self.graph_view.currentTextChanged.connect(self._refresh_graph)
        graph_controls.addWidget(self.graph_view)
        graph_controls.addWidget(QLabel("Date:"))
        self.graph_date = QDateEdit()
        self.graph_date.setCalendarPopup(True)
        self.graph_date.setDisplayFormat("yyyy-MM-dd")
        self.graph_date.setDate(QDate.currentDate())
        self.graph_date.dateChanged.connect(self._refresh_graph)
        graph_controls.addWidget(self.graph_date)
        graph_controls.addStretch(1)
        graph_layout.addLayout(graph_controls)
        self.graph = PullGraph()
        graph_layout.addWidget(self.graph)
        layout.addWidget(graph_card)

        leaderboard_card, leaderboard_layout = self._card(
            "Longest stretches without pulling", "🏆")
        self.highscores_container = QWidget()
        self.highscores_layout = QVBoxLayout(self.highscores_container)
        self.highscores_layout.setContentsMargins(0, 0, 0, 0)
        self.highscores_layout.setSpacing(0)
        leaderboard_layout.addWidget(self.highscores_container)
        layout.addWidget(leaderboard_card)

        footer = QLabel("All data stays on your device")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet("color:#9ca3af; font-size:11px; padding-top:4px;")
        layout.addWidget(footer)

        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setWidget(page)
        return scroll

    # ------------------------------------------------------------ form helpers
    def _populate(self, cfg):
        self.points_edit.setText(",".join(str(p) for p in cfg["contact_points"]))
        self.cooldown_spin.setValue(float(cfg["cooldown_seconds"]))
        self.interval_spin.setValue(int(round(cfg["frame_interval_ms"])))
        self.radius_spin.setValue(float(cfg["radius_px"]))
        self.color_step_spin.setValue(int(cfg["heatmap_green_step"]))
        self.hand_confirm_spin.setValue(int(cfg["hand_confirm_frames"]))
        self.hair_threshold_spin.setValue(float(cfg["hair_threshold"]))
        self.face_conf_spin.setValue(float(cfg["face_min_confidence"]))
        self.hand_detect_spin.setValue(float(cfg["hand_detection_confidence"]))
        self.hand_presence_spin.setValue(float(cfg["hand_presence_confidence"]))
        self.hand_tracking_spin.setValue(float(cfg["hand_tracking_confidence"]))
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
            "heatmap_green_step": self.color_step_spin.value(),
            "hand_confirm_frames": self.hand_confirm_spin.value(),
            "hair_threshold": self.hair_threshold_spin.value(),
            "face_min_confidence": self.face_conf_spin.value(),
            "hand_detection_confidence": self.hand_detect_spin.value(),
            "hand_presence_confidence": self.hand_presence_spin.value(),
            "hand_tracking_confidence": self.hand_tracking_spin.value(),
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

        self._pulls = all_pulls(data)
        self._update_streak()

        year = int(self.year_combo.currentText())
        counts = counts_per_day(data, year)
        step = int(self.cfg["heatmap_green_step"])
        self.heatmap.set_data(year, counts, step)
        self.description_label.setText(
            "Each square is one day. A day with no pulling stays bright green; "
            f"every pull removes {step} from the green channel, so darker "
            "squares mean more pulling that day.")
        self.total_label.setText(f"{sum(counts.values())} pulls in {year}")

        self._refresh_graph()
        self._refresh_highscores(data)

    def _on_tab_changed(self, index):
        if self.tabs.tabText(index) == "Activity":
            self._refresh_activity()

    def _refresh_graph(self, *_args):
        if not hasattr(self, "graph"):
            return
        label = self.graph_view.currentText()
        period, step = VIEWS[label]
        ref = self.graph_date.date().toPython()
        counts, labels, ticks, title, descriptions = build_graph_data(
            period, step, ref, self._pulls)
        self.graph.set_data(counts, labels, ticks, title, descriptions)

    def _refresh_highscores(self, data):
        while self.highscores_layout.count():
            item = self.highscores_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        scores = load_highscores(data)
        if not scores:
            empty = QLabel(
                "No durations yet. A stretch starts when you launch the app "
                "and ends at your first pull.")
            empty.setWordWrap(True)
            empty.setStyleSheet("color:#6b7280; padding:6px 2px;")
            self.highscores_layout.addWidget(empty)
            return

        medals = ("🥇", "🥈", "🥉")
        colors = ("#D4AF37", "#A8A9AD", "#CD7F32")
        sizes = (20, 18, 16)

        for rank, score in enumerate(scores):
            duration = format_duration(score["seconds"])
            span = format_range(score["start"], score["end"])

            if rank < 3:
                text = (f"{medals[rank]} {duration}"
                        f"<span style='color:#777; font-weight:normal; "
                        f"font-size:11px;'>&nbsp;&nbsp;{span}</span>")
                label = QLabel(text)
                label.setStyleSheet(
                    f"font-size:{sizes[rank]}px; font-weight:bold; "
                    f"color:{colors[rank]}; padding:6px 2px;")
            else:
                text = (f"{rank + 1}. {duration}"
                        f"<span style='color:#777; font-size:11px;'>"
                        f"&nbsp;&nbsp;{span}</span>")
                label = QLabel(text)
                label.setStyleSheet("font-size:13px; padding:6px 2px;")

            label.setTextFormat(Qt.TextFormat.RichText)
            self.highscores_layout.addWidget(label)

            if rank < len(scores) - 1:
                line = QFrame()
                line.setFrameShape(QFrame.Shape.HLine)
                line.setFixedHeight(1)
                line.setStyleSheet("background:#eceff3; border:none;")
                self.highscores_layout.addWidget(line)

    @staticmethod
    def _plural(value, unit):
        return f"{value} {unit}" + ("" if value == 1 else "s")

    def _update_streak(self):
        start = self._last_pull or self._session_start
        elapsed = int((datetime.datetime.now() - start).total_seconds())
        elapsed = max(0, elapsed)
        days, rest = divmod(elapsed, 86400)
        hours, rest = divmod(rest, 3600)
        minutes, seconds = divmod(rest, 60)
        self.streak_label.setText(
            "I have not pulled my hair in "
            f"{self._plural(days, 'day')}, {self._plural(hours, 'hour')}, "
            f"{self._plural(minutes, 'minute')}, "
            f"{self._plural(seconds, 'second')}")

    def _on_pull_recorded(self):
        self._last_pull = datetime.datetime.now()
        self._refresh_activity()

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
        self._refresh_activity()
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
