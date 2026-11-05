#!/usr/bin/env python3
"""Shared detection core for the Tricho detector.

No GUI dependencies live here; both the CLI (detect.py) and the GUI (gui.py)
build on the Detector class and the config helpers below.
"""

import datetime
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from desktop_notifier import DesktopNotifierSync, Icon
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from activity import load_activity, record_duration, record_pull, save_activity
from sound import SoundPlayer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

HAIR_SEGMENTER = os.path.join(BASE_DIR, "models", "hair_segmenter.tflite")
HAND_LANDMARKER = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
FACE_DETECTOR = os.path.join(BASE_DIR, "models",
                             "blaze_face_short_range.tflite")
NOTIFICATION_ICON = Path(BASE_DIR) / "dont_pull_square.png"

# Max normalized distance between palm centers in consecutive frames for two
# detections to count as the same hand.
HAND_MATCH_DISTANCE = 0.2

# Built-in defaults. Any key present in config.json overrides its value here.
DEFAULTS = {
    "contact_points": [4, 8, 12, 16, 20, 3, 7, 11, 15, 19],
    "cooldown_seconds": 5.0,
    "frame_interval_ms": 100,
    "radius_px": 12.0,
    "heatmap_green_step": 10,
    "hand_confirm_frames": 3,
    "hair_threshold": 0.7,
    "face_min_confidence": 0.55,
    "hand_detection_confidence": 0.8,
    "hand_presence_confidence": 0.8,
    "hand_tracking_confidence": 0.8,
    "notification_title": "Stop pulling your hair!",
    "notification_message": "Take a breath and move your hand away.",
    "sound_path": "",
    "warn_on_face_lost": False,
    "face_warn_cooldown_seconds": 1.5,
    "dark_mode": False,
}

CONFIG_KEYS = tuple(DEFAULTS)


class ConfigError(Exception):
    """Raised when a config file or GUI value is invalid."""


def validate_config(cfg):
    """Validate and normalise a config dict. Returns a new dict."""
    out = dict(cfg)

    pts = out.get("contact_points")
    if (not isinstance(pts, list) or not pts
            or not all(isinstance(p, int) and not isinstance(p, bool)
                       and 0 <= p <= 20 for p in pts)):
        raise ConfigError(
            "contact_points must be a non-empty list of integers in the "
            "range 0..20")
    out["contact_points"] = [int(p) for p in pts]

    def number(key, minimum, exclusive=False):
        v = out.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{key} must be a number")
        if not (v > minimum if exclusive else v >= minimum):
            op = ">" if exclusive else ">="
            raise ConfigError(f"{key} must be {op} {minimum}")
        return float(v)

    out["cooldown_seconds"] = number("cooldown_seconds", 0)
    out["frame_interval_ms"] = number("frame_interval_ms", 0, exclusive=True)
    out["radius_px"] = number("radius_px", 0)
    out["face_warn_cooldown_seconds"] = number(
        "face_warn_cooldown_seconds", 0)

    step = out.get("heatmap_green_step")
    if (isinstance(step, bool) or not isinstance(step, int)
            or not 1 <= step <= 255):
        raise ConfigError(
            "heatmap_green_step must be an integer in the range 1..255")
    out["heatmap_green_step"] = int(step)

    def unit(key):
        v = out.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ConfigError(f"{key} must be a number")
        if not 0.0 <= v <= 1.0:
            raise ConfigError(f"{key} must be between 0 and 1")
        return float(v)

    out["hair_threshold"] = unit("hair_threshold")
    if out["hair_threshold"] <= 0.0:
        raise ConfigError("hair_threshold must be greater than 0")
    out["face_min_confidence"] = unit("face_min_confidence")
    out["hand_detection_confidence"] = unit("hand_detection_confidence")
    out["hand_presence_confidence"] = unit("hand_presence_confidence")
    out["hand_tracking_confidence"] = unit("hand_tracking_confidence")

    frames = out.get("hand_confirm_frames")
    if (isinstance(frames, bool) or not isinstance(frames, int)
            or not 1 <= frames <= 30):
        raise ConfigError(
            "hand_confirm_frames must be an integer in the range 1..30")
    out["hand_confirm_frames"] = int(frames)

    for key in ("notification_title", "notification_message", "sound_path"):
        if not isinstance(out.get(key), str):
            raise ConfigError(f"{key} must be a string")

    if not isinstance(out.get("warn_on_face_lost"), bool):
        raise ConfigError("warn_on_face_lost must be a boolean")

    if not isinstance(out.get("dark_mode"), bool):
        raise ConfigError("dark_mode must be a boolean")

    return out


def load_config(path=CONFIG_PATH):
    """Load and validate config.json. Raises ConfigError on any problem."""
    cfg = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}

    if not os.path.exists(path):
        raise ConfigError(f"{path} not found")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ConfigError(f"could not read {path}: {e}")

    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")

    unknown = sorted(set(data) - set(DEFAULTS))
    if unknown:
        print(f"[config] ignoring unknown keys: {', '.join(unknown)}",
              file=sys.stderr)

    cfg.update({k: v for k, v in data.items() if k in DEFAULTS})
    return validate_config(cfg)


def save_config(path, cfg):
    """Validate and write cfg to path as pretty-printed JSON."""
    cfg = validate_config(cfg)
    data = {k: cfg[k] for k in CONFIG_KEYS}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
    return cfg


@dataclass
class Result:
    warn: bool
    pulling: bool
    points: list
    hair_mask: np.ndarray | None
    episode_started: bool = False


class Detector:
    """Holds the MediaPipe models and the temporal/notification state."""

    @staticmethod
    def _build_segmenter():
        return vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=python.BaseOptions(
                    model_asset_path=HAIR_SEGMENTER),
                running_mode=vision.RunningMode.VIDEO,
                output_category_mask=False,
                output_confidence_masks=True,
            ))

    @staticmethod
    def _build_hand_landmarker(cfg):
        return vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=HAND_LANDMARKER),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=cfg["hand_detection_confidence"],
                min_hand_presence_confidence=cfg["hand_presence_confidence"],
                min_tracking_confidence=cfg["hand_tracking_confidence"],
            ))

    @staticmethod
    def _build_face_detector(cfg):
        return vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=python.BaseOptions(
                    model_asset_path=FACE_DETECTOR),
                running_mode=vision.RunningMode.VIDEO,
                min_detection_confidence=cfg["face_min_confidence"],
            ))

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.segmenter = self._build_segmenter()
        self.landmarker = self._build_hand_landmarker(cfg)
        self.face_detector = self._build_face_detector(cfg)
        self._icon = Icon(path=NOTIFICATION_ICON)
        try:
            self.notifier = DesktopNotifierSync(
                app_name="Tricho",
                app_icon=self._icon,
            )
        except Exception as e:
            self.notifier = None
            print(f"[notify] disabled: {e}", file=sys.stderr)

        self._sound = SoundPlayer()
        self._hand_tracks = []
        self._streak = 0
        self._last_alert = float("-inf")
        self._last_warn = False
        self._face_present = False
        self._face_lost_since = None
        self._face_warned = False
        self._session_start = datetime.datetime.now()
        self._last_episode = None
        self._closed = False

    def reconfigure(self, cfg):
        """Rebuild models whose creation-time settings changed."""
        cfg = dict(cfg)
        hand_keys = ("hand_detection_confidence",
                     "hand_presence_confidence",
                     "hand_tracking_confidence")
        if any(cfg[k] != self.cfg.get(k) for k in hand_keys):
            self.landmarker.close()
            self.landmarker = self._build_hand_landmarker(cfg)
            self._hand_tracks = []
        if cfg["face_min_confidence"] != self.cfg.get("face_min_confidence"):
            self.face_detector.close()
            self.face_detector = self._build_face_detector(cfg)
        self.cfg = cfg

    def process(self, frame, timestamp_ms, cfg, dwell_frames=3):
        """Run one frame. cfg supplies contact_points/radius/cooldown/text."""
        H, W = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # --- face gate + hair mask ----------------------------------------
        # Only look for hair when a face is present. Without this, the
        # segmenter can hallucinate hair on background objects.
        face_result = self.face_detector.detect_for_video(mp_img, timestamp_ms)
        face_present = bool(face_result.detections)
        now_mono = time.monotonic()

        if face_present:
            self._face_lost_since = None
            self._face_warned = False
        else:
            if self._face_present:
                self._face_lost_since = now_mono
            if (cfg.get("warn_on_face_lost", False)
                    and not self._face_warned
                    and self._face_lost_since is not None
                    and now_mono - self._face_lost_since
                    >= cfg["face_warn_cooldown_seconds"]):
                self._face_warned = True
                if self.notifier is not None:
                    try:
                        self.notifier.send(
                            title="Face not detected",
                            message=("Your face is no longer apparent. Adjust "
                                     "your position to be visible to the "
                                     "camera."),
                            icon=self._icon,
                        )
                    except Exception as e:
                        print(f"[notify] send failed: {e}", file=sys.stderr)
        self._face_present = face_present

        if face_present:
            seg = self.segmenter.segment_for_video(mp_img, timestamp_ms)
            # confidence_masks[1] is the hair class; threshold it ourselves.
            hair_conf = seg.confidence_masks[1].numpy_view()  # (H, W, 1)
            hair_bin = (hair_conf[:, :, 0] >= cfg["hair_threshold"]).astype(np.uint8)
            if hair_bin.any():
                dist = cv2.distanceTransform(1 - hair_bin, cv2.DIST_L2, 5)
            else:
                dist = None
        else:
            hair_bin = None
            dist = None  # no face -> no hair to test against

        # --- contact points ----------------------------------------------
        hand_result = self.landmarker.detect_for_video(mp_img, timestamp_ms)
        hands = self._confirmed_hands(hand_result.hand_landmarks,
                                      cfg["hand_confirm_frames"])
        pulling = False
        points = []

        if dist is not None:
            for hand in hands:
                for idx in cfg["contact_points"]:
                    lm = hand[idx]
                    x = min(max(int(round(lm.x * W)), 0), W - 1)
                    y = min(max(int(round(lm.y * H)), 0), H - 1)
                    if float(dist[y, x]) < cfg["radius_px"]:
                        pulling = True
                    points.append((x, y))

        # --- dwell-time smoothing + cooldown-limited notification --------
        self._streak = self._streak + 1 if pulling else 0
        warn = self._streak >= dwell_frames

        # Count the start of each pulling episode once (rising edge of warn).
        episode_started = warn and not self._last_warn
        self._last_warn = warn
        if episode_started:
            now = datetime.datetime.now()
            start = self._last_episode or self._session_start
            try:
                data = load_activity()
                record_pull(data, when=now)
                record_duration(data, start, now)
                save_activity(data)
            except OSError as e:
                print(f"[activity] save failed: {e}", file=sys.stderr)
            self._last_episode = now

        if warn:
            now = time.monotonic()
            if now - self._last_alert >= cfg["cooldown_seconds"]:
                self._last_alert = now
                if self.notifier is not None:
                    try:
                        self.notifier.send(
                            title=cfg["notification_title"],
                            message=cfg["notification_message"],
                            icon=self._icon,
                        )
                    except Exception as e:
                        print(f"[notify] send failed: {e}", file=sys.stderr)
                self._sound.play(cfg.get("sound_path", ""))

        return Result(warn=warn, pulling=pulling, points=points,
                      hair_mask=hair_bin, episode_started=episode_started)

    def _confirmed_hands(self, hands, confirm_frames):
        """Keep only hands that have appeared for `confirm_frames` frames."""
        centroids = []
        for hand in hands:
            cx = sum(lm.x for lm in hand) / len(hand)
            cy = sum(lm.y for lm in hand) / len(hand)
            centroids.append((cx, cy))

        confirmed = []
        new_tracks = []
        used = [False] * len(self._hand_tracks)

        for hand, (cx, cy) in zip(hands, centroids):
            best_index = None
            best_distance = HAND_MATCH_DISTANCE
            for i, track in enumerate(self._hand_tracks):
                if used[i]:
                    continue
                distance = math.hypot(cx - track["cx"], cy - track["cy"])
                if distance < best_distance:
                    best_distance = distance
                    best_index = i

            if best_index is None:
                track = {"cx": cx, "cy": cy, "streak": 1}
            else:
                used[best_index] = True
                track = self._hand_tracks[best_index]
                track["streak"] += 1
                track["cx"], track["cy"] = cx, cy

            new_tracks.append(track)
            if track["streak"] >= confirm_frames:
                confirmed.append(hand)

        self._hand_tracks = new_tracks
        return confirmed

    def finalize_session(self):
        """Record the currently open no-pull stretch (app close)."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        now = datetime.datetime.now()
        start = self._last_episode or self._session_start
        try:
            data = load_activity()
            record_duration(data, start, now)
            save_activity(data)
        except OSError as e:
            print(f"[activity] save failed: {e}", file=sys.stderr)

    def close(self):
        self.finalize_session()
        self.segmenter.close()
        self.landmarker.close()
        self.face_detector.close()


def draw_overlay(frame, result):
    """Return a BGR copy of frame with the hair mask and contact points drawn."""
    out = frame.copy()
    if result.hair_mask is not None:
        mask = cv2.cvtColor((result.hair_mask * 255).astype(np.uint8),
                            cv2.COLOR_GRAY2BGR)
        out = cv2.addWeighted(out, 0.6, mask, 0.4, 0)
    for (x, y) in result.points:
        cv2.circle(out, (x, y), 6, (0, 0, 255), -1)
    label = "PULLING" if result.warn else "safe"
    color = (0, 0, 255) if result.warn else (0, 255, 0)
    cv2.putText(out, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    return out
