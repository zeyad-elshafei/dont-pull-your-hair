#!/usr/bin/env python3
"""Shared detection core for the Tricho detector.

No GUI dependencies live here; both the CLI (detect.py) and the GUI (gui.py)
build on the Detector class and the config helpers below.
"""

import json
import os
import sys
import time
from dataclasses import dataclass

import cv2
import numpy as np
import mediapipe as mp
from desktop_notifier import DesktopNotifierSync
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

HAIR_SEGMENTER = os.path.join(BASE_DIR, "models", "hair_segmenter.tflite")
HAND_LANDMARKER = os.path.join(BASE_DIR, "models", "hand_landmarker.task")

# Built-in defaults. Any key present in config.json overrides its value here.
DEFAULTS = {
    "contact_points": [4, 8, 12, 16, 20, 3, 7, 11, 15, 19],
    "cooldown_seconds": 5.0,
    "frame_interval_ms": 100,
    "radius_px": 12.0,
    "notification_title": "Stop pulling your hair!",
    "notification_message": "Take a breath and move your hand away.",
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

    for key in ("notification_title", "notification_message"):
        if not isinstance(out.get(key), str):
            raise ConfigError(f"{key} must be a string")

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


class Detector:
    """Holds the MediaPipe models and the temporal/notification state."""

    def __init__(self):
        self.segmenter = vision.ImageSegmenter.create_from_options(
            vision.ImageSegmenterOptions(
                base_options=python.BaseOptions(
                    model_asset_path=HAIR_SEGMENTER),
                running_mode=vision.RunningMode.VIDEO,
                output_category_mask=True,
            ))
        self.landmarker = vision.HandLandmarker.create_from_options(
            vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(
                    model_asset_path=HAND_LANDMARKER),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
            ))
        try:
            self.notifier = DesktopNotifierSync(app_name="Tricho")
        except Exception as e:
            self.notifier = None
            print(f"[notify] disabled: {e}", file=sys.stderr)

        self._streak = 0
        self._last_notify = float("-inf")

    def process(self, frame, timestamp_ms, cfg, dwell_frames=3):
        """Run one frame. cfg supplies contact_points/radius/cooldown/text."""
        H, W = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # --- hair mask (frame resolution, 1 = hair) ----------------------
        seg = self.segmenter.segment_for_video(mp_img, timestamp_ms)
        hair = seg.category_mask.numpy_view().copy()  # (H, W, 1) uint8
        hair_bin = (hair[:, :, 0] > 0).astype(np.uint8)

        if hair_bin.any():
            dist = cv2.distanceTransform(1 - hair_bin, cv2.DIST_L2, 5)
        else:
            dist = None  # no hair detected -> nothing can be "in hair"

        # --- contact points ----------------------------------------------
        hands = self.landmarker.detect_for_video(mp_img, timestamp_ms)
        pulling = False
        points = []

        if dist is not None:
            for hand in hands.hand_landmarks:
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

        if warn and self.notifier is not None:
            now = time.monotonic()
            if now - self._last_notify >= cfg["cooldown_seconds"]:
                self._last_notify = now
                try:
                    self.notifier.send(
                        title=cfg["notification_title"],
                        message=cfg["notification_message"],
                    )
                except Exception as e:
                    print(f"[notify] send failed: {e}", file=sys.stderr)

        return Result(warn=warn, pulling=pulling, points=points,
                      hair_mask=hair_bin)

    def close(self):
        self.segmenter.close()
        self.landmarker.close()


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
