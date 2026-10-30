#!/usr/bin/env python3
"""Command-line Tricho detector.

    python detect.py [--config config.json] [--camera 0] [--no-display]

All tunables live in config.json; --threshold and --update-ms exist only as
one-off overrides for testing.
"""

import argparse
import sys
import time

import cv2

from detector import (
    CONFIG_PATH,
    DEFAULTS,
    ConfigError,
    Detector,
    draw_overlay,
    load_config,
    validate_config,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=CONFIG_PATH,
                    help="path to JSON config (default: config.json "
                         "next to this script)")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=None,
                    help="override radius_px from the config file")
    ap.add_argument("--update-ms", type=float, default=None,
                    help="override frame_interval_ms from the config file")
    ap.add_argument("--dwell-frames", type=int, default=3,
                    help="consecutive 'pulling' frames required before warning")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        print(f"[config] {e}; using built-in defaults", file=sys.stderr)
        cfg = validate_config(dict(DEFAULTS))

    if args.threshold is not None:
        cfg["radius_px"] = float(args.threshold)
    if args.update_ms is not None:
        cfg["frame_interval_ms"] = float(args.update_ms)

    print(f"[config] contact_points={tuple(cfg['contact_points'])} "
          f"cooldown={cfg['cooldown_seconds']}s "
          f"interval={cfg['frame_interval_ms']}ms "
          f"radius={cfg['radius_px']}px", file=sys.stderr)

    detector = Detector()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        detector.close()
        raise SystemExit(f"Could not open camera index {args.camera}")

    next_t = time.monotonic()
    show = not args.no_display

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            ts = int(time.monotonic() * 1000)
            result = detector.process(frame, ts, cfg, args.dwell_frames)

            print("STOP PULLING YOUR HAIR" if result.warn else "safe")

            if show:
                try:
                    cv2.imshow("tricho", draw_overlay(frame, result))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show = False  # no display available; keep running

            next_t += cfg["frame_interval_ms"] / 1000.0
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
