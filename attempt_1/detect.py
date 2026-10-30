#!/usr/bin/env python3
"""Tricho detector: hair segmentation + fingertip proximity.

Every ~100 ms:
  1. Segment hair with MediaPipe Hair Segmenter (class 1 = hair).
  2. Locate fingertips with MediaPipe Hand Landmarker.
  3. For each fingertip, compute the distance to the nearest hair pixel.
  4. If any fingertip is within DIST_THRESHOLD pixels of hair for
     DWELL_FRAMES consecutive frames, warn.

Both models are fed the same frame, and the hair mask is returned at the
input frame's resolution, so the mask and the hand landmarks already share
the same (frame-pixel) coordinate system.
"""

import argparse
import time

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Fingertip landmark indices: thumb, index, middle, ring, pinky tips.
FINGERTIPS = (4, 8, 12, 16, 20)

HAIR_SEGMENTER = "models/hair_segmenter.tflite"
HAND_LANDMARKER = "models/hand_landmarker.task"


def build_segmenter():
    options = vision.ImageSegmenterOptions(
        base_options=python.BaseOptions(model_asset_path=HAIR_SEGMENTER),
        running_mode=vision.RunningMode.VIDEO,
        output_category_mask=True,
    )
    return vision.ImageSegmenter.create_from_options(options)


def build_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_LANDMARKER),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(options)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=12.0,
                    help="max distance (px) from fingertip to hair to warn")
    ap.add_argument("--update-ms", type=int, default=100)
    ap.add_argument("--dwell-frames", type=int, default=3,
                    help="consecutive 'pulling' frames required before warning")
    ap.add_argument("--no-display", action="store_true")
    args = ap.parse_args()

    segmenter = build_segmenter()
    landmarker = build_hand_landmarker()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera}")

    next_t = time.monotonic()
    show = not args.no_display
    streak = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            H, W = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int(time.monotonic() * 1000)

            # --- hair mask (frame resolution, 1 = hair) ------------------
            seg = segmenter.segment_for_video(mp_img, ts)
            hair = seg.category_mask.numpy_view().copy()  # (H, W, 1) uint8
            hair_bin = (hair[:, :, 0] > 0).astype(np.uint8)  # (H, W)

            # distance from every pixel to the nearest hair pixel
            if hair_bin.any():
                dist = cv2.distanceTransform(1 - hair_bin, cv2.DIST_L2, 5)
            else:
                dist = None  # no hair detected -> nothing can be "in hair"

            # --- fingertips ----------------------------------------------
            hands = landmarker.detect_for_video(mp_img, ts)
            pulling = False
            tips = []  # (x, y) in frame pixels, for debug overlay

            if dist is not None:
                for hand in hands.hand_landmarks:
                    for idx in FINGERTIPS:
                        lm = hand[idx]
                        x = min(max(int(round(lm.x * W)), 0), W - 1)
                        y = min(max(int(round(lm.y * H)), 0), H - 1)
                        if float(dist[y, x]) < args.threshold:
                            pulling = True
                        tips.append((x, y))

            # --- dwell-time smoothing ------------------------------------
            streak = streak + 1 if pulling else 0
            warn = streak >= args.dwell_frames

            print("STOP PULLING YOUR HAIR" if warn else "safe")

            # --- debug window -------------------------------------------
            if show:
                try:
                    overlay = frame.copy()
                    mask_rgb = cv2.cvtColor(
                        (hair_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    overlay = cv2.addWeighted(overlay, 0.6, mask_rgb, 0.4, 0)
                    for (x, y) in tips:
                        cv2.circle(overlay, (x, y), 6, (0, 0, 255), -1)
                    label = "PULLING" if warn else "safe"
                    color = (0, 0, 255) if warn else (0, 255, 0)
                    cv2.putText(overlay, label, (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                    cv2.imshow("tricho", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    show = False  # no display available; keep running

            # --- throttle to update-ms -----------------------------------
            next_t += args.update_ms / 1000.0
            sleep = next_t - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.monotonic()

    finally:
        cap.release()
        segmenter.close()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
