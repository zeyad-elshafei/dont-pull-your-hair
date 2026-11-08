<p align="center">
  <img src="dont_pull_square.png" alt="Don't Pull!" width="280">
</p>

<h1 align="center">Don't Pull!</h1>

**Tricho Detector** is a local, webcam-based awareness tool for
**trichotillomania** (hair-pulling disorder). It watches for the moment a hand
comes into contact with hair, and gives a gentle but hard-to-miss cue (a
desktop notification and an optional sound) so the behavior is interrupted
before it becomes automatic.

Everything runs on your machine. No video is recorded, uploaded, or sent
anywhere.

---

## Why

Hair pulling usually happens outside awareness: while reading, thinking, on a
call. The first step in breaking the loop is *noticing* it. This app acts as a
second pair of eyes that never gets bored, and keeps a private log of when it
happened so patterns become visible over time.

It is a **wellness tool, not a medical device**. See the disclaimer at the
bottom.

---

## How it works

Every frame interval (100 ms by default) the app runs three models on the same
camera frame:

```
camera frame
   ├─ MediaPipe Face Detector ──► no face? ──► skip hair segmentation
   ├─ MediaPipe Hair Segmenter (only when a face is present)
   │        └─► hair mask ──► distance transform
   └─ MediaPipe Hand Landmarker ─► 21 landmarks ─► contact points

   contact point distance to hair  <  radius_px ?
                    │
                    ├─ no  ──► safe
                    └─ yes ──► dwell (N consecutive frames) ──► episode
                                                                (log + notify + sound)
```

**1. Face gate and hair segmentation.** A MediaPipe Face Detector runs first.
If no face is present, hair segmentation is skipped entirely, so background
objects cannot be mistaken for hair. If the optional face-loss warning is
enabled, the face must stay missing for the configured cooldown (1.5 s by
default) before a notification is sent, which ignores brief flicker. When a
face is present, the Hair Segmenter produces per-pixel hair/background
probabilities, and we keep only pixels whose hair probability is at least the
configured `hair_threshold` (0.7 by default).

**2. Distance transform.** For every pixel we compute the distance to the
nearest hair pixel (`cv2.distanceTransform`). This is the key trick; see the
next section.

**3. Hand landmarks.** MediaPipe's Hand Landmarker returns 21 landmarks per
hand in normalized coordinates, which we map to frame pixels. A hand must be
seen for several consecutive frames before its landmarks are used, which
suppresses transient false positives (for example, facial features briefly
misread as a hand).

**4. Contact test.** For each *contact point* (fingertips and the knuckle just
before each tip), we look up the distance to the nearest hair pixel. If that
distance is below `radius_px`, the point is considered to be touching hair.

**5. Dwell time.** A single frame can flicker, so the app requires several
consecutive "touching" frames (3 by default) before it declares an episode.
This kills most false positives from landmark jitter.

**6. Alert + log.** When an episode starts, the app records it, shows a desktop
notification, and plays your chosen sound. Alerts are rate-limited by
`cooldown_seconds` so continuous pulling does not spam you.

### Why distance, not "is the fingertip inside the hair mask?"

The obvious approach, checking whether the fingertip falls inside the hair
region, fails exactly when it matters. During a pull, **the hand occludes the
hair**, so the segmenter sees a hand-shaped hole in the mask. A fingertip at the
center of that hole would read as "not hair" precisely while the person is
pulling.

Measuring the distance to the *nearest* hair pixel sidesteps this: the hole is
bounded by hair, so a fingertip inside it is still only about half a finger
width from real hair. Occlusion stops being a problem by construction.

---

## Contact points

The Hand Landmarker numbers a hand's 21 keypoints `0`–`20` (`0` is the wrist):

<img src="handpose-demo-keypoints.jpg" alt="MediaPipe hand landmark numbering, 0 to 20" width="360">

The app checks **ten** of them by default: the five fingertips plus the DIP
joint just below each tip.

| Finger | Fingertip (TIP) | DIP joint |
|---|---|---|
| Thumb    | 4  | 3  |
| Index    | 8  | 7  |
| Middle   | 12 | 11 |
| Ring     | 16 | 15 |
| Pinky    | 20 | 19 |

**Why the DIP joints too?** When the fingers curl against the head, the tips
tuck underneath the hand and can end up far from the hair even though the
knuckles are physically pressed against it. The DIP joints sit on the outside
of the curl, so they catch that case. Without them, a curled-finger pull can be
missed.

You can change the set in `config.json`: `contact_points` accepts any subset of
`0`–`20`. For example, a much stricter (fewer false positives) configuration:

```json
"contact_points": [4, 8, 12, 16, 20]
```

Or a very permissive one that also includes the PIP and MCP knuckles:

```json
"contact_points": [4, 8, 12, 16, 20, 3, 7, 11, 15, 19, 6, 10, 14, 18, 5, 9, 13, 17]
```

More points means higher recall and more false positives; it is a single knob.

---

## Features

- **Live camera preview** with the hair mask and every contact point overlaid,
  so you can see exactly what the detector sees.
- **Desktop notification + optional sound** when an episode starts.
- **Optional face-loss warning**: a notification when the face detector loses
  you, so you know to move back into view. Off by default, with a 1.5 s
  cooldown to ignore brief flicker.
- **Calibrate mode**: a Camera-tab toggle that shows live PULLING feedback
  without sending pull notifications, recording pulls, or resetting the streak
  timer. A reminder notification fires after two minutes so it is not left on.
- **Radius visualization**: red circles around each contact point show the
  exact reach of `radius_px`; a hair pixel inside one triggers the pull test.
- **Configurable** contact points, radius, frame rate, cooldown, notification
  text, and sound file, editable in the GUI or in `config.json`.
- **Activity calendar**: a GitHub-style yearly heatmap of pulls per day; hover
  a square to see that day's exact count.
- **Activity graph**: pulls per minute, hour, day, week, or month over a
  selectable window; hover a bar to see the exact bucket and count.
- **Streak timer**: "I have not pulled my hair in X days, Y hours, …",
  reset when the app starts and after every pull.
- **Highscore leaderboard**: the ten longest stretches without pulling, with
  gold, silver, and bronze medals for the top three.
- **100% local**: no account, no server, no cloud.

---

## Install

Requires **Python 3.10+**.

```bash
# Linux / macOS
bash install.sh

# Windows
install.bat
```

Or directly:

```bash
python install.py
```

The installer creates a `.venv`, installs the pinned dependencies, downloads the
three MediaPipe models into `models/`, and creates `config.json` and
`activity.json` on first run.
It is safe to re-run: existing environments, models, and settings are reused.

---

## Run

**Desktop app (recommended):**

```bash
# Linux / macOS
.venv/bin/python gui.py

# Windows
.venv\Scripts\python gui.py
```

The app opens with three tabs:

| Tab | What it does |
|---|---|
| **Camera** | Live preview; the status bar turns red while pulling is detected. Toggle **Calibrate mode** to test settings without recording anything |
| **Settings** | Edit all tunables and appearance, then **Save**. **Reset to Defaults** restores the form |
| **Activity** | Yearly heatmap, pulls-over-time graph, streak timer, top-10 leaderboard, and the feedback link |

Saving settings hot-applies them; no restart needed.

**Calibrate mode** is a runtime-only toggle: it is not saved to
`config.json`, and it resets to off when the app restarts. While it is on, the
Camera preview still shows the red PULLING state, but detections do not send
pull notifications, write to `activity.json`, or reset the current no-pull
timer.

> Desktop notifications are suppressed by your OS's **Do Not Disturb / Focus**
> mode. The in-app status bar (and the sound, if configured) still work.

**Command line:**

```bash
.venv/bin/python detect.py                 # console + OpenCV preview window
.venv/bin/python detect.py --no-display    # console only
```

---

## Screenshots

Dark mode is shown below; click an image to open it full size. Switch themes
under **Settings → Appearance**.

### Settings

<p align="center">
  <a href="ArcoLinux_2026-11-06_02-12-15.png">
    <img src="ArcoLinux_2026-11-06_02-12-15.png" alt="Settings: detection, calendar, and appearance" width="100%">
  </a>
</p>

<p align="center">
  <a href="ArcoLinux_2026-11-06_02-12-43.png">
    <img src="ArcoLinux_2026-11-06_02-12-43.png" alt="Settings: alerts, sound, and advanced options" width="100%">
  </a>
</p>

<p align="center">
  <a href="ArcoLinux_2026-11-06_02-12-57.png">
    <img src="ArcoLinux_2026-11-06_02-12-57.png" alt="Settings: advanced model tuning" width="100%">
  </a>
</p>

### Activity

<p align="center">
  <a href="ArcoLinux_2026-11-06_02-13-19.png">
    <img src="ArcoLinux_2026-11-06_02-13-19.png" alt="Activity: pull calendar, streak, graph, and leaderboard" width="100%">
  </a>
</p>

<p align="center">
  <a href="ArcoLinux_2026-11-06_02-13-34.png">
    <img src="ArcoLinux_2026-11-06_02-13-34.png" alt="Activity: graph, full leaderboard, and footer" width="100%">
  </a>
</p>

---

## Configuration

Settings live in `config.json` next to the scripts. The GUI writes this file;
you can also edit it by hand.

| Key | Default | Meaning |
|---|---|---|
| `contact_points` | `[4, 8, 12, 16, 20, 3, 7, 11, 15, 19]` | Landmark indices to test (0–20) |
| `cooldown_seconds` | `5.0` | Minimum seconds between alerts |
| `frame_interval_ms` | `100` | Milliseconds between processed frames (100 = 10 FPS) |
| `radius_px` | `12.0` | Max distance from a contact point to hair, in frame pixels |
| `heatmap_green_step` | `10` | Green removed from a day's square per pull (1–255) |
| `hand_confirm_frames` | `3` | Consecutive frames a hand must be seen before use (1–30) |
| `hair_threshold` | `0.7` | Minimum hair probability per pixel (0–1) |
| `face_min_confidence` | `0.55` | Minimum score for the face gate (0–1) |
| `hand_detection_confidence` | `0.8` | Palm-detector confidence to find a hand (0–1) |
| `hand_presence_confidence` | `0.8` | Confidence to keep tracking a hand (0–1) |
| `hand_tracking_confidence` | `0.8` | IoU threshold for hand tracking (0–1) |
| `notification_title` | `"Stop pulling your hair!"` | Notification heading |
| `notification_message` | `"Take a breath and move your hand away."` | Notification body |
| `sound_path` | `""` | Path to a `.wav` / `.mp3` / `.ogg` / `.flac` file, played with each alert; empty = silent |
| `warn_on_face_lost` | `false` | Send one notification when the face detector loses you |
| `face_warn_cooldown_seconds` | `1.5` | Continuous face loss before that warning fires (0–60 s) |
| `dark_mode` | `false` | Use the dark color palette in the app |

Five notes:

- **`radius_px` is in frame pixels**, so it depends on the camera resolution.
  If you change resolution, re-tune it. 12 px at 720p is fairly tight; 15–30 is
  a reasonable range for most people.
- **`heatmap_green_step`** controls how fast the calendar fades: each pull
  removes that many points from the green channel (1–255).
- **Advanced model settings** (`hand_confirm_frames`, `hair_threshold`,
  `face_min_confidence`, and the three hand confidence values) live under
  Settings → Advanced, with a description next to each one.
- **Dark mode** lives under Settings → Appearance. It is applied when you click
  **Save**, like the other settings.
- **The CLI-only `--dwell-frames`** flag (default 3) controls how many
  consecutive frames are needed before an episode is declared. It is not in
  `config.json`.

---

## Activity tracking

Episodes are counted once per pull *episode* (the rising edge of the detected
state), not once per frame. Every pull is stored with its exact timestamp in
`activity.json`:

```json
{
  "2026": [
    "2026-10-29T09:05:00",
    "2026-10-30T14:25:00",
    "2026-10-30T14:27:12"
  ],
  "highscores": [
    {
      "start": "2026-10-01T10:00:00",
      "end": "2026-10-03T12:00:00",
      "seconds": 180000
    }
  ]
}
```

- The year keys are **years**; a new year is created automatically.
- Each year is a list of ISO timestamps, one per pull.
- `highscores` keeps the ten longest stretches without pulling.
- A stretch runs from the previous episode to the next. The first stretch of a
  session runs from app launch to the first episode, and the open stretch is
  recorded when the app closes.
- New stretches are inserted, sorted, and the shortest is dropped. A tie does
  not displace an existing entry.
- The streak timer and the graph are derived from the timestamps; there is no
  separate summary field.

### Calendar and graph

The heatmap starts **bright green** `(0, 255, 0)` for a day with no pulling.
Every pull removes `heatmap_green_step` from the green channel (10 by default),
so with the default 26+ pulls in a day is black: the more you pull, the more the
reward drains out of the calendar. Change the step in Settings if you want the
calendar to fade faster or slower.

Below it, the graph shows pulls per bucket for a selectable window:

| Period | Step | X-axis |
|---|---|---|
| Day | 1 minute | hours, plus a mark at every minute with a pull |
| Day | 1 hour | each hour |
| Week | 1 day | Mon-Sun |
| Month | 1 week | weeks of the month |
| Month | 1 day | each day |
| Year | 1 month | Jan-Dec |
| Year | 1 week | every fourth week |
| Year | 1 day | month boundaries |

Pick the window with the **Graph** dropdown and the reference date with
**Date**.

Delete `activity.json` to start with a clean history.

---

## Project layout

```
tricho_detector/
├── gui.py                    # PySide6 app: Camera / Settings / Activity
├── detect.py                 # command-line runner
├── detector.py               # shared core: models, decision logic, alerts
├── activity.py               # pull timestamps, streaks, daily counts
├── heatmap.py                # yearly heatmap widget
├── graph.py                  # pulls-per-bucket bar chart
├── theme.py                  # light/dark color tokens and global Qt styling
├── sound.py                  # cross-platform sound playback
├── install.py                # cross-platform installer
├── install.sh / install.bat  # installer wrappers
├── config.json               # user settings
├── activity.json             # pull history (gitignored; created by the installer)
├── requirements.txt          # pinned dependencies
├── dont_pull.jpeg            # source logo
├── dont_pull_square.png      # square notification icon
├── handpose-demo-keypoints.jpg
└── models/
    ├── blaze_face_short_range.tflite
    ├── hair_segmenter.tflite
    └── hand_landmarker.task
```

---

## Requirements and platform support

- **Python 3.10+**
- **Webcam** (built-in or USB)
- No GPU required; both models run on CPU

| Platform | Status |
|---|---|
| Windows 10/11 (x86-64, ARM64) | Supported |
| macOS 11+ (Apple Silicon) | Supported |
| Linux (glibc 2.28+, x86-64 / ARM64) | Supported |
| macOS on Intel | Not supported (MediaPipe ships arm64-only wheels) |
| Android / iOS | Not supported (no Python MediaPipe; background camera is also restricted on mobile) |

The MediaPipe models are Apache-2.0 licensed. They are downloaded by the
installer and are not committed to this repository.

---

## Privacy

- Camera frames are processed in memory and discarded immediately.
- Nothing is recorded, uploaded, or sent to any server.
- `activity.json` stores only the timestamp of each pull, with no images, no
  video, and no location data.
- `activity.json` is listed in `.gitignore`, so it is never committed to the
  repository.
- The feedback link opens an external Google Form in your browser; nothing is
  sent unless you choose to submit it.

---

## Disclaimer

This is an awareness and habit-interruption aid, **not a medical device**. It
cannot diagnose trichotillomania, and it cannot reliably tell a hair pull from
scratching, adjusting glasses, or resting your head on your hand. It is not a
substitute for professional care.

If hair pulling is affecting your life, speak to a doctor or mental-health
professional. Trichotillomania is treatable, and habit-reversal therapy in
particular has good evidence behind it.

---

## Feedback

Found a bug, or have an idea? Use the **Contact us** link at the bottom of the
Activity tab, or open the form directly:

<https://docs.google.com/forms/d/e/1FAIpQLSfDUIuLhHDg1paNdIf2CJqjNX1tQKJW5cqKV-TVLTISETO0Cw/viewform?usp=publish-editor>
