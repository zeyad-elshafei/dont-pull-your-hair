#!/usr/bin/env python3
"""Individual hair-pulling timestamps, persisted to activity.json.

Layout:

    {
      "2026": [
        "2026-10-30T14:25:00",
        "2026-10-30T14:27:12"
      ],
      "2027": [
        "2027-01-01T09:00:00"
      ]
    }

Top-level keys are years. Each year maps to a sorted list of ISO timestamps.
A year key is created automatically on the first pull in it.
"""

import datetime
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACTIVITY_PATH = os.path.join(BASE_DIR, "activity.json")


def load_activity(path=ACTIVITY_PATH):
    """Return the activity dict, or an empty dict if missing/unreadable."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_activity(data, path=ACTIVITY_PATH):
    """Write atomically so a crash cannot leave a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def record_pull(data, when=None):
    """Append the timestamp of a pull to its year's list. Mutates and returns."""
    when = when or datetime.datetime.now()
    if not isinstance(when, datetime.datetime):
        when = datetime.datetime.combine(when, datetime.time.min)
    stamp = when.replace(microsecond=0).isoformat()
    data.setdefault(str(when.year), []).append(stamp)
    return data


def pulls_for_year(data, year):
    """Return the sorted datetimes recorded for a year (empty if none)."""
    raw = data.get(str(year), [])
    if not isinstance(raw, list):
        return []
    pulls = []
    for stamp in raw:
        try:
            pulls.append(datetime.datetime.fromisoformat(stamp))
        except (TypeError, ValueError):
            continue
    pulls.sort()
    return pulls


def years_in(data):
    """Return every year key that is a plain integer, sorted ascending."""
    return sorted(int(y) for y in data if str(y).isdigit())


def all_pulls(data):
    """Return every recorded pull as a sorted list of datetimes."""
    pulls = []
    for year in years_in(data):
        pulls.extend(pulls_for_year(data, year))
    pulls.sort()
    return pulls


def last_pull_time(data):
    """Return the most recent pull datetime, or None if there is none."""
    pulls = all_pulls(data)
    return pulls[-1] if pulls else None


def counts_per_day(data, year):
    """Return {iso_date: count} for a year, used by the heatmap."""
    counts = {}
    for when in pulls_for_year(data, year):
        key = when.date().isoformat()
        counts[key] = counts.get(key, 0) + 1
    return counts
