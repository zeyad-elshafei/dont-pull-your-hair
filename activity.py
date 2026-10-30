#!/usr/bin/env python3
"""Per-day hair-pulling counts, persisted to activity.json.

Layout:

    {
      "2026": {
        "2026-10-29": 1,
        "2026-10-30": 3
      }
    }

Top-level keys are years. A year key is created automatically the first time
a pull is recorded in it, so the file grows one year at a time.
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
    """Write atomically so a crash can't leave a half-written file."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def record_pull(data, when=None):
    """Increment the count for `when` (default: now) and stamp last_pull."""
    when = when or datetime.datetime.now()
    if isinstance(when, datetime.datetime):
        day = when.date()
        stamp = when.replace(microsecond=0).isoformat()
    else:
        day = when
        stamp = datetime.datetime.combine(day, datetime.time.min).isoformat()

    data["last_pull"] = stamp
    year = str(day.year)
    key = day.isoformat()
    days = data.setdefault(year, {})
    days[key] = int(days.get(key, 0)) + 1
    return data


def last_pull_time(data):
    """Return the parsed last_pull timestamp, or None if absent/invalid."""
    raw = data.get("last_pull")
    if not isinstance(raw, str):
        return None
    try:
        return datetime.datetime.fromisoformat(raw)
    except ValueError:
        return None


def counts_for_year(data, year):
    """Return the {iso_date: count} mapping for a year (may be empty)."""
    counts = data.get(str(year), {})
    return counts if isinstance(counts, dict) else {}


def years_in(data):
    """Return every year key that is a plain integer, sorted ascending."""
    return sorted(int(y) for y in data if str(y).isdigit())
