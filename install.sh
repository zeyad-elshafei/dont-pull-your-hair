#!/usr/bin/env bash
# Tricho installer for Linux and macOS.
set -euo pipefail
cd "$(dirname "$0")"

find_python() {
  for candidate in python3.14 python3.13 python3.12 python3.11 python3.10 python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
      if "$candidate" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
        printf '%s\n' "$candidate"
        return 0
      fi
    fi
  done
  return 1
}

if ! PYTHON=$(find_python); then
  echo "Error: Python 3.10 or newer is required but was not found." >&2
  echo "Install it from https://www.python.org/downloads/ and try again." >&2
  exit 1
fi

echo "Using $PYTHON ($("$PYTHON" --version 2>&1))"
exec "$PYTHON" install.py "$@"
