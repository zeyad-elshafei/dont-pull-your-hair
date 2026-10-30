#!/usr/bin/env python3
"""Cross-platform sound-effect playback via the pygame mixer.

The surface is deliberately tiny: construct one SoundPlayer and call
play(path). Missing pygame, a missing audio device, or a bad file only
disable/skip playback with a warning; nothing here raises into the detector.
"""

import os
import sys

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

try:
    import pygame
except Exception:                       # pygame not installed / broken
    pygame = None


class SoundPlayer:
    def __init__(self):
        self._enabled = False
        self._cache = {}
        if pygame is None:
            print("[sound] pygame is not installed; sound disabled",
                  file=sys.stderr)
            return
        try:
            pygame.mixer.init()
            self._enabled = True
        except Exception as e:
            print(f"[sound] audio unavailable: {e}", file=sys.stderr)

    @property
    def enabled(self):
        return self._enabled

    def play(self, path):
        if not self._enabled or not path:
            return
        if not os.path.isfile(path):
            print(f"[sound] file not found: {path}", file=sys.stderr)
            return
        try:
            sound = self._cache.get(path)
            if sound is None:
                sound = pygame.mixer.Sound(path)
                self._cache[path] = sound
            sound.play()
        except Exception as e:
            print(f"[sound] could not play {path}: {e}", file=sys.stderr)
