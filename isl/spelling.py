"""
Fingerspelling logic: turns a stream of per-frame predictions into typed letters.

Features:
1. Smoothing via majority vote over recent window.
2. Hold time: Supports both real-time hold (seconds) and frame-based hold (frames).
3. Re-arming / debounce lockout: Requires hand release or sign change before repeating a letter.
4. Auto-completion, space, backspace, and clear helpers.
"""
from __future__ import annotations

import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from .config import DEFAULT_HOLD_FRAMES


@dataclass
class SpellingSnapshot:
    text: str
    current_word: str
    candidate: Optional[str]
    progress: float              # 0..1 - how close candidate is to being typed
    last_event: Optional[str]    # e.g. "letter:H", "space", "word:HELLO"


class SpellingBuffer:
    def __init__(self, hold_frames: int = DEFAULT_HOLD_FRAMES, window: int = 5,
                 release_frames: int = 4, majority: float = 0.6, word_mode: bool = False,
                 hold_seconds: Optional[float] = None):
        self.word_mode = word_mode
        self.hold_seconds = hold_seconds
        self.min_streak = 3
        self.hold_frames = max(1, int(hold_frames))
        self.window = max(1, int(window))
        self.release_frames = max(1, int(release_frames))
        self.majority = majority
        self._lock = threading.RLock()
        self._votes: Deque[Optional[str]] = deque(maxlen=self.window)
        self._text = ""
        self._candidate: Optional[str] = None
        self._streak = 0
        self._score = 0.0
        self._since = 0.0
        self._last_t = 0.0
        self._armed = True
        self._release = 0
        self._last_typed: Optional[str] = None
        self._last_event: Optional[str] = None
        self.events: List[str] = []

    def update(self, label: Optional[str], now: Optional[float] = None) -> Optional[str]:
        with self._lock:
            now = time.monotonic() if now is None else now
            self._last_t = now
            self._votes.append(label)
            voted = self._vote()

            if voted is None:
                if self.hold_seconds:
                    # In real-time live mode, gently decay instead of instantaneous reset
                    self._score = max(0.0, self._score - 0.4)
                    if self._score < 0.2:
                        self._candidate, self._streak = None, 0
                else:
                    self._candidate, self._streak = None, 0
                self._release += 1
                if self._release >= self.release_frames:
                    self._armed = True
                return None

            if voted == self._candidate:
                self._streak += 1
                self._score += 1.0
            else:
                self._candidate = voted
                self._streak = 1
                self._score = 1.0
                self._since = now
                if voted != self._last_typed:
                    self._armed = True
                    self._release = 0

            if self.hold_seconds:
                elapsed = max(0.0, now - self._since)
                held = (self._streak >= self.min_streak) and (elapsed >= self.hold_seconds)
            else:
                held = self._streak >= self.hold_frames

            if held and self._armed:
                return self._commit(voted)
            return None

    def _vote(self) -> Optional[str]:
        real = [v for v in self._votes if v is not None]
        if not real or len(real) < max(1, int(self.window * 0.5)):
            return None
        top, n = Counter(real).most_common(1)[0]
        return top if n / len(self._votes) >= self.majority else None

    def _commit(self, token: str) -> str:
        if self.word_mode:
            self._text += token + " "
            self.events.append(f"word:{token}")
        else:
            self._text += token
        self._last_typed = token
        self._armed = False
        self._release = 0
        self._streak = 0
        self._score = 0.0
        self._last_event = f"letter:{token}"
        return token

    def append(self, token: str) -> None:
        with self._lock:
            self._commit(token)
            self._armed = True

    def add_space(self) -> None:
        with self._lock:
            word = self.current_word
            if self._text and not self._text.endswith(" "):
                self._text += " "
                self._last_event = "space"
                if word:
                    self.events.append(f"word:{word}")
            self._last_typed, self._armed = None, True

    def backspace(self) -> None:
        with self._lock:
            self._text = self._text[:-1]
            self._last_typed, self._armed = None, True

    def clear(self) -> None:
        with self._lock:
            self._text, self._last_typed, self._armed = "", None, True
            self._candidate, self._streak, self._score = None, 0, 0.0
            self._votes.clear()

    def set_text(self, text: str) -> None:
        with self._lock:
            self._text = text

    def replace_current_word(self, word: str) -> None:
        with self._lock:
            head = self._text[: len(self._text) - len(self.current_word)]
            self._text = head + word + " "
            self.events.append(f"word:{word}")
            self._last_typed, self._armed = None, True

    def pop_events(self) -> List[str]:
        with self._lock:
            ev, self.events = self.events, []
            return ev

    @property
    def text(self) -> str:
        with self._lock:
            return self._text

    @property
    def current_word(self) -> str:
        with self._lock:
            return self._text.split(" ")[-1] if self._text and not self._text.endswith(" ") else ""

    def snapshot(self) -> SpellingSnapshot:
        with self._lock:
            if self.hold_seconds:
                progress = min(max(self._last_t - self._since, 0.0) / self.hold_seconds, 1.0) if self._candidate else 0.0
            else:
                progress = min(self._streak / self.hold_frames, 1.0)
            progress = progress if self._armed else 0.0
            return SpellingSnapshot(self._text, self.current_word, self._candidate,
                                    progress, self._last_event)

    def reconfigure(self, hold_frames: Optional[int] = None, word_mode: Optional[bool] = None,
                    hold_seconds: Optional[float] = None) -> None:
        with self._lock:
            if hold_frames:
                self.hold_frames = max(1, int(hold_frames))
            if hold_seconds is not None:
                self.hold_seconds = float(hold_seconds)
            if word_mode is not None:
                self.word_mode = word_mode
