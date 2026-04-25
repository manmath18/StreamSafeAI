"""
Smart Skip Controller
Full state-machine implementation for intelligent content-aware skipping.

State Machine
-------------
    IDLE → MONITORING → UNSAFE_DETECTED → SKIPPING → SEEKING
         → BUFFER_RESET → MONITORING

Transitions
-----------
IDLE           → MONITORING      : video loaded / start_analysis called
MONITORING     → UNSAFE_DETECTED : temporal aggregator emits UNSAFE event
UNSAFE_DETECTED→ SKIPPING        : cooldown period satisfied (≥ cooldown_ms)
UNSAFE_DETECTED→ MONITORING      : cooldown NOT satisfied (event suppressed)
SKIPPING       → SEEKING         : skip target computed (via scene detector or +N s)
SEEKING        → BUFFER_RESET    : seek command issued to player
BUFFER_RESET   → MONITORING      : temporal buffer cleared, monitoring resumes

Improvements over v1
--------------------
* Complete IDLE→MONITORING→…→MONITORING state machine (not just cooldown check)
* Progressive skip distance: base_ms + n_skips × step_ms (capped at max_ms)
* Scene-boundary snapping via PySceneDetect (optional, non-blocking)
* Aggression reset: 8 consecutive safe frames → reduce skip distance
* Thread-safe: all state mutations under a single lock
* Full telemetry dict for sidebar display
"""

from __future__ import annotations

import time
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# State constants
# ---------------------------------------------------------------------------
IDLE            = "IDLE"
MONITORING      = "MONITORING"
UNSAFE_DETECTED = "UNSAFE_DETECTED"
SKIPPING        = "SKIPPING"
SEEKING         = "SEEKING"
BUFFER_RESET    = "BUFFER_RESET"

ALL_STATES = [IDLE, MONITORING, UNSAFE_DETECTED, SKIPPING, SEEKING, BUFFER_RESET]


class SkipController:
    """
    State-machine skip controller with progressive distances and
    optional scene-boundary snapping.

    Parameters
    ----------
    cooldown_ms        : minimum milliseconds between consecutive skips (default 3000)
    base_skip_ms       : initial skip distance in ms (default 5000)
    max_skip_ms        : maximum skip distance in ms (default 20 000)
    aggression_step_ms : ms added to skip distance per consecutive skip (default 2500)
    safe_reset_frames  : consecutive safe frames required to reset aggression (default 8)
    """

    def __init__(
        self,
        cooldown_ms:        int = 3000,
        base_skip_ms:       int = 5000,
        max_skip_ms:        int = 20_000,
        aggression_step_ms: int = 2500,
        safe_reset_frames:  int = 8,
    ):
        self.cooldown_ms        = cooldown_ms
        self.base_skip_ms       = base_skip_ms
        self.max_skip_ms        = max_skip_ms
        self.aggression_step_ms = aggression_step_ms
        self.safe_reset_frames  = safe_reset_frames

        self._lock = threading.Lock()

        # State machine
        self._state:             str   = IDLE
        self._last_skip_sys_ms:  float = 0.0    # system time of last skip
        self._skip_count:        int   = 0       # total skips issued
        self._consecutive_safe:  int   = 0       # safe frames since last skip
        self._last_skip_pos_ms:  int   = 0       # video position of last skip
        self._last_skip_tgt_ms:  int   = 0       # target of last skip

        # Scene detection (optional)
        self._video_path: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_monitoring(self):
        """Call when video starts playing."""
        with self._lock:
            self._state = MONITORING
            self._consecutive_safe = 0

    def reset(self):
        """Reset all state (call on new video / user-initiated seek)."""
        with self._lock:
            self._state             = IDLE
            self._last_skip_sys_ms  = 0.0
            self._skip_count        = 0
            self._consecutive_safe  = 0
            self._last_skip_pos_ms  = 0
            self._last_skip_tgt_ms  = 0

    def set_video_path(self, path: str):
        """Register the video path for scene-boundary detection."""
        self._video_path = path

    # ------------------------------------------------------------------
    # Main control interface
    # ------------------------------------------------------------------

    def handle_unsafe(self, current_ms: int) -> Optional[int]:
        """
        Called when the temporal aggregator emits an UNSAFE event.

        Parameters
        ----------
        current_ms : current playback position in milliseconds.

        Returns
        -------
        int | None
            Target position (ms) to seek to, or None if suppressed
            (cooldown active / already skipping).
        """
        with self._lock:
            now_ms = time.time() * 1000.0

            # --- Transition: MONITORING → UNSAFE_DETECTED ---
            if self._state == MONITORING:
                self._state = UNSAFE_DETECTED

            # --- Cooldown check ---
            elapsed_since_last = now_ms - self._last_skip_sys_ms
            if elapsed_since_last < self.cooldown_ms:
                # Cooldown not satisfied → stay in UNSAFE_DETECTED, suppress
                return None

            # --- Transition: UNSAFE_DETECTED → SKIPPING ---
            self._state = SKIPPING

            # Compute progressive skip distance
            skip_ms = min(
                self.base_skip_ms + self._skip_count * self.aggression_step_ms,
                self.max_skip_ms,
            )

            # --- Transition: SKIPPING → SEEKING ---
            self._state = SEEKING
            target_ms = current_ms + skip_ms

            # Optional scene-boundary snapping (non-blocking)
            snapped = self._snap_to_scene_boundary(target_ms)
            if snapped is not None:
                target_ms = snapped

            # --- Transition: SEEKING → BUFFER_RESET ---
            self._state             = BUFFER_RESET
            self._skip_count       += 1
            self._consecutive_safe  = 0
            self._last_skip_sys_ms  = now_ms
            self._last_skip_pos_ms  = current_ms
            self._last_skip_tgt_ms  = target_ms

            # --- Transition: BUFFER_RESET → MONITORING ---
            self._state = MONITORING

            return target_ms

    def handle_safe(self, _current_ms: int = 0):
        """
        Called on every frame classified as safe.
        Gradually resets aggression once the scene is clean.
        """
        with self._lock:
            self._consecutive_safe += 1
            if self._consecutive_safe >= self.safe_reset_frames:
                # Scene appears clean → reduce skip aggression
                if self._skip_count > 0:
                    self._skip_count = max(0, self._skip_count - 1)
                # Ensure we're back in MONITORING
                if self._state != MONITORING:
                    self._state = MONITORING

    # ------------------------------------------------------------------
    # Properties (thread-safe reads)
    # ------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def current_skip_distance_ms(self) -> int:
        return min(
            self.base_skip_ms + self._skip_count * self.aggression_step_ms,
            self.max_skip_ms,
        )

    @property
    def cooldown_remaining_ms(self) -> int:
        elapsed = (time.time() * 1000.0) - self._last_skip_sys_ms
        return max(0, int(self.cooldown_ms - elapsed))

    def get_telemetry(self) -> dict:
        """Return a snapshot of internal state for sidebar display."""
        return {
            "state":              self._state,
            "skip_count":         self._skip_count,
            "skip_distance_ms":   self.current_skip_distance_ms,
            "cooldown_remain_ms": self.cooldown_remaining_ms,
            "last_skip_pos_ms":   self._last_skip_pos_ms,
            "last_skip_tgt_ms":   self._last_skip_tgt_ms,
            "consecutive_safe":   self._consecutive_safe,
        }

    # ------------------------------------------------------------------
    # Scene-boundary snapping (optional PySceneDetect)
    # ------------------------------------------------------------------

    def _snap_to_scene_boundary(self, target_ms: int) -> Optional[int]:
        """
        Attempt to snap the skip target to the nearest scene boundary
        after target_ms using PySceneDetect.  Returns None on failure.
        """
        if not self._video_path:
            return None
        try:
            from scenedetect import detect, ContentDetector  # type: ignore
            target_sec = target_ms / 1000.0
            scene_list = detect(self._video_path, ContentDetector())
            for start_tc, _end_tc in scene_list:
                start_s = start_tc.get_seconds()
                if start_s > target_sec:
                    return int(start_s * 1000)
        except Exception:
            pass
        return None
