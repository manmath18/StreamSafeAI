"""
Spatio-Temporal Action Tube Aggregator
Implements temporal aggregation inspired by action tube modeling to produce
reliable scene-level safety decisions across consecutive frames rather than
isolated frame-level predictions.

Design
------
The core idea mirrors "action tube detection" from video action recognition:
 • A tube is a sequence of per-frame detection boxes / scores linked temporally.
 • Tubes are scored by their average confidence and spatial consistency.
 • Here we adapt this concept to *safety score tubes*: we track how long
   a high-confidence unsafe signal has persisted and weight recent frames
   more heavily, producing a temporally anchored decision.

Key Concepts
------------
TUBE SCORE
    A sliding window of N frame-level scores.  The tube score is a
    momentum-weighted average that gives more weight to:
    • Consecutive unsafe frames (spatial continuity proxy)
    • Recent frames (temporal recency bias via exponential weight)

TUBE CONFIRMATION
    The tube must remain above the ON-threshold for `min_tube_len` consecutive
    frames before an UNSAFE event is emitted.  This directly prevents false
    positives from isolated noisy frames.

HYSTERESIS DECAY
    After a tube drops below the OFF-threshold, the system requires
    `decay_frames` safe frames before it fully resets, bridging short
    "dropout" moments (e.g., camera cut within an intimate scene).
"""

from __future__ import annotations

from collections import deque
from typing import List, Tuple, Optional
import numpy as np


class ActionTubeAggregator:
    """
    Spatio-temporal action tube aggregator for scene-level safety decisions.

    Parameters
    ----------
    window_size : int
        Sliding window length N (default 12 frames ≈ 2.4 s at 5 FPS).
    on_threshold : float
        Tube score required to start a potential UNSAFE tube (default 0.60).
    off_threshold : float
        Tube score below which a tube is considered broken (default 0.42).
    min_tube_len : int
        Minimum consecutive frames above on_threshold to confirm UNSAFE
        (default 4 frames ≈ 0.8 s — prevents single-frame false positives).
    decay_frames : int
        Number of consecutive safe frames required after a tube breaks
        before the state fully resets to SAFE (default 6).
    ewm_alpha : float
        Exponential weight for recency bias in the sliding window
        (default 0.35 — 0 = equal weight, 1 = only latest frame).
    tube_momentum : float
        How much the previous tube score persists when current score is
        below the on_threshold (0 = no memory, 1 = never decays).
        Default 0.55 — provides the "bridge" effect for short dropouts.
    """

    def __init__(
        self,
        window_size:    int   = 12,
        on_threshold:   float = 0.60,
        off_threshold:  float = 0.42,
        min_tube_len:   int   = 4,
        decay_frames:   int   = 6,
        ewm_alpha:      float = 0.35,
        tube_momentum:  float = 0.55,
    ):
        self.window_size   = window_size
        self.on_threshold  = on_threshold
        self.off_threshold = off_threshold
        self.min_tube_len  = min_tube_len
        self.decay_frames  = decay_frames
        self.ewm_alpha     = ewm_alpha
        self.tube_momentum = tube_momentum

        # Sliding window buffer
        self._window: deque[float] = deque(maxlen=window_size)

        # EWM state
        self._ewm_score: float = 0.0

        # Tube state
        self._tube_score:         float = 0.0
        self._tube_len:           int   = 0   # consecutive frames in current tube
        self._consecutive_safe:   int   = 0   # consecutive safe frames after break
        self._state_unsafe:       bool  = False
        self._hold_counter:       int   = 0   # frames to hold UNSAFE after tube breaks

        # Statistics (for telemetry / sidebar)
        self.total_tubes_confirmed: int  = 0
        self.current_tube_length:   int  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame_score: float) -> Tuple[bool, float]:
        """
        Ingest a new frame-level safety score and return the current
        tube-based decision.

        Parameters
        ----------
        frame_score : float
            Raw frame-level unsafe probability in [0, 1].

        Returns
        -------
        (is_unsafe: bool, tube_score: float)
            is_unsafe  — True when an active unsafe tube has been confirmed.
            tube_score — Smoothed tube score in [0, 1].
        """
        # 1. Update sliding window
        self._window.append(frame_score)

        # 2. EWM update (recency-biased average)
        a = self.ewm_alpha
        self._ewm_score = a * frame_score + (1.0 - a) * self._ewm_score

        # 3. Window average (equal-weighted baseline)
        win_avg = float(np.mean(self._window)) if self._window else 0.0

        # 4. Blended input to tube tracker (EWM + window)
        blended = 0.65 * self._ewm_score + 0.35 * win_avg

        # 5. Tube score update with momentum
        if blended >= self.on_threshold:
            # Active tube: score rises quickly
            self._tube_score = max(blended, self.tube_momentum * self._tube_score)
            self._tube_len  += 1
            self.current_tube_length = self._tube_len
            self._consecutive_safe   = 0
        else:
            # Below threshold: decay with momentum
            self._tube_score = self.tube_momentum * self._tube_score
            self._tube_len   = 0
            self.current_tube_length = 0
            self._consecutive_safe  += 1

        # 6. Hold counter (keeps UNSAFE alive for a few frames after tube breaks)
        if self._hold_counter > 0:
            self._hold_counter -= 1

        # 7. State machine transitions
        if not self._state_unsafe:
            # Transition to UNSAFE: tube must be confirmed for min_tube_len frames
            if (self._tube_score >= self.on_threshold
                    and self._tube_len >= self.min_tube_len):
                self._state_unsafe = True
                self._hold_counter = self.decay_frames
                self.total_tubes_confirmed += 1
        else:
            # Transition back to SAFE: tube score + hold + decay frames
            if (self._tube_score < self.off_threshold
                    and self._hold_counter == 0
                    and self._consecutive_safe >= self.decay_frames):
                self._state_unsafe = False
                self._consecutive_safe = 0

        return self._state_unsafe, float(np.clip(self._tube_score, 0.0, 1.0))

    def reset(self):
        """Clear all state (call on seek / new video load)."""
        self._window.clear()
        self._ewm_score          = 0.0
        self._tube_score         = 0.0
        self._tube_len           = 0
        self._consecutive_safe   = 0
        self._state_unsafe       = False
        self._hold_counter       = 0
        self.current_tube_length = 0

    # ------------------------------------------------------------------
    # Properties for telemetry
    # ------------------------------------------------------------------

    @property
    def is_unsafe(self) -> bool:
        return self._state_unsafe

    @property
    def tube_score(self) -> float:
        return float(np.clip(self._tube_score, 0.0, 1.0))

    @property
    def ewm_score(self) -> float:
        return self._ewm_score

    @property
    def tube_active(self) -> bool:
        """True if a tube is currently building (may not yet be confirmed)."""
        return self._tube_score >= self.on_threshold

    def get_diagnostics(self) -> dict:
        """Return a dict of internal state for sidebar telemetry."""
        return {
            "tube_score":         round(self.tube_score, 3),
            "ewm_score":          round(self._ewm_score, 3),
            "tube_length":        self._tube_len,
            "consecutive_safe":   self._consecutive_safe,
            "hold_remaining":     self._hold_counter,
            "is_unsafe":          self._state_unsafe,
            "tubes_confirmed":    self.total_tubes_confirmed,
        }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agg = ActionTubeAggregator(window_size=12, on_threshold=0.60, min_tube_len=4)

    # Simulate: 5 safe frames, then 8 unsafe, then 6 safe
    scores = [0.1]*5 + [0.85]*8 + [0.1]*6
    print("Frame | Score | is_unsafe | tube_score")
    print("-" * 45)
    for i, s in enumerate(scores):
        unsafe, tube = agg.update(s)
        print(f"  {i:3d} | {s:.2f}  | {'YES' if unsafe else 'no ':>3}       | {tube:.3f}")
    print(f"\nTotal tubes confirmed: {agg.total_tubes_confirmed}")
