"""
Temporal Smoothing Engine
Prevents flickering unsafe/safe decisions by combining:
  • Exponential weighted moving average (EWM)
  • Hysteresis thresholds (separate on/off thresholds)
  • Consecutive-frame confirmation gate
  • Debounce: once unsafe, stays unsafe for a minimum hold period
"""
from collections import deque


class TemporalSmoother:
    """
    Enhanced temporal smoother for streaming content-safety decisions.

    Algorithm:
        EWM(t) = alpha * score(t) + (1 - alpha) * EWM(t-1)
        State transitions:
            SAFE  → UNSAFE : EWM ≥ on_threshold  AND  consecutive_unsafe ≥ confirm_frames
            UNSAFE → SAFE  : EWM <  off_threshold AND  consecutive_safe  ≥ confirm_frames
        Additionally a minimum hold timer prevents rapid oscillation once
        the UNSAFE state is entered.

    Parameters
    ----------
    window_size : int
        Legacy sliding-window size (kept for backward compatibility; also
        used as a secondary average alongside EWM).
    threshold : float
        Primary on-threshold to enter UNSAFE state (passed via legacy API).
    alpha : float
        EWM decay factor. Higher → more responsive. Lower → smoother.
    off_threshold : float
        Threshold to return to SAFE (defaults to 80 % of on_threshold).
    confirm_frames : int
        Consecutive frames required before a state transition fires.
    hold_frames : int
        Minimum frames to stay in UNSAFE state before it can clear.
    """

    def __init__(
        self,
        window_size: int = 12,
        threshold: float = 0.72,
        alpha: float = 0.35,
        off_threshold: float = None,
        confirm_frames: int = 4,
        hold_frames: int = 6,
    ):
        self.window_size = window_size
        self.on_threshold = threshold
        self.off_threshold = off_threshold if off_threshold is not None else threshold * 0.80
        self.alpha = alpha
        self.confirm_frames = confirm_frames
        self.hold_frames = hold_frames

        # State
        self.buffer = deque(maxlen=window_size)
        self.ewm_score: float = 0.0
        self.state_unsafe: bool = False
        self._consecutive_unsafe: int = 0
        self._consecutive_safe: int = 0
        self._hold_counter: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, current_score: float):
        """
        Feed the latest raw frame score and get a smoothed decision.

        Returns
        -------
        (state_unsafe: bool, smoothed_score: float)
            `state_unsafe` — True when content should be flagged.
            `smoothed_score` — EWM-blended score in [0, 1].
        """
        # Update sliding window (legacy compatibility)
        self.buffer.append(current_score)
        window_avg = sum(self.buffer) / len(self.buffer)

        # EWM update (blends with window avg for extra stability)
        blended_input = 0.7 * current_score + 0.3 * window_avg
        self.ewm_score = self.alpha * blended_input + (1.0 - self.alpha) * self.ewm_score
        smoothed = self.ewm_score

        # Decrement hold counter
        if self._hold_counter > 0:
            self._hold_counter -= 1

        # State machine transitions
        if not self.state_unsafe:
            if smoothed >= self.on_threshold:
                self._consecutive_unsafe += 1
                self._consecutive_safe = 0
                if self._consecutive_unsafe >= self.confirm_frames:
                    self.state_unsafe = True
                    self._hold_counter = self.hold_frames
            else:
                self._consecutive_unsafe = 0
        else:
            if smoothed < self.off_threshold and self._hold_counter == 0:
                self._consecutive_safe += 1
                self._consecutive_unsafe = 0
                if self._consecutive_safe >= self.confirm_frames:
                    self.state_unsafe = False
                    self._consecutive_safe = 0
            else:
                self._consecutive_safe = 0

        return self.state_unsafe, smoothed

    def reset(self):
        """Reset all state (call on seek / new video)."""
        self.buffer.clear()
        self.ewm_score = 0.0
        self.state_unsafe = False
        self._consecutive_unsafe = 0
        self._consecutive_safe = 0
        self._hold_counter = 0

    @property
    def is_unsafe(self) -> bool:
        return self.state_unsafe

    @property
    def score(self) -> float:
        return self.ewm_score
