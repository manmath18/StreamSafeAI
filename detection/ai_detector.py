"""
SafeVision AI Detector Engine  (v3 – Full Spatio-Temporal Pipeline)

Architecture
------------

  ┌─────────────────────────────────────────────────────────────┐
  │                  SafeVisionEngine (QThread)                 │
  │                                                             │
  │  ┌──────────────┐    ┌────────────────────────────────┐    │
  │  │_FramePrefetcher│   │  Batch Inference Queue (4 fps) │    │
  │  │ (daemon thread)│──►│  Groups frames into batches of │    │
  │  │ reads 1.5s    │   │  size 4 for GPU efficiency     │    │
  │  │ ahead, ×0.5   │   └──────────┬─────────────────────┘    │
  │  └──────────────┘              │                            │
  │                                ▼                            │
  │            ┌───────────────────────────────────┐           │
  │            │    Parallel Inference Stage        │           │
  │            │  1. EfficientNet-B0  (PyTorch)     │           │
  │            │  2. ONNX Nude Detector             │           │
  │            │  3. CV Heuristics (skin/face/blur) │           │
  │            └──────────────┬────────────────────┘           │
  │                           │                                 │
  │                           ▼                                 │
  │            ┌───────────────────────────────────┐           │
  │            │   Late Fusion (weighted average)   │           │
  │            │  S = α_eff*S_eff + α_onnx*S_onnx  │           │
  │            │      + α_cv*S_cv + α_blur*S_blur   │           │
  │            └──────────────┬────────────────────┘           │
  │                           │                                 │
  │                           ▼                                 │
  │            ┌───────────────────────────────────┐           │
  │            │   ActionTubeAggregator             │           │
  │            │   Spatio-temporal tube confirmation│           │
  │            │   (N=12 frame window, min_tube=4)  │           │
  │            └──────────────┬────────────────────┘           │
  │                           │                                 │
  │                           ▼                                 │
  │            ┌───────────────────────────────────┐           │
  │            │   SkipController (state machine)   │           │
  │            │   IDLE→MONITORING→UNSAFE_DETECTED  │           │
  │            │   →SKIPPING→SEEKING→BUFFER_RESET   │           │
  │            │   →MONITORING                      │           │
  │            └──────────────┬────────────────────┘           │
  │                           │                                 │
  │            stats_updated  │  skip_requested                 │
  │            skip_detail_updated (rich telemetry)             │
  └───────────────────────────┼─────────────────────────────────┘
                              ▼
                         PyQt6 UI (RightSidebar)
"""

from __future__ import annotations

import cv2
import time
import threading
import queue
from typing import Optional, List

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

# Internal modules
from temporal_engine.action_tube import ActionTubeAggregator
from skip_controller.smart_skip import SkipController, MONITORING
from detection.simple_detector import analyse_frame_cv, DETECTOR_CONFIG
from detection.efficientnet_classifier import EfficientNetClassifier
from live import LiveNudeDetector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENGINE_CONFIG = {
    # ---------------------------------------------------------------
    # Fusion weights (must sum to 1.0)
    # EfficientNet is NOT fine-tuned → weight=0 until a checkpoint is
    # provided via EfficientNetClassifier.load_weights(path).
    # ONNX nude-detector + CV heuristics carry the full detection load.
    # ---------------------------------------------------------------
    "W_EFFICIENTNET": 0.00,   # Disabled: random head = ~0.5 noise on every frame
    "W_ONNX":         0.55,   # Primary: ONNX nude-detector (trained)
    "W_CV":           0.35,   # Key for kiss/skin scenes not in ONNX
    "W_BLUR":         0.10,   # Motion blur signal

    # Sampling
    "SAMPLE_MS":      200,    # 5 FPS analysis rate
    "BATCH_SIZE":     1,      # Process every frame individually (no batch delay)
    "PREFETCH_QUEUE": 12,     # max frames buffered ahead
    "LOOKAHEAD_MS":   1500,   # how far ahead to pre-read (ms)
    "PREFETCH_SCALE": 0.5,    # downscale factor for pre-read frames

    # Temporal tube — lowered thresholds so real detections fire
    "TUBE_WINDOW":    12,     # sliding window size
    "TUBE_ON_THRESH": 0.45,   # was 0.60 — lower so CV+ONNX can trigger
    "TUBE_OFF_THRESH":0.28,   # was 0.42
    "TUBE_MIN_LEN":   3,      # confirmed unsafe after N consecutive frames (was 4)
    "TUBE_DECAY":     5,      # safe frames to fully reset

    # Skip controller
    "COOLDOWN_MS":    3000,
    "BASE_SKIP_MS":   5000,
    "MAX_SKIP_MS":    20_000,
    "AGGR_STEP_MS":   2500,
}


# ---------------------------------------------------------------------------
# Frame Prefetcher Thread
# ---------------------------------------------------------------------------
class _FramePrefetcher(threading.Thread):
    """
    Daemon thread that reads frames 1.5 s ahead of the player, downscales them,
    and queues them for the inference pipeline.

    Decoupling reading from inference ensures the player thread is never blocked
    and the inference engine always has work queued.
    """

    def __init__(
        self,
        cap:          cv2.VideoCapture,
        out_q:        queue.Queue,
        lookahead_ms: float = 1500,
        sample_ms:    float = 200,
        scale:        float = 0.5,
    ):
        super().__init__(daemon=True, name="FramePrefetcher")
        self.cap          = cap
        self.out_q        = out_q
        self.lookahead_ms = lookahead_ms
        self.sample_ms    = sample_ms
        self.scale        = scale
        self.running      = True
        self._player_ref  = None
        self._lock        = threading.Lock()
        self._pending_seek: Optional[float] = None

    def set_player(self, player_ref):
        self._player_ref = player_ref

    def request_seek(self, target_ms: float):
        """Thread-safe seek request issued by the engine on skip."""
        with self._lock:
            self._pending_seek = target_ms

    def run(self):
        while self.running:
            # --- Handle pending seek (drain stale frames first) ---
            with self._lock:
                if self._pending_seek is not None:
                    while not self.out_q.empty():
                        try:
                            self.out_q.get_nowait()
                        except queue.Empty:
                            break
                    self.cap.set(cv2.CAP_PROP_POS_MSEC, self._pending_seek)
                    self._pending_seek = None

            if self._player_ref is None:
                time.sleep(0.05)
                continue

            player_ms = self._player_ref.get_time()
            if player_ms < 0:
                time.sleep(0.05)
                continue

            cap_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            # Catch up if reader fell behind
            if cap_ms < player_ms:
                self.cap.set(cv2.CAP_PROP_POS_MSEC, player_ms)
                cap_ms = player_ms

            # Slow down if we're sufficiently ahead
            if cap_ms > player_ms + self.lookahead_ms or self.out_q.full():
                time.sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)

            if self.scale < 1.0:
                frame_small = cv2.resize(
                    frame, (0, 0),
                    fx=self.scale, fy=self.scale,
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                frame_small = frame

            try:
                self.out_q.put_nowait((pos_ms, frame_small))
            except queue.Full:
                time.sleep(0.02)

    def stop(self):
        self.running = False


# ---------------------------------------------------------------------------
# Batch Frame Buffer (groups frames for EfficientNet batch inference)
# ---------------------------------------------------------------------------
class _BatchBuffer:
    """Accumulates frames and flushes them when batch_size is reached or on demand."""

    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self._items: List[tuple] = []   # list of (pos_ms, frame)

    def push(self, pos_ms: float, frame: np.ndarray):
        self._items.append((pos_ms, frame))

    def ready(self) -> bool:
        return len(self._items) >= self.batch_size

    def flush(self) -> List[tuple]:
        items = self._items[:]
        self._items = []
        return items

    def flush_all(self) -> List[tuple]:
        """Flush whatever is buffered, even if not full."""
        return self.flush()

    def clear(self):
        self._items = []


# ---------------------------------------------------------------------------
# SafeVisionEngine
# ---------------------------------------------------------------------------
class SafeVisionEngine(QThread):
    """
    Main real-time detection engine — runs on a dedicated QThread.

    Signals
    -------
    stats_updated(score: float, fps: float, is_safe: bool)
        Emitted per analysis batch.  Connected to the right-sidebar risk bar.

    skip_requested(start_ms: int, end_ms: int)
        Emitted when a skip is actually executed.  Connected to the player.

    skip_detail_updated(telemetry: dict)
        Rich per-frame telemetry dict for the skip-logic monitor panel.
    """

    stats_updated      = pyqtSignal(float, float, bool)
    skip_requested     = pyqtSignal(int, int)
    skip_detail_updated = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.video_path:  Optional[str] = None
        self.player_ref                 = None
        self.running:     bool          = False

        cfg = ENGINE_CONFIG

        # --- Spatio-temporal tube aggregator ---
        self.tube = ActionTubeAggregator(
            window_size   = cfg["TUBE_WINDOW"],
            on_threshold  = cfg["TUBE_ON_THRESH"],
            off_threshold = cfg["TUBE_OFF_THRESH"],
            min_tube_len  = cfg["TUBE_MIN_LEN"],
            decay_frames  = cfg["TUBE_DECAY"],
        )

        # --- Skip controller (full state machine) ---
        self.skip_ctrl = SkipController(
            cooldown_ms        = cfg["COOLDOWN_MS"],
            base_skip_ms       = cfg["BASE_SKIP_MS"],
            max_skip_ms        = cfg["MAX_SKIP_MS"],
            aggression_step_ms = cfg["AGGR_STEP_MS"],
        )

        # --- Deep-learning classifiers ---
        self.eff_clf = EfficientNetClassifier(batch_size=cfg["BATCH_SIZE"])
        self.onnx_det = LiveNudeDetector()
        print("SafeVisionEngine: all models loaded")

        # --- CV cooldown bridges ---
        self._kiss_cd: int = 0
        self._skin_cd: int = 0

        # --- Prefetch pipeline (created in run()) ---
        self._prefetch_q: Optional[queue.Queue]    = None
        self._prefetcher: Optional[_FramePrefetcher] = None
        self._read_cap:   Optional[cv2.VideoCapture] = None

        # --- Batch buffer ---
        self._batch_buf = _BatchBuffer(batch_size=cfg["BATCH_SIZE"])

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start_analysis(self, video_path: str, player_ref):
        self.video_path = video_path
        self.player_ref = player_ref
        self.skip_ctrl.set_video_path(video_path)
        self._reset_all_state()
        self.running = True
        self.start()

    def stop_analysis(self):
        self.running = False
        if self._prefetcher:
            self._prefetcher.stop()
        self.wait()

    # ------------------------------------------------------------------
    # Main run loop
    # ------------------------------------------------------------------

    def run(self):
        cfg = ENGINE_CONFIG

        self._read_cap = cv2.VideoCapture(self.video_path)
        if not self._read_cap.isOpened():
            return

        self._prefetch_q = queue.Queue(maxsize=cfg["PREFETCH_QUEUE"])
        self._prefetcher  = _FramePrefetcher(
            cap          = self._read_cap,
            out_q        = self._prefetch_q,
            lookahead_ms = cfg["LOOKAHEAD_MS"],
            sample_ms    = cfg["SAMPLE_MS"],
            scale        = cfg["PREFETCH_SCALE"],
        )
        self._prefetcher.set_player(self.player_ref)
        self._prefetcher.start()

        self.skip_ctrl.start_monitoring()

        last_sample_ms: float = -1.0

        while self.running and self.player_ref:
            if not self.player_ref.is_playing():
                time.sleep(0.05)
                continue

            current_ms = float(self.player_ref.get_time())
            if current_ms < 0:
                time.sleep(0.05)
                continue

            # Pull next prefetched frame
            frame_data = self._pull_frame(current_ms)
            if frame_data is None:
                time.sleep(0.02)
                continue

            pos_ms, frame = frame_data

            if pos_ms <= last_sample_ms:
                time.sleep(0.01)
                continue
            last_sample_ms = pos_ms

            t0 = time.time()

            # ----------------------------------------------------------------
            # Process the single frame immediately — no batch buffer.
            # Batching was causing frames to be silently discarded when
            # flush_all() + flush() were called in sequence.
            # ----------------------------------------------------------------
            frames_batch = [frame]
            last_pos_ms  = pos_ms

            # ----------------------------------------------------------------
            # Stage 1 – EfficientNet (disabled until fine-tuned checkpoint)
            # Weight is 0.0 so this is a no-op, but the hook remains for
            # when a proper NSFW-trained checkpoint is available.
            # ----------------------------------------------------------------
            eff_score = 0.0  # not used in fusion (W_EFFICIENTNET = 0.0)

            # ----------------------------------------------------------------
            # Stage 2 – ONNX Nude Detector
            # ----------------------------------------------------------------
            onnx_score = self._run_onnx(frames_batch[-1])

            # ----------------------------------------------------------------
            # Stage 3 – CV Heuristics (kiss, skin, motion)
            # ----------------------------------------------------------------
            cv_result = analyse_frame_cv(frames_batch[-1], int(last_pos_ms))
            cv_score, cv_result = self._apply_cooldown(cv_result)

            # ----------------------------------------------------------------
            # Stage 4 – Motion blur signal
            # ----------------------------------------------------------------
            gray = cv2.cvtColor(frames_batch[-1], cv2.COLOR_BGR2GRAY)
            lap_var    = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_score = float(max(0.0, min(1.0, 50.0 / (lap_var + 1.0))))

            # ----------------------------------------------------------------
            # Stage 5 – Late fusion
            # ----------------------------------------------------------------
            w = ENGINE_CONFIG
            S = (
                w["W_EFFICIENTNET"] * eff_score
                + w["W_ONNX"]       * onnx_score
                + w["W_CV"]         * cv_score
                + w["W_BLUR"]       * blur_score
            )
            S = float(np.clip(S, 0.0, 1.0))

            # ----------------------------------------------------------------
            # Stage 6 – Spatio-temporal tube aggregation
            # ----------------------------------------------------------------
            is_unsafe, tube_score = self.tube.update(S)

            infer_time = time.time() - t0
            fps        = min(1.0 / (infer_time + 0.001), 30.0)

            self.stats_updated.emit(tube_score, fps, not is_unsafe)

            # ----------------------------------------------------------------
            # Stage 7 – Skip control state machine
            # ----------------------------------------------------------------
            skip_fired     = False
            skip_target_ms: Optional[int] = None

            if is_unsafe:
                target = self.skip_ctrl.handle_unsafe(int(current_ms))
                if target is not None:
                    skip_fired     = True
                    skip_target_ms = target
                    # Realign prefetcher and clear tube buffer
                    self._prefetcher.request_seek(float(target))
                    self._reset_all_state()
                    self.skip_requested.emit(int(current_ms), target)
            else:
                self.skip_ctrl.handle_safe(int(current_ms))

            # ----------------------------------------------------------------
            # Stage 8 – Emit rich telemetry for sidebar
            # ----------------------------------------------------------------
            tube_diag  = self.tube.get_diagnostics()
            skip_telem = self.skip_ctrl.get_telemetry()

            telemetry = {
                # Event type
                "event":              "SKIP" if skip_fired else "FRAME",
                "position_ms":        int(current_ms),
                # Per-model scores
                "eff_score":          round(eff_score,  3),
                "onnx_score":         round(onnx_score, 3),
                "cv_score":           round(cv_score,   3),
                "blur_score":         round(blur_score, 3),
                "fused_score":        round(S,          3),
                # Tube aggregator
                "tube_score":         tube_diag["tube_score"],
                "smoothed_score":     tube_diag["tube_score"],  # legacy key for sidebar bars
                "tube_length":        tube_diag["tube_length"],
                "tubes_confirmed":    tube_diag["tubes_confirmed"],
                # Decision
                "is_unsafe":          is_unsafe,
                # CV detections (for trigger badges)
                "detections":         cv_result.get("detections", []),
                # Skip controller
                "skip_count":         skip_telem["skip_count"],
                "skip_distance_ms":   skip_telem["skip_distance_ms"],
                "cooldown_remain_ms": skip_telem["cooldown_remain_ms"],
                "skip_target_ms":     skip_target_ms,
                "skip_state":         skip_telem["state"],
                # Performance
                "fps":                round(fps, 1),
                "infer_ms":           round(infer_time * 1000, 1),
                "batch_size":         len(frames_batch),
            }
            self.skip_detail_updated.emit(telemetry)

            # Throttle
            elapsed    = time.time() - t0
            sleep_time = (cfg["SAMPLE_MS"] / 1000.0) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Cleanup
        if self._prefetcher:
            self._prefetcher.stop()
        if self._read_cap:
            self._read_cap.release()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset_all_state(self):
        """Reset tube and cooldown bridges (called on seek/init)."""
        self.tube.reset()
        self._kiss_cd = 0
        self._skin_cd = 0

    def _pull_frame(self, current_ms: float) -> Optional[tuple]:
        """
        Pull the best matching prefetched frame from the queue.
        Accepts frames within [current_ms, current_ms + LOOKAHEAD_MS].
        Frames older than current_ms are dropped (stale).
        Frames beyond the deadline are put back (too far ahead).
        """
        best    = None
        drained = []
        deadline = current_ms + ENGINE_CONFIG["LOOKAHEAD_MS"]

        while True:
            try:
                item = self._prefetch_q.get_nowait()
                pos_ms, _ = item
                if pos_ms < current_ms:
                    # Stale — behind player, discard
                    continue
                elif pos_ms <= deadline:
                    best = item   # keep pulling to get the most recent valid frame
                else:
                    drained.append(item)   # too far ahead – put back
                    break
            except queue.Empty:
                break

        for item in drained:
            try:
                self._prefetch_q.put_nowait(item)
            except queue.Full:
                pass

        return best

    def _run_onnx(self, frame: np.ndarray) -> float:
        """Run ONNX nude detector and return a unified threat score."""
        try:
            detections = self.onnx_det.detect_frame(frame)
        except Exception:
            return 0.1

        max_score  = 0.0
        has_unsafe = False
        frame_area = frame.shape[0] * frame.shape[1]

        for d in detections:
            label    = d.get("class", "")
            severity = d.get("severity", "MODERATE")
            score    = float(d.get("score", 0.0))

            if not self.onnx_det.should_apply_blur(label):
                continue

            if severity == "CRITICAL":
                has_unsafe = True
                max_score  = max(max_score, min(1.0, score + 0.40))
            elif severity == "HIGH":
                has_unsafe = True
                max_score  = max(max_score, min(1.0, score + 0.30))
            elif severity == "MODERATE" and score >= 0.40:
                has_unsafe = True
                max_score  = max(max_score, score)
            elif severity == "LOW" and score >= 0.60:
                has_unsafe = True
                max_score  = max(max_score, score)

            # Area-weighted skin-exposure bonus
            if "EXPOSED" in label and score > 0.35:
                box = d.get("box", [0, 0, 0, 0])
                area_ratio = (box[2] * box[3]) / max(frame_area, 1)
                if area_ratio > 0.05:
                    has_unsafe = True
                    max_score  = max(max_score, min(1.0, score + 0.15))

        return max_score if has_unsafe else 0.1

    def _apply_cooldown(self, cv_result: dict):
        """Apply face/skin cooldown bridges to CV score."""
        cv_score = cv_result.get("raw_score", 0.0)
        detections = cv_result.get("detections", [])

        face_tags = [t for t in detections if "face" in t or "kiss" in t]
        skin_tags = [t for t in detections if "skin" in t]

        if face_tags:
            self._kiss_cd = DETECTOR_CONFIG["KISS_COOLDOWN"]
        elif self._kiss_cd > 0:
            self._kiss_cd -= 1
            cv_score = max(cv_score, 0.85)
            detections.append("kiss_cooldown_bridge")

        if skin_tags:
            self._skin_cd = DETECTOR_CONFIG["SKIN_COOLDOWN"]
        elif self._skin_cd > 0:
            self._skin_cd -= 1
            cv_score = max(cv_score, 0.80)
            detections.append("skin_cooldown_bridge")

        cv_result["raw_score"]  = cv_score
        cv_result["detections"] = detections
        return cv_score, cv_result
