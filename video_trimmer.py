#!/usr/bin/env python3
"""
SafeVision Video Trimmer
========================
Analyses a video frame-by-frame using the ONNX nudity detection model,
identifies obscene/unsafe scene segments, and uses FFmpeg to trim them out.
The result is a clean MP4 with the original audio preserved.

Usage:
    from video_trimmer import VideoTrimmer
    trimmer = VideoTrimmer("input.mp4")
    segments = trimmer.analyze()          # returns list of (start_sec, end_sec, risk)
    output   = trimmer.trim("output.mp4") # returns path to trimmed file
"""

import os
import cv2
import math
import json
import time
import subprocess
import tempfile
import threading
import numpy as np
from pathlib import Path

# ─── ONNX / Model imports ────────────────────────────────────────────────────
try:
    import onnxruntime
    import onnx
    from onnx import version_converter
    ONNX_OK = True
except ImportError:
    ONNX_OK = False
    print("[VideoTrimmer] WARNING: onnxruntime not installed.")

# ─── Labels & risk classification (identical to video.py / live_streamer.py) ─
LABELS = [
    "FEMALE_GENITALIA_COVERED",
    "FACE_FEMALE",
    "BUTTOCKS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_BREAST_EXPOSED",
    "ANUS_EXPOSED",
    "FEET_EXPOSED",
    "BELLY_COVERED",
    "FEET_COVERED",
    "ARMPITS_COVERED",
    "ARMPITS_EXPOSED",
    "FACE_MALE",
    "BELLY_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "ANUS_COVERED",
    "FEMALE_BREAST_COVERED",
    "BUTTOCKS_COVERED",
]

RISK_LEVELS = {
    "CRITICAL": {"FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED"},
    "HIGH":     {"FEMALE_BREAST_EXPOSED", "ANUS_EXPOSED"},
    "MODERATE": {"BUTTOCKS_EXPOSED"},
    "LOW":      {"MALE_BREAST_EXPOSED", "BELLY_EXPOSED", "ARMPITS_EXPOSED", "FEET_EXPOSED"},
    "SAFE":     {
        "FACE_FEMALE", "FACE_MALE", "FEMALE_GENITALIA_COVERED",
        "BELLY_COVERED", "FEET_COVERED", "ARMPITS_COVERED", "ANUS_COVERED",
        "FEMALE_BREAST_COVERED", "BUTTOCKS_COVERED",
    },
}

RISK_PRIORITY = ["SAFE", "LOW", "MODERATE", "HIGH", "CRITICAL"]


def label_risk(label: str) -> str:
    for level, labels in RISK_LEVELS.items():
        if label in labels:
            return level
    return "UNKNOWN"


def risk_index(risk: str) -> int:
    try:
        return RISK_PRIORITY.index(risk)
    except ValueError:
        return 0


# ─── Preprocessing / postprocessing (identical to video.py) ──────────────────

def _preprocess(frame, target_size=320):
    h, w = frame.shape[:2]
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    aspect = w / h
    if h > w:
        nh = target_size
        nw = int(round(target_size * aspect))
    else:
        nw = target_size
        nh = int(round(target_size / aspect))

    resize_factor = math.sqrt((w ** 2 + h ** 2) / (nw ** 2 + nh ** 2))
    img = cv2.resize(img, (nw, nh))

    pad_x = target_size - nw
    pad_y = target_size - nh
    pad_top  = int(np.floor(pad_y) / 2)
    pad_left = int(np.floor(pad_x) / 2)
    pad_bottom = pad_y - pad_top
    pad_right  = pad_x - pad_left

    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (target_size, target_size))

    data = img.astype("float32") / 255.0
    data = np.transpose(data, (2, 0, 1))
    data = np.expand_dims(data, axis=0)
    return data, resize_factor, pad_left, pad_top


def _postprocess(outputs, resize_factor, pad_left, pad_top, threshold=0.25):
    outputs = np.transpose(np.squeeze(outputs[0]))
    rows = outputs.shape[0]
    boxes, scores, class_ids = [], [], []

    for i in range(rows):
        cls_scores = outputs[i][4:]
        max_score  = float(np.amax(cls_scores))
        if max_score >= threshold:
            class_id = int(np.argmax(cls_scores))
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left   = int(round((x - w * 0.5 - pad_left) * resize_factor))
            top    = int(round((y - h * 0.5 - pad_top)  * resize_factor))
            width  = int(round(w * resize_factor))
            height = int(round(h * resize_factor))
            boxes.append([left, top, width, height])
            scores.append(max_score)
            class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45)
    detections = []
    for i in indices:
        label = LABELS[class_ids[i]] if class_ids[i] < len(LABELS) else "UNKNOWN"
        detections.append({
            "class":    label,
            "score":    scores[i],
            "box":      boxes[i],
            "risk":     label_risk(label),
        })
    return detections


def _ensure_opset15(model_path: str) -> str:
    base, ext = os.path.splitext(model_path)
    conv_path = f"{base}_opset15{ext}"
    if not os.path.exists(conv_path):
        model     = onnx.load(model_path)
        converted = version_converter.convert_version(model, 15)
        onnx.save(converted, conv_path)
    return conv_path


# ─── Main Trimmer Class ───────────────────────────────────────────────────────

class VideoTrimmer:
    """
    Analyses a video and trims out obscene/unsafe scene segments.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    model_dir : str, optional
        Directory containing best.onnx (defaults to ./Models).
    frame_skip : int
        Analyse every Nth frame (default=2 for speed; 1 = every frame).
    min_risk : str
        Minimum risk level to flag as unsafe.
        One of SAFE / LOW / MODERATE / HIGH / CRITICAL.
    scene_gap_sec : float
        Two unsafe segments closer than this (seconds) are merged into one scene.
    scene_pad_sec : float
        Extra seconds to add before/after each unsafe scene boundary.
    threshold : float
        Detection confidence threshold (0–1).
    """

    def __init__(
        self,
        video_path: str,
        model_dir: str = None,
        frame_skip: int = 2,
        min_risk: str = "MODERATE",
        scene_gap_sec: float = 0.5,
        scene_pad_sec: float = 0.3,
        threshold: float = 0.25,
    ):
        self.video_path    = video_path
        self.frame_skip    = max(1, frame_skip)
        self.min_risk      = min_risk
        self.scene_gap_sec = scene_gap_sec
        self.scene_pad_sec = scene_pad_sec
        self.threshold     = threshold

        # Progress tracking (for UI polling)
        self.progress        = 0.0   # 0.0 – 100.0
        self.status          = "idle"  # idle | analysing | trimming | done | error
        self.status_message  = ""
        self._lock           = threading.Lock()

        # Results
        self.frame_results   = []   # [{frame, time, risk, detections}, ...]
        self.unsafe_segments = []   # [(start_sec, end_sec, max_risk), ...]
        self.safe_segments   = []   # [(start_sec, end_sec), ...]
        self.total_duration  = 0.0
        self.trim_output     = None

        # Load model
        self.session    = None
        self.input_name = None
        self.input_size = 320

        if not ONNX_OK:
            raise RuntimeError("onnxruntime is required. Install with: pip install onnxruntime onnx")

        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")

        orig_model = os.path.join(model_dir, "best.onnx")
        if not os.path.exists(orig_model):
            raise FileNotFoundError(f"Model not found: {orig_model}")

        model_path = _ensure_opset15(orig_model)

        providers = onnxruntime.get_available_providers()
        self.session = onnxruntime.InferenceSession(model_path, providers=providers)
        inp = self.session.get_inputs()[0]
        self.input_name = inp.name
        self.input_size = inp.shape[2]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _is_unsafe(self, detections) -> bool:
        """Return True if any detection meets or exceeds self.min_risk."""
        min_idx = risk_index(self.min_risk)
        for d in detections:
            if risk_index(d["risk"]) >= min_idx:
                return True
        return False

    def _detect(self, frame):
        data, rf, pl, pt = _preprocess(frame, self.input_size)
        outputs = self.session.run(None, {self.input_name: data})
        return _postprocess(outputs, rf, pl, pt, self.threshold)

    def _merge_segments(self, raw_unsafe_times):
        """
        Given a sorted list of (time_sec,) floats that are unsafe,
        group them into segments and merge segments closer than scene_gap_sec.
        Applies scene_pad_sec padding and clamps to [0, duration].
        Returns [(start, end, risk_placeholder), ...]
        """
        if not raw_unsafe_times:
            return []

        # Build raw segments from consecutive unsafe timestamps
        segments = []
        seg_start = raw_unsafe_times[0]
        seg_end   = raw_unsafe_times[0]

        for t in raw_unsafe_times[1:]:
            if t - seg_end <= self.scene_gap_sec:
                seg_end = t
            else:
                segments.append((seg_start, seg_end))
                seg_start = t
                seg_end   = t
        segments.append((seg_start, seg_end))

        # Apply padding + clamp
        padded = []
        dur = self.total_duration
        for s, e in segments:
            ps = max(0.0, s - self.scene_pad_sec)
            pe = min(dur, e + self.scene_pad_sec)
            padded.append((round(ps, 3), round(pe, 3)))

        # Merge overlapping after padding
        merged = [padded[0]]
        for s, e in padded[1:]:
            if s <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
            else:
                merged.append((s, e))

        return [(s, e, "UNSAFE") for s, e in merged]

    def _compute_safe_segments(self):
        """Invert unsafe_segments to get safe (keep) segments."""
        safe = []
        cursor = 0.0
        for s, e, _ in self.unsafe_segments:
            if cursor < s:
                safe.append((round(cursor, 3), round(s, 3)))
            cursor = e
        if cursor < self.total_duration:
            safe.append((round(cursor, 3), round(self.total_duration, 3)))
        self.safe_segments = [seg for seg in safe if (seg[1] - seg[0]) > 0.05]

    def _update_progress(self, pct, message=""):
        with self._lock:
            self.progress       = round(pct, 1)
            self.status_message = message

    # ── Public API ────────────────────────────────────────────────────────────

    def analyze(self, callback=None):
        """
        Run frame-by-frame analysis.

        Parameters
        ----------
        callback : callable(progress_pct, message), optional
            Called periodically with progress updates.

        Returns
        -------
        dict with keys:
            segments      — list of {start, end, risk} dicts for ALL segments
            unsafe_count  — number of unsafe scenes found
            safe_duration — total safe playback seconds
            total_duration
        """
        self.status = "analysing"
        self._update_progress(0, "Opening video…")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            self.status = "error"
            raise IOError(f"Cannot open video: {self.video_path}")

        fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.total_duration = total_frames / fps

        raw_unsafe_times = []
        self.frame_results = []

        frame_idx = 0
        analysed  = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t = frame_idx / fps
            frame_idx += 1

            # Only analyse every Nth frame for speed
            if frame_idx % self.frame_skip != 0:
                continue

            detections = self._detect(frame)
            unsafe     = self._is_unsafe(detections)
            max_risk   = max(
                (d["risk"] for d in detections), key=risk_index, default="SAFE"
            )

            self.frame_results.append({
                "frame":      frame_idx,
                "time":       round(t, 3),
                "risk":       max_risk,
                "unsafe":     unsafe,
                "detections": len(detections),
            })

            if unsafe:
                raw_unsafe_times.append(t)

            analysed += 1
            pct = min(99.0, (frame_idx / max(total_frames, 1)) * 100)
            self._update_progress(pct, f"Analysing frame {frame_idx}/{total_frames}…")
            if callback:
                callback(pct, f"Frame {frame_idx}/{total_frames}")

        cap.release()

        # Build scene segments
        self.unsafe_segments = self._merge_segments(raw_unsafe_times)
        self._compute_safe_segments()

        self._update_progress(100, "Analysis complete")
        self.status = "done"

        # Build combined timeline for UI
        all_segments = []
        for s, e in self.safe_segments:
            all_segments.append({"start": s, "end": e, "type": "safe"})
        for s, e, r in self.unsafe_segments:
            all_segments.append({"start": s, "end": e, "type": "unsafe", "risk": r})
        all_segments.sort(key=lambda x: x["start"])

        safe_dur = sum(e - s for s, e in self.safe_segments)

        return {
            "segments":       all_segments,
            "unsafe_count":   len(self.unsafe_segments),
            "safe_duration":  round(safe_dur, 2),
            "total_duration": round(self.total_duration, 2),
            "fps":            fps,
            "frames_analysed": analysed,
        }

    def trim(self, output_path: str = None, callback=None) -> str:
        """
        Trim unsafe scenes from the video using FFmpeg concat demuxer.

        Parameters
        ----------
        output_path : str, optional
            Where to save the trimmed video. Auto-generated if None.
        callback : callable(progress_pct, message), optional

        Returns
        -------
        str — absolute path to the trimmed output video.
        """
        if not self.safe_segments:
            raise RuntimeError("No analysis results. Call analyze() first.")

        self.status = "trimming"
        self._update_progress(0, "Preparing trim…")

        if output_path is None:
            base = os.path.splitext(os.path.basename(self.video_path))[0]
            out_dir = os.path.join(os.path.dirname(self.video_path), "video_outputs")
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"{base}_trimmed.mp4")

        # Build FFmpeg filter_complex with select+aselect to cut segments
        # We use the concat demuxer approach (most reliable for variable-length cuts)

        # Step 1: Write a concat list of safe segment clips
        tmp_dir = tempfile.mkdtemp(prefix="safevision_trim_")
        segment_files = []

        for idx, (start, end) in enumerate(self.safe_segments):
            duration = end - start
            if duration <= 0:
                continue
            seg_file = os.path.join(tmp_dir, f"seg_{idx:04d}.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i",  self.video_path,
                "-t",  str(duration),
                "-c:v", "libx264",
                "-c:a", "aac",
                "-avoid_negative_ts", "1",
                seg_file,
            ]
            self._update_progress(
                (idx / max(len(self.safe_segments), 1)) * 60,
                f"Cutting segment {idx+1}/{len(self.safe_segments)}…",
            )
            if callback:
                callback(self.progress, self.status_message)

            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0 and os.path.exists(seg_file):
                segment_files.append(seg_file)
            else:
                print(f"[VideoTrimmer] WARNING: segment {idx} failed:\n{result.stderr.decode()}")

        if not segment_files:
            self.status = "error"
            raise RuntimeError("FFmpeg failed to produce any safe segments.")

        # Step 2: Write concat list
        concat_file = os.path.join(tmp_dir, "concat_list.txt")
        with open(concat_file, "w") as f:
            for sf in segment_files:
                f.write(f"file '{sf}'\n")

        # Step 3: Concatenate all segments into final output
        self._update_progress(70, "Concatenating segments…")
        if callback:
            callback(70, "Concatenating…")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True)

        if result.returncode != 0:
            # Fallback: re-encode concat
            cmd[-3:-1] = ["-c:v", "libx264", "-c:a", "aac"]
            result = subprocess.run(cmd, capture_output=True)

        # Cleanup temp files
        try:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

        if result.returncode != 0:
            self.status = "error"
            raise RuntimeError(
                f"FFmpeg concat failed:\n{result.stderr.decode()}"
            )

        self._update_progress(100, "Trim complete!")
        if callback:
            callback(100, "Trim complete!")

        self.status = "done"
        self.trim_output = output_path
        return output_path

    def get_progress(self) -> dict:
        """Thread-safe progress snapshot for UI polling."""
        with self._lock:
            return {
                "progress": self.progress,
                "status":   self.status,
                "message":  self.status_message,
                "output":   self.trim_output,
            }

    def timeline_json(self) -> list:
        """Return the full per-frame analysis as a JSON-serialisable list."""
        return self.frame_results
