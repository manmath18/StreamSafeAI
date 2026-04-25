"""
Enhanced Computer Vision Based Detector
Uses OpenCV face detection, skin analysis, motion heuristics, and
a pre-render/ahead-of-time frame pipeline for faster streaming decisions.
"""
import cv2
import numpy as np
import threading
import queue
import time
from collections import deque
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DETECTOR_CONFIG = {
    # Face proximity
    "CLOSE_FACE_RATIO": 1.2,        # faces closer than 1.2× avg width → flag
    "FACE_ASPECT_WIDE": 1.35,       # wide merged-face box ratio → kissing heuristic
    # Skin — lowered thresholds to catch more scenes
    "SKIN_RATIO_HIGH": 0.22,        # >22% skin pixels → unsafe  (was 0.28)
    "SKIN_RATIO_MEDIUM": 0.13,      # 13-22% skin → add partial score  (was 0.18)
    # Brightness
    "DARK_BRIGHTNESS": 80,          # below this with people → flag
    # Red tones
    "RED_RATIO_THRESH": 0.20,       # red/pink concentration threshold
    # Motion blur
    "BLUR_VAR_THRESH": 90,          # Laplacian variance below this → high blur
    # Scoring weights for multi-signal fusion
    # Boosted FACE (kissing) and SKIN weights for better recall
    "W_FACE": 0.55,   # was 0.40
    "W_SKIN": 0.30,   # was 0.30
    "W_MOTION": 0.08, # was 0.15
    "W_COLOR": 0.07,  # was 0.15
    # Temporal EWM
    "EWM_ALPHA": 0.35,              # exponential weight for latest frame
    "SCORE_UNSAFE_THRESH": 0.45,    # EWM score ≥ this → unsafe (was 0.55)
    # Pre-render pipeline
    "PREFETCH_QUEUE_SIZE": 8,       # frames to buffer ahead of playback
    "PREFETCH_DOWNSCALE": 0.5,      # downscale factor for pre-analysis pass
    # Cooldown bridges (frames)
    "KISS_COOLDOWN": 8,   # was 6
    "SKIN_COOLDOWN": 6,   # was 4
}


# ---------------------------------------------------------------------------
# HSV Skin Detection Helper (vectorised)
# ---------------------------------------------------------------------------
_LOWER_SKIN = np.array([0, 20, 70], dtype=np.uint8)
_UPPER_SKIN = np.array([20, 255, 255], dtype=np.uint8)

_LOWER_RED1 = np.array([0,   50, 50], dtype=np.uint8)
_UPPER_RED1 = np.array([10, 255, 255], dtype=np.uint8)
_LOWER_RED2 = np.array([170, 50, 50], dtype=np.uint8)
_UPPER_RED2 = np.array([180, 255, 255], dtype=np.uint8)


def _skin_ratio(hsv: np.ndarray) -> float:
    mask = cv2.inRange(hsv, _LOWER_SKIN, _UPPER_SKIN)
    return float(np.count_nonzero(mask)) / (hsv.shape[0] * hsv.shape[1])


def _torso_skin_ratio(hsv: np.ndarray, faces: list) -> float:
    """Check skin density directly underneath detected faces (chest/torso area)."""
    if not faces:
        return 0.0
        
    mask = cv2.inRange(hsv, _LOWER_SKIN, _UPPER_SKIN)
    max_torso_ratio = 0.0
    frame_height, frame_width = hsv.shape[:2]
    
    for x, y, w, h in faces:
        # Define torso box: below the face, slightly wider, spanning down 2.5x face height
        tx1 = max(0, x - int(w * 0.25))
        ty1 = min(frame_height, y + int(h * 0.9)) # start slightly above chin
        tx2 = min(frame_width, x + int(w * 1.25))
        ty2 = min(frame_height, y + int(h * 3.5))
        
        torso_area = (tx2 - tx1) * (ty2 - ty1)
        if torso_area > 0:
            torso_mask = mask[ty1:ty2, tx1:tx2]
            skin_pixels = np.count_nonzero(torso_mask)
            ratio = skin_pixels / float(torso_area)
            max_torso_ratio = max(max_torso_ratio, ratio)
            
    return max_torso_ratio


def _red_ratio(hsv: np.ndarray) -> float:
    m1 = cv2.inRange(hsv, _LOWER_RED1, _UPPER_RED1)
    m2 = cv2.inRange(hsv, _LOWER_RED2, _UPPER_RED2)
    combined = cv2.bitwise_or(m1, m2)
    return float(np.count_nonzero(combined)) / (hsv.shape[0] * hsv.shape[1])


def _blur_score(gray: np.ndarray) -> float:
    """Laplacian variance → 0 (sharp) to 1 (very blurry)."""
    var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(max(0.0, 1.0 - min(var / DETECTOR_CONFIG["BLUR_VAR_THRESH"], 1.0)))


def _brightness(gray: np.ndarray) -> float:
    return float(np.mean(gray))


# ---------------------------------------------------------------------------
# Face detector (singleton cascades to avoid reload overhead)
# ---------------------------------------------------------------------------
_frontal_cascade: Optional[cv2.CascadeClassifier] = None
_profile_cascade: Optional[cv2.CascadeClassifier] = None
_cascade_lock = threading.Lock()


def _get_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
    global _frontal_cascade, _profile_cascade
    if _frontal_cascade is None:
        with _cascade_lock:
            if _frontal_cascade is None:
                _frontal_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                _profile_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_profileface.xml"
                )
    return _frontal_cascade, _profile_cascade


def _detect_faces(gray: np.ndarray) -> Tuple[List, List]:
    frontal, profile = _get_cascades()
    faces_f = frontal.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28)
    )
    faces_p = profile.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(28, 28)
    )
    f = list(faces_f) if len(faces_f) > 0 else []
    p = list(faces_p) if len(faces_p) > 0 else []
    return f, p


def _check_close_faces(faces: list, frame_height: int) -> bool:
    """Return True if any two faces overlap or are dangerously close (kissing)."""
    if len(faces) < 2:
        return False
        
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            x1, y1, w1, h1 = faces[i]
            x2, y2, w2, h2 = faces[j]
            
            # Center points
            cx1, cy1 = x1 + w1 / 2.0, y1 + h1 / 2.0
            cx2, cy2 = x2 + w2 / 2.0, y2 + h2 / 2.0
            
            # 1. Proximity check (relative to face size)
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
            avg_w = (w1 + w2) / 2.0
            if dist < avg_w * DETECTOR_CONFIG["CLOSE_FACE_RATIO"]:
                return True
                
            # 2. Bounding Box Intersection (IoU overlap)
            # If face boxes physically intersect, they are extremely close
            ix1 = max(x1, x2)
            iy1 = max(y1, y2)
            ix2 = min(x1 + w1, x2 + w2)
            iy2 = min(y1 + h1, y2 + h2)
            
            if ix1 < ix2 and iy1 < iy2:
                # Calculate intersection area ratio
                inter_area = (ix2 - ix1) * (iy2 - iy1)
                box1_area = w1 * h1
                box2_area = w2 * h2
                # If they overlap by more than 10% of the smallest face's area -> kissing
                if inter_area > 0.10 * min(box1_area, box2_area):
                    return True
                    
    return False


def _check_wide_face(faces: list) -> bool:
    """Wide merged bounding box → two faces very close (kissing heuristic)."""
    for x, y, w, h in faces:
        if h > 0 and (w / h) > DETECTOR_CONFIG["FACE_ASPECT_WIDE"]:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-frame analysis (pure function, thread-safe)
# ---------------------------------------------------------------------------
def analyse_frame_cv(frame: np.ndarray, frame_number: int = 0) -> Dict:
    """
    Multi-signal CV analysis of a single frame.
    Returns a rich result dict including a raw [0,1] threat score.
    """
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Signal extraction ---
    faces_f, faces_p = _detect_faces(gray)
    all_faces = faces_f + faces_p
    n_faces = len(all_faces)

    sr = _skin_ratio(hsv)
    torso_sr = _torso_skin_ratio(hsv, all_faces)
    rr = _red_ratio(hsv)
    bv = _brightness(gray)
    blur = _blur_score(gray)

    # --- Rule-based flags ---
    is_close = _check_close_faces(all_faces, h)
    is_wide = _check_wide_face(all_faces)
    face_profile_mix = len(faces_f) >= 1 and len(faces_p) >= 1
    high_skin = sr >= DETECTOR_CONFIG["SKIN_RATIO_HIGH"]
    medium_skin = sr >= DETECTOR_CONFIG["SKIN_RATIO_MEDIUM"]
    dark_scene = bv < DETECTOR_CONFIG["DARK_BRIGHTNESS"] and n_faces >= 1
    red_with_face = rr > DETECTOR_CONFIG["RED_RATIO_THRESH"] and n_faces >= 1
    heavy_blur = blur > 0.6 and n_faces >= 1

    # --- Compute per-signal partial scores (0–1) ---
    # Face signal
    face_score = 0.0
    face_tags = []
    if is_close or is_wide:
        face_score = 0.95
        face_tags.append("close_faces_kissing")
    elif face_profile_mix:
        face_score = 0.85
        face_tags.append("face_profile_mix")
    elif n_faces >= 3:
        face_score = 0.55
        face_tags.append("multiple_faces")

    # Skin signal
    skin_score = 0.0
    skin_tags = []
    
    # Torso skin (bare chest/shoulders) is the strongest indicator
    if torso_sr > 0.45:
        skin_score = 0.95
        skin_tags.append("exposed_torso")
    elif torso_sr > 0.25:
        skin_score = max(skin_score, 0.70)
        skin_tags.append("partial_torso_skin")
        
    # Global skin falls back
    if high_skin:
        skin_score = max(skin_score, 0.90)
        skin_tags.append("high_skin_exposure")
    elif medium_skin and n_faces >= 1:
        skin_score = max(skin_score, 0.55)
        skin_tags.append("medium_skin_exposure")

    # Motion/blur signal
    motion_score = 0.0
    motion_tags = []
    if heavy_blur:
        motion_score = blur  # 0.6–1.0 range
        motion_tags.append("motion_blur_with_people")

    # Color/scene signal
    color_score = 0.0
    color_tags = []
    if dark_scene:
        color_score = max(color_score, 0.65)
        color_tags.append("dark_scene_with_people")
    if red_with_face:
        color_score = max(color_score, 0.60 + rr)
        color_tags.append("red_tones_with_faces")

    # Weighted fusion
    cfg = DETECTOR_CONFIG
    raw_score = (
        cfg["W_FACE"] * face_score
        + cfg["W_SKIN"] * skin_score
        + cfg["W_MOTION"] * motion_score
        + cfg["W_COLOR"] * min(color_score, 1.0)
    )

    detections = face_tags + skin_tags + motion_tags + color_tags

    return {
        "frame_number": frame_number,
        "raw_score": float(np.clip(raw_score, 0.0, 1.0)),
        "face_score": face_score,
        "skin_score": skin_score,
        "motion_score": motion_score,
        "color_score": min(color_score, 1.0),
        "n_faces": n_faces,
        "skin_ratio": sr,
        "brightness": bv,
        "blur": blur,
        "detections": detections,
        # legacy keys expected by SafeVisionEngine
        "is_safe": raw_score < cfg["SCORE_UNSAFE_THRESH"],
        "confidence": float(np.clip(raw_score, 0.0, 1.0)),
        "reason": "; ".join(detections) if detections else "clean",
    }


# ---------------------------------------------------------------------------
# SimpleDetector – drop-in enhanced class
# ---------------------------------------------------------------------------
class SimpleDetector:
    """
    Enhanced offline detector using computer vision.

    Key improvements over the original:
    • Pre-render pipeline: frames are downscaled and buffered ahead of time
      so streaming decisions are made faster than real-time.
    • Parallel worker thread for analysis so the caller never blocks.
    • Exponential weighted smoothing to prevent flickering decisions.
    • Cooldown bridges to hold detections across face-dropout gaps.
    • Rich multi-signal score (face proximity + skin + motion + color).
    """

    def __init__(self, prefetch_size: int = DETECTOR_CONFIG["PREFETCH_QUEUE_SIZE"]):
        _get_cascades()  # warm up singleton cascades
        self.prefetch_size = prefetch_size

        # EWM smoother state
        self._ewm_score: float = 0.0
        self._alpha = DETECTOR_CONFIG["EWM_ALPHA"]
        self._unsafe_thresh = DETECTOR_CONFIG["SCORE_UNSAFE_THRESH"]

        # Cooldown bridges
        self._kiss_cd = 0
        self._skin_cd = 0

        # Internal async pipeline (for streaming mode)
        self._in_q: queue.Queue = queue.Queue(maxsize=prefetch_size)
        self._out_q: queue.Queue = queue.Queue(maxsize=prefetch_size)
        self._worker: Optional[threading.Thread] = None
        self._streaming = False

        print("✓ SimpleDetector initialized (offline, CV-only, prefetch pipeline ready)")

    # ------------------------------------------------------------------
    # Synchronous API (batch / single-frame use)
    # ------------------------------------------------------------------

    def analyze_frame(self, frame: np.ndarray, frame_number: int = 0) -> Dict:
        """
        Synchronously analyse a single frame.
        Applies cooldown bridges and EWM smoothing.
        """
        result = analyse_frame_cv(frame, frame_number)
        result = self._apply_cooldown_bridge(result)
        result = self._apply_ewm(result)
        return result

    # ------------------------------------------------------------------
    # Streaming / pre-render pipeline API
    # ------------------------------------------------------------------

    def start_stream(self):
        """
        Start the background worker thread.
        Call submit_frame() to push frames and get_result() to pull results.
        """
        if self._streaming:
            return
        self._streaming = True
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="SimpleDetector-Worker"
        )
        self._worker.start()
        print("✓ SimpleDetector streaming pipeline started")

    def stop_stream(self):
        """Gracefully shut down the background worker."""
        self._streaming = False
        # Unblock worker
        try:
            self._in_q.put_nowait(None)
        except queue.Full:
            pass
        if self._worker:
            self._worker.join(timeout=2.0)
        self._worker = None
        print("✓ SimpleDetector streaming pipeline stopped")

    def submit_frame(
        self, frame: np.ndarray, frame_number: int = 0, timeout: float = 0.05
    ) -> bool:
        """
        Push a frame into the pre-render pipeline (non-blocking).
        Returns True if accepted, False if the buffer is full.

        The pipeline downscales the frame before analysis so the worker
        runs ahead of the player. Full-res censoring can still be applied
        by the caller using the returned detections.
        """
        scale = DETECTOR_CONFIG["PREFETCH_DOWNSCALE"]
        small = cv2.resize(
            frame,
            (0, 0),
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_LINEAR,
        )
        try:
            self._in_q.put_nowait((small, frame_number))
            return True
        except queue.Full:
            return False

    def get_result(self, timeout: float = 0.05) -> Optional[Dict]:
        """
        Pull the next analysis result from the pipeline (non-blocking).
        Returns None if no result is ready yet.
        """
        try:
            return self._out_q.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_all_pending_results(self) -> List[Dict]:
        """Drain all ready results from the output queue."""
        results = []
        while True:
            try:
                results.append(self._out_q.get_nowait())
            except queue.Empty:
                break
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _worker_loop(self):
        """Background thread: pull frames, analyse, push results."""
        while self._streaming:
            try:
                item = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                break  # shutdown sentinel
            small_frame, frame_number = item
            result = analyse_frame_cv(small_frame, frame_number)
            result = self._apply_cooldown_bridge(result)
            result = self._apply_ewm(result)
            try:
                self._out_q.put(result, timeout=0.05)
            except queue.Full:
                pass  # drop oldest if consumer can't keep up

    def _apply_cooldown_bridge(self, result: Dict) -> Dict:
        """Hold flagged state across face-detection dropout gaps."""
        face_tags = [t for t in result["detections"] if "face" in t or "kiss" in t]
        skin_tags = [t for t in result["detections"] if "skin" in t]

        if face_tags:
            self._kiss_cd = DETECTOR_CONFIG["KISS_COOLDOWN"]
        elif self._kiss_cd > 0:
            self._kiss_cd -= 1
            result["face_score"] = max(result["face_score"], 0.85)
            if "kiss_cooldown_bridge" not in result["detections"]:
                result["detections"].append("kiss_cooldown_bridge")

        if skin_tags:
            self._skin_cd = DETECTOR_CONFIG["SKIN_COOLDOWN"]
        elif self._skin_cd > 0:
            self._skin_cd -= 1
            result["skin_score"] = max(result["skin_score"], 0.80)
            if "skin_cooldown_bridge" not in result["detections"]:
                result["detections"].append("skin_cooldown_bridge")

        # Recompute fused score after cooldown adjustments
        cfg = DETECTOR_CONFIG
        raw = (
            cfg["W_FACE"] * result["face_score"]
            + cfg["W_SKIN"] * result["skin_score"]
            + cfg["W_MOTION"] * result["motion_score"]
            + cfg["W_COLOR"] * result["color_score"]
        )
        result["raw_score"] = float(np.clip(raw, 0.0, 1.0))
        return result

    def _apply_ewm(self, result: Dict) -> Dict:
        """Apply exponential weighted mean for temporal smoothing."""
        a = self._alpha
        self._ewm_score = a * result["raw_score"] + (1.0 - a) * self._ewm_score
        result["ewm_score"] = self._ewm_score
        result["is_safe"] = self._ewm_score < self._unsafe_thresh
        result["confidence"] = self._ewm_score
        return result

    def reset(self):
        """Reset temporal state (call when seeking/rewinding)."""
        self._ewm_score = 0.0
        self._kiss_cd = 0
        self._skin_cd = 0


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def test_simple_detector():
    print("Testing Enhanced SimpleDetector …")
    det = SimpleDetector()

    # Single-frame sync test
    frame = np.random.randint(0, 200, (480, 640, 3), dtype=np.uint8)
    r = det.analyze_frame(frame, frame_number=1)
    print(f"  Sync result: ewm={r['ewm_score']:.3f}  safe={r['is_safe']}  "
          f"tags={r['detections']}")

    # Async pipeline test
    det.start_stream()
    for i in range(10):
        ok = det.submit_frame(frame, frame_number=i)
        time.sleep(0.02)
    time.sleep(0.3)
    results = det.get_all_pending_results()
    print(f"  Pipeline produced {len(results)} results")
    det.stop_stream()
    print("✓ SimpleDetector tests passed!")


if __name__ == "__main__":
    test_simple_detector()
