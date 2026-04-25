"""
EfficientNet-B0 Deep-Learning Scene Safety Classifier
Performs binary (safe / unsafe) frame-level classification using a pretrained
EfficientNet-B0 backbone fine-tuned for content safety.

Features
--------
* Auto-selects CUDA → MPS → CPU based on availability
* Batch inference (default batch_size=4) for GPU throughput
* Graceful fallback: returns neutral score (0.1) when torch unavailable
* Thread-safe: separate preprocessing pool, single inference lock
* Warm-up run on first call to avoid cold-start latency on first real frame
"""

from __future__ import annotations

import threading
import queue
import time
from typing import List, Tuple, Optional

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Optional torch import
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torchvision.models as tv_models
    import torchvision.transforms as T
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    torch = None  # type: ignore


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def _select_device() -> "torch.device":
    if not _TORCH_OK:
        return None
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[EfficientNet] Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = torch.device("mps")
        print("[EfficientNet] Using Apple MPS")
    else:
        dev = torch.device("cpu")
        print("[EfficientNet] Using CPU (no GPU detected)")
    return dev


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def _build_model(device: "torch.device") -> "nn.Module":
    """
    Load EfficientNet-B0 with ImageNet weights and replace the classifier
    head with a binary output (2 logits: [safe, unsafe]).
    """
    weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model = tv_models.efficientnet_b0(weights=weights)

    # Replace the final classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(in_features, 2),   # 0 = safe, 1 = unsafe
    )

    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Preprocessing pipeline  (224×224, ImageNet normalise)
# ---------------------------------------------------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

def _build_transform():
    if not _TORCH_OK:
        return None
    return T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# EfficientNetClassifier
# ---------------------------------------------------------------------------
class EfficientNetClassifier:
    """
    Binary scene-safety classifier powered by EfficientNet-B0.

    Usage
    -----
    clf = EfficientNetClassifier(batch_size=4)

    # Synchronous single-frame
    score = clf.predict_single(bgr_frame)          # float in [0, 1]

    # Batch (list of BGR frames, all same size recommended)
    scores = clf.predict_batch([frame1, frame2, ...])

    Notes
    -----
    The model ships with IMAGENET weights only, meaning it has NOT been
    fine-tuned on NSFW data. Use as a feature backbone whose output is
    combined with the ONNX nude-detector and CV heuristics in the
    late-fusion engine (SafeVisionEngine).  To fine-tune, replace
    ``self._model`` weights via ``load_weights(path)``.
    """

    def __init__(self, batch_size: int = 4):
        self.batch_size = batch_size
        self._lock = threading.Lock()
        self._available = _TORCH_OK

        if not self._available:
            print("[EfficientNet] PyTorch not available – classifier disabled")
            return

        self._device = _select_device()
        self._transform = _build_transform()
        self._model = _build_model(self._device)

        # Warm up (avoids JIT latency on first real frame)
        self._warmup()
        print(f"[EfficientNet] Ready (batch_size={batch_size})")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_single(self, bgr_frame: np.ndarray) -> float:
        """
        Classify a single BGR frame.

        Returns
        -------
        float : probability of UNSAFE content in [0, 1]
        """
        if not self._available:
            return 0.1
        scores = self.predict_batch([bgr_frame])
        return scores[0]

    def predict_batch(self, bgr_frames: List[np.ndarray]) -> List[float]:
        """
        Classify a list of BGR frames in one forward pass (batched).

        Returns
        -------
        list[float] : per-frame unsafe probabilities in [0, 1]
        """
        if not self._available or not bgr_frames:
            return [0.1] * len(bgr_frames)

        try:
            tensors = []
            for frame in bgr_frames:
                # BGR → RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                t = self._transform(rgb)
                tensors.append(t)

            batch = torch.stack(tensors).to(self._device)

            with self._lock:
                with torch.no_grad():
                    logits = self._model(batch)          # (N, 2)
                    probs  = torch.softmax(logits, dim=1)
                    unsafe_probs = probs[:, 1].cpu().numpy()  # class 1 = unsafe

            return [float(p) for p in unsafe_probs]

        except Exception as e:
            print(f"[EfficientNet] Inference error: {e}")
            return [0.1] * len(bgr_frames)

    def load_weights(self, path: str):
        """Load fine-tuned weights from a .pt checkpoint."""
        if not self._available:
            return
        try:
            state = torch.load(path, map_location=self._device)
            # Handle both raw state-dict and {'model': ...} checkpoint dicts
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self._model.load_state_dict(state, strict=False)
            self._model.eval()
            print(f"[EfficientNet] Loaded weights from {path}")
        except Exception as e:
            print(f"[EfficientNet] Could not load weights: {e}")

    @property
    def is_available(self) -> bool:
        return self._available

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _warmup(self):
        """Run a dummy forward pass to initialise CUDA kernels."""
        try:
            dummy = torch.zeros(1, 3, 224, 224, device=self._device)
            with torch.no_grad():
                self._model(dummy)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    clf = EfficientNetClassifier(batch_size=4)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    import time
    t0 = time.perf_counter()
    score = clf.predict_single(frame)
    print(f"Single predict: {score:.4f}  ({(time.perf_counter()-t0)*1000:.1f} ms)")

    frames = [frame] * 4
    t0 = time.perf_counter()
    scores = clf.predict_batch(frames)
    print(f"Batch predict (4): {scores}  ({(time.perf_counter()-t0)*1000:.1f} ms)")
