import cv2
import time
from PyQt6.QtCore import QThread, pyqtSignal

from temporal_engine.smoothing import TemporalSmoother
from skip_controller.smart_skip import SkipController
from live import LiveNudeDetector
import numpy as np

class SafeVisionEngine(QThread):
    stats_updated = pyqtSignal(float, float, bool) # score, fps, is_safe
    skip_requested = pyqtSignal(int, int) # start_ms, end_ms
    
    def __init__(self):
        super().__init__()
        self.video_path = None
        self.player_ref = None
        self.running = False
        
        self.smoother = TemporalSmoother(window_size=12, threshold=0.72)
        self.skip_controller = SkipController(cooldown_ms=3000)
        
        # Initialize the real ONNX-based detector
        self.detector = LiveNudeDetector()
        print("LiveNudeDetector initialized in SafeVisionEngine")

    def start_analysis(self, video_path, player_ref):
        self.video_path = video_path
        self.player_ref = player_ref
        self.running = True
        self.start()

    def stop_analysis(self):
        self.running = False
        self.wait()

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval_ms = (1.0 / (fps if fps > 0 else 30)) * 1000
        
        # We sample at ~5-10 FPS
        sample_interval_ms = 200  # 5 FPS
        
        last_process_time = time.time()
        
        while self.running and self.player_ref:
            if not self.player_ref.is_playing():
                time.sleep(0.1)
                continue
                
            current_ms = self.player_ref.get_time()
            if current_ms < 0:
                time.sleep(0.1)
                continue
                
            # Seek cap to current video player time
            cap.set(cv2.CAP_PROP_POS_MSEC, current_ms)
            ret, frame = cap.read()
            
            if not ret:
                time.sleep(0.1)
                continue
                
            # Simulate inference delay and get dummy score
            start_infer = time.time()
            
            # --- REAL INFERENCE CODE ---
            detections = self.detector.detect_frame(frame)
            
            # Check for generic unsafe content based on LiveNudeDetector logic
            has_unsafe = False
            max_unsafe_score = 0.0
            
            for d in detections:
                label = d["class"]
                severity = d.get("severity", "MODERATE")
                score = d.get("score", 0.0)
                
                # Check if it should be censored/marked unsafe
                if self.detector.should_apply_blur(label):
                    if severity in ['CRITICAL', 'HIGH']:
                        has_unsafe = True
                        max_unsafe_score = max(max_unsafe_score, score)
                    elif severity == 'MODERATE' and score >= 0.4:
                        has_unsafe = True
                        max_unsafe_score = max(max_unsafe_score, score)
                    elif severity == 'LOW' and score >= 0.6:
                        has_unsafe = True
                        max_unsafe_score = max(max_unsafe_score, score)

            # Kiss detection logic: Face proximity algorithm
            face_boxes = []
            for d in detections:
                if d["class"] in ["FACE_FEMALE", "FACE_MALE"] and d.get("score", 0.0) > 0.4:
                    face_boxes.append(d["box"])
            
            is_kiss = False
            if len(face_boxes) >= 2:
                for i in range(len(face_boxes)):
                    for j in range(i+1, len(face_boxes)):
                        x1, y1, w1, h1 = face_boxes[i]
                        x2, y2, w2, h2 = face_boxes[j]
                        c1x, c1y = x1 + w1/2.0, y1 + h1/2.0
                        c2x, c2y = x2 + w2/2.0, y2 + h2/2.0
                        dist = np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
                        avg_width = (w1 + w2) / 2.0
                        
                        # Distance closer than 0.85 of avg face width is considered a kiss/unsafe proximity
                        if dist < (avg_width * 0.85):
                            is_kiss = True
                            has_unsafe = True
                            max_unsafe_score = max(max_unsafe_score, 0.9)  # High confidence of unsafe proximity
                            break

            raw_score = max_unsafe_score if has_unsafe else 0.1
            
            # Temporal Smoothing
            is_unsafe_smoothed, avg_score = self.smoother.process(raw_score)
            
            infer_time = time.time() - start_infer
            calc_fps = 1.0 / (infer_time + 0.001)
            
            self.stats_updated.emit(avg_score, min(calc_fps, 30.0), not is_unsafe_smoothed)
            
            if is_unsafe_smoothed:
                # Ask skip controller what to do
                skip_target_ms = self.skip_controller.handle_unsafe(current_ms)
                if skip_target_ms is not None:
                    # Jump ahead 5 seconds when unsafe content detected
                    self.skip_requested.emit(current_ms, current_ms + 5000)
            
            # Optional: Sleep to not overload CPU if processing too fast
            time_to_sleep = (sample_interval_ms / 1000.0) - infer_time
            if time_to_sleep > 0:
                time.sleep(time_to_sleep)
            
        cap.release()
