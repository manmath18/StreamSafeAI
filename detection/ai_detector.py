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
        
        # Sliding window aggregator parameters
        self.score_history = []
        self.W = 15
        self.decision_tau = 0.60
        
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
                    if severity == 'CRITICAL':
                        has_unsafe = True
                        # Boost critical scores to ensure they trigger the temporal threshold
                        max_unsafe_score = max(max_unsafe_score, min(1.0, score + 0.40))
                    elif severity == 'HIGH':
                        has_unsafe = True
                        # Boost high severity scores slightly
                        max_unsafe_score = max(max_unsafe_score, min(1.0, score + 0.30))
                    elif severity == 'MODERATE' and score >= 0.4:
                        has_unsafe = True
                        max_unsafe_score = max(max_unsafe_score, score)
                    elif severity == 'LOW' and score >= 0.6:
                        has_unsafe = True
                        max_unsafe_score = max(max_unsafe_score, score)

            # --- Additional Detection Algorithms ---
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Face Detection Using Haar Cascades
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            faces_frontal = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
            faces_profile = profile_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
            
            faces = []
            if len(faces_frontal) > 0:
                faces.extend(faces_frontal)
            if len(faces_profile) > 0:
                faces.extend(faces_profile)
                
            # Combine faces detected by the ONNX model!
            for d in detections:
                if d["class"] in ["FACE_FEMALE", "FACE_MALE"] and d.get("score", 0.0) > 0.4:
                    faces.append(d["box"])
            
            # 2. Motion Blur Detection Using Laplacian Operator
            laplacian_var = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
            is_blurry = laplacian_var < 100.0  # Threshold indicating rapid motion
            
            # 3. Skin Exposure Detection Using HSV Color Space
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv_frame, lower_skin, upper_skin)
            skin_ratio = cv2.countNonZero(skin_mask) / (frame.shape[0] * frame.shape[1])
            is_high_skin = skin_ratio > 0.25  # Threshold for high skin exposure
            
            if is_high_skin:
                has_unsafe = True
                max_unsafe_score = max(max_unsafe_score, 0.85)
            
            # 4. Kissing and Intimate Interaction Detection (Face Proximity)
            frame_height = frame.shape[0]
            tau_f = frame_height * 0.25 # Increased Face-distance threshold proportional to frame height
            is_kiss = False
            
            if len(faces) >= 2:
                for i in range(len(faces)):
                    for j in range(i+1, len(faces)):
                        x1, y1, w1, h1 = faces[i]
                        x2, y2, w2, h2 = faces[j]
                        c1x, c1y = x1 + w1/2.0, y1 + h1/2.0
                        c2x, c2y = x2 + w2/2.0, y2 + h2/2.0
                        dist = np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
                        
                        # Use either strict frame proportion or bounding box size heuristics
                        if dist <= tau_f or dist <= ((w1 + w2) / 1.2):
                            is_kiss = True
                            has_unsafe = True
                            max_unsafe_score = max(max_unsafe_score, 0.95)
                            break
                    if is_kiss:
                        break

            # --- Multi-Modal Frame-Level Classification ---
            S_v = max_unsafe_score if has_unsafe else 0.1
            
            # Convert laplacian variance to a motion blur probability [0, 1]
            S_m = max(0.0, min(1.0, 50.0 / (laplacian_var + 1.0)))
            S_a = 0.0 # Audio not processed in this thread
            
            alpha = 0.85
            beta = 0.15
            gamma = 0.0
            
            # Late-fusion classifier final score
            S = (alpha * S_v) + (beta * S_m) + (gamma * S_a)
            
            # --- Temporal Decision Aggregation and Unsafe-Segment Detection ---
            self.score_history.append(S)
            if len(self.score_history) > self.W:
                self.score_history.pop(0)
                
            S_t = sum(self.score_history) / len(self.score_history)
            
            is_unsafe_smoothed = S_t >= self.decision_tau
            avg_score = S_t
            
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
