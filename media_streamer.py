#!/usr/bin/env python3
"""
Media Streamer (PyQt5 version)
Plays an uploaded video file and skips or blurs sensitive content automatically at runtime.
"""

import sys
import os
import cv2
import time
import argparse
import numpy as np

# Set environment variable to avoid Wayland/xcb plugin issues
os.environ["QT_QPA_PLATFORM"] = "xcb"

from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QFileDialog, QMessageBox, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt
from live import LiveNudeDetector

class VideoStreamer(QMainWindow):
    def __init__(self, video_path=None, mode="blur"):
        super().__init__()
        self.setWindowTitle(f"SafeVision Media Streamer - Mode: {mode.upper()}")
        self.mode = mode
        
        # Central Widget & Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.layout.addWidget(self.video_label)
        
        # Initialize the detector
        print("Initializing Live Nude Detector...")
        self.detector = LiveNudeDetector()
        print("Detector Initialized.")
        
        self.paused = False
        self.frame_count = 0
        
        # Determine video path
        if not video_path:
            video_path, _ = QFileDialog.getOpenFileName(
                self, "Select a Video File to Stream", "",
                "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*.*)"
            )
            
        if not video_path:
            QMessageBox.critical(self, "Error", "No video selected. Exiting.")
            sys.exit(1)
            
        self.video_path = video_path
        print(f"Loading video: {self.video_path}")
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", f"Could not open video: {self.video_path}")
            sys.exit(1)
            
        # Try to read first frame to get dimensions
        ret, frame = self.cap.read()
        if ret:
            h, w = frame.shape[:2]
            self.resize(w, h)
            # Re-seek back to frame 0
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Frame timer setup
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps or fps == 0:
            fps = 30.0
            
        self.frame_delay_ms = max(1, int(1000 / fps))
        print(f"Video FPS: {fps:.2f}, Frame Delay: {self.frame_delay_ms}ms")
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(self.frame_delay_ms)
        
    def update_frame(self):
        if self.paused:
            return
            
        ret, frame = self.cap.read()
        if not ret:
            print("End of video stream.")
            self.timer.stop()
            self.cap.release()
            return
            
        self.frame_count += 1
        
        # Run detection
        detections = self.detector.detect_frame(frame)
        
        has_sensitive_content = False
        for d in detections:
            label = d["class"]
            severity = d.get("severity", "MODERATE")
            score = d.get("score", 0.0)
            
            if self.detector.should_apply_blur(label):
                if severity in ['CRITICAL', 'HIGH']:
                    has_sensitive_content = True
                    break
                elif severity == 'MODERATE' and score >= 0.4:
                    has_sensitive_content = True
                    break
                elif severity == 'LOW' and score >= 0.6:
                    has_sensitive_content = True
                    break

        if self.mode.lower() == 'skip' and has_sensitive_content:
            display_frame = np.zeros_like(frame)
            text = "SENSITIVE CONTENT SKIPPED"
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_size = cv2.getTextSize(text, font, 1.5, 3)[0]
            text_x = (display_frame.shape[1] - text_size[0]) // 2
            text_y = (display_frame.shape[0] + text_size[1]) // 2
            cv2.putText(display_frame, text, (text_x, text_y), font, 1.5, (0, 0, 255), 3)
            cv2.putText(display_frame, f"Frame: {self.frame_count}", (10, 30), font, 0.7, (255, 255, 255), 2)
        else:
            display_frame = self.detector.apply_censoring(frame, detections)
            
        # Convert map format for PyQt display
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_frame.shape
        bytes_per_line = ch * w
        
        q_img = QImage(display_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(q_img))
        
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space:
            self.paused = not self.paused
            print(f"Video {'paused' if self.paused else 'resumed'}.")
        elif event.key() == Qt.Key_Q or event.key() == Qt.Key_Escape:
            print("Streamer closed by user.")
            self.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SafeVision Media Streamer")
    parser.add_argument("video", nargs="?", help="Path to the uploaded video file to stream")
    parser.add_argument("--mode", choices=["blur", "skip"], default="blur", 
                        help="Action when sensitive content is found (blur regions or skip entire frame)")
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    # Allows Ctrl+C to stop app easily
    import signal
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    streamer = VideoStreamer(video_path=args.video, mode=args.mode)
    streamer.show()
    
    # Run the Qt Event Loop
    sys.exit(app.exec_())
