from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QFrame
from PyQt6.QtCore import Qt
import os
import cv2

class RightSidebar(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 41, 59, 0.7);
                border-left: 1px solid #334155;
            }
            QLabel {
                border: none;
                color: #cbd5e1;
            }
        """)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(20)
        
        title = QLabel("AI Monitor Panel")
        title.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)
        
        # Status Badge
        self.status_badge = QLabel("SAFE")
        self.status_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_badge.setStyleSheet("""
            background-color: rgba(34, 197, 94, 0.2);
            color: #4ade80;
            padding: 10px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 16px;
            border: 1px solid rgba(34, 197, 94, 0.5);
        """)
        layout.addWidget(self.status_badge)
        
        # Risk Score
        layout.addWidget(QLabel("Current Risk Score"))
        self.risk_bar = QProgressBar()
        self.risk_bar.setMaximum(100)
        self.risk_bar.setValue(0)
        self.risk_bar.setTextVisible(True)
        self.risk_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #334155;
                border-radius: 4px;
                background-color: #0f172a;
                height: 12px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4ade80, stop:0.5 #facc15, stop:1 #ef4444);
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.risk_bar)
        
        # Hardware Stats
        self.fps_label = QLabel("FPS: --")
        layout.addWidget(self.fps_label)
        self.gpu_label = QLabel("GPU Usage: --%")
        layout.addWidget(self.gpu_label)
        
        # Diagnostics & Metadata
        layout.addWidget(self._create_separator())
        meta_title = QLabel("Diagnostics & Metadata")
        meta_title.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: bold;")
        layout.addWidget(meta_title)
        
        self.file_size_label = QLabel("Size: -- MB")
        self.resolution_label = QLabel("Resolution: -- x --")
        self.bitrate_label = QLabel("Bitrate: -- kbps")
        self.codec_label = QLabel("Video Codec: --")
        
        for lbl in [self.file_size_label, self.resolution_label, self.bitrate_label, self.codec_label]:
            lbl.setStyleSheet("color: #cbd5e1; font-size: 13px;")
            layout.addWidget(lbl)
            
        layout.addWidget(self._create_separator())
        log_title = QLabel("Skipping Logs")
        log_title.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: bold;")
        layout.addWidget(log_title)
        
        from PyQt6.QtWidgets import QTextEdit
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 4px;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.log_area)
        
        layout.addStretch()

    def log_skip(self, msg):
        self.log_area.append(msg)
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #334155;")
        return line

    def update_metadata(self, file_path):
        if not os.path.exists(file_path):
            return
            
        # 1. File Size
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        self.file_size_label.setText(f"Size: {size_mb:.2f} MB")
        
        # 2. Extract info with OpenCV
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            
            # Codec (fourcc to string)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = "Unknown"
            if fourcc != 0:
                try:
                    codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                except:
                    pass
            self.codec_label.setText(f"Video Codec: {codec_str}")
            
            if width > 0 and height > 0:
                self.resolution_label.setText(f"Resolution: {width} x {height}")
            
            # Calculate rough bitrate (Size in bits / Duration in seconds)
            if fps > 0 and frame_count > 0:
                duration_sec = frame_count / fps
                bitrate_kbps = (size_bytes * 8) / (duration_sec * 1000)
                self.bitrate_label.setText(f"Bitrate: {bitrate_kbps:.0f} kbps")
                
        cap.release()

    def update_stats(self, score, fps, is_safe):
        self.risk_bar.setValue(int(score * 100))
        self.fps_label.setText(f"Extraction FPS: {fps:.1f}")
        
        if is_safe:
            self.status_badge.setText("SAFE")
            self.status_badge.setStyleSheet("""
                background-color: rgba(34, 197, 94, 0.2);
                color: #4ade80;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
                border: 1px solid rgba(34, 197, 94, 0.5);
            """)
        else:
            self.status_badge.setText("UNSAFE DETECTED")
            self.status_badge.setStyleSheet("""
                background-color: rgba(239, 68, 68, 0.2);
                color: #f87171;
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
                border: 1px solid rgba(239, 68, 68, 0.5);
            """)
