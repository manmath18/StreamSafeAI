from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
                             QSlider, QLabel, QFileDialog, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
import os

from player.vlc_backend import VLCBackend
from detection.ai_detector import SafeVisionEngine

class TimelineWidget(QFrame):
    """Custom timeline widget to show unsafe segments in red."""
    def __init__(self):
        super().__init__()
        self.setFixedHeight(12)
        self.setStyleSheet("""
            QFrame {
                background-color: #334155;
                border-radius: 4px;
            }
        """)
        self.segments = []

    def set_segments(self, segments):
        self.segments = segments
        self.update()

    # We would override paintEvent to draw red segments, but for simplicity, 
    # we leave the UI structure ready.

class PlayerPanel(QWidget):
    frame_extracted = pyqtSignal(float, float, bool)  # score, fps, is_safe
    unsafe_detected = pyqtSignal(str)
    video_loaded = pyqtSignal(str) # file_path
    skip_detail_updated = pyqtSignal(dict)  # rich skip telemetry dict
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #020617;
            }
        """)
        
        self.player = VLCBackend()
        self.ai_engine = SafeVisionEngine()
        
        # Connect AI engine signals
        self.ai_engine.stats_updated.connect(self.frame_extracted.emit)
        self.ai_engine.skip_requested.connect(self.handle_skip_request)
        self.ai_engine.skip_detail_updated.connect(self.skip_detail_updated.emit)
        
        self.init_ui()

        # Update timer for UI
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(500)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Top toolbar
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(15, 15, 15, 15)
        
        self.open_btn = QPushButton("📂 Open Video")
        self.open_btn.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
        """)
        self.open_btn.clicked.connect(self.open_file)
        toolbar.addWidget(self.open_btn)
        
        self.title_label = QLabel("No Video Loaded")
        self.title_label.setStyleSheet("color: #94a3b8; font-weight: bold;")
        toolbar.addWidget(self.title_label)
        toolbar.addStretch()
        
        layout.addLayout(toolbar)
        
        # Video Frame
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("background-color: black;")
        layout.addWidget(self.video_frame, stretch=1)
        self.player.set_window(self.video_frame)
        
        # Controls area
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(20, 10, 20, 20)
        
        # Timeline
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 1000)
        self.timeline.setStyleSheet("""
            QSlider::groove:horizontal {
                border-radius: 4px;
                height: 8px;
                margin: 0px;
                background-color: #334155;
            }
            QSlider::handle:horizontal {
                background-color: #38bdf8;
                border: none;
                height: 16px;
                width: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::sub-page:horizontal {
                background-color: #3b82f6;
                border-radius: 4px;
            }
        """)
        self.timeline.sliderMoved.connect(self.set_position)
        controls_layout.addWidget(self.timeline)
        
        # Buttons
        btns_layout = QHBoxLayout()
        
        self.skip_back_btn = QPushButton("⏪ 10s")
        self.skip_back_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: white;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        self.skip_back_btn.clicked.connect(self.skip_backward)
        btns_layout.addWidget(self.skip_back_btn)

        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #4f46e5;
                color: white;
                border-radius: 20px;
                font-size: 18px;
            }
            QPushButton:hover {
                background-color: #4338ca;
            }
        """)
        self.play_btn.clicked.connect(self.toggle_play)
        btns_layout.addWidget(self.play_btn)
        
        self.skip_forward_btn = QPushButton("10s ⏩")
        self.skip_forward_btn.setStyleSheet("""
            QPushButton {
                background-color: #334155;
                color: white;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton:hover { background-color: #475569; }
        """)
        self.skip_forward_btn.clicked.connect(self.skip_forward)
        btns_layout.addWidget(self.skip_forward_btn)
        
        self.time_label = QLabel("00:00 / 00:00")
        self.time_label.setStyleSheet("color: #cbd5e1; font-weight: bold; margin-left: 10px;")
        btns_layout.addWidget(self.time_label)
        
        btns_layout.addStretch()
        controls_layout.addLayout(btns_layout)
        
        layout.addLayout(controls_layout)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "Video Files (*.mp4 *.mkv *.avi *.mov *.webm)")
        if file_name:
            self.title_label.setText(os.path.basename(file_name))
            self.player.play_media(file_name)
            self.play_btn.setText("▶")
            # Emit signal for metadata extraction
            self.video_loaded.emit(file_name)
            # Start AI Engine
            self.ai_engine.start_analysis(file_name, self.player)

    def toggle_play(self):
        if self.player.is_playing():
            self.player.pause()
            self.play_btn.setText("▶")
        else:
            self.player.play()
            self.play_btn.setText("⏸")

    def set_position(self, pos):
        self.player.set_position(pos / 1000.0)

    def update_ui(self):
        if self.player.is_playing():
            pos = self.player.get_position()
            self.timeline.setValue(int(pos * 1000))
            
            time_str = f"{self.format_time(self.player.get_time())} / {self.format_time(self.player.get_length())}"
            self.time_label.setText(time_str)

    def format_time(self, ms):
        if ms < 0: return "00:00"
        s = int(ms / 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    def handle_skip_request(self, start_ms, end_ms):
        msg = f"Unsafe scene detected at {self.format_time(start_ms)}. Auto-skipping to {self.format_time(end_ms)}..."
        self.unsafe_detected.emit(msg)
        try:
            from utils.db_manager import log_event
            # Default to 95.0% confidence for logged skip events
            file_path = "Unknown"
            if hasattr(self.player, 'media') and self.player.media is not None:
                file_path = self.player.media.get_mrl()
            log_event("AUTO_SKIP", msg, file_path, 95.0)
        except Exception as e:
            print("DB Log Error:", e)
        self.player.set_time(end_ms)

    def skip_backward(self):
        current_time = self.player.get_time()
        new_time = max(0, current_time - 10000)
        self.player.set_time(new_time)

    def skip_forward(self):
        current_time = self.player.get_time()
        length = self.player.get_length()
        if length > 0:
            new_time = min(length, current_time + 10000)
        else:
            new_time = current_time + 10000
        self.player.set_time(new_time)
