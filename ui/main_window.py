from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QSplitter, QLabel, QPushButton, QFrame, QSizePolicy, QStackedWidget)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QFont

from ui.player_panel import PlayerPanel
from ui.sidebar_left import LeftSidebar
from ui.sidebar_right import RightSidebar
from ui.bottom_panel import BottomPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SafeVision - Intelligent Media Player")
        self.setMinimumSize(1280, 800)
        self.init_ui()

    def init_ui(self):
        # Main central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main layout
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # Horizontal layout for the three columns
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setHandleWidth(1)
        self.content_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #1e293b;
            }
        """)
        
        # 1. Left Sidebar
        self.left_sidebar = LeftSidebar()
        self.content_splitter.addWidget(self.left_sidebar)
        
        # 2. Center Stacked Widget
        self.center_stack = QStackedWidget()
        
        # Tab 0: Player Panel
        self.player_panel = PlayerPanel()
        self.center_stack.addWidget(self.player_panel)
        
        # Tab 1: Preprocess & Upload
        self.preprocess_panel = self._create_placeholder("Preprocess & Upload Area\n\nUpload a video for batch processing, scanning, and generating a clean trimmed version.")
        self.center_stack.addWidget(self.preprocess_panel)
        
        # Tab 2: Blur Settings
        self.blur_settings_panel = self._create_placeholder("Blur Settings Configuration\n\nToggle which explicit content classes to blur and set blur strength thresholds.")
        self.center_stack.addWidget(self.blur_settings_panel)
        
        # Tab 3: Scan Library
        self.scan_lib_panel = self._create_placeholder("Scan Library\n\nBatch scan multiple media files in a directory.")
        self.center_stack.addWidget(self.scan_lib_panel)

        # Tab 4: History
        self.history_panel = self._create_placeholder("Scan History\n\nView previous detections and video reports.")
        self.center_stack.addWidget(self.history_panel)
        
        self.content_splitter.addWidget(self.center_stack)
        
        # 3. Right Sidebar
        self.right_sidebar = RightSidebar()
        self.content_splitter.addWidget(self.right_sidebar)
        
        # Set initial splitter sizes (Left: 200, Center: 800, Right: 280)
        self.content_splitter.setSizes([200, 800, 280])
        
        self.main_layout.addWidget(self.content_splitter, stretch=1)
        
        # 4. Bottom Panel (Logs / Detection Events)
        self.bottom_panel = BottomPanel()
        self.main_layout.addWidget(self.bottom_panel)
        
        # Connect signals
        self.left_sidebar.tab_selected.connect(self.center_stack.setCurrentIndex)
        self.player_panel.frame_extracted.connect(self.right_sidebar.update_stats)
        self.player_panel.unsafe_detected.connect(self.bottom_panel.log_event)
        self.player_panel.video_loaded.connect(self.right_sidebar.update_metadata)

    def _create_placeholder(self, text):
        widget = QWidget()
        widget.setStyleSheet("background-color: #020617;")
        layout = QVBoxLayout(widget)
        label = QLabel(text)
        label.setStyleSheet("color: #cbd5e1; font-size: 16px; font-weight: bold;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return widget
        
