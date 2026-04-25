from PyQt6.QtWidgets import (QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, 
                             QSplitter, QLabel, QPushButton, QFrame, QSizePolicy, QStackedWidget, QFileDialog,
                             QGroupBox, QGridLayout, QTextEdit, QTabWidget)
from PyQt6.QtCore import Qt, QSize, QThread, pyqtSignal
from PyQt6.QtGui import QIcon, QFont, QImage, QPixmap
import os
import cv2
import numpy as np

class BatchProcessingThread(QThread):
    progress_updated = pyqtSignal(int)
    log_updated = pyqtSignal(str)
    frame_updated = pyqtSignal(QImage)
    finished_processing = pyqtSignal()

    def __init__(self, input_path, output_dir):
        super().__init__()
        self.input_path = input_path
        self.output_dir = output_dir
        self.running = True

    def run(self):
        self.log_updated.emit("[SYSTEM] Initializing models (Haar, ONNX)...")
        try:
            from live import LiveNudeDetector
            detector = LiveNudeDetector()
            
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
            
            cap = cv2.VideoCapture(self.input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            out_name = os.path.splitext(os.path.basename(self.input_path))[0]
            blur_out_path = os.path.join(self.output_dir, f"{out_name}_blurred.mp4")
            trim_out_path = os.path.join(self.output_dir, f"{out_name}_trimmed.mp4")
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out_blur = cv2.VideoWriter(blur_out_path, fourcc, fps, (width, height))
            out_trim = cv2.VideoWriter(trim_out_path, fourcc, fps, (width, height))
            
            self.log_updated.emit(f"[INFO] Video Details: {width}x{height} @ {fps:.1f}fps. Total frames: {total_frames}")
            self.log_updated.emit(f"[INFO] Generating: {blur_out_path}")
            self.log_updated.emit(f"[INFO] Generating: {trim_out_path}")
            
            frame_idx = 0
            score_history = []
            W = 8
            kiss_cooldown = 0
            skin_cooldown = 0
            
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                if total_frames > 0 and frame_idx % 10 == 0:
                    self.progress_updated.emit(int((frame_idx / total_frames) * 100))
                    
                blur_frame = frame.copy()
                
                # Fast processing (subsample or run full? Let's run full for accuracy)
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces_frontal = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
                faces_profile = profile_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
                
                faces = list(faces_frontal) + list(faces_profile)
                
                detections = detector.detect_frame(frame)
                
                for d in detections:
                    if d["class"] in ["FACE_FEMALE", "FACE_MALE"] and d.get("score", 0.0) > 0.4:
                        faces.append(d["box"])
                        
                has_unsafe = False
                max_unsafe_score = 0.0
                
                for d in detections:
                    label = d["class"]
                    severity = d.get("severity", "MODERATE")
                    score = d.get("score", 0.0)
                    
                    if detector.should_apply_blur(label):
                        if severity == 'CRITICAL':
                            has_unsafe = True
                            max_unsafe_score = max(max_unsafe_score, min(1.0, score + 0.40))
                        elif severity == 'HIGH':
                            has_unsafe = True
                            max_unsafe_score = max(max_unsafe_score, min(1.0, score + 0.30))
                        elif severity == 'MODERATE' and score >= 0.4:
                            has_unsafe = True
                            max_unsafe_score = max(max_unsafe_score, score)
                        elif severity == 'LOW' and score >= 0.6:
                            has_unsafe = True
                            max_unsafe_score = max(max_unsafe_score, score)
                            
                        # Blur
                        box = d["box"]
                        if len(box) == 4:
                            x, y, w, h = box
                            if w > 0 and h > 0:
                                roi = blur_frame[y:y+h, x:x+w]
                                if roi.size > 0:
                                    blurred_roi = cv2.GaussianBlur(roi, (99, 99), 30)
                                    blur_frame[y:y+h, x:x+w] = blurred_roi
                                    
                # Skin check
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                skin_mask = cv2.inRange(hsv_frame, np.array([0, 20, 70], dtype=np.uint8), np.array([20, 255, 255], dtype=np.uint8))
                skin_ratio = cv2.countNonZero(skin_mask) / (height * width)
                
                if skin_ratio > 0.18:
                    skin_cooldown = 4
                    has_unsafe = True
                    max_unsafe_score = max(max_unsafe_score, 0.90)
                elif skin_cooldown > 0:
                    skin_cooldown -= 1
                    has_unsafe = True
                    max_unsafe_score = max(max_unsafe_score, 0.90)
                    
                # Kiss check
                is_kiss = False
                for box in faces:
                    if len(box) == 4:
                        x, y, w, h = box
                        if h > 0 and (w / h) > 1.35:
                            is_kiss = True
                            break
                if not is_kiss and len(faces) >= 2:
                    for i in range(len(faces)):
                        for j in range(i+1, len(faces)):
                            x1, y1, w1, h1 = faces[i]
                            x2, y2, w2, h2 = faces[j]
                            c1x, c1y = x1 + w1/2.0, y1 + h1/2.0
                            c2x, c2y = x2 + w2/2.0, y2 + h2/2.0
                            dist = np.sqrt((c1x - c2x)**2 + (c1y - c2y)**2)
                            if dist <= height * 0.25 or dist <= ((w1 + w2) / 1.1):
                                is_kiss = True
                                break
                        if is_kiss: break
                        
                if is_kiss:
                    kiss_cooldown = 6
                    has_unsafe = True
                    max_unsafe_score = max(max_unsafe_score, 0.95)
                elif kiss_cooldown > 0:
                    kiss_cooldown -= 1
                    has_unsafe = True
                    max_unsafe_score = max(max_unsafe_score, 0.95)
                    
                score_history.append(max_unsafe_score if has_unsafe else 0.1)
                if len(score_history) > W:
                    score_history.pop(0)
                    
                S_t = np.mean(score_history) if score_history else 0.1
                
                out_blur.write(blur_frame)
                
                if S_t < 0.60:
                    out_trim.write(frame)
                else:
                    if frame_idx % 30 == 0:
                        self.log_updated.emit(f"[ACTION] Skipped explicit scene at frame {frame_idx} (Score: {S_t:.2f})")
                        
                if frame_idx % 3 == 0:
                    rgb_image = cv2.cvtColor(blur_frame, cv2.COLOR_BGR2RGB)
                    h_img, w_img, ch_img = rgb_image.shape
                    bytes_per_line = ch_img * w_img
                    qt_img = QImage(rgb_image.data, w_img, h_img, bytes_per_line, QImage.Format.Format_RGB888).copy()
                    self.frame_updated.emit(qt_img)
                    
            cap.release()
            out_blur.release()
            out_trim.release()
            
            self.progress_updated.emit(100)
            if self.running:
                self.log_updated.emit("[SUCCESS] Batch processing completed successfully!")
            else:
                self.log_updated.emit("[WARN] Batch processing stopped by user.")
            self.finished_processing.emit()
            
        except Exception as e:
            self.log_updated.emit(f"[ERROR] Process failed: {str(e)}")
            self.finished_processing.emit()


from ui.player_panel import PlayerPanel
from ui.sidebar_left import LeftSidebar
from ui.sidebar_right import RightSidebar
from ui.bottom_panel import BottomPanel

class PreprocessPanel(QWidget):
    log_message = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #020617; color: white;")
        layout = QVBoxLayout(self)
        
        # Header
        title = QLabel("Preprocess & Upload")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title)

        # 1. Input/Output Group
        io_group = QGroupBox("Video Selection")
        io_group.setStyleSheet("""
            QGroupBox { 
                color: #3b82f6; 
                font-weight: bold; 
                border: 1px solid #1e293b; 
                border-radius: 8px; 
                margin-top: 15px;
                padding-top: 15px; 
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QLabel { color: white; }
        """)
        io_layout = QGridLayout(io_group)
        
        io_layout.addWidget(QLabel("Input Video:"), 0, 0)
        self.input_path_lbl = QLabel("No file selected")
        self.input_path_lbl.setStyleSheet("color: #94a3b8;")
        io_layout.addWidget(self.input_path_lbl, 0, 1)
        self.btn_input = QPushButton("📂 Browse")
        self.btn_input.setStyleSheet("background-color: #1e293b; padding: 5px; border-radius: 4px;")
        self.btn_input.clicked.connect(self.select_input)
        io_layout.addWidget(self.btn_input, 0, 2)

        io_layout.addWidget(QLabel("Save Output To:"), 1, 0)
        self.output_path_lbl = QLabel("Default: ./output/")
        self.output_path_lbl.setStyleSheet("color: #94a3b8;")
        io_layout.addWidget(self.output_path_lbl, 1, 1)
        self.btn_output = QPushButton("💾 Set Path")
        self.btn_output.setStyleSheet("background-color: #1e293b; padding: 5px; border-radius: 4px;")
        self.btn_output.clicked.connect(self.select_output)
        io_layout.addWidget(self.btn_output, 1, 2)
        
        layout.addWidget(io_group)

        # 2. Preview and Output (Horizontal Layout)
        mid_layout = QHBoxLayout()
        
        self.preview_tabs = QTabWidget()
        self.preview_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #1e293b;
                border-radius: 8px;
                background-color: #0f172a;
            }
            QTabBar::tab {
                background-color: #1e293b;
                color: #94a3b8;
                padding: 10px 20px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #3b82f6;
                color: white;
            }
        """)
        
        # Tab 1: Live Processing Preview
        live_preview_tab = QWidget()
        live_layout = QVBoxLayout(live_preview_tab)
        self.preview_label = QLabel("Frame preview will appear here when processing starts")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("background-color: #020617; border-radius: 4px; min-height: 250px; color: #475569;")
        live_layout.addWidget(self.preview_label)
        self.preview_tabs.addTab(live_preview_tab, "📺 Live Processing Preview")
        
        # Tab 2: Output Preview
        output_preview_tab = QWidget()
        output_layout = QVBoxLayout(output_preview_tab)
        self.output_status_lbl = QLabel("No outputs generated yet.\nProcess a video to view results.")
        self.output_status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_status_lbl.setStyleSheet("color: #94a3b8; font-size: 14px; font-style: italic;")
        output_layout.addWidget(self.output_status_lbl)
        
        out_btn_layout = QHBoxLayout()
        self.btn_play_blurred = QPushButton("▶ Play Blurred Output")
        self.btn_play_blurred.setStyleSheet("background-color: #f59e0b; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        self.btn_play_blurred.setEnabled(False)
        self.btn_play_blurred.clicked.connect(lambda: self.play_output("blurred"))
        
        self.btn_play_trimmed = QPushButton("▶ Play Trimmed Output")
        self.btn_play_trimmed.setStyleSheet("background-color: #10b981; color: white; padding: 10px; border-radius: 5px; font-weight: bold;")
        self.btn_play_trimmed.setEnabled(False)
        self.btn_play_trimmed.clicked.connect(lambda: self.play_output("trimmed"))
        
        out_btn_layout.addWidget(self.btn_play_blurred)
        out_btn_layout.addWidget(self.btn_play_trimmed)
        output_layout.addLayout(out_btn_layout)
        
        self.preview_tabs.addTab(output_preview_tab, "📤 Generated Output Preview")
        
        mid_layout.addWidget(self.preview_tabs)
        
        layout.addLayout(mid_layout)

        # 3. Progress Bar
        from PyQt6.QtWidgets import QProgressBar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #1e293b;
                border-radius: 5px;
                text-align: center;
                background-color: #0f172a;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
            }
        """)
        layout.addWidget(self.progress_bar)

        # 4. Action Buttons
        action_layout = QHBoxLayout()
        self.start_btn = QPushButton("🚀 Start Processing")
        self.start_btn.setStyleSheet("background-color: #2563eb; color: white; padding: 12px; border-radius: 6px; font-weight: bold;")
        self.start_btn.setEnabled(False)
        
        self.start_btn.clicked.connect(self.start_processing)
        
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet("background-color: #dc2626; color: white; padding: 12px; border-radius: 6px; font-weight: bold;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_processing)
        
        action_layout.addWidget(self.start_btn)
        action_layout.addWidget(self.stop_btn)
        layout.addLayout(action_layout)

        layout.addStretch()
        
        self.current_input_path = None
        self.current_output_dir = "./output/"
        self.processor = None

    def select_input(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Input Video", "", "Videos (*.mp4 *.mkv *.avi *.mov *.webm)")
        if file_path:
            self.current_input_path = file_path
            self.input_path_lbl.setText(os.path.basename(file_path))
            self.start_btn.setEnabled(True)
            self.log_message.emit(f"[INFO] Selected input: {file_path}")

    def select_output(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.current_output_dir = dir_path
            self.output_path_lbl.setText(dir_path)
            self.log_message.emit(f"[INFO] Selected output directory: {dir_path}")

    def start_processing(self):
        if not self.current_input_path: return
        
        os.makedirs(self.current_output_dir, exist_ok=True)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.btn_input.setEnabled(False)
        self.btn_output.setEnabled(False)
        self.btn_play_blurred.setEnabled(False)
        self.btn_play_trimmed.setEnabled(False)
        self.output_status_lbl.setText("Processing in progress...")
        self.preview_tabs.setCurrentIndex(0) # Switch to live preview
        self.progress_bar.setValue(0)
        self.log_message.emit(f"[INFO] Starting processing for {os.path.basename(self.current_input_path)}")
        
        self.processor = BatchProcessingThread(self.current_input_path, self.current_output_dir)
        self.processor.log_updated.connect(self.append_log)
        self.processor.progress_updated.connect(self.progress_bar.setValue)
        self.processor.frame_updated.connect(self.update_preview)
        self.processor.finished_processing.connect(self.on_processing_finished)
        self.processor.start()

    def stop_processing(self):
        if self.processor:
            self.processor.running = False
            self.log_message.emit("[INFO] Stopping processor...")
            self.stop_btn.setEnabled(False)

    def on_processing_finished(self):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.btn_input.setEnabled(True)
        self.btn_output.setEnabled(True)
        self.btn_play_blurred.setEnabled(True)
        self.btn_play_trimmed.setEnabled(True)
        self.output_status_lbl.setText(f"Processing Complete!\nOutputs saved to: {self.current_output_dir}")
        self.preview_tabs.setCurrentIndex(1) # Switch to output tab
        
    def play_output(self, file_type):
        if not self.current_input_path: return
        out_name = os.path.splitext(os.path.basename(self.current_input_path))[0]
        file_name = f"{out_name}_{file_type}.mp4"
        full_path = os.path.join(self.current_output_dir, file_name)
        
        if os.path.exists(full_path):
            from PyQt6.QtGui import QDesktopServices
            from PyQt6.QtCore import QUrl
            QDesktopServices.openUrl(QUrl.fromLocalFile(full_path))
        else:
            self.log_message.emit(f"[ERROR] Could not find output file: {full_path}")
        
    def append_log(self, text):
        self.log_message.emit(text)
        
    def update_preview(self, qimg):
        pixmap = QPixmap.fromImage(qimg)
        scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.preview_label.setPixmap(scaled_pixmap)


class BlurSettingsPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #020617; color: white;")
        layout = QVBoxLayout(self)
        title = QLabel("Blur Settings")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title)
        
        from PyQt6.QtWidgets import QCheckBox, QSlider
        self.cb1 = QCheckBox("Blur Nudity")
        self.cb1.setChecked(True)
        self.cb2 = QCheckBox("Blur Kissing/Intimacy")
        self.cb2.setChecked(True)
        layout.addWidget(self.cb1)
        layout.addWidget(self.cb2)
        
        layout.addWidget(QLabel("Blur Intensity:"))
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(1, 50)
        slider.setValue(23)
        layout.addWidget(slider)
        
        layout.addStretch()

class ScanLibraryPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #020617; color: white;")
        layout = QVBoxLayout(self)
        title = QLabel("Scan Library")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title)
        
        self.btn = QPushButton("📁 Select Directory to Scan")
        self.btn.setStyleSheet("background-color: #10b981; padding: 10px; border-radius: 5px; font-weight: bold;")
        self.btn.clicked.connect(self.select_directory)
        layout.addWidget(self.btn)
        
        from PyQt6.QtWidgets import QListWidget
        self.list = QListWidget()
        self.list.setStyleSheet("background-color: #0f172a; border: 1px solid #334155; border-radius: 4px; padding: 5px;")
        layout.addWidget(self.list)

    def select_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if dir_path:
            self.list.clear()
            try:
                files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.webm'))]
                if files:
                    self.list.addItems(files)
                    from utils.db_manager import log_event
                    log_event("SCAN_LIBRARY", f"Scanned directory: {dir_path}", dir_path, 0.0)
                else:
                    self.list.addItem("No video files found in directory.")
            except Exception as e:
                self.list.addItem(f"Error scanning directory: {str(e)}")


class HistoryPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("background-color: #020617; color: white;")
        layout = QVBoxLayout(self)
        title = QLabel("Scan History & Logs")
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin-bottom: 20px;")
        layout.addWidget(title)
        
        btn = QPushButton("Refresh Logs")
        btn.setStyleSheet("background-color: #3b82f6; padding: 10px; border-radius: 5px;")
        btn.clicked.connect(self.load_logs)
        layout.addWidget(btn)
        
        from PyQt6.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Timestamp", "Event Type", "Message", "File Path", "Confidence"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setStyleSheet("background-color: #0f172a; border: 1px solid #334155;")
        layout.addWidget(self.table)
        
        self.load_logs()

    def load_logs(self):
        try:
            from utils.db_manager import get_all_logs
            from PyQt6.QtWidgets import QTableWidgetItem
            logs = get_all_logs()
            self.table.setRowCount(0)
            for row_idx, row_data in enumerate(logs):
                self.table.insertRow(row_idx)
                for col_idx, item_data in enumerate(row_data):
                    self.table.setItem(row_idx, col_idx, QTableWidgetItem(str(item_data)))
        except Exception as e:
            print("Could not load logs:", e)


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
        self.left_sidebar.tab_selected.connect(self.on_tab_selected)
        
        # 2. Center Stacked Widget
        self.center_stack = QStackedWidget()
        
        # Tab 0: Preprocess & Upload
        self.preprocess_panel = PreprocessPanel()
        self.center_stack.addWidget(self.preprocess_panel)

        
        # Tab 1: Player Panel
        self.player_panel = PlayerPanel()
        self.center_stack.addWidget(self.player_panel)

        
        # Tab 2: Blur Settings
        self.blur_settings_panel = BlurSettingsPanel()
        self.center_stack.addWidget(self.blur_settings_panel)
        
        # Tab 3: Scan Library
        self.scan_lib_panel = ScanLibraryPanel()
        self.center_stack.addWidget(self.scan_lib_panel)

        # Tab 4: History
        self.history_panel = HistoryPanel()
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
        self.left_sidebar.tab_selected.connect(self.on_tab_selected)
        self.player_panel.frame_extracted.connect(self.right_sidebar.update_stats)
        self.player_panel.unsafe_detected.connect(self.bottom_panel.log_event)
        self.player_panel.unsafe_detected.connect(self.right_sidebar.log_skip)
        self.player_panel.video_loaded.connect(self.right_sidebar.update_metadata)
        self.player_panel.skip_detail_updated.connect(self.right_sidebar.update_skip_detail)
        self.preprocess_panel.log_message.connect(self.right_sidebar.log_batch)
        
        # Initialize default mode
        self.right_sidebar.set_mode("batch")

    def on_tab_selected(self, index):
        self.center_stack.setCurrentIndex(index)
        if index == 0:
            self.right_sidebar.set_mode("batch")
        elif index == 1:
            self.right_sidebar.set_mode("live")
    def _create_placeholder(self, text):
        widget = QWidget()
        widget.setStyleSheet("background-color: #020617;")
        layout = QVBoxLayout(widget)
        label = QLabel(text)
        label.setStyleSheet("color: #cbd5e1; font-size: 16px; font-weight: bold;")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        return widget
        
