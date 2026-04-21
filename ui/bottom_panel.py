from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import Qt

class BottomPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #0f172a;
                border-top: 1px solid #334155;
            }
            QLabel {
                color: #94a3b8;
                font-family: monospace;
            }
        """)
        self.setFixedHeight(40)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 5, 15, 5)
        
        self.log_label = QLabel("SafeVision Engine Initialized. Ready to play.")
        layout.addWidget(self.log_label)

    def log_event(self, message):
        self.log_label.setText(f"System: {message}")
        self.log_label.setStyleSheet("color: #38bdf8;")
