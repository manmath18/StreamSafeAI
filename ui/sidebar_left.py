from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QFrame, QButtonGroup
from PyQt6.QtCore import Qt, pyqtSignal

class LeftSidebar(QWidget):
    tab_selected = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(30, 41, 59, 0.7);
                border-right: 1px solid #334155;
            }
            QPushButton {
                background-color: transparent;
                color: #cbd5e1;
                border: none;
                text-align: left;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: 500;
                border-radius: 8px;
                margin: 4px 10px;
            }
            QPushButton:hover {
                background-color: rgba(56, 189, 248, 0.1);
                color: #38bdf8;
            }
            QPushButton:checked {
                background-color: rgba(56, 189, 248, 0.2);
                color: #38bdf8;
                font-weight: bold;
                border-left: 4px solid #38bdf8;
            }
        """)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 20, 0, 20)
        
        # Logo Area
        logo_label = QLabel("🛡️ SafeVision")
        logo_label.setStyleSheet("""
            color: #f8fafc;
            font-size: 20px;
            font-weight: 800;
            padding: 0px 20px 20px 20px;
            border: none;
        """)
        layout.addWidget(logo_label)
        
        # Navigation Buttons
        self.nav_items = [
            ("Live Stream Player", 0),
            ("Preprocess & Upload", 1),
            ("Blur Settings", 2),
            ("Scan Library", 3),
            ("History", 4)
        ]
        
        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)
        
        for name, idx in self.nav_items:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            
            # Connect the button click to our custom emit function
            btn.clicked.connect(lambda checked, i=idx: self.tab_selected.emit(i))
            
            self.btn_group.addButton(btn, idx)
            layout.addWidget(btn)
            
            if idx == 0:
                btn.setChecked(True)
            
        layout.addStretch()
