import sys
import os
from PyQt6.QtWidgets import QApplication
from ui.main_window import MainWindow

def main():
    # Setup high DPI scaling
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
    
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    # Global dark theme and modern aesthetics
    app.setStyleSheet("""
        QWidget {
            background-color: #0f172a;
            color: #f8fafc;
            font-family: 'Inter', 'Segoe UI', sans-serif;
        }
        QScrollBar:vertical {
            background: #1e293b;
            width: 10px;
            margin: 0px 0px 0px 0px;
        }
        QScrollBar::handle:vertical {
            background: #334155;
            min-height: 20px;
            border-radius: 5px;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
    """)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
