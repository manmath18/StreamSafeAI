"""
Right Sidebar – SafeVision AI Monitor Panel
Displays live engine telemetry, skip logic state, and a colour-coded event log.
"""
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QFrame, QStackedWidget, QTextEdit, QScrollArea, QSizePolicy,
)
from PyQt6.QtCore import Qt, QDateTime
from PyQt6.QtGui import QColor
import os
import cv2


def _fmt_ms(ms: int) -> str:
    """Format milliseconds as M:SS."""
    if ms < 0:
        return "--:--"
    s = ms // 1000
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def _pct(v: float) -> int:
    return max(0, min(100, int(v * 100)))


class _MiniBar(QWidget):
    """Compact labelled progress bar for score breakdown."""
    def __init__(self, label: str, color: str = "#3b82f6"):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        lbl = QLabel(label)
        lbl.setFixedWidth(58)
        lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(lbl)

        self.bar = QProgressBar()
        self.bar.setMaximum(100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(7)
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                border: none;
                border-radius: 3px;
                background-color: #1e293b;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)
        layout.addWidget(self.bar)

        self.val_lbl = QLabel("0%")
        self.val_lbl.setFixedWidth(32)
        self.val_lbl.setStyleSheet("color: #cbd5e1; font-size: 11px;")
        self.val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self.val_lbl)

    def set_value(self, v: float):
        pct = _pct(v)
        self.bar.setValue(pct)
        self.val_lbl.setText(f"{pct}%")


class _SkipStateCard(QWidget):
    """Card showing the current skip engine state."""

    def __init__(self):
        super().__init__()
        self.setStyleSheet("""
            QWidget {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 8px;
            }
            QLabel { border: none; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Header row
        hdr = QHBoxLayout()
        lbl = QLabel("⚡ Skip Engine")
        lbl.setStyleSheet("color: #f8fafc; font-weight: bold; font-size: 13px;")
        hdr.addWidget(lbl)
        hdr.addStretch()
        self.state_badge = QLabel("MONITORING")
        self.state_badge.setStyleSheet(
            "background-color: rgba(34,197,94,0.15); color: #4ade80;"
            "padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;"
        )
        hdr.addWidget(self.state_badge)
        layout.addLayout(hdr)

        # Skip count + distance
        row1 = QHBoxLayout()
        self.skip_count_lbl = QLabel("Skips: 0")
        self.skip_count_lbl.setStyleSheet("color: #cbd5e1; font-size: 12px;")
        row1.addWidget(self.skip_count_lbl)
        row1.addStretch()
        self.skip_dist_lbl = QLabel("Next skip: +5s")
        self.skip_dist_lbl.setStyleSheet("color: #60a5fa; font-size: 12px;")
        row1.addWidget(self.skip_dist_lbl)
        layout.addLayout(row1)

        # Cooldown bar
        cd_hdr = QLabel("Cooldown remaining")
        cd_hdr.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(cd_hdr)
        self.cooldown_bar = QProgressBar()
        self.cooldown_bar.setMaximum(3000)
        self.cooldown_bar.setValue(0)
        self.cooldown_bar.setTextVisible(False)
        self.cooldown_bar.setFixedHeight(6)
        self.cooldown_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                border-radius: 3px;
                background-color: #1e293b;
            }
            QProgressBar::chunk {
                background-color: #f59e0b;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.cooldown_bar)
        self.cooldown_lbl = QLabel("Ready")
        self.cooldown_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.cooldown_lbl)

        # Last skip target
        self.last_skip_lbl = QLabel("Last skip: --")
        self.last_skip_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(self.last_skip_lbl)

    def update_from(self, t: dict):
        skip_count  = t.get("skip_count", 0)
        dist_ms     = t.get("skip_distance_ms", 5000)
        cd_ms       = t.get("cooldown_remain_ms", 0)
        is_unsafe   = t.get("is_unsafe", False)
        skip_target = t.get("skip_target_ms")
        event       = t.get("event", "FRAME")
        sm_state    = t.get("skip_state", "MONITORING")
        tube_len    = t.get("tube_length", 0)

        self.skip_count_lbl.setText(f"Skips fired: {skip_count}  |  Tube len: {tube_len}")
        self.skip_dist_lbl.setText(f"Next skip: +{dist_ms // 1000}s")

        cd_capped = min(cd_ms, 3000)
        self.cooldown_bar.setValue(cd_capped)
        if cd_ms > 0:
            self.cooldown_lbl.setText(f"Cooling down {cd_ms / 1000:.1f}s …")
        else:
            self.cooldown_lbl.setText("Ready to skip")

        if event == "SKIP" and skip_target is not None:
            pos_ms = t.get("position_ms", 0)
            self.last_skip_lbl.setText(
                f"Last: {_fmt_ms(pos_ms)} → {_fmt_ms(skip_target)}"
            )

        # State-machine badge
        STATE_STYLES = {
            "IDLE":            ("IDLE",             "rgba(100,116,139,0.2)",  "#94a3b8"),
            "MONITORING":      ("✓ MONITORING",     "rgba(34,197,94,0.15)",   "#4ade80"),
            "UNSAFE_DETECTED": ("⚠ UNSAFE DETECT",  "rgba(239,68,68,0.25)",   "#f87171"),
            "SKIPPING":        ("⚡ SKIPPING",       "rgba(245,158,11,0.25)",  "#fbbf24"),
            "SEEKING":         ("🔍 SEEKING",        "rgba(96,165,250,0.25)",  "#60a5fa"),
            "BUFFER_RESET":    ("🔄 BUFFER RESET",   "rgba(167,139,250,0.25)","#a78bfa"),
        }
        label, bg, fg = STATE_STYLES.get(
            sm_state,
            (sm_state, "rgba(100,116,139,0.2)", "#94a3b8")
        )
        if is_unsafe and sm_state == "MONITORING":
            label, bg, fg = ("⚠ UNSAFE", "rgba(239,68,68,0.2)", "#f87171")
        self.state_badge.setText(label)
        self.state_badge.setStyleSheet(
            f"background-color: {bg}; color: {fg};"
            "padding: 2px 8px; border-radius: 4px; font-size: 11px; font-weight: bold;"
        )


class _TriggerBadges(QWidget):
    """Row of small badges showing which CV signals fired this frame."""

    _TAGS = {
        "close_faces_kissing":   ("👄 Kiss",         "#ef4444"),
        "face_profile_mix":      ("👥 Face Mix",      "#f97316"),
        "high_skin_exposure":    ("🔴 Skin Hi",       "#ef4444"),
        "medium_skin_exposure":  ("🟠 Skin Med",      "#f97316"),
        "dark_scene_with_people":("🌑 Dark",          "#8b5cf6"),
        "red_tones_with_faces":  ("🔴 Red+Face",      "#f87171"),
        "motion_blur_with_people":("💨 Blur",         "#60a5fa"),
        "multiple_faces":        ("👥 3+Faces",       "#a78bfa"),
        "kiss_cooldown_bridge":  ("⏳ KissCool",      "#fbbf24"),
        "skin_cooldown_bridge":  ("⏳ SkinCool",      "#fbbf24"),
    }

    def __init__(self):
        super().__init__()
        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(4)
        self._layout.addStretch()
        self._badges: dict = {}

    def update_detections(self, detections: list):
        active = set(detections)

        for tag, (text, color) in self._TAGS.items():
            if tag in active:
                if tag not in self._badges:
                    badge = QLabel(text)
                    badge.setStyleSheet(
                        f"background-color: rgba({self._hex_to_rgb(color)}, 0.2);"
                        f"color: {color}; padding: 2px 5px;"
                        "border-radius: 4px; font-size: 10px; font-weight: bold;"
                    )
                    self._badges[tag] = badge
                    self._layout.insertWidget(self._layout.count() - 1, badge)
                self._badges[tag].setVisible(True)
            else:
                if tag in self._badges:
                    self._badges[tag].setVisible(False)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"


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
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.stack = QStackedWidget()
        main_layout.addWidget(self.stack)

        # Page 0: Live Stream Mode
        self.live_page = QWidget()
        self.init_live_page()
        self.stack.addWidget(self.live_page)

        # Page 1: Batch Process Mode
        self.batch_page = QWidget()
        self.init_batch_page()
        self.stack.addWidget(self.batch_page)

        self.stack.setCurrentIndex(0)

    def set_mode(self, mode):
        if mode == "live":
            self.stack.setCurrentIndex(0)
        elif mode == "batch":
            self.stack.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Live page
    # ------------------------------------------------------------------

    def init_live_page(self):
        # Use a scroll area so nothing is cut off on small screens
        scroll = QScrollArea(self.live_page)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(14)

        # ---- Title ----
        title = QLabel("AI Monitor Panel (Live)")
        title.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # ---- Status Badge ----
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

        # ---- Risk Score ----
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
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4ade80, stop:0.5 #facc15, stop:1 #ef4444
                );
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.risk_bar)

        # ---- Hardware Stats ----
        self.fps_label = QLabel("FPS: --")
        self.infer_label = QLabel("Infer: -- ms")
        layout.addWidget(self.fps_label)
        layout.addWidget(self.infer_label)

        layout.addWidget(self._create_separator())

        # ========================================================
        # SKIP LOGIC SECTION
        # ========================================================
        skip_title = QLabel("Skip Logic Monitor")
        skip_title.setStyleSheet("color: #f8fafc; font-size: 14px; font-weight: bold;")
        layout.addWidget(skip_title)

        # State card
        self.skip_card = _SkipStateCard()
        layout.addWidget(self.skip_card)

        # Score breakdown
        breakdown_title = QLabel("Score Breakdown (per frame)")
        breakdown_title.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: bold;")
        layout.addWidget(breakdown_title)

        self._bar_eff    = _MiniBar("EffNet",  "#8b5cf6")
        self._bar_onnx   = _MiniBar("ONNX",   "#6366f1")
        self._bar_cv     = _MiniBar("CV",      "#10b981")
        self._bar_blur   = _MiniBar("Blur",    "#60a5fa")
        self._bar_fused  = _MiniBar("Fused",   "#f59e0b")
        self._bar_smooth = _MiniBar("Tube",    "#ef4444")

        for bar in [self._bar_eff, self._bar_onnx, self._bar_cv,
                    self._bar_blur, self._bar_fused, self._bar_smooth]:
            layout.addWidget(bar)

        # Active trigger badges
        triggers_title = QLabel("Active Triggers")
        triggers_title.setStyleSheet("color: #94a3b8; font-size: 12px; font-weight: bold;")
        layout.addWidget(triggers_title)

        self.trigger_badges = _TriggerBadges()
        layout.addWidget(self.trigger_badges)

        # Skip event log
        layout.addWidget(self._create_separator())
        log_title = QLabel("Skip Event Log")
        log_title.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: bold;")
        layout.addWidget(log_title)

        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setFixedHeight(160)
        self.log_area.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 4px;
                font-family: monospace;
                font-size: 11px;
            }
        """)
        layout.addWidget(self.log_area)

        # ========================================================
        # DIAGNOSTICS & METADATA
        # ========================================================
        layout.addWidget(self._create_separator())
        meta_title = QLabel("Diagnostics & Metadata")
        meta_title.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: bold;")
        layout.addWidget(meta_title)

        self.file_size_label   = QLabel("Size: -- MB")
        self.resolution_label  = QLabel("Resolution: -- x --")
        self.bitrate_label     = QLabel("Bitrate: -- kbps")
        self.codec_label       = QLabel("Video Codec: --")
        self.model_acc_label   = QLabel("Real-time Confidence: --%")
        self.model_status_label = QLabel("Late-Fusion Engine: Active")
        self.window_size_label = QLabel("EWM α: 0.38 | Window: 12")

        for lbl in [
            self.file_size_label, self.resolution_label,
            self.bitrate_label, self.codec_label,
            self.model_acc_label, self.model_status_label, self.window_size_label,
        ]:
            lbl.setStyleSheet("color: #cbd5e1; font-size: 12px;")
            layout.addWidget(lbl)

        layout.addStretch()

        scroll.setWidget(container)

        # Put scroll area inside the live_page
        page_layout = QVBoxLayout(self.live_page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.addWidget(scroll)

    # ------------------------------------------------------------------
    # Batch page
    # ------------------------------------------------------------------

    def init_batch_page(self):
        layout = QVBoxLayout(self.batch_page)
        layout.setContentsMargins(15, 20, 15, 20)
        layout.setSpacing(20)

        title = QLabel("Batch Processing Panel")
        title.setStyleSheet("color: #f8fafc; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        model_title = QLabel("Model Metrics")
        model_title.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: bold;")
        layout.addWidget(model_title)

        self.batch_model_acc_label  = QLabel("Late-Fusion Engine: Active")
        self.batch_window_size_label = QLabel("Aggregator Window: 8 frames")

        for lbl in [self.batch_model_acc_label, self.batch_window_size_label]:
            lbl.setStyleSheet("color: #cbd5e1; font-size: 13px;")
            layout.addWidget(lbl)

        layout.addWidget(self._create_separator())

        log_title = QLabel("Processing Logs")
        log_title.setStyleSheet("color: #94a3b8; font-size: 14px; font-weight: bold;")
        layout.addWidget(log_title)

        self.batch_log_area = QTextEdit()
        self.batch_log_area.setReadOnly(True)
        self.batch_log_area.setStyleSheet("""
            QTextEdit {
                background-color: #0f172a;
                color: #cbd5e1;
                border: 1px solid #334155;
                border-radius: 4px;
                font-family: monospace;
                font-size: 12px;
            }
        """)
        layout.addWidget(self.batch_log_area)

    # ------------------------------------------------------------------
    # Slot: rich engine telemetry (called ~5x per second while streaming)
    # ------------------------------------------------------------------

    def update_skip_detail(self, t: dict):
        """Receive live telemetry dict from SafeVisionEngine and refresh UI."""
        # Score breakdown bars
        self._bar_eff.set_value(t.get("eff_score", 0))
        self._bar_onnx.set_value(t.get("onnx_score", 0))
        self._bar_cv.set_value(t.get("cv_score", 0))
        self._bar_blur.set_value(t.get("blur_score", 0))
        self._bar_fused.set_value(t.get("fused_score", 0))
        self._bar_smooth.set_value(t.get("tube_score", t.get("smoothed_score", 0)))

        # Infer time + batch size
        batch = t.get("batch_size", 1)
        self.infer_label.setText(
            f"Infer: {t.get('infer_ms', 0):.0f} ms  |  batch={batch}"
        )

        # Trigger badges
        self.trigger_badges.update_detections(t.get("detections", []))

        # Skip state card
        self.skip_card.update_from(t)

        # Log skip events only (not every frame, to avoid log spam)
        if t.get("event") == "SKIP" and t.get("skip_target_ms") is not None:
            pos  = t.get("position_ms", 0)
            tgt  = t.get("skip_target_ms", 0)
            sc   = t.get("skip_count", 0)
            dist = t.get("skip_distance_ms", 0)
            tube = t.get("tube_length", 0)
            now  = QDateTime.currentDateTime().toString("hh:mm:ss")
            detections  = t.get("detections", [])
            trigger_str = ", ".join(detections) if detections else "high-score"

            html = (
                f"<span style='color:#fbbf24'>[{now}]</span> "
                f"<span style='color:#f87171;font-weight:bold'>⚡ SKIP #{sc}</span> "
                f"<span style='color:#cbd5e1'>"
                f"{_fmt_ms(pos)} → {_fmt_ms(tgt)} "
                f"(+{dist // 1000}s)</span><br>"
                f"<span style='color:#94a3b8'>  Trigger: {trigger_str} "
                f"| tube={tube}f</span>"
            )
            self.log_area.append(html)
            sb = self.log_area.verticalScrollBar()
            sb.setValue(sb.maximum())

    # ------------------------------------------------------------------
    # Existing slots kept for backward compatibility
    # ------------------------------------------------------------------

    def log_batch(self, msg):
        if hasattr(self, "batch_log_area"):
            self.batch_log_area.append(msg)
            sb = self.batch_log_area.verticalScrollBar()
            sb.setValue(sb.maximum())

    def log_skip(self, msg):
        """Legacy slot – called from player_panel unsafe_detected signal."""
        now = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_area.append(
            f"<span style='color:#94a3b8'>[{now}]</span> "
            f"<span style='color:#94a3b8'>{msg}</span>"
        )
        sb = self.log_area.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _create_separator(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        line.setStyleSheet("background-color: #334155; max-height: 1px;")
        return line

    def update_metadata(self, file_path):
        if not os.path.exists(file_path):
            return

        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        self.file_size_label.setText(f"Size: {size_mb:.2f} MB")

        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            codec_str = "Unknown"
            if fourcc != 0:
                try:
                    codec_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
                except Exception:
                    pass
            self.codec_label.setText(f"Video Codec: {codec_str}")

            if width > 0 and height > 0:
                self.resolution_label.setText(f"Resolution: {width} x {height}")

            if fps > 0 and frame_count > 0:
                duration_sec = frame_count / fps
                bitrate_kbps = (size_bytes * 8) / (duration_sec * 1000)
                self.bitrate_label.setText(f"Bitrate: {bitrate_kbps:.0f} kbps")

        cap.release()

    def update_stats(self, score, fps, is_safe):
        self.risk_bar.setValue(_pct(score))
        self.fps_label.setText(f"Extraction FPS: {fps:.1f}")
        self.model_acc_label.setText(f"Real-time Confidence: {score * 100:.1f}%")

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
