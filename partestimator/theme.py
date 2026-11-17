"""One light theme for the whole tool, so both tabs look like one program.

Fusion style with a light palette, plus a small stylesheet for the pieces
Fusion leaves plain. The machining tab's existing light look is the
reference; the sheet metal tab's dark SMTv13 theme was retired.
"""
from __future__ import annotations

ACCENT = "#1f5fbf"

QSS = f"""
QWidget {{ font-size: 12px; }}
QMainWindow, QDialog {{ background: #f4f5f7; }}
QGroupBox {{ border: 1px solid #d3d6db; border-radius: 5px; margin-top: 10px;
            padding-top: 8px; background: #ffffff; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px;
                   color: #374151; font-weight: 600; }}
QPushButton, QToolButton {{ background: #ffffff; border: 1px solid #c4c8ce;
                           border-radius: 4px; padding: 4px 10px; }}
QPushButton:hover, QToolButton:hover {{ background: #eef3fb; border-color: {ACCENT}; }}
QPushButton:disabled, QToolButton:disabled {{ color: #9ca3af; background: #f3f4f6; }}
QToolButton:checked {{ background: #dbe7fb; border-color: {ACCENT}; color: #0b3a82; }}
QPushButton#primary {{ background: {ACCENT}; color: white; border-color: {ACCENT};
                      font-weight: 600; }}
QPushButton#primary:hover {{ background: #2a6fd6; }}
QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {{ background: #ffffff;
    border: 1px solid #c4c8ce; border-radius: 3px; padding: 2px 4px; }}
QTableWidget, QTextEdit {{ background: #ffffff; border: 1px solid #d3d6db;
    gridline-color: #e5e7eb; alternate-background-color: #f8fafc; }}
QHeaderView::section {{ background: #eef0f3; border: 0; border-bottom: 1px solid #d3d6db;
                       padding: 4px; font-weight: 600; }}
QTabWidget::pane {{ border: 1px solid #d3d6db; background: #ffffff; }}
QTabBar::tab {{ background: #e9ebef; padding: 5px 12px; border: 1px solid #d3d6db;
               border-bottom: 0; border-top-left-radius: 4px; border-top-right-radius: 4px; }}
QTabBar::tab:selected {{ background: #ffffff; }}
QTabBar#modeTabs::tab {{ padding: 8px 22px; font-size: 13px; font-weight: 600;
                        color: #4b5563; min-width: 120px; }}
QTabBar#modeTabs::tab:selected {{ color: {ACCENT}; border-top: 3px solid {ACCENT}; }}
QScrollArea {{ border: 0; }}
QStatusBar {{ background: #eef0f3; }}
QSlider::groove:horizontal {{ height: 4px; background: #d3d6db; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {ACCENT}; width: 12px; margin: -5px 0;
                             border-radius: 6px; }}
"""


def apply(app) -> None:
    from PySide6.QtGui import QColor, QPalette

    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#f4f5f7"))
    pal.setColor(QPalette.Base, QColor("#ffffff"))
    pal.setColor(QPalette.AlternateBase, QColor("#f8fafc"))
    pal.setColor(QPalette.Text, QColor("#111827"))
    pal.setColor(QPalette.WindowText, QColor("#111827"))
    pal.setColor(QPalette.Button, QColor("#ffffff"))
    pal.setColor(QPalette.ButtonText, QColor("#111827"))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    pal.setColor(QPalette.ToolTipText, QColor("#111827"))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
