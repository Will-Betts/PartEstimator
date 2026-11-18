"""One light theme for the whole tool, so both tabs look like one program.

Fusion style with a light palette, plus a small stylesheet for the pieces
Fusion leaves plain. The machining tab's existing light look is the
reference; the sheet metal tab's dark SMTv13 theme was retired.
"""
from __future__ import annotations

from pathlib import Path

ACCENT = "#1f5fbf"
HOVER_BG = "#dbe7fb"      # a highlighted list row or menu entry
HOVER_FG = "#0b3a82"

# Styling a spin box's border makes Qt draw its buttons from the style sheet,
# and a style sheet has no arrows of its own: they have to be images.
_ICONS = (Path(__file__).parent / "data" / "icons").as_posix()

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
QComboBox {{ selection-background-color: {HOVER_BG}; selection-color: {HOVER_FG}; }}
QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {{
    border-color: {ACCENT}; }}
QComboBox:disabled, QDoubleSpinBox:disabled, QSpinBox:disabled, QLineEdit:disabled {{
    color: #9ca3af; background: #f3f4f6; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: center right;
    width: 18px; border: 0; }}
QComboBox::down-arrow {{ image: url({_ICONS}/arrow-down.svg); width: 8px; height: 5px; }}
QComboBox::down-arrow:disabled {{ image: url({_ICONS}/arrow-down-disabled.svg); }}

/* Dropdown lists, search suggestions and tables: a highlighted row keeps
   dark text on a pale blue, so the item under the pointer stays readable. */
QAbstractItemView {{ selection-background-color: {HOVER_BG}; selection-color: {HOVER_FG}; }}
QComboBox QAbstractItemView, QCompleter QAbstractItemView, QListView#completerPopup {{
    background: #ffffff; color: #111827; border: 1px solid #c4c8ce; outline: 0; }}
QComboBox QAbstractItemView::item {{ min-height: 20px; padding: 0 6px; color: #111827; }}
QComboBox QAbstractItemView::item:hover, QComboBox QAbstractItemView::item:selected {{
    background: {HOVER_BG}; color: {HOVER_FG}; }}

QAbstractSpinBox {{ padding-right: 18px; }}
QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {{ subcontrol-origin: border;
    width: 16px; border-left: 1px solid #d3d6db; background: #f8fafc; }}
QAbstractSpinBox::up-button {{ subcontrol-position: top right; border-top-right-radius: 3px; }}
QAbstractSpinBox::down-button {{ subcontrol-position: bottom right;
    border-bottom-right-radius: 3px; border-top: 1px solid #e5e7eb; }}
QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {{
    background: {HOVER_BG}; }}
QAbstractSpinBox::up-button:pressed, QAbstractSpinBox::down-button:pressed {{
    background: #c5d8f7; }}
QAbstractSpinBox::up-arrow {{ image: url({_ICONS}/arrow-up.svg); width: 8px; height: 5px; }}
QAbstractSpinBox::down-arrow {{ image: url({_ICONS}/arrow-down.svg); width: 8px; height: 5px; }}
QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::up-arrow:off {{
    image: url({_ICONS}/arrow-up-disabled.svg); }}
QAbstractSpinBox::down-arrow:disabled, QAbstractSpinBox::down-arrow:off {{
    image: url({_ICONS}/arrow-down-disabled.svg); }}

QMenu {{ background: #ffffff; border: 1px solid #c4c8ce; padding: 3px 0; }}
QMenu::item {{ padding: 4px 22px 4px 20px; color: #111827; }}
QMenu::item:selected {{ background: {HOVER_BG}; color: {HOVER_FG}; }}
QMenu::item:disabled {{ color: #9ca3af; }}
QMenu::right-arrow {{ image: url({_ICONS}/arrow-right.svg); width: 5px; height: 8px; }}
QMenu::indicator {{ width: 12px; height: 12px; left: 4px; }}
QMenu::separator {{ height: 1px; background: #e5e7eb; margin: 3px 6px; }}
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
