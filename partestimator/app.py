"""The one window: two tabs, Sheet Metal and Machining.

    [Sheet Metal | Machining]  [Open STEP...]  part.step  <why it opened here>  Settings...
    +-----------------------------------------------------------------------------+
    |  panels                         |  3D view (same VTK viewer on both tabs)   |
    +-----------------------------------------------------------------------------+

Both tools are their original windows, embedded whole, so every feature either
one had is still there. This window owns opening a file: it classifies the part
(in a separate process, so a large file cannot freeze the window), selects the
matching tab, and hands it the path. Clicking the other tab while a part is open
overrides the detection: the same file opens there.
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTabBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import __appname__, __version__
from .classify import MACHINED, SHEET, Classification, classify_to_dict, from_dict
from .settings import MAX_PAD_IN, SETTINGS_PATH, Settings

PAGE_SHEET, PAGE_MACHINED = 0, 1
TAB_INDEX = {0: "sheet_metal", 1: "machined"}

BANNER_STYLE = {
    SHEET: "background:#e8f0fd; color:#0b3a82; border:1px solid #bcd0f5;"
           " border-radius:4px; padding:6px 10px;",
    MACHINED: "background:#fdf4e3; color:#7a4a05; border:1px solid #f1d7a6;"
              " border-radius:4px; padding:6px 10px;",
    None: "background:#ffffff; color:#4b5563; border:1px solid #d3d6db;"
          " border-radius:4px; padding:6px 10px;",
}
KIND_LABEL = {SHEET: "Sheet metal", MACHINED: "Machined"}
TAB_NAME = {SHEET: "Sheet Metal", MACHINED: "Machining"}


def _inch_spin(value: float, high: float, step: float = 0.125, decimals: int = 3,
               tip: str = "", suffix: str = " in") -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setDecimals(decimals)
    spin.setRange(0.0, high)
    spin.setSingleStep(step)
    spin.setSuffix(suffix)
    spin.setValue(value)
    spin.setToolTip(tip)
    return spin


def _note(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color:#777;")
    return label


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        outer = QVBoxLayout(self)
        tabs = QTabWidget()
        outer.addWidget(tabs)

        # -- general ---------------------------------------------------------
        page = QWidget()
        form = QFormLayout(page)
        tabs.addTab(page, "General")
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setDecimals(3)
        self.cutoff.setRange(0.010, 1.000)
        self.cutoff.setSingleStep(0.005)
        self.cutoff.setSuffix(" in")
        self.cutoff.setValue(settings.sheet_cutoff_in)
        form.addRow("Flat sheet metal thickness cutoff:", self.cutoff)
        form.addRow(_note(
            "A FLAT part of uniform thickness at or under this is treated as "
            "sheet metal; thicker is machined plate. Bent parts are always "
            "sheet metal."))

        form.addRow(QLabel("<b>Machining stock allowances</b>"))
        self.pads = {}
        rows = (
            ("stock_length_pad_in", "Length:", "Added to the part's length (plate, bar and "
                                               "extrusion) for work-holding."),
            ("stock_width_pad_in", "Width:", "Added to the part's width (plate) for "
                                             "work-holding."),
            ("stock_thickness_pad_in", "Thickness:", "Added to the part's thickness (plate "
                                                     "over the sheet break) for cleanup."),
            ("stock_diameter_pad_in", "Bar diameter:", "Added to the part's diameter (round "
                                                       "bar) for cleanup."),
        )
        for key, label, tip in rows:
            spin = _inch_spin(getattr(settings, key), MAX_PAD_IN, tip=tip)
            form.addRow(label, spin)
            self.pads[key] = spin
        form.addRow(_note(
            "The recommended stock is the part plus these, rounded up to the "
            "next 1/4 in. Sheet (0.190 in thick or under) is used as-is."))

        # -- tabbed fixturing ------------------------------------------------
        page = QWidget()
        form = QFormLayout(page)
        tabs.addTab(page, "Tabbed fixturing")
        self.fixture_enabled = QCheckBox("Tab milled plate parts by default")
        self.fixture_enabled.setChecked(settings.fixture_enabled)
        self.fixture_enabled.setToolTip(
            "Each part can still be taken off tabs in Part Inputs > Fixturing.")
        form.addRow(self.fixture_enabled)
        form.addRow(_note(
            "The part is cut out of its plate with a channel round its outline, held "
            "to the frame of left-over stock by tabs. Thick enough plate stands on "
            "one edge in a 5-axis vise and is done in one operation, turned 180 "
            "degrees between sides; thinner or taller plate lies flat for two "
            "operations, the second facing the floor off."))
        self.fx = {}

        def add(key, label, high, step=0.125, decimals=3, tip="", suffix=" in"):
            spin = _inch_spin(getattr(settings, key), high, step, decimals, tip, suffix)
            form.addRow(label, spin)
            self.fx[key] = spin

        form.addRow(QLabel("<b>Tabs</b>"))
        add("tab_height_in", "Tab height:", 1.0, 0.010,
            tip="Through the thickness. Each tab is centred where the part's wall "
                "reaches the outline at its spot.")
        add("tab_width_in", "Tab width:", 2.0, 0.0625, tip="Along the part's edge.")
        add("tab_spacing_in", "One tab per:", 100.0, 0.5, decimals=2,
            tip="Of outline. Tabs go on straight edges, away from corners and arcs.")
        self.min_tabs = QSpinBox()
        self.min_tabs.setRange(1, 50)
        self.min_tabs.setValue(int(settings.min_tabs))
        form.addRow("At least:", self.min_tabs)

        form.addRow(QLabel("<b>Channel width (the profiling cutter)</b>"))
        add("channel_depth_1_in", "Cut from one side up to:", 12.0, 0.125,
            tip="Standing in the vise each side cuts half the thickness; flat, the "
                "whole thickness.")
        add("channel_width_1_in", "    width:", 2.0, 0.0625)
        add("channel_depth_2_in", "Up to:", 12.0, 0.125)
        add("channel_width_2_in", "    width:", 2.0, 0.0625)
        add("channel_width_3_in", "Deeper:", 2.0, 0.0625)

        form.addRow(QLabel("<b>Stock and vise</b>"))
        add("min_frame_in", "Frame outside the channel, at least:", 6.0,
            tip="The recommended plate grows (to the next 1/4 in) where it would be "
                "thinner.")
        add("vise_grip_in", "Vise grip on the clamped edge:", 3.0, 0.0625)
        add("vise_clearance_in", "Clearance above the jaws:", 6.0, 0.0625,
            tip="So the holder clears the vise; the clamped edge grows by grip plus "
                "clearance beyond the frame.")
        add("vertical_min_thickness_in", "Stand on edge from:", 12.0, 0.0625,
            tip="Thinner plate lies flat for two operations.")
        add("max_stickout_ratio", "Stick-out above the jaws, at most:", 100.0, 0.5,
            decimals=1, suffix=" x thickness",
            tip="A blank sticking out further than this lies flat instead: tall, thin "
                "plate chatters on edge.")

        outer.addWidget(_note(f"All saved to {SETTINGS_PATH.name} beside the launcher."))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def apply_to(self, settings: Settings) -> None:
        """Copy what was entered into `settings`."""
        settings.sheet_cutoff_in = float(self.cutoff.value())
        for key, spin in {**self.pads, **self.fx}.items():
            setattr(settings, key, float(spin.value()))
        settings.min_tabs = int(self.min_tabs.value())
        settings.fixture_enabled = self.fixture_enabled.isChecked()
        if settings.channel_depth_2_in <= settings.channel_depth_1_in:
            settings.channel_depth_2_in = settings.channel_depth_1_in + 0.125


class UnifiedWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(__appname__)
        self.resize(1500, 950)
        self.setAcceptDrops(True)

        self.settings = Settings.load()
        self.settings.apply()       # the stock allowances, before any part opens
        self.path: Path | None = None
        self.detected: Classification | None = None
        self.mode: str | None = None
        self._loaded_into: dict[str, Path | None] = {SHEET: None, MACHINED: None}
        self._future = None
        self._pending_path: Path | None = None
        self._pool: cf.ProcessPoolExecutor | None = None
        self._last_dir = ""

        self._poll = QTimer(self)
        self._poll.setInterval(80)
        self._poll.timeout.connect(self._check_classification)

        self._build()
        self._warm_pool()

    # -- construction -----------------------------------------------------

    def _build(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(6, 4, 6, 0)
        outer.setSpacing(4)

        # Row 1: the two features of the tool, as tabs. Clicking the other
        # tab while a part is open is the override: the same file opens there.
        top = QHBoxLayout()
        self.mode_tabs = QTabBar()
        self.mode_tabs.setObjectName("modeTabs")
        self.mode_tabs.setDrawBase(False)
        self.mode_tabs.setExpanding(False)
        self.mode_tabs.addTab("Sheet Metal")
        self.mode_tabs.addTab("Machining")
        self.mode_tabs.setTabToolTip(0, "Flat pattern, bend table, DXF export.")
        self.mode_tabs.setTabToolTip(1, "Geometry features, stock, cycle-time estimate "
                                        "(mill or turn-mill).")
        self.mode_tabs.tabBarClicked.connect(self._tab_clicked)
        top.addWidget(self.mode_tabs)
        top.addSpacing(10)

        self.btn_open = QPushButton("Open STEP...")
        self.btn_open.setObjectName("primary")
        self.btn_open.setMinimumHeight(30)
        self.btn_open.clicked.connect(self.open_dialog)
        top.addWidget(self.btn_open)

        self.lbl_file = QLabel("No file open")
        self.lbl_file.setStyleSheet("font-weight:600; padding:0 8px;")
        top.addWidget(self.lbl_file)

        self.banner = QLabel("Open a STEP file. It is checked automatically and opened "
                             "on the right tab.")
        self.banner.setStyleSheet(BANNER_STYLE[None])
        self.banner.setWordWrap(True)
        top.addWidget(self.banner, 1)

        self.btn_settings = QPushButton("Settings...")
        self.btn_settings.clicked.connect(self.edit_settings)
        top.addWidget(self.btn_settings)
        outer.addLayout(top)

        self.stack = QStackedWidget()

        from .smt.gui.app import MainWindow as SheetWindow

        self.sheet = SheetWindow()
        self._embed(self.sheet)
        self.sheet.btn_open.hide()
        self.sheet.menuBar().clear()
        self.sheet.menuBar().hide()
        self.stack.addWidget(self.sheet)

        from .machining.ui.main_window import MainWindow as MachWindow

        self.mach = MachWindow()
        self._embed(self.mach)
        self.mach.open_button.hide()
        self.mach.fileOpened.connect(self._opened_from_library)
        self.stack.addWidget(self.mach)

        outer.addWidget(self.stack, 1)
        self.setCentralWidget(central)
        self._menu()

    def _tab_clicked(self, index: int) -> None:
        if index < 0:
            return
        kind = TAB_INDEX[index]
        if self.path is None:
            self.stack.setCurrentIndex(PAGE_SHEET if kind == SHEET else PAGE_MACHINED)
            self.mode = kind
            return
        if self._future is not None:
            return
        self.set_mode(kind, user=True)

    @staticmethod
    def _embed(win: QMainWindow) -> None:
        win.setWindowFlags(Qt.Widget)

    def _menu(self) -> None:
        m = self.menuBar().addMenu("&File")
        a = QAction("&Open STEP...", self)
        a.setShortcut(QKeySequence.Open)
        a.triggered.connect(self.open_dialog)
        m.addAction(a)
        e = QAction("&Export flat pattern (sheet metal)...", self)
        e.setShortcut("Ctrl+E")
        e.triggered.connect(self._export_sheet)
        m.addAction(e)
        m.addSeparator()
        t = QAction("Shop &tuning...", self)
        t.setToolTip("The library of measured parts the machining estimate is fitted to.")
        t.triggered.connect(self._open_tuning)
        m.addAction(t)
        s = QAction("&Settings...", self)
        s.triggered.connect(self.edit_settings)
        m.addAction(s)
        m.addSeparator()
        q = QAction("&Quit", self)
        q.setShortcut(QKeySequence.Quit)
        q.triggered.connect(self.close)
        m.addAction(q)
        h = self.menuBar().addMenu("&Help")
        about = QAction("&About", self)
        about.triggered.connect(self._about)
        h.addAction(about)

    # -- classification process --------------------------------------------

    def _warm_pool(self) -> None:
        """Start the classifier process now, so the first file opens quickly."""
        try:
            self._pool = cf.ProcessPoolExecutor(
                max_workers=1, mp_context=mp.get_context("spawn"))
            self._pool.submit(_warm_child)
        except Exception:  # noqa: BLE001 -- fall back to in-process
            self._pool = None

    # -- opening -------------------------------------------------------------

    def open_dialog(self) -> None:
        path, _f = QFileDialog.getOpenFileName(
            self, "Open STEP file", self._last_dir,
            "STEP files (*.step *.stp *.STEP *.STP);;All files (*)")
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        path = Path(path)
        if self._future is not None:
            return  # one at a time; the button is disabled while classifying
        self._last_dir = str(path.parent)
        self.path = path
        self.detected = None
        self._loaded_into = {SHEET: None, MACHINED: None}
        self.lbl_file.setText(path.name)
        self.setWindowTitle(f"{__appname__} -- {path.name}")
        self._show_banner(None, f"Checking whether {path.name} is sheet metal...")
        self.mode_tabs.setEnabled(False)
        self.btn_open.setEnabled(False)
        QApplication.setOverrideCursor(Qt.BusyCursor)

        cutoff = self.settings.sheet_cutoff_in
        if self._pool is not None:
            try:
                self._future = self._pool.submit(classify_to_dict, str(path), cutoff)
                self._pending_path = path
                self._poll.start()
                return
            except Exception:  # noqa: BLE001 -- pool broke; do it here
                self._pool = None
        try:
            result = classify_to_dict(str(path), cutoff)
            self._classified(path, from_dict(result), None)
        except Exception as exc:  # noqa: BLE001
            self._classified(path, None, exc)

    def _check_classification(self) -> None:
        fut = self._future
        if fut is None or not fut.done():
            return
        self._poll.stop()
        self._future = None
        path = self._pending_path
        try:
            c = from_dict(fut.result())
            self._classified(path, c, None)
        except Exception as exc:  # noqa: BLE001
            # A dead pool (e.g. the kernel crashed on this file) is replaced.
            if isinstance(exc, cf.process.BrokenProcessPool):
                self._warm_pool()
            self._classified(path, None, exc)

    def _classified(self, path, c: Classification | None, exc) -> None:
        QApplication.restoreOverrideCursor()
        self.btn_open.setEnabled(True)
        self.mode_tabs.setEnabled(True)
        if c is None:
            # Could not classify: the machining side reads almost anything,
            # so default there and say why.
            self.detected = None
            self._show_banner(MACHINED, f"Could not classify ({type(exc).__name__}: {exc}). "
                                        f"Opened on the Machining tab -- click Sheet Metal "
                                        f"to try it there.")
            self.set_mode(MACHINED, user=False)
            return
        self.detected = c
        self.set_mode(c.kind, user=False)

    def set_mode(self, kind: str, user: bool) -> None:
        if self.path is None:
            return
        self.mode = kind
        self.mode_tabs.blockSignals(True)
        self.mode_tabs.setCurrentIndex(PAGE_SHEET if kind == SHEET else PAGE_MACHINED)
        self.mode_tabs.blockSignals(False)
        self.stack.setCurrentIndex(PAGE_SHEET if kind == SHEET else PAGE_MACHINED)

        if self.detected is not None:
            if kind == self.detected.kind:
                self._show_banner(kind, self.detected.reason)
            else:
                self._show_banner(kind, f"Opened on the {TAB_NAME[kind]} tab by hand. "
                                        f"Detected: {self.detected.reason}")
        elif user:
            self._show_banner(kind, f"Opened on the {TAB_NAME[kind]} tab by hand.")

        if self._loaded_into.get(kind) != self.path:
            self._loaded_into[kind] = self.path
            if kind == SHEET:
                self.sheet.opts.sheet_face = None
                try:
                    self.sheet.view3d.set_highlight(None)
                except Exception:  # noqa: BLE001
                    pass
                self.sheet.load(self.path)
            else:
                self.mach._load_and_process(str(self.path))

    def _open_tuning(self) -> None:
        self.mach.open_tuning()

    def _opened_from_library(self, path: str) -> None:
        """The tuning library opened a part straight into the Machining tab."""
        p = Path(path)
        self.path = p
        self.detected = None
        self._loaded_into = {SHEET: None, MACHINED: p}
        self.lbl_file.setText(p.name)
        self.setWindowTitle(f"{__appname__} -- {p.name}")
        self.mode = MACHINED
        self.mode_tabs.blockSignals(True)
        self.mode_tabs.setCurrentIndex(PAGE_MACHINED)
        self.mode_tabs.blockSignals(False)
        self.stack.setCurrentIndex(PAGE_MACHINED)
        self._show_banner(MACHINED, "Opened from the shop tuning library, with its saved inputs.")

    def _show_banner(self, kind, text: str) -> None:
        self.banner.setStyleSheet(BANNER_STYLE.get(kind, BANNER_STYLE[None]))
        prefix = f"<b>{KIND_LABEL[kind]}</b> &nbsp;" if kind else ""
        self.banner.setText(prefix + text)

    # -- misc ------------------------------------------------------------------

    def _export_sheet(self) -> None:
        if self.mode == SHEET and self.sheet.btn_export.isEnabled():
            self.sheet.export()
        else:
            QMessageBox.information(self, "Export", "Flat-pattern export is for sheet "
                                                    "metal parts. Open one first.")

    def edit_settings(self) -> None:
        dlg = SettingsDialog(self.settings, self)
        if dlg.exec() != QDialog.Accepted:
            return
        old_cutoff = self.settings.sheet_cutoff_in
        old = (self.settings.stock_allowances(), self.settings.fixture())
        dlg.apply_to(self.settings)
        try:
            self.settings.save()
        except OSError as exc:
            QMessageBox.warning(self, "Settings", f"Could not save settings: {exc}")
        if (self.settings.stock_allowances(), self.settings.fixture()) != old:
            self.settings.apply()
            self.mach.refresh_stock_allowances()
        if (self.path is not None and self._future is None
                and self.settings.sheet_cutoff_in != old_cutoff):
            if QMessageBox.question(
                    self, "Settings", "Re-check the open part with the new cutoff?"
            ) == QMessageBox.Yes:
                self.open_path(self.path)

    def _about(self) -> None:
        QMessageBox.information(
            self, f"About {__appname__}",
            f"{__appname__} {__version__}\n\n"
            "Sheet metal tool (from SMTv13) and machining cycle-time estimator "
            "(from STEP Estimator v2.5) in one window.\n\n"
            "Runs entirely on this computer. No network connection is made at runtime.")

    def dragEnterEvent(self, event):  # noqa: N802
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(u.toLocalFile().lower().endswith((".step", ".stp")) for u in urls):
            event.acceptProposedAction()

    def dropEvent(self, event):  # noqa: N802
        for u in event.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith((".step", ".stp")):
                self.open_path(Path(p))
                break

    def closeEvent(self, event):  # noqa: N802
        self._poll.stop()
        for w in (self.sheet, self.mach):
            try:
                w.close()
            except Exception:  # noqa: BLE001
                pass
        if self._pool is not None:
            try:
                self._pool.shutdown(wait=False, cancel_futures=True)
            except Exception:  # noqa: BLE001
                pass
        super().closeEvent(event)


def _warm_child() -> bool:
    """Import the kernel in the classifier process ahead of the first file."""
    import OCP.STEPControl  # noqa: F401

    from . import classify  # noqa: F401

    return True


def run(path: Path | None = None) -> int:
    app = QApplication.instance() or QApplication([])
    app.setApplicationName(__appname__)
    from . import theme

    theme.apply(app)
    win = UnifiedWindow()
    win.show()
    if path is not None:
        QTimer.singleShot(150, lambda: win.open_path(Path(path)))
    return app.exec()
